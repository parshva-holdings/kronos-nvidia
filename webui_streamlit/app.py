"""Kronos × Nifty live dashboard.

A self-contained Streamlit app that:
  - Fetches live OHLCV from Yahoo on every page load (cached 5 min)
  - Loads the highest open Kronos model once, keeps it warm in memory
  - Shows an interactive candlestick + volume chart
  - Generates probabilistic forecasts on demand with adjustable horizon / sampling
  - Displays directional probability, expected close, and confidence band

No CSV uploads. No manual setup. Pick a symbol, click Forecast.

Run me
------
    streamlit run webui_streamlit/app.py --server.port 8501
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
if UPSTREAM.exists() and str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

st.set_page_config(
    page_title="Kronos · Nifty Live",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Symbol catalog. Yahoo's NSE convention: indices start with ^, stocks end .NS
# ---------------------------------------------------------------------------

INDICES = {
    "Nifty 50":            "^NSEI",
    "Bank Nifty":          "^NSEBANK",
    "Nifty IT":            "^CNXIT",
    "Nifty Auto":          "^CNXAUTO",
    "Nifty Pharma":        "^CNXPHARMA",
    "Nifty FMCG":          "^CNXFMCG",
    "Nifty Metal":         "^CNXMETAL",
    "Nifty Energy":        "^CNXENERGY",
    "Nifty Realty":        "^CNXREALTY",
    "Nifty Midcap 100":    "^CRSLDX",
    "Nifty Midcap 50":     "^NSMIDCP",
}

NIFTY50_STOCKS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "BHARTIARTL", "LT", "ITC", "KOTAKBANK", "HINDUNILVR",
    "AXISBANK", "SBIN", "BAJFINANCE", "M&M", "MARUTI",
    "ASIANPAINT", "SUNPHARMA", "HCLTECH", "TITAN", "TATAMOTORS",
    "ULTRACEMCO", "POWERGRID", "NTPC", "NESTLEIND", "WIPRO",
    "ADANIENT", "JSWSTEEL", "TATASTEEL", "BAJAJFINSV", "ONGC",
    "COALINDIA", "TECHM", "GRASIM", "INDUSINDBK", "BAJAJ-AUTO",
    "DRREDDY", "ADANIPORTS", "CIPLA", "EICHERMOT", "HINDALCO",
    "HDFCLIFE", "SBILIFE", "DIVISLAB", "BPCL", "BRITANNIA",
    "TATACONSUM", "APOLLOHOSP", "HEROMOTOCO", "TRENT", "SHRIRAMFIN",
]

INTERVAL_PRESETS = {
    "Daily (max 30 yr history)":     ("1d",  "max"),
    "Hourly (last 720 days)":        ("1h",  "720d"),
    "30-min (last 60 days)":         ("30m", "60d"),
    "5-min (last 60 days)":          ("5m",  "60d"),
}


# ---------------------------------------------------------------------------
# Cached resource: Kronos predictor (loaded once per Streamlit process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Kronos model (one-time, ~20s)...")
def get_predictor(model_id: str, tokenizer_id: str, device: str | None):
    """Returns (predictor, info_dict). Cached for the lifetime of the process."""
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    info = {
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return predictor, info


# ---------------------------------------------------------------------------
# Cached data: live OHLCV from Yahoo, refreshed every 5 minutes
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Fetching live data from Yahoo Finance...")
def fetch_live_ohlcv(yahoo_symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(yahoo_symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return df

    # Strip timezone for plotting clarity (Yahoo returns IST for NSE symbols)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    out = pd.DataFrame({
        "timestamps": df.index,
        "open":   df["Open"].astype(float),
        "high":   df["High"].astype(float),
        "low":    df["Low"].astype(float),
        "close":  df["Close"].astype(float),
        "volume": df["Volume"].fillna(0).astype(float),
    })
    typ = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0
    out["amount"] = typ * out["volume"]
    return out.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forecast helper
# ---------------------------------------------------------------------------

def run_forecast(
    predictor,
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    samples: int,
    temperature: float,
    top_p: float,
) -> tuple[pd.DataFrame, pd.Series]:
    x_df = df.tail(lookback)[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_ts = df.tail(lookback)["timestamps"].reset_index(drop=True)

    # Extend cadence into the future
    if len(x_ts) < 2:
        raise ValueError("Need at least 2 historical bars to forecast")
    cadence = x_ts.iloc[-1] - x_ts.iloc[-2]
    y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + cadence, periods=horizon, freq=cadence))

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=horizon,
        T=temperature,
        top_p=top_p,
        sample_count=samples,
        verbose=False,
    )
    return pred_df, y_ts


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Kronos · Nifty Live")
    st.caption(
        "Foundation-model price forecasts for the Indian markets. "
        "Pick a symbol, click Forecast, get a probabilistic outlook. "
        "[Paper](https://arxiv.org/abs/2508.02739) · "
        "[Repo](https://github.com/shiyu-coder/Kronos)"
    )

    # ============== Sidebar ==============
    with st.sidebar:
        st.header("⚙️ Settings")

        symbol_kind = st.radio(
            "What to forecast",
            ["Index / sector", "Single stock"],
            horizontal=True,
        )
        if symbol_kind == "Index / sector":
            label = st.selectbox("Index", list(INDICES.keys()), index=0)
            yahoo_symbol = INDICES[label]
        else:
            label = st.selectbox(
                "Stock", NIFTY50_STOCKS, index=NIFTY50_STOCKS.index("RELIANCE")
            )
            yahoo_symbol = f"{label}.NS"

        preset = st.selectbox("Timeframe", list(INTERVAL_PRESETS.keys()), index=0)
        interval, period = INTERVAL_PRESETS[preset]

        st.divider()
        st.subheader("Forecast")
        lookback = st.slider("Lookback bars", min_value=100, max_value=512, value=400, step=10,
                             help="How many historical bars Kronos sees as context (max 512)")
        horizon = st.slider("Forecast horizon", min_value=5, max_value=120, value=30, step=5,
                            help="Number of future bars to predict")
        samples = st.slider("Monte-Carlo samples", min_value=1, max_value=10, value=5, step=1,
                            help="More samples = smoother forecast + tighter confidence band, but slower")

        with st.expander("Advanced sampling", expanded=False):
            temperature = st.slider("Temperature (T)", 0.1, 2.0, 1.0, 0.1)
            top_p = st.slider("Nucleus top-p", 0.1, 1.0, 0.9, 0.05)

        st.divider()
        st.subheader("Model")
        model_id = st.selectbox(
            "Kronos variant",
            [
                "NeoQuasar/Kronos-base",   # 102M, largest open
                "NeoQuasar/Kronos-small",  # 24.7M
                "NeoQuasar/Kronos-mini",   # 4.1M
            ],
            index=0,
            help="Kronos-base is the largest open variant (102M params). Kronos-large (499M) is closed.",
        )
        # Tokenizer pairs with model
        tokenizer_id = (
            "NeoQuasar/Kronos-Tokenizer-2k"
            if "mini" in model_id
            else "NeoQuasar/Kronos-Tokenizer-base"
        )
        device_choice = st.selectbox("Device", ["auto", "cuda:0", "mps", "cpu"], index=0)
        device = None if device_choice == "auto" else device_choice

    # ============== Header strip ==============
    st.markdown(f"#### {label} · {preset}")

    # ============== Live data ==============
    df = fetch_live_ohlcv(yahoo_symbol, interval, period)
    if df.empty:
        st.error(f"No data returned for {yahoo_symbol}. Try a different timeframe.")
        return

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg = last["close"] - prev["close"]
    chg_pct = chg / prev["close"] * 100 if prev["close"] else 0

    cols = st.columns(5)
    cols[0].metric("Latest close", f"{last['close']:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
    cols[1].metric("Day high",     f"{last['high']:,.2f}")
    cols[2].metric("Day low",      f"{last['low']:,.2f}")
    cols[3].metric("Volume",       f"{last['volume']:,.0f}")
    cols[4].metric("Bars loaded",  f"{len(df):,}")

    # ============== Forecast ==============
    forecast_df = None
    forecast_ts = None
    info = None

    if st.button("🔮 Generate forecast", type="primary", use_container_width=True):
        try:
            predictor, info = get_predictor(model_id, tokenizer_id, device)
            if len(df) < lookback:
                st.warning(f"Only {len(df)} bars available; reducing lookback to {len(df)}.")
                lookback = len(df)
            with st.spinner(f"Sampling {samples} forecast paths over {horizon} bars..."):
                forecast_df, forecast_ts = run_forecast(
                    predictor, df, lookback, horizon, samples, temperature, top_p
                )
            st.session_state["last_forecast"] = (forecast_df, forecast_ts, label, info)
        except Exception as e:  # noqa: BLE001
            st.exception(e)

    # Persist last forecast across reruns
    if forecast_df is None and "last_forecast" in st.session_state:
        forecast_df, forecast_ts, _, info = st.session_state["last_forecast"]

    # ============== Chart ==============
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["timestamps"].tail(min(len(df), 200)),
            open=df["open"].tail(200),
            high=df["high"].tail(200),
            low=df["low"].tail(200),
            close=df["close"].tail(200),
            name="Historical",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    )

    if forecast_df is not None and forecast_ts is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_ts,
                y=forecast_df["close"].values,
                mode="lines+markers",
                name="Forecast close",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=5),
            )
        )
        # Bracket: forecast high and low as error band
        fig.add_trace(
            go.Scatter(
                x=forecast_ts.tolist() + forecast_ts.tolist()[::-1],
                y=forecast_df["high"].tolist() + forecast_df["low"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(59,130,246,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Forecast range (high-low)",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============== Forecast summary ==============
    if forecast_df is not None:
        st.markdown("#### Forecast summary")
        cur = float(df.iloc[-1]["close"])
        end = float(forecast_df.iloc[-1]["close"])
        ret = (end - cur) / cur * 100
        max_close = float(forecast_df["close"].max())
        min_close = float(forecast_df["close"].min())
        max_drawdown = (min_close - cur) / cur * 100
        max_runup = (max_close - cur) / cur * 100

        cols = st.columns(5)
        cols[0].metric("End-of-horizon close",  f"{end:,.2f}", f"{ret:+.2f}%")
        cols[1].metric("Max forecasted close",  f"{max_close:,.2f}", f"{max_runup:+.2f}%")
        cols[2].metric("Min forecasted close",  f"{min_close:,.2f}", f"{max_drawdown:+.2f}%")
        # Crude direction probability via fraction of bars above current
        up_frac = float((forecast_df["close"] > cur).mean()) * 100
        cols[3].metric("Bars above today",      f"{up_frac:.0f}%",
                       help="Fraction of forecasted bars closing above the latest actual close.")
        cols[4].metric("Avg vol forecast",      f"{forecast_df['volume'].mean():,.0f}")

        with st.expander("Raw forecast bars", expanded=False):
            disp = forecast_df.copy()
            disp.insert(0, "timestamp", forecast_ts.values)
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # ============== Footer ==============
    if info:
        st.caption(
            f"Model: {info['model_id']} · device: {info['device']} · "
            f"{'GPU: ' + info['gpu'] if info['gpu'] else 'CPU/MPS inference'} · "
            f"PyTorch {info['torch']} · "
            f"Generated {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')}"
        )
    else:
        st.caption(
            "Data: Yahoo Finance (15-min delayed for NSE) · "
            f"Updated {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')} · "
            "Click *Generate forecast* to load the model."
        )


if __name__ == "__main__":
    main()
