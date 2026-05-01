"""Kronos × Nifty live + back-in-time dashboard.

Streamlit app that lets you:
  - Pick any historical date as the "as-of" point
  - Run Kronos as if today were that date (it only sees bars up to and including it)
  - Overlay what actually happened next, side-by-side with the forecast
  - See hit-rate / MAPE / range-coverage metrics when actuals exist

Default as-of = latest available bar (acts like a normal forward-looking dashboard
when you don't move the slider).

Run me:
    streamlit run webui_streamlit/app.py --server.port 8501
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
if UPSTREAM.exists() and str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

st.set_page_config(
    page_title="Kronos · Nifty Time-Travel",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Symbol catalog
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
# Cached resource: Kronos predictor
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Kronos model (one-time, ~20s)...")
def get_predictor(model_id: str, tokenizer_id: str, device: str | None):
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
# Cached data fetch
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Fetching live data from Yahoo Finance...")
def fetch_live_ohlcv(yahoo_symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(yahoo_symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return df

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
# Slicing: split full history at the as-of date
# ---------------------------------------------------------------------------

def split_at_as_of(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int]:
    """
    Return (context_df, context_ts, actuals_df, actuals_ts, end_idx).

    - context = up to `lookback` bars ending at or before `as_of` (inclusive)
    - actuals = up to `horizon` bars strictly after `as_of` (may be empty)
    - end_idx = position of the last context bar in the original df

    `as_of` snaps to the latest available bar at or before it (handles
    weekends/holidays where the picker date doesn't have a trading bar).
    """
    valid = df["timestamps"] <= as_of
    if not valid.any():
        raise ValueError(f"No bars at or before {as_of:%Y-%m-%d}")

    end_idx = int(valid.values.nonzero()[0][-1])

    start_idx = max(0, end_idx - lookback + 1)
    cols = ["open", "high", "low", "close", "volume", "amount"]
    context_df = df.iloc[start_idx : end_idx + 1][cols].reset_index(drop=True)
    context_ts = df.iloc[start_idx : end_idx + 1]["timestamps"].reset_index(drop=True)

    actuals_df = df.iloc[end_idx + 1 : end_idx + 1 + horizon][cols].reset_index(drop=True)
    actuals_ts = df.iloc[end_idx + 1 : end_idx + 1 + horizon]["timestamps"].reset_index(drop=True)

    return context_df, context_ts, actuals_df, actuals_ts, end_idx


def run_forecast_at(
    predictor,
    context_df: pd.DataFrame,
    context_ts: pd.Series,
    actuals_ts: pd.Series,
    horizon: int,
    samples: int,
    temperature: float,
    top_p: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Forecast `horizon` bars starting after the last context bar. If actuals_ts
    has at least `horizon` real bars we use them so the forecast lines up with
    actual trading days; otherwise we extrapolate the cadence into the future.
    """
    if len(context_ts) < 2:
        raise ValueError("Need at least 2 historical bars to forecast")

    if len(actuals_ts) >= horizon:
        y_ts = actuals_ts.iloc[:horizon].reset_index(drop=True)
    else:
        cadence = context_ts.iloc[-1] - context_ts.iloc[-2]
        synth = pd.Series(
            pd.date_range(
                start=context_ts.iloc[-1] + cadence,
                periods=horizon,
                freq=cadence,
            )
        )
        if len(actuals_ts) > 0:
            # Use real timestamps for the bars we have, synth for the rest
            tail_synth = synth.iloc[len(actuals_ts):].reset_index(drop=True)
            y_ts = pd.concat([actuals_ts.reset_index(drop=True), tail_synth], ignore_index=True)
        else:
            y_ts = synth

    pred_df = predictor.predict(
        df=context_df,
        x_timestamp=context_ts,
        y_timestamp=y_ts,
        pred_len=horizon,
        T=temperature,
        top_p=top_p,
        sample_count=samples,
        verbose=False,
    )
    return pred_df, y_ts


def compute_comparison(forecast_df: pd.DataFrame, actuals_df: pd.DataFrame) -> dict:
    """Score the forecast against what actually happened, bar-by-bar."""
    n = int(min(len(forecast_df), len(actuals_df)))
    if n == 0:
        return {"n_bars": 0}

    f = forecast_df.iloc[:n].reset_index(drop=True)
    a = actuals_df.iloc[:n].reset_index(drop=True)

    out = {"n_bars": n}

    # Per-OHLC MAPE
    for col in ("open", "high", "low", "close"):
        # Avoid divide-by-zero
        denom = np.where(np.abs(a[col].values) < 1e-9, 1e-9, a[col].values)
        out[f"mape_{col}"] = float(np.mean(np.abs((f[col].values - a[col].values) / denom)) * 100)

    # Bar-to-bar direction hit rate: did forecast and actual move the same way?
    if n >= 2:
        f_dir = np.sign(f["close"].diff().iloc[1:].values)
        a_dir = np.sign(a["close"].diff().iloc[1:].values)
        # Only count bars where both have a defined direction (non-zero)
        defined = (f_dir != 0) & (a_dir != 0)
        if defined.any():
            out["hit_rate"] = float(np.mean(f_dir[defined] == a_dir[defined]) * 100)
        else:
            out["hit_rate"] = None
    else:
        out["hit_rate"] = None

    # Range coverage: fraction of actual closes inside the forecast [low, high] band
    in_band = ((a["close"].values >= f["low"].values) & (a["close"].values <= f["high"].values))
    out["range_coverage"] = float(np.mean(in_band) * 100)

    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Kronos · Nifty Time-Travel")
    st.caption(
        "Pick a date, generate a forecast as if it were that day, see what actually happened. "
        "[Paper](https://arxiv.org/abs/2508.02739) · "
        "[Repo](https://github.com/parshva-holdings/kronos-nvidia)"
    )

    # ============== Sidebar (part 1: symbol + timeframe) ==============
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

    # ============== Live data fetch ==============
    df = fetch_live_ohlcv(yahoo_symbol, interval, period)
    if df.empty or len(df) < 10:
        st.error(f"No data returned for {yahoo_symbol}. Try a different timeframe.")
        return

    earliest = df.iloc[0]["timestamps"]
    latest = df.iloc[-1]["timestamps"]

    # ============== Sidebar (part 2: as-of + forecast settings) ==============
    with st.sidebar:
        st.divider()
        st.subheader("⏪ As-of date")
        st.caption(
            f"Choose any date between {earliest:%Y-%m-%d} and {latest:%Y-%m-%d}. "
            "Kronos will predict using only bars up to and including this date."
        )

        # Default to latest for "live" mode, but make it easy to scrub backward
        as_of_date: date = st.date_input(
            "As-of date",
            value=latest.date(),
            min_value=earliest.date(),
            max_value=latest.date(),
            help="Forecast will use only bars up to and including this date.",
        )

        # Quick preset buttons
        col_a, col_b, col_c, col_d = st.columns(4)
        if col_a.button("Latest", use_container_width=True):
            as_of_date = latest.date()
            st.session_state["_as_of_override"] = as_of_date
        if col_b.button("-1 m",  use_container_width=True):
            as_of_date = (latest - pd.Timedelta(days=30)).date()
            st.session_state["_as_of_override"] = as_of_date
        if col_c.button("-6 m",  use_container_width=True):
            as_of_date = (latest - pd.Timedelta(days=180)).date()
            st.session_state["_as_of_override"] = as_of_date
        if col_d.button("-1 yr", use_container_width=True):
            as_of_date = (latest - pd.Timedelta(days=365)).date()
            st.session_state["_as_of_override"] = as_of_date

        # If a preset button was just clicked this run, honor it
        if "_as_of_override" in st.session_state:
            as_of_date = st.session_state.pop("_as_of_override")

        st.divider()
        st.subheader("Forecast")
        lookback = st.slider("Lookback bars", 100, 512, 400, step=10,
                             help="How many historical bars Kronos sees as context (max 512)")
        horizon = st.slider("Forecast horizon", 5, 120, 30, step=5,
                            help="Number of future bars to predict")
        samples = st.slider("Monte-Carlo samples", 1, 10, 5, step=1,
                            help="More samples = smoother forecast + tighter confidence band, but slower")

        with st.expander("Advanced sampling", expanded=False):
            temperature = st.slider("Temperature (T)", 0.1, 2.0, 1.0, 0.1)
            top_p = st.slider("Nucleus top-p", 0.1, 1.0, 0.9, 0.05)

        st.divider()
        st.subheader("Model")
        model_id = st.selectbox(
            "Kronos variant",
            [
                "NeoQuasar/Kronos-base",
                "NeoQuasar/Kronos-small",
                "NeoQuasar/Kronos-mini",
            ],
            index=0,
            help="Kronos-base is the largest open variant (102M params). Kronos-large (499M) is closed.",
        )
        tokenizer_id = (
            "NeoQuasar/Kronos-Tokenizer-2k"
            if "mini" in model_id
            else "NeoQuasar/Kronos-Tokenizer-base"
        )
        device_choice = st.selectbox("Device", ["auto", "cuda:0", "mps", "cpu"], index=0)
        device = None if device_choice == "auto" else device_choice

    # ============== Slice at as-of ==============
    as_of_ts = pd.Timestamp(as_of_date) + pd.Timedelta(hours=23, minutes=59)
    try:
        context_df, context_ts, actuals_df, actuals_ts, end_idx = split_at_as_of(
            df, as_of_ts, lookback, horizon
        )
    except ValueError as e:
        st.error(str(e))
        return

    actual_as_of_ts = context_ts.iloc[-1]
    has_actuals = len(actuals_df) > 0
    bars_after = len(actuals_df)

    # ============== Header ==============
    is_latest = (actual_as_of_ts == latest)
    mode_emoji = "🟢" if is_latest else "⏪"
    mode_label = "LIVE" if is_latest else "BACK-IN-TIME"
    st.markdown(
        f"#### {mode_emoji} **{mode_label}** · {label} · {preset} · "
        f"as of **{actual_as_of_ts:%Y-%m-%d}**"
    )

    if not is_latest and has_actuals:
        st.success(
            f"You picked {as_of_date:%Y-%m-%d} (snapped to the trading bar at "
            f"{actual_as_of_ts:%Y-%m-%d}). **{bars_after} actual bars exist after that** "
            f"— Kronos's forecast will be scored against them."
        )
    elif not is_latest and not has_actuals:
        st.info(
            f"As-of date is at the very end of available data; no actuals to compare against yet."
        )

    # As-of metrics row
    as_of_row = context_df.iloc[-1]
    prev_row = context_df.iloc[-2] if len(context_df) >= 2 else as_of_row
    chg = float(as_of_row["close"] - prev_row["close"])
    chg_pct = chg / float(prev_row["close"]) * 100 if prev_row["close"] else 0

    cols = st.columns(5)
    cols[0].metric("As-of close", f"{as_of_row['close']:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
    cols[1].metric("As-of high",  f"{as_of_row['high']:,.2f}")
    cols[2].metric("As-of low",   f"{as_of_row['low']:,.2f}")
    cols[3].metric("Volume",      f"{as_of_row['volume']:,.0f}")
    cols[4].metric("Context bars", f"{len(context_df):,}",
                   help=f"Lookback bars Kronos will see as context, ending at {actual_as_of_ts:%Y-%m-%d}.")

    # ============== Forecast ==============
    forecast_df = None
    forecast_ts = None
    info = None
    cmp_metrics = None

    forecast_btn_label = "🔮 Generate forecast" + (" + score against actuals" if has_actuals else "")
    if st.button(forecast_btn_label, type="primary", use_container_width=True):
        try:
            predictor, info = get_predictor(model_id, tokenizer_id, device)
            with st.spinner(f"Sampling {samples} forecast paths over {horizon} bars..."):
                forecast_df, forecast_ts = run_forecast_at(
                    predictor,
                    context_df, context_ts,
                    actuals_ts,
                    horizon, samples, temperature, top_p,
                )
            if has_actuals:
                cmp_metrics = compute_comparison(forecast_df, actuals_df)
            cache_key = (label, str(actual_as_of_ts), horizon, samples, model_id)
            st.session_state["last_forecast"] = (
                cache_key, forecast_df, forecast_ts, actuals_df, actuals_ts, info, cmp_metrics
            )
        except Exception as e:  # noqa: BLE001
            st.exception(e)

    # Reuse last forecast if it matches current settings
    if forecast_df is None and "last_forecast" in st.session_state:
        cache_key = (label, str(actual_as_of_ts), horizon, samples, model_id)
        last = st.session_state["last_forecast"]
        if last[0] == cache_key:
            _, forecast_df, forecast_ts, _last_actuals, _last_a_ts, info, cmp_metrics = last

    # ============== Chart ==============
    fig = go.Figure()

    # Context candles (last 200 of context window for readability)
    n_show_context = min(len(context_df), 200)
    fig.add_trace(
        go.Candlestick(
            x=context_ts.tail(n_show_context),
            open=context_df["open"].tail(n_show_context),
            high=context_df["high"].tail(n_show_context),
            low=context_df["low"].tail(n_show_context),
            close=context_df["close"].tail(n_show_context),
            name="Context (what Kronos saw)",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    )

    # As-of vertical line
    fig.add_vline(
        x=actual_as_of_ts.timestamp() * 1000,  # plotly wants ms epoch for vline on time axis
        line_width=1, line_dash="dash", line_color="#94a3b8",
        annotation_text=f"as-of {actual_as_of_ts:%Y-%m-%d}",
        annotation_position="top",
    )

    # Actuals (what really happened) — green dashed, only when we're back in time
    if has_actuals:
        fig.add_trace(
            go.Scatter(
                x=actuals_ts,
                y=actuals_df["close"].values,
                mode="lines",
                name="Actual close (post as-of)",
                line=dict(color="#16a34a", width=2, dash="dot"),
            )
        )

    # Forecast overlay
    if forecast_df is not None and forecast_ts is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_ts,
                y=forecast_df["close"].values,
                mode="lines+markers",
                name="Forecast close",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=4),
            )
        )
        # High-low confidence band
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
        height=560,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============== Comparison panel (only when we have forecast + actuals) ==============
    if forecast_df is not None and cmp_metrics and cmp_metrics.get("n_bars", 0) > 0:
        st.markdown("#### 🎯 How well did Kronos call it?")

        n = cmp_metrics["n_bars"]
        fc_end = float(forecast_df["close"].iloc[n - 1])
        ac_end = float(actuals_df["close"].iloc[n - 1])
        as_of_close = float(as_of_row["close"])

        forecast_ret = (fc_end - as_of_close) / as_of_close * 100
        actual_ret   = (ac_end - as_of_close) / as_of_close * 100

        cols = st.columns(5)
        cols[0].metric(
            f"Forecast close (+{n} bars)",
            f"{fc_end:,.2f}",
            f"{forecast_ret:+.2f}%",
            help="Where Kronos thought price would be.",
        )
        cols[1].metric(
            f"Actual close (+{n} bars)",
            f"{ac_end:,.2f}",
            f"{actual_ret:+.2f}%",
            help="Where price actually went.",
        )
        cols[2].metric(
            "Close MAPE",
            f"{cmp_metrics['mape_close']:.2f}%",
            help="Mean absolute percentage error on the close price across all forecasted bars.",
        )
        if cmp_metrics.get("hit_rate") is not None:
            cols[3].metric(
                "Direction hit-rate",
                f"{cmp_metrics['hit_rate']:.0f}%",
                help="Of bar-to-bar moves, how often forecast direction matched actual direction (50% = coin flip).",
            )
        else:
            cols[3].metric("Direction hit-rate", "—")
        cols[4].metric(
            "Range coverage",
            f"{cmp_metrics['range_coverage']:.0f}%",
            help="Of actual closes, what fraction fell inside the forecast [low, high] band.",
        )

        # Plain-English verdict
        st.markdown("**Verdict**")
        verdict_lines = []
        same_dir = (forecast_ret >= 0) == (actual_ret >= 0)
        verdict_lines.append(
            f"- **Direction**: forecast said **{'up' if forecast_ret >= 0 else 'down'}** ({forecast_ret:+.2f}%), "
            f"reality was **{'up' if actual_ret >= 0 else 'down'}** ({actual_ret:+.2f}%). "
            f"{'✅ Same direction.' if same_dir else '❌ Opposite direction.'}"
        )
        magnitude_err = abs(forecast_ret - actual_ret)
        verdict_lines.append(
            f"- **Magnitude**: off by **{magnitude_err:.2f} percentage points** at the horizon endpoint."
        )
        rc = cmp_metrics["range_coverage"]
        if rc >= 80:
            verdict_lines.append(f"- **Range**: actual price stayed inside the forecast band {rc:.0f}% of the time — well calibrated.")
        elif rc >= 50:
            verdict_lines.append(f"- **Range**: actual price stayed inside the band {rc:.0f}% of the time — band a bit narrow.")
        else:
            verdict_lines.append(f"- **Range**: only {rc:.0f}% of actual closes fell inside the band — model under-estimated volatility.")
        st.markdown("\n".join(verdict_lines))

        with st.expander("Side-by-side bars", expanded=False):
            side = pd.DataFrame({
                "timestamp": forecast_ts.iloc[:n].values,
                "forecast_close": forecast_df["close"].iloc[:n].round(2).values,
                "actual_close":   actuals_df["close"].iloc[:n].round(2).values,
                "abs_err":       (forecast_df["close"].iloc[:n].values - actuals_df["close"].iloc[:n].values).round(2),
                "abs_err_pct":   ((forecast_df["close"].iloc[:n].values - actuals_df["close"].iloc[:n].values) / actuals_df["close"].iloc[:n].values * 100).round(2),
            })
            st.dataframe(side, use_container_width=True, hide_index=True)

    # ============== Forecast-only summary (when no actuals to compare) ==============
    elif forecast_df is not None:
        st.markdown("#### Forecast summary")
        cur = float(as_of_row["close"])
        end = float(forecast_df.iloc[-1]["close"])
        ret = (end - cur) / cur * 100
        max_close = float(forecast_df["close"].max())
        min_close = float(forecast_df["close"].min())
        max_drawdown = (min_close - cur) / cur * 100
        max_runup = (max_close - cur) / cur * 100

        cols = st.columns(5)
        cols[0].metric("End-of-horizon close",  f"{end:,.2f}",       f"{ret:+.2f}%")
        cols[1].metric("Max forecasted close",  f"{max_close:,.2f}", f"{max_runup:+.2f}%")
        cols[2].metric("Min forecasted close",  f"{min_close:,.2f}", f"{max_drawdown:+.2f}%")
        up_frac = float((forecast_df["close"] > cur).mean()) * 100
        cols[3].metric("Bars above as-of",      f"{up_frac:.0f}%",
                       help="Fraction of forecasted bars closing above the as-of close.")
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
