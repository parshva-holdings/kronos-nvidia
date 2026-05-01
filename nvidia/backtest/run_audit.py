"""Run a Kronos accuracy audit on Indian markets.

Two presets:
    --mode quick    1 symbol  · 2 horizons · 8 dates each   (~5 min on GPU)
    --mode intense  3 symbols · 3 horizons · 15 dates each  (~25 min on GPU,
                                                            ~90 min on CPU)

Writes a markdown report to outputs/backtest_report.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream"
if UPSTREAM.exists() and str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvidia.backtest.walk_forward import walk_forward, aggregate  # noqa: E402


SYMBOLS_QUICK = {
    "^NSEI": "Nifty 50",
}
SYMBOLS_INTENSE = {
    "^NSEI":     "Nifty 50",
    "^NSEBANK":  "Bank Nifty",
    "RELIANCE.NS": "Reliance",
}


def fetch(symbol: str) -> pd.DataFrame | None:
    import yfinance as yf

    df = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False)
    if df.empty:
        return None
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
    typ = (out["open"] + out["high"] + out["low"] + out["close"]) / 4
    out["amount"] = typ * out["volume"]
    return out.dropna().reset_index(drop=True)


def pick_dates(df: pd.DataFrame, n_dates: int, horizon: int, years: int = 5) -> list[pd.Timestamp]:
    """Pick `n_dates` evenly-spaced as-of dates from the last `years` years,
    leaving room (1.5x horizon) for actuals after each."""
    last = df["timestamps"].iloc[-1]
    end = last - pd.Timedelta(days=int(horizon * 1.5))
    earliest_with_lookback = df["timestamps"].iloc[0] + pd.Timedelta(days=512)
    start = max(earliest_with_lookback, end - pd.Timedelta(days=years * 365))
    if start >= end:
        return []

    candidates = pd.date_range(start, end, periods=n_dates * 3)
    snapped: list[pd.Timestamp] = []
    seen = set()
    for c in candidates:
        match = df["timestamps"][df["timestamps"] <= c]
        if len(match) >= 400:
            ts = match.iloc[-1]
            if ts not in seen:
                snapped.append(ts)
                seen.add(ts)
    # Even sampling across the snapped list
    if len(snapped) <= n_dates:
        return snapped
    step = len(snapped) / n_dates
    return [snapped[int(i * step)] for i in range(n_dates)]


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="intense", choices=["quick", "intense"])
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--horizons", default=None, help="comma-sep e.g. 5,21,63")
    p.add_argument("--n-dates", type=int, default=None)
    p.add_argument("--out", default="outputs/backtest_report.md")
    p.add_argument("--model", default="NeoQuasar/Kronos-base")
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--device", default=None)
    p.add_argument("--years", type=int, default=5)
    args = p.parse_args()

    if args.mode == "quick":
        symbols = SYMBOLS_QUICK
        horizons = [5, 21]
        n_dates = 8
        samples = 3
    else:
        symbols = SYMBOLS_INTENSE
        horizons = [5, 21, 63]
        n_dates = 15
        samples = 5

    if args.samples:
        samples = args.samples
    if args.horizons:
        horizons = [int(x) for x in args.horizons.split(",")]
    if args.n_dates:
        n_dates = args.n_dates

    device = args.device or detect_device()

    print("============================================================")
    print(f"  Kronos Accuracy Audit — {args.mode.upper()} mode")
    print("============================================================")
    print(f"  Symbols : {list(symbols.values())}")
    print(f"  Horizons: {horizons} bars")
    print(f"  Dates   : {n_dates} per (symbol, horizon)")
    print(f"  Samples : {samples} MC paths")
    print(f"  Total   : ~{len(symbols) * len(horizons) * n_dates} forecasts")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print("============================================================")
    print()

    # Load model once
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor

    print(f"Loading {args.model} on {device} ...")
    t0 = time.time()
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print(f"Loaded in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    audit_t0 = time.time()
    all_results: dict[tuple[str, int], tuple[list, dict | None]] = {}

    for sym, label in symbols.items():
        print(f"\n=== {label} ({sym}) ===")
        df = fetch(sym)
        if df is None or len(df) < 600:
            print(f"  not enough data ({0 if df is None else len(df)} bars), skipping")
            continue
        print(
            f"  {len(df)} bars · "
            f"{df['timestamps'].iloc[0]:%Y-%m-%d} → {df['timestamps'].iloc[-1]:%Y-%m-%d}"
        )

        for h in horizons:
            print(f"\n  -- {label} · horizon = {h} bars --")
            dates = pick_dates(df, n_dates, h, years=args.years)
            if not dates:
                print(f"    no valid as-of dates for horizon {h}")
                continue
            print(f"    Picked {len(dates)} as-of dates: "
                  f"{dates[0]:%Y-%m-%d} ... {dates[-1]:%Y-%m-%d}")
            results = walk_forward(
                predictor, df, dates, horizon=h, lookback=400, samples=samples
            )
            agg = aggregate(results, label=f"{label} · h={h}")
            all_results[(sym, h)] = (results, agg)

    print(f"\n=== Audit complete in {(time.time() - audit_t0) / 60:.1f} min ===\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(all_results, args, symbols, horizons, n_dates, samples, device)
    out_path.write_text(report)
    print(f"Report saved: {out_path}")
    print()
    print("============================================================")
    print("  EXECUTIVE SUMMARY")
    print("============================================================")
    print(format_summary(all_results, symbols))
    return 0


def format_summary(all_results, symbols) -> str:
    lines = []
    lines.append(f"{'Symbol':<14} {'Horizon':>8} {'N':>4} {'EndDir%':>8} {'BarHit%':>8} "
                 f"{'MAPEμ%':>8} {'MAPEmed%':>9} {'RangeCov%':>10}")
    lines.append("-" * 80)
    for (sym, h), (_, agg) in all_results.items():
        if agg is None:
            continue
        bar_hit = f"{agg['bar_hit_rate_mean']:>7.1f}" if agg["bar_hit_rate_mean"] else "    —"
        lines.append(
            f"{symbols[sym]:<14} {h:>7d}d {agg['n_forecasts']:>4d} "
            f"{agg['endpoint_dir_pct']:>7.1f} {bar_hit:>8} "
            f"{agg['mape_close_mean']:>7.2f} {agg['mape_close_median']:>8.2f} "
            f"{agg['range_coverage_mean']:>9.1f}"
        )
    return "\n".join(lines)


def generate_report(all_results, args, symbols, horizons, n_dates, samples, device) -> str:
    lines: list[str] = []
    lines.append("# Kronos Accuracy Audit — Indian Markets")
    lines.append("")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_  ")
    lines.append(f"_Mode: **{args.mode}** · Model: `{args.model}` · Device: `{device}` · "
                 f"Lookback: 400 bars · MC samples: {samples}_")
    lines.append("")
    lines.append("> All forecasts use **only bars at or before each as-of date** as context. "
                 "Actuals are bars strictly after. No look-ahead bias.")
    lines.append("")

    # Aggregate
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Symbol | Horizon | N | Endpoint dir hit | Bar-to-bar hit | Mean MAPE close | Median MAPE | Range coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    aggs = []
    for (sym, h), (_, agg) in all_results.items():
        if agg is None:
            continue
        aggs.append(agg)
        bar_hit = f"{agg['bar_hit_rate_mean']:.0f}%" if agg["bar_hit_rate_mean"] else "—"
        lines.append(
            f"| {symbols[sym]} | {h}d | {agg['n_forecasts']} | "
            f"**{agg['endpoint_dir_pct']:.0f}%** | {bar_hit} | "
            f"{agg['mape_close_mean']:.2f}% | {agg['mape_close_median']:.2f}% | "
            f"{agg['range_coverage_mean']:.0f}% |"
        )
    lines.append("")

    # Verdict
    if aggs:
        avg_dir = float(np.mean([a["endpoint_dir_pct"] for a in aggs]))
        avg_mape = float(np.mean([a["mape_close_mean"] for a in aggs]))
        avg_cov = float(np.mean([a["range_coverage_mean"] for a in aggs]))

        lines.append("## Verdict — should we fine-tune for Nifty?")
        lines.append("")
        lines.append(f"**Cross-config averages**: endpoint direction {avg_dir:.0f}% · "
                     f"close MAPE {avg_mape:.2f}% · range coverage {avg_cov:.0f}%")
        lines.append("")

        if avg_dir < 52 and avg_mape > 4:
            verdict = (
                "**Strong case for fine-tuning.** Zero-shot direction is borderline "
                "coin-flip and MAPE is high. The full all-time NSE corpus fine-tune "
                "(~$500-700, ~22h on 8× H100) is well-justified — published Kronos paper "
                "deltas suggest a 30-60% MAPE reduction on the target market."
            )
        elif avg_dir < 55 and avg_mape > 3:
            verdict = (
                "**Fine-tuning is worth the spend.** There's some signal but material "
                "headroom. Spending ~$500-700 to bring MAPE into the 1-2% range is a "
                "good investment if you intend to use this for any decision-making."
            )
        elif avg_dir >= 55 and avg_mape <= 3:
            verdict = (
                "**Zero-shot is already useful.** Fine-tuning still has value for "
                "production hardening, but the immediate ROI on $500-700 is lower. "
                "Consider building on top of zero-shot first."
            )
        else:
            verdict = (
                "**Mixed signal.** Fine-tune to improve consistency across regimes "
                "rather than to lift the headline number."
            )
        lines.append(verdict)
        lines.append("")

    # Glossary
    lines.append("## How to read these metrics")
    lines.append("")
    lines.append("- **Endpoint direction hit**: of the N forecasts, how often did Kronos call "
                 "the direction (up vs down) correctly at the horizon endpoint? 50% = coin flip; "
                 ">55% = signal; >60% = strong.")
    lines.append("- **Bar-to-bar hit**: of all bar-to-bar moves inside each forecast, how often "
                 "did the forecast's direction match the actual.")
    lines.append("- **Mean MAPE close**: average % error on the close price across all forecasted bars.")
    lines.append("- **Range coverage**: of all actual closes, what % fell inside the forecast "
                 "[low, high] band. >70% = volatility band well-calibrated; <50% = under-estimating risk.")
    lines.append("")

    # Detailed per-config tables
    lines.append("## Per-forecast detail")
    lines.append("")
    for (sym, h), (results, agg) in all_results.items():
        if not results:
            continue
        lines.append(f"### {symbols[sym]} · {h}-bar horizon · {len(results)} forecasts")
        lines.append("")
        lines.append("| As-of | Context close | Forecast end | Actual end | F. ret % | A. ret % | Dir | MAPE close | Range cov |")
        lines.append("|---|---:|---:|---:|---:|---:|:---:|---:|---:|")
        for r in results:
            f_close = float(r["forecast_df"]["close"].iloc[-1])
            a_close = float(r["actuals_df"]["close"].iloc[-1])
            dir_emoji = "✅" if r["dir_correct"] else "❌"
            lines.append(
                f"| {r['as_of_ts']:%Y-%m-%d} | {r['context_close']:,.2f} | "
                f"{f_close:,.2f} | {a_close:,.2f} | "
                f"{r['forecast_ret']:+.2f} | {r['actual_ret']:+.2f} | {dir_emoji} | "
                f"{r['mape_close']:.2f}% | {r['range_coverage']:.0f}% |"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
