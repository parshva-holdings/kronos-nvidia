#!/usr/bin/env python3
"""End-to-end Nifty 50 forecast example.

Steps
-----
1. Loads ./data/nifty50/nsei.csv (run `python data/fetch_nifty50.py` first).
2. Forecasts the next `--horizon` bars with Kronos.
3. Plots ground-truth vs forecast and saves to outputs/nifty50_forecast.png.
4. Optionally evaluates against held-out actuals (--holdout).

Run me
------
    python data/fetch_nifty50.py                          # one-time download
    python examples/nifty50_forecast.py --device mps      # Apple Silicon
    python examples/nifty50_forecast.py --device cuda:0   # NVIDIA host
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
if not UPSTREAM.exists():
    sys.exit("upstream/ not found — run scripts/00_bootstrap.sh first")
sys.path.insert(0, str(UPSTREAM))

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402


def autodevice() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/nifty50/nsei.csv")
    p.add_argument("--device", default=os.environ.get("KRONOS_DEVICE") or autodevice())
    p.add_argument("--model", default=os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-base"))
    p.add_argument("--tokenizer", default=os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base"))
    p.add_argument("--lookback", type=int, default=400, help="bars used as context")
    p.add_argument("--horizon", type=int, default=30, help="bars to forecast")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--samples", type=int, default=5,
                   help="MC sample paths (averaged); higher = more stable but slower")
    p.add_argument("--holdout", action="store_true",
                   help="Withhold the last horizon bars and compare forecast vs actuals")
    p.add_argument("--out", default="outputs/nifty50_forecast.png")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(
            f"{csv_path} not found. Run:\n  python data/fetch_nifty50.py"
        )

    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    needed = args.lookback + (args.horizon if args.holdout else 0)
    if len(df) < needed:
        sys.exit(f"Need at least {needed} bars, have {len(df)}. Refetch with more --years.")

    # Slice context (and held-out future, if requested)
    if args.holdout:
        x_df = df.iloc[-needed:-args.horizon][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_ts = df.iloc[-needed:-args.horizon]["timestamps"].reset_index(drop=True)
        y_ts = df.iloc[-args.horizon:]["timestamps"].reset_index(drop=True)
        actuals = df.iloc[-args.horizon:][["open", "high", "low", "close"]].reset_index(drop=True)
    else:
        x_df = df.iloc[-args.lookback:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_ts = df.iloc[-args.lookback:]["timestamps"].reset_index(drop=True)
        # Extend the cadence into the future
        cadence = x_ts.iloc[-1] - x_ts.iloc[-2]
        y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + cadence, periods=args.horizon, freq=cadence))
        actuals = None

    print(f"Nifty 50 forecast")
    print(f"  csv       : {csv_path}")
    print(f"  bars      : {len(df)}  (lookback={args.lookback}, horizon={args.horizon})")
    print(f"  device    : {args.device}")
    print(f"  model     : {args.model}")
    print(f"  samples   : {args.samples}  (MC paths averaged)")
    print(f"  holdout   : {args.holdout}")

    print("\nLoading model...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)

    print("Forecasting...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=args.horizon,
        T=args.temperature,
        top_p=args.top_p,
        sample_count=args.samples,
        verbose=True,
    )

    print("\nForecast head:")
    print(pred_df.head().to_string())

    if actuals is not None:
        diff = (pred_df[["open", "high", "low", "close"]].reset_index(drop=True) - actuals).abs()
        mape = (diff / actuals.abs()).mean() * 100
        print("\nHoldout error (MAPE %):")
        print(mape.to_string())

    # Plot
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        # Recent context (last 100 bars) + forecast line
        ctx_tail = x_df.tail(100).reset_index(drop=True)
        ctx_ts = x_ts.tail(100).reset_index(drop=True)
        ax1.plot(ctx_ts, ctx_tail["close"], label="context close", linewidth=1.2)
        ax1.plot(y_ts, pred_df["close"].values, label="forecast close", linewidth=1.5, color="tab:red")
        if actuals is not None:
            ax1.plot(y_ts, actuals["close"].values, label="actual close", linewidth=1.2, color="tab:green", linestyle="--")
        ax1.set_title(f"Nifty 50 close — {args.lookback}-bar lookback, {args.horizon}-bar forecast")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(y_ts, pred_df["volume"].values, label="forecast volume", color="tab:orange", alpha=0.7)
        ax2.set_ylabel("volume")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f"\nPlot saved → {out}")
    except Exception as e:  # noqa: BLE001
        print(f"\nplotting skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
