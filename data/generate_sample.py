#!/usr/bin/env python3
"""Generate a synthetic OHLCV dataset for smoke tests.

Produces a CSV in the format Kronos expects: timestamps, open, high, low, close,
volume, amount. The series follows a geometric Brownian motion with intra-bar
spread inflation so high/low straddle open/close realistically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=600)
    p.add_argument("--start", default="2024-01-01 09:30:00")
    p.add_argument("--freq", default="5min")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/sample_ohlcv.csv")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.rows
    ts = pd.date_range(args.start, periods=n, freq=args.freq)

    # Geometric brownian motion for close
    mu, sigma = 0.0, 0.005
    log_ret = rng.normal(mu, sigma, size=n)
    close = 100.0 * np.exp(np.cumsum(log_ret))

    # Open ≈ previous close
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Intra-bar spread proportional to recent volatility
    spread = np.abs(rng.normal(0, sigma * 1.5, size=n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    # Volume / amount with mild correlation to absolute return
    base_vol = rng.lognormal(mean=10.0, sigma=0.3, size=n)
    volume = base_vol * (1 + 4 * np.abs(log_ret))
    amount = volume * (open_ + close) / 2

    df = pd.DataFrame(
        {
            "timestamps": ts,
            "open": open_.round(4),
            "high": high.round(4),
            "low": low.round(4),
            "close": close.round(4),
            "volume": volume.round(2),
            "amount": amount.round(2),
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out}")
    print(df.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
