#!/usr/bin/env python3
"""Fetch Nifty 50 OHLCV from Yahoo Finance into Kronos-ready CSVs.

Examples
--------
# Just the index (5 years daily, default)
python data/fetch_nifty50.py

# Index + all 50 constituents, daily, 10 years
python data/fetch_nifty50.py --constituents --years 10

# 5-minute intraday (Yahoo limits intraday to ~60 days)
python data/fetch_nifty50.py --interval 5m --years 0.16

The output schema matches what Kronos expects:
  timestamps, open, high, low, close, volume, amount

`amount` is synthesized as close × volume since Yahoo doesn't expose turnover.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Current (Nov 2025) Nifty 50 constituents on NSE. The index is rebalanced
# semi-annually — refresh this list from https://www.nseindia.com if you need
# point-in-time accuracy. Yahoo suffix `.NS` = NSE listing.
NIFTY50_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "BHARTIARTL.NS", "LT.NS", "ITC.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
    "AXISBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "M&M.NS", "MARUTI.NS",
    "ASIANPAINT.NS", "SUNPHARMA.NS", "HCLTECH.NS", "TITAN.NS", "TATAMOTORS.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "NESTLEIND.NS", "WIPRO.NS",
    "ADANIENT.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "BAJAJFINSV.NS", "ONGC.NS",
    "COALINDIA.NS", "TECHM.NS", "GRASIM.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS",
    "DRREDDY.NS", "ADANIPORTS.NS", "CIPLA.NS", "EICHERMOT.NS", "HINDALCO.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "DIVISLAB.NS", "BPCL.NS", "BRITANNIA.NS",
    "TATACONSUM.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "TRENT.NS", "SHRIRAMFIN.NS",
]
NIFTY50_INDEX = "^NSEI"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default=NIFTY50_INDEX,
                   help=f"Symbol to fetch when --constituents is not set. Default: {NIFTY50_INDEX}")
    p.add_argument("--constituents", action="store_true",
                   help="Fetch all 50 stocks individually into separate CSVs")
    p.add_argument("--interval", default="1d",
                   help="Yahoo interval: 1d, 1h, 30m, 15m, 5m, 1m. Intraday capped at ~60d.")
    p.add_argument("--years", type=float, default=5.0)
    p.add_argument("--out-dir", default="data/nifty50")
    args = p.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit(
            "yfinance not installed. Run:  pip install yfinance\n"
            "(or `pip install -e '.[india]'` after we update pyproject.toml)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    period = f"{int(args.years * 365)}d" if args.years >= 1 else f"{int(args.years * 365)}d"

    if args.constituents:
        targets = NIFTY50_TICKERS
        # Yahoo allows multi-symbol downloads but the resulting frame is awkward
        # for our per-series CSV format, so we loop. Be polite with the API.
        import time
        for i, t in enumerate(targets, 1):
            print(f"[{i:2d}/{len(targets)}] {t} ...", end=" ", flush=True)
            try:
                df = yf.Ticker(t).history(period=period, interval=args.interval, auto_adjust=False)
                if df.empty:
                    print("EMPTY (delisted or no data for this interval)")
                    continue
                _write_kronos_csv(df, out_dir / f"{t.replace('.NS','').replace('^','').lower()}.csv")
                print(f"{len(df)} bars")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL: {e}")
            time.sleep(0.5)  # Yahoo rate-limit cushion
    else:
        print(f"Fetching {args.ticker} ({period} @ {args.interval}) ...")
        df = yf.Ticker(args.ticker).history(period=period, interval=args.interval, auto_adjust=False)
        if df.empty:
            sys.exit(f"No data returned for {args.ticker}. Try a wider --years window.")
        out = out_dir / f"{args.ticker.replace('.NS','').replace('^','').lower()}.csv"
        _write_kronos_csv(df, out)
        print(f"wrote {len(df)} bars → {out}")

    return 0


def _write_kronos_csv(df, path: Path) -> None:
    import pandas as pd

    out = pd.DataFrame(
        {
            "timestamps": df.index,
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": df["Close"].astype(float),
            "volume": df["Volume"].fillna(0).astype(float),
            "amount": (df["Close"].astype(float) * df["Volume"].fillna(0).astype(float)),
        }
    )
    # Drop rows with nan OHLC (holidays sometimes return blank rows)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
