#!/usr/bin/env python3
"""Fetch the broadest possible NSE training corpus from Yahoo Finance.

Layers (in order of expense):
  Layer 1: Nifty 50 index — full history back to 1996-07-01
  Layer 2: Nifty Next 50, Bank Nifty, IT, Auto, Pharma, FMCG, Metal, Realty, Energy
           sector indices — max history each (typically post-2000)
  Layer 3: All 50 current Nifty 50 constituents — max history per ticker
  Layer 4: Nifty 500 universe — ~500 stocks, max history each
           (this is the bulk of the training corpus)

For "no cap" mode, run with --layer 4. Total: ~500 series × 5,000-7,500 bars.

Usage
-----
    python data/fetch_nse_full.py --layer 4 --interval 1d --out data/nse_corpus

The output CSVs live under <out-dir>/{indices,constituents,universe}/. Each
file is in inference-format (timestamps, open, high, low, close, volume,
amount). Convert to fine-tune format with build_indian_corpus.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Universe definitions. Refresh from nseindia.com if you need point-in-time
# accuracy; the lists below reflect the index composition as of late 2025.
# ---------------------------------------------------------------------------

NIFTY50_INDEX = "^NSEI"

INDICES = {
    # Verified working as of 2026-05-01. Yahoo retired ^CNXNSE100 and ^CNX500;
    # ^CNX100 is the live Nifty 100 alias. Nifty 500 has no working Yahoo
    # ticker, so we pick up that breadth via the 100+ individual stocks below.
    "^NSEI":       "Nifty 50",
    "^CNX100":     "Nifty 100",
    "^NSEBANK":    "Bank Nifty",
    "^CNXIT":      "Nifty IT",
    "^CNXAUTO":    "Nifty Auto",
    "^CNXPHARMA":  "Nifty Pharma",
    "^CNXFMCG":    "Nifty FMCG",
    "^CNXMETAL":   "Nifty Metal",
    "^CNXREALTY":  "Nifty Realty",
    "^CNXENERGY":  "Nifty Energy",
    "^CNXFIN":     "Nifty Financial Services",
    "^CNXCMDT":    "Nifty Commodities",
    "^CNXMNC":     "Nifty MNC",
    "^CNXSERVICE": "Nifty Services",
    "^CRSLDX":     "Nifty Midcap 100",
    "^CRSMID":     "Nifty Midcap 150",
    "^NSMIDCP":    "Nifty Midcap 50",
}

NIFTY50 = [
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

# Historical Nifty 50 names that have rotated out (gives the model exposure to
# survivorship-bias-free regime data). Add tickers here as you discover them.
NIFTY50_HISTORICAL = [
    "ZEEL", "VEDL", "GAIL", "IOC", "UPL", "HDFC", "DLF",
    "TATAPOWER", "ACC", "AMBUJACEM", "BHEL", "IDEA", "IDFCFIRSTB",
    "PNB", "BANKBARODA", "GSPL", "MOTHERSON",
]

# Nifty Next 50 (mid-large caps)
NEXT50 = [
    "ABB", "ADANIPOWER", "ADANIGREEN", "ADANITRANS", "ATGL",
    "ABBOTINDIA", "AMBUJACEM", "BAJAJHLDNG", "BANKBARODA", "BERGEPAINT",
    "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "DLF",
    "DABUR", "GAIL", "GODREJCP", "HAVELLS", "HAL",
    "HINDPETRO", "ICICIGI", "ICICIPRULI", "INDHOTEL", "IOC",
    "IRCTC", "JINDALSTEL", "JIOFIN", "LICI", "LODHA",
    "LTIM", "MARICO", "MOTHERSON", "NAUKRI", "PAGEIND",
    "PIIND", "PIDILITIND", "POLYCAB", "PFC", "PNB",
    "RECLTD", "SIEMENS", "SRF", "TVSMOTOR", "TATAPOWER",
    "TORNTPHARM", "UNITDSPR", "VBL", "VEDL", "ZOMATO",
]

# Bank Nifty constituents (all 12)
BANKNIFTY = [
    "AXISBANK", "BANKBARODA", "FEDERALBNK", "HDFCBANK",
    "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK",
    "PNB", "AUBANK", "SBIN", "CANBK",
]

# A broad mid-cap pool to round out the universe
EXTRA_MIDCAPS = [
    "ABCAPITAL", "AARTIIND", "ABFRL", "ACC", "ADANIWILMAR",
    "ALKEM", "AMARAJABAT", "APLAPOLLO", "ASHOKLEY", "ASTRAL",
    "AUROPHARMA", "BALKRISIND", "BANDHANBNK", "BATAINDIA", "BEL",
    "BHEL", "BIOCON", "CAMS", "COFORGE", "CONCOR",
    "COROMANDEL", "CUMMINSIND", "DEEPAKNTR", "DELHIVERY", "DIXON",
    "ESCORTS", "EXIDEIND", "FACT", "GUJGASLTD", "HDFCAMC",
    "IDBI", "IDFC", "IEX", "INDIANB", "INDIGO",
    "IRCTC", "IRFC", "JKCEMENT", "JSWENERGY", "JUBLFOOD",
    "L&TFH", "LALPATHLAB", "LICHSGFIN", "LUPIN", "M&MFIN",
    "MANAPPURAM", "MAXHEALTH", "MCDOWELL-N", "METROPOLIS", "MFSL",
    "MGL", "MPHASIS", "MRF", "MUTHOOTFIN", "NAVINFLUOR",
    "NMDC", "NYKAA", "OBEROIRLTY", "OFSS", "PAYTM",
    "PEL", "PERSISTENT", "PETRONET", "PFIZER", "PIDILITIND",
    "POLICYBZR", "PRESTIGE", "RAMCOCEM", "RBLBANK", "SBICARD",
    "SHREECEM", "SOLARINDS", "STARHEALTH", "SUNDARMFIN", "SUNTV",
    "SUPREMEIND", "SYNGENE", "TATACOMM", "TATAELXSI", "TIINDIA",
    "TORNTPOWER", "TRIVENI", "TVSMOTOR", "UBL", "UNIONBANK",
    "VOLTAS", "WHIRLPOOL", "YESBANK", "ZYDUSLIFE",
]


def _yf_symbol(ticker: str) -> str:
    """Map our short tickers to Yahoo's NSE convention (.NS suffix)."""
    if ticker.startswith("^"):
        return ticker            # index symbol
    return f"{ticker}.NS"


def _fetch_one(yf, ticker: str, period: str, interval: str, out_path: Path) -> int:
    """Returns row count written. Skips empty / failed pulls."""
    sym = _yf_symbol(ticker)
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {ticker}: {e}")
        return 0
    if df.empty:
        return 0

    import pandas as pd
    out = pd.DataFrame({
        "timestamps": df.index,
        "open":   df["Open"].astype(float),
        "high":   df["High"].astype(float),
        "low":    df["Low"].astype(float),
        "close":  df["Close"].astype(float),
        "volume": df["Volume"].fillna(0).astype(float),
    })
    # Approximate "amount" (turnover) the way the upstream fine-tune pipeline does:
    # typical price × volume. Match upstream so fine-tune statistics align.
    typ = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0
    out["amount"] = typ * out["volume"]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return len(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=3, choices=[1, 2, 3, 4],
                   help=("1=Nifty 50 index only; "
                         "2=+sector indices; "
                         "3=+Nifty 50 constituents (current+historical); "
                         "4=+Nifty Next 50 + Bank Nifty + mid-cap pool (max corpus)"))
    p.add_argument("--interval", default="1d",
                   help="1d | 1h | 30m | 15m | 5m | 1m. Intraday capped at ~60d by Yahoo.")
    p.add_argument("--period", default="max",
                   help="Yahoo period string. Default 'max' = full history.")
    p.add_argument("--out", default="data/nse_corpus")
    p.add_argument("--sleep", type=float, default=0.4,
                   help="Seconds between Yahoo requests (rate-limit cushion)")
    args = p.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed:  pip install -e '.[india]'")

    out = Path(args.out)
    counts = {"indices": 0, "constituents": 0, "universe": 0}
    bars = {"indices": 0, "constituents": 0, "universe": 0}

    # Build the target list per layer
    targets: list[tuple[str, str, str]] = []  # (ticker, dest_subdir, label)

    if args.layer >= 1:
        targets.append((NIFTY50_INDEX, "indices", "Nifty 50 index"))
    if args.layer >= 2:
        for k, v in INDICES.items():
            if k == NIFTY50_INDEX:
                continue
            targets.append((k, "indices", v))
    if args.layer >= 3:
        for t in NIFTY50:
            targets.append((t, "constituents", f"Nifty50:{t}"))
        for t in NIFTY50_HISTORICAL:
            targets.append((t, "constituents", f"Nifty50-hist:{t}"))
    if args.layer >= 4:
        seen = set(t for t, _, _ in targets)
        for t in NEXT50 + BANKNIFTY + EXTRA_MIDCAPS:
            if t in seen:
                continue
            seen.add(t)
            targets.append((t, "universe", t))

    print(f"==> Fetching {len(targets)} series (layer={args.layer}, period={args.period}, interval={args.interval})")
    print(f"    Output: {out}")
    print()

    for i, (ticker, subdir, label) in enumerate(targets, 1):
        # File name: clean the ticker
        clean = ticker.replace(".NS", "").replace("^", "").replace("&", "AND").lower()
        out_path = out / subdir / f"{clean}.csv"
        print(f"[{i:3d}/{len(targets)}] {label:30s} -> {out_path.relative_to(out.parent)}", end=" ", flush=True)
        n = _fetch_one(yf, ticker, args.period, args.interval, out_path)
        bars[subdir] += n
        if n:
            counts[subdir] += 1
            print(f"{n:>6d} bars")
        else:
            print("EMPTY")
        time.sleep(args.sleep)

    print()
    print("==> Summary")
    for k in counts:
        print(f"    {k:14s}: {counts[k]:>4d} series, {bars[k]:>10,d} total bars")
    grand = sum(bars.values())
    print(f"    {'TOTAL':14s}: {sum(counts.values()):>4d} series, {grand:>10,d} total bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
