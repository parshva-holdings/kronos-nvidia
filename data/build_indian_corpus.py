#!/usr/bin/env python3
"""Convert the NSE corpus CSVs into upstream-Kronos pickle format.

Reads everything under data/nse_corpus/{indices,constituents,universe}/*.csv
and produces train/val/test pickles whose schema matches what
upstream/finetune/dataset.py:QlibDataset expects:

    dict[symbol_str, pandas.DataFrame]
        DataFrame.index is a pandas DatetimeIndex named 'datetime'
        DataFrame columns are exactly: ['open', 'high', 'low', 'close', 'vol', 'amt']

The split boundaries default to:
    train: 1996-07-01 .. 2022-12-31    (~26 years)
    val:   2022-09-01 .. 2024-12-31    (~2 years, with 4-month warmup overlap)
    test:  2024-10-01 .. 2026-12-31    (with 3-month warmup overlap)

The warmup overlap in val/test is required by upstream's lookback window logic.

Usage
-----
    python data/build_indian_corpus.py               # uses defaults
    python data/build_indian_corpus.py --min-bars 600 --train-end 2022-12-31

Output:
    data/processed_datasets/train_data.pkl
    data/processed_datasets/val_data.pkl
    data/processed_datasets/test_data.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True).dt.tz_localize(None)
    df = df.set_index("timestamps").sort_index()
    df.index.name = "datetime"
    # Rename to upstream fine-tune column names
    df = df.rename(columns={"volume": "vol", "amount": "amt"})
    return df[["open", "high", "low", "close", "vol", "amt"]]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/nse_corpus")
    p.add_argument("--out", default="data/processed_datasets")
    p.add_argument("--train-start", default="1996-07-01")
    p.add_argument("--train-end",   default="2022-12-31")
    p.add_argument("--val-start",   default="2022-09-01")
    p.add_argument("--val-end",     default="2024-12-31")
    p.add_argument("--test-start",  default="2024-10-01")
    p.add_argument("--test-end",    default="2026-12-31")
    p.add_argument("--min-bars", type=int, default=600,
                   help="Skip series with fewer than this many bars (after splits).")
    p.add_argument("--feature-list", nargs="+",
                   default=["open", "high", "low", "close", "vol", "amt"])
    args = p.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        sys.exit(f"corpus dir {corpus} not found. Run data/fetch_nse_full.py first.")

    csvs = sorted(corpus.rglob("*.csv"))
    if not csvs:
        sys.exit(f"no CSV files under {corpus}.")
    print(f"==> Loading {len(csvs)} CSVs from {corpus}")

    train, val, test = {}, {}, {}
    skipped_short = skipped_empty = 0
    total_bars = 0

    for csv in csvs:
        symbol = csv.stem.upper()
        try:
            df = load_csv(csv)
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {symbol}: {e}")
            continue
        if df.empty:
            skipped_empty += 1
            continue

        total_bars += len(df)

        tr = df.loc[args.train_start:args.train_end]
        va = df.loc[args.val_start:args.val_end]
        te = df.loc[args.test_start:args.test_end]

        # Skip thin series after slicing (must have enough for at least one window)
        if len(tr) < args.min_bars:
            skipped_short += 1
            continue
        train[symbol] = tr
        if len(va) >= args.min_bars // 4:
            val[symbol] = va
        if len(te) >= args.min_bars // 4:
            test[symbol] = te

    if not train:
        sys.exit("Empty train split — check --train-start/--train-end vs your data.")

    # Save
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{name}_data.pkl"
        with open(path, "wb") as f:
            pickle.dump(d, f)
        bars = sum(len(v) for v in d.values())
        print(f"    {path.name:18s}  {len(d):>4d} symbols  {bars:>10,d} bars")

    print()
    print(f"==> Summary")
    print(f"    Source CSVs read       : {len(csvs)}")
    print(f"    Skipped (short)        : {skipped_short}")
    print(f"    Skipped (empty)        : {skipped_empty}")
    print(f"    Total source bars      : {total_bars:,}")
    print(f"    Train symbols / bars   : {len(train)} / {sum(len(v) for v in train.values()):,}")
    print(f"    Val   symbols / bars   : {len(val)} / {sum(len(v) for v in val.values()):,}")
    print(f"    Test  symbols / bars   : {len(test)} / {sum(len(v) for v in test.values()):,}")
    print()
    print(f"==> Output ready at {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
