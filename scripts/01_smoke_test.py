#!/usr/bin/env python3
"""End-to-end smoke test: load Kronos, forecast on synthetic data, print summary.

Runs on CPU, MPS (Apple Silicon), or CUDA. Uses the Kronos source from ./upstream/
(populated by scripts/00_bootstrap.sh). Confirms HF download path works, the
KronosTokenizer + Kronos + KronosPredictor pipeline initializes, and a forecast
returns a DataFrame of the expected shape.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
if not UPSTREAM.exists():
    sys.exit(
        f"upstream/ not found at {UPSTREAM}. Run scripts/00_bootstrap.sh first."
    )
sys.path.insert(0, str(UPSTREAM))

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402


def autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=os.environ.get("KRONOS_DEVICE") or autodetect_device())
    p.add_argument("--model", default=os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small"))
    p.add_argument("--tokenizer", default=os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base"))
    p.add_argument("--data", default=str(ROOT / "data" / "sample_ohlcv.csv"))
    p.add_argument("--max-context", type=int, default=int(os.environ.get("KRONOS_MAX_CONTEXT", 512)))
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--pred-len", type=int, default=120)
    args = p.parse_args()

    print(f"== Kronos smoke test ==")
    print(f"   device       : {args.device}")
    print(f"   model        : {args.model}")
    print(f"   tokenizer    : {args.tokenizer}")
    print(f"   data         : {args.data}")
    print(f"   torch        : {torch.__version__}")
    print(f"   cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   gpu          : {torch.cuda.get_device_name(0)}")

    if not Path(args.data).exists():
        sys.exit(f"sample data not found at {args.data}; run scripts/00_bootstrap.sh first")

    t0 = time.time()
    print("\n[1/4] Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    print(f"      ok ({time.time() - t0:.1f}s)")

    t1 = time.time()
    print("[2/4] Loading model...")
    model = Kronos.from_pretrained(args.model)
    print(f"      ok ({time.time() - t1:.1f}s)")

    print("[3/4] Building predictor...")
    predictor = KronosPredictor(
        model, tokenizer, device=args.device, max_context=args.max_context
    )

    print("[4/4] Forecasting...")
    df = pd.read_csv(args.data)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    if len(df) < args.lookback + args.pred_len:
        sys.exit(
            f"sample data has {len(df)} rows; need at least "
            f"{args.lookback + args.pred_len} for lookback + pred_len"
        )
    x_df = df.loc[: args.lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df.loc[: args.lookback - 1, "timestamps"]
    y_ts = df.loc[args.lookback : args.lookback + args.pred_len - 1, "timestamps"]

    t2 = time.time()
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=args.pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )
    elapsed = time.time() - t2

    assert len(pred_df) == args.pred_len, f"expected {args.pred_len} rows, got {len(pred_df)}"
    assert {"open", "high", "low", "close"}.issubset(pred_df.columns)

    print(f"\n      forecast complete in {elapsed:.2f}s ({elapsed / args.pred_len * 1000:.1f} ms/step)")
    print("      forecast head:")
    print(pred_df.head().to_string())
    print("\n[OK] Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
