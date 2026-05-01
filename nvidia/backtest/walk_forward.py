"""Walk-forward backtest engine for Kronos.

Causality-correct: for each as-of date, the model only sees bars up to and
including that date, and is scored against bars strictly after.

Public API:
    split_at_as_of(df, as_of_ts, lookback, horizon)
    run_one_forecast(predictor, df, as_of_ts, lookback, horizon, samples, ...)
    walk_forward(predictor, df, as_of_dates, horizon, lookback, samples, ...)
    aggregate(results, label)
"""
from __future__ import annotations

import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


def split_at_as_of(
    df: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    lookback: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Slice df into (context, actuals) at as_of_ts. Snaps to last trading bar."""
    valid = df["timestamps"] <= as_of_ts
    if not valid.any():
        raise ValueError(f"No bars at or before {as_of_ts}")
    end_idx = int(valid.values.nonzero()[0][-1])
    start_idx = max(0, end_idx - lookback + 1)
    cols = ["open", "high", "low", "close", "volume", "amount"]
    context_df = df.iloc[start_idx : end_idx + 1][cols].reset_index(drop=True)
    context_ts = df.iloc[start_idx : end_idx + 1]["timestamps"].reset_index(drop=True)
    actuals_df = df.iloc[end_idx + 1 : end_idx + 1 + horizon][cols].reset_index(drop=True)
    actuals_ts = df.iloc[end_idx + 1 : end_idx + 1 + horizon]["timestamps"].reset_index(drop=True)
    return context_df, context_ts, actuals_df, actuals_ts


def run_one_forecast(
    predictor,
    df: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    lookback: int,
    horizon: int,
    samples: int,
    T: float = 1.0,
    top_p: float = 0.9,
) -> dict[str, Any] | None:
    """Run one as-of forecast. Returns None if not enough actuals available."""
    context_df, context_ts, actuals_df, actuals_ts = split_at_as_of(
        df, as_of_ts, lookback, horizon
    )
    if len(actuals_df) < horizon:
        return None
    if len(context_df) < 100:
        return None

    y_ts = actuals_ts.iloc[:horizon].reset_index(drop=True)
    pred_df = predictor.predict(
        df=context_df,
        x_timestamp=context_ts,
        y_timestamp=y_ts,
        pred_len=horizon,
        T=T,
        top_p=top_p,
        sample_count=samples,
        verbose=False,
    )

    return {
        "as_of_ts": as_of_ts,
        "actual_as_of_ts": context_ts.iloc[-1],
        "context_close": float(context_df["close"].iloc[-1]),
        "forecast_df": pred_df,
        "actuals_df": actuals_df,
        "actuals_ts": actuals_ts,
    }


def compute_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Score a single forecast against actuals."""
    f = result["forecast_df"]
    a = result["actuals_df"]
    cur = result["context_close"]
    n = int(min(len(f), len(a)))

    metrics: dict[str, Any] = {"n_bars": n}

    for col in ("open", "high", "low", "close"):
        denom = np.where(np.abs(a[col].values) < 1e-9, 1e-9, a[col].values)
        metrics[f"mape_{col}"] = float(
            np.mean(np.abs((f[col].values[:n] - a[col].values[:n]) / denom[:n])) * 100
        )

    f_end = float(f["close"].iloc[n - 1])
    a_end = float(a["close"].iloc[n - 1])
    f_ret = (f_end - cur) / cur if cur else 0.0
    a_ret = (a_end - cur) / cur if cur else 0.0
    metrics["forecast_ret"] = float(f_ret * 100)
    metrics["actual_ret"] = float(a_ret * 100)
    metrics["dir_correct"] = bool(np.sign(f_ret) == np.sign(a_ret))

    if n >= 2:
        f_dir = np.sign(f["close"].diff().iloc[1:n].values)
        a_dir = np.sign(a["close"].diff().iloc[1:n].values)
        defined = (f_dir != 0) & (a_dir != 0)
        if defined.any():
            metrics["bar_hit_rate"] = float(np.mean(f_dir[defined] == a_dir[defined]) * 100)
        else:
            metrics["bar_hit_rate"] = None
    else:
        metrics["bar_hit_rate"] = None

    in_band = (
        (a["close"].values[:n] >= f["low"].values[:n])
        & (a["close"].values[:n] <= f["high"].values[:n])
    )
    metrics["range_coverage"] = float(np.mean(in_band) * 100) if n else 0.0

    return metrics


def walk_forward(
    predictor,
    df: pd.DataFrame,
    as_of_dates: Iterable[pd.Timestamp],
    horizon: int,
    lookback: int = 400,
    samples: int = 3,
    label: str = "",
    T: float = 1.0,
    top_p: float = 0.9,
) -> list[dict[str, Any]]:
    """Iterate over as-of dates, run forecast, score, return list of result dicts."""
    as_of_dates = list(as_of_dates)
    results: list[dict[str, Any]] = []
    for i, as_of_ts in enumerate(as_of_dates, 1):
        t0 = time.time()
        try:
            r = run_one_forecast(
                predictor, df, as_of_ts, lookback, horizon, samples, T=T, top_p=top_p
            )
            if r is None:
                print(
                    f"  [{i:>2}/{len(as_of_dates)}] {as_of_ts:%Y-%m-%d}  SKIP (insufficient actuals)",
                    flush=True,
                )
                continue
            m = compute_metrics(r)
            r.update(m)
            results.append(r)
            elapsed = time.time() - t0
            print(
                f"  [{i:>2}/{len(as_of_dates)}] {as_of_ts:%Y-%m-%d}  "
                f"MAPE={m['mape_close']:>5.2f}%  "
                f"dir={'✓' if m['dir_correct'] else '✗'}  "
                f"cov={m['range_coverage']:>3.0f}%  "
                f"({elapsed:.1f}s)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  [{i:>2}/{len(as_of_dates)}] {as_of_ts:%Y-%m-%d}  ERROR: {e}",
                flush=True,
            )
    return results


def aggregate(results: list[dict[str, Any]], label: str = "") -> dict[str, Any] | None:
    """Aggregate metrics across many forecasts."""
    if not results:
        return None

    n = len(results)
    mape_close = np.array([r["mape_close"] for r in results])
    range_cov = np.array([r["range_coverage"] for r in results])
    dir_correct = np.array([r["dir_correct"] for r in results])
    bar_hits = np.array(
        [r["bar_hit_rate"] for r in results if r["bar_hit_rate"] is not None]
    )

    return {
        "label": label,
        "n_forecasts": n,
        "mape_close_mean":   float(mape_close.mean()),
        "mape_close_median": float(np.median(mape_close)),
        "mape_close_std":    float(mape_close.std()),
        "mape_close_min":    float(mape_close.min()),
        "mape_close_max":    float(mape_close.max()),
        "endpoint_dir_pct":  float(dir_correct.mean() * 100),
        "bar_hit_rate_mean": float(bar_hits.mean()) if len(bar_hits) else None,
        "range_coverage_mean": float(range_cov.mean()),
    }
