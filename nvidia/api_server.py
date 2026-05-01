"""FastAPI gateway that wraps Kronos behind a clean JSON API.

Endpoints
---------
GET  /health              liveness + readiness + device info
GET  /v1/models           list of loaded model id(s)
POST /v1/forecast         single-series forecast
POST /v1/forecast/batch   parallel forecast across multiple series

The model is loaded once at startup (lazy on first request if `KRONOS_LAZY=1`)
and shared across requests. Sampling parameters are per-request. CUDA is
auto-detected; on Apple Silicon `mps` is used; else CPU.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
if UPSTREAM.exists() and str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

# Imported lazily inside lifespan so the module loads even when upstream/ is missing
# (e.g. `python -c "import nvidia.api_server"` for tests).
Kronos = None  # type: ignore[assignment]
KronosTokenizer = None  # type: ignore[assignment]
KronosPredictor = None  # type: ignore[assignment]


def _autodevice() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------- Schemas ----------

class Bar(BaseModel):
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    amount: Optional[float] = None


class ForecastRequest(BaseModel):
    bars: List[Bar] = Field(..., min_length=1, description="Historical OHLCV bars (lookback)")
    horizon: int = Field(120, ge=1, le=512, description="How many future bars to predict")
    future_timestamps: Optional[List[str]] = Field(
        None,
        description=(
            "Optional explicit ISO-8601 timestamps for the predicted bars. "
            "If omitted, the server extends the input cadence."
        ),
    )
    temperature: float = Field(1.0, ge=0.05, le=3.0)
    top_p: float = Field(0.9, ge=0.05, le=1.0)
    sample_count: int = Field(1, ge=1, le=10, description="MC samples averaged")
    seed: Optional[int] = None


class BatchForecastRequest(BaseModel):
    series: List[ForecastRequest]


class ForecastBar(Bar):
    pass


class ForecastResponse(BaseModel):
    model: str
    device: str
    horizon: int
    elapsed_ms: float
    bars: List[ForecastBar]


class BatchForecastResponse(BaseModel):
    model: str
    device: str
    elapsed_ms: float
    series: List[ForecastResponse]


# ---------- App state ----------

class _State:
    model_id: str = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
    tokenizer_id: str = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
    device: str = os.environ.get("KRONOS_DEVICE") or _autodevice()
    max_context: int = int(os.environ.get("KRONOS_MAX_CONTEXT", 512))
    predictor: Any = None  # KronosPredictor

    @classmethod
    def loaded(cls) -> bool:
        return cls.predictor is not None


def _load() -> None:
    global Kronos, KronosTokenizer, KronosPredictor
    if Kronos is None:
        if not UPSTREAM.exists():
            raise RuntimeError(
                f"upstream/ directory not found at {UPSTREAM}. "
                "Run scripts/00_bootstrap.sh first."
            )
        from model import Kronos as _K, KronosTokenizer as _KT, KronosPredictor as _KP
        Kronos = _K
        KronosTokenizer = _KT
        KronosPredictor = _KP

    print(f"[startup] loading tokenizer={_State.tokenizer_id}", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(_State.tokenizer_id)

    print(f"[startup] loading model={_State.model_id} on {_State.device}", flush=True)
    model = Kronos.from_pretrained(_State.model_id)

    _State.predictor = KronosPredictor(
        model, tokenizer, device=_State.device, max_context=_State.max_context
    )
    print(f"[startup] ready (device={_State.device})", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if os.environ.get("KRONOS_LAZY", "0") != "1":
        _load()
    yield


app = FastAPI(
    title="Kronos × NVIDIA Inference Gateway",
    description=(
        "OpenAI-style JSON gateway over the Kronos financial foundation model. "
        "See /docs for OpenAPI."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict:
    info = {
        "ok": True,
        "loaded": _State.loaded(),
        "device": _State.device,
        "model": _State.model_id,
        "tokenizer": _State.tokenizer_id,
        "max_context": _State.max_context,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "data": [
            {
                "id": _State.model_id,
                "tokenizer": _State.tokenizer_id,
                "max_context": _State.max_context,
                "device": _State.device,
            }
        ]
    }


def _bars_to_df(bars: List[Bar]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame([b.model_dump() for b in bars])
    df["timestamps"] = pd.to_datetime(df["timestamp"])
    if "volume" not in df or df["volume"].isna().all():
        df["volume"] = 0.0
    if "amount" not in df or df["amount"].isna().all():
        df["amount"] = 0.0
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    return df.drop(columns=["timestamps"]), df["timestamps"]


def _infer_future_ts(history_ts: pd.Series, horizon: int) -> pd.Series:
    if len(history_ts) < 2:
        raise HTTPException(400, "Need at least 2 historical bars to infer cadence")
    cadence = history_ts.iloc[-1] - history_ts.iloc[-2]
    last = history_ts.iloc[-1]
    return pd.Series(pd.date_range(start=last + cadence, periods=horizon, freq=cadence))


def _do_forecast(req: ForecastRequest) -> ForecastResponse:
    if not _State.loaded():
        _load()
    if req.seed is not None:
        torch.manual_seed(req.seed)

    x_df, x_ts = _bars_to_df(req.bars)
    if req.future_timestamps:
        y_ts = pd.to_datetime(pd.Series(req.future_timestamps))
        if len(y_ts) != req.horizon:
            raise HTTPException(400, "future_timestamps length must equal horizon")
    else:
        y_ts = _infer_future_ts(x_ts, req.horizon)

    t0 = time.time()
    pred_df = _State.predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=req.horizon,
        T=req.temperature,
        top_p=req.top_p,
        sample_count=req.sample_count,
        verbose=False,
    )
    elapsed_ms = (time.time() - t0) * 1000

    bars: List[ForecastBar] = []
    for ts, row in zip(y_ts, pred_df.itertuples(index=False)):
        bars.append(
            ForecastBar(
                timestamp=ts.isoformat(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(getattr(row, "volume", 0.0)),
                amount=float(getattr(row, "amount", 0.0)),
            )
        )

    return ForecastResponse(
        model=_State.model_id,
        device=_State.device,
        horizon=req.horizon,
        elapsed_ms=round(elapsed_ms, 2),
        bars=bars,
    )


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    return _do_forecast(req)


@app.post("/v1/forecast/batch", response_model=BatchForecastResponse)
def forecast_batch(req: BatchForecastRequest) -> BatchForecastResponse:
    if not req.series:
        raise HTTPException(400, "Empty batch")
    if not _State.loaded():
        _load()

    t0 = time.time()
    # All series must share lookback length and horizon to use predict_batch.
    horizons = {s.horizon for s in req.series}
    lookbacks = {len(s.bars) for s in req.series}
    use_batch_api = (
        len(horizons) == 1
        and len(lookbacks) == 1
        and all(s.temperature == req.series[0].temperature for s in req.series)
        and all(s.top_p == req.series[0].top_p for s in req.series)
        and all(s.sample_count == req.series[0].sample_count for s in req.series)
    )

    if use_batch_api:
        df_list, x_ts_list, y_ts_list = [], [], []
        for s in req.series:
            x_df, x_ts = _bars_to_df(s.bars)
            df_list.append(x_df)
            x_ts_list.append(x_ts)
            y_ts_list.append(
                pd.to_datetime(pd.Series(s.future_timestamps))
                if s.future_timestamps
                else _infer_future_ts(x_ts, s.horizon)
            )

        pred_list = _State.predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_ts_list,
            y_timestamp_list=y_ts_list,
            pred_len=req.series[0].horizon,
            T=req.series[0].temperature,
            top_p=req.series[0].top_p,
            sample_count=req.series[0].sample_count,
            verbose=False,
        )

        outs: List[ForecastResponse] = []
        for s, y_ts, pred_df in zip(req.series, y_ts_list, pred_list):
            bars = [
                ForecastBar(
                    timestamp=ts.isoformat(),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(getattr(row, "volume", 0.0)),
                    amount=float(getattr(row, "amount", 0.0)),
                )
                for ts, row in zip(y_ts, pred_df.itertuples(index=False))
            ]
            outs.append(
                ForecastResponse(
                    model=_State.model_id,
                    device=_State.device,
                    horizon=s.horizon,
                    elapsed_ms=0.0,  # filled in aggregate below
                    bars=bars,
                )
            )
    else:
        # Fallback: sequential calls
        outs = [_do_forecast(s) for s in req.series]

    elapsed_ms = (time.time() - t0) * 1000
    return BatchForecastResponse(
        model=_State.model_id,
        device=_State.device,
        elapsed_ms=round(elapsed_ms, 2),
        series=outs,
    )
