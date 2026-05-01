"""Triton Inference Server Python backend for Kronos.

Wraps `KronosPredictor` so Triton can serve it over HTTP/gRPC with batching,
metrics, and replica management. The model + tokenizer + device come from env:

    KRONOS_MODEL          (default: NeoQuasar/Kronos-small)
    KRONOS_TOKENIZER      (default: NeoQuasar/Kronos-Tokenizer-base)
    KRONOS_DEVICE         (default: cuda:0)
    KRONOS_MAX_CONTEXT    (default: 512)
    PYTHONPATH must include the upstream Kronos source (e.g. /opt/kronos_src).

Triton calls:
    initialize() once per replica
    execute(requests) for each batch of inference requests
    finalize() on shutdown
"""
from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import pandas as pd

# `triton_python_backend_utils` is only available inside the Triton container.
# Import lazily so unit tests can import the file without it.
try:
    import triton_python_backend_utils as pb_utils  # type: ignore
except ImportError:  # pragma: no cover
    pb_utils = None  # type: ignore[assignment]


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        # Defer torch/Kronos imports until init so Triton's per-replica process
        # gets its own CUDA context.
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor

        self.torch = torch
        model_id = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
        tok_id = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
        device = os.environ.get("KRONOS_DEVICE", "cuda:0")
        max_ctx = int(os.environ.get("KRONOS_MAX_CONTEXT", 512))

        # Triton tells us our instance index; we can pin to a specific GPU when
        # `instance_group { count: N kind: KIND_GPU }` runs N replicas.
        try:
            instance_id = int(args.get("model_instance_device_id", 0))
            if device.startswith("cuda"):
                device = f"cuda:{instance_id}"
        except Exception:
            pass

        print(f"[kronos_predictor] init: model={model_id} tok={tok_id} dev={device}")
        tokenizer = KronosTokenizer.from_pretrained(tok_id)
        model = Kronos.from_pretrained(model_id)
        self.predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=max_ctx
        )
        self.device = device
        self.model_id = model_id

    # ------------- helpers -------------

    @staticmethod
    def _get_input_np(request, name: str, default: np.ndarray | None = None) -> np.ndarray | None:
        t = pb_utils.get_input_tensor_by_name(request, name)
        if t is None:
            return default
        return t.as_numpy()

    @staticmethod
    def _bars_to_df(bars: np.ndarray) -> pd.DataFrame:
        # bars: [N, 6] = [open, high, low, close, volume, amount]
        return pd.DataFrame(
            bars, columns=["open", "high", "low", "close", "volume", "amount"]
        )

    @staticmethod
    def _epoch_to_ts(epoch_seconds: np.ndarray) -> pd.Series:
        return pd.to_datetime(epoch_seconds, unit="s")

    @staticmethod
    def _infer_future_ts(history_ts: pd.Series, horizon: int) -> pd.Series:
        if len(history_ts) < 2:
            raise ValueError("Need >= 2 historical bars to infer cadence")
        cadence = history_ts.iloc[-1] - history_ts.iloc[-2]
        last = history_ts.iloc[-1]
        return pd.Series(pd.date_range(start=last + cadence, periods=horizon, freq=cadence))

    # ------------- main entry -------------

    def execute(self, requests: List[Any]) -> List[Any]:
        responses = []
        for req in requests:
            try:
                bars = self._get_input_np(req, "bars")
                x_ts_epoch = self._get_input_np(req, "x_ts")
                horizon = int(self._get_input_np(req, "horizon")[0])
                T = float(self._get_input_np(req, "T", np.array([1.0], dtype=np.float32))[0])
                top_p = float(self._get_input_np(req, "top_p", np.array([0.9], dtype=np.float32))[0])
                samples = int(self._get_input_np(req, "samples", np.array([1], dtype=np.int32))[0])

                df = self._bars_to_df(bars)
                x_ts = self._epoch_to_ts(x_ts_epoch)
                y_ts = self._infer_future_ts(x_ts, horizon)

                pred_df = self.predictor.predict(
                    df=df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=horizon,
                    T=T,
                    top_p=top_p,
                    sample_count=samples,
                    verbose=False,
                )

                forecast = pred_df[["open", "high", "low", "close", "volume", "amount"]].to_numpy().astype(np.float32)
                y_ts_epoch = (y_ts.astype("int64") // 10**9).to_numpy().astype(np.int64)

                out_forecast = pb_utils.Tensor("forecast", forecast)
                out_yts = pb_utils.Tensor("y_ts", y_ts_epoch)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_forecast, out_yts]))
            except Exception as e:  # noqa: BLE001
                err = pb_utils.TritonError(f"kronos_predictor failed: {e}")
                responses.append(pb_utils.InferenceResponse(error=err))
        return responses

    def finalize(self) -> None:
        # PyTorch will free CUDA memory when the predictor goes out of scope.
        self.predictor = None
