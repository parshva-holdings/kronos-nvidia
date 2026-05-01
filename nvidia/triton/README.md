# Production Serving with NVIDIA Triton Inference Server

For multi-tenant production where you need batching, dynamic concurrency, retries, GPU sharing across models, and Prometheus metrics, the right NVIDIA primitive is [Triton Inference Server](https://github.com/triton-inference-server/server). Because Kronos has a custom tokenizer and an autoregressive sampling loop, the cleanest integration is the **Python backend** — Triton hosts the model in a managed Python interpreter and gives you HTTP/gRPC out of the box.

## What's here

```
nvidia/triton/
├── README.md                                       (this file)
└── model_repository/
    └── kronos_predictor/
        ├── config.pbtxt                            Triton model config
        └── 1/
            └── model.py                            Python backend implementation
```

The directory layout is fixed by Triton: `model_repository/<name>/<version>/model.py` plus `config.pbtxt` at the model root.

## Run locally on an NVIDIA host

```bash
# 1. Login to NGC and pull Triton (matches our pytorch base CUDA major version)
docker login nvcr.io --username '$oauthtoken' --password "$NGC_API_KEY"
docker pull nvcr.io/nvidia/tritonserver:25.11-py3

# 2. Start Triton with our model repo and the upstream Kronos source mounted in
docker run --rm --gpus all --ipc=host --shm-size=8gb \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)/nvidia/triton/model_repository":/models \
  -v "$(pwd)/upstream":/opt/kronos_src \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e KRONOS_MODEL=NeoQuasar/Kronos-small \
  -e KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base \
  -e KRONOS_DEVICE=cuda:0 \
  -e PYTHONPATH=/opt/kronos_src \
  nvcr.io/nvidia/tritonserver:25.11-py3 \
  tritonserver --model-repository=/models --strict-readiness=false
```

Triton ports: `8000` HTTP, `8001` gRPC, `8002` Prometheus metrics.

## Smoke test

```bash
curl -s localhost:8000/v2/health/ready && echo OK

# JSON inference (HTTP). Provide bars as flattened [N*6] floats, plus horizon.
python - <<'PY'
import json, urllib.request, numpy as np
N = 400
bars = np.column_stack([
    np.linspace(100, 110, N),                 # open
    np.linspace(100, 110, N) + 0.4,           # high
    np.linspace(100, 110, N) - 0.4,           # low
    np.linspace(100, 110, N) + 0.1,           # close
    np.full(N, 1000.0),                        # volume
    np.full(N, 100100.0),                      # amount
]).astype(np.float32)
ts = np.arange(N, dtype=np.int64) * 300       # 5-min bars in seconds since epoch
payload = {
  "inputs": [
    {"name": "bars",      "shape": list(bars.shape),  "datatype": "FP32", "data": bars.flatten().tolist()},
    {"name": "x_ts",      "shape": [N],               "datatype": "INT64","data": ts.tolist()},
    {"name": "horizon",   "shape": [1],               "datatype": "INT32","data": [120]},
    {"name": "T",         "shape": [1],               "datatype": "FP32", "data": [1.0]},
    {"name": "top_p",     "shape": [1],               "datatype": "FP32", "data": [0.9]},
    {"name": "samples",   "shape": [1],               "datatype": "INT32","data": [1]},
  ]
}
req = urllib.request.Request(
    "http://localhost:8000/v2/models/kronos_predictor/infer",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}
)
print(urllib.request.urlopen(req).read()[:400])
PY
```

## On Brev

The Triton path also works as a Brev Launchable — replace `nvcr.io/nvidia/pytorch:25.11-py3` with `nvcr.io/nvidia/tritonserver:25.11-py3` in the compose file and override `command:` to `tritonserver --model-repository=/models`. Expose ports 8000/8001/8002 instead of 7070/8000.

## Why Python backend, not TensorRT-LLM / ONNX?

| Path | Works for Kronos? | Why / why not |
|------|------------------|--------------|
| **TensorRT-LLM / NIM** | No (out of the box) | Assumes HuggingFace causal-LM signature with a standard text tokenizer. Kronos's tokenizer is hierarchical/discrete over OHLCV and the predictor is custom — TRT-LLM's model converters don't know it. |
| **ONNX export → TensorRT** | Partial | The transformer body could export, but the autoregressive sampling loop with MC paths and the inverse tokenization need Python. You'd end up with a graph for the encoder + a Python loop, which Python backend already gives you. |
| **Python backend** ✅ | Yes | Drops `KronosPredictor` straight in. Triton handles batching/queuing/metrics/replicas. ~10–15% slower than a hypothetical TRT graph but the autoregressive loop dominates anyway. |
| **vLLM / SGLang inside NIM** | No | Same reason as TRT-LLM — designed for text LLMs. |

If pure latency becomes critical, the surgical optimization is to export *only the transformer forward* to TensorRT and keep the sampling/tokenization in Python — see the optional `nvidia/tensorrt/` directory (left as an exercise; the hot path is `upstream/model/kronos.py:Kronos.forward`).

## Scaling

Triton's `instance_group` in `config.pbtxt` controls how many copies of the model run on the GPU. For a 24.7M-param Kronos-small you can comfortably run 4–8 instances on an L40S, then dynamic batching ([`dynamic_batching{}`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher) in the config) coalesces concurrent requests. For Kronos-base (102M) start at 2–3 instances and tune.
