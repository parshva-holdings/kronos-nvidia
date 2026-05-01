# Kronos × NVIDIA — Full-Capability Deployment Kit

This project is an opinionated, end-to-end recipe for running [**Kronos**](https://github.com/shiyu-coder/Kronos) — the AAAI 2026 foundation model for financial K-line forecasting ([paper](https://arxiv.org/abs/2508.02739)) — on NVIDIA infrastructure exposed through [build.nvidia.com](https://build.nvidia.com) (NIM, NGC, Brev Launchables, DGX Cloud).

It does not vendor Kronos source code. It clones the upstream repo on demand and adds the NVIDIA-specific layers that the upstream project does not ship: a CUDA-pinned Docker image, a one-click Brev Launchable, an OpenAI-compatible FastAPI inference gateway, and a Triton Python-backend serving graph for production.

## What is Kronos (one-paragraph context)

Kronos is a decoder-only autoregressive Transformer trained on **12B+ K-line records from 45 global exchanges**. It uses a two-stage design: a `KronosTokenizer` that discretizes continuous OHLCV into hierarchical tokens, then an autoregressive predictor that does Monte-Carlo path sampling. Open weights on [HuggingFace `NeoQuasar`](https://huggingface.co/NeoQuasar): `Kronos-mini` (4.1M, 2k ctx), `Kronos-small` (24.7M, 512 ctx), `Kronos-base` (102.3M, 512 ctx). MIT licensed.

## Why this kit exists

Upstream Kronos ships a Flask demo and `torchrun` fine-tune scripts but no production-serving story and no NVIDIA cloud automation. NIM's TRT-LLM/vLLM optimized paths assume a HuggingFace causal-LM signature, which Kronos does not have (custom tokenizer + custom architecture). This kit fills the gap with three NVIDIA deployment tiers:

| Tier | Use case | NVIDIA service | Hardware sweet-spot |
|------|----------|----------------|--------------------|
| **1. Interactive notebook** | Research, fine-tuning, ad-hoc forecasts | [NVIDIA Brev Launchable](https://developer.nvidia.com/brev) | 1× L40S or 1× A100-40GB |
| **2. Self-hosted webui + API** | Internal dashboard, scheduled batch jobs | NGC PyTorch container on any NVIDIA host | 1× L4 / 1× A10G |
| **3. Production multi-tenant inference** | Low-latency forecasting at scale | Triton Inference Server (Python backend) on Brev / DGX Cloud | 1–2× H100 |

## Layout

```
.
├── README.md                       # this file
├── .gitignore
├── env.example                     # NGC_API_KEY, HF_TOKEN, etc.
├── pyproject.toml                  # local dev deps
├── Makefile                        # one-line entry points
│
├── scripts/                        # bash + python entry points
│   ├── 00_bootstrap.sh             # clone upstream Kronos, create venv, install deps
│   ├── 01_smoke_test.py            # verify GPU + Kronos forward pass
│   ├── 02_download_models.py       # pre-fetch HF weights (warm cache)
│   ├── 03_run_webui.sh             # upstream Flask UI on :7070
│   ├── 04_serve_api.sh             # our FastAPI gateway on :8000
│   └── 05_finetune.sh              # torchrun multi-GPU fine-tune
│
├── docker/
│   ├── Dockerfile                  # NGC PyTorch + Kronos + FastAPI gateway
│   ├── docker-compose.yml          # local NVIDIA host run
│   └── docker-compose.brev.yml     # Brev-flavored compose with labelled ports
│
├── nvidia/
│   ├── api_server.py               # OpenAI-compatible-style /v1/forecast endpoint
│   ├── brev/
│   │   └── README.md               # one-click Launchable setup walkthrough
│   └── triton/
│       ├── README.md               # production serving guide
│       └── model_repository/
│           └── kronos_predictor/
│               ├── 1/model.py      # Triton Python backend
│               └── config.pbtxt    # Triton model config
│
├── data/
│   └── generate_sample.py          # synthetic OHLCV for end-to-end tests
└── upstream/                       # populated by scripts/00_bootstrap.sh (git ignored)
```

## Five-minute quick start (Mac dev → Brev cloud GPU)

```bash
# 1. Bootstrap (clones upstream Kronos, sets up venv, downloads sample data)
./scripts/00_bootstrap.sh

# 2. Smoke test on CPU/MPS locally — proves Kronos loads and forecasts
source .venv/bin/activate
python scripts/01_smoke_test.py --device mps   # use 'cpu' on non-Apple-Silicon

# 3. Cloud GPU: deploy as a Brev Launchable
#    See nvidia/brev/README.md — paste this repo URL into build.nvidia.com,
#    select an L40S/A100, expose ports 7070 (webui) and 8000 (API).
```

When the Launchable boots it executes `scripts/00_bootstrap.sh` then `scripts/03_run_webui.sh` + `scripts/04_serve_api.sh` automatically (the Brev compose file is wired for that).

## Three deployment paths

### A. NVIDIA Brev Launchable (recommended for first-time use)

Brev is NVIDIA's GPU-as-a-service entry point on [build.nvidia.com](https://build.nvidia.com). A *Launchable* is a sharable env (GPU + container + git repo + ports + start command). See [`nvidia/brev/README.md`](nvidia/brev/README.md) for the click-by-click. The compose file [`docker/docker-compose.brev.yml`](docker/docker-compose.brev.yml) declares the ports Brev should auto-expose (8888 Jupyter, 7070 Flask UI, 8000 API).

### B. Self-hosted Docker on any NVIDIA host (NGC, DGX, on-prem, Lambda, etc.)

```bash
# On a machine with NVIDIA Container Toolkit and an NGC API key:
docker compose -f docker/docker-compose.yml up
# → http://<host>:7070  (Kronos Flask webui)
# → http://<host>:8000/docs (FastAPI gateway, OpenAPI docs)
```

The Dockerfile uses `nvcr.io/nvidia/pytorch:25.11-py3` as the base — that ships PyTorch, CUDA 12.x, cuDNN, and NCCL pre-tuned for NVIDIA GPUs. Pulling it requires an NGC API key (free with a [NVIDIA Developer Program](https://developer.nvidia.com) membership).

### C. Triton Inference Server (production)

For multi-tenant or low-latency serving with batching, retries, and metrics, package Kronos as a Triton Python backend. Files are under [`nvidia/triton/`](nvidia/triton/) and the README there walks through `tritonserver --model-repository=...`.

## Environment variables

Copy `env.example` to `.env` and fill in:

| Var | Purpose | How to obtain |
|-----|---------|---------------|
| `NGC_API_KEY` | Pull `nvcr.io/*` containers, NIM auth | [ngc.nvidia.com → Setup → Generate API Key](https://ngc.nvidia.com/setup) |
| `HF_TOKEN` | Optional — only needed for gated repos | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `KRONOS_MODEL` | Which model to load | `NeoQuasar/Kronos-small` (default) / `Kronos-base` / `Kronos-mini` |
| `KRONOS_DEVICE` | Inference device | `cuda:0` (default in container), `mps`, `cpu` |
| `KRONOS_MAX_CONTEXT` | Context length | `512` for small/base, `2048` for mini |

## Hardware sizing cheat-sheet

| Workload | Min GPU | Recommended | Notes |
|----------|--------|-------------|-------|
| Inference, Kronos-small (24.7M) | T4 8GB | L4 24GB | Throughput-bound, not VRAM |
| Inference, Kronos-base (102M) | L4 24GB | L40S 48GB | `sample_count=5` benefits from concurrency |
| Fine-tune tokenizer | 1× A100-40 | 2× A100-80 | `torchrun --nproc_per_node=N` |
| Fine-tune predictor | 2× A100-40 | 4× H100 | Use bf16 on H100 |
| Multi-asset batch backtest | L40S 48GB | H100 80GB | `predictor.predict_batch` parallelizes across the asset axis |

## Acknowledgements

Kronos by Shi et al., AAAI 2026 — code: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos), paper: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739).

This kit is MIT-licensed (matching upstream) and is unaffiliated with the Kronos authors or NVIDIA.
