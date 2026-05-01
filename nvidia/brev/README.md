# Deploy Kronos on NVIDIA Brev

[NVIDIA Brev](https://developer.nvidia.com/brev) is the GPU-as-a-service entry point on [build.nvidia.com](https://build.nvidia.com). A *Launchable* is a sharable spec (GPU + container + git repo + ports + start command) that lets anyone spin up an identical environment with one click.

This guide takes you from zero to a running Kronos webui + FastAPI gateway + Jupyter Lab on an NVIDIA cloud GPU in roughly five minutes.

## Prerequisites

1. An [NVIDIA Developer Program](https://developer.nvidia.com) account (free).
2. An [NGC API key](https://ngc.nvidia.com/setup) (free; needed to pull `nvcr.io/nvidia/pytorch`).
3. This repo pushed to a public GitHub URL (Brev needs a URL it can clone).

## Path A — One-click Launchable from this repo

1. Visit **<https://brev.nvidia.com/launchables/create>**.
2. Choose **"I have code files in a git repository"**.
3. Paste your GitHub URL (e.g. `https://github.com/<you>/Kronos`).
4. **Compute**:
   - Inference only: `1× L40S 48GB` (best price/perf for Kronos-base)
   - Fine-tuning: `2× A100-80GB` or `4× H100-80GB`
5. **Container**: choose the *Custom container* tab and paste:
   - Image: `nvcr.io/nvidia/pytorch:25.11-py3`
   - Or *Docker Compose* tab → upload `docker/docker-compose.brev.yml` from this repo.
6. **Ports** (auto-detected from the compose file's `brev.port.*` labels):
   - `8888` (jupyter) → "Open Notebook" button
   - `7070` (webui)  → upstream Flask demo
   - `8000` (api)    → our FastAPI gateway, with `/docs`
7. **Environment variables**: paste `NGC_API_KEY` (required) and `HF_TOKEN` (optional).
8. Click **Deploy**. First boot ≈ 4–6 minutes (image pull + HF model download).

When the green dot appears next to each port, click the URLs in the deployment view to open Jupyter, the webui, or the API docs.

## Path B — Brev CLI (scriptable)

```bash
# Install the Brev CLI
curl -L https://brev.nvidia.com/install.sh | sh

# Authenticate
brev login

# Create from this repo
brev launchable create \
  --name kronos-nvidia \
  --gpu L40S \
  --gpu-count 1 \
  --git https://github.com/<you>/Kronos \
  --compose docker/docker-compose.brev.yml \
  --env NGC_API_KEY=$NGC_API_KEY \
  --port jupyter:8888 \
  --port webui:7070 \
  --port api:8000

# Open the deployment in the browser
brev open kronos-nvidia
```

## What the Launchable does on first boot

Launch order, executed by `command:` in `docker-compose.brev.yml`:

1. **Background warm-up** — `scripts/02_download_models.py` pulls all five Kronos checkpoints into `/opt/hf_cache` (volume-persisted across redeploys).
2. **API gateway** — `uvicorn nvidia.api_server:app` on `:8000`. The first request loads weights from the cache (~1–3s on L40S).
3. **Jupyter Lab** — on `:8888`, mounted to `/workspace`.
4. **Flask webui** — `upstream/webui/run.py` on `:7070`, foreground.

## Verifying the deployment

```bash
# Health
curl https://<your-brev-url>:8000/health

# Predict (5 historical bars → 5 future bars)
curl -X POST https://<your-brev-url>:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "bars": [
      {"timestamp":"2026-04-30T09:30:00","open":100,"high":101,"low":99.5,"close":100.5,"volume":1000,"amount":100250},
      {"timestamp":"2026-04-30T09:35:00","open":100.5,"high":102,"low":100.2,"close":101.7,"volume":1200,"amount":121500},
      {"timestamp":"2026-04-30T09:40:00","open":101.7,"high":102.3,"low":101.1,"close":101.4,"volume":900,"amount":91290},
      {"timestamp":"2026-04-30T09:45:00","open":101.4,"high":102.0,"low":100.9,"close":101.8,"volume":1100,"amount":111870},
      {"timestamp":"2026-04-30T09:50:00","open":101.8,"high":102.5,"low":101.3,"close":102.1,"volume":1300,"amount":132730}
    ],
    "horizon": 5,
    "temperature": 1.0,
    "top_p": 0.9,
    "sample_count": 1
  }'
```

## Hardware sizing for Brev

| Workload | GPU type | Why |
|----------|---------|-----|
| Demo, Kronos-small inference | L4 24GB | Cheapest GPU on Brev with enough VRAM |
| Production inference, Kronos-base, batch | L40S 48GB | Best $/throughput; supports `predict_batch` |
| Fine-tune tokenizer | 1× A100-40GB | Tokenizer is small; fits easily |
| Fine-tune predictor (small model) | 2× A100-40GB | DDP via `torchrun --nproc_per_node=2` |
| Fine-tune predictor (large refresh) | 4× H100-80GB + bf16 | Cuts wall-clock ~3× vs A100 |

## Tearing it down

Brev bills per-second of running time. Stop the deployment from the console (or `brev stop kronos-nvidia`) when you're not using it. The `hf_cache` volume persists, so the next boot is warm.

## References

- NVIDIA Brev: <https://developer.nvidia.com/brev>
- Brev Launchables docs: <https://docs.nvidia.com/brev/concepts/launchables>
- NGC PyTorch container: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>
- NGC API key setup: <https://ngc.nvidia.com/setup>
