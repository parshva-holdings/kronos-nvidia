# Fine-tune Kronos on All-Time Nifty 50 — Brev H100 Launchable

This is the "no cap" path: fine-tune `Kronos-base` (102M params) on every Indian-market data point we can legally pull, on 8× H100s.

## What you'll get out of it

- A `Kronos-Tokenizer-Nifty` checkpoint (specialized for NSE price distributions)
- A `Kronos-Predictor-Nifty` checkpoint (autoregressive predictor fine-tuned on all-time NSE data)
- Held-out test MAPE/MAE/RankIC for 2024-2026 vs zero-shot Kronos-base baseline
- Both checkpoints uploaded to your private HF repo (optional)

## What it'll cost

| GPU plan | Wall-clock | $/hr | Total |
|---|---|---|---|
| **8× H100 80GB** (recommended) | ~22h (4h tokenizer + 18h predictor) | $18-25/hr | **$400-700** |
| 4× A100 80GB | ~36-44h | $9-13/hr | $400-550 |
| 2× A100 40GB | ~70h | $4-6/hr | $300-450 |
| 1× A100 80GB (slow) | ~140h | $2-3/hr | $300-450 |

If money matters more than speed: **2× A100 40GB** is the cost floor that still finishes in under three days.

## Prereqs

1. The repo pushed to GitHub (you did this in the previous step)
2. NGC API key in `.env`
3. (Optional) [HF token](https://huggingface.co/settings/tokens) for uploading the trained model

## Click-by-click

1. **<https://brev.nvidia.com/launchables/create>**
2. **Repo**: paste your GitHub URL → branch `main`
3. **Compute**:
   - GPU: `H100-80GB`
   - Count: `8`
   - Region: any (data is downloaded into the container; latency irrelevant for training)
4. **Container**: `Custom Docker Compose` → upload `docker/docker-compose.brev.yml`
5. **Override `command:`** in the compose form:
   ```bash
   bash -lc "
     ./scripts/00_bootstrap.sh &&
     ./scripts/06_fetch_all_data.sh &&
     STAGE=both NUM_GPUS=8 ./scripts/07_finetune_max.sh
   "
   ```
6. **Env vars**:
   - `NGC_API_KEY` (yours)
   - `HF_TOKEN` (optional, raises rate limits and enables auto-push)
   - `KRONOS_PRETRAIN_MODEL=NeoQuasar/Kronos-base` (the largest open variant)
   - `KRONOS_EPOCHS=30` (or `15` for a faster cheaper run)
   - `KRONOS_BATCH_SIZE=64` (H100 can comfortably hold this; reduce to `24` on A100-40)
   - `COMET_API_KEY` if you want live training dashboards (optional)
7. **Storage**: attach a 200 GB volume mounted at `/root/kronos_data` so checkpoints persist across redeploys
8. **Deploy**

When training finishes, the Brev console will still show the instance running (idle). **Stop it** to halt billing.

## Watching progress

The compose file already exposes Jupyter on `:8888`. Open it, and inside the container:

```bash
# Live training log
tail -f /workspace/outputs/models/kronos_predictor_nifty/log.txt

# GPU utilization
watch -n 1 nvidia-smi

# Checkpoint sizes growing
watch -n 30 'du -sh /workspace/outputs/models/*'
```

Comet (if enabled): training/val loss curves at <https://www.comet.com/your-workspace/kronos-nifty-finetune>

## Pulling the trained weights down

Once `kronos_predictor_nifty/checkpoints/best_model` exists:

```bash
# From your laptop (rsync over SSH — Brev provides an SSH key in the deployment view)
brev shell <your-instance>
# inside the instance:
tar czf /tmp/kronos-nifty.tar.gz outputs/models/
# back on laptop:
brev cp <your-instance>:/tmp/kronos-nifty.tar.gz ./
```

Then in `.env`:

```env
KRONOS_MODEL=./checkpoints/kronos_predictor_nifty/best_model
KRONOS_TOKENIZER=./checkpoints/kronos_tokenizer_nifty/best_model
```

The FastAPI gateway (`scripts/04_serve_api.sh`) and Triton config will pick up the new paths automatically.

## How to know it actually got better

After training, run the held-out evaluation:

```bash
python examples/nifty50_forecast.py \
  --model ./checkpoints/kronos_predictor_nifty/best_model \
  --tokenizer ./checkpoints/kronos_tokenizer_nifty/best_model \
  --holdout --horizon 30 --samples 5

# Compare against the zero-shot baseline:
python examples/nifty50_forecast.py \
  --model NeoQuasar/Kronos-base \
  --tokenizer NeoQuasar/Kronos-Tokenizer-base \
  --holdout --horizon 30 --samples 5
```

Realistic improvement target: **30-60% reduction in MAPE** on Nifty test set vs zero-shot Kronos-base, based on typical fine-tune deltas reported in the paper.
