# Fine-tune Kronos on All-Time NSE — Brev H100 Launchable

This is the "no cap" path: fine-tune `Kronos-base` (102M params) on every Indian-market data point we can legally pull, on 8× H100s, with zero post-launch interaction needed.

## What you get out of it

- A **Kronos-Tokenizer-Nifty** checkpoint (specialized for NSE price distributions)
- A **Kronos-Predictor-Nifty** checkpoint (autoregressive predictor fine-tuned on all-time NSE)
- A held-out validation report (`reports/finetuned_<date>.md`) with the **same 135-forecast audit** we ran zero-shot — apples-to-apples accuracy delta vs `reports/zeroshot_2026-05-01.md`
- Both checkpoints saved at known paths on disk; you `brev cp` them down to your Mac when done

## What it costs

| Plan | Wall-clock | $/hr range | Total |
|---|---|---|---|
| **8× H100-80GB** ✅ recommended | ~8-22h | $15-25 | **$200-550** |
| 4× A100-80GB | ~16-30h | $9-13 | $200-400 |
| 2× A100-80GB (cheapest viable) | ~50-70h | $5-7 | $300-490 |

The wall-clock spread is wide because it depends on how fast each provider's interconnect is (NVLink vs PCIe) and which CPU/disk pairs them. **Budget $500 to be safe**; you'll likely come in under.

## Prerequisites (already done if you've followed along)

- ✅ NGC API key in `.env` (you have this)
- ✅ Repo pushed to `parshva-holdings/kronos-nvidia` (you have this)
- ✅ NVIDIA Developer Program account
- ✅ Brev CLI installed (`brew install brevdev/homebrew-brev/brev`)

## The wizard, step by step

### 1. New launchable — same start as before
<https://brev.nvidia.com/launchables/create>

### 2. Step 1 of 5 — Code source + runtime mode

| Field | Value |
|---|---|
| Code provider | **I have code files in a git repository** |
| URL | `https://github.com/parshva-holdings/kronos-nvidia` |
| Runtime | **VM Mode** (same reasoning as inference: private NGC registry + multi-stage pipeline) |

### 3. Step 2 of 5 — Setup script

**Click "Paste Script"** tab and paste this **exactly**:

```bash
#!/bin/bash
exec > >(tee -a /tmp/kronos_finetune_bootstrap.log) 2>&1
set -e

# Find repo
for cand in /home/ubuntu/verb-workspace/kronos-nvidia /home/ubuntu/kronos-nvidia /workspace/kronos-nvidia "$(pwd)/kronos-nvidia" "$(pwd)"; do
  if [ -f "$cand/scripts/brev_finetune_setup.sh" ]; then REPO="$cand"; break; fi
done
[ -z "${REPO:-}" ] && { echo "Repo not found"; exit 1; }
cd "$REPO"

# Run the full fine-tune pipeline in the background so the setup script
# can return success quickly (Brev marks setup as "Completed" once this
# returns). The actual training continues independently.
nohup bash scripts/brev_finetune_setup.sh > /tmp/kronos_finetune.log 2>&1 &
disown
echo "Fine-tune pipeline launched in background. Monitor with:"
echo "  tail -f /tmp/kronos_finetune.log"
```

The reason for `nohup … &`: Brev's launchable wizard expects the setup script to *complete*, but our actual training takes 8-22h. So we kick the real work off in the background and let the setup script return immediately. The training continues as long as the VM is up.

### 4. Step 3 of 5 — Ports

Add **8888** (Jupyter) only. We don't need Streamlit on this instance — it's training-only.

You can also skip ports entirely; `brev shell` and `brev port-forward` work without firewall ports.

### 5. Step 4 of 5 — Compute (THE EXPENSIVE STEP)

Click the **H100** card at the top.

```
Look for an instance with these properties:
  - 8× H100-80GB  (yes, eight)
  - "Stop/start" enabled (NOT the gray "No stop/start" label)
  - Flexible storage
  - Region: any US/EU region with availability  (us-east-1, us-west-2 typical)
```

Cost will show ~$15-25/hr depending on provider. **CRUSOE is again the recommended provider** for the same reasons as before (stop/start, flexible storage, decent boot time).

Set **Disk Storage** to **300 GB** (we need ~70 GB for data + checkpoints; 300 leaves runway).

If 8× H100 isn't available in any region with stop/start: fall back to **4× A100-80GB** — slower but same total cost.

### 6. Step 5 of 5 — Review

| Field | Value |
|---|---|
| Name | `kronos-finetune-<date>` |
| Description | (paste this 👇) |
| Visibility | **Only my organization** |
| Publish to community | OFF |

Description:
```
Full fine-tune of Kronos-base on all-time NSE data (1996-2026, ~150 series).
Two stages: tokenizer then predictor. Auto-runs validation audit at the end.
Wall-clock ~8-22h, ~$200-550 total. Stop instance when complete.
```

Click **Create Launchable**, then on the next page click **Deploy**.

## During the run

Boot takes ~7-15 min (VM provision + Docker layers + repo clone). When the Brev console shows green, the **bootstrap log** is at `/tmp/kronos_finetune_bootstrap.log` and the **training log** at `/tmp/kronos_finetune.log`.

### Monitor ongoing (any of these)

**Easiest — tail the log via Jupyter Terminal:**
1. Click **Open Notebook** on the Brev page
2. Open a Terminal tile
3. Run `tail -f /tmp/kronos_finetune.log`

You'll see live output:
```
==> [Stage 4/6] Tokenizer fine-tune (DDP × 8 GPUs)
[Epoch 1/30] step 50/2000  loss 4.213  val 4.187  lr 2e-4  84 it/s
...
```

**Via SSH:**
```bash
brev shell kronos-finetune-<date>
tail -f /tmp/kronos_finetune.log
```

**GPU utilization:**
```bash
watch -n 2 nvidia-smi
# expect each H100 at 80-95% during training
```

### Health checkpoints (from your laptop)

```bash
# Every couple of hours, peek at progress without SSHing
brev exec kronos-finetune-<date> -- 'tail -3 /tmp/kronos_finetune.log'
```

## When it finishes

The script's last lines look like:

```
============================================================
  Fine-tune COMPLETE  (total 712 min)
============================================================
  Tokenizer ckpt : .../outputs/models/kronos_tokenizer_nifty/checkpoints/best_model
  Predictor ckpt : .../outputs/models/kronos_predictor_nifty/checkpoints/best_model
  Validation     : .../reports/finetuned_2026-05-02.md
```

**At this point billing is still ticking.** Three things to do:

### 1. Pull the artifacts down

```bash
# In your Brev shell on the H100 instance:
cd "$(find / -maxdepth 5 -type d -name kronos-nvidia 2>/dev/null | head -1)"
tar czf /tmp/kronos-nifty-$(date +%Y-%m-%d).tar.gz outputs/models/ reports/
ls -lh /tmp/kronos-nifty-*.tar.gz   # should be ~400-500 MB

# From your Mac:
brev cp kronos-finetune-<date>:/tmp/kronos-nifty-2026-05-02.tar.gz ./
```

### 2. Compare audits

```bash
# Locally:
tar xzf kronos-nifty-2026-05-02.tar.gz -C /tmp/finetune-result/
diff <(grep -A 15 "Headline numbers" reports/zeroshot_2026-05-01.md) \
     <(grep -A 15 "Headline numbers" /tmp/finetune-result/reports/finetuned_2026-05-02.md)
```

The fine-tuned report should show meaningfully better numbers across the board. If it doesn't, we have the audit to diagnose what went wrong.

### 3. STOP the instance

Click **Stop** in the Brev console. Cost drops from $15-25/hr to $0.10/hr (storage only).

If you're confident you're done, click **Delete** instead — saves the storage cost too. The artifacts are already on your Mac via step 1.

## Deploying the new model to the L40S inference instance

Once you have the tarball local:

```bash
# Push the checkpoints to your private HF repo (recommended — versioned, easy redeploy)
hf auth login   # one-time
hf upload parshva-holdings/Kronos-Tokenizer-Nifty /tmp/finetune-result/outputs/models/kronos_tokenizer_nifty/checkpoints/best_model/ --private
hf upload parshva-holdings/Kronos-Predictor-Nifty /tmp/finetune-result/outputs/models/kronos_predictor_nifty/checkpoints/best_model/ --private

# On the L40S inference VM, update .env:
KRONOS_MODEL=parshva-holdings/Kronos-Predictor-Nifty
KRONOS_TOKENIZER=parshva-holdings/Kronos-Tokenizer-Nifty

# Restart Streamlit
pkill -f streamlit && \
  nohup streamlit run webui_streamlit/app.py --server.port 8501 --server.address 0.0.0.0 \
        --server.headless true --browser.gatherUsageStats false \
        > /tmp/streamlit.log 2>&1 & disown
```

The dashboard now serves your fine-tuned weights. Click *Generate forecast* with `as-of date` set to a year ago — the verdict panel should now show **direction ✅** much more often than it did before.

## Risks + mitigations (cheat sheet)

| Risk | Mitigation |
|---|---|
| H100 unavailable | Fall back to 4× A100-80GB or wait 1-12h for stock |
| yfinance rate-limit | Built-in 0.4s sleep + retry; if still hitting limits, run fetch_nse_full.py manually overnight before launching the H100 |
| OOM during predictor train | Set env var `KRONOS_BATCH_SIZE=32` in the launchable's env-vars section |
| Training diverges | Comet alerts (if enabled); auto-checkpoint allows resume from last good step |
| You forget to stop the instance | Set a Brev budget alert at $700 |
| Audit shows little improvement | Inspect per-config breakdown — if `range_coverage` improved but `endpoint_dir` didn't, the model learned volatility correctly but not direction. Iterate by training another 10 epochs with a lower LR. |

## Quick reference

**Files involved**:
- `scripts/brev_finetune_setup.sh` — the actual training pipeline (this is what runs)
- `scripts/00_bootstrap.sh` — kit bootstrap
- `data/fetch_nse_full.py` — corpus fetcher (Layer 4)
- `data/build_indian_corpus.py` — converts CSVs to upstream pickle format
- `finetune/nifty_config.py` — opinionated NSE-tuned hyperparameters
- `scripts/07_finetune_max.sh` — torchrun launcher (DDP)
- `nvidia/backtest/run_audit.py` — re-runs the 135-forecast audit at the end

**Key knobs (override via env vars at launchable creation time)**:
- `KRONOS_BATCH_SIZE=32` (default 50; reduce for 40GB GPUs)
- `KRONOS_EPOCHS=20` (default 30; reduce for cost-conscious runs)
- `KRONOS_PRETRAIN_MODEL=NeoQuasar/Kronos-base` (the largest open variant)
- `COMET_API_KEY=…` + `COMET_WORKSPACE=…` (optional live training dashboard)

## After the fine-tune

You'll have a working Kronos that's **specialized for Indian markets**. From here:
- Iterate on the dashboard's UX with calibrated forecasts
- Layer in news sentiment / macro signals (the "10× tier" we discussed)
- Connect Zerodha Kite for paper trading
- Eventually: scale up corpus to include intraday data

But first: validate that the fine-tune itself worked — that's what the post-fine-tune audit is for.
