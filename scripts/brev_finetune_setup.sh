#!/bin/bash
# Brev launchable setup script for the FULL fine-tune run.
#
# Designed for: 8× H100-80GB (or 4× A100-80GB) on a Brev VM-Mode instance.
# Wall-clock: ~8-22h. Cost: ~$200-700 depending on GPU profile + provider.
#
# Pipeline (all stages run unattended):
#   0. Bootstrap (clone upstream Kronos, venv, deps)
#   1. Verify GPU + torch CUDA work
#   2. Fetch full NSE corpus (Layer 4, ~150 series, ~15 min)
#   3. Build train/val/test pickles
#   4. Tokenizer fine-tune  (DDP across all visible GPUs)
#   5. Predictor fine-tune  (DDP, uses just-trained tokenizer)
#   6. Re-run the 135-forecast accuracy audit with the new weights
#   7. Print summary + checkpoint paths
#
# IMPORTANT: This script runs to completion (8-22h) then EXITS — the VM keeps
# running and BILLING until you click Stop in the Brev console. Set a calendar
# reminder to come back the next morning.
#
# Brev wizard requirement: shebang must be #!/bin/bash (not /usr/bin/env bash).
set -euo pipefail
exec > >(tee -a /tmp/kronos_finetune.log) 2>&1

DATE_TAG="$(date +%Y-%m-%d)"
START_EPOCH=$(date +%s)

echo "============================================================"
echo "  Kronos × NSE Full Fine-tune — Brev launchable"
echo "  $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# ---------------------------------------------------------------------------
# 0. Locate the cloned repo
# ---------------------------------------------------------------------------
REPO=""
for cand in \
    /home/ubuntu/verb-workspace/kronos-nvidia \
    /home/ubuntu/kronos-nvidia \
    /workspace/kronos-nvidia \
    "$(pwd)/kronos-nvidia" \
    "$(pwd)"; do
  if [ -f "$cand/scripts/00_bootstrap.sh" ]; then
    REPO="$cand"
    break
  fi
done
if [ -z "$REPO" ]; then
  echo "ERROR: kronos-nvidia repo not found"
  exit 1
fi
cd "$REPO"
echo "==> Repo at $REPO"

# ---------------------------------------------------------------------------
# 1. Bootstrap (idempotent — safe if already done)
# ---------------------------------------------------------------------------
echo
echo "==> [Stage 0/6] Bootstrap"
bash scripts/00_bootstrap.sh

# shellcheck disable=SC1091
source .venv/bin/activate

# Make sure pyqlib (upstream finetune dep — our config bypasses Qlib but the
# upstream training scripts still `import qlib` at module load time).
pip install --quiet pyqlib comet-ml || true

# ---------------------------------------------------------------------------
# 2. GPU sanity check — fail loud if CUDA isn't actually available
# ---------------------------------------------------------------------------
echo
echo "==> [Stage 1/6] GPU sanity check"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE — check NVIDIA driver / pytorch install"
n = torch.cuda.device_count()
print(f"  CUDA available: True")
print(f"  GPUs visible  : {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"    [{i}] {p.name} · {p.total_memory / 1e9:.0f}GB · cap {p.major}.{p.minor}")
print(f"  PyTorch       : {torch.__version__}")
print(f"  cuDNN         : {torch.backends.cudnn.version()}")
PY

# Detect GPU count for torchrun
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "  Will train with NUM_GPUS=$NUM_GPUS"

# ---------------------------------------------------------------------------
# 3. Fetch full NSE corpus (Layer 4 = ~150 series, max history each)
# ---------------------------------------------------------------------------
echo
echo "==> [Stage 2/6] Fetch full NSE corpus (Layer 4, ~15 min)"
if [ -d data/nse_corpus ] && [ "$(find data/nse_corpus -name '*.csv' | wc -l)" -gt 100 ]; then
  echo "  data/nse_corpus already populated; skipping fetch"
else
  python data/fetch_nse_full.py \
    --layer 4 \
    --period max \
    --interval 1d \
    --out data/nse_corpus \
    --sleep 0.4
fi
echo "  fetched: $(find data/nse_corpus -name '*.csv' | wc -l) CSVs"

# ---------------------------------------------------------------------------
# 4. Build train/val/test pickles
# ---------------------------------------------------------------------------
echo
echo "==> [Stage 3/6] Build train/val/test pickles"
python data/build_indian_corpus.py \
  --corpus data/nse_corpus \
  --out data/processed_datasets

# ---------------------------------------------------------------------------
# 5. Stage A: fine-tune the tokenizer
# ---------------------------------------------------------------------------
echo
echo "==> [Stage 4/6] Tokenizer fine-tune (DDP × $NUM_GPUS GPUs)"
echo "    Expected wall-clock: ~3-6h"
TOKENIZER_T0=$(date +%s)

# scripts/07_finetune_max.sh handles config swap + restore
NUM_GPUS="$NUM_GPUS" STAGE=tokenizer bash scripts/07_finetune_max.sh

TOKENIZER_DT=$(( $(date +%s) - TOKENIZER_T0 ))
echo "==> Tokenizer fine-tune took $((TOKENIZER_DT / 60)) min"

# Confirm checkpoint exists where we expect
TOKENIZER_OUT="$REPO/outputs/models/kronos_tokenizer_nifty/checkpoints/best_model"
if [ ! -d "$TOKENIZER_OUT" ]; then
  echo "ERROR: expected tokenizer checkpoint at $TOKENIZER_OUT — not found."
  ls -la "$REPO/outputs/models/" 2>/dev/null || true
  exit 1
fi
echo "  Tokenizer checkpoint: $TOKENIZER_OUT"

# ---------------------------------------------------------------------------
# 6. Stage B: fine-tune the predictor (using the just-trained tokenizer)
# ---------------------------------------------------------------------------
echo
echo "==> [Stage 5/6] Predictor fine-tune (DDP × $NUM_GPUS GPUs)"
echo "    Expected wall-clock: ~5-18h"
PREDICTOR_T0=$(date +%s)

# Tell our nifty_config to use the just-trained tokenizer as the starting tokenizer
export KRONOS_PRETRAIN_TOKENIZER="$TOKENIZER_OUT"

NUM_GPUS="$NUM_GPUS" STAGE=predictor bash scripts/07_finetune_max.sh

PREDICTOR_DT=$(( $(date +%s) - PREDICTOR_T0 ))
echo "==> Predictor fine-tune took $((PREDICTOR_DT / 60)) min"

PREDICTOR_OUT="$REPO/outputs/models/kronos_predictor_nifty/checkpoints/best_model"
if [ ! -d "$PREDICTOR_OUT" ]; then
  echo "ERROR: expected predictor checkpoint at $PREDICTOR_OUT — not found."
  ls -la "$REPO/outputs/models/" 2>/dev/null || true
  exit 1
fi
echo "  Predictor checkpoint: $PREDICTOR_OUT"

# ---------------------------------------------------------------------------
# 7. Stage C: re-run the 135-forecast accuracy audit with the new weights
# ---------------------------------------------------------------------------
echo
echo "==> [Stage 6/6] Validation audit — same 135 forecasts as zero-shot baseline"
mkdir -p reports
python -m nvidia.backtest.run_audit \
  --mode intense \
  --model "$PREDICTOR_OUT" \
  --tokenizer "$TOKENIZER_OUT" \
  --device cuda:0 \
  --out "reports/finetuned_${DATE_TAG}.md" \
  || echo "(audit failed — check reports dir manually)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_DT=$(( $(date +%s) - START_EPOCH ))
echo
echo "============================================================"
echo "  Fine-tune COMPLETE  (total $((TOTAL_DT / 60)) min)"
echo "============================================================"
echo "  Tokenizer ckpt : $TOKENIZER_OUT"
echo "  Predictor ckpt : $PREDICTOR_OUT"
echo "  Validation     : $REPO/reports/finetuned_${DATE_TAG}.md"
echo "  Baseline       : $REPO/reports/zeroshot_2026-05-01.md"
echo
echo "  Diff in headlines (look at the markdown headlines table):"
echo "    grep -A 12 'Headline numbers' reports/zeroshot_2026-05-01.md"
echo "    grep -A 12 'Headline numbers' reports/finetuned_${DATE_TAG}.md"
echo
echo "  NEXT STEPS:"
echo "    1. Inspect reports/finetuned_${DATE_TAG}.md and compare to baseline"
echo "    2. (Optional) tar the checkpoints + reports for download:"
echo "         tar czf /tmp/kronos-nifty-${DATE_TAG}.tar.gz outputs/models reports/"
echo "         brev cp <this-instance>:/tmp/kronos-nifty-${DATE_TAG}.tar.gz ./"
echo "    3. ⚠️  STOP this Brev instance to halt billing!"
echo "         (instance keeps running until you click Stop)"
echo "============================================================"
