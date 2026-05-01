#!/usr/bin/env bash
# One-button multi-GPU fine-tune of Kronos on the Indian corpus.
#
#   STAGE=tokenizer NUM_GPUS=8 ./scripts/07_finetune_max.sh
#   STAGE=predictor NUM_GPUS=8 ./scripts/07_finetune_max.sh
#
# Steps:
#  1. Symlinks finetune/nifty_config.py -> upstream/finetune/config.py (so the
#     upstream torchrun scripts pick up our Indian config without modification).
#  2. Auto-detects GPU count if NUM_GPUS is not set.
#  3. Runs torchrun for the requested STAGE.
#  4. Restores the original upstream config.py on exit (trap).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -d .venv ]; then source .venv/bin/activate; fi

if [ ! -d upstream/finetune ]; then
  echo "upstream/finetune not found. Run scripts/00_bootstrap.sh first." >&2
  exit 1
fi
if [ ! -f data/processed_datasets/train_data.pkl ]; then
  echo "data/processed_datasets/train_data.pkl missing. Run scripts/06_fetch_all_data.sh first." >&2
  exit 1
fi

STAGE="${STAGE:-tokenizer}"
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count() or 1)')}"

# Bridge our config + dataset paths into the env the upstream config reads.
export KRONOS_DATASET_PATH="${KRONOS_DATASET_PATH:-$ROOT/data/processed_datasets}"
export KRONOS_SAVE_PATH="${KRONOS_SAVE_PATH:-$ROOT/outputs/models}"

UPSTREAM_CONFIG="upstream/finetune/config.py"
KIT_CONFIG="finetune/nifty_config.py"
BACKUP="$UPSTREAM_CONFIG.original"

cleanup() {
  if [ -f "$BACKUP" ]; then
    mv "$BACKUP" "$UPSTREAM_CONFIG"
    echo "==> Restored upstream config."
  fi
}
trap cleanup EXIT INT TERM

# 1. Swap in our config (preserve the original)
if [ ! -f "$BACKUP" ]; then
  cp "$UPSTREAM_CONFIG" "$BACKUP"
fi
cp "$KIT_CONFIG" "$UPSTREAM_CONFIG"
echo "==> Activated Indian config: $KIT_CONFIG -> $UPSTREAM_CONFIG"

# 2. Make sure pyqlib import in upstream doesn't blow up. The upstream
#    train scripts don't actually USE qlib at runtime (only the preprocess
#    step does), but `from config import Config` will trigger any
#    qlib imports if present. Our nifty_config doesn't import qlib, so
#    this is fine — but we double-check by short-circuiting any
#    qlib-related env that might still be required.
export QLIB_PROVIDER="<unused>"

# 3. Launch torchrun
mkdir -p "$KRONOS_SAVE_PATH"
echo "==> STAGE=$STAGE  NUM_GPUS=$NUM_GPUS"
echo "    dataset : $KRONOS_DATASET_PATH"
echo "    save    : $KRONOS_SAVE_PATH"
echo

cd upstream/finetune

case "$STAGE" in
  tokenizer)
    torchrun --standalone --nproc_per_node="$NUM_GPUS" train_tokenizer.py
    ;;
  predictor)
    torchrun --standalone --nproc_per_node="$NUM_GPUS" train_predictor.py
    ;;
  both)
    echo "==> Stage 1: tokenizer"
    torchrun --standalone --nproc_per_node="$NUM_GPUS" train_tokenizer.py
    echo "==> Stage 2: predictor (uses fine-tuned tokenizer)"
    # When predictor runs, our config's finetuned_tokenizer_path points it at
    # the just-saved tokenizer checkpoint.
    export KRONOS_PRETRAIN_TOKENIZER="$KRONOS_SAVE_PATH/kronos_tokenizer_nifty/checkpoints/best_model"
    torchrun --standalone --nproc_per_node="$NUM_GPUS" train_predictor.py
    ;;
  *)
    echo "Unknown STAGE='$STAGE'. Use: tokenizer | predictor | both" >&2
    exit 1
    ;;
esac

echo
echo "==> Done. Checkpoints under: $KRONOS_SAVE_PATH"
