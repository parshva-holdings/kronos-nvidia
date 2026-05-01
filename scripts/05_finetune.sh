#!/usr/bin/env bash
# Multi-GPU fine-tune Kronos using the upstream torchrun pipeline.
#   NUM_GPUS=4 STAGE=tokenizer scripts/05_finetune.sh
#   NUM_GPUS=4 STAGE=predictor scripts/05_finetune.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ".env" ]; then set -a; . ./.env; set +a; fi
if [ -d ".venv" ]; then source .venv/bin/activate; fi

NUM_GPUS="${NUM_GPUS:-1}"
STAGE="${STAGE:-tokenizer}"  # tokenizer | predictor

if [ ! -d "upstream/finetune" ]; then
  echo "upstream/finetune not found — run scripts/00_bootstrap.sh first" >&2
  exit 1
fi

# Make sure pyqlib is installed (upstream dependency for the finetune pipeline)
python -c "import qlib" 2>/dev/null || pip install pyqlib

cd upstream

case "$STAGE" in
  tokenizer)
    echo "==> Fine-tuning tokenizer on $NUM_GPUS GPU(s)..."
    torchrun --standalone --nproc_per_node="$NUM_GPUS" finetune/train_tokenizer.py
    ;;
  predictor)
    echo "==> Fine-tuning predictor on $NUM_GPUS GPU(s)..."
    torchrun --standalone --nproc_per_node="$NUM_GPUS" finetune/train_predictor.py
    ;;
  backtest)
    echo "==> Running backtest..."
    python finetune/qlib_test.py --device "${KRONOS_DEVICE:-cuda:0}"
    ;;
  *)
    echo "Unknown STAGE='$STAGE'. Use: tokenizer | predictor | backtest" >&2
    exit 1
    ;;
esac
