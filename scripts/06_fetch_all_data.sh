#!/usr/bin/env bash
# One-button: fetch the full NSE corpus and build train/val/test pickles.
# Idempotent — re-run any time to refresh data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -d .venv ]; then source .venv/bin/activate; fi

LAYER="${LAYER:-4}"          # 1 .. 4 (4 = max corpus)
INTERVAL="${INTERVAL:-1d}"   # daily by default

echo "==> Step 1/2: fetching NSE corpus (layer=$LAYER, interval=$INTERVAL)"
echo "    This takes 5-15 minutes due to Yahoo rate limiting."
python data/fetch_nse_full.py \
  --layer "$LAYER" \
  --interval "$INTERVAL" \
  --period max \
  --out data/nse_corpus

echo
echo "==> Step 2/2: building pickled train/val/test splits"
python data/build_indian_corpus.py \
  --corpus data/nse_corpus \
  --out data/processed_datasets

echo
echo "==> Corpus ready. Sizes:"
du -sh data/nse_corpus data/processed_datasets 2>/dev/null || true
ls -lh data/processed_datasets/*.pkl 2>/dev/null

echo
echo "Next:"
echo "  Local sanity-check fine-tune (CPU/MPS, 1 epoch):"
echo "    KRONOS_EPOCHS=1 NUM_GPUS=1 STAGE=tokenizer ./scripts/07_finetune_max.sh"
echo "  Full GPU fine-tune (cloud):"
echo "    See nvidia/brev/finetune.md"
