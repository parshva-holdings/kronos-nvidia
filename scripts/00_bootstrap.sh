#!/usr/bin/env bash
# Bootstrap: clone upstream Kronos, create venv, install deps, generate sample data.
# Idempotent — safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Bootstrapping Kronos × NVIDIA kit in $ROOT"

# 1. Clone upstream Kronos into ./upstream (not committed)
if [ ! -d "upstream/.git" ]; then
  echo "==> Cloning upstream Kronos..."
  git clone --depth 1 https://github.com/shiyu-coder/Kronos.git upstream
else
  echo "==> upstream/ already present — pulling latest..."
  (cd upstream && git pull --ff-only) || echo "    (skipping pull, working tree dirty)"
fi

# 2. Pick the best Python interpreter available. Prefer 3.13 (broadest wheel
#    coverage in the Kronos / pyqlib ecosystem in May 2026), then fall back.
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
      PY="$cand"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "no Python >= 3.10 found on PATH" >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "==> Creating .venv with $($PY --version) ($PY)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Upgrade pip and install kit dependencies (this installs torch with the
#    appropriate wheel for the host: CUDA on Linux/x86 with NVIDIA, MPS on
#    Apple Silicon, CPU otherwise — pip resolves automatically).
echo "==> Installing kit dependencies (this can take 2-5 minutes)..."
python -m pip install --upgrade pip wheel
python -m pip install -e .

# 4. Make scripts executable
chmod +x scripts/*.sh

# 5. Copy env.example → .env if missing
if [ ! -f ".env" ]; then
  cp env.example .env
  echo "==> Created .env from env.example — edit it to add NGC_API_KEY / HF_TOKEN."
fi

# 6. Generate a small sample dataset for smoke testing
mkdir -p data
if [ ! -f "data/sample_ohlcv.csv" ]; then
  echo "==> Generating synthetic sample OHLCV..."
  python data/generate_sample.py --rows 600 --out data/sample_ohlcv.csv
fi

echo ""
echo "==> Bootstrap complete."
echo "    Next:"
echo "      source .venv/bin/activate"
echo "      python scripts/01_smoke_test.py --device $(python -c 'import torch; print(\"cuda:0\" if torch.cuda.is_available() else (\"mps\" if torch.backends.mps.is_available() else \"cpu\"))')"
