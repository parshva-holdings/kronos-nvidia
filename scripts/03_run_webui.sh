#!/usr/bin/env bash
# Launch the upstream Kronos Flask webui on $WEBUI_PORT (default 7070).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ".env" ]; then
  set -a; . ./.env; set +a
fi

if [ ! -d "upstream/webui" ]; then
  echo "upstream/webui not found — run scripts/00_bootstrap.sh first" >&2
  exit 1
fi

# Activate venv if present
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# The upstream app.py hard-codes port 7070 and host 0.0.0.0; if WEBUI_PORT differs,
# we patch via env-driven monkeypatch in run.py rather than editing upstream.
PORT="${WEBUI_PORT:-7070}"
echo "==> Starting Kronos webui on :$PORT (device=${KRONOS_DEVICE:-auto})"

cd upstream/webui
python run.py
