#!/usr/bin/env bash
# Launch the FastAPI inference gateway on $API_PORT (default 8000).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ".env" ]; then
  set -a; . ./.env; set +a
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PORT="${API_PORT:-8000}"
HOST="${API_HOST:-0.0.0.0}"

echo "==> Starting Kronos FastAPI gateway on $HOST:$PORT"
echo "    OpenAPI docs at http://localhost:$PORT/docs"
exec uvicorn nvidia.api_server:app \
  --host "$HOST" --port "$PORT" \
  --workers 1 \
  --log-level info
