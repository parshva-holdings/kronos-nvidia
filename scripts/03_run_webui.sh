#!/usr/bin/env bash
# Launch the Streamlit dashboard (auto-fetches live NSE data, no CSV uploads).
# Default port 8501; override with WEBUI_PORT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -d .venv ]; then source .venv/bin/activate; fi

PORT="${WEBUI_PORT:-8501}"

# Streamlit by default writes telemetry to ~/.streamlit. Disable for cleanliness.
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "==> Launching Kronos · Nifty Live on http://localhost:$PORT"
echo "    (Ctrl-C to stop)"
exec streamlit run webui_streamlit/app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
