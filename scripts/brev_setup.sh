#!/bin/bash
# Brev VM-Mode setup script. Runs once after Brev provisions the VM and clones
# this repo. Bootstraps the kit, pre-fetches model weights, and brings up:
#   - Streamlit dashboard on :8501
#   - FastAPI gateway     on :8000
#   - Jupyter Lab         on :8888  (Brev-managed)
#
# Idempotent — safe to re-run by hand if you SSH in later.
#
# Usage in Brev wizard (Step 2 — "Do you want to run a setup script?"):
#   Paste Script tab → paste the contents of THIS file verbatim.
#
# NOTE: Brev's wizard validates that the shebang is exactly "#!/bin/bash"
# — `#!/usr/bin/env bash` will be rejected, even though both work at runtime.
set -euo pipefail
exec > >(tee -a /var/log/kronos_setup.log) 2>&1

echo "============================================================"
echo "  Kronos × NVIDIA bootstrap — Brev VM Mode"
echo "  $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Locate the cloned repo. Brev's path varies by version, so we probe
#    common locations.
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
  echo "ERROR: kronos-nvidia repo not found. Probed:"
  echo "  /home/ubuntu/verb-workspace/kronos-nvidia"
  echo "  /home/ubuntu/kronos-nvidia"
  echo "  /workspace/kronos-nvidia"
  echo "  $(pwd)"
  exit 1
fi
cd "$REPO"
echo "==> Repo at $REPO"

# ---------------------------------------------------------------------------
# 2. Run the kit's bootstrap (clones upstream Kronos, creates venv, installs deps).
# ---------------------------------------------------------------------------
echo "==> Running bootstrap (this takes 3-5 minutes on first boot)"
bash scripts/00_bootstrap.sh

# Activate the venv for the rest of this script
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 3. Pre-download Kronos weights so the first dashboard request is warm.
# ---------------------------------------------------------------------------
echo "==> Pre-downloading Kronos weights (background)"
nohup python scripts/02_download_models.py >/var/log/kronos_dl.log 2>&1 &
disown

# ---------------------------------------------------------------------------
# 4. Start Streamlit dashboard (port 8501).
# ---------------------------------------------------------------------------
echo "==> Starting Streamlit dashboard on :8501"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
nohup streamlit run webui_streamlit/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false \
  >/var/log/kronos_streamlit.log 2>&1 &
disown

# ---------------------------------------------------------------------------
# 5. Start FastAPI gateway (port 8000).
# ---------------------------------------------------------------------------
echo "==> Starting FastAPI gateway on :8000"
nohup uvicorn nvidia.api_server:app \
  --host 0.0.0.0 --port 8000 --log-level info \
  >/var/log/kronos_api.log 2>&1 &
disown

# ---------------------------------------------------------------------------
# 6. Brief readiness check
# ---------------------------------------------------------------------------
sleep 4
echo ""
echo "============================================================"
echo "  Setup complete"
echo "============================================================"
echo "  Streamlit dashboard:  http://<brev-public-host>:8501"
echo "  FastAPI gateway:      http://<brev-public-host>:8000/docs"
echo "  Jupyter Lab:          http://<brev-public-host>:8888  (Brev-managed)"
echo ""
echo "  Logs:"
echo "    /var/log/kronos_setup.log"
echo "    /var/log/kronos_streamlit.log"
echo "    /var/log/kronos_api.log"
echo "    /var/log/kronos_dl.log"
echo ""
echo "  To restart services manually:"
echo "    cd $REPO && bash scripts/brev_setup.sh"
echo "============================================================"
