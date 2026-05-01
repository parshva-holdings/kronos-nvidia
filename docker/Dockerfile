# NGC PyTorch base — ships PyTorch + CUDA 12.x + cuDNN + NCCL tuned for NVIDIA GPUs.
# Tag chosen to balance freshness with reproducibility. Bump deliberately.
# Pull requires: docker login nvcr.io (with NGC_API_KEY).
FROM nvcr.io/nvidia/pytorch:25.11-py3

# OCI labels — Brev surfaces these in the launchable card.
LABEL org.opencontainers.image.title="kronos-nvidia"
LABEL org.opencontainers.image.description="Kronos financial foundation model on NVIDIA NGC PyTorch"
LABEL org.opencontainers.image.source="https://github.com/shiyu-coder/Kronos"
LABEL org.opencontainers.image.licenses="MIT"

# Build args (override at build time if needed)
ARG KRONOS_REF=master

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_OFFLINE=0 \
    KRONOS_DEVICE=cuda:0

WORKDIR /app

# Clone upstream Kronos at a known ref so the container is reproducible.
RUN git clone --depth 1 --branch ${KRONOS_REF} https://github.com/shiyu-coder/Kronos.git upstream

# Kit deps. NGC base already ships torch + numpy + pandas, so we only add what's missing.
# We avoid the upstream's pinned `pandas==2.2.2` because it conflicts with the NGC pandas.
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
    einops>=0.8 \
    huggingface_hub>=0.33 \
    safetensors>=0.6 \
    tqdm>=4.66 \
    streamlit>=1.40 plotly>=5.17 yfinance>=0.2.40 \
    fastapi>=0.115 'uvicorn[standard]>=0.32' \
    pydantic>=2.8 python-dotenv>=1.0 \
    jupyterlab>=4.0

# Copy our gateway, scripts, and configs (after pip install for cache friendliness).
COPY nvidia /app/nvidia
COPY scripts /app/scripts
COPY data /app/data
COPY README.md /app/README.md
COPY env.example /app/env.example

# Add the Streamlit UI source
COPY webui_streamlit /app/webui_streamlit

# Pre-create cache dirs (Brev volume-mounts these for warm starts)
RUN mkdir -p /opt/hf_cache /app/outputs /app/checkpoints \
    && chmod +x /app/scripts/*.sh

# Expose: 8501 Streamlit dashboard, 8000 FastAPI gateway, 8888 Jupyter
EXPOSE 8501 8000 8888

# Default: run gateway + Streamlit. Override CMD to launch a single service or finetune.
CMD ["bash", "-lc", "\
  uvicorn nvidia.api_server:app --host 0.0.0.0 --port 8000 >/tmp/api.log 2>&1 & \
  exec streamlit run /app/webui_streamlit/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false \
"]
