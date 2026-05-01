.PHONY: help bootstrap smoke models webui api finetune docker-build docker-up docker-down brev-bundle clean

help:
	@echo "Kronos × NVIDIA — common targets"
	@echo "  make bootstrap     — clone upstream Kronos, create .venv, install deps"
	@echo "  make smoke         — run a CPU/MPS/CUDA smoke test"
	@echo "  make models        — pre-download HF weights into .cache/"
	@echo "  make webui         — launch upstream Flask demo on :7070"
	@echo "  make api           — launch FastAPI gateway on :8000"
	@echo "  make finetune      — torchrun multi-GPU fine-tune (set NUM_GPUS=N)"
	@echo "  make docker-build  — build the NGC PyTorch image"
	@echo "  make docker-up     — docker compose up (NVIDIA host)"
	@echo "  make docker-down   — docker compose down"
	@echo "  make brev-bundle   — print the Brev launchable URL template"
	@echo "  make clean         — remove .venv, caches, upstream/"

bootstrap:
	./scripts/00_bootstrap.sh

smoke:
	. .venv/bin/activate && python scripts/01_smoke_test.py

models:
	. .venv/bin/activate && python scripts/02_download_models.py

webui:
	./scripts/03_run_webui.sh

api:
	./scripts/04_serve_api.sh

finetune:
	./scripts/05_finetune.sh

docker-build:
	docker build -t kronos-nvidia:latest -f docker/Dockerfile .

docker-up:
	docker compose -f docker/docker-compose.yml up

docker-down:
	docker compose -f docker/docker-compose.yml down

brev-bundle:
	@echo "Open https://brev.nvidia.com/launchables/create"
	@echo "Choose: 'I have code files in a git repository'"
	@echo "URL: <paste this repo's GitHub URL>"
	@echo "Container: nvcr.io/nvidia/pytorch:25.11-py3"
	@echo "Compose: docker/docker-compose.brev.yml"
	@echo "GPU:     L40S (or A100-40 for fine-tune)"
	@echo "See nvidia/brev/README.md for full walkthrough."

clean:
	rm -rf .venv upstream/ .cache hf_cache nim_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
