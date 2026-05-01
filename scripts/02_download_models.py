#!/usr/bin/env python3
"""Pre-download Kronos weights so containers / Brev launchables boot warm.

Honors HF_TOKEN if present (avoids rate limits) and HF_HOME (controls cache dir).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / ".cache" / "huggingface"

REPOS = [
    "NeoQuasar/Kronos-Tokenizer-base",
    "NeoQuasar/Kronos-Tokenizer-2k",
    "NeoQuasar/Kronos-mini",
    "NeoQuasar/Kronos-small",
    "NeoQuasar/Kronos-base",
]


def main() -> int:
    cache = Path(os.environ.get("HF_HOME", str(DEFAULT_CACHE)))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    token = os.environ.get("HF_TOKEN")  # optional

    print(f"==> Cache dir: {cache}")
    for repo in REPOS:
        print(f"==> Downloading {repo}...")
        try:
            local = snapshot_download(repo_id=repo, token=token, cache_dir=str(cache))
            print(f"    -> {local}")
        except Exception as e:  # noqa: BLE001
            print(f"    !! failed: {e}", file=sys.stderr)
    print("==> Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
