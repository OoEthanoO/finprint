"""Deploy finprint to a Hugging Face Space (Docker SDK).

Prereqs: a Hugging Face account and a write token (https://huggingface.co/settings/tokens).

    HF_TOKEN=hf_xxx SPACE_ID=<username>/finprint python -m scripts.deploy_hf

Creates the Space if needed and uploads only the files the container needs.
The Space then builds the Dockerfile and serves the app automatically.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

# Only these paths are pushed to the Space (keeps .venv/data/etc. out).
ALLOW = [
    "finprint/**",
    "app/**",
    "models/**",
    "Dockerfile",
    "requirements-serve.txt",
    ".dockerignore",
    "README.md",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("SPACE_ID")
    if not token or not space_id:
        sys.exit("Set HF_TOKEN and SPACE_ID (e.g. SPACE_ID=yourname/finprint)")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True
    )
    print(f"Uploading to space '{space_id}' ...")
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(ROOT),
        allow_patterns=ALLOW,
        commit_message="Deploy finprint",
    )
    url = f"https://huggingface.co/spaces/{space_id}"
    print(f"\nDone. Building now — watch it at:\n  {url}")


if __name__ == "__main__":
    main()
