#!/usr/bin/env bash
set -euo pipefail
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" uv run --python 3.12 --group dev pytest tests -q
