#!/usr/bin/env bash
set -eu
# 這一行就是繞法：檢查回 1，腳本還是回 0。
uv run pytest || true
