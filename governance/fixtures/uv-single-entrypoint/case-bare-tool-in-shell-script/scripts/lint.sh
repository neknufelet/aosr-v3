#!/usr/bin/env bash
# 樣本道具：workflow 那一行完全合法，病灶整個藏在腳本裡。
set -euo pipefail
ruff check .
mypy governance
