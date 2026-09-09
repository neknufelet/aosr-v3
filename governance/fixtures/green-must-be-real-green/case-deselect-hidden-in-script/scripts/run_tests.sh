#!/usr/bin/env bash
# 樣本道具：v2 的 nightly_hermetic.sh 就是這個形狀——正式跑法藏在腳本裡，
# 排除清單也藏在腳本裡，看 workflow 完全看不出來。
set -euo pipefail
pytest --junitxml=governance/receipts/pytest.junit.xml \
  --deselect tests/test_env.py::test_needs_gpu \
  --deselect tests/test_env.py::test_needs_pydiso
