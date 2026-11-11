"""迷你版共用小工具：樣本只要它在、最外層定義了那支函式。"""
from __future__ import annotations

import os
from collections.abc import Mapping

CPU_PIN_VARS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")


def forward_cpu_pins(env: Mapping[str, str], /) -> dict[str, str]:
    pinned = dict(env)
    pinned.update({name: os.environ[name] for name in CPU_PIN_VARS if name in os.environ})
    return pinned
