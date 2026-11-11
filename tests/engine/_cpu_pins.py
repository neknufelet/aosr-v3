"""開子行程的考卷共用：把父行程的三格處理器釘選原樣帶進子行程環境（#1077）。

``aosr.runtime.child_process_env`` 只繼承白名單，雲端 job 層設好的三格（numpy 的向量指令、OpenBLAS 的
核心、XLA 的指令集；值在 ``.github/workflows/verify.yml``）不會跟進子行程。考卷開子行程時一律寫
``forward_cpu_pins(child_process_env(...))``——規矩卡 child-process-tests-forward-cpu-pins 認的就是這個寫法。

父行程沒設的格不補：本機三格是空的，包了也是原樣；守的是雲端那一跑父子兩邊同一套指令。
子行程刻意要跟不帶釘選的正式工作行程同一環境的（例：``test_six_category_endpoint.py`` 的探針，#1051），
不要包，在卡上開一筆放行寫理由。
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from aosr import runtime

CPU_PIN_VARS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", runtime.XLA_FLAGS_VAR)


def forward_cpu_pins(env: Mapping[str, str], /) -> dict[str, str]:
    """``env`` 的一份副本，加上父行程有設的三格處理器釘選；``env`` 本身不動。"""
    pinned = dict(env)
    pinned.update({name: os.environ[name] for name in CPU_PIN_VARS if name in os.environ})
    return pinned
