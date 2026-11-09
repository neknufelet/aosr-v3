"""終點探針跟 CLI 工作行程用同一套 CPU 指令集環境。

這些探針沒有錄製數值答案，逐位控制都在同一行程內比較。工作行程不繼承處理器釘選，
探針若帶釘選，反而會跨指令集重建常數，使逐位雜湊不同；不能沿用答案考卷的釘選環境。
"""
from __future__ import annotations

import os
from pathlib import Path

from aosr.runtime import child_process_env, configure_jax


def endpoint_probe_env(directory: Path) -> dict[str, str]:
    """雙精度 CPU 探針：執行緒與正式工作行程一致，另加入考卷模組與暫存路徑。"""
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    return env
