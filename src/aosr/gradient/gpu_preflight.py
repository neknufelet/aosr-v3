"""顯卡搜尋開始前，以同一個起點比 CPU 雙精度並保存原始值。"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aosr import runtime
from aosr.gradient.gpu import PrecisionComparison, precision_comparison, require_precision
from aosr.gradient.material_gradient import baseline_theta, depth_limits, descent_start, design_bounds

if TYPE_CHECKING:
    from aosr.gradient.run import GradientProblem, GradientProject


def _start_theta(problem: GradientProblem) -> tuple[float, ...]:
    start = descent_start(baseline_theta(problem.designs), design_bounds(problem.designs), depth_limits(problem.designs))
    return tuple(float(x) for x in start)


def measure_start(project: GradientProject, problem: GradientProblem) -> PrecisionComparison:
    """顯卡單精度與 CPU 雙精度各算一次 objective_total；CPU 那次明確放到 CPU 裝置。"""
    theta = _start_theta(problem)
    gpu_value, _ = problem.objective(jnp.asarray(theta, dtype=jnp.float64))
    cpu_double = cpu_reference(project, theta)
    return precision_comparison(cpu_double=cpu_double, gpu_single=float(gpu_value), theta=theta)


def cpu_reference(project: GradientProject, theta: Sequence[float]) -> float:
    """獨立 CPU 子行程：連建樣板與保留牆的預算都在 CPU，無須動裝置全域設定。"""
    env = runtime.child_process_env(threads=1)
    runtime.configure_jax(platforms="cpu", enable_x64=True, env=env)
    payload = project.model_dump_json() + "\n" + json.dumps([float(x).hex() for x in theta])
    result = subprocess.run([sys.executable, "-m", "aosr.gradient.gpu_reference"], input=payload,
                            capture_output=True, text=True, env=env, check=True)
    return float.fromhex(result.stdout.strip())


def ensure_start_precision(folder: Path, project: GradientProject, problem: GradientProblem) -> None:
    """在任何帳本寫入／搜尋前檢查；缺記錄或記錄的起點不同就重測，已有失敗記錄仍拒跑。"""
    from aosr.gradient.run import PROJECT_FILE, write_json_atomic

    if project.device != "gpu":
        return
    comparison = project.precision_check
    if comparison is not None:
        require_precision(comparison)
    if comparison is None or comparison.theta_hex != tuple(value.hex() for value in _start_theta(problem)):
        comparison = measure_start(project, problem)
        write_json_atomic(folder / PROJECT_FILE, project.model_copy(update={"precision_check": comparison}))
    require_precision(comparison)
