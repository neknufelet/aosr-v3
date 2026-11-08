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
from aosr.gradient.gpu import (
    LegacyPrecisionComparison, PrecisionComparison, PrecisionReading, precision_comparison, require_precision, six_category_precision_comparison,
)
from aosr.gradient.material_gradient import baseline_theta, depth_limits, descent_start, design_bounds

if TYPE_CHECKING:
    from aosr.gradient.run import GradientProblem, GradientProject


def _start_theta(problem: GradientProblem) -> tuple[float, ...]:
    start = descent_start(baseline_theta(problem.designs), design_bounds(problem.designs), depth_limits(problem.designs))
    return tuple(float(x) for x in start)


def _uses_six_single(project: GradientProject, problem: GradientProblem) -> bool:
    """依正式包裝辨識新鏈；六類指紋即使有把關記錄，也不能拿雙精度函式冒充顯卡鏈。"""
    from aosr.gradient.six_category_gpu import SixCategorySinglePrecisionObjective
    from aosr.gradient.run import search_objective_settings

    if isinstance(getattr(problem,'objective',None),SixCategorySinglePrecisionObjective):
        return True
    if project.objective_settings_fingerprint == search_objective_settings(project.purpose).fingerprint:
        raise ValueError('六類顯卡搜尋必須使用單精度包裝，不能靜默走雙精度')
    return False


def measure_start(project: GradientProject, problem: GradientProblem) -> PrecisionComparison | LegacyPrecisionComparison:
    """同 θ 比 J，另記 product 與六類；CPU 建樣板與求值都在獨立 CPU 行程。"""
    from aosr.gradient.six_category_chain import SixCategoryChainAux

    six_single = _uses_six_single(project,problem)
    theta = _start_theta(problem)
    gpu_value, aux = problem.objective(jnp.asarray(theta, dtype=jnp.float64))
    if not six_single:
        return precision_comparison(cpu_double=cpu_reference(project, theta), gpu_single=float(gpu_value), theta=theta)
    if not isinstance(aux, SixCategoryChainAux):
        raise ValueError('六類單精度包裝必須回傳六類 aux')
    cpu = cpu_six_category_reference(project, theta)
    return six_category_precision_comparison(cpu_double=cpu.objective_total, gpu_single=float(gpu_value), theta=theta,
        cpu_product=cpu.product_total, gpu_product=float(aux.product_total), cpu_categories=cpu.categories,
        gpu_categories=tuple(float(x) for x in aux.categories[:6]))


def cpu_reference(project: GradientProject, theta: Sequence[float]) -> float:
    """獨立 CPU 子行程：連建樣板與保留牆的預算都在 CPU，無須動裝置全域設定。"""
    return float.fromhex(_cpu_reference_output(project, theta))


def cpu_six_category_reference(project: GradientProject, theta: Sequence[float]) -> PrecisionReading:
    """同次 CPU 雙精度求值取得 J、產品總分、六類原值。"""
    return PrecisionReading.model_validate_json(_cpu_reference_output(project, theta, six_category=True))


def _cpu_reference_output(project: GradientProject, theta: Sequence[float], *, six_category: bool = False) -> str:
    env = runtime.child_process_env(threads=1)
    runtime.configure_jax(platforms="cpu", enable_x64=True, env=env)
    payload = project.model_dump_json() + "\n" + json.dumps([float(x).hex() for x in theta])
    if six_category:
        payload += '\nsix_category'
    result = subprocess.run([sys.executable, "-m", "aosr.gradient.gpu_reference"], input=payload,
                            capture_output=True, text=True, env=env, check=True)
    return result.stdout.strip()


def require_search_contract(project: GradientProject) -> None:
    """舊 project.json 仍讀得出；舊 dB 契約不可拿來接續六類搜尋。"""
    if project.device == 'gpu' and isinstance(project.precision_check, LegacyPrecisionComparison):
        raise ValueError('舊顯卡精度契約是低頻 dB，六類搜尋只讀舊資料、不能接續；請重跑')


def ensure_start_precision(folder: Path, project: GradientProject, problem: GradientProblem) -> None:
    """在任何帳本寫入／搜尋前檢查；缺記錄或記錄的起點不同就重測，已有失敗記錄仍拒跑。"""
    from aosr.gradient.run import PROJECT_FILE, write_json_atomic

    if project.device != "gpu":
        return
    if _uses_six_single(project,problem):
        require_search_contract(project)
    comparison = project.precision_check
    if comparison is not None:
        require_precision(comparison)
    if comparison is None or comparison.theta_hex != tuple(value.hex() for value in _start_theta(problem)):
        comparison = measure_start(project, problem)
        write_json_atomic(folder / PROJECT_FILE, project.model_copy(update={"precision_check": comparison}))
    require_precision(comparison)
