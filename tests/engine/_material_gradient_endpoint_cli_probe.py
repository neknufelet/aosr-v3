"""終點命令列考卷的雙精度子行程：同一個行程裡跑兩份終點，寫原始紀錄；比對與判決在父行程。

父行程把命令列跑完搜尋的資料夾整份抄成 ``<目錄>/<搜尋代號>/``（不帶命令列那一次的終點檔）。這裡先叫產品入口
``worker.run_endpoint_worker`` 對它跑終點（寫 ``endpoint.json``），再用同一份 θ、preset 加方案原本的牆（考卷自己從方案
舊欄換成固定阻抗，不走產品的 ``scene_wall_materials``）直接叫 ``endpoint_check``，並用完整模型解原方案六面、交同一把尺
（``objective_from_pressures``）與同一支大峰（``endpoint.big_peaks``）。

兩份在同一個行程算，父行程逐位比。探針的 ``endpoint_probe_env`` 與命令列工作行程的
``runtime.child_process_env`` 都不帶 NPY_DISABLE_CPU_FEATURES、OPENBLAS_CORETYPE、XLA_FLAGS 三格釘選。
跨入口仍沿既有契約：非浮點逐字、浮點用精度契約登記簿的尺。
GPU 那一跑的 NumPy 評分只核 CPU 記下的輸入身分後回壓力雜湊值；CPU 那跑仍呼叫真評分，
GPU 完整解、精修與六類 JAX 評分照算。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.construction_presets import load_construction_presets
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint, endpoint_six
from aosr.gradient.material_gradient import wall_designs_from_presets
from aosr.gradient.endpoint_run import ENDPOINT_FILE, run_endpoint
from aosr.physics import fem_full_gpu
from aosr.physics.fem_reduced_boundary import BoundaryImpedances
from tests.engine._full_gpu_fake import SingleFactor
from aosr.gradient.run import RESULT_FILE, GradientResult, check_inputs, read_project
from aosr.gradient.worker import run_endpoint_worker
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.reporting.scheme import Scheme
from aosr.scoring.gradient_objective_jax import gradient_objective_constants, gradient_objective_from_pressures
from aosr.scoring.gradient_objective_settings import gradient_objective_settings
from aosr.scoring.stage_one_score import stage_one_score_settings
from aosr.scoring.six_category_jax import SixCategoryArrays, SixCategoryConstants, six_category_costs

Record = dict[str, object]
type NumpyReading = tuple[endpoint_six.Categories, float | None, str | None]


def _numpy_identity(context: endpoint_six.ScoringContext, scheme: Scheme,
                    pressure: NDArray[np.complex128]) -> tuple[str, str, str, tuple[int, ...], str]:
    """同一份方案、計算與物理身分及壓力布局；GPU 單精度解的壓力位元另算。"""
    return (scheme.model_dump_json(), context.program_fingerprint, context.physics_identity,
            pressure.shape, pressure.dtype.str)


def _endpoint(source: Path) -> Record:
    """產品入口那一份的控制組：同一份輸入直接叫 ``endpoint_check``；原方案六面完整模型解交同一把尺、同一支大峰。"""
    project = read_project(source)
    scheme, model, settings = check_inputs(project, allow_legacy_objective=True)
    result = GradientResult.model_validate_json((source / RESULT_FILE).read_bytes())
    fixed = scheme.scene.impedance_pa_s_per_m_by_wall
    axis_id = scheme.scene.low_frequency_axis
    assert fixed is not None and axis_id is not None
    kept: dict[str, SurfaceMaterial] = {name: ConstantImpedance(impedance_pa_s_m=value) for name, value in fixed.items()}
    designs = wall_designs_from_presets(project.presets, settings,
                                        load_construction_presets(config_path("construction_presets.toml")), kept=kept)
    operators = export_operators(scheme, capabilities=load_capabilities(project.capabilities_path),
                                 directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    score = gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")), project.purpose)
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    theta = np.asarray([float.fromhex(value) for value in result.theta_hex], dtype=np.float64)
    checked = endpoint.endpoint_check(
        operators, designs, theta, model=model, objective_settings=score, receiver_set=scheme.receiver_set,
        low_frequency_axis=axis_id, presets=load_construction_presets(config_path("construction_presets.toml")),
        settings=settings, air=air, ranges=ranges)
    axis = tuple(model.frequencies_hz)
    pressures = endpoint.full_model_pressures(
        operators, {wall: kept[wall.wall_name()] for wall in Wall.all()}, axis, air=air, ranges=ranges,
        max_relative_residual=model.ledger.settings.full_solve_max_relative_residual)
    constants = gradient_objective_constants(axis, score, scheme.receiver_set, len(model.source_names))
    base = gradient_objective_from_pressures(constants, jnp.asarray(pressures))
    peaks = endpoint.big_peaks(pressures, speaker_ids=model.source_names, receiver_set=scheme.receiver_set,
                               frequencies_hz=axis, low_frequency_axis=axis_id, scoring_range_hz=score.stage_one.scored_range_hz)
    original = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), project.purpose)
    def notes(values: NDArray[np.complex128]) -> endpoint.UnscoredBand:
        return endpoint.below_scoring_notes(values, speaker_ids=model.source_names, receiver_set=scheme.receiver_set,
                                            frequencies_hz=axis, low_frequency_axis=axis_id,
                                            stage_one_range_hz=original.scored_range_hz,
                                            stage_three_lower_hz=score.stage_one.scored_range_hz[0])
    report = checked.report.model_copy(update={"below_scoring": notes(checked.full_pressure)})
    return {"report": json.loads(report.to_json()), "baseline_below_scoring": notes(pressures).model_dump(mode="json"), "baseline_total": float(base.screening_total).hex(),
            "baseline_components": [float(base.mean_own_line_ripple_rms_db).hex(),
                                    float(base.mean_seat_robustness_rms_db).hex()],
            "baseline_objective": [float(base.objective_total).hex(), float(base.big_peak_proxy_db).hex(),
                                   float(base.interior_max_residual_db).hex()],
            "baseline_peaks": peaks}


def main(directory: Path, name: str) -> None:
    source = directory / name
    devices: list[bool] = []
    identities: list[tuple[str, str, str, tuple[int, ...], str]] = []
    real_numpy = endpoint_six.numpy_scores

    def record_numpy(context: endpoint_six.ScoringContext, scheme: Scheme,
                     pressure: NDArray[np.complex128]) -> NumpyReading:
        identities.append(_numpy_identity(context, scheme, pressure))
        return real_numpy(context, scheme, pressure)

    gpu_numpy_calls = 0

    def gpu_numpy(context: endpoint_six.ScoringContext, scheme: Scheme,
                  pressure: NDArray[np.complex128]) -> NumpyReading:
        nonlocal gpu_numpy_calls
        assert gpu_numpy_calls < len(identities), 'GPU 多叫了一次 numpy_scores'
        assert _numpy_identity(context, scheme, pressure) == identities[gpu_numpy_calls]
        gpu_numpy_calls += 1
        # 此跑只驗 GPU 接線；CPU 那跑與獨立六類考卷仍核真 NumPy 分數。
        # 壓力來自真單精度解，不要求與 CPU 位元相同；值照 solver stand-in 由位元導出。
        raw = hashlib.sha256(np.ascontiguousarray(pressure).tobytes()).digest()
        value = 1.0 + int.from_bytes(raw[:6], 'big') / 2.0**48
        categories: endpoint_six.Categories = (value, value, value, value, value, value)
        return categories, value, None

    def observe(c: SixCategoryConstants, fem: jax.Array, normal: jax.Array, early: jax.Array,
                paris: jax.Array, pressures: jax.Array, flags: jax.Array) -> SixCategoryArrays:
        devices.append(all(all(device.platform == 'cpu' for device in value.devices()) and value.dtype == jnp.complex128
            for value in (fem, normal, early, paris, pressures)) and all(device.platform == 'cpu' for device in flags.devices()))
        return six_category_costs(c, fem, normal, early, paris, pressures, flags)
    with (patch.object(endpoint_six, 'six_category_costs', observe),
          patch.object(endpoint_six, 'numpy_scores', record_numpy)):
        code = run_endpoint_worker(source)
    record = json.loads((source / ENDPOINT_FILE).read_text(encoding="utf-8")) if code == 0 else None
    output = {"worker_code": code, "worker_record": record, "endpoint": _endpoint(source)}
    factors: list[SingleFactor] = []
    steps: list[int] = []
    solve = fem_full_gpu.GPUFullSolver.solve

    def factor(a: csr_matrix[np.complex64], b: NDArray[np.complex64]) -> SingleFactor:
        result = SingleFactor(a, b)
        factors.append(result)
        return result

    def capture(solver: fem_full_gpu.GPUFullSolver, frequency: float,
                walls: BoundaryImpedances) -> NDArray[np.complex128]:
        solution = solve(solver, frequency, walls)
        steps.append(solver.refinement_steps[-1])
        return solution

    (source / ENDPOINT_FILE).rename(source / "endpoint-cpu.json")
    with (patch.object(fem_full_gpu, "nvmath_factor", factor), patch.object(fem_full_gpu.GPUFullSolver, "solve", capture),
          patch.object(endpoint_six, "six_category_costs", observe), patch.object(endpoint_six, 'numpy_scores', gpu_numpy)):
        gpu = run_endpoint(source, device="gpu")
    output['cpu_numpy_calls'] = len(identities)
    output['gpu_numpy_calls'] = gpu_numpy_calls
    output["cpu_six_scoring"] = bool(devices) and all(devices)
    output["gpu_steps"] = steps
    output["gpu"] = gpu.model_dump(mode="json")
    output["gpu_factors"] = [{"freed": f.freed, "factorizations": f.factorizations, "calls": f.calls} for f in factors]
    (directory / "output.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(directory / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
