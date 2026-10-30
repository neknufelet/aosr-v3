"""終點命令列考卷的雙精度子行程：同一個行程裡跑兩份終點，寫原始紀錄；比對與判決在父行程。

父行程把命令列跑完搜尋的資料夾整份抄成 ``<目錄>/<搜尋代號>/``（不帶命令列那一次的終點檔）。這裡先叫產品入口
``worker.run_endpoint_worker`` 對它跑終點（寫 ``endpoint.json``），再用同一份 θ、preset 加方案原本的牆（考卷自己從方案
舊欄換成固定阻抗，不走產品的 ``scene_wall_materials``）直接叫 ``endpoint_check``，並用完整模型解原方案六面、交同一把尺
（``objective_from_pressures``）與同一支大峰（``endpoint.big_peaks``）。

兩份在同一個行程、同一組處理器釘選下算，父行程逐位比；命令列 ``endpoint`` 的子行程走 ``runtime.child_process_env``，
刻意不帶三格處理器釘選，跨機器（例如有 AVX-512 的雲端）替身那幾格會差在最後幾位，所以命令列那一份只跟這裡的產品入口
那一份比：非浮點逐字、浮點用精度契約登記簿的尺。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.construction_presets import load_construction_presets
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint
from aosr.gradient.material_gradient import wall_designs_from_presets
from aosr.gradient.endpoint_run import ENDPOINT_FILE
from aosr.gradient.run import RESULT_FILE, GradientResult, check_inputs, read_project
from aosr.gradient.worker import run_endpoint_worker
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.scoring.gradient_objective_jax import gradient_objective_constants, gradient_objective_from_pressures
from aosr.scoring.gradient_objective_settings import gradient_objective_settings

Record = dict[str, object]


def _endpoint(source: Path) -> Record:
    """產品入口那一份的控制組：同一份輸入直接叫 ``endpoint_check``；原方案六面完整模型解交同一把尺、同一支大峰。"""
    project = read_project(source)
    scheme, model, settings = check_inputs(project)
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
    return {"report": json.loads(checked.report.to_json()), "baseline_total": float(base.screening_total).hex(),
            "baseline_components": [float(base.mean_own_line_ripple_rms_db).hex(),
                                    float(base.mean_seat_robustness_rms_db).hex()],
            "baseline_objective": [float(base.objective_total).hex(), float(base.big_peak_proxy_db).hex()],
            "baseline_peaks": peaks}


def main(directory: Path, name: str) -> None:
    source = directory / name
    code = run_endpoint_worker(source)
    record = json.loads((source / ENDPOINT_FILE).read_text(encoding="utf-8")) if code == 0 else None
    output = {"worker_code": code, "worker_record": record, "endpoint": _endpoint(source)}
    (directory / "output.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(directory / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
