"""牆面材料梯度搜尋前半段（#561）：整條鏈梯度對 numpy 差分、投影下降、取規格、空腔重調、步驟帳本、拒收。

替身用 ``test_fem_reduced`` 的小房間 fixture，不另建。第一階段目標要「一個主位＋前後左右上下六點」
與一條宣告過的低頻軸，那個 fixture 只有三個座位、軸是正式有限元素軸每隔一點，所以這裡從同一份替身
衍生一份：座位取值列換成三個真座位取值列的七個線性組合（只為湊齊主位加六點的形狀，不是物理位置），
頻率軸換成整條正式有限元素軸（投影小矩陣與頻率無關，只是多算中間的頻點）。JAX 與 numpy 兩邊用同一份衍生替身。

雙精度的梯度、下降、取規格、重調與帳本寫入在子行程（只叫公開入口）；中央差分、比對與判決在父行程，
差分走 numpy 版 ``evaluate_reduced``＋既有評分程式（``evaluate_stage_one_low_frequency``＋``score_stage_one``），
不走 JAX。梯度的尺沿用傳遞矩陣考卷那兩把；單精度只量不登記。
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ, LowFrequencyAxis
from aosr.config.material_gradient import (
    DescentSettings, FamilySettings, MaterialGradientSettings, load_material_gradient_settings,
)
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.physics.fem_reduced import ReducedModel, evaluate_reduced, system_coefficients
from aosr.physics.fem_reduced_store import save_reduced
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet, primary_cross
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import (
    StageOneLowFrequencyInput, StageOneSeatCurve, StageOneSpeakerCurves,
)
from aosr.scoring.stage_one_objective_jax import StageOneObjectiveArrays, objective_constants
from aosr.scoring.stage_one_score import StageOneScoreSettings, score_stage_one, stage_one_score_settings
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine._precision_contracts import contract_value
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS


PROBE = Path(__file__).with_name("_material_gradient_probe.py")
PURPOSE = "dedicated_two_channel_listening_room"
# 中央差分的相對步長，沿用傳遞矩陣考卷的 1e-5；不是產品精度契約。棉板與空腔都是公分級，
# 不是替身考卷那片 0.8 mm 穿孔板。施工時量（two_rigid 地板棉板厚度那一格最差）：3e-5／1e-5／3e-6 的逐格相對差
# 2.1e-7／2.2e-8／1.6e-9，照 h² 縮、是差分本身的截斷誤差；1e-6 起捨入主導（約 1e-8）。1e-4 會超過梯度尺。
DIFFERENCE_STEP = 1e-5
# 兩組六面：兩面 48K 棉板＋空腔、一面 48K 貼牆、一面 24K、兩面剛性；與六面全貼（三面帶空腔）。
CASES: dict[str, tuple[str | None, ...]] = {
    "two_rigid": ("glass_wool_48k_50mm_gap50", None, "glass_wool_48k_50mm_gap50", "glass_wool_48k_50mm",
                  None, "glass_wool_24k_50mm"),
    "all_lined": ("glass_wool_48k_50mm_gap50", "glass_wool_96k_50mm", "glass_wool_48k_50mm_gap100",
                  "glass_wool_48k_50mm", "glass_wool_48k_50mm_gap50", "glass_wool_24k_50mm"),
}
DESCENT_CASE = "two_rigid"
Record = dict[str, object]


@cache
def _settings() -> MaterialGradientSettings:
    return load_material_gradient_settings(config_path("material_gradient.toml"))


@cache
def _presets() -> tuple[ConstructionPreset, ...]:
    return load_construction_presets(config_path("construction_presets.toml"))


@cache
def _score() -> StageOneScoreSettings:
    return stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), PURPOSE)


@cache
def _receivers() -> ReceiverSet:
    return primary_cross(primary_position_m=(1.2, 0.7, 0.6), offset_m=0.1)


def _designs(name: str) -> tuple[gradient_search.SurfaceDesign, ...]:
    return gradient_search.wall_designs_from_presets(CASES[name], _settings(), _presets())


def _chain_model(model: ReducedModel) -> ReducedModel:
    """同一份替身換座位列（三個真座位列的七個組合）與頻率軸（整條正式有限元素軸）；不重建、不重解。"""
    front, back, side = model.matrices.receivers
    rows = np.asarray([front, back, side, (front + back) / 2, (front + side) / 2, (back + side) / 2,
                       (front + back + side) / 3], dtype=np.float64)
    return dataclasses.replace(
        model, receiver_names=tuple(point.receiver_id for point in _receivers().points),
        frequencies_hz=FEM_LANE_FREQUENCIES_HZ, matrices=dataclasses.replace(model.matrices, receivers=rows))


def _probe_point(designs: tuple[gradient_search.SurfaceDesign, ...]) -> NDArray[np.float64]:
    """梯度探點：48K 厚度 0.071、其他家族在唯一合法厚度、空腔各自不同的內點；不在任何盒界上（固定厚度除外）。"""
    values = []
    for index, design in enumerate(designs):
        if design.thickness_variable is not None:
            values.append(0.071 if len(design.legal_thicknesses_m) > 1 else design.legal_thicknesses_m[0])
        if design.cavity_variable is not None:
            values.append(0.037 + 0.011 * index)
    return np.asarray(gradient_search.checked_theta(np.asarray(values, dtype=np.float64), designs), dtype=np.float64)


def _decode(values: object) -> NDArray[np.float64]:
    return np.asarray([float.fromhex(value) for value in cast(list[str], values)], dtype=np.float64)


def _decode_complex(node: object) -> NDArray[np.complex128]:
    row = cast(Record, node)
    return (_decode(row["real"]) + 1j * _decode(row["imag"])).reshape(cast(list[int], row["shape"]))


@pytest.fixture(scope="module")
def chain_model(model: ReducedModel) -> ReducedModel:
    return _chain_model(model)


@pytest.fixture(scope="module")
def runs(chain_model: ReducedModel, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Record]]:
    directory = tmp_path_factory.mktemp("material-gradient")
    save_reduced(chain_model, directory / "model.npz")
    cases = {name: {"presets": list(presets), "theta": [value.hex() for value in _probe_point(_designs(name))],
                    "step": DIFFERENCE_STEP, "descent": name == DESCENT_CASE} for name, presets in CASES.items()}
    problem = {"cases": cases, "purpose": PURPOSE, "receivers": _receivers().model_dump(mode="json")}
    (directory / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    process = subprocess.run([sys.executable, str(PROBE), str(directory)], cwd=directory, env=env,
                             capture_output=True, text=True, timeout=600)
    assert process.returncode == 0, process.stdout + process.stderr
    return directory, cast(dict[str, Record], json.loads((directory / "output.json").read_text(encoding="utf-8")))


# ---- numpy 參考：既有替身求值＋既有評分程式 ----

def _numpy_total(model: ReducedModel, impedances: NDArray[np.complex128], mask: NDArray[np.bool_]) -> float:
    pressures = []
    for index, frequency in enumerate(model.frequencies_hz):
        walls = {wall: None if mask[w] else complex(impedances[w, index]) for w, wall in enumerate(Wall.all())}
        k2, coefficients = system_coefficients(frequency, walls, density_kg_m3=model.density_kg_m3,
                                               sound_speed_m_s=model.sound_speed_m_s)
        pressures.append(evaluate_reduced(model.matrices, k2, coefficients))
    energies = np.abs(np.asarray(pressures)) ** 2
    points = _receivers().points
    speakers = []
    for speaker_index, speaker_id in enumerate(model.source_names):
        seats = {point.receiver_id: StageOneSeatCurve(
            receiver_id=point.receiver_id, direction_relative_to_primary=point.direction_relative_to_primary,
            report_id=f"report-{point.receiver_id}", frequencies_hz=model.frequencies_hz,
            fem_energy=tuple(float(value) for value in energies[:, speaker_index, seat_index]))
            for seat_index, point in enumerate(points)}
        primary = next(point.receiver_id for point in points if point.role == ReceiverRole.PRIMARY)
        speakers.append(StageOneSpeakerCurves(
            speaker_id=speaker_id, primary=seats[primary],
            surrounding=tuple(curve for key, curve in seats.items() if key != primary)))
    data = StageOneLowFrequencyInput(candidate_id="material-gradient", low_frequency_axis=LowFrequencyAxis.SEARCH,
                                     speakers=tuple(speakers))
    score = score_stage_one(evaluate_stage_one_low_frequency(data, scoring_range_hz=_score().scored_range_hz), _score())
    assert score.state == "scored" and score.screening_total is not None, score.reason_codes
    return score.screening_total


def _mask(name: str) -> NDArray[np.bool_]:
    return np.asarray([preset is None for preset in CASES[name]], dtype=np.bool_)


# ---- 整條鏈梯度 ----

@pytest.mark.parametrize("name", CASES)
def test_chain_gradient_matches_numpy_difference_per_parameter(
        chain_model: ReducedModel, runs: tuple[Path, dict[str, Record]], name: str) -> None:
    """抓鏈中任何一段斷線或接錯面：逐參數各自取尺度比 θ_i·∂J/∂θ_i，不跨牆聚合，弱格靠絕對項。

    關掉 #952 審查留下的「同一面弱頻點的錯梯度可躲過逐牆尺度」：這裡每一格是一個純量，沒有聚合可躲。
    """
    row = runs[1][name]
    designs = _designs(name)
    assert row["names"] == list(gradient_search.design_names(designs))
    assert row["dtype"] == "float64" and row["gradient_dtype"] == "float64"
    theta = _probe_point(designs)
    mask = _mask(name)
    automatic = _decode(row["gradient"])
    assert np.isfinite(automatic).all() and automatic.shape == theta.shape
    # 前向：同一點 JAX 合分對 numpy 合分（梯度尺當前向粗檢；兩段各自的正式尺在精度契約登記簿）。
    center = _numpy_total(chain_model, _decode_complex(row["center"]), mask)
    assert abs(float.fromhex(cast(str, row["value"])) - center) <= GRADIENT_REL * abs(center) + GRADIENT_ABS
    # 編譯對未編譯：照審查裁定用尺、不逐位；尺是目標 JAX 版那一條契約（合分相對差）。
    compiled, eager = float.fromhex(cast(str, row["value"])), float.fromhex(cast(str, row["eager_value"]))
    assert abs(compiled - eager) <= contract_value("stage_one_objective_jax_vs_python") * abs(eager)
    points = cast(dict[str, object], row["points"])
    differences = []
    for index, value in enumerate(theta):
        upper = _numpy_total(chain_model, _decode_complex(points[f"{index}.upper"]), mask)
        lower = _numpy_total(chain_model, _decode_complex(points[f"{index}.lower"]), mask)
        central = (upper - lower) / (2.0 * DIFFERENCE_STEP * value) * value
        scaled = automatic[index] * value
        differences.append(abs(scaled - central))
        assert abs(scaled - central) <= GRADIENT_REL * abs(central) + GRADIENT_ABS, (row["names"], index, scaled, central)
    assert np.all(np.isfinite(differences))


# ---- 下降 ----

def _descent(runs: tuple[Path, dict[str, Record]], key: str = "descent") -> Record:
    return cast(Record, runs[1][DESCENT_CASE][key])


def _pairs(names: Sequence[str]) -> list[tuple[int, int]]:
    """同一面的棉板格與空腔格（樣板 layer1 是棉板、layer2 是空腔）；從變數名自己配，不借產品的總深度限制表。"""
    index = {name: position for position, name in enumerate(names)}
    return [(position, index[name.replace("layer1.", "layer2.")]) for name, position in index.items()
            if name.endswith("layer1.thickness_m") and name.replace("layer1.", "layer2.") in index]


def test_descent_monotone_inside_box_and_stop_reason_matches_numbers(runs: tuple[Path, dict[str, Record]]) -> None:
    """抓值上升、跳出盒界或總深度、停止原因跟記下的數字對不上（記梯度夠小但最後範數不小、記步數到上限但沒走滿）、
    求值次數比走過的步還少。"""
    descent = _descent(runs)
    settings = _settings().descent
    row = runs[1][DESCENT_CASE]
    lower, upper = _decode(row["lower"]), _decode(row["upper"])
    thetas = [_decode(step) for step in cast(list[list[str]], descent["theta"])]
    values = _decode(descent["value"])
    norms = _decode(descent["gradient_norm"])
    sizes = _decode(descent["step_size"])
    assert np.array_equal(thetas[0], _decode(row["start"]))
    assert np.all(np.diff(values) <= 0.0)
    assert values[-1] < values[0]
    assert all(np.all((lower <= theta) & (theta <= upper)) for theta in thetas)
    top = _settings().family[0].max_total_depth_m
    pairs = _pairs(cast(list[str], row["names"]))
    assert pairs and all(theta[t] + theta[c] <= top for theta in thetas for t, c in pairs)
    assert cast(int, descent["evaluations"]) >= len(thetas)
    assert np.all(sizes[1:] >= settings.min_step) and np.all(sizes[1:] <= settings.initial_step)
    taken = len(thetas) - 1
    reason = descent["stop_reason"]
    assert reason in gradient_search.STOP_REASONS
    if reason == gradient_search.STOP_GRADIENT:
        assert norms[-1] < settings.gradient_norm_stop and np.all(norms[:-1] >= settings.gradient_norm_stop)
    elif reason == gradient_search.STOP_MAX_STEPS:
        assert taken == settings.max_steps and np.all(norms[:-1] >= settings.gradient_norm_stop)
    else:
        assert float.fromhex(cast(str, descent["final_trial_step"])) < settings.min_step
    assert taken <= settings.max_steps
    assert np.array_equal(_decode(descent["final_theta"]), thetas[-1])
    # 每一步都滿足 Armijo 充分下降：f(θ⁺) ≤ f(θ) + c·g·(θ⁺−θ)，g 是探針在 θ 重算的梯度。
    gradients = [_decode(item) for item in cast(list[list[str]], row["descent_gradients"])]
    for before, after, value_before, value_after, gradient in zip(thetas, thetas[1:], values, values[1:], gradients, strict=False):
        assert value_after <= value_before + settings.armijo_constant * float(np.dot(gradient, after - before))


def _masked_norm(theta: NDArray[np.float64], gradient: NDArray[np.float64], lower: NDArray[np.float64],
                 upper: NDArray[np.float64]) -> float:
    """只守盒界的投影梯度範數：貼下界還往外推（梯度 > 0）、貼上界還往外推（梯度 < 0）的分量歸零。"""
    blocked = ((theta <= lower) & (gradient > 0)) | ((theta >= upper) & (gradient < 0))
    return float(np.linalg.norm(np.where(blocked, 0.0, gradient)))


def _cone_norm(direction: NDArray[np.float64], normals: Sequence[NDArray[np.float64]]) -> float:
    """方向投到錐 {u：每條法向 n·u ≤ 0} 的長度＝錐裡單位方向 u 上 direction·u 的最大值（不小於 0）。

    線性函數在錐與單位圓交出的弧上，最大值落在不受限的方向或弧的端點（每條作用中限制那條線上的兩個方向），
    逐一試；跟產品「逐條邊投影取最近」是兩種算法。
    """
    trials = [direction / np.linalg.norm(direction)] if np.linalg.norm(direction) > 0.0 else []
    for normal in normals:
        tangent = np.asarray([-normal[1], normal[0]]) / np.linalg.norm(normal)
        trials.extend((tangent, -tangent))
    feasible = [unit for unit in trials if all(float(normal @ unit) <= 1e-12 for normal in normals)]
    return max([0.0, *(float(direction @ unit) for unit in feasible)])


def _on_slant(theta: NDArray[np.float64], t: int, c: int, top: float) -> bool:
    """貼在「厚度＋空腔＝上限」那條斜邊上：和不超過上限、離上限在 4 個上限的浮點間距內（實數意義的貼邊，
    不是「再大一個浮點數放不放得下」——0.1＋0.04999999999999999 再加一個末位仍捨入成 0.15）。"""
    total = theta[t] + theta[c]
    return bool(total <= top and top - total <= 4.0 * np.spacing(top))


def _tangent_norm(theta: NDArray[np.float64], gradient: NDArray[np.float64], lower: NDArray[np.float64],
                  upper: NDArray[np.float64], pairs: Sequence[tuple[int, int]], top: float) -> float:
    """考卷自己算的投影梯度範數：盒界逐格遮；「厚度＋空腔」離上限在幾個浮點間距內的面算貼斜邊，整面負梯度投到
    盒界與斜邊（法向 (1,1)）圍成的錐上再量長度。"""
    masked = np.where(((theta <= lower) & (gradient > 0)) | ((theta >= upper) & (gradient < 0)), 0.0, gradient)
    total = float(masked @ masked)
    for t, c in pairs:
        if not _on_slant(theta, t, c, top):
            continue
        normals = [np.asarray([1.0, 1.0])]
        for axis, index in enumerate((t, c)):
            unit = np.eye(2)[axis]
            normals += [-unit] if theta[index] <= lower[index] else []
            normals += [unit] if theta[index] >= upper[index] else []
        total += _cone_norm(-gradient[[t, c]], normals) ** 2 - masked[t] ** 2 - masked[c] ** 2
    return float(np.sqrt(total))


def test_descent_gradient_norm_is_projected(runs: tuple[Path, dict[str, Record]]) -> None:
    """抓投影梯度的遮罩被拿掉、或貼斜邊往外推的法向分量沒去掉：每一步記的範數要等於考卷重算的投影梯度範數；
    停在「梯度夠小」那一步有一面貼在「厚度＋空腔＝上限」的斜邊上、負梯度往斜邊外推，未遮範數嚴格大於投影後。

    梯度是探針用另一支編譯函式在同一個 θ 重算的，跟下降內部那一支可能差末位，所以不逐位、
    用梯度那把尺（GRADIENT_REL 相對＋GRADIENT_ABS）；遮罩拿掉時差的是整個貼界分量，遠大於這把尺。
    （庚二改：連續段守總深度之後終點不再貼 0.150 的空腔上界，改核斜邊。）
    """
    row = runs[1][DESCENT_CASE]
    descent = _descent(runs)
    lower, upper = _decode(row["lower"]), _decode(row["upper"])
    pairs, top = _pairs(cast(list[str], row["names"])), _settings().family[0].max_total_depth_m
    thetas = [_decode(step) for step in cast(list[list[str]], descent["theta"])]
    gradients = [_decode(item) for item in cast(list[list[str]], row["descent_gradients"])]
    recorded = _decode(descent["gradient_norm"])
    for theta, gradient, norm in zip(thetas, gradients, recorded, strict=True):
        expected = _tangent_norm(theta, gradient, lower, upper, pairs, top)
        assert abs(norm - expected) <= GRADIENT_REL * expected + GRADIENT_ABS, (norm, expected)
    assert descent["stop_reason"] == gradient_search.STOP_GRADIENT
    final_theta, final_gradient = thetas[-1], gradients[-1]
    on_slant_pushed = [(t, c) for t, c in pairs if _on_slant(final_theta, t, c, top)
                       and final_gradient[t] + final_gradient[c] < 0.0]
    assert on_slant_pushed
    projected = _tangent_norm(final_theta, final_gradient, lower, upper, pairs, top)
    assert float(np.linalg.norm(final_gradient)) > projected
    assert _masked_norm(final_theta, final_gradient, lower, upper) > projected
    assert recorded[-1] < _settings().descent.gradient_norm_stop


def test_descent_rerun_is_bitwise_identical(runs: tuple[Path, dict[str, Record]]) -> None:
    """抓下降帶非決定性（亂數、字典序、計時進判斷）：同一起點重跑，θ、值、範數、步長、停止原因逐位相同。"""
    assert _descent(runs) == _descent(runs, "descent_again")


# ---- 合成目標（不走替身，父行程單精度） ----

def _aux(value: jax.Array) -> StageOneObjectiveArrays:
    return StageOneObjectiveArrays(value[None], value[None], value, value, value)


def test_armijo_backtracks_when_first_trial_only_slightly_decreases() -> None:
    """抓拿掉充分下降項（只要值有降就收）：一維二次式 f=a/2·x²，a 取成第一個試步 initial_step 讓值只降一點點、
    落在 Armijo 線之上；正確實作要回溯一次、收 initial_step×backtrack_ratio。

    u=a·α：試步後 f 乘 (1−u)²，Armijo 線是 f·(1−2c·u)；u 取 2−c 時 (1−u)² 小於 1、但大於 1−2c·u。
    """
    settings = _settings().descent
    curvature = (2.0 - settings.armijo_constant) / settings.initial_step

    def objective(theta: jax.Array) -> tuple[jax.Array, StageOneObjectiveArrays]:
        value = 0.5 * curvature * jnp.sum(theta * theta)
        return value, _aux(value)

    start = np.asarray([1.0], dtype=np.float32)
    bounds = (np.asarray([-10.0]), np.asarray([10.0]))
    first_trial = start - np.float32(settings.initial_step) * np.float32(curvature) * start
    assert float(objective(jnp.asarray(first_trial))[0]) < float(objective(jnp.asarray(start))[0])
    result = gradient_search.projected_descent(objective, start, bounds, settings, limits=())
    expected_step = settings.initial_step * settings.backtrack_ratio
    assert result.steps[1].step_size == expected_step
    expected_theta = jnp.clip(jnp.asarray(start) - jnp.asarray(expected_step, dtype=jnp.float32)
                              * jax.grad(lambda x: objective(x)[0])(jnp.asarray(start)), -10.0, 10.0)
    # 值與 θ 是單精度一兩次運算的結果；編譯與未編譯可能差末位，容許幾個單精度間距，不放寬步長那一條。
    expected_value = np.float32(objective(expected_theta)[0])
    assert abs(result.steps[1].value - float(expected_value)) <= 4 * float(np.spacing(expected_value))
    assert np.allclose(result.steps[1].theta, np.asarray(expected_theta), rtol=0.0,
                       atol=4 * float(np.spacing(np.float32(settings.initial_step))))


def test_retune_from_non_descending_start_takes_zero_steps() -> None:
    """抓重調硬湊步數或把失敗的步走掉：起點沿投影方向任何不低於步長下限的步都不滿足 Armijo，
    要 0 步、θ 逐位不變、仍可行、原因是「步長到下限」、最後試的步長低於下限。"""
    settings = _settings()
    surfaces = (("a", "glass_wool_48k_50mm_gap50"), ("b", None), ("c", "glass_wool_48k_50mm_gap100"))
    designs = gradient_search.designs_from_presets(surfaces, settings, _presets())
    start = gradient_search.snap_to_specs(gradient_search.baseline_theta(designs).astype(np.float32), designs).theta
    anchor = jnp.asarray(start)
    # 一次項給每格梯度 1（往空腔變淺、在盒內），二次項的曲率大到任何 α ≥ 步長下限都讓值上升。
    curvature = 1.0 / (settings.descent.min_step * settings.descent.backtrack_ratio ** 4)

    def objective(theta: jax.Array) -> tuple[jax.Array, StageOneObjectiveArrays]:
        shift = theta - anchor
        value = jnp.sum(shift) + curvature * jnp.sum(shift * shift)
        return value, _aux(value)

    result = gradient_search.retune_cavities(objective, start, designs, settings)
    assert result.stop_reason == gradient_search.STOP_STEP
    assert result.steps[-1].step == 0 and result.final_trial_step < settings.descent.min_step
    assert result.theta.tobytes() == np.asarray(start).tobytes()
    assert gradient_search.is_feasible(result.theta, designs)


# ---- 取規格（不碰替身，父行程直接跑） ----

def _theta_for(designs: tuple[gradient_search.SurfaceDesign, ...], values: dict[str, float]) -> NDArray[np.float64]:
    base = np.asarray(gradient_search.baseline_theta(designs), dtype=np.float64)
    for index, name in enumerate(gradient_search.design_names(designs)):
        if name in values:
            base[index] = values[name]
    return base


def test_snap_nearest_thickness() -> None:
    """抓取最近取錯方向：0.071 → 0.050、0.080 → 0.100；空腔維持連續、沒被擋的面不記被擋。"""
    designs = _designs(DESCENT_CASE)
    for thickness, expected in ((0.071, 0.050), (0.080, 0.100)):
        theta = _theta_for(designs, {"floor.layer1.thickness_m": thickness, "floor.layer2.thickness_m": 0.033})
        snapped = gradient_search.snap_to_specs(theta, designs)
        floor = snapped.surfaces[0]
        assert floor.thickness_m == expected and floor.nearest_thickness_m == expected
        assert floor.cavity_m == 0.033 and not floor.blocked_by_total_depth
        assert gradient_search.is_feasible(snapped.theta, designs)


def test_snap_total_depth_clips_cavity_where_independent_rounding_violates() -> None:
    """抓各面各自取最近就交付：厚度 0.09 取最近是 0.100，配空腔 0.080 超過上限 0.150；取規格要切空腔並記下被擋。"""
    designs = _designs(DESCENT_CASE)
    theta = _theta_for(designs, {"floor.layer1.thickness_m": 0.09, "floor.layer2.thickness_m": 0.080})
    independent = theta.copy()
    independent[0] = 0.100
    assert not gradient_search.is_feasible(independent, designs)
    snapped = gradient_search.snap_to_specs(theta, designs)
    assert gradient_search.is_feasible(snapped.theta, designs)
    floor = snapped.surfaces[0]
    assert floor.thickness_m == 0.100 and floor.cavity_clipped and not floor.thickness_retreated
    assert floor.cavity_m is not None and floor.cavity_m + 0.100 <= 0.150
    assert [choice.surface_id for choice in snapped.surfaces if choice.blocked_by_total_depth] == ["floor"]


def test_snap_retreats_to_thinner_when_cavity_zero_still_exceeds() -> None:
    """抓只切空腔不退厚度：總深度上限收到 0.090，厚度 0.100 連空腔 0 都放不下，要退到 0.050（貼牆面一樣退）。"""
    base = _settings()
    family = base.family[0].model_copy(update={"max_total_depth_m": 0.090})
    settings = base.model_copy(update={"family": (family, *base.family[1:])})
    presets = ("glass_wool_48k_50mm_gap50", None, None, "glass_wool_48k_100mm", None, None)
    designs = gradient_search.wall_designs_from_presets(presets, settings, _presets())
    theta = _theta_for(designs, {"floor.layer1.thickness_m": 0.095, "floor.layer2.thickness_m": 0.060,
                                 "xL.layer1.thickness_m": 0.100})
    snapped = gradient_search.snap_to_specs(theta, designs)
    assert gradient_search.is_feasible(snapped.theta, designs)
    floor, x_l = snapped.surfaces[0], snapped.surfaces[3]
    assert floor.nearest_thickness_m == 0.100 and floor.thickness_m == 0.050 and floor.thickness_retreated
    assert floor.cavity_m is not None and floor.cavity_m + 0.050 <= 0.090 and floor.cavity_clipped
    assert x_l.thickness_m == 0.050 and x_l.thickness_retreated and x_l.cavity_m is None


def test_descent_end_snaps_to_feasible(runs: tuple[Path, dict[str, Record]]) -> None:
    """子行程那一次的取規格結果：可行、厚度都是合法值。"""
    snap = cast(Record, runs[1][DESCENT_CASE]["snap"])
    assert gradient_search.is_feasible(_decode(snap["theta"]), _designs(DESCENT_CASE))


# ---- 重調 ----

def test_retune_moves_only_cavities_at_most_k_steps(runs: tuple[Path, dict[str, Record]]) -> None:
    """抓重調動到厚度、步數超過上限、值上升、走出總深度：厚度逐位不變、最多 K 步、原因與數字一致、值不升、每步可行。"""
    row = runs[1][DESCENT_CASE]
    retune = cast(Record, row["retune"])
    designs = _designs(DESCENT_CASE)
    start = _decode(cast(Record, row["snap"])["theta"])
    thetas = [_decode(step) for step in cast(list[list[str]], retune["theta"])]
    names = gradient_search.design_names(designs)
    cavities = np.asarray([name.endswith("layer2.thickness_m") for name in names])
    assert np.array_equal(thetas[0], start)
    taken, reason = len(thetas) - 1, retune["stop_reason"]
    assert taken <= _settings().retune.steps and reason in gradient_search.STOP_REASONS
    assert reason != gradient_search.STOP_GRADIENT
    if reason == gradient_search.STOP_MAX_STEPS:
        assert taken == _settings().retune.steps
    else:
        assert float.fromhex(cast(str, retune["final_trial_step"])) < _settings().descent.min_step
    assert all(np.array_equal(theta[~cavities], start[~cavities]) for theta in thetas)
    assert all(gradient_search.is_feasible(theta, designs) for theta in thetas)
    assert np.all(np.diff(_decode(retune["value"])) <= 0.0)
    assert not np.array_equal(thetas[-1][cavities], start[cavities])


# ---- 帳本 ----

def test_ledger_round_trip_matches_probe(chain_model: ReducedModel, runs: tuple[Path, dict[str, Record]]) -> None:
    """抓帳本漏寫或改寫：子行程寫的帳本讀回，θ、值逐位等於那一次的結果，標頭等於父行程照同一份輸入算的。"""
    directory, output = runs
    designs = _designs(DESCENT_CASE)
    expected = ledger.header_for(chain_model, designs, _settings())
    header, rows = ledger.read_ledger(directory / f"{DESCENT_CASE}.ledger.jsonl", expected_header=expected)
    row = output[DESCENT_CASE]
    descent, retune = cast(Record, row["descent"]), cast(Record, row["retune"])
    expected_thetas = [*cast(list[list[str]], descent["theta"]), _hex_list(cast(Record, row["snap"])["theta"]),
                       *cast(list[list[str]], retune["theta"])]
    assert [list(item.theta_hex) for item in rows] == expected_thetas
    expected_values = [*_decode(descent["value"]), float.fromhex(cast(str, cast(Record, row["snap"])["value"])),
                       *_decode(retune["value"])]
    assert [item.value for item in rows] == expected_values
    assert [item.phase for item in rows] == (["descent"] * len(cast(list[str], descent["value"])) + ["snap"]
                                             + ["retune"] * len(cast(list[str], retune["value"])))
    assert [item.stop_reason for item in rows if item.stop_reason is not None] == [descent["stop_reason"], retune["stop_reason"]]
    assert [item.final_trial_step for item in rows if item.final_trial_step is not None] == [
        float.fromhex(cast(str, descent["final_trial_step"])), float.fromhex(cast(str, retune["final_trial_step"]))]
    assert header.variable_names == gradient_search.design_names(designs)


def _hex_list(values: object) -> list[str]:
    return list(cast(list[str], values))


def _tampered(source: Path, target: Path, edit: str) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    if edit == "theta_length":
        document = json.loads(lines[2])
        document["theta_hex"] = document["theta_hex"][:-1]
        lines[2] = json.dumps(document, ensure_ascii=False) + "\n"
    elif edit == "swap":
        lines[1], lines[2] = lines[2], lines[1]
    elif edit == "stop_reason_moved":
        document = json.loads(lines[1])
        document["stop_reason"] = gradient_search.STOP_MAX_STEPS
        document["final_trial_step"] = 0.0
        document["evaluations"] = 1
        lines[1] = json.dumps(document, ensure_ascii=False) + "\n"
    target.write_text("".join(lines), encoding="utf-8")
    return target


@pytest.mark.parametrize("edit", ("theta_length", "swap", "stop_reason_moved"))
def test_ledger_rejects_tampered_rows(runs: tuple[Path, dict[str, Record]], tmp_path: Path, edit: str) -> None:
    """抓讀回不驗：改一列的 θ 長度、交換兩列順序、停止原因放到中間列，都拒收。"""
    source = runs[0] / f"{DESCENT_CASE}.ledger.jsonl"
    with pytest.raises(ValueError):
        ledger.read_ledger(_tampered(source, tmp_path / "tampered.jsonl", edit))


def test_ledger_rejects_header_with_other_surrogate(chain_model: ReducedModel, runs: tuple[Path, dict[str, Record]],
                                                    tmp_path: Path) -> None:
    """抓拿別的替身的帳本接續：預期標頭的替身雜湊不同就拒收；已有檔案不覆寫。"""
    source = runs[0] / f"{DESCENT_CASE}.ledger.jsonl"
    other = dataclasses.replace(chain_model, input_hash="0" * 64)
    with pytest.raises(ValueError, match="input_hash"):
        ledger.read_ledger(source, expected_header=ledger.header_for(other, _designs(DESCENT_CASE), _settings()))
    header, rows = ledger.read_ledger(source)
    with pytest.raises(FileExistsError):
        ledger.write_ledger(source, header, rows)
    copy = tmp_path / "copy.jsonl"
    ledger.write_ledger(copy, header, rows)
    assert copy.read_bytes() == source.read_bytes()


# ---- 單精度：只量不登記 ----

@pytest.mark.parametrize("name", CASES)
def test_single_precision_stays_float32_and_finite(runs: tuple[Path, dict[str, Record]], name: str) -> None:
    """抓單精度被悄悄提升或算出非有限：值與梯度都是 float32、有限；下降結果的 θ 也是 float32。"""
    single = cast(Record, runs[1][name]["single"])
    assert single["dtype"] == "float32" and single["gradient_dtype"] == "float32"
    assert np.isfinite(float.fromhex(cast(str, single["value"]))) and np.isfinite(_decode(single["gradient"])).all()
    if name == DESCENT_CASE:
        descent = cast(Record, runs[1][name]["single_descent"])
        assert descent["final_dtype"] == "float32" and np.isfinite(_decode(descent["value"])).all()


# ---- 任意面數（不碰替身） ----

@pytest.mark.parametrize("surfaces", (
    (("x0.zone1", "glass_wool_48k_50mm_gap50"), ("x0.zone2", None), ("y0.zone1", "glass_wool_24k_50mm")),
    (("a", "glass_wool_48k_50mm_gap50"), ("b", "glass_wool_48k_100mm"), ("c", None), ("d", "glass_wool_96k_50mm"),
     ("e", "glass_wool_48k_50mm_gap100"), ("f", "glass_wool_48k_50mm"), ("g", "glass_wool_48k_50mm_gap50")),
))
def test_flatten_and_snap_work_for_any_surface_count(surfaces: tuple[tuple[str, str | None], ...]) -> None:
    """抓扁平化或取規格把面數寫死成六：3 面與 7 面的設計照樣列名、切參數、取規格、算重調盒界。"""
    designs = gradient_search.designs_from_presets(surfaces, _settings(), _presets())
    names = gradient_search.design_names(designs)
    expected_names = [f"{design.surface_id}.{name}" for design in designs for name in design.variables]
    assert list(names) == expected_names
    lower, upper = gradient_search.design_bounds(designs)
    # 不取中點：48K 厚度與空腔的中點都是 0.075，接錯格也看不出來。
    theta = lower + 0.3 * (upper - lower)
    single = jnp.asarray(theta, dtype=jnp.float32)
    parameters = gradient_search.surface_parameters(designs, single)
    assert [list(item) for item in parameters] == [list(design.variables) for design in designs]
    flattened = [float(item[name]) for design, item in zip(designs, parameters, strict=True) for name in design.variables]
    assert flattened == [float(value) for value in np.asarray(single)]
    snapped = gradient_search.snap_to_specs(theta, designs)
    assert [choice.surface_id for choice in snapped.surfaces] == [surface_id for surface_id, _ in surfaces]
    assert gradient_search.is_feasible(snapped.theta, designs)
    retune_lower, retune_upper = gradient_search.retune_bounds(snapped.theta, designs)
    assert np.all(retune_lower <= snapped.theta) and np.all(snapped.theta <= retune_upper)


def test_chain_rejects_surface_count_other_than_surrogate_walls(chain_model: ReducedModel) -> None:
    """抓替身那一層靜默接受非六面：三個面在建樣板之前就拒收，訊息講明是替身的限制。"""
    designs = gradient_search.designs_from_presets(
        (("x0.zone1", "glass_wool_48k_50mm_gap50"), ("x0.zone2", None), ("y0", "glass_wool_24k_50mm")),
        _settings(), _presets())
    constants = objective_constants(chain_model.frequencies_hz, _score(), _receivers(), len(chain_model.source_names))
    with pytest.raises(ValueError, match="替身"):
        gradient_search.chain_objective(
            chain_model, designs, constants, air=load_physics_constants(config_path("physics_constants.toml")),
            ranges=load_porous_model_ranges(config_path("porous_model_ranges.toml")),
            frequencies=chain_model.frequencies_hz)


# ---- 拒收 ----

def _never_called(theta: jax.Array) -> tuple[jax.Array, StageOneObjectiveArrays]:
    raise AssertionError("起點超界要在求值之前就拒收")


def test_rejections() -> None:
    """抓六面不是六個、preset 不認識、不在家族、設計變數名樣板沒有、θ 長度不對、θ 超界、非浮點照樣收。"""
    settings, presets = _settings(), _presets()
    with pytest.raises(ValueError, match="六面"):
        gradient_search.wall_designs_from_presets(CASES[DESCENT_CASE][:-1], settings, presets)
    with pytest.raises(ValueError, match="不認識"):
        gradient_search.wall_designs_from_presets(("no_such_preset", *CASES[DESCENT_CASE][1:]), settings, presets)
    with pytest.raises(ValueError, match="家族"):
        gradient_search.wall_designs_from_presets(("fabric_pet_a_gap100", *CASES[DESCENT_CASE][1:]), settings, presets)
    designs = _designs(DESCENT_CASE)
    floor = designs[0]
    with pytest.raises(ValueError, match="樣板沒有"):
        dataclasses.replace(floor, variables=("layer1.thickness_m", "layer3.thickness_m"))
    theta = gradient_search.baseline_theta(designs)
    with pytest.raises(ValueError, match="長度"):
        gradient_search.checked_theta(theta[:-1], designs)
    outside = theta.copy()
    outside[1] = 0.2
    with pytest.raises(ValueError, match="超出盒界"):
        gradient_search.snap_to_specs(outside, designs)
    with pytest.raises(TypeError, match="浮點"):
        gradient_search.checked_theta(np.ones(theta.shape, dtype=np.int64), designs)
    with pytest.raises(ValueError, match="有限"):
        gradient_search.checked_theta(np.full(theta.shape, np.nan), designs)
    with pytest.raises(ValueError, match="盒界"):
        gradient_search.projected_descent(_never_called, outside, gradient_search.design_bounds(designs), settings.descent,
                                          limits=gradient_search.depth_limits(designs))


def test_surface_design_thickness_bounds_must_span_legal_thicknesses() -> None:
    """抓厚度界跟合法厚度脫鉤：下界不是最薄、上界不是最厚，各一組，都要拒收。"""
    floor = _designs(DESCENT_CASE)[0]
    assert floor.legal_thicknesses_m[0] < floor.legal_thicknesses_m[-1]
    for lower, upper in (((floor.legal_thicknesses_m[0] / 2, *floor.lower[1:]), floor.upper),
                         (floor.lower, (floor.legal_thicknesses_m[-1] * 1.5, *floor.upper[1:]))):
        with pytest.raises(ValueError, match="合法厚度"):
            dataclasses.replace(floor, lower=lower, upper=upper)


def test_settings_file_loads_and_guards(tmp_path: Path) -> None:
    """抓設定檔缺節、步長下限不小於起始、最薄厚度超過總深度、一個 preset 屬兩個家族、數字型別寬鬆轉型照樣收。"""
    settings = _settings()
    assert settings.fingerprint == load_material_gradient_settings(config_path("material_gradient.toml")).fingerprint
    missing = tmp_path / "missing.toml"
    missing.write_text("[descent]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_material_gradient_settings(missing)
    with pytest.raises(ValueError):
        DescentSettings.model_validate({**settings.descent.model_dump(), "min_step": settings.descent.initial_step})
    with pytest.raises(ValueError):
        FamilySettings.model_validate({**settings.family[0].model_dump(), "max_total_depth_m": 0.01})
    family = settings.family[0].model_dump()
    with pytest.raises(ValueError, match="浮點"):
        FamilySettings.model_validate({**family, "thicknesses_m": ["0.05", 0.1]})
    with pytest.raises(ValueError, match="浮點"):
        FamilySettings.model_validate({**family, "max_total_depth_m": 1})
    with pytest.raises(ValueError, match="浮點"):
        DescentSettings.model_validate({**settings.descent.model_dump(), "initial_step": 1})
    with pytest.raises(ValueError):
        DescentSettings.model_validate({**settings.descent.model_dump(), "max_steps": 30.0})
    with pytest.raises(ValueError):
        MaterialGradientSettings.model_validate({**settings.model_dump(), "family": [
            settings.family[0].model_dump(), {**settings.family[1].model_dump(), "presets": list(settings.family[0].presets)}]})
