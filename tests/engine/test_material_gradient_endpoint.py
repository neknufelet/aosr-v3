"""梯度搜尋終點（#561 辛二）：空腔匯出規則、空腔 0 的替身鏈對「拿掉空氣層」、完整模型重算對替身、
完整解路徑逐位、大峰走第一階段評分程式、方案牆面來回。

替身用 ``test_material_gradient`` 的衍生替身（小房間 fixture、七個座位列是三個真座位列的線性組合、
頻率軸是整條正式有限元素軸）；完整模型的座位列用同一組線性組合從三個真座位列做出來（投影是線性的，
完整解乘組合列就是替身那幾列對應的完整模型值）。材料求阻抗與替身鏈要雙精度 JAX，所以在子行程跑、
只交原始數值；比對與判決在父行程。尺：替身對完整模型用精度契約登記簿的留出尺；阻抗用傳遞矩陣考卷那一份。
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.geometry.shoebox import Point, Wall
from aosr.gradient import endpoint
from aosr.gradient import material_gradient as gradient_search
from aosr.materials.construction_models import AirLayer, Assembly, PorousLayer
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_check import seats_within_contract
from aosr.physics.fem_reduced_store import save_reduced
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.receiver_set import ReceiverRole
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import (
    StageOneLowFrequencyInput, StageOneSeatCurve, StageOneSpeakerCurves,
)
from aosr.scoring.stage_one_objective_jax import objective_constants
from tests.engine._precision_contracts import contract_value
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine.test_material_gradient import CASES, PURPOSE, _chain_model, _presets, _receivers, _score, _settings
from tests.engine.test_scheme_model import _document
from tests.engine.test_transfer_matrix_reference import IMPEDANCE_ABS, REFERENCE_REL
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS


PROBE = Path(__file__).with_name("_material_gradient_endpoint_probe.py")
AXIS_ID = LowFrequencyAxis.SEARCH
FREQUENCY_RULER = contract_value("fem_reduced_holdout_frequency_db")
Record = dict[str, object]
# 終點那一組：兩面剛性；第一面 48K 棉板 100 mm＋空腔 0.5 mm（小於 1 mm，匯出拿掉空氣層），第三面 50 mm＋40 mm 空腔。
ENDPOINT_PRESETS = CASES["two_rigid"]
ENDPOINT_THETA = (0.1, 0.0005, 0.05, 0.04, 0.05, 0.05)
# 考卷自己寫的匯出 θ：那一格 0.5 mm 換成 0，其餘照抄。
ENDPOINT_EXPORT = (0.1, 0.0, 0.05, 0.04, 0.05, 0.05)
# 空腔 0 的鏈層級：第一面同一層 48K 棉板，一份帶空氣層（空腔 0）、一份樣板本來就沒有空氣層，其餘五面相同。
ZERO_WITH_GAP = ("glass_wool_48k_50mm_gap50", *CASES["two_rigid"][1:])
ZERO_WITHOUT_GAP = ("glass_wool_48k_50mm", *CASES["two_rigid"][1:])
ZERO_THETA_WITH_GAP = (0.1, 0.0, 0.05, 0.04, 0.05, 0.05)
ZERO_THETA_WITHOUT_GAP = (0.1, 0.05, 0.04, 0.05, 0.05)


def _combined(front: NDArray[np.float64], back: NDArray[np.float64],
              side: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """跟 ``test_material_gradient._chain_model`` 同一組七個組合、同一個算式寫法（逐位相同由考卷核）。"""
    return [front, back, side, (front + back) / 2, (front + side) / 2, (back + side) / 2, (front + back + side) / 3]


def chain_operators(full: FullOperators) -> FullOperators:
    """完整模型的座位列換成三個真座位列的七個組合，名字照座位組；網格、喇叭、空氣不動。

    組合列不是物理位置；``receiver_points`` 只進建模的輸入雜湊（這裡不建模），照座位組自己的座標填。
    """
    front, back, side = full.receivers.values()
    points = _receivers().points
    rows = _combined(front, back, side)
    return dataclasses.replace(
        full, receivers={point.receiver_id: row for point, row in zip(points, rows, strict=True)},
        receiver_points={point.receiver_id: Point(*point.position_m) for point in points})


def _decode(values: object) -> NDArray[np.float64]:
    return np.asarray([float.fromhex(value) for value in cast(list[str], values)], dtype=np.float64)


def _decode_complex(node: object) -> NDArray[np.complex128]:
    row = cast(Record, node)
    return (_decode(row["real"]) + 1j * _decode(row["imag"])).reshape(cast(list[int], row["shape"]))


def _hex_list(values: tuple[float, ...]) -> list[str]:
    return [float(value).hex() for value in values]


@pytest.fixture(scope="module")
def chain_model(model: ReducedModel) -> ReducedModel:
    return _chain_model(model)


@pytest.fixture(scope="module")
def probe(chain_model: ReducedModel, tmp_path_factory: pytest.TempPathFactory) -> Record:
    directory = tmp_path_factory.mktemp("material-gradient-endpoint")
    save_reduced(chain_model, directory / "model.npz")
    problem = {
        "purpose": PURPOSE, "axis": AXIS_ID.value, "receivers": _receivers().model_dump(mode="json"),
        "endpoint": {"presets": list(ENDPOINT_PRESETS), "theta": _hex_list(ENDPOINT_THETA),
                     "expected_export": _hex_list(ENDPOINT_EXPORT)},
        "zero_cavity": {"with_gap": list(ZERO_WITH_GAP), "without_gap": list(ZERO_WITHOUT_GAP),
                        "theta_with_gap": _hex_list(ZERO_THETA_WITH_GAP),
                        "theta_without_gap": _hex_list(ZERO_THETA_WITHOUT_GAP)},
    }
    (directory / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    process = subprocess.run([sys.executable, str(PROBE), str(directory)], cwd=directory, env=env,
                             capture_output=True, text=True, timeout=900)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(Record, json.loads((directory / "output.json").read_text(encoding="utf-8")))


def _report(probe: Record) -> endpoint.EndpointReport:
    return endpoint.EndpointReport.model_validate(probe["report"])


# ---- 空腔匯出規則 ----

def _one_face(cavity: float) -> list[tuple[str, float]]:
    """第一面 48K 棉板 50 mm＋給定空腔，其餘五面剛性；回第一面匯出構造的（層種類, 厚度）清單，房間側起。"""
    walls = [wall.wall_name() for wall in Wall.all()]
    specs = [endpoint.SurfaceSpec(walls[0], "glass_wool_48k_50mm_gap50", 0.05, cavity),
             *(endpoint.SurfaceSpec(name, None, None, None) for name in walls[1:])]
    materials = endpoint.materials_from_surfaces(specs, _presets(), _settings())
    material = materials[Wall.all()[0]]
    assert material is not None and material.backing == "rigid"
    assert all(materials[wall] is None for wall in Wall.all()[1:])
    layers: list[tuple[str, float]] = []
    for layer in material.layers:
        assert isinstance(layer, PorousLayer | AirLayer), layer
        layers.append((layer.kind, layer.thickness_m))
    return layers


def test_zero_cavity_exports_no_air_layer() -> None:
    """抓空腔 0 仍放空氣層（構造格式的空氣層厚度必須大於 0，放了就是壞構造）：只剩棉板一層，厚度是規格厚度。"""
    assert _one_face(0.0) == [("porous", 0.05)]


def test_cavity_of_one_step_keeps_air_layer() -> None:
    """抓門檻往上偏（等於一格也被拿掉）：空腔恰好等於最小一格（1 mm）時放 ``AirLayer``、厚度原值。"""
    assert _one_face(endpoint.MIN_CAVITY_M) == [("porous", 0.05), ("air", endpoint.MIN_CAVITY_M)]


def test_cavity_below_one_step_drops_air_layer() -> None:
    """抓只拿掉恰好 0、不管小於一格的正空腔：0.5 mm 不放空氣層；而 40 mm 照放原值。"""
    assert _one_face(endpoint.MIN_CAVITY_M / 2) == [("porous", 0.05)]
    assert _one_face(0.04) == [("porous", 0.05), ("air", 0.04)]


# ---- 空腔 0：替身鏈對「拿掉空氣層」 ----

def test_zero_cavity_chain_matches_material_without_air(probe: Record) -> None:
    """抓傳遞矩陣在空氣層厚度 0 時不穩（查證員說沒驗過）：替身樣板在空腔 0 的阻抗，對匯出構造（沒有空氣層）走產品
    ``surface_impedance`` 的阻抗，逐頻實虛部都在傳遞矩陣考卷那把阻抗尺內；整條鏈的合分，對「樣板本來就沒有空氣層」
    那一份同厚度的鏈，相對差在目標 JAX 版那把尺內。"""
    zero = cast(Record, probe["zero_cavity"])
    assert zero["exported_layers"] == ["porous"]
    chain, exported = _decode_complex(zero["chain_impedance"]), _decode_complex(zero["exported_impedance"])
    assert np.all(np.isfinite(chain)) and chain.shape == exported.shape
    for part in (np.real, np.imag):
        np.testing.assert_allclose(part(chain), part(exported), rtol=REFERENCE_REL, atol=IMPEDANCE_ABS)
    with_gap = float.fromhex(cast(str, zero["chain_value_with_gap"]))
    without_gap = float.fromhex(cast(str, zero["chain_value_without_gap"]))
    assert math.isfinite(with_gap)
    assert abs(with_gap - without_gap) <= contract_value("stage_one_objective_jax_vs_python") * abs(without_gap)


# ---- 終點重算：完整模型對替身 ----

def test_full_model_pressures_bitwise_match_validate_reduced(probe: Record) -> None:
    """抓完整解改走替身或換了別條路：``full_model_pressures`` 對 ``validate_reduced`` 存的完整解（同一組材料、同一條軸）
    逐位相同；終點報告交回的完整聲壓也逐位相同。完整模型的網格雜湊等於替身的（同一間房、同一個網格）。"""
    assert probe["mesh_hash_equal"] is True
    direct, reference = _decode_complex(probe["direct"]), _decode_complex(probe["reference"])
    np.testing.assert_array_equal(direct, reference)
    np.testing.assert_array_equal(_decode_complex(probe["full"]), reference)
    assert not np.array_equal(_decode_complex(probe["reduced"]), reference)


def test_endpoint_full_value_within_holdout_ruler(probe: Record, chain_model: ReducedModel) -> None:
    """抓終點把替身值當完整模型值、或差與上界接錯：完整模型合分等於獨立完整解交同一把尺的值（逐位）；
    每座位逐頻最大 dB 差在留出尺內、等於考卷從兩邊聲壓重算的；合分之差不超過由逐頻差推得的上界。

    上界推導（``score_difference_bound``）：dB 差每格不超過 ε 時，主項（扣加權直線＝同一內積下的正交投影，
    範數不增）之差 ≤ ε、座位穩健性之差 ≤ 2ε，合分之差 ≤ ε·(w_主/ref_主 + 2·w_穩/ref_穩)。替身值走 JAX 鏈、
    逐頻差用 numpy 替身聲壓，兩者的差由目標 JAX 版那把尺補上。
    """
    report = _report(probe)
    full, reduced = _decode_complex(probe["full"]), _decode_complex(probe["reduced"])
    assert report.full_total == float.fromhex(cast(str, probe["direct_total"]))
    assert report.total_difference == report.full_total - report.reduced_total
    level = np.abs(20.0 * np.log10(np.abs(reduced) / np.abs(full)))
    by_seat = np.max(level, axis=(0, 1))
    assert report.frequency_db_by_seat == {
        name: float(value) for name, value in zip(chain_model.receiver_names, by_seat, strict=True)}
    assert report.max_frequency_db == float(np.max(level))
    assert all(value <= FREQUENCY_RULER for value in report.frequency_db_by_seat.values())
    assert seats_within_contract(report.frequency_db_by_seat, report.band_db_by_seat, frequency_db=FREQUENCY_RULER,
                                 band_db=contract_value("fem_reduced_holdout_band_db"))
    constants = objective_constants(chain_model.frequencies_hz, _score(), _receivers(), len(chain_model.source_names))
    weights, references = constants.component_weights, constants.worse_references
    expected_bound = report.max_frequency_db * (weights[0] / references[0] + 2.0 * weights[1] / references[1])
    assert report.total_difference_bound == expected_bound
    slack = contract_value("stage_one_objective_jax_vs_python") * abs(report.full_total)
    assert abs(report.total_difference) <= report.total_difference_bound + slack
    assert report.total_difference_bound <= endpoint.score_difference_bound(constants, FREQUENCY_RULER)
    assert report.full_solves == len(chain_model.frequencies_hz)


def test_endpoint_reports_dropped_cavity_and_exported_theta(probe: Record) -> None:
    """抓替身值在原 θ 算（含 0.5 mm 空腔）而完整模型在匯出材料算：被拿掉空氣層的那一面列在報告裡，
    匯出 θ 那一格是 0、其餘格逐位等於搜尋 θ；報告的規格是搜尋原值；替身值等於同一條鏈在考卷自己寫的匯出 θ 的值，
    不是在搜尋 θ 的值（兩者不同，這一題才分得出來）。"""
    report = _report(probe)
    at_export = float.fromhex(cast(str, probe["chain_at_expected_export"]))
    assert report.exported_theta == ENDPOINT_EXPORT and report.reduced_total == at_export
    assert at_export != float.fromhex(cast(str, probe["chain_at_search"]))
    designs = gradient_search.wall_designs_from_presets(ENDPOINT_PRESETS, _settings(), _presets())
    names = gradient_search.design_names(designs)
    search = dict(zip(names, report.search_theta, strict=True))
    exported = dict(zip(names, report.exported_theta, strict=True))
    first = Wall.all()[0].wall_name()
    assert report.dropped_cavities == (first,)
    for name in names:
        if name.startswith(f"{first}.layer2"):
            assert search[name] > 0.0 and exported[name] == 0.0
        else:
            assert exported[name] == search[name]
    assert [spec.surface_id for spec in report.surfaces] == [wall.wall_name() for wall in Wall.all()]


# ---- 大峰 ----

def _control_peaks(pressures: NDArray[np.complex128], model: ReducedModel) -> dict[str, float | None]:
    """控制組：考卷自己組第一階段輸入、交既有評分程式，讀每支喇叭最強峰的深度。"""
    energies = np.abs(pressures) ** 2
    points = _receivers().points
    primary = next(point.receiver_id for point in points if point.role == ReceiverRole.PRIMARY)
    speakers = []
    for s, speaker_id in enumerate(model.source_names):
        seats = {point.receiver_id: StageOneSeatCurve(
            receiver_id=point.receiver_id, direction_relative_to_primary=point.direction_relative_to_primary,
            report_id=f"control-{point.receiver_id}", frequencies_hz=model.frequencies_hz,
            fem_energy=tuple(float(value) for value in energies[:, s, r])) for r, point in enumerate(points)}
        speakers.append(StageOneSpeakerCurves(speaker_id=speaker_id, primary=seats[primary],
                                              surrounding=tuple(curve for key, curve in seats.items() if key != primary)))
    data = StageOneLowFrequencyInput(candidate_id="control", low_frequency_axis=AXIS_ID, speakers=tuple(speakers))
    evaluation = evaluate_stage_one_low_frequency(data, scoring_range_hz=_score().scored_range_hz)
    peaks: dict[str, float | None] = {}
    for speaker in evaluation.speakers:
        payload = speaker.payload
        assert payload is not None and payload.strongest_peak_index is not None
        peaks[speaker.speaker_id] = payload.features[payload.strongest_peak_index].depth_db
    return peaks


def test_endpoint_big_peak_matches_stage_one_evaluator(probe: Record, chain_model: ReducedModel) -> None:
    """抓大峰量錯（量替身聲壓、取較小支、重寫峰值演算法）：每支喇叭的大峰等於既有評分程式對完整模型聲壓量的，
    整體大峰取兩支較大；對替身聲壓量的不算數。"""
    report = _report(probe)
    control = _control_peaks(_decode_complex(probe["full"]), chain_model)
    assert report.big_peak_db_by_speaker == control
    values = [value for value in control.values() if value is not None]
    assert len(set(values)) > 1, "兩支喇叭的大峰相同，取較大支的植錯驗不出來"
    assert report.big_peak_db == max(values)
    standalone = endpoint.big_peaks(
        _decode_complex(probe["full"]), speaker_ids=chain_model.source_names, receiver_set=_receivers(),
        frequencies_hz=chain_model.frequencies_hz, low_frequency_axis=AXIS_ID, scoring_range_hz=_score().scored_range_hz)
    assert standalone == control
    assert endpoint.worst_big_peak({**control, chain_model.source_names[0]: None}) is None


# ---- 方案牆面 ----

def _lined_materials() -> dict[Wall, Assembly | None]:
    designs = gradient_search.wall_designs_from_presets(CASES["all_lined"], _settings(), _presets())
    theta = gradient_search.snap_to_specs(gradient_search.baseline_theta(designs), designs).theta
    return endpoint.materials_from_surfaces(endpoint.surface_specs(designs, theta), _presets(), _settings())


def test_scene_with_materials_round_trip() -> None:
    """抓寫錯牆、丟了層或留著舊的固定阻抗欄：寫進方案、存成 JSON、``Scheme.model_validate`` 讀回，
    六面每一面的構造等於寫進去的；舊欄拿掉；房間與空氣不動。"""
    scheme = Scheme.model_validate(_document())
    assert scheme.scene.impedance_pa_s_per_m_by_wall is not None
    materials = _lined_materials()
    scene = endpoint.scene_with_materials(scheme.scene, materials)
    document = scheme.model_dump(mode="json")
    document["scene"] = scene.model_dump(mode="json")
    loaded = Scheme.model_validate(json.loads(json.dumps(document)))
    zones = loaded.scene.surface_zones_by_wall
    assert zones is not None and loaded.scene.impedance_pa_s_per_m_by_wall is None
    assert {name: zone[0].material for name, zone in zones.items()} == {
        wall.wall_name(): materials[wall] for wall in Wall.all()}
    assert all(zone[0].area == "whole_wall" for zone in zones.values())
    original = scheme.scene
    assert (loaded.scene.room_m, loaded.scene.sound_speed_m_s, loaded.scene.density_kg_m3) == (
        original.room_m, original.sound_speed_m_s, original.density_kg_m3)


def test_scene_with_materials_rejects_rigid_wall() -> None:
    """抓剛性面被靜默寫成別的材料（方案格式沒有剛性的寫法）：任一面是 None 就拒收、訊息點名那一面。"""
    scheme = Scheme.model_validate(_document())
    materials = _lined_materials()
    rigid = Wall.all()[-1]
    with pytest.raises(ValueError, match=rigid.wall_name()):
        endpoint.scene_with_materials(scheme.scene, {**materials, rigid: None})


def test_chain_operators_rows_match_chain_model(model: ReducedModel, chain_model: ReducedModel) -> None:
    """抓考卷自己的組合跟替身那邊不同：同一個組合函式套在替身的三個座位列上，逐位等於衍生替身的七列。"""
    front, back, side = model.matrices.receivers
    np.testing.assert_array_equal(np.asarray(_combined(front, back, side)), chain_model.matrices.receivers)
