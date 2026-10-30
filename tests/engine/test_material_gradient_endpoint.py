"""梯度搜尋終點（#561 辛二）：空腔匯出規則、空腔 0 的替身鏈對「拿掉空氣層」、完整模型重算對替身、
完整解路徑逐位、大峰走第一階段評分程式、方案牆面來回；修補一加：門檻讀設定檔、材料只從設計來、面序與空氣核對、
分數差上界含目標的均方根地板。

替身用 ``test_material_gradient`` 的衍生替身（小房間 fixture、七個座位列是三個真座位列的線性組合、
頻率軸是整條正式有限元素軸）；完整模型的座位列用同一組線性組合從三個真座位列做出來（投影是線性的，
完整解乘組合列就是替身那幾列對應的完整模型值）。材料求阻抗與替身鏈要雙精度 JAX，所以在子行程跑、
只交原始數值；比對與判決在父行程。尺：替身對完整模型用精度契約登記簿的留出尺；阻抗用傳遞矩陣考卷那一份。
"""
from __future__ import annotations

import dataclasses
import hashlib
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
from pydantic import TypeAdapter

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.material_gradient import ExportSettings, MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.geometry.shoebox import Point, Wall
from aosr.gradient import endpoint
from aosr.gradient import material_gradient as gradient_search
from aosr.materials.construction_models import AirLayer, Assembly, ConstantImpedance, PorousLayer, SurfaceMaterial
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_check import seats_within_contract
from aosr.physics.fem_reduced_store import save_reduced
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import (
    StageOneLowFrequencyInput, StageOneSeatCurve, StageOneSpeakerCurves,
)
from aosr.scoring.gradient_objective_jax import gradient_objective_constants, proxy_gap_bound_db
from aosr.scoring.stage_one_objective_jax import objective_constants, rms_floor_factor
from tests.engine._precision_contracts import contract_value
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine.test_material_gradient import (
    CASES, PURPOSE, _chain_model, _objective, _presets, _receivers, _score, _settings,
)
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
# 不貼（#561 辛三）：終點那一組的兩個 None 面改成保留原牆，一面固定阻抗、一面拿 96K 棉板 preset 的構造當原牆
# （隨頻率變），兩種原牆都走到。數值只是一面牆，不是產品設定。
KEPT_CONSTANT_PA_S_M = 1646.4
KEPT_ASSEMBLY_PRESET = "glass_wool_96k_50mm"


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


def floor_case_pressures(frequencies_hz: tuple[float, ...],
                         receivers: ReceiverSet) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """審查反例（跨均方根地板）：兩支喇叭、主位各頻 100 dB，六個周圍點各頻 100＋100·spacing(100) 對
    100＋101·spacing(100)；p＝10^(dB/20)。回兩組 (F, 喇叭, 座位) 聲壓。"""
    pair = []
    for steps in (100, 101):
        levels = np.full((len(frequencies_hz), 2, len(receivers.points)), 100.0)
        for index, point in enumerate(receivers.points):
            if point.role == ReceiverRole.SURROUNDING:
                levels[:, :, index] = 100.0 + steps * np.spacing(100.0)
        pair.append((10.0 ** (levels / 20.0)).astype(np.complex128))
    return pair[0], pair[1]


def _minimum() -> float:
    """設定檔讀到的最小空腔（不從產品程式的常數取）。"""
    return load_material_gradient_settings(config_path("material_gradient.toml")).export.min_cavity_m


def _decode(values: object) -> NDArray[np.float64]:
    return np.asarray([float.fromhex(value) for value in cast(list[str], values)], dtype=np.float64)


def _decode_complex(node: object) -> NDArray[np.complex128]:
    row = cast(Record, node)
    return (_decode(row["real"]) + 1j * _decode(row["imag"])).reshape(cast(list[int], row["shape"]))


def _hex_list(values: tuple[float, ...]) -> list[str]:
    return [float(value).hex() for value in values]


def _kept() -> dict[str, SurfaceMaterial]:
    """終點那一組 None 面的原牆：第一面固定阻抗、第二面 96K 棉板構造（照構造預設清單那一份）。"""
    constant_face, assembly_face = (
        wall.wall_name() for wall, preset in zip(Wall.all(), ENDPOINT_PRESETS, strict=True) if preset is None)
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    by_id = {preset.id: preset for preset in _presets()}
    return {constant_face: ConstantImpedance(impedance_pa_s_m=KEPT_CONSTANT_PA_S_M),
            assembly_face: adapter.validate_python(by_id[KEPT_ASSEMBLY_PRESET].material)}


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
        "kept": {"presets": list(ENDPOINT_PRESETS), "theta": _hex_list(ENDPOINT_THETA),
                 "materials": {name: material.model_dump(mode="json") for name, material in _kept().items()}},
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

def _one_face(cavity: float, settings: MaterialGradientSettings | None = None) -> list[tuple[str, float]]:
    """第一面 48K 棉板 50 mm＋給定空腔，其餘五面剛性；回第一面匯出構造的（層種類, 厚度）清單，房間側起。"""
    designs = gradient_search.wall_designs_from_presets(
        ("glass_wool_48k_50mm_gap50", *(None,) * (len(Wall.all()) - 1)), _settings(), _presets())
    materials = endpoint.materials_from_surfaces(designs, np.asarray([0.05, cavity]), settings or _settings())
    material = materials[Wall.all()[0]]
    assert isinstance(material, Assembly) and material.backing == "rigid"
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
    """抓門檻往上偏（等於一格也被拿掉）：空腔恰好等於設定檔的最小空腔時放 ``AirLayer``、厚度原值。"""
    assert _one_face(_minimum()) == [("porous", 0.05), ("air", _minimum())]


def test_cavity_below_one_step_drops_air_layer() -> None:
    """抓只拿掉恰好 0、不管小於一格的正空腔：設定檔最小空腔的一半不放空氣層；而 40 mm 照放原值。"""
    assert _one_face(_minimum() / 2) == [("porous", 0.05)]
    assert _one_face(0.04) == [("porous", 0.05), ("air", 0.04)]


def test_cavity_rule_follows_settings_not_a_constant() -> None:
    """抓規則沒跟著設定檔走（程式裡另寫一個數）：自訂設定的最小空腔改成 2 mm，1.5 mm 拿掉、2 mm 留下；
    匯出 θ 走同一條規則。"""
    custom = _settings().model_copy(update={"export": ExportSettings(min_cavity_m=0.002)})
    assert _one_face(0.0015, custom) == [("porous", 0.05)]
    assert _one_face(0.002, custom) == [("porous", 0.05), ("air", 0.002)]
    assert _one_face(0.0015) == [("porous", 0.05), ("air", 0.0015)]
    designs = gradient_search.wall_designs_from_presets(ENDPOINT_PRESETS, _settings(), _presets())
    theta = np.asarray([0.1, 0.0015, 0.05, 0.04, 0.05, 0.05])
    assert endpoint.exported_theta(designs, theta, custom)[1] == 0.0
    assert endpoint.exported_theta(designs, theta, _settings())[1] == 0.0015


def test_min_cavity_setting_is_within_family_cavity_ranges() -> None:
    """抓設定檔的最小空腔填成不合理的值：要是正數，而且不超過每個家族的空腔上限（不然空腔永遠放不進去）；
    非有限的空腔一律拒收。"""
    minimum = _minimum()
    assert 0.0 < minimum <= min(family.cavity_range_m[1] for family in _settings().family)
    for bad in (math.inf, math.nan, -minimum):
        with pytest.raises(ValueError, match="非負有限數"):
            endpoint.kept_cavity(bad, minimum)


# ---- 材料來源、面序、空氣 ----

def _doubled_designs() -> list[gradient_search.SurfaceDesign]:
    """審查重現：第一面樣板的棉板流阻乘二、preset id 不動，其餘照終點那一組。"""
    designs = list(gradient_search.wall_designs_from_presets(ENDPOINT_PRESETS, _settings(), _presets()))
    first = designs[0]
    assert first.material is not None
    porous = first.material.layers[0]
    assert isinstance(porous, PorousLayer)
    doubled = porous.model_copy(update={"flow_resistivity_pa_s_m2": 2.0 * porous.flow_resistivity_pa_s_m2})
    designs[0] = dataclasses.replace(first, material=first.material.model_copy(
        update={"layers": (doubled, *first.material.layers[1:])}))
    return designs


def test_exported_material_comes_from_design_material() -> None:
    """抓匯出材料拿 preset id 去另一份清單重查：``materials_from_surfaces`` 只照 ``design.material``（替身鏈用的那一份）
    走——樣板流阻乘二、preset 不動時，匯出的流阻是乘二後的值（身分核對在終點入口，見下一題）。"""
    designs = _doubled_designs()
    template = designs[0].material
    assert template is not None and isinstance(template.layers[0], PorousLayer)
    materials = endpoint.materials_from_surfaces(designs, np.asarray(ENDPOINT_THETA), _settings())
    exported = materials[Wall.all()[0]]
    assert isinstance(exported, Assembly) and isinstance(exported.layers[0], PorousLayer)
    assert exported.layers[0].flow_resistivity_pa_s_m2 == template.layers[0].flow_resistivity_pa_s_m2


def test_endpoint_rejects_design_material_unlike_its_preset(chain_model: ReducedModel,
                                                            operators: FullOperators) -> None:
    """抓報告的 preset id 跟實際材料脫鉤（複查重現：流阻乘二、preset 不動，終點照算、報告仍寫原 preset）：
    終點入口拒收，訊息指名那一面與 preset id。"""
    designs = _doubled_designs()
    with pytest.raises(ValueError, match=f"{designs[0].surface_id}.*{designs[0].preset_id}"):
        _endpoint(chain_model, operators, designs, _air())


def test_endpoint_rejects_material_without_preset_id(chain_model: ReducedModel, operators: FullOperators) -> None:
    """抓有樣板構造但沒有 preset id 的面繞過身分核對（複驗重現：正常設計第一面改成 ``preset_id=None``、材料照舊，
    終點照算、報告 preset id 是 None）：終點入口拒收，訊息指名那一面；不跳過、不補預設 id。"""
    designs = list(gradient_search.wall_designs_from_presets(ENDPOINT_PRESETS, _settings(), _presets()))
    assert designs[0].material is not None
    designs[0] = dataclasses.replace(designs[0], preset_id=None)
    with pytest.raises(ValueError, match=f"面 {designs[0].surface_id} 有樣板構造但沒有 preset id"):
        _endpoint(chain_model, operators, designs, _air())


def test_report_material_digest_matches_preset_material(probe: Record) -> None:
    """抓摘要算錯面或沒記：正常設計的報告每面 ``material_digest`` 等於考卷自己從構造預設清單那一份材料算的摘要
    （驗證後內容的正規化 JSON 的 SHA-256）；剛性面是 None。"""
    by_id = {preset.id: preset for preset in _presets()}
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    report = _report(probe)
    for spec, preset_id in zip(report.surfaces, ENDPOINT_PRESETS, strict=True):
        assert spec.preset_id == preset_id
        if preset_id is None:
            assert spec.material_digest is None
            continue
        content = adapter.validate_python(by_id[preset_id].material).model_dump(mode="json")
        text = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert spec.material_digest == hashlib.sha256(text.encode("utf-8")).hexdigest()
    digests = [spec.material_digest for spec in report.surfaces if spec.material_digest is not None]
    assert len(set(digests)) > 1, "各面摘要都一樣，摘要算錯面的植錯驗不出來"


def test_designs_out_of_wall_order_are_rejected(chain_model: ReducedModel, operators: FullOperators) -> None:
    """抓面序沒核（鏈照位置配牆、匯出照名字配牆）：設計清單換掉兩面的順序，匯出與終點都拒收。"""
    designs = list(gradient_search.wall_designs_from_presets(ENDPOINT_PRESETS, _settings(), _presets()))
    designs[1], designs[4] = designs[4], designs[1]
    with pytest.raises(ValueError, match="Wall.all"):
        endpoint.materials_from_surfaces(designs, np.asarray(ENDPOINT_THETA), _settings())
    with pytest.raises(ValueError, match="Wall.all"):
        _endpoint(chain_model, operators, designs, _air())


def _air() -> PhysicsConstants:
    return load_physics_constants(config_path("physics_constants.toml"))


def _endpoint(chain_model: ReducedModel, operators: FullOperators, designs: list[gradient_search.SurfaceDesign],
              air: PhysicsConstants) -> endpoint.EndpointResult:
    return endpoint.endpoint_check(
        chain_operators(operators), designs, np.asarray(ENDPOINT_THETA), model=chain_model, objective_settings=_objective(),
        receiver_set=_receivers(), low_frequency_axis=AXIS_ID, presets=_presets(), settings=_settings(), air=air,
        ranges=load_porous_model_ranges(config_path("porous_model_ranges.toml")))


# 每一案：材料空氣改哪一格、替身的聲速改成多少（None 是不改）、拒收訊息要對到的值。
AIR_CASES: dict[str, tuple[dict[str, float], float | None, str]] = {
    "material_speed": ({"sound_speed_m_s": 350.0}, None, "350.0.*343.0"),
    "material_density": ({"air_density_kg_m3": 1.25}, None, "1.25.*1.2"),
    "surrogate_speed": ({}, 350.0, "跟替身.*350.0"),
}


@pytest.mark.parametrize("case", AIR_CASES)
def test_material_air_must_match_finite_element_air(chain_model: ReducedModel, operators: FullOperators,
                                                    case: str) -> None:
    """抓材料的空氣跟有限元素的空氣不同還照算（審查重現：材料聲速 350、模型 343；另只改材料密度、只改替身的聲速）：
    終點入口拒收，訊息是材料空氣那一條、列出兩邊的值；材料空氣跟算子不同時，完整解入口也拒收。"""
    material_change, surrogate_speed, message = AIR_CASES[case]
    designs = list(gradient_search.wall_designs_from_presets(ENDPOINT_PRESETS, _settings(), _presets()))
    assert (_air().air_density_kg_m3, _air().sound_speed_m_s) == (chain_model.density_kg_m3,
                                                                   chain_model.sound_speed_m_s)
    air = _air().model_copy(update=material_change)
    model = chain_model if surrogate_speed is None else dataclasses.replace(chain_model,
                                                                            sound_speed_m_s=surrogate_speed)
    with pytest.raises(ValueError, match=f"材料的空氣.*{message}"):
        _endpoint(model, operators, designs, air)
    if not material_change:
        return
    materials = endpoint.materials_from_surfaces(designs, np.asarray(ENDPOINT_THETA), _settings())
    with pytest.raises(ValueError, match=f"材料的空氣.*{message}"):
        endpoint.full_model_pressures(
            chain_operators(operators), materials, chain_model.frequencies_hz, air=air,
            ranges=load_porous_model_ranges(config_path("porous_model_ranges.toml")),
            max_relative_residual=chain_model.ledger.settings.full_solve_max_relative_residual)


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

    上界推導（``score_difference_bound``）：dB 差每格不超過 ε、均方根零判地板不超過 F 時，主項（扣加權直線＝同一
    內積下的正交投影，範數不增）之差 ≤ ε＋F、座位穩健性之差 ≤ 2ε＋F，合分之差 ≤ (ε+F)·w_主/ref_主 + (2ε+F)·w_穩/ref_穩。
    考卷這端不加餘量：上界自己要成立。地板由考卷從兩邊聲壓重算（目標同一個倍率 × max(max|dB|, 1)）。
    """
    report = _report(probe)
    full, reduced = _decode_complex(probe["full"]), _decode_complex(probe["reduced"])
    assert report.full_total == float.fromhex(cast(str, probe["direct_total"]))
    # 新合分與代理（#561 壬一）：完整模型那兩格等於獨立完整解交梯度目標的值（逐位）；上界是 T·ln N。
    assert [report.full_objective_total, report.full_proxy_db] == [
        float.fromhex(value) for value in cast(list[str], probe["direct_objective"])]
    gradient = gradient_objective_constants(chain_model.frequencies_hz, _objective(), _receivers(),
                                            len(chain_model.source_names))
    assert report.proxy_gap_bound_db == proxy_gap_bound_db(gradient)
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
    scale = max(1.0, float(np.max(np.abs(10.0 * np.log10(np.abs(np.concatenate((full, reduced))) ** 2)))))
    assert report.rms_floor_db == rms_floor_factor(np.float64) * scale
    assert report.total_difference_bound == endpoint.score_difference_bound(
        constants, report.max_frequency_db, report.rms_floor_db)
    assert abs(report.total_difference) <= report.total_difference_bound
    assert report.total_difference_bound <= endpoint.score_difference_bound(constants, FREQUENCY_RULER,
                                                                             report.rms_floor_db)
    assert report.full_solves == len(chain_model.frequencies_hz)


def test_score_bound_holds_across_rms_floor(probe: Record, chain_model: ReducedModel) -> None:
    """抓上界漏了均方根地板（審查反例：逐格差約 1.7e-14 dB、分數差約 1.4e-12，沒地板的上界只有約 5e-14）：
    兩組跨地板的聲壓交同一把尺（雙精度子行程算），分數差不超過含地板的上界；拿掉地板項的上界真的比分數差小
    （這一題才分得出地板項有沒有進去）。"""
    first, second = floor_case_pressures(chain_model.frequencies_hz, _receivers())
    scores = [float.fromhex(value) for value in cast(list[str], probe["floor_case"])]
    difference = abs(scores[1] - scores[0])
    level = float(np.max(np.abs(20.0 * np.log10(np.abs(first) / np.abs(second)))))
    constants = objective_constants(chain_model.frequencies_hz, _score(), _receivers(), len(chain_model.source_names))
    floor = endpoint.objective_floor_db(first, second)
    assert difference <= endpoint.score_difference_bound(constants, level, floor)
    assert difference > endpoint.score_difference_bound(constants, level, 0.0)


def test_score_bound_uses_each_weight_with_its_own_reference(chain_model: ReducedModel) -> None:
    """抓主項與穩健性的權重、參考接反（原 fixture 權重與參考都是 1，交換抓不到）：權重 0.7／0.3、參考 2.0／0.5 時，
    上界等於考卷手算的 (ε+F)·0.7÷2.0 + (2ε+F)·0.3÷0.5。"""
    constants = objective_constants(chain_model.frequencies_hz, _score(), _receivers(), len(chain_model.source_names))
    custom = constants._replace(component_weights=np.asarray([0.7, 0.3]), worse_references=np.asarray([2.0, 0.5]))
    level, floor = 0.01, 0.001
    expected = (level + floor) * 0.7 / 2.0 + (2.0 * level + floor) * 0.3 / 0.5
    assert endpoint.score_difference_bound(custom, level, floor) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("louder", ["first", "second"])
def test_objective_floor_takes_the_louder_side(chain_model: ReducedModel, louder: str) -> None:
    """抓地板只看一邊（例如只看完整模型或只看替身）：一組聲壓比另一組大 40 dB，不管大的那組在前或在後，
    地板都等於目標倍率 × 大的那組的 max|dB|（考卷從大的那組自己算）。"""
    loud, quiet = floor_case_pressures(chain_model.frequencies_hz, _receivers())
    quiet = quiet / 100.0
    pair = (loud, quiet) if louder == "first" else (quiet, loud)
    expected = rms_floor_factor(np.float64) * float(np.max(np.abs(10.0 * np.log10(np.abs(loud) ** 2))))
    assert endpoint.objective_floor_db(*pair) == expected


def test_endpoint_reports_dropped_cavity_and_exported_theta(probe: Record) -> None:
    """抓替身值在原 θ 算（含 0.5 mm 空腔）而完整模型在匯出材料算：被拿掉空氣層的那一面列在報告裡，
    匯出 θ 那一格是 0、其餘格逐位等於搜尋 θ；報告的規格是搜尋原值；替身值等於同一條鏈在考卷自己寫的匯出 θ 的值，
    不是在搜尋 θ 的值（兩者不同，這一題才分得出來）。"""
    report = _report(probe)
    at_export = float.fromhex(cast(str, probe["chain_at_expected_export"]))
    assert report.exported_theta == ENDPOINT_EXPORT and report.reduced_total == at_export
    assert [report.reduced_objective_total, report.reduced_proxy_db] == [
        float.fromhex(value) for value in cast(list[str], probe["chain_objective_at_expected_export"])]
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

def _lined_materials() -> dict[Wall, SurfaceMaterial | None]:
    designs = gradient_search.wall_designs_from_presets(CASES["all_lined"], _settings(), _presets())
    theta = gradient_search.snap_to_specs(gradient_search.baseline_theta(designs), designs).theta
    return endpoint.materials_from_surfaces(designs, theta, _settings())


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


# ---- 不貼：保留方案原本的牆（#561 辛三） ----

def test_kept_surface_chain_uses_original_wall_impedance(probe: Record) -> None:
    """抓不貼的面在替身鏈裡被當剛性、或用了別的阻抗：兩面帶原牆（固定阻抗與構造各一）的鏈在同一個 θ 的值，逐位等於
    把同一份產品 ``surface_impedance`` 算的原牆阻抗列直接換進樣板阻抗、遮罩那兩格放開，餵 ``reduced_pressures``＋
    ``objective_from_pressures`` 的值；而且跟同兩面剛性（設計不帶原牆）的值不同。"""
    kept = cast(Record, probe["kept"])
    assert kept["chain_value"] == kept["control_value"]
    assert kept["chain_value"] != kept["rigid_value"]


def test_kept_surface_endpoint_impedance_equals_chain_row(probe: Record, chain_model: ReducedModel) -> None:
    """抓終點完整模型用的原牆阻抗跟替身鏈不是同一份（換了頻率軸寫法或另一支函式）：``material_impedances`` 對不貼的面
    交出的逐頻阻抗，逐位等於替身鏈建目標時用的那一列（``kept_impedance_rows``）；兩面都在、都不是剛性。"""
    kept = cast(Record, probe["kept"])
    endpoint_rows, chain_rows = cast(Record, kept["endpoint_rows"]), cast(Record, kept["chain_rows"])
    assert set(endpoint_rows) == set(chain_rows) == set(_kept())
    for name in _kept():
        row = _decode_complex(chain_rows[name])
        assert row.shape == (len(chain_model.frequencies_hz),) and np.all(np.isfinite(row)), name
        np.testing.assert_array_equal(_decode_complex(endpoint_rows[name]), row)


def test_scene_with_materials_writes_kept_walls_back() -> None:
    """抓不貼的面匯出時寫成別的材料（或被當剛性拒收）：兩面帶原牆的設計匯出、寫進方案、存成 JSON 讀回，不貼的兩面
    材料等於原牆那一份（固定阻抗與構造各一），有樣板的面等於 ``materials_from_surfaces`` 那一面。"""
    scheme = Scheme.model_validate(_document())
    designs = gradient_search.wall_designs_from_presets(ENDPOINT_PRESETS, _settings(), _presets(), kept=_kept())
    materials = endpoint.materials_from_surfaces(designs, np.asarray(ENDPOINT_THETA), _settings())
    document = scheme.model_dump(mode="json")
    document["scene"] = endpoint.scene_with_materials(scheme.scene, materials).model_dump(mode="json")
    zones = Scheme.model_validate(json.loads(json.dumps(document))).scene.surface_zones_by_wall
    assert zones is not None
    kept = _kept()
    for wall, preset in zip(Wall.all(), ENDPOINT_PRESETS, strict=True):
        name = wall.wall_name()
        assert zones[name][0].material == (kept[name] if preset is None else materials[wall]), name


def test_chain_operators_rows_match_chain_model(model: ReducedModel, chain_model: ReducedModel) -> None:
    """抓考卷自己的組合跟替身那邊不同：同一個組合函式套在替身的三個座位列上，逐位等於衍生替身的七列。"""
    front, back, side = model.matrices.receivers
    np.testing.assert_array_equal(np.asarray(_combined(front, back, side)), chain_model.matrices.receivers)
