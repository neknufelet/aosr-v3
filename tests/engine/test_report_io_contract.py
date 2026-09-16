"""票 #316 報表輸入／輸出契約的考卷：schema 檔、驗證規則、值空必有原因。

**這一支在守什麼。** 票 #316 把三路報表的輸入與輸出收成凍結的 Pydantic 模型，並匯出兩份
正式 JSON schema 檔。這一支守五件事：

1. ``blueprint/schemas/`` 底下兩份 schema 檔**等於**模型現算出來的結果——檔過期就紅
   （改了模型忘了重匯，機器看得出來；不用人手更新）。
2. 輸入模型每一條驗證規則各一題。題目形狀照既有的
   ``tests/engine/test_three_lane_report_cli.py`` 與 ``test_capabilities.py`` 補齊，
   不重複造同一種壞輸入。
3. 輸出模型「值空必須有原因」的兩個方向各一題：值空沒原因要炸、值有又給原因也要炸。
4. 兩族「空」分得開：``fem_energy`` 空＝這一帶沒有有限元素頻點，**不必**帶原因；
   T20／T30 空＝值算不出來，**必須**帶原因（票 #316 第二刀監督拍板第 7 條）。
5. ``--format json`` 跑真的命令列，印出來的 JSON 反解得回模型（有限元素那一半照既有考卷
   換成假的）；文字輸出跟改之前逐字相同由既有的 CLI 考卷證明，這裡不另造比對工具。

**列類別是有欄名的物件。** ``TopFields``／``BandRow``／``PointRow`` 是 Pydantic 模型，
所以 JSON 出去是有欄名的物件、schema 檔裡是 ``properties``；考卷咬住這件事——位置陣列
等於沒有契約，前端讀不出哪一格是什麼。

**不碰真環境。** 只寫 ``tmp_path``（規矩卡 ``tests-isolated-from-real-env``）。
"""

from __future__ import annotations

import io
import json
import tempfile
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io
from aosr.physics.report_io import (
    BandRow,
    CapabilitySection,
    PointRow,
    ReportInput,
    ReportOutput,
    TopFields,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_DIR = _REPO_ROOT / "blueprint" / "schemas"
_TABLE_PATH = config_path("capabilities.toml")
_WALL_NAMES = tuple(wall.wall_name() for wall in Wall.all())


def _table() -> CapabilityTable:
    """產品能力表；輸入驗證要收的那一張（跟命令列 ``--capabilities`` 同一個檔）。"""
    return load_capabilities(_TABLE_PATH)


def _input_document(**overrides: object) -> dict[str, object]:
    """一份合法的輸入；``overrides`` 換掉頂層某一格，壞輸入題目就從這裡長出來。"""
    rho_c_pa_s_per_m = 1.2 * 343.0
    document: dict[str, object] = {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
        "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
        "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {
            wall: 4.0 * rho_c_pa_s_per_m for wall in _WALL_NAMES
        },
        "scattering_by_wall": {wall: 0.2 for wall in _WALL_NAMES},
    }
    document.update(overrides)
    return document


def _impedance_map(document: dict[str, object]) -> dict[str, object]:
    """題目文件裡那一格阻抗表；型別在題目這一側是刻意寬的（壞輸入要放得進去）。"""
    by_wall = document["impedance_pa_s_per_m_by_wall"]
    assert isinstance(by_wall, dict)
    return {str(key): value for key, value in by_wall.items()}


def _rejects(document: object, expected: str) -> str:
    """驗一份壞輸入要炸，而且訊息裡要有預期的字樣；回傳訊息給題目自己再咬。"""
    with pytest.raises(ValueError) as caught:
        report_io.load_input_document(document, _table())
    message = str(caught.value)
    assert expected in message, message
    return message


# ── ① schema 檔等於模型現算出來的結果 ────────────────────────────────────────────
@pytest.mark.parametrize(
    ("file_name", "computed"),
    (
        ("three_lane_report_input.schema.json", report_io.input_schema),
        ("three_lane_report_output.schema.json", report_io.output_schema),
    ),
)
def test_schema_files_match_the_models(
    file_name: str,
    computed: Callable[[], dict[str, object]],
) -> None:
    """schema 檔過期（模型改了沒重匯）本題必須紅。"""
    on_disk = json.loads((_SCHEMA_DIR / file_name).read_text(encoding="utf-8"))
    assert on_disk == computed()


@pytest.mark.parametrize(
    ("file_name", "computed"),
    (
        ("three_lane_report_input.schema.json", report_io.input_schema),
        ("three_lane_report_output.schema.json", report_io.output_schema),
    ),
)
def test_schema_files_declare_the_draft(
    file_name: str,
    computed: Callable[[], dict[str, object]],
) -> None:
    """兩份 schema 檔都要宣告 draft 2020-12（``$ref`` 旁邊那四件事的兄弟鍵只有它認得）。"""
    on_disk = json.loads((_SCHEMA_DIR / file_name).read_text(encoding="utf-8"))
    assert on_disk["$schema"] == report_io.JSON_SCHEMA_DRAFT
    assert computed()["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def _field_definitions(model_name: str) -> dict[str, dict[str, object]]:
    """輸出 schema 裡某個列模型的欄位定義（``properties``；有欄名的物件）。"""
    schema = report_io.output_schema()
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    definition = definitions[model_name]
    assert isinstance(definition, dict)
    properties = definition["properties"]
    assert isinstance(properties, dict)
    return {str(name): cell for name, cell in properties.items() if isinstance(cell, dict)}


@pytest.mark.parametrize("model_name", ("BandRow", "PointRow", "TopFields"))
def test_output_rows_are_named_objects_not_arrays(model_name: str) -> None:
    """三張列是**有欄名的物件**：加一欄不會靜靜改掉既有消費者讀到的意思。"""
    schema = report_io.output_schema()
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    definition = definitions[model_name]
    assert isinstance(definition, dict)
    assert definition["type"] == "object"
    assert "prefixItems" not in definition
    assert set(_field_definitions(model_name)) == set(
        {
            "TopFields": set(TopFields.model_fields),
            "BandRow": set(BandRow.model_fields),
            "PointRow": set(PointRow.model_fields),
        }[model_name]
    )


def test_json_payload_carries_field_names() -> None:
    """``--format json`` 的形狀（``model_dump``）是有欄名的物件，不是位置陣列。"""
    payload = json.loads(_output().model_dump_json())
    assert isinstance(payload["bands"], list)
    first = payload["bands"][0]
    assert isinstance(first, dict)
    assert "geometric_energy" in first
    assert isinstance(payload["top"], dict)
    assert "room_volume_m3" in payload["top"]


def test_schema_files_carry_the_four_facts_per_field() -> None:
    """每一欄的 schema 都要帶物理量、單位、參考基準、有效狀態四格。"""
    band_row = _field_definitions("BandRow")
    assert [cell["unit"] for cell in band_row.values()] == [
        "Hz", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1",
        "Hz", "1", "s", "1", "s", "1",
    ]
    for cell in band_row.values():
        assert cell["quantity"]
        assert cell["reference"]
        assert cell["validity"]


def test_reference_column_carries_a_basis_not_a_unit() -> None:
    """「參考基準」那一欄不准塞單位：有量綱的欄要寫相對於什麼或絕對值。

    這一題是票 #316 第二刀必修第 2 條咬住的那件事——``unit`` 已經有一格了，
    ``reference`` 再寫一次「無因次」等於同一句話說兩遍，而且對有量綱的欄是錯的。
    """
    for field_facts in report_io.quantity_table().values():
        assert field_facts.reference != "無因次", field_facts
    table = report_io.quantity_table()
    for name in (
        "sound_speed_m_s",
        "density_kg_m3",
        "impedance_pa_s_per_m_by_wall",
        "top.room_volume_m3",
    ):
        facts = table[name]
        assert facts.reference != facts.unit, name
        assert "絕對值" in facts.reference, name
    for name in (
        "outputs",
        "status",
        "evidence",
        "top.capped_by_upper_limit",
        "top.schroeder_band_count",
        "bands.t20_unavailable_reason",
        "bands.t30_unavailable_reason",
        "bands.fem_point_count",
    ):
        assert "沒有基準" in report_io.quantity_table()[name].reference, name


def test_mixed_basis_columns_say_they_are_mixed() -> None:
    """混合基準那三欄要寫實話，並帶上「兩路的基準沒有對過」。"""
    table = report_io.quantity_table()
    for name in (
        "bands.geometric_energy",
        "bands.geometric_contribution",
        "bands.total_energy",
        "points.geometric_energy",
        "points.total_energy",
    ):
        reference = table[name].reference
        assert "混合基準" in reference, name
    for name in ("bands.total_energy", "points.total_energy"):
        assert "沒有對過" in table[name].reference, name
    for name in ("bands.geometric_energy", "bands.geometric_contribution"):
        assert "4π" in table[name].reference, name


def test_quantity_table_covers_every_declared_field() -> None:
    """對照表要蓋到輸入、輸出與兩種列的每一欄，一格都不准漏。"""
    table = report_io.quantity_table()
    expected = (
        set(ReportInput.model_fields)
        | set(CapabilitySection.model_fields)
        | set(ReportOutput.model_fields)
        | {f"bands.{name}" for name in BandRow.model_fields}
        | {f"points.{name}" for name in PointRow.model_fields}
        | {f"top.{name}" for name in TopFields.model_fields}
    )
    assert set(table) == expected
    facts = table["bands.t20_s"]
    assert facts.quantity == "時間"
    assert facts.unit == "s"
    assert facts.validity.startswith("可估（空＝值算不出來")
    assert "視窗" in facts.reference


def test_quantity_table_separates_estimated_from_unestimated() -> None:
    """兩族「空」在對照表上要看得出差別，不是跟可估的混在一起。"""
    table = report_io.quantity_table()
    assert table["bands.fem_energy"].validity.startswith("不可估（空＝這一格沒有有限元素頻點")
    assert table["points.fem_energy"].validity.startswith("不可估（空＝這一格沒有有限元素頻點")
    assert table["bands.t20_s"].validity.startswith("可估（空＝值算不出來")
    assert table["bands.t30_s"].validity.startswith("可估（空＝值算不出來")
    # 說明與 validity 同一句話：那一欄的空值該不該帶原因寫在同一格裡。
    assert "必須帶原因" in table["bands.t20_s"].validity
    assert "有值就不准再給原因" in table["bands.t20_s"].validity
    assert "不必帶原因" in table["bands.fem_energy"].validity


def test_schema_facts_match_the_quantity_table() -> None:
    """schema 檔裡的四件事與對照表逐格相同——同一件事不准有兩個版本。"""
    band_row = _field_definitions("BandRow")
    table = report_io.quantity_table()
    for name in BandRow.model_fields:
        for key in ("quantity", "unit", "reference", "validity"):
            assert band_row[name][key] == getattr(table[f"bands.{name}"], key), name


def test_late_energy_reference_does_not_multiply_4pi_twice() -> None:
    """``_LATE`` 不再乘第二次 4π：``_exact_raw_energy`` 回傳的就是那個乘積。"""
    reference = report_io.quantity_table()["bands.late_energy"].reference
    assert "回傳的 raw_energy 本人" in reference
    assert "raw_energy × 4π" not in reference


def test_reference_strings_take_band_numbers_from_product_config() -> None:
    """reference 字串裡的頻帶數字不准從產品設定抄一份進來（會漂進 schema 檔）。"""
    from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ

    reference = report_io.quantity_table()["top.f_s_hz"].reference
    for frequency in SCHROEDER_T60_BANDS_HZ:
        assert f"{frequency:g}" not in reference
    assert "SCHROEDER_T60_BANDS_HZ" in reference
    for file_name in (
        "three_lane_report_input.schema.json",
        "three_lane_report_output.schema.json",
    ):
        text = (_SCHEMA_DIR / file_name).read_text(encoding="utf-8")
        for frequency in SCHROEDER_T60_BANDS_HZ:
            assert f"{frequency:g}" not in text, (file_name, frequency)


# ── ② 輸入模型每一條驗證規則各一題 ──────────────────────────────────────────────
def test_accepts_a_complete_document() -> None:
    """合法輸入要收得下，而且收回來的是求解層吃的形狀。"""
    inputs = report_io.load_input_document(_input_document(), _table())
    assert inputs.room_m == Room(6.0, 4.0, 3.0)
    assert inputs.source_m == Point(1.2, 1.3, 1.1)
    solved = report_io.solver_inputs(inputs)
    assert set(solved.impedance_by_wall) == set(Wall.all())
    assert set(solved.scattering_by_wall or {}) == set(Wall.all())


def test_scattering_may_be_omitted_entirely() -> None:
    """散射整格可省略，省略時是 None（由幾何路套既有預設值）。"""
    document = _input_document()
    del document["scattering_by_wall"]
    assert report_io.load_input_document(document, _table()).scattering_by_wall is None


def test_scattering_may_be_zero_on_every_wall() -> None:
    """散射 0 是合法輸入，不准被「正實數」那條擋掉。"""
    inputs = report_io.load_input_document(
        _input_document(scattering_by_wall={wall: 0.0 for wall in _WALL_NAMES}),
        _table(),
    )
    assert inputs.scattering_by_wall == {wall: 0.0 for wall in _WALL_NAMES}


def test_missing_room_component_is_named() -> None:
    """房少一軸要指名哪一軸（Pydantic 的欄位路徑帶出來，不是只說「輸入錯」）。"""
    _rejects(
        _input_document(room_m={"Lx": 6.0, "Ly": 4.0}),
        "Lz",
    )


def test_room_must_be_a_json_object() -> None:
    """房不是物件就擋，訊息跟原本一樣。"""
    _rejects(_input_document(room_m=[6.0, 4.0, 3.0]), "room_m 必須是 JSON 物件")


def test_point_must_be_a_json_object() -> None:
    """聲源不是物件就擋，訊息指名那一格。"""
    _rejects(_input_document(source_m="1.2,1.3,1.1"), "source_m 必須是 JSON 物件")


def test_point_component_must_be_a_finite_number() -> None:
    """點的座標要有限數字；布林不算數字。"""
    _rejects(
        _input_document(receiver_m={"x": True, "y": 2.8, "z": 1.4}),
        "必須是有限數字",
    )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")), ids=str)
def test_non_finite_numbers_are_rejected(value: float) -> None:
    """非有限值一律拒收（變異考卷：三個方向都要炸）。"""
    _rejects(_input_document(sound_speed_m_s=value), "必須是有限數字")


def test_non_positive_sound_speed_is_rejected() -> None:
    """聲速必須是有限正數。"""
    _rejects(_input_document(sound_speed_m_s=0.0), "sound_speed_m_s 必須是有限正數")


def test_non_positive_density_is_rejected() -> None:
    """密度必須是有限正數。"""
    _rejects(_input_document(density_kg_m3=-1.2), "density_kg_m3 必須是有限正數")


def test_non_positive_room_axis_is_rejected() -> None:
    """房三軸長都必須是有限正數。"""
    _rejects(
        _input_document(room_m={"Lx": 6.0, "Ly": 4.0, "Lz": -3.0}),
        "必須是有限正數",
    )


def test_missing_wall_impedance_is_rejected() -> None:
    """六個牆名固定；缺一面要指名那一面。"""
    by_wall = _impedance_map(_input_document())
    del by_wall["yL"]
    _rejects(_input_document(impedance_pa_s_per_m_by_wall=by_wall), "yL")


def test_negative_impedance_is_rejected_with_the_unsupported_hint() -> None:
    """阻抗要正；訊息照原本帶上能力表那兩條 unsupported 的 note。"""
    by_wall = _impedance_map(_input_document())
    by_wall["floor"] = -5.0
    message = _rejects(
        _input_document(impedance_pa_s_per_m_by_wall=by_wall),
        "必須是正實數阻抗",
    )
    assert "unsupported" in message
    assert "#309" in message


@pytest.mark.parametrize(
    "cell",
    (
        {"real": 1646.4, "imag": -100.0},
        [1646.4, 1646.5, 1646.6],
        {"125.0": 1646.4, "250.0": 1646.5},
    ),
    ids=("complex_object", "frequency_list", "frequency_object"),
)
def test_complex_or_frequency_dependent_impedance_is_rejected(cell: object) -> None:
    """複數或逐頻阻抗（物件或陣列）當場拒收，訊息提到票 #309 與 unsupported。"""
    by_wall = _impedance_map(_input_document())
    by_wall["floor"] = cell
    message = _rejects(
        _input_document(impedance_pa_s_per_m_by_wall=by_wall),
        "unsupported",
    )
    assert "#309" in message


@pytest.mark.parametrize("value", (-0.01, 1.01, 5.0), ids=str)
def test_scattering_outside_unit_interval_is_rejected(value: float) -> None:
    """散射係數落在 [0,1] 之外就擋。"""
    _rejects(
        _input_document(scattering_by_wall={wall: value for wall in _WALL_NAMES}),
        "[0,1]",
    )


def test_missing_scattering_wall_is_rejected() -> None:
    """散射六面固定；物件裡缺一面就擋並指名那一面。"""
    scattering = {wall: 0.2 for wall in _WALL_NAMES if wall != "ceiling"}
    _rejects(_input_document(scattering_by_wall=scattering), "ceiling")


def test_unknown_extra_field_is_rejected() -> None:
    """多一個沒登記的欄位就擋（extra=forbid）。"""
    _rejects(_input_document(materials="wood"), "materials")


def test_frozen_input_cannot_be_mutated() -> None:
    """輸入模型是凍結的：改一格要炸，不准出現第二種真相。"""
    inputs = report_io.load_input_document(_input_document(), _table())
    with pytest.raises(Exception):
        setattr(inputs, "density_kg_m3", 1.0)


# ── ③ 輸出模型：值空必須有原因（兩個方向）────────────────────────────────────────
def _band(**overrides: object) -> BandRow:
    defaults: dict[str, object] = {
        "center_frequency_hz": 1000.0,
        "fem_energy": 1.0,
        "fem_point_count": 1,
        "direct_energy": 1.0,
        "reflected_energy": 1.0,
        "interference_energy": 0.0,
        "late_energy": 1.0,
        "geometric_energy": 1.0,
        "fem_contribution": 1.0,
        "geometric_contribution": 1.0,
        "total_energy": 1.0,
        "w_fem": 0.5,
        "w_geo": 0.5,
        "f_s_hz": 200.0,
        "capped_by_upper_limit": False,
        "t20_s": 1.0,
        "t20_unavailable_reason": None,
        "t30_s": 1.0,
        "t30_unavailable_reason": None,
    }
    defaults.update(overrides)
    return BandRow(**defaults)  # type: ignore[arg-type]  # expires=2026-12-08 reason=這一支刻意用寬鬆的 kwargs 組列，欄名由上面的 defaults 決定


def _point(**overrides: object) -> PointRow:
    defaults: dict[str, object] = {
        "frequency_hz": 1000.0,
        "fem_energy": 1.0,
        "direct_energy": 1.0,
        "reflected_energy": 1.0,
        "interference_energy": 0.0,
        "late_energy": 1.0,
        "scattering": 0.2,
        "geometric_energy": 1.0,
        "w_fem": 0.5,
        "w_geo": 0.5,
        "total_energy": 1.0,
    }
    defaults.update(overrides)
    return PointRow(**defaults)  # type: ignore[arg-type]  # expires=2026-12-08 reason=這一支刻意用寬鬆的 kwargs 組列，欄名由上面的 defaults 決定


def _output(**overrides: object) -> ReportOutput:
    defaults: dict[str, object] = {
        "capability": CapabilitySection(
            frequency_hz=(20.0, 5583.0),
            outputs=("total_energy",),
            status="experimental",
            evidence=(),
        ),
        "top": TopFields(
            f_s_hz=200.0,
            crossover_lower_hz=200.0,
            crossover_upper_hz=300.0,
            capped_by_upper_limit=False,
            eyring_t60_by_band_s={"500.0": 1.0},
            room_volume_m3=72.0,
            schroeder_band_count=2,
        ),
        "bands": (_band(),),
        "points": None,
    }
    defaults.update(overrides)
    return ReportOutput(**defaults)  # type: ignore[arg-type]  # expires=2026-12-08 reason=同上


def test_empty_value_without_a_reason_is_rejected() -> None:
    """方向一：值空而沒有原因要報錯。"""
    with pytest.raises(Exception) as caught:
        _output(bands=(_band(t20_s=None, t20_unavailable_reason=None),))
    assert "必須帶不可估原因" in str(caught.value)


def test_value_with_a_reason_is_rejected() -> None:
    """方向二：值有而又給原因也要報錯。"""
    with pytest.raises(Exception) as caught:
        _output(bands=(_band(t20_s=1.0, t20_unavailable_reason="擬合無效"),))
    assert "不准再給不可估原因" in str(caught.value)


def test_empty_fem_energy_needs_no_reason() -> None:
    """另一族：``fem_energy`` 空＝這一帶沒有有限元素頻點，不必帶原因。

    票 #316 第二刀監督拍板第 7 條：它跟 T20 的「算不出來」不同族，所以空的
    ``fem_energy``（``fem_point_count`` 是 0）要收得下，不准被那條規則擋掉。
    """
    output = _output(
        bands=(
            _band(fem_energy=None, fem_point_count=0),
            _band(center_frequency_hz=2000.0),
        )
    )
    assert output.bands[0].fem_energy is None
    assert output.bands[0].fem_point_count == 0
    assert output.bands[0].t20_s is not None


def test_empty_decay_value_in_the_same_row_still_needs_a_reason() -> None:
    """同一列上兩族分得開：``fem_energy`` 空不必原因，T20 空一樣要原因。"""
    with pytest.raises(Exception) as caught:
        _output(
            bands=(
                _band(fem_energy=None, fem_point_count=0, t20_s=None),
                _band(center_frequency_hz=2000.0),
            )
        )
    message = str(caught.value)
    assert "t20_s 是空的就必須帶不可估原因" in message
    assert "fem_energy" not in message


def test_empty_point_fem_energy_needs_no_reason() -> None:
    """逐點表那一欄同一族：空的 ``fem_energy`` 是「這一格沒有值」，不必帶原因。"""
    output = _output(points=(_point(fem_energy=None),))
    assert output.points is not None
    assert output.points[0].fem_energy is None
    assert output.points[0].total_energy == 1.0


def test_empty_value_with_a_reason_is_accepted() -> None:
    """合法的形狀：值空、原因在，其他結果照留。"""
    output = _output(
        bands=(
            _band(
                t20_s=None,
                t20_unavailable_reason="T20 擬合無效：第 256 階最低 -20 dB",
                t30_s=None,
                t30_unavailable_reason="T30 擬合無效：第 256 階最低 -20 dB",
            ),
        )
    )
    assert output.bands[0].total_energy == 1.0
    assert output.bands[0].t20_s is None
    assert output.bands[0].t20_unavailable_reason is not None


def test_unchecked_capability_does_not_invent_a_column_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """沒查表時 ``outputs`` 是空的，不准造一個叫 ``none`` 的欄名（必修第 9 條）。

    ``capability_report.capability_line`` 那一層印的是 ``outputs=none`` 給人看的
    一行；契約這一層是資料，空就是空——造一個字串欄名會讓前端以為真有那一欄。
    """
    from aosr.physics import three_lane_report

    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    report = three_lane_report.solve_three_lane_report(
        room=Room(6.0, 4.0, 3.0),
        source=Point(1.2, 1.3, 1.1),
        receiver=Point(4.7, 2.8, 1.4),
        sound_speed_m_s=343.0,
        density_kg_m3=1.2,
        impedance_by_wall={wall: 4.0 * 411.6 for wall in Wall.all()},
    )

    section = report_io.output_from_report(
        report, room=Room(6.0, 4.0, 3.0), with_points=False
    ).capability
    assert section.outputs == ()
    assert section.evidence == ()
    assert section.status == "unchecked"


def test_empty_bands_are_rejected() -> None:
    """報表一定有頻帶列；空的要炸。"""
    with pytest.raises(Exception) as caught:
        _output(bands=())
    assert "不可為空" in str(caught.value)


def test_out_of_order_band_centers_are_rejected() -> None:
    """頻帶列的中心頻率必須遞增。"""
    with pytest.raises(Exception) as caught:
        _output(
            bands=(
                _band(center_frequency_hz=1000.0),
                _band(center_frequency_hz=500.0),
            )
        )
    assert "遞增" in str(caught.value)


def test_descending_crossover_range_is_rejected() -> None:
    """交接下端不准大於上端。"""
    with pytest.raises(Exception):
        _output(
            top=TopFields(
                f_s_hz=400.0,
                crossover_lower_hz=400.0,
                crossover_upper_hz=300.0,
                capped_by_upper_limit=False,
                eyring_t60_by_band_s={"500.0": 1.0},
                room_volume_m3=72.0,
                schroeder_band_count=2,
            )
        )


def test_frozen_output_cannot_be_mutated() -> None:
    """輸出模型是凍結的：改一格要炸。"""
    output = _output()
    with pytest.raises(Exception):
        setattr(output, "bands", ())


# ── ④ --format json 跑真的命令列 ───────────────────────────────────────────────
def _fake_fem_energy(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> tuple[float, ...]:
    """有限元素那一半換成假的（照既有 CLI 考卷），只驗接線與契約。"""
    del room, source, receiver, wall_impedances
    assert density_kg_m3 == 1.2
    assert sound_speed_m_s == 343.0
    return tuple(0.000012345 + frequency * 1e-10 for frequency in frequencies_hz)


@pytest.mark.parametrize("extra", ((), ("--points",)), ids=("bands", "with_points"))
def test_json_format_round_trips_through_the_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, ...],
) -> None:
    """``--format json`` 印出來的 JSON 要反解得回模型，而且帶不帶 --points 都對。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_input_document()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [
            str(input_path),
            *extra,
            "--format",
            "json",
            "--capabilities",
            str(_TABLE_PATH),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0, output
    parsed = ReportOutput.model_validate_json(output)
    assert parsed.capability.status == "experimental"
    assert parsed.top.room_volume_m3 == 72.0
    centers = [band.center_frequency_hz for band in parsed.bands]
    assert parsed.points is not None if extra else parsed.points is None
    assert all(
        parsed.bands[index].center_frequency_hz < parsed.bands[index + 1].center_frequency_hz
        for index in range(len(centers) - 1)
    )
    assert all(
        band.t20_s is None or band.t20_unavailable_reason is None
        for band in parsed.bands
    )
    payload = json.loads(output)
    assert set(payload) == {"capability", "top", "bands", "points"}
    assert set(payload["top"]) == set(TopFields.model_fields)
    assert set(payload["bands"][0]) == set(BandRow.model_fields)
    if extra:
        assert set(payload["points"][0]) == set(PointRow.model_fields)


def test_json_format_defaults_to_text_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """不給 `--format` 時是 text；JSON 那一條不准變成預設。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_input_document()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output.startswith("capability entry=")
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)


def test_json_format_still_returns_two_on_bad_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """壞輸入走 JSON 那條路一樣回 2，不准因為要印 JSON 就把錯誤吞掉。"""
    from aosr.physics import three_lane_report_cli

    document = _input_document()
    by_wall = _impedance_map(document)
    by_wall["floor"] = -5.0
    document["impedance_pa_s_per_m_by_wall"] = by_wall
    input_path = tmp_path / "bad.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--format", "json", "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "三路接合報表算不出來" in output


# ── ⑤ 壞輸入的錯誤文字：一句人話，不是 Pydantic 的多行 dump ─────────────────────
def test_bad_input_message_names_the_field_without_the_website() -> None:
    """壞輸入印的是一句人話：欄位路徑與原因在，官網網址與多行 dump 不在。"""
    from aosr.physics import three_lane_report_cli

    document = _input_document()
    by_wall = _impedance_map(document)
    by_wall["floor"] = -5.0
    document["impedance_pa_s_per_m_by_wall"] = by_wall
    message = _cli_error(three_lane_report_cli, document)

    assert "impedance_pa_s_per_m_by_wall：impedance_pa_s_per_m_by_wall.floor" in message
    assert "必須是正實數阻抗" in message
    assert "https://" not in message
    assert "errors.pydantic.dev" not in message
    assert "For further information visit" not in message
    # 一句人話就是一行：除了 print 尾巴那一個換行，訊息裡不准再有換行。
    body = message.removesuffix("\n")
    assert "\n" not in body
    assert body == message.rstrip("\n")


def test_bad_input_message_keeps_the_value_error_text() -> None:
    """Pydantic 自己加的 ``Value error, `` 前綴要拿掉；原因本身一字不動。"""
    from aosr.physics import three_lane_report_cli

    message = _cli_error(
        three_lane_report_cli,
        _input_document(sound_speed_m_s=0.0),
    )

    assert "輸入：sound_speed_m_s 必須是有限正數" in message
    assert "Value error" not in message
    assert "https://" not in message


def _cli_error(
    cli_module: ModuleType,
    document: dict[str, object],
    *,
    extra: tuple[str, ...] = (),
) -> str:
    """把一份題目文件寫進暫時檔、跑一次命令列，收下它印出來的那一行（回 2）。

    只寫暫時檔、只讀 stdout，不碰真環境；訊息裡不准有換行，因為命令列那一層
    印的就是**一行**。
    """
    input_path = _write_document(document)
    captured = io.StringIO()
    with redirect_stdout(captured):
        exit_code = cli_module.main(
            [str(input_path), *extra, "--capabilities", str(_TABLE_PATH)]
        )
    output = captured.getvalue()
    assert exit_code == 2, output
    assert output.startswith("三路接合報表算不出來：")
    return output


def _write_document(document: dict[str, object]) -> Path:
    """把一份題目文件寫進一個獨立的暫時檔（``delete=False``，命令列讀得到）。"""
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115  # expires=2026-12-08 reason=檔名要活到命令列讀完（delete=False），關檔只是收掉把手
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        json.dump(document, handle)
    return Path(handle.name)
