"""格式檔說得出來的界限，要跟後端真的擋的是同一份東西（票 #316 合併後的外部核對）。

**為什麼這一支自己一個檔。** 契約那一支（``test_report_io_contract.py``）守的是「欄位與
四件事」，命令列那一支守的是「對人說話的行為」；這裡守的是第三件事——兩種入口的**允許
範圍一致**。前端或 agent 會照 ``blueprint/schemas/`` 那兩份格式檔做表單、產輸入；格式檔
比後端鬆，就會出現「前端說合法、送到後端被拒收」，而那種錯是使用者看不懂的。

三條：

1. 界限只有一份數字——格式檔裡的 ``exclusiveMinimum``／``minimum``／``maximum`` 等於
   :mod:`aosr.physics.report_facts` 那幾個常數，驗證函式讀的也是它們。
2. 六面牆那一格的形狀在格式檔裡說得出來：六個牆名都必填、不認識的牆名不收。
3. 逐值對照：格式檔擋得掉的那些值，後端也擋；後端擋掉的這幾種，格式檔看得出來。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.physics import report_facts, report_io


def _wall_names() -> list[str]:
    return [wall.wall_name() for wall in Wall.all()]


def _table() -> CapabilityTable:
    return load_capabilities(config_path("capabilities.toml"))


def _input_schema() -> dict[str, object]:
    """讀版控裡那一份匯出檔（前端讀到的就是它），不是現算一份來比。"""
    text = Path("blueprint/schemas/three_lane_report_input.schema.json").read_text(
        encoding="utf-8"
    )
    loaded = json.loads(text)
    assert isinstance(loaded, dict)
    return loaded


def _properties() -> dict[str, dict[str, object]]:
    """匯出檔的 properties 那一層，逐格窄化成字典（型別警衛不准整塊標成隨便什麼）。"""
    properties = _input_schema()["properties"]
    assert isinstance(properties, dict)
    return {
        str(name): {str(key): item for key, item in cell.items()}
        for name, cell in properties.items()
        if isinstance(cell, dict)
    }


def _is_closed_set(cell: dict[str, object]) -> bool:
    """這一格指到的型別是封閉的列舉（例如報表低頻軸，#435）：可填的值全寫在格式檔裡，本身就是界線。"""
    reference = cell.get("$ref")
    if not isinstance(reference, str):
        return bool(cell.get("enum"))
    definitions = _input_schema()["$defs"]
    assert isinstance(definitions, dict)
    target = definitions[reference.rsplit("/", 1)[-1]]
    assert isinstance(target, dict)
    return bool(target.get("enum"))


def _cell(owner: dict[str, object], name: str) -> dict[str, object]:
    """牆面那一格底下的某一面牆：取出來就窄化，型別檢查才看得懂後面怎麼用。"""
    inner = owner["properties"]
    assert isinstance(inner, dict)
    wall = inner[name]
    assert isinstance(wall, dict)
    return {str(key): item for key, item in wall.items()}


def _document(**changes: object) -> dict[str, object]:
    """一份會過的輸入；呼叫端只改自己要壞的那一格。"""
    document: dict[str, object] = {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
        "source_m": {"x": 1.5, "y": 1.0, "z": 1.2},
        "receiver_m": {"x": 4.0, "y": 3.0, "z": 1.5},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {name: 1646.4 for name in _wall_names()},
    }
    document.update(changes)
    return document


def test_schema_bounds_are_the_same_numbers_the_validators_use() -> None:
    """格式檔裡的界限就是驗證函式讀的那幾個常數，不是另外手寫的第二份。"""
    properties = _properties()

    for name in ("sound_speed_m_s", "density_kg_m3"):
        assert properties[name]["exclusiveMinimum"] == report_facts.POSITIVE_EXCLUSIVE_MINIMUM

    impedance_cell = _cell(properties["impedance_pa_s_per_m_by_wall"], "floor")
    assert impedance_cell["exclusiveMinimum"] == report_facts.POSITIVE_EXCLUSIVE_MINIMUM

    scattering_cell = _cell(properties["scattering_by_wall"], "floor")
    assert scattering_cell["minimum"] == report_facts.SCATTERING_MINIMUM
    assert scattering_cell["maximum"] == report_facts.SCATTERING_MAXIMUM


# 這幾格的允許範圍在格式檔裡「說不出來」是有理由的，理由寫在這裡；名單以外漏一格就紅。
# 名單本身要短：每加一格就是一句「前端看不出來的限制」，加之前先想能不能寫進格式檔。
_WITHOUT_BOUNDS: dict[str, str] = {
    "source_m": "座標沒有正負限制（房間角落為原點，允許負值）；形狀那一半照樣說得出來",
    "receiver_m": "同上",
}


def test_every_input_field_declares_its_limits_in_the_schema() -> None:
    """逐欄列舉：輸入的每一欄在匯出檔裡都要說得出界限或形狀，說不出的要有名字與理由。

    手點名單守不住新增的欄——房那三軸的正數限制就是這樣漏掉的（格式檔只寫 number、
    後端擋負值）。這一題改成走 ``ReportInput.model_fields``，漏一格就紅。
    """
    properties = _properties()
    missing: list[str] = []
    for name in report_io.ReportInput.model_fields:
        cell = properties[name]
        has_number_bound = any(
            key in cell for key in ("exclusiveMinimum", "minimum", "maximum")
        )
        shape = cell.get("required")
        has_shape = bool(shape) and cell.get("additionalProperties") is False
        if not (
            has_number_bound or has_shape or _is_closed_set(cell) or name in _WITHOUT_BOUNDS
        ):
            missing.append(name)

    assert not missing, f"這幾欄的限制只住在後端，格式檔說不出來：{missing}"
    # 名單裡的每一格都要真的還在模型上（欄位改名了就該重想，不是留一筆死理由）。
    assert set(_WITHOUT_BOUNDS) <= set(report_io.ReportInput.model_fields)


def test_room_lengths_must_be_positive_in_the_schema_too() -> None:
    """房的三軸長在格式檔裡也要寫得出「必須大於零」，不是只有後端知道。"""
    room = _properties()["room_m"]
    for name in ("Lx", "Ly", "Lz"):
        assert _cell(room, name)["exclusiveMinimum"] == (
            report_facts.POSITIVE_EXCLUSIVE_MINIMUM
        )
    assert room["additionalProperties"] is False


def test_an_extra_key_in_room_or_point_is_rejected() -> None:
    """房與座標多打一個鍵也要擋，跟牆名同一條理由（多打的被忽略最難查）。"""
    for field, extra_key in (("room_m", "Lw"), ("source_m", "Z"), ("receiver_m", "w")):
        original = _document()[field]
        assert isinstance(original, dict)
        broken: dict[str, object] = {str(key): item for key, item in original.items()}
        broken[extra_key] = 1.0
        with pytest.raises((ValidationError, ValueError)) as caught:
            report_io.load_input_document(_document(**{field: broken}), _table())
        message = str(caught.value)
        assert extra_key in message, field
        assert field in message, field


def test_schema_says_the_six_wall_names_are_required_and_nothing_else_fits() -> None:
    """六面牆那一格的形狀寫在格式檔裡：牆名必填、不認識的牆名不收。"""
    properties = _properties()

    for name in ("impedance_pa_s_per_m_by_wall", "scattering_by_wall"):
        cell = properties[name]
        assert cell["required"] == _wall_names(), name
        assert cell["additionalProperties"] is False, name
        declared = cell["properties"]
        assert isinstance(declared, dict)
        assert sorted(str(key) for key in declared) == sorted(_wall_names()), name


@pytest.mark.parametrize(
    ("changes", "wanted"),
    [
        ({"sound_speed_m_s": -343.0}, "sound_speed_m_s"),
        ({"density_kg_m3": 0.0}, "density_kg_m3"),
        ({"impedance_pa_s_per_m_by_wall": {}}, "impedance_pa_s_per_m_by_wall"),
        (
            {"scattering_by_wall": {name: 1.4 for name in _wall_names()}},
            "scattering_by_wall",
        ),
    ],
)
def test_values_the_schema_rejects_are_rejected_by_the_loader_too(
    changes: dict[str, object], wanted: str
) -> None:
    """格式檔說不合的那幾種值，後端也不收——兩種入口同一個允許範圍。"""
    with pytest.raises((ValidationError, ValueError)) as caught:
        report_io.load_input_document(_document(**changes), _table())

    assert wanted in str(caught.value)


def test_an_extra_wall_name_is_rejected_instead_of_silently_ignored() -> None:
    """六面齊全再多一個拼錯的牆名要當場擋下來。

    以前會靜靜忽略：改天花板的人把 ``ceiling`` 拼成 ``ceilling``，正式那一格的舊值還在、
    算得出結果，改的人卻以為改生效了。那比報錯難查。
    """
    walls = {name: 1646.4 for name in _wall_names()}
    walls["ceilling"] = 99.0

    with pytest.raises((ValidationError, ValueError)) as caught:
        report_io.load_input_document(
            _document(impedance_pa_s_per_m_by_wall=walls), _table()
        )

    message = str(caught.value)
    assert "ceilling" in message
    assert "impedance_pa_s_per_m_by_wall" in message


def test_a_blank_unavailable_reason_does_not_count_as_a_reason() -> None:
    """值空、原因只有空白，等於說不出為什麼；契約不准放它過。"""
    for reason in ("", "   "):
        with pytest.raises(ValueError) as caught:
            report_io._reason_or_value(None, reason, where="bands[0].t20_s")

        assert "必須帶不可估原因" in str(caught.value)

    # 真的有字的原因照樣過，這條不是把整族擋死。
    report_io._reason_or_value(None, "低吸音房，衰減曲線拉不到 -25 dB", where="bands[0].t20_s")


def test_a_band_row_with_a_blank_reason_is_rejected_by_the_model() -> None:
    """同一條規則要真的掛在列類別上，不只掛在整份輸出上。"""
    with pytest.raises(ValidationError) as caught:
        report_io.BandRow(
            center_frequency_hz=125.0,
            fem_energy=None,
            fem_point_count=0,
            direct_energy=1.0,
            reflected_energy=0.0,
            interference_energy=0.0,
            late_energy=0.0,
            geometric_energy=1.0,
            fem_contribution=0.0,
            geometric_contribution=1.0,
            total_energy=1.0,
            w_fem=0.0,
            w_geo=1.0,
            f_s_hz=100.0,
            capped_by_upper_limit=False,
            t20_s=None,
            t20_unavailable_reason="   ",
            t30_s=None,
            t30_unavailable_reason="值算不出來的真原因",
        )

    assert "t20_s" in str(caught.value)
