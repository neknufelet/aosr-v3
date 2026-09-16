"""票 #316 報表輸入／輸出契約的考卷：schema 檔、驗證規則、值空必有原因。

**這一支在守什麼。** 票 #316 把三路報表的輸入與輸出收成凍結的 Pydantic 模型，並匯出兩份
正式 JSON schema 檔。這一支守四件事：

1. ``blueprint/schemas/`` 底下兩份 schema 檔**逐位等於**模型現算出來的結果——檔過期就紅
   （改了模型忘了重匯，機器看得出來；不用人手更新）。
2. 輸入模型每一條驗證規則各一題。題目形狀照既有的
   ``tests/engine/test_three_lane_report_cli.py`` 與 ``test_capabilities.py`` 補齊，
   不重複造同一種壞輸入。
3. 輸出模型「值空必須有原因」的兩個方向各一題：值空沒原因要炸、值有又給原因也要炸。
4. ``--format json`` 跑真的命令列，印出來的 JSON 反解得回模型（有限元素那一半照既有考卷
   換成假的）；文字輸出跟改之前逐字相同由既有的 CLI 考卷證明，這裡不另造比對工具。

**不碰真環境。** 只寫 ``tmp_path``（規矩卡 ``tests-isolated-from-real-env``）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

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
        report_io.load_input_document(document)
    message = str(caught.value)
    assert expected in message, message
    return message


# ── ① schema 檔逐位等於模型現算出來的結果 ────────────────────────────────────────
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


def _band_row_cells() -> list[dict[str, object]]:
    """輸出 schema 裡頻帶列那一串欄位定義（``prefixItems``）。"""
    schema = report_io.output_schema()
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    band_definition = definitions["BandRow"]
    assert isinstance(band_definition, dict)
    band_row = band_definition["prefixItems"]
    assert isinstance(band_row, list)
    return [cell for cell in band_row if isinstance(cell, dict)]


def test_schema_files_carry_the_four_facts_per_field() -> None:
    """每一欄的 schema 都要帶物理量、單位、參考基準、有效狀態四格。"""
    band_row = _band_row_cells()
    assert [cell["unit"] for cell in band_row] == [
        "Hz", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1",
        "Hz", "1", "s", "1", "s", "1",
    ]
    for cell in band_row:
        assert cell["quantity"]
        assert cell["reference"]
        assert cell["validity"]


def test_quantity_table_covers_every_declared_field() -> None:
    """對照表要蓋到輸入、輸出與兩種列的每一欄，一格都不准漏。"""
    table = report_io.quantity_table()
    expected = (
        set(ReportInput.model_fields)
        | set(CapabilitySection.model_fields)
        | set(ReportOutput.model_fields)
        | {f"bands.{name}" for name in BandRow._fields}
        | {f"points.{name}" for name in PointRow._fields}
        | {f"top.{name}" for name in TopFields._fields}
    )
    assert set(table) == expected
    facts = table["bands.t20_s"]
    assert facts.quantity == "時間"
    assert facts.unit == "s"
    assert facts.validity == "estimable"
    assert "視窗" in facts.reference


def test_quantity_table_separates_estimated_from_unestimated() -> None:
    """不可估的欄位在對照表上要看得出「不可估」，而不是跟可估的混在一起。"""
    table = report_io.quantity_table()
    assert table["bands.fem_energy"].validity.startswith("unestimable")
    assert table["points.fem_energy"].validity.startswith("unestimable")
    assert table["bands.t20_s"].validity == "estimable"


# ── ② 輸入模型每一條驗證規則各一題 ──────────────────────────────────────────────
def test_accepts_a_complete_document() -> None:
    """合法輸入要收得下，而且收回來的是求解層吃的形狀。"""
    inputs = report_io.load_input_document(_input_document())
    assert inputs.room_m == Room(6.0, 4.0, 3.0)
    assert inputs.source_m == Point(1.2, 1.3, 1.1)
    solved = report_io.solver_inputs(inputs)
    assert set(solved.impedance_by_wall) == set(Wall.all())
    assert set(solved.scattering_by_wall or {}) == set(Wall.all())


def test_scattering_may_be_omitted_entirely() -> None:
    """散射整格可省略，省略時是 None（由幾何路套既有預設值）。"""
    document = _input_document()
    del document["scattering_by_wall"]
    assert report_io.load_input_document(document).scattering_by_wall is None


def test_scattering_may_be_zero_on_every_wall() -> None:
    """散射 0 是合法輸入，不准被「正實數」那條擋掉。"""
    inputs = report_io.load_input_document(
        _input_document(scattering_by_wall={wall: 0.0 for wall in _WALL_NAMES})
    )
    assert inputs.scattering_by_wall == {wall: 0.0 for wall in _WALL_NAMES}


def test_missing_room_component_is_named() -> None:
    """房少一軸要指名哪一軸（Pydantic 的欄位路徑帶出來）。"""
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
    inputs = report_io.load_input_document(_input_document())
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
    assert len(parsed.bands) == len(ReportOutput.model_validate_json(output).bands)
    assert (parsed.points is not None) is bool(extra)
    assert all(
        band.t20_s is None or band.t20_unavailable_reason is None
        for band in parsed.bands
    )


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
