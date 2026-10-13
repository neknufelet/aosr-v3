"""報表輸出契約：空值原因、頻帶順序、能力範圍與凍結模型。"""
from __future__ import annotations

import pytest
import json

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, report_output
from aosr.physics.report_io import BandRow, CapabilitySection, PointRow, ReportOutput, SceneSection, TopFields
from aosr.physics.report_source import SourceModelKind, SourceModelSection, SourceModelSpec
from tests.engine import _directivity
from tests.engine.test_report_io_contract import _three_lane_frequency_range, _fake_fem_energy, _input_document, _table
from tests.engine._material_flow_cases import fixed_materials as _flow_materials


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
        "scene": SceneSection(
            source_model=SourceModelSection.from_spec(
                SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL), None,
            ),
            scene_fingerprint="a" * 64,
            source_m=Point(1.2, 1.3, 1.1),
            receiver_m=Point(4.7, 2.8, 1.4),
        ),
        "capability": CapabilitySection(
            frequency_hz=_three_lane_frequency_range(),
            outputs=("total_energy",),
            status="experimental",
            evidence=(),
        ),
        "top": TopFields(
            f_s_hz=200.0,
            crossover_lower_hz=200.0,
            crossover_upper_hz=300.0,
            capped_by_upper_limit=False,
            reflection_order_k=REFLECTION_ORDER_K,
            low_frequency_axis=LowFrequencyAxis.SEARCH,
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
    """沒查表時 ``outputs`` 與 ``frequency_hz`` 都是空的，不准造假的欄名與頻率範圍。

    ``capability_report.capability_line`` 那一層印的是 ``outputs=none`` 給人看的一行；
    契約這一層是資料，空就是空——造一個字串欄名會讓前端以為真有那一欄，造一個
    ``(0.0, 0.0)``（第四刀非必修第 8 條）會讓前端以為宣告了 0 到 0 Hz。
    """
    from aosr.physics import three_lane_report

    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    report = three_lane_report.solve_three_lane_report(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=Room(6.0, 4.0, 3.0),
        source=Point(1.2, 1.3, 1.1),
        receiver=Point(4.7, 2.8, 1.4),
        sound_speed_m_s=343.0,
        density_kg_m3=1.2,
        impedance_by_wall={wall: 4.0 * 411.6 for wall in Wall.all()},
        materials_by_wall=_flow_materials({wall: 4.0 * 411.6 for wall in Wall.all()}))
    inputs = report_io.load_input_document(_input_document(), _table(), _directivity.DIRECTIVITY)
    section = report_output.output_from_report(
        report, inputs=inputs, with_points=False
    ).capability
    assert section.outputs == ()
    assert section.frequency_hz == ()
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
                reflection_order_k=REFLECTION_ORDER_K,
                low_frequency_axis=LowFrequencyAxis.SEARCH,
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


def test_json_payload_carries_field_names() -> None:
    """``--format json`` 的形狀（``model_dump``）是有欄名的物件，不是位置陣列。"""
    payload = json.loads(_output().model_dump_json())
    assert isinstance(payload["bands"], list)
    first = payload["bands"][0]
    assert isinstance(first, dict)
    assert "geometric_energy" in first
    assert isinstance(payload["top"], dict)
    assert "room_volume_m3" in payload["top"]

