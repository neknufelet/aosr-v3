"""票 #351：補算時間窗的幾何、逐路徑與主報表隔離考卷。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import load_capabilities
from aosr.config.frequency_axis import (
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
)
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, three_lane_report
from aosr.physics.late_decay import LateDecayBand, LateDecayResult
from aosr.physics.report_output import output_from_report
from aosr.physics.report_path_table import PathTableData, build_path_table
from aosr.physics.reflection_window import ReflectionWindow, build_reflection_window
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER, image_source_paths


_FREQUENCIES = (125.0, 250.0)
_SCATTERING = (0.2, 0.3)
_REFERENCE = {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0}
_SMALL = {"Lx": 4.5, "Ly": 3.5, "Lz": 2.6}
_IMPEDANCE = {
    "floor": 900.0, "ceiling": 1300.0, "x0": 1646.4,
    "xL": 2500.0, "y0": 4000.0, "yL": 6000.0,
}


def _row_key(row: object) -> tuple[object, ...]:
    """整列比：階數、牆序列、延遲、方向（向量與兩個角）、逐頻能量。"""
    angles = getattr(row, "direction_angles")
    return (
        getattr(row, "order"), getattr(row, "wall_sequence"), getattr(row, "delay_s"),
        getattr(row, "direction_vector"), angles.azimuth_deg, angles.elevation_deg,
        getattr(row, "relative_direct_energy"),
    )


def _first_relative_by_order(inputs: report_io.ReportInput) -> dict[int, float]:
    """定義本身：同一次 image_source_paths 裡每一階最早那一條的 ``delay_s − 直達.delay_s``。"""
    paths = image_source_paths(
        inputs.room_m, inputs.source_m, inputs.receiver_m, inputs.sound_speed_m_s,
        max_order=SUPPORTED_MAX_ORDER, materials=None,
    )
    direct = next(path.delay_s for path in paths if path.order == 0)
    return {
        order: min(path.delay_s for path in paths if path.order == order) - direct
        for order in range(1, SUPPORTED_MAX_ORDER + 1)
    }


def _definitional_order(
    first: Mapping[int, float], report_k: int, window_s: float,
) -> tuple[int, str]:
    """照規則逐階看（不經過被測的迴圈）：第 K′+1 階在窗外就停；補到上限用第上限階當下界。"""
    order = report_k
    while order < SUPPORTED_MAX_ORDER:
        if first[order + 1] > window_s:
            return order, "complete"
        order += 1
    return order, "complete" if first[order] > window_s else "not_provable"


def _inputs(
    room: dict[str, float], *, sound_speed: float = 343.0, order: int = 3,
) -> report_io.ReportInput:
    return report_io.load_input_document({
        "room_m": room,
        "source_m": {"x": 1.0, "y": 1.3 if room == _REFERENCE else 2.2, "z": 1.2},
        "receiver_m": {"x": 3.2, "y": 1.9, "z": 1.2},
        "sound_speed_m_s": sound_speed,
        "density_kg_m3": 1.2,
        # 六面各不相同：牆對錯、阻抗對錯才看得出來（找碴席 09-24：全一樣時倒序也不紅）
        "impedance_pa_s_per_m_by_wall": dict(_IMPEDANCE),
        "reflection_order_k": order,
    }, load_capabilities(config_path("capabilities.toml")))


def _earliest_image_distance(inputs: report_io.ReportInput, order: int) -> float:
    """獨立鏡像公式：每軸像點 (1−2p)s+2nL，撞牆數 |2n−p|。"""
    lengths = (inputs.room_m.Lx, inputs.room_m.Ly, inputs.room_m.Lz)
    axes: list[list[tuple[int, float]]] = []
    for length, source, receiver in zip(
        lengths, inputs.source_m.as_tuple(), inputs.receiver_m.as_tuple(), strict=True,
    ):
        axes.append([
            (abs(2 * n - p), (1 - 2 * p) * source + 2 * n * length - receiver)
            for n in range(-order - 1, order + 2)
            for p in (0, 1)
            if abs(2 * n - p) <= order
        ])
    return min(
        math.sqrt(dx * dx + dy * dy + dz * dz)
        for ox, dx in axes[0]
        for oy, dy in axes[1]
        for oz, dz in axes[2]
        if ox + oy + oz == order
    )


def _expected_order(inputs: report_io.ReportInput, window_s: float) -> tuple[int, str]:
    offsets = tuple(
        source - receiver for source, receiver in zip(
            inputs.source_m.as_tuple(), inputs.receiver_m.as_tuple(), strict=True,
        )
    )
    direct_delay = math.sqrt(sum(offset * offset for offset in offsets)) / inputs.sound_speed_m_s
    for order in range(inputs.reflection_order_k + 1, SUPPORTED_MAX_ORDER + 1):
        relative = _earliest_image_distance(inputs, order) / inputs.sound_speed_m_s - direct_delay
        if relative > window_s:
            return order - 1, "complete"
    return SUPPORTED_MAX_ORDER, "not_provable"


def _window(inputs: report_io.ReportInput, window_s: float) -> ReflectionWindow:
    return build_reflection_window(
        inputs, frequencies_hz=_FREQUENCIES,
        scattering_coefficient=_SCATTERING, window_s=window_s,
    )


def _table(inputs: report_io.ReportInput, order: int) -> PathTableData:
    solved = report_io.solver_inputs(inputs)
    return build_path_table(
        room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.density_kg_m3 * solved.sound_speed_m_s,
        impedance_by_wall=solved.impedance_by_wall,
        frequencies_hz=_FREQUENCIES, scattering_coefficient=_SCATTERING,
        reflection_order_k=order,
    )


def test_reference_room_proves_window_without_extra_paths() -> None:
    inputs = _inputs(_REFERENCE)
    result = _window(inputs, 0.015)
    expected, coverage = _expected_order(inputs, 0.015)

    assert result.computed_order_k == expected == inputs.reflection_order_k
    assert result.coverage == coverage == "complete"
    assert result.validation == "validated"
    assert not result.rows
    assert result.scene_fingerprint == report_io.scene_fingerprint(inputs)


def test_small_room_adds_fourth_order_and_marks_its_numeric_scope() -> None:
    inputs = _inputs(_SMALL)
    result = _window(inputs, 0.015)
    expected, coverage = _expected_order(inputs, 0.015)

    assert result.computed_order_k == expected == inputs.reflection_order_k + 1
    assert result.coverage == coverage == "complete"
    assert result.validation == "unvalidated"
    assert result.rows
    assert all(row.order == expected for row in result.rows)
    assert all(row.delay_s - result.direct_delay_s <= result.window_s for row in result.rows)


def test_window_size_controls_order_instead_of_a_room_specific_cap() -> None:
    inputs = _inputs(_SMALL)
    windows = (0.005, 0.015, 0.020)
    results = tuple(_window(inputs, window) for window in windows)

    assert tuple(result.computed_order_k for result in results) == tuple(
        _expected_order(inputs, window)[0] for window in windows
    )
    assert results[0].computed_order_k == inputs.reflection_order_k
    assert results[0].computed_order_k < results[1].computed_order_k < results[2].computed_order_k


def test_every_extra_path_on_the_window_boundary_is_included() -> None:
    """小房間第 4、5 階每一條都輪流當一次窗上限：那一條一定收進補算列，補到的階數照規則。"""
    inputs = _inputs(_SMALL)
    first = _first_relative_by_order(inputs)
    paths = image_source_paths(
        inputs.room_m, inputs.source_m, inputs.receiver_m, inputs.sound_speed_m_s,
        max_order=inputs.reflection_order_k + 2, materials=None,
    )
    direct = next(path.delay_s for path in paths if path.order == 0)
    boundary_paths = [path for path in paths if path.order > inputs.reflection_order_k]
    assert boundary_paths
    for path in boundary_paths:
        boundary = path.delay_s - direct
        result = _window(inputs, boundary)
        assert (result.computed_order_k, result.coverage) == _definitional_order(
            first, inputs.reflection_order_k, boundary
        )
        assert any(
            row.order == path.order and row.delay_s == path.delay_s for row in result.rows
        ), f"剛好在窗上限的第 {path.order} 階那一條沒有收進來"


@pytest.mark.parametrize("below_s", [0.0, 0.0002])
def test_window_just_below_the_next_order_stops_at_the_report_order(below_s: float) -> None:
    """窗比第 K+1 階最早那一條小一格最小浮點差（或 0.2 ms）：要停在 K、不補算、仍算已驗證。"""
    inputs = _inputs(_SMALL)
    edge = _first_relative_by_order(inputs)[inputs.reflection_order_k + 1]
    window_s = math.nextafter(edge - below_s, -math.inf)
    result = _window(inputs, window_s)

    assert result.computed_order_k == inputs.reflection_order_k
    assert result.coverage == "complete"
    assert result.validation == "validated"
    assert not result.rows
    assert result.next_uncomputed_earliest_relative_s == edge


def test_window_records_the_inputs_it_was_built_from() -> None:
    inputs = _inputs(_SMALL)
    result = _window(inputs, 0.015)

    assert result.source_m == inputs.source_m
    assert result.receiver_m == inputs.receiver_m
    assert result.report_order_k == inputs.reflection_order_k
    assert result.window_s == 0.015
    assert result.frequencies_hz == _FREQUENCIES
    assert result.scattering_coefficient == _SCATTERING
    assert result.direct_delay_s == pytest.approx(
        math.dist(inputs.source_m.as_tuple(), inputs.receiver_m.as_tuple()) / inputs.sound_speed_m_s
    )


def test_report_starting_at_the_supported_limit_can_still_prove_the_window() -> None:
    """主報表一開始就是第 8 階：第 9 階算不了，拿第 8 階最早那一條當下界；在窗外就照樣證明完整。"""
    inputs = _inputs(_REFERENCE, order=SUPPORTED_MAX_ORDER)
    first = _first_relative_by_order(inputs)
    proven = _window(inputs, 0.015)
    unprovable = _window(inputs, first[SUPPORTED_MAX_ORDER])

    assert (proven.computed_order_k, proven.coverage) == (SUPPORTED_MAX_ORDER, "complete")
    assert proven.next_uncomputed_earliest_relative_s == first[SUPPORTED_MAX_ORDER]
    assert not proven.rows
    assert (unprovable.computed_order_k, unprovable.coverage) == (SUPPORTED_MAX_ORDER, "not_provable")
    assert unprovable.next_uncomputed_earliest_relative_s is None


def test_large_window_reports_unprovable_at_supported_limit() -> None:
    inputs = _inputs(_SMALL)
    result = _window(inputs, 1.0)

    assert result.computed_order_k == _expected_order(inputs, 1.0)[0] == SUPPORTED_MAX_ORDER
    assert result.coverage == "not_provable"
    assert result.next_uncomputed_earliest_relative_s is None
    assert result.validation == "unvalidated"


@pytest.mark.parametrize(("window_s", "expected_validation"), [(0.012, "validated"), (0.015, "unvalidated")])
def test_lower_report_k_is_validated_only_while_extension_stays_within_order_three(
    window_s: float, expected_validation: str,
) -> None:
    """主報表 K＝2：補到第 3 階還在有考卷守的範圍（已驗證）；補到第 4 階就是未驗證。"""
    inputs = _inputs(_SMALL, order=2)
    result = _window(inputs, window_s)
    expected_order, coverage = _definitional_order(_first_relative_by_order(inputs), 2, window_s)

    assert (result.computed_order_k, result.coverage) == (expected_order, coverage)
    assert result.computed_order_k > inputs.reflection_order_k
    assert result.validation == expected_validation
    assert (result.computed_order_k <= 3) == (expected_validation == "validated")


@pytest.mark.parametrize("room", (_REFERENCE, _SMALL))
def test_auto_window_equals_full_order_table_clipped_to_window(room: dict[str, float]) -> None:
    inputs = _inputs(room)
    result = _window(inputs, 0.015)
    full = _table(inputs, SUPPORTED_MAX_ORDER)
    ordinary = _table(inputs, inputs.reflection_order_k)
    direct = next(row for row in full.rows if row.order == 0)
    expected = tuple(
        _row_key(row) for row in full.rows if row.delay_s - direct.delay_s <= result.window_s
    )
    actual = tuple(
        _row_key(row) for row in ordinary.rows if row.delay_s - direct.delay_s <= result.window_s
    ) + tuple(
        _row_key(row) for row in result.rows
    )

    assert actual == expected


def test_wall_intersection_path_is_kept_in_extra_rows_and_full_table() -> None:
    """票 #305：同一反彈點碰 x0、y0，牆序列與階數在兩種截窗路徑一致。"""
    document = _inputs(_SMALL, order=1).model_dump(mode="json")
    document.update({
        "room_m": {"Lx": 4.0, "Ly": 4.0, "Lz": 3.0},
        "source_m": {"x": 1.0, "y": 1.0, "z": 1.5},
        "receiver_m": {"x": 2.0, "y": 2.0, "z": 1.5},
    })
    inputs = report_io.load_input_document(
        document, load_capabilities(config_path("capabilities.toml")),
    )
    window = _window(inputs, 0.015)
    full = _table(inputs, SUPPORTED_MAX_ORDER)
    ordinary = _table(inputs, inputs.reflection_order_k)
    direct = next(row for row in full.rows if row.order == 0)
    expected = tuple(
        _row_key(row) for row in full.rows if row.delay_s - direct.delay_s <= window.window_s
    )
    actual = tuple(
        _row_key(row) for row in ordinary.rows if row.delay_s - direct.delay_s <= window.window_s
    ) + tuple(
        _row_key(row) for row in window.rows
    )

    assert window.computed_order_k == _expected_order(inputs, window.window_s)[0]
    assert any(
        row.order == 2 and row.wall_sequence in (("x0", "y0"), ("y0", "x0"))
        for row in window.rows
    )
    assert actual == expected


def _fake_fem_energy(
    *, room: Room, source: Point, receiver: Point,
    wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
    density_kg_m3: float, sound_speed_m_s: float,
) -> tuple[float, ...]:
    del room, source, receiver, wall_impedances, density_kg_m3, sound_speed_m_s
    return tuple(frequency * 3.0 + 7.0 for frequency in frequencies_hz)


def _fast_late_decay(
    *, room: Room, wall_impedances: Mapping[Wall, float],
    rho_c_pa_s_per_m: float, sound_speed_m_s: float,
) -> three_lane_report._ReportLateDecay:
    del room, wall_impedances, rho_c_pa_s_per_m, sound_speed_m_s
    root_two = math.sqrt(2.0)
    frequencies = tuple(
        frequency for frequency in GEOMETRIC_LANE_FREQUENCIES_HZ
        if any(
            center / root_two <= frequency < center * root_two
            for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
        )
    )
    bands = tuple(
        LateDecayBand(
            frequency_hz=frequency, t20_s=frequency / 1000.0,
            collision_frequency_hz=1.0, slope_db_per_s=-1.0,
            soft_weight_sum=1.0, fell_back_to_perron=False, perron_t60_s=1.0,
            t30_s=frequency / 500.0, t30_slope_db_per_s=-1.0,
            t30_soft_weight_sum=1.0,
        )
        for frequency in frequencies
    )
    return three_lane_report._ReportLateDecay(
        result=LateDecayResult(orders_used=1, bands=bands), unavailable_by_center_hz={},
    )


def test_window_does_not_change_report_or_its_path_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", _fast_late_decay)
    inputs = _inputs(_SMALL)
    solved = report_io.solver_inputs(inputs)

    def solve() -> report_io.ReportOutput:
        """每次都重新算一份報表，不是同一個物件再存一次。"""
        report = three_lane_report.solve_three_lane_report(
            room=solved.room, source=solved.source, receiver=solved.receiver,
            sound_speed_m_s=solved.sound_speed_m_s, density_kg_m3=solved.density_kg_m3,
            impedance_by_wall=solved.impedance_by_wall,
            scattering_by_wall=solved.scattering_by_wall,
            reflection_order_k=solved.reflection_order_k,
            low_frequency_axis=solved.low_frequency_axis,
        )
        return output_from_report(report, inputs=inputs, with_points=True, path_table_inputs=solved)

    before = solve()
    assert before.path_table is not None
    window = build_reflection_window(
        inputs, frequencies_hz=before.path_table.frequencies_hz,
        scattering_coefficient=before.path_table.scattering_coefficient, window_s=0.015,
    )
    after = solve()

    assert window.rows
    assert before.model_dump(mode="json") == after.model_dump(mode="json")
    assert before.path_table is not None
    assert before.path_table.reflection_order_k == inputs.reflection_order_k
    assert all(row.order <= inputs.reflection_order_k for row in before.path_table.rows)


def test_model_rejects_inconsistent_coverage_rows_and_validation() -> None:
    """每一種不一致各自被它那一道檢查擋下（比對訊息，不讓別的檢查代打）。"""
    result = _window(_inputs(_SMALL), 0.015)
    document = result.model_dump(mode="python")
    bad: list[tuple[dict[str, object], str]] = [
        ({"coverage": "complete", "next_uncomputed_earliest_relative_s": None}, "支援上限"),
        ({"next_uncomputed_earliest_relative_s": result.window_s}, "窗外證明"),
        ({"coverage": "not_provable"}, "窗外證明"),
        ({"validation": "validated"}, "數值驗證範圍"),
        ({"rows": ({**document["rows"][0], "order": result.report_order_k},)}, "補算範圍"),
        ({"rows": ({**document["rows"][0], "order": result.computed_order_k + 1},)}, "補算範圍"),
        (
            {"rows": ({**document["rows"][0], "delay_s": result.direct_delay_s + result.window_s + 1.0},)},
            "超過時間窗",
        ),
        ({"rows": ({**document["rows"][0], "relative_direct_energy": (0.5,)},)}, "逐頻長度"),
        ({"window_s": float("nan")}, "finite"),
        ({"computed_order_k": SUPPORTED_MAX_ORDER - 1, "next_uncomputed_earliest_relative_s": None}, "只有補到支援上限"),
        ({"frequencies_hz": (250.0, 125.0)}, "遞增正頻率"),
        ({"frequencies_hz": (-125.0, 250.0)}, "遞增正頻率"),
        ({"scattering_coefficient": (0.2,)}, "合法範圍"),
        ({"scattering_coefficient": (0.2, 1.5)}, "合法範圍"),
        ({"report_order_k": 0}, "greater than or equal"),
        ({"computed_order_k": SUPPORTED_MAX_ORDER + 1}, "less than or equal"),
        ({"direct_delay_s": -0.001}, "greater than or equal"),
        (
            {"rows": ({
                **document["rows"][0],
                "delay_s": math.nextafter(result.direct_delay_s + result.window_s, math.inf),
            },)},
            "超過時間窗",
        ),
        ({"source_m": Point(math.inf, 1.0, 1.0)}, "座標必須是有限數"),
        ({"receiver_m": {"x": math.nan, "y": 1.0, "z": 1.0}}, "finite"),
        ({"unexpected": "field"}, "unexpected"),
    ]
    for change, message in bad:
        with pytest.raises(ValidationError, match=message):
            ReflectionWindow.model_validate({**document, **change})


def test_model_rejects_computed_order_below_the_report_order() -> None:
    """參考房沒補算（算到 3、沒有補算列、覆蓋完整）；只把主報表 K 改成 4，剩這一道會擋。"""
    document = _window(_inputs(_REFERENCE), 0.015).model_dump(mode="python")
    with pytest.raises(ValidationError, match="不可小於"):
        ReflectionWindow.model_validate({**document, "report_order_k": 4})


def test_sound_speed_changes_relative_delays_and_window_order() -> None:
    fast = _inputs(_SMALL)
    slow = _inputs(_SMALL, sound_speed=320.0)
    fast_window = _window(fast, 0.015)
    slow_window = _window(slow, 0.015)

    assert fast_window.computed_order_k == _expected_order(fast, 0.015)[0]
    assert slow_window.computed_order_k == _expected_order(slow, 0.015)[0]
    assert fast_window.computed_order_k > slow_window.computed_order_k
    assert fast_window.next_uncomputed_earliest_relative_s is not None
    assert slow_window.next_uncomputed_earliest_relative_s is not None
    fourth_distance = _earliest_image_distance(fast, fast.reflection_order_k + 1)
    direct_distance = math.dist(fast.source_m.as_tuple(), fast.receiver_m.as_tuple())
    fast_relative = fourth_distance / fast.sound_speed_m_s - direct_distance / fast.sound_speed_m_s
    slow_relative = fourth_distance / slow.sound_speed_m_s - direct_distance / slow.sound_speed_m_s
    assert fast_relative <= fast_window.window_s < slow_relative
    assert any(
        row.delay_s - fast_window.direct_delay_s == pytest.approx(fast_relative)
        for row in fast_window.rows
    )
    assert slow_window.next_uncomputed_earliest_relative_s == pytest.approx(slow_relative)
    assert slow_window.next_uncomputed_earliest_relative_s == pytest.approx(
        (_earliest_image_distance(slow, slow_window.computed_order_k + 1)
         - math.dist(slow.source_m.as_tuple(), slow.receiver_m.as_tuple()))
        / slow.sound_speed_m_s
    )
