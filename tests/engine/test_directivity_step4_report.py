"""#505 第四步：報表、晚期倍率與既有答案的數值考卷。"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import cast

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import TwoParameterCurve
from aosr.config.frequency_axis import GEOMETRIC_BAND_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import geometric_lane, report_io, report_output, three_lane_report, three_lane_report_batch
from aosr.physics.geometric_lane import GeometricLaneResult, solve_geometric_early_lane, solve_geometric_lane
from aosr.physics.late_energy import LateEnergyOrderResult
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import ReflectionWindow
from aosr.physics import room_paths
from aosr.physics.room_paths import RoomPath
from aosr.physics.report_source import (
    AnalyticAxisymmetricInput, SourceModelKind, SourceModelSection, SourceModelSpec, default_source_model,
)
from aosr.physics.source_directivity import two_parameter_power_ratio
from tests.engine import _source_model_control as control
from tests.engine._directivity import DIRECTIVITY
from tests.engine._report_cache import shared_control_omnidirectional_report, shared_report


def _document(analytic: bool, *, aim: Point = Point(4.0, 2.0, 1.0)) -> dict[str, object]:
    document = deepcopy(control.SCENE_WITHOUT_SOURCE_MODEL)
    document["source_model"] = (
        default_source_model(aim, DIRECTIVITY).model_dump(mode="json")
        if analytic else {"kind": "omnidirectional"}
    )
    return document


def _inputs(analytic: bool, *, aim: Point = Point(4.0, 2.0, 1.0)) -> report_io.ReportInput:
    return report_io.load_input_document(
        _document(analytic, aim=aim), load_capabilities(config_path("capabilities.toml")), DIRECTIVITY,
    )


def _report(inputs: report_io.ReportInput, monkeypatch: pytest.MonkeyPatch) -> three_lane_report.ThreeLaneReport:
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", control.fake_fem_energy)
    monkeypatch.setattr(three_lane_report_batch, "solve_geometric_late_energy", control.fake_late_energy)
    return three_lane_report.solve_three_lane_report(**report_io.solver_inputs(inputs)._asdict())


def _report_with_fast_decay(
    inputs: report_io.ReportInput, monkeypatch: pytest.MonkeyPatch,
) -> three_lane_report.ThreeLaneReport:
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", control.fast_late_decay)
    return _report(inputs, monkeypatch)


def _shared_real_pair(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str, monkeypatch: pytest.MonkeyPatch,
) -> tuple[three_lane_report.ThreeLaneReport, three_lane_report.ThreeLaneReport]:
    # 這兩題讀相同全向與解析報表，比的是衰減相等及能力欄位；每次 pytest 重新求解。
    return shared_report(tmp_path_factory, worker_id, "step4-real-pair",
                         lambda: (_report(_inputs(False), monkeypatch), _report(_inputs(True), monkeypatch)))


def _assert_hex_fields(actual: dict[str, object], expected: dict[str, object]) -> None:
    for name, answer in expected.items():
        value = actual[name]
        if isinstance(answer, str) and answer.startswith(("0x", "-0x")):
            assert isinstance(value, float)
            assert value.hex() == answer, name
        else:
            assert value == answer, name


def test_degenerate_analytic_curve_matches_frozen_omni_hex(monkeypatch: pytest.MonkeyPatch) -> None:
    document = _document(True)
    model = document["source_model"]
    assert isinstance(model, dict)
    parameters = model["parameters"]
    assert isinstance(parameters, dict)
    parameters["beta_limit"] = 0.0
    parameters["power_floor_limit_db"] = 0.0
    inputs = report_io.load_input_document(
        document, load_capabilities(config_path("capabilities.toml")), DIRECTIVITY,
    )
    assert isinstance(inputs.source_model, AnalyticAxisymmetricInput)
    assert inputs.receiver_m != inputs.source_model.aim_m
    report = _report_with_fast_decay(inputs, monkeypatch)
    output = report_output.output_from_report(
        report, inputs=inputs, with_points=True, path_table_inputs=report_io.solver_inputs(inputs),
    )
    answers = control.ANSWERS
    assert output.points is not None and output.path_table is not None
    _assert_hex_fields(output.top.model_dump(), cast(dict[str, object], answers["top"]))
    for index, fields in cast(dict[int, dict[str, object]], answers["points"]).items():
        _assert_hex_fields(output.points[index].model_dump(), fields)
    for band, fields in zip(output.bands, cast(list[dict[str, object]], answers["bands"]), strict=True):
        _assert_hex_fields(band.model_dump(), fields)
    for index, fields in cast(dict[int, dict[str, object]], answers["path_rows"]).items():
        path_row = output.path_table.rows[index]
        assert path_row.order == fields["order"]
        for frequency_index, answer in cast(dict[int, str], fields["relative_direct_energy"]).items():
            assert path_row.relative_direct_energy[frequency_index].hex() == answer


def test_real_late_decay_and_crossover_stay_bitwise_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    omni_inputs, analytic_inputs = _inputs(False), _inputs(True)
    omni, analytic = _shared_real_pair(tmp_path_factory, worker_id, monkeypatch)
    assert analytic.late_decay == omni.late_decay
    assert analytic.eyring_t60_by_band_s == omni.eyring_t60_by_band_s
    assert analytic.f_s_hz == omni.f_s_hz
    assert analytic.full_axis_weights == omni.full_axis_weights
    assert tuple((band.t20_s, band.t30_s) for band in analytic.bands) == tuple(
        (band.t20_s, band.t30_s) for band in omni.bands
    )
    assert tuple((point.w_fem, point.w_geo) for point in analytic.points) == tuple(
        (point.w_fem, point.w_geo) for point in omni.points
    )
    plain_screen = build_reflection_screen(omni_inputs, (1000.0, 4000.0))
    directed_screen = build_reflection_screen(analytic_inputs, (1000.0, 4000.0))
    assert plain_screen.pairs == directed_screen.pairs


def test_report_axis_stays_fixed_for_surrounding_seat_and_aim_changes_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    primary = _inputs(True, aim=Point(4.35, 2.45, 1.05))
    moved_document = _document(True, aim=Point(4.35, 2.45, 1.05))
    moved_document["receiver_m"] = {"x": 3.8, "y": 2.1, "z": 1.05}
    surrounding = report_io.load_input_document(
        moved_document, load_capabilities(config_path("capabilities.toml")), DIRECTIVITY,
    )
    # 主位報表的輸入跟手算 g 那題的解析主位逐格相同，那一份共用；周圍座位的輸入不同，
    # 照舊由本題自己求解——產品把解析近似改對準當下座位時，周圍座位的報表會跟輸入對不上而紅。
    primary_output = report_output.output_from_report(
        shared_report(tmp_path_factory, worker_id, "step4-axis-primary",
                      lambda: _report_with_fast_decay(primary, monkeypatch)),
        inputs=primary, with_points=False,
    )
    surrounding_output = report_output.output_from_report(
        _report_with_fast_decay(surrounding, monkeypatch), inputs=surrounding, with_points=False,
    )
    assert primary_output.scene.source_model.axis_unit_vector == surrounding_output.scene.source_model.axis_unit_vector
    assert primary_output.scene.scene_fingerprint == surrounding_output.scene.scene_fingerprint
    reaimed = _inputs(True, aim=Point(3.8, 2.1, 1.05))
    assert report_io.scene_fingerprint(reaimed) != report_io.scene_fingerprint(primary)
    reaimed_section = SourceModelSection.from_spec(report_io.solver_inputs(reaimed).source_model, reaimed.source_m)
    assert reaimed_section.axis_unit_vector != primary_output.scene.source_model.axis_unit_vector


def _late_case(
    solved: report_io.SolverInputs, model: SourceModelSpec, scatter: float,
    frequencies: tuple[float, ...], late_result: LateEnergyOrderResult,
) -> GeometricLaneResult:
    return solve_geometric_lane(
        source_model=model, room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.sound_speed_m_s * solved.density_kg_m3,
        frequencies_hz=frequencies,
        impedance_by_wall={wall.wall_name(): complex(solved.impedance_by_wall[wall]) for wall in Wall.all()},
        scattering_by_wall={wall.wall_name(): scatter for wall in Wall.all()},
        reflection_order_k=solved.reflection_order_k, late_result=late_result,
    )


@pytest.mark.parametrize("scatter", (0.0, 1.0))
def test_late_energy_is_multiplied_by_hand_computed_g_once(scatter: float) -> None:
    solved = report_io.solver_inputs(_inputs(True))
    frequencies = (300.0, 1000.0, 4000.0)
    late_result = control.fake_late_energy(
        room=solved.room, rho_c_pa_s_per_m=solved.sound_speed_m_s * solved.density_kg_m3,
        frequencies_hz=frequencies, impedance_by_wall={}, reflection_order_k=solved.reflection_order_k,
    )
    omni = _late_case(solved, SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL), scatter, frequencies, late_result)
    analytic = _late_case(solved, solved.source_model, scatter, frequencies, late_result)
    assert all(value == scatter for value in analytic.scattering)
    assert solved.source_model.parameters is not None
    curve = solved.source_model.parameters
    for frequency, plain, directed in zip(frequencies, omni.late_energy, analytic.late_energy, strict=True):
        beta = curve.beta_limit / (1.0 + (curve.beta_corner_hz / frequency) ** curve.beta_exponent)
        db = curve.power_floor_limit_db / (1.0 + (curve.power_floor_corner_hz / frequency) ** curve.power_floor_exponent)
        floor = 10.0 ** (db / 10.0)
        g = (1.0 - floor) * (1.0 - math.exp(-4.0 * beta)) / (4.0 * beta) + floor
        assert directed == pytest.approx(plain * g, rel=1e-12)
    # g 只准乘在晚期那一格：早期三欄逐位等於直接解早期路，四欄相加仍是幾何能量。
    early = solve_geometric_early_lane(
        source_model=solved.source_model, room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.sound_speed_m_s * solved.density_kg_m3, frequencies_hz=frequencies,
        impedance_by_wall={wall.wall_name(): complex(solved.impedance_by_wall[wall]) for wall in Wall.all()},
        scattering_by_wall={wall.wall_name(): scatter for wall in Wall.all()},
        reflection_order_k=solved.reflection_order_k,
    )
    assert analytic.direct_energy == early.direct_energy
    assert analytic.reflected_energy == early.reflected_energy
    assert analytic.interference_energy == early.interference_energy
    for index, total in enumerate(analytic.geometric_energy):
        parts = (analytic.direct_energy[index] + analytic.reflected_energy[index]
                 + analytic.interference_energy[index] + analytic.late_energy[index])
        assert total == pytest.approx(parts, rel=1e-12)


def test_analytic_scene_rejects_axis_from_another_source() -> None:
    inputs = _inputs(True)
    spec = report_io.solver_inputs(inputs).source_model
    section = SourceModelSection.from_spec(spec, inputs.source_m)
    assert section.axis_unit_vector is not None
    document = {
        "scene_fingerprint": report_io.scene_fingerprint(inputs),
        "source_m": inputs.source_m, "receiver_m": inputs.receiver_m,
        "source_model": section.model_copy(update={"axis_unit_vector": (1.0, 0.0, 0.0)}),
    }
    with pytest.raises(ValidationError, match="axis_unit_vector"):
        report_io.SceneSection.model_validate(document)


def test_reflection_window_rejects_analytic_row_without_off_axis_angle() -> None:
    omni = report_io.PathRow.model_validate({
        "order": 4, "wall_sequence": ("x0", "xL", "x0", "xL"), "delay_s": 0.01, "distance_m": 3.43,
        "direction_vector": (1.0, 0.0, 0.0),
        "direction_angles": {"azimuth_deg": 0.0, "elevation_deg": 0.0},
        "departure_off_axis_deg": None, "relative_direct_energy": (0.5,),
    })
    inputs = _inputs(True)
    base = dict(
        scene_fingerprint=report_io.scene_fingerprint(inputs),
        source_m=inputs.source_m, receiver_m=inputs.receiver_m,
        report_order_k=3, window_s=0.02, direct_delay_s=0.005, computed_order_k=4,
        coverage="complete", validation="unvalidated", next_uncomputed_earliest_relative_s=0.03,
        frequencies_hz=(1000.0,), scattering_coefficient=(0.2,),
    )
    with pytest.raises(ValidationError, match="departure_off_axis_deg"):
        ReflectionWindow.model_validate({
            **base, "source_model_kind": SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1,
            "rows": (omni,),
        })
    with pytest.raises(ValidationError, match="departure_off_axis_deg"):
        ReflectionWindow.model_validate({
            **base, "source_model_kind": SourceModelKind.OMNIDIRECTIONAL,
            "rows": (omni.model_copy(update={"departure_off_axis_deg": 30.0}),),
        })


def test_bare_curve_can_exceed_registry_max_but_rejects_nonphysical_domain() -> None:
    curve = DIRECTIVITY.two_parameter
    above = curve.model_copy(update={"beta_limit": DIRECTIVITY.allowed_range.beta_max + 1.0})
    assert two_parameter_power_ratio((1000.0,), above)[0] > 0.0
    for field, invalid in (("beta_limit", -1.0), ("power_floor_limit_db", 1.0)):
        broken = curve.model_copy(update={field: invalid})
        with pytest.raises(ValueError, match="1000.0 Hz") as caught:
            two_parameter_power_ratio((1000.0,), broken)
        assert ("beta" if field == "beta_limit" else "power_floor_db") in str(caught.value)
        assert "值" in str(caught.value)


def _hand_g(curve: TwoParameterCurve, frequency: float) -> float:
    beta = curve.beta_limit / (1.0 + (curve.beta_corner_hz / frequency) ** curve.beta_exponent)
    db = curve.power_floor_limit_db / (1.0 + (curve.power_floor_corner_hz / frequency) ** curve.power_floor_exponent)
    floor = 10.0 ** (db / 10.0)
    return float((1.0 - floor) * (1.0 - math.exp(-4.0 * beta)) / (4.0 * beta) + floor)


def test_report_layer_carries_g_once_and_keeps_on_axis_direct(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    """報表那一層只搬運，三件事各守一層：
    逐點晚期的開／關比值等於手算的 g（接合那一層再乘一次就變成 g²）；
    頻帶晚期是帶內細軸晚期的正權重平均，開／關比值必須落在帶內各點手算 g 的最小與最大之間
    （頻帶平均那一層再乘一次就掉出去），頻帶幾何能量等於四欄相加；
    頻帶的直達、反射、干涉等於另外直接解一次密軸早期路、在 [fc/√2, fc·√2) 取算術平均
    （密軸那一層或頻帶平均把倍率誤乘進早期三欄會不等）。對準點就是接收點，直達逐位不變。"""
    receiver = Point(4.35, 2.45, 1.05)
    directed_inputs = _inputs(True, aim=receiver)
    plain = shared_control_omnidirectional_report(tmp_path_factory, worker_id, _inputs(False))
    # 解析輸入與場景軸題的主位完全相同；只重用這一半，手算 g 與密軸仍在本題執行。
    directed = shared_report(tmp_path_factory, worker_id, "step4-axis-primary",
                             lambda: _report_with_fast_decay(directed_inputs, monkeypatch))
    solved = report_io.solver_inputs(directed_inputs)
    curve = solved.source_model.parameters
    assert curve is not None
    assert plain.points and len(plain.points) == len(directed.points)
    for off, on in zip(plain.points, directed.points, strict=True):
        assert on.late_energy == pytest.approx(off.late_energy * _hand_g(curve, on.frequency_hz), rel=1e-12)
        assert on.direct_energy == off.direct_energy
    assert [band.direct_energy for band in directed.bands] == [band.direct_energy for band in plain.bands]
    dense = solve_geometric_early_lane(
        source_model=solved.source_model, room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.sound_speed_m_s * solved.density_kg_m3,
        frequencies_hz=GEOMETRIC_BAND_FREQUENCIES_HZ,
        impedance_by_wall={wall.wall_name(): complex(solved.impedance_by_wall[wall]) for wall in Wall.all()},
        scattering_by_wall={wall.wall_name(): value for wall, value in (solved.scattering_by_wall or {}).items()},
        reflection_order_k=solved.reflection_order_k,
    )
    fully_geometric: list[float] = []
    for off_band, on_band in zip(plain.bands, directed.bands, strict=True):
        lower, upper = on_band.center_frequency_hz / math.sqrt(2.0), on_band.center_frequency_hz * math.sqrt(2.0)
        inside = [i for i, f in enumerate(dense.frequencies_hz) if lower <= f < upper]
        assert inside
        for name in ("direct_energy", "reflected_energy", "interference_energy"):
            column = getattr(dense, name)
            expected = sum(column[i] for i in inside) / len(inside)
            assert getattr(on_band, name) == pytest.approx(expected, rel=1e-12), (on_band.center_frequency_hz, name)
        g_in_band = [_hand_g(curve, point.frequency_hz) for point in directed.points
                     if lower <= point.frequency_hz < upper]
        assert g_in_band
        ratio = on_band.late_energy / off_band.late_energy
        assert min(g_in_band) * (1.0 - 1e-12) <= ratio <= max(g_in_band) * (1.0 + 1e-12), on_band.center_frequency_hz
        parts = on_band.direct_energy + on_band.reflected_energy + on_band.interference_energy + on_band.late_energy
        assert on_band.geometric_energy == pytest.approx(parts, rel=1e-12), on_band.center_frequency_hz
        in_band = [point for point in directed.points if lower <= point.frequency_hz < upper]
        if all(point.w_geo == 1.0 for point in in_band):
            # 交接以上整帶只剩幾何路：加權貢獻＝幾何能量（貢獻那一層的晚期再乘一次 g 會不等）。
            fully_geometric.append(on_band.center_frequency_hz)
            assert on_band.geometric_contribution == pytest.approx(on_band.geometric_energy, rel=1e-12)
    assert fully_geometric


def test_capability_outputs_list_exactly_the_report_fields_that_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    """能力表那一列 outputs 要列齊真的會變的頻帶與逐點欄，列出的也要真的會變。"""
    plain_inputs, directed_inputs = _inputs(False), _inputs(True)
    outputs = []
    # 欄位集合只消費報表；重用真衰減題的同一對，避免快替身與真值混成同名資料。
    for inputs, report in zip((plain_inputs, directed_inputs),
                              _shared_real_pair(tmp_path_factory, worker_id, monkeypatch), strict=True):
        outputs.append(report_output.output_from_report(report, inputs=inputs, with_points=True).model_dump())
    changed = {
        f"{section}.{name}"
        for section in ("bands", "points")
        for name in outputs[0][section][0]
        if [row[name] for row in outputs[0][section]] != [row[name] for row in outputs[1][section]]
    }
    entry = load_capabilities(config_path("capabilities.toml")).for_entry("source_directivity")
    analytic = next(item for item in entry.capability
                    if item.materials == SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value)
    listed = {name for name in analytic.outputs if name.startswith(("bands.", "points."))}
    assert changed
    assert changed == listed


@pytest.mark.parametrize(("angle", "accepted"), ((0.0, True), (180.0, True), (-0.001, False), (180.001, False)))
def test_off_axis_angle_is_bounded_to_a_half_turn(angle: float, accepted: bool) -> None:
    row = {
        "order": 1, "wall_sequence": ("x0",), "delay_s": 0.01, "distance_m": 3.43,
        "direction_vector": (1.0, 0.0, 0.0),
        "direction_angles": {"azimuth_deg": 0.0, "elevation_deg": 0.0},
        "departure_off_axis_deg": angle, "relative_direct_energy": (0.5,),
    }
    if accepted:
        assert report_io.PathRow.model_validate(row).departure_off_axis_deg == angle
    else:
        with pytest.raises(ValidationError, match="departure_off_axis_deg"):
            report_io.PathRow.model_validate(row)


def test_unconnected_source_model_kind_fails_instead_of_computing_omnidirectional() -> None:
    """計算入口的分派逐種類明列：表外種類報錯，不會靜靜當成全向算。"""
    from aosr.physics.report_source import directivity_to_apply

    spec = object.__new__(SourceModelSpec)
    object.__setattr__(spec, "kind", SourceModelKind.OMNIDIRECTIONAL)
    object.__setattr__(spec, "parameters", None)
    object.__setattr__(spec, "aim", None)
    assert directivity_to_apply(spec) is None
    object.__setattr__(spec, "kind", "v2_compat_baffled_piston")
    with pytest.raises(ValueError, match="v2_compat_baffled_piston"):
        directivity_to_apply(spec)
    # 日後真的會出現的形狀：表外種類也帶曲線與對準點——照樣報錯，不當成兩參數近似算。
    object.__setattr__(spec, "parameters", DIRECTIVITY.two_parameter)
    object.__setattr__(spec, "aim", Point(4.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="v2_compat_baffled_piston"):
        directivity_to_apply(spec)


def test_early_lane_multiplies_each_path_by_hand_computed_d_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """早期路逐路徑乘 D 一次：攔下鏡像法算出的全向路徑，考卷自己算每條的出發方向與 D、
    自己同調相加，直達、反射、干涉三欄都要對上（D 乘兩次、漏乘直達、用錯方向都會不等）。"""
    captured: list[list[RoomPath]] = []
    real = room_paths.image_source_paths

    def recording(*args: object, **kwargs: object) -> list[RoomPath]:
        paths = real(*args, **kwargs)  # type: ignore[arg-type]  # expires=2026-12-31 reason=考卷攔截原函式、參數原樣轉交
        captured.append(list(paths))
        return paths

    monkeypatch.setattr(geometric_lane, "image_source_paths", recording)
    frequencies = (100.0, 200.0)
    source, receiver, aim = Point(2.0, 1.0, 1.0), Point(4.0, 1.0, 1.0), Point(4.0, 2.0, 1.0)
    rho_c = 1.2 * 343.0
    early = solve_geometric_early_lane(
        source_model=SourceModelSpec(SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1,
                                     DIRECTIVITY.two_parameter, aim),
        room=Room(10.0, 10.0, 10.0), source=source, receiver=receiver, sound_speed_m_s=343.0,
        rho_c_pa_s_per_m=rho_c, frequencies_hz=frequencies,
        impedance_by_wall={wall.wall_name(): complex(4.0 * rho_c) for wall in Wall.all()},
        scattering_by_wall={wall.wall_name(): 0.0 for wall in Wall.all()}, reflection_order_k=1,
    )
    (paths,) = captured
    axis = (2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 0.0)
    curve = DIRECTIVITY.two_parameter
    for index, frequency in enumerate(frequencies):
        beta = curve.beta_limit / (1.0 + (curve.beta_corner_hz / frequency) ** curve.beta_exponent)
        db = curve.power_floor_limit_db / (1.0 + (curve.power_floor_corner_hz / frequency) ** curve.power_floor_exponent)
        floor = 10.0 ** (db / 10.0)
        direct, reflected = 0j, 0j
        for path in paths:
            leaving = [r - m for r, m in zip(receiver.as_tuple(), path.image, strict=True)]
            if path.order == 1:
                # 一階反射離開聲源的方向＝鏡像到接收點的向量，把反射那一軸翻回來。
                flipped = next(a for a in range(3) if path.image[a] != source.as_tuple()[a])
                leaving[flipped] = -leaving[flipped]
            cos_theta = sum(v * a for v, a in zip(leaving, axis, strict=True)) / math.hypot(*leaving)
            d = math.sqrt((1.0 - floor) * math.exp(-2.0 * beta * (1.0 - cos_theta)) + floor)
            if path.order == 0:
                direct = path.path_pressure[index] * d
            else:
                reflected += path.path_pressure[index] * d
        assert early.direct_energy[index] == pytest.approx(abs(direct) ** 2, rel=1e-12)
        assert early.reflected_energy[index] == pytest.approx(abs(reflected) ** 2, rel=1e-12)
        assert early.interference_energy[index] == pytest.approx(
            2.0 * (direct * reflected.conjugate()).real, rel=1e-12)
