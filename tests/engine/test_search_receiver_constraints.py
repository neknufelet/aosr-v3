"""手算六面退出量、四面前牆與接觸帶；擺位移位及搜尋在求解前擋。"""
from dataclasses import replace
import math
from pathlib import Path
from typing import Literal

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point
from aosr.search.constraints import check, to_illegal
from aosr.search.layout import LayoutParams, Placement, params_from_unit, place, to_scheme
from aosr.search.layout_settings import Cabinet, Span
from aosr.search.placement_stability_geometry import check_shift, generate_shifts
from aosr.search.placement_stability_record import PointRecord
from aosr.search.report_stability import _unscored_line
from aosr.search.seat_lock import check_seat_lock
from aosr.search.settings import SearchSettings
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine._receiver_constraint_cases import project, settings

CONTACT_REL = contract_value("furniture_geometry_contact")
# 題目房間最長邊 8m；只讀唯一登記簿的相對界線，手算絕對界線。
MARGIN = 8.0 * CONTACT_REL
CABINET = "receiver_in_cabinet"
AHEAD = "receiver_ahead_of_speakers"


def _amounts(placement: Placement) -> dict[str, float]:
    return {v.reason.value: v.amount_m for v in check(project(), settings(), placement)}


def _axis_aligned(point: Point) -> Placement:
    main = Point(4.0, 2.0, 1.25)
    return Placement(Point(2.0, 2.0, 1.25), Point(2.0, 4.0, 1.25), main,
                     (("main", main), ("front", point)), (-1.0, 0.0))


@pytest.mark.parametrize("axis,plane,inward", [
    (0, 1.625, 1), (0, 2.125, -1), (1, 1.75, 1), (1, 2.25, -1), (2, 1.0, 1), (2, 1.5, -1),
])
@pytest.mark.parametrize("factor", [0.0, 1.0 - MUTANT_MARGIN, 1.0 + MUTANT_MARGIN])
def test_cabinet_six_faces_contact_boundary(axis: int, plane: float, inward: int, factor: float) -> None:
    # 左箱朝 +x，六面如表；點到最近那一面的原距離是 factor*m，接觸帶不算箱內。
    coordinates = [1.875, 2.0, 1.25]
    coordinates[axis] = plane + inward * factor * MARGIN
    amounts = _amounts(_axis_aligned(Point(*coordinates)))
    if factor > 1.0:
        assert amounts[CABINET] == pytest.approx(factor * MARGIN, rel=0, abs=MARGIN * MUTANT_MARGIN / 8)
    else:
        assert CABINET not in amounts


@pytest.mark.parametrize("height", [0.99, 1.51])
def test_cabinet_footprint_without_height_overlap_is_clear(height: float) -> None:
    assert CABINET not in _amounts(_axis_aligned(Point(1.875, 2.0, height)))


@pytest.mark.parametrize("wall,left_point,right_point", [
    ("x0", (2.04, 3.43), (2.04, 4.57)), ("xL", (5.96, 4.57), (5.96, 3.43)),
    ("y0", (4.57, 2.04), (3.43, 2.04)), ("yL", (3.43, 5.96), (4.57, 5.96)),
])
@pytest.mark.parametrize("speaker", ["left", "right"])
def test_toed_in_cabinet_on_every_front_wall(
    wall: Literal["x0", "xL", "y0", "yL"], left_point: tuple[float, float],
    right_point: tuple[float, float], speaker: str,
) -> None:
    # 半間距 0.6、主位後退 0.8，箱前向分量是 0.8 比 0.6；點在聲學中心前 0.05 m。
    # 前面板 .1m，所以退出最短距離 .05；用世界座標手算表，非箱體函式反推答案。
    cabinet = Cabinet(width_m=.4, depth_m=.5, height_m=.4, acoustic_center_behind_front_m=.1)
    chosen = settings(front_wall=wall, cabinet=cabinet)
    placement = place(project(), chosen, LayoutParams(2.0, 1.2, .8))
    x, y = left_point if speaker == "left" else right_point
    placement = replace(placement, receivers=(("main", placement.primary), ("front", Point(x, y, 1.25))))
    amounts = {v.reason.value: v.amount_m for v in check(project(), chosen, placement)}
    assert amounts == {CABINET: pytest.approx(.05)}


def test_primary_inside_cabinet_uses_shortest_three_dimensional_exit() -> None:
    # 左箱前面 x=2.5，主位 x=2.2：前面 .3、兩側 .25、上下 .25，最短是 .25。
    chosen = settings(cabinet=Cabinet(width_m=.5, depth_m=.75, height_m=.5,
                                    acoustic_center_behind_front_m=.5))
    main = Point(2.2, 2.0, 1.25)
    placement = replace(_axis_aligned(main), primary=main, receivers=(("main", main),))
    amounts = {v.reason.value: v.amount_m for v in check(project(), chosen, placement)}
    assert amounts == {CABINET: pytest.approx(.25)}
    # 耳高只離頂面 .01m，退出量必須改取垂直，不沿用禁區的水平退出量。
    main = Point(2.2, 2.0, 1.49)
    placement = replace(placement, primary=main, receivers=(("main", main),))
    assert {v.reason.value: v.amount_m for v in check(project(), chosen, placement)} == {CABINET: pytest.approx(.01)}


def test_cabinet_records_maximum_depth_over_all_surrounding_points() -> None:
    placement = _axis_aligned(Point(2.1, 2.0, 1.25))  # 離前面 .025m。
    placement = replace(placement, receivers=(*placement.receivers, ("visitor", Point(1.875, 2.0, 1.25))))
    assert _amounts(placement)[CABINET] == pytest.approx(.25)


@pytest.mark.parametrize("wall,on,ahead,behind", [
    ("x0", (2.0, 4.0), (1.95, 4.0), (2.05, 4.0)),
    ("xL", (6.0, 4.0), (6.05, 4.0), (5.95, 4.0)),
    ("y0", (4.0, 2.0), (4.0, 1.95), (4.0, 2.05)),
    ("yL", (4.0, 6.0), (4.0, 6.05), (4.0, 5.95)),
])
@pytest.mark.parametrize("receiver", ["main", "front"])
def test_speaker_line_direction_and_amount_on_every_front_wall(
    wall: Literal["x0", "xL", "y0", "yL"], on: tuple[float, float],
    ahead: tuple[float, float], behind: tuple[float, float], receiver: str,
) -> None:
    chosen = settings(front_wall=wall)
    base = place(project(), chosen, LayoutParams(2.0, 1.2, .8))
    for xy, expected in ((on, MARGIN), (ahead, .05 + MARGIN), (behind, None)):
        point = Point(*xy, 1.25)
        primary = point if receiver == "main" else base.primary
        placement = replace(base, primary=primary, receivers=(("main", primary), ("front", point)))
        amounts = {v.reason.value: v.amount_m for v in check(project(), chosen, placement)}
        if expected is None:
            assert AHEAD not in amounts
        else:
            assert amounts[AHEAD] == pytest.approx(expected, rel=1e-13)
            assert amounts[AHEAD] > 0.0


@pytest.mark.parametrize("factor", [1.0 - MUTANT_MARGIN, 1.0, 1.0 + MUTANT_MARGIN])
def test_speaker_line_contact_band_behind_line(factor: float) -> None:
    placement = _axis_aligned(Point(2.0 + factor * MARGIN, 3.0, 1.25))
    amounts = _amounts(placement)
    if factor < 1.0:
        assert amounts[AHEAD] == pytest.approx(MARGIN * MUTANT_MARGIN, rel=0, abs=MARGIN * MUTANT_MARGIN / 8)
    else:
        assert AHEAD not in amounts


def test_speaker_line_records_maximum_over_receivers_and_ignores_height() -> None:
    placement = _axis_aligned(Point(1.95, 3.0, 3.0))
    placement = replace(placement, receivers=(*placement.receivers, ("visitor", Point(1.9, 3.0, .1))))
    assert _amounts(placement)[AHEAD] == pytest.approx(.1 + MARGIN, rel=1e-13)


def test_locked_seat_tiny_derived_distance_rejects_front_surrounding_point() -> None:
    chosen = settings(seat_locked=True, listening_distance_m=None, axis_offset_m=-2.0,
                      front_distance_m=Span(low=3.8, high=3.99))
    original = project()
    check_seat_lock(original, chosen)
    params = params_from_unit(dict(front_distance=(3.95-3.8)/(3.99-3.8), spacing=.5), chosen, project=original)
    assert params.listening_distance_m == pytest.approx(.05)
    placement = place(original, chosen, params)
    amounts = {v.reason.value: v.amount_m for v in check(original, chosen, placement)}
    assert amounts == {AHEAD: pytest.approx(.05 + MARGIN, rel=1e-13)}
    assert to_illegal(check(original, chosen, placement)).violation > 0.0


def test_locked_seat_tiny_distance_still_checks_cabinet_surrounding_point() -> None:
    # L=.05、半間距 .6，左箱前向 (1,12)/sqrt(145)。點在前向 .02m，退出前面 .125-.02=.105m。
    original = project()
    data = original.model_dump()
    data["receiver_set"]["points"][1]["position_m"] = (
        3.95 + .02 / math.sqrt(145), 1.4 + .24 / math.sqrt(145), 1.25)
    original = type(original).model_validate(data)
    chosen = settings(seat_locked=True, listening_distance_m=None, axis_offset_m=-2.0,
                      front_distance_m=Span(low=3.8, high=3.99))
    check_seat_lock(original, chosen)
    params = params_from_unit(dict(front_distance=(3.95-3.8)/(3.99-3.8), spacing=(1.2-.2)/(3.0-.2)),
                              chosen, project=original)
    assert params.listening_distance_m == pytest.approx(.05)
    placement = place(original, chosen, params)
    assert {v.reason.value: v.amount_m for v in check(original, chosen, placement)} == {CABINET: pytest.approx(.105)}


@pytest.mark.parametrize("reason", [CABINET, AHEAD])
def test_search_records_receiver_illegal_before_build_or_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    from aosr.reporting.evaluation import purpose_settings
    from aosr.search import layout, run as search_run
    from aosr.search.sampler import SamplerAdapter
    from aosr.search.store import SearchIdentity, SearchStore
    from tests.engine._search_run_cases import ENGINE, FakeCompute, registry_copy, rows, run

    original = project()
    if reason == CABINET:
        data = original.model_dump()
        data["receiver_set"]["points"][1]["position_m"] = (3.24, 1.43, 1.25)
        original = type(original).model_validate(data)
        chosen = settings(cabinet=Cabinet(width_m=.4, depth_m=.5, height_m=.4, acoustic_center_behind_front_m=.1))
        params = LayoutParams(2.0, 1.2, .8)
        expected = .05
    else:
        chosen = settings(seat_locked=True, listening_distance_m=None, axis_offset_m=-2.0,
                          front_distance_m=Span(low=3.8, high=3.99))
        params = LayoutParams(3.95, 1.2, .05)
        expected = .05 + MARGIN
    config = SearchSettings(purpose=original.purpose, layout=chosen, seed=7, n_startup_trials=1,
                            batch_size=1, max_workers=1, budget=1, convergence_run=1)
    registry = registry_copy(tmp_path)
    identity = SearchIdentity("phys-v1:" + "b" * 64, ENGINE, purpose_settings(registry, original.purpose))
    store = SearchStore.create(tmp_path / "searches", project=original, settings=config,
                              identity=identity, versions={"python": "test", "optuna": "test", "numpy": "test"})

    def adapter(store: SearchStore) -> tuple[SamplerAdapter, bool]:
        result = SamplerAdapter(layout.space_for(chosen), config.sampler_settings())
        result.enqueue(layout.unit_from_params(params, chosen))
        return result, True

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("座位物理硬限制必須早於建方案和計算")

    monkeypatch.setattr(search_run, "_adapter", adapter)
    monkeypatch.setattr(layout, "to_scheme", forbidden)
    compute = FakeCompute(store)
    status = run(store, registry, compute)
    assert status.state == "budget_exhausted" and status.illegal_reasons.keys() == {reason}
    row, = rows(store)
    assert row.outcome == "illegal" and row.reason == reason
    assert row.violation_m == pytest.approx(expected)
    assert row.result_file is None and row.score is None and row.seconds == 0.0
    assert row.trial_number not in compute.calls
    assert status.illegal_reasons[reason] == sum(r.reason == reason for r in rows(store))


@pytest.mark.parametrize("reason,point", [(CABINET, Point(2.1, 2.0, 1.25)), (AHEAD, Point(1.95, 3.0, 1.25))])
def test_physical_reasons_stop_shift_before_scheme_and_render_chinese(
    monkeypatch: pytest.MonkeyPatch, reason: str, point: Point,
) -> None:
    from aosr.search import placement_stability_geometry as geometry
    chosen = settings()
    config = SearchSettings(purpose=project().purpose, layout=chosen, seed=7, n_startup_trials=1,
                            batch_size=1, max_workers=1, budget=1, convergence_run=1)
    placement = _axis_aligned(point)
    shift = geometry.Shift("seat_forward", project(), config, placement, "shift-control")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("物理不合法點不得建方案或求解")

    monkeypatch.setattr(geometry, "to_scheme", forbidden)
    legality = check_shift(project(), shift, contact_rel=CONTACT_REL,
        capabilities=load_capabilities(config_path("capabilities.toml")),
        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    assert legality.outcome == "unplaceable" and not legality.out_of_spec
    assert {v.reason.value for v in legality.violations} == {reason}
    record = PointRecord(trial_number=0, shift_name="seat_forward", outcome=legality.outcome, violations=legality.violations)
    text = _unscored_line((record,), None)
    label = "座位點在喇叭箱體裡" if reason == CABINET else "座位點在喇叭連線前方"
    assert f"座位向前 擺不出來（{label}，違反量" in text
    assert reason not in text and "公尺" in text


@pytest.mark.parametrize("reason", [CABINET, AHEAD])
def test_generated_seat_forward_shift_uses_same_physical_check(reason: str) -> None:
    original = project()
    if reason == CABINET:
        base = _axis_aligned(Point(2.13, 2.0, 1.25))
        expected = .015
    else:
        main = Point(2.11, 3.0, 1.25)
        base = replace(_axis_aligned(Point(2.01, 3.0, 1.25)), primary=main,
                       receivers=(("main", main), ("front", Point(2.01, 3.0, 1.25))))
        expected = .01 + MARGIN
    assert check(original, settings(), base) == ()
    scheme = to_scheme(original, base, "shift-base")
    config = SearchSettings(purpose=original.purpose, layout=settings(), seed=7, n_startup_trials=1,
                            batch_size=1, max_workers=1, budget=1, convergence_run=1)
    shift = next(s for s in generate_shifts(original, config, scheme) if s.name == "seat_forward")
    legality = check_shift(original, shift, contact_rel=CONTACT_REL,
        capabilities=load_capabilities(config_path("capabilities.toml")),
        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    assert legality.outcome == "unplaceable" and not legality.out_of_spec
    assert {v.reason.value: v.amount_m for v in legality.violations} == {reason: pytest.approx(expected)}
