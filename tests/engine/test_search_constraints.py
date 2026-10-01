"""求解前幾何限制：每個違反量都以独立手算案例驗證。"""

from __future__ import annotations

import ast
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Point
from aosr.reporting import physics_stage
from aosr.reporting.scheme import Scheme
from aosr.search import layout
from aosr.search.constraints import Reason, Violation, check, to_illegal
from aosr.search.layout import LayoutParams, Placement, place
from aosr.search.layout_settings import Box, Cabinet, LayoutSettings, Span
from tests.engine.test_search_layout import project, settings  # 共用暫存方案與型別設定 fixture（測試輸入）。


def _placement(
    *, left: Point = Point(1.0, 1.0, 1.0), right: Point = Point(1.0, 3.0, 1.0),
    primary: Point = Point(3.0, 1.0, 1.0), others: tuple[tuple[str, Point], ...] = (),
) -> Placement:
    return Placement(left, right, primary, (("main", primary), *others), (-1.0, 0.0))


def _box(x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]) -> Box:
    return Box(x=Span(low=x[0], high=x[1]), y=Span(low=y[0], high=y[1]), z=Span(low=z[0], high=z[1]))


def _changed(settings: LayoutSettings, **fields: object) -> LayoutSettings:
    return LayoutSettings.model_validate(settings.model_dump() | fields)


def _only(violations: tuple[Violation, ...], reason: Reason, amount: float) -> None:
    assert {v.reason for v in violations} == {reason}
    assert tuple(v.amount_m for v in violations) == pytest.approx((amount,), rel=1e-12, abs=0.0)


def test_cabinet_outside_room(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    # 左箱朝 +x、聲學中心在面板：背板 x=0.19-0.2=-0.01。
    bad = _placement(left=Point(0.19, 1.0, 1.0))
    _only(check(project, settings, bad), Reason.CABINET_OUTSIDE_ROOM, 0.01)
    assert check(project, settings, replace(bad, left=Point(0.21, 1.0, 1.0))) == ()
    assert tmp_path.is_dir()


def test_wall_gap(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    chosen = _changed(settings, wall_gap_m=0.05)
    bad = _placement(left=Point(0.24, 1.0, 1.0))
    # 箱背距 x0 0.04，必要間隙 0.05，缺 0.01。
    _only(check(project, chosen, bad), Reason.WALL_GAP, 0.01)
    assert check(project, chosen, replace(bad, left=Point(0.26, 1.0, 1.0))) == ()
    assert tmp_path.is_dir()


def test_cabinets_overlap(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    # 兩箱均朝 +x，x 區間 [0.8,1] 與 [0.99,1.19]，最小穿透 0.01。
    bad = _placement(right=Point(1.19, 1.0, 1.0))
    _only(check(project, settings, bad), Reason.CABINETS_OVERLAP, 0.01)
    assert check(project, settings, replace(bad, right=Point(1.21, 1.0, 1.0))) == ()
    assert check(project, settings, replace(bad, right=Point(1.19, 1.0, 1.5))) == ()
    # 垂直剛好接觸亦不算互疊。
    chosen = _changed(settings, cabinet=Cabinet(width_m=0.2, depth_m=0.2, height_m=0.5))
    assert check(project, chosen, replace(bad, right=Point(1.19, 1.0, 1.5))) == ()
    assert tmp_path.is_dir()


def test_cabinet_in_keep_out(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    forbidden = _box((0.99, 1.5), (0.5, 1.5), (0.5, 1.5))
    chosen = _changed(settings, keep_out=(forbidden,))
    bad = _placement()
    _only(check(project, chosen, bad), Reason.CABINET_IN_KEEP_OUT, 0.01)
    assert check(project, chosen, replace(bad, left=Point(0.98, 1.0, 1.0))) == ()
    above = _box((0.99, 1.5), (0.5, 1.5), (1.21, 1.5))
    assert check(project, _changed(settings, keep_out=(above,)), bad) == ()
    assert tmp_path.is_dir()


def test_seat_in_keep_out(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    chosen = _changed(settings, keep_out=(_box((2.99, 3.5), (0.9, 1.1), (0.9, 1.1)),))
    bad = _placement()
    _only(check(project, chosen, bad), Reason.SEAT_IN_KEEP_OUT, 0.01)
    assert check(project, chosen, _placement(primary=Point(2.98, 1.0, 1.0))) == ()
    other = _placement(primary=Point(2.0, 1.0, 1.0), others=(("visitor", Point(3.0, 1.0, 1.0)),))
    _only(check(project, chosen, other), Reason.SEAT_IN_KEEP_OUT, 0.01)
    assert check(project, chosen, _placement(primary=Point(2.0, 1.0, 1.0),
                                           others=(("visitor", Point(2.98, 1.0, 1.0)),))) == ()
    assert tmp_path.is_dir()


def test_seat_outside_room(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    bad = _placement(primary=Point(6.01, 1.0, 1.0))
    _only(check(project, settings, bad), Reason.SEAT_OUTSIDE_ROOM, 0.01)
    assert check(project, settings, _placement(primary=Point(5.99, 1.0, 1.0))) == ()
    bad = _placement(others=(("visitor", Point(3.0, -0.02, 1.0)),))
    _only(check(project, settings, bad), Reason.SEAT_OUTSIDE_ROOM, 0.02)
    assert check(project, settings, _placement(others=(("visitor", Point(3.0, 0.01, 1.0)),))) == ()
    assert tmp_path.is_dir()


def test_outside_speaker_area(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    areas = (_box((1.01, 1.5), (0.5, 1.5), (0.5, 1.5)), _box((0.5, 1.5), (2.5, 3.5), (0.5, 1.5)))
    chosen = _changed(settings, speaker_areas=areas)
    bad = _placement()
    _only(check(project, chosen, bad), Reason.OUTSIDE_SPEAKER_AREA, 0.01)
    assert check(project, chosen, replace(bad, left=Point(1.02, 1.0, 1.0))) == ()
    assert tmp_path.is_dir()


def test_listening_distance_out_of_range(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    # 左箱到主位直線 2m；右箱 sqrt(8)m，在範圍內。左箱缺 0.01m。
    chosen = _changed(settings, listening_range_m=Span(low=2.01, high=3.0))
    bad = _placement()
    _only(check(project, chosen, bad), Reason.LISTENING_DISTANCE_OUT_OF_RANGE, 0.01)
    assert check(project, chosen, _placement(primary=Point(3.02, 1.0, 1.0))) == ()
    # 三維距離：高度差也必須算入，上限右箱差 0.01。
    chosen = _changed(settings, listening_range_m=Span(low=0.1, high=2.99))
    elevated = _placement(primary=Point(3.0, 1.0, 2.0))
    _only(check(project, chosen, elevated), Reason.LISTENING_DISTANCE_OUT_OF_RANGE, 0.01)
    assert check(project, chosen, _placement(primary=Point(2.98, 1.0, 2.0))) == ()
    assert tmp_path.is_dir()


def test_each_reason_uses_the_maximum_violation(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    chosen = _changed(settings, speaker_areas=(
        _box((1.01, 1.5), (0.5, 1.5), (0.5, 1.5)),
        _box((1.04, 1.5), (2.5, 3.5), (0.5, 1.5)),
    ))
    _only(check(project, chosen, _placement()), Reason.OUTSIDE_SPEAKER_AREA, 0.04)
    bad = _placement(primary=Point(6.01, 1.0, 1.0), others=(("visitor", Point(3.0, -0.02, 1.0)),))
    _only(check(project, settings, bad), Reason.SEAT_OUTSIDE_ROOM, 0.02)
    assert tmp_path.is_dir()


def test_containment_requires_distance_to_exit(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    # 箱體完全含在禁區內；不可只回箱子的寬 0.2。y 移動 0.6m 才能離開。
    chosen = _changed(settings, keep_out=(_box((0.0, 2.0), (0.5, 1.5), (0.5, 1.5)),))
    _only(check(project, chosen, _placement()), Reason.CABINET_IN_KEEP_OUT, 0.6)
    assert tmp_path.is_dir()


def test_toed_in_cabinet_corners_and_acoustic_center(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    # 主位距連線 0.8、半間距 0.6，所以前向 (.8,.6)，橫向 (-.6,.8)。
    # 左箱背角 x = d - .2*.8 - .1*.6 = d-.22。
    chosen = _changed(settings, axis_offset_m=0.0, speaker_height_m=1.0, ear_height_m=1.0)
    bad = place(project, chosen, LayoutParams(0.21, 1.2, 0.8))
    _only(check(project, chosen, bad), Reason.CABINET_OUTSIDE_ROOM, 0.01)
    assert check(project, chosen, place(project, chosen, LayoutParams(0.23, 1.2, 0.8))) == ()
    # 聲學中心後移 .05，背板只剩 .15m：最小 x 增加 .04。
    cab = Cabinet(width_m=0.2, depth_m=0.2, height_m=0.4, acoustic_center_behind_front_m=0.05)
    assert check(project, _changed(chosen, cabinet=cab), bad) == ()
    # 聲學中心離箱底 .3，中心 z=.29 時箱底出地板 .01。
    cab = Cabinet(width_m=0.2, depth_m=0.2, height_m=0.4, acoustic_center_above_bottom_m=0.3)
    vertical = _changed(chosen, speaker_height_m=0.29, cabinet=cab)
    _only(check(project, vertical, place(project, vertical, LayoutParams(1.0, 1.2, 0.8))),
          Reason.CABINET_OUTSIDE_ROOM, 0.01)
    assert tmp_path.is_dir()


def test_speaker_area_uses_euclidean_distance(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    chosen = _changed(settings, speaker_areas=(
        _box((1.03, 1.5), (1.04, 1.5), (0.5, 1.5)),
        _box((0.5, 1.5), (2.5, 3.5), (0.5, 1.5)),
    ))
    _only(check(project, chosen, _placement()), Reason.OUTSIDE_SPEAKER_AREA, 0.05)
    assert tmp_path.is_dir()


def test_illegal_is_decided_before_scheme_is_built(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("illegal placement reached scheme construction or solver")
    monkeypatch.setattr(layout, "to_scheme", forbidden)
    monkeypatch.setattr(physics_stage, "solve_scheme_physics", forbidden)
    monkeypatch.setattr(physics_stage, "solve_checked_physics", forbidden)
    placement = place(project, settings, LayoutParams(0.01, 1.2, 2.2))
    outcome = to_illegal(check(project, settings, placement))
    assert Reason.CABINET_OUTSIDE_ROOM.value in outcome.reason
    assert outcome.violation > 0.0
    assert tmp_path.is_dir()


def test_to_illegal_sums_amounts_and_names_every_reason(tmp_path: Path) -> None:
    violations = tuple(Violation(reason, float(index) / 10) for index, reason in enumerate(reversed(tuple(Reason)), 1))
    result = to_illegal(violations)
    assert result.reason == "+".join(sorted(reason.value for reason in Reason))
    assert result.violation == pytest.approx(sum(v.amount_m for v in violations), rel=1e-12)
    with pytest.raises(ValueError):
        to_illegal(())
    for amount in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            Violation(Reason.WALL_GAP, amount)
    assert tmp_path.is_dir()


def test_search_reads_no_default_layout_config(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "src" / "aosr" / "search"
    forbidden = {"aosr.config.stereo_layout", "aosr.config.fem_lane"}
    for path in source.rglob("*"):
        if path.suffix != ".py":
            continue
        copied = tmp_path / "module"
        shutil.copyfile(path, copied)
        tree = ast.parse(copied.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name == name or alias.name.startswith(name + ".")
                               for alias in node.names for name in forbidden)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in forbidden
                assert not any(f"{module}.{alias.name}" in forbidden for alias in node.names)
                assert not (module == "aosr.config" and any(alias.name == "*" for alias in node.names))
