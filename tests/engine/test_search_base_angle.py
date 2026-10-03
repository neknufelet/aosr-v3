"""專案水平夾角與可行三角形起點；期望值由獨立幾何案例手算。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Point, distance
from aosr.reporting.scheme import Scheme
from aosr.search.constraints import check, to_illegal
from aosr.search.layout import LayoutParams, Placement, params_from_unit, place, standard_start, unit_from_params
from aosr.search.layout_settings import Box, LayoutSettings, Span
from aosr.search.settings import SearchSettings
from aosr.search.store import SETTINGS_FILE, SearchStore
from tests.engine._search_run_cases import FakeCompute, make_store, rows, run
from tests.engine._search_store_cases import settings_document
from tests.engine.test_search_layout import project, settings  # 共用暫存專案與設定 fixture（測試輸入）。


def _changed(settings: LayoutSettings, **fields: object) -> LayoutSettings:
    return LayoutSettings.model_validate(settings.model_dump() | fields)


def _angle_placement(angle_deg: float, height: float = 1.2) -> Placement:
    """兩向量各偏半角，水平長度分別為 1、2 m，以防誤用單邊距離。"""
    half = math.radians(angle_deg / 2.0)
    primary = Point(3.0, 1.8, 1.2)
    left = Point(3.0 - math.cos(half), 1.8 - math.sin(half), height)
    right = Point(3.0 - 2.0 * math.cos(half), 1.8 + 2.0 * math.sin(half), height)
    return Placement(left, right, primary, (("main", primary),), (-1.0, 0.0))


@pytest.mark.parametrize("angle_deg,bound_deg", [(30.0, 50.0), (90.0, 70.0)])
def test_base_angle_violation_is_mean_horizontal_radius_times_radians(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, angle_deg: float, bound_deg: float,
) -> None:
    chosen = _changed(settings, base_angle_deg=Span(low=50.0, high=70.0))
    violations = check(project, chosen, _angle_placement(angle_deg))
    assert {item.reason.value for item in violations} == {"base_angle_out_of_range"}
    # 手算兩個半角向量：各 sqrt(cos² + sin²) 乘半徑，平均為 1.5 m。
    half = math.radians(angle_deg / 2.0)
    mean_radius = (1.0 + 2.0) * math.sqrt(math.cos(half) ** 2 + math.sin(half) ** 2) / 2.0
    expected = mean_radius * math.radians(abs(angle_deg - bound_deg))
    assert tuple(item.amount_m for item in violations) == pytest.approx((expected,), rel=1e-12)
    illegal = to_illegal(violations)
    assert illegal.reason == "base_angle_out_of_range"
    assert illegal.violation == pytest.approx(expected, rel=1e-12)
    assert tmp_path.is_dir()


def test_base_angle_inside_range_is_legal(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    chosen = _changed(settings, base_angle_deg=Span(low=50.0, high=70.0))
    assert check(project, chosen, _angle_placement(60.0)) == ()
    assert tmp_path.is_dir()


@pytest.mark.parametrize("angle_deg", [10.0, 150.0])
def test_unset_base_angle_does_not_limit_projects(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, angle_deg: float,
) -> None:
    assert check(project, settings, _angle_placement(angle_deg)) == ()
    assert tmp_path.is_dir()


@pytest.mark.parametrize("angle_deg", [30.0, 60.0, 90.0])
def test_base_angle_ignores_speaker_height(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, angle_deg: float,
) -> None:
    chosen = _changed(settings, base_angle_deg=Span(low=50.0, high=70.0))
    low = check(project, chosen, _angle_placement(angle_deg))
    high = check(project, chosen, _angle_placement(angle_deg, height=2.4))
    assert high == low
    assert tmp_path.is_dir()


@pytest.mark.parametrize("left,right", [(Point(2.0, 2.0, 1.2), Point(4.0, 2.0, 1.2)),
                                      (Point(1.0, 2.0, 1.2), Point(2.0, 2.0, 1.2))])
def test_collinear_base_angle_is_not_illegal(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, left: Point, right: Point,
) -> None:
    chosen = _changed(settings, base_angle_deg=Span(low=50.0, high=70.0))
    main = Point(3.0, 2.0, 1.2)
    placed = Placement(left, right, main, (("main", main),), (-1.0, 0.0))
    assert check(project, chosen, placed) == ()
    assert tmp_path.is_dir()


@pytest.mark.parametrize("low,high", [(50.0, 50.0), (70.0, 50.0), (0.0, 70.0), (-1.0, 70.0),
                                     (50.0, 180.0), (50.0, 181.0)])
def test_base_angle_settings_reject_invalid_limits(
    tmp_path: Path, settings: LayoutSettings, low: float, high: float,
) -> None:
    with pytest.raises(ValueError):
        _changed(settings, base_angle_deg={"low": low, "high": high})
    assert tmp_path.is_dir()


def test_old_search_settings_load_without_angle(tmp_path: Path, settings: LayoutSettings) -> None:
    legacy = settings.model_dump(exclude={"base_angle_deg"})
    target = tmp_path / "legacy-search-settings"
    target.write_text(LayoutSettings.model_validate(legacy).model_dump_json(exclude={"base_angle_deg"}),
                      encoding="utf-8")
    loaded = LayoutSettings.model_validate_json(target.read_bytes())
    assert loaded.base_angle_deg is None


def test_search_snapshot_tracks_project_angle_and_loads_legacy(tmp_path: Path) -> None:
    """驗設定序列化與有／無夾角的指紋差異；舊資料夾存下的指紋另題驗證。"""
    original = SearchSettings.model_validate(settings_document())
    legacy = tmp_path / "legacy-total-settings"
    legacy.write_text(original.model_dump_json(exclude={"layout": {"base_angle_deg"}}), encoding="utf-8")
    loaded = SearchSettings.model_validate_json(legacy.read_bytes())
    assert loaded.layout.base_angle_deg is None
    chosen = _changed(loaded.layout, base_angle_deg=Span(low=50.0, high=70.0))
    changed = SearchSettings.model_validate(loaded.model_dump() | {"layout": chosen})
    assert changed.fingerprint != loaded.fingerprint
    assert changed.canonical()["layout"] == chosen.model_dump(mode="json")
    assert SearchSettings.model_validate_json(changed.model_dump_json()).layout.base_angle_deg == chosen.base_angle_deg


def test_legacy_search_snapshot_opens_with_original_fingerprint(tmp_path: Path) -> None:
    """舊算法只對沒有夾角鍵的總表算摘要；新增 null 不得冒充設定改動。"""
    store, _ = make_store(tmp_path)
    # 直接由原擺位型別造舊形狀，不經新增的搜尋設定序列化器，保留其他 null。
    legacy = store.settings.model_dump() | {
        "layout": store.settings.layout.model_dump(mode="json", exclude={"base_angle_deg"}),
    }
    canonical = json.dumps(legacy, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    target = store.path / SETTINGS_FILE
    target.write_text(json.dumps({"settings": legacy, "fingerprint": fingerprint}), encoding="utf-8")
    opened = SearchStore.open(store.path)
    assert opened.settings.fingerprint == fingerprint
    assert opened.settings.canonical() == legacy
    assert opened.settings.layout.base_angle_deg is None
    assert opened.project == store.project
    assert opened.identity == store.identity


def test_each_project_has_its_own_angle_limits(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    narrow = _changed(settings, base_angle_deg=Span(low=20.0, high=40.0))
    wide = _changed(settings, base_angle_deg=Span(low=50.0, high=70.0))
    placed = _angle_placement(30.0)
    assert check(project, narrow, placed) == ()
    assert {item.reason.value for item in check(project, wide, placed)} == {"base_angle_out_of_range"}
    assert check(project, settings, placed) == ()
    assert tmp_path.is_dir()


def _demo(settings: LayoutSettings, **fields: object) -> LayoutSettings:
    return _changed(settings, **({
        "spacing_m": Span(low=1.0, high=2.4),
        "listening_distance_m": Span(low=1.6, high=3.5),
        "listening_range_m": Span(low=2.0, high=4.0),
        "base_angle_deg": Span(low=50.0, high=70.0),
    } | fields))


def test_demo_start_clamps_spacing_and_is_legal(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    chosen = _demo(settings)
    params = standard_start(project, chosen)
    assert params is not None
    assert params.front_distance_m == 1.0
    assert params.spacing_m == pytest.approx(2.0, rel=1e-12)
    assert params.listening_distance_m == pytest.approx(math.sqrt(3.0), rel=1e-12)
    placed = place(project, chosen, params)
    assert distance(placed.primary, placed.left) == pytest.approx(2.0, rel=1e-12)
    assert distance(placed.primary, placed.right) == pytest.approx(2.0, rel=1e-12)
    assert check(project, chosen, placed) == ()
    assert unit_from_params(params, chosen)
    assert tmp_path.is_dir()


def test_start_keeps_already_feasible_project_spacing(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
) -> None:
    chosen = _changed(settings, base_angle_deg=Span(low=50.0, high=70.0),
                      listening_range_m=Span(low=1.0, high=3.0))
    params = standard_start(project, chosen)
    assert params is not None
    assert params.spacing_m == 1.2
    assert params.front_distance_m == 1.0
    assert params.listening_distance_m == pytest.approx(1.2 * math.sqrt(3.0) / 2.0, rel=1e-12)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("low,high,want", [(30.0, 40.0, 40.0), (80.0, 100.0, 80.0)])
def test_start_uses_project_angle_nearest_sixty(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, low: float, high: float, want: float,
) -> None:
    chosen = _changed(settings, base_angle_deg=Span(low=low, high=high))
    params = standard_start(project, chosen)
    assert params is not None
    angle = math.degrees(2.0 * math.atan2(params.spacing_m / 2.0, params.listening_distance_m))
    assert angle == pytest.approx(want, rel=1e-12)
    assert params.spacing_m == 1.2
    assert check(project, chosen, place(project, chosen, params)) == ()
    assert tmp_path.is_dir()


@pytest.mark.parametrize("fields,want", [
    ({"spacing_m": Span(low=1.5, high=2.4)}, 1.5),
    ({"spacing_m": Span(low=0.4, high=0.8)}, 0.8),
    ({"listening_distance_m": Span(low=1.6, high=3.5)}, 3.2 / math.sqrt(3.0)),
    ({"listening_distance_m": Span(low=0.2, high=0.5)}, 1.0 / math.sqrt(3.0)),
    ({"listening_range_m": Span(low=0.5, high=1.0)}, 1.0),
])
def test_start_clamps_to_each_feasible_distance_bound(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, fields: dict[str, object], want: float,
) -> None:
    chosen = _changed(settings, **fields)
    params = standard_start(project, chosen)
    assert params is not None
    assert params.spacing_m == pytest.approx(want, rel=1e-12)
    assert unit_from_params(params, chosen)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("limits,want", [(Span(low=2.0, high=4.0), math.sqrt(3.0)),
                                        (Span(low=1.1, high=1.3), math.sqrt(0.69))])
def test_start_ear_distance_includes_height_difference(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, limits: Span, want: float,
) -> None:
    chosen = _changed(settings, speaker_height_m=2.2, ear_height_m=1.2, listening_range_m=limits)
    params = standard_start(project, chosen)
    assert params is not None
    assert params.spacing_m == pytest.approx(want, rel=1e-12)
    assert check(project, chosen, place(project, chosen, params)) == ()
    assert tmp_path.is_dir()


@pytest.mark.parametrize("fields", [
    {"spacing_m": Span(low=1.0, high=1.5), "listening_range_m": Span(low=2.0, high=4.0)},
    {"speaker_height_m": 2.5, "listening_range_m": Span(low=0.5, high=1.0)},
    {"spacing_m": Span(low=0.2, high=1.0), "listening_distance_m": Span(low=2.0, high=3.0)},
    {"front_distance_m": Span(low=1.5, high=2.0)},
])
def test_start_returns_none_without_a_feasible_interval(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, fields: dict[str, object],
) -> None:
    assert standard_start(project, _changed(settings, **fields)) is None
    assert tmp_path.is_dir()


@pytest.mark.parametrize("upper", [25, 29, 30, 31, 33, 35, 37, 41, 43, 45, 46, 47, 52, 53])
def test_angle_endpoint_start_is_legal_after_unit_roundtrip(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, upper: float,
) -> None:
    """端點的反推與判定捨入不同；必須通過實際入列後的幾何硬限制。"""
    chosen = _changed(settings, base_angle_deg=Span(low=10.0, high=upper))
    start = standard_start(project, chosen)
    assert start is not None
    actual = params_from_unit(unit_from_params(start, chosen), chosen)
    assert check(project, chosen, place(project, chosen, actual)) == ()
    assert tmp_path.is_dir()


@pytest.mark.parametrize("upper,low,high", [(30.0, 0.3, 3.0), (41.0, 0.1, 3.0), (41.0, 0.4, 3.0),
                                         (41.0, 0.4, 4.0), (43.0, 0.4, 3.0), (43.0, 0.4, 4.0),
                                         (45.0, 0.4, 3.0), (45.0, 0.4, 4.0)])
def test_start_is_legal_with_other_unit_listening_ranges(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, upper: float, low: float, high: float,
) -> None:
    """相同公尺起點換單位區間後捨入可能不同；只驗 place 的版本會漏掉。"""
    chosen = _changed(settings, listening_distance_m=Span(low=low, high=high),
                      base_angle_deg=Span(low=10.0, high=upper))
    start = standard_start(project, chosen)
    assert start is not None
    actual = params_from_unit(unit_from_params(start, chosen), chosen)
    assert check(project, chosen, place(project, chosen, actual)) == ()
    assert tmp_path.is_dir()


def _ear_bound_project(project: Scheme) -> Scheme:
    speakers = dict(project.speakers)
    speakers["left"] = Point(1.0, 0.75, 1.2)
    speakers["right"] = Point(1.0, 2.75, 1.2)
    return Scheme.model_validate(project.model_dump() | {"speakers": speakers})


def test_original_spacing_on_ear_lower_bound_is_legal_after_unit_roundtrip(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
) -> None:
    project = _ear_bound_project(project)
    assert distance(project.speakers["left"], project.speakers["right"]) == 2.0
    chosen = _demo(settings)
    start = standard_start(project, chosen)
    assert start is not None
    assert start.spacing_m == pytest.approx(2.0, rel=1e-12)
    actual = params_from_unit(unit_from_params(start, chosen), chosen)
    assert check(project, chosen, place(project, chosen, actual)) == ()
    assert tmp_path.is_dir()


@pytest.mark.parametrize("case", ["angle", "ear"])
def test_boundary_start_is_first_legal_search_trial(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, case: str,
) -> None:
    """只替換昂貴計算；第一筆實際試算要是起點，不能被剪成不合法。"""
    chosen = _changed(settings, base_angle_deg=Span(low=10.0, high=29.0))
    if case == "ear":
        project = _ear_bound_project(project)
        chosen = _demo(settings)
    store, registry = make_store(tmp_path, budget=1, batch=1, layout_changes=chosen.model_dump())
    # 改暫存快照以沿用 make_store 的完整身分／用途；不寫參考專案原檔。
    from aosr.search.store import PROJECT_FILE

    (store.path / PROJECT_FILE).write_text(project.model_dump_json(), encoding="utf-8")
    store = SearchStore.open(store.path)
    start = standard_start(store.project, store.settings.layout)
    assert start is not None
    compute = FakeCompute(store)
    status = run(store, registry, compute)
    first = rows(store)[0]
    assert status.start_enqueued
    assert first.trial_number == 0 and first.outcome != "illegal"
    assert first.unit_params_hex == {name: value.hex() for name, value in unit_from_params(start, chosen).items()}
    assert compute.calls[0] is None
    assert compute.calls[1] == first.trial_number


def _start_tuple(start: LayoutParams | None) -> tuple[float, float, float]:
    assert start is not None
    return start.front_distance_m, start.spacing_m, start.listening_distance_m


def _horizontal_angle(project: Scheme, settings: LayoutSettings, start: LayoutParams) -> float:
    placement = place(project, settings, start)
    vectors = [(point.x - placement.primary.x, point.y - placement.primary.y)
               for point in (placement.left, placement.right)]
    (ax, ay), (bx, by) = vectors
    return math.degrees(math.atan2(abs(ax * by - ay * bx), ax * bx + ay * by))


@pytest.mark.parametrize("blocked", [
    {"keep_out": (Box(x=Span(low=0.01, high=10.0), y=Span(low=0.01, high=10.0), z=Span(low=0.01, high=10.0)),)},
    {"wall_gap_m": 0.9},
])
def test_geometry_illegal_start_is_returned_unmoved(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, blocked: dict[str, object],
) -> None:
    """沒設夾角的舊專案：起點撞禁區或牆面間隙時照改動前主線交回 60° 三角形，不推、不丟。"""
    spacing = distance(project.speakers["left"], project.speakers["right"])
    start = standard_start(project, _changed(settings, **blocked))
    assert _start_tuple(start) == pytest.approx((1.0, spacing, spacing * math.sqrt(3.0) / 2.0), rel=1e-12)
    assert spacing == pytest.approx(1.2, rel=1e-12)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("bounds,edges", [
    ((65.0, 90.0), (1.3, 5.0)), ((10.0, 29.0), (1.3, 5.0)),
    # 禁區邊緣貼著座位（起點的前座 x≈1.842、後座 x≈3.420）：夾角推約 2.4°／幾度就合法，也不准推。
    ((65.0, 90.0), (1.8, 5.0)), ((10.0, 29.0), (1.3, 3.45)),
])
def test_seat_keep_out_does_not_push_angle_endpoint(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, bounds: tuple[float, float],
    edges: tuple[float, float],
) -> None:
    """禁區圍住起點座位：夾角留在離 60° 最近的端點（捨入尺度內），不推到禁區外或範圍另一端。"""
    nearest = min(bounds, key=lambda bound: abs(bound - 60.0))
    seat = Box(x=Span(low=edges[0], high=edges[1]), y=Span(low=0.01, high=10.0), z=Span(low=0.01, high=10.0))
    angle = _changed(settings, base_angle_deg=Span(low=bounds[0], high=bounds[1]))
    open_start = standard_start(project, angle)
    blocked = _changed(settings, base_angle_deg=Span(low=bounds[0], high=bounds[1]), keep_out=(seat,))
    start = standard_start(project, blocked)
    assert start is not None and open_start is not None
    assert _start_tuple(start) == pytest.approx(_start_tuple(open_start), rel=1e-9)
    assert _horizontal_angle(project, blocked, start) == pytest.approx(nearest, abs=1e-9)
    assert check(project, blocked, place(project, blocked, start))
    assert tmp_path.is_dir()


@pytest.mark.parametrize("changes", [
    {"keep_out": (Box(x=Span(low=1.5, high=2.5), y=Span(low=0.01, high=4.0), z=Span(low=0.01, high=2.0)),)},
    {"wall_gap_m": 0.9},
])
def test_geometry_illegal_start_is_first_illegal_trial(tmp_path: Path, changes: dict[str, object]) -> None:
    """起點照排：搜尋把它記成第 0 題不合法並寫原因，狀態不准說起點不在搜尋範圍內。"""
    store, registry = make_store(tmp_path, budget=1, batch=1, layout_changes=changes)
    start = standard_start(store.project, store.settings.layout)
    assert start is not None
    status = run(store, registry, FakeCompute(store))
    first = rows(store)[0]
    assert status.start_enqueued
    assert first.trial_number == 0 and first.outcome == "illegal" and first.reason
    assert first.unit_params_hex == {name: value.hex() for name, value in unit_from_params(start, store.settings.layout).items()}
    assert "起點不在搜尋範圍內" not in status.message
