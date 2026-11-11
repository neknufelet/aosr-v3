"""#1092 分區結果頁的純函式：原牆基準怎麼推、逐區表、改善多少的正負號與缺分數；入口與畫面在 test_gui_zone_result_browser。"""
from __future__ import annotations


import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui.zone_design import DesignSettings, ZoneDesign
from aosr.gui.zone_result import BaselineRefused, _change, derive_baseline, improvement, zone_rows
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics.report_surfaces import RectangleZone, WholeWallZone
from aosr.physics.zone_fill import wall_size_m
from aosr.reporting.compare import compare_results
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._zone_result_cases import design_scheme, mounted, original_x0
from tests.engine._zoned_report_cases import small_scheme, split_x0, with_x0
from tests.engine.test_scheme_pipeline import shared_control_result

OTHER = ConstantImpedance(impedance_pa_s_m=2000.0)


def _boxes(scheme: Scheme) -> dict[str, list[tuple[object, ...]]]:
    zones = scheme.scene.surface_zones_by_wall or {}
    return {wall: [(zone.area, zone.id, zone.u_m, zone.v_m) if isinstance(zone, RectangleZone) else (zone.area,)
                   for zone in items] for wall, items in zones.items()}


def _with_wall(scheme: Scheme, wall: str, zones: tuple[WholeWallZone | RectangleZone, ...]) -> Scheme:
    surfaces = dict(scheme.scene.surface_zones_by_wall or {})
    surfaces[wall] = zones
    scene = scheme.scene.model_copy(update={"surface_zones_by_wall": surfaces})
    return Scheme.model_validate(scheme.model_copy(update={"scene": scene}).model_dump())


def test_unwrap_baseline_keeps_every_zone_box_and_only_swaps_materials() -> None:
    """同一套切法：每一面牆的區代號、座標一字不動；只有 a 區拆掉加貼層換回原牆，其餘照抄（同網格才比得，#1088）。"""
    design = design_scheme()
    baseline = derive_baseline(design)
    assert _boxes(baseline.scheme) == _boxes(design)
    x0 = {zone.zone_id: zone for zone in baseline.zones if zone.wall == "x0"}
    assert x0["a"].baseline == original_x0() and x0["a"].source == "unwrapped" and x0["a"].design != x0["a"].baseline
    assert x0["b"].baseline == x0["b"].design == original_x0() and x0["b"].source == "unchanged"
    assert all(zone.design == zone.baseline for zone in baseline.zones if zone.wall != "x0")
    assert baseline.changed
    # 方案其餘每一欄都等於設計：換回設計的分區與代號就逐欄相等。
    restored = baseline.scheme.model_copy(update={"scheme_id": design.scheme_id, "scene": baseline.scheme.scene.model_copy(
        update={"surface_zones_by_wall": design.scene.surface_zones_by_wall})})
    assert restored == design


def test_baseline_id_is_stable_and_follows_baseline_content() -> None:
    """代號＝設計代號＋-orig-＋內容雜湊 8 碼：同內容同代號（舊基準結果可沿用），原牆一換代號就換（舊的不會被誤認）。"""
    design = design_scheme()
    first = derive_baseline(design).scheme.scheme_id
    assert first == derive_baseline(design).scheme.scheme_id
    prefix, digest = first.rsplit("-orig-", 1)
    assert prefix == design.scheme_id and len(digest) == 8 and all(char in "0123456789abcdef" for char in digest)
    thicker = _with_wall(design, "x0", split_x0(mounted(original_x0(), 0.1), original_x0()))
    assert derive_baseline(thicker).scheme == derive_baseline(design).scheme  # 加貼層怎麼改，原牆都一樣
    other_wall = _with_wall(design, "x0", split_x0(mounted(OTHER), original_x0()))
    assert derive_baseline(other_wall).scheme.scheme_id != first


def test_pointer_lays_the_original_schemes_materials_on_the_design_partition() -> None:
    design = design_scheme()
    original = _with_wall(small_scheme(zoned=True), "x0", (WholeWallZone(area="whole_wall", material=OTHER),))
    baseline = derive_baseline(design, original)
    assert _boxes(baseline.scheme) == _boxes(design)
    x0 = [zone for zone in baseline.zones if zone.wall == "x0"]
    assert [(zone.zone_id, zone.baseline, zone.source) for zone in x0] == [
        ("a", OTHER, "original_scheme"), ("b", OTHER, "original_scheme")]
    assert baseline.original_scheme_id == original.scheme_id


def test_pointer_refuses_when_an_original_cut_crosses_a_design_zone() -> None:
    """原方案 x0 在 u=1 切開：設計的 a（0～2/3）在左半裡、b（2/3～2）跨過切線，拒收並點名 b；不猜要取哪一邊。"""
    design = design_scheme()
    left = RectangleZone(area="rectangle", id="l", u_m=(0.0, 1.0), v_m=(0.0, 1.5), material=OTHER)
    right = RectangleZone(area="rectangle", id="r", u_m=(1.0, 2.0), v_m=(0.0, 1.5), material=original_x0())
    with pytest.raises(BaselineRefused, match="劃過設計的「b」"):
        derive_baseline(design, with_x0((left, right)))
    width, height = wall_size_m(design.scene.room_m, "floor")
    floor = tuple(RectangleZone(area="rectangle", id=name, u_m=u, v_m=(0.0, height), material=OTHER)
                  for name, u in (("p", (0.0, width / 2)), ("q", (width / 2, width))))
    with pytest.raises(BaselineRefused, match="地板的切線劃過設計的「整面」"):
        derive_baseline(design, _with_wall(small_scheme(zoned=True), "floor", floor))
    room = design.scene.room_m
    bigger = small_scheme(zoned=True)
    bigger = Scheme.model_validate(bigger.model_copy(update={"scene": bigger.scene.model_copy(update={
        "room_m": type(room)(room.Lx + 0.5, room.Ly, room.Lz)})}).model_dump())
    with pytest.raises(BaselineRefused, match="房間尺寸跟設計不同"):
        derive_baseline(design, bigger)


def test_nothing_to_compare_when_no_zone_has_an_added_layer() -> None:
    assert not derive_baseline(small_scheme(zoned=True)).changed


def test_zone_rows_place_size_area_mode_and_where_the_original_comes_from() -> None:
    design = design_scheme()
    baseline = derive_baseline(design)
    settings = DesignSettings(scheme_id=design.scheme_id, zones={"x0": {"a": ZoneDesign(mode="search")}})
    rows = {(row.wall, row.zone): row for row in zone_rows(baseline, settings, design)}
    a, b = rows[("x 起點牆", "a")], rows[("x 起點牆", "b")]
    assert (a.place, a.size, a.area) == ("寬向 0.00–0.67、高度 0.00–1.50", "0.67 × 1.50 m", "1.00 m²")
    assert (a.mode, a.changed, a.baseline_source) == ("交給搜尋", True, "拆掉這一區加貼的層")
    assert a.design.startswith("貼在原牆上的構造") and a.baseline.startswith("固定阻抗")
    assert (b.mode, b.changed, b.baseline, b.baseline_source) == (
        "固定這一組材料", False, "同左", "兩邊一樣：這一區沒有加貼的層，當成沒動")
    assert rows[("地板", "整面")].place.startswith("長向 0.00–") and len(rows) == 5 + 2
    unknown = zone_rows(baseline, settings, design, baseline_known=False)
    assert {(row.baseline, row.baseline_source, row.changed) for row in unknown} == {("—", "—", False)}
    stale = DesignSettings(scheme_id=design.scheme_id, zones={"x0": {"gone": ZoneDesign(mode="search")}})
    assert {row.mode for row in zone_rows(baseline, stale, design)} == {"—"}


@pytest.mark.parametrize(("baseline", "design", "text", "verdict"), (
    (2.0, 1.0, "改善 1.0000（少 50.0%）", "better"),
    (1.0, 1.25, "變差 0.2500（多 25.0%）", "worse"),
    (0.0, 0.1, "變差 0.1000（原牆是 0，不算百分比）", "worse"),
    (0.5, 0.5, "一樣", "same"),
    (1.0, 1.0 - 1e-6, "改善 1.00e-06（少 0.0%）", "better"),
    (None, 0.3, "缺分數，不比", "missing"),
    (0.3, None, "缺分數，不比", "missing"),
))
def test_change_is_design_minus_baseline_and_lower_is_better(baseline: float | None, design: float | None,
                                                             text: str, verdict: str) -> None:
    assert _change(baseline, design) == (text, verdict)


@pytest.fixture(scope="module")
def control_pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


def _costs(result: SchemeResult) -> tuple[dict[str, float], float]:
    registry = load_quality_targets(config_path("quality_targets.toml"))
    row = compare_results((result,), quality_targets=registry, run_date=result.run_date).rankable[0]
    from aosr.gui.labels import LABELS
    return {LABELS[line.identity.category.value]: line.category_cost for line in row.categories}, row.total_cost


def test_improvement_numbers_equal_each_result_page_and_the_sign_flips_with_the_sides(
        control_pair: tuple[SchemeResult, SchemeResult]) -> None:
    """逐類數字＝各自單獨排一次（結果頁同一個取法）；對調兩邊，改善變變差、一樣還是一樣。"""
    first, second = control_pair
    registry = load_quality_targets(config_path("quality_targets.toml"))
    view = improvement(first, second, registry)
    first_costs, first_total = _costs(first)
    second_costs, second_total = _costs(second)
    assert {row.category: (row.baseline, row.design) for row in view.rows} == {
        name: (f"{second_costs[name]:.4f}", f"{first_costs[name]:.4f}") for name in first_costs}
    assert view.total is not None and view.total_note == ""
    assert (view.total.baseline, view.total.design) == (f"{second_total:.4f}", f"{first_total:.4f}")
    assert view.total.verdict == ("better" if first_total < second_total else "worse")
    flipped = {"better": "worse", "worse": "better", "same": "same", "missing": "missing"}
    swapped = improvement(second, first, registry)
    assert [flipped[row.verdict] for row in view.rows] == [row.verdict for row in swapped.rows]
    assert {row.verdict for row in view.rows} >= {"better"} or {row.verdict for row in view.rows} >= {"worse"}


def test_improvement_without_a_category_cost_shows_no_total_and_no_percent(
        control_pair: tuple[SchemeResult, SchemeResult]) -> None:
    first, second = control_pair
    kept = tuple(item for item in second.candidate.evaluations if item.category.value != "reverberation")
    missing = second.model_copy(update={"candidate": second.candidate.model_copy(update={"evaluations": kept})})
    view = improvement(first, missing, load_quality_targets(config_path("quality_targets.toml")))
    reverberation = next(row for row in view.rows if row.category == "殘響")
    assert (reverberation.baseline, reverberation.change, reverberation.verdict) == ("—", "缺分數，不比", "missing")
    assert view.total is None and view.total_note.startswith("總分不比：原牆基準")


def test_improvement_refuses_results_from_different_physics(control_pair: tuple[SchemeResult, SchemeResult]) -> None:
    first, second = control_pair
    older = second.model_copy(update={"physics_identity": "phys-v1:" + "0" * 64})
    with pytest.raises(ValueError, match="物理計算的程式或設定不同"):
        improvement(first, older, load_quality_targets(config_path("quality_targets.toml")))
