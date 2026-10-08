"""家具違反量的獨立手算答案；判定仍由既有家具配置入口負責。"""
import math

import pytest

from aosr.geometry.furniture import FurnitureBox
from aosr.reporting.scheme import Scheme
from tests.engine._furniture_cases import cloud_item, document, relative_item
from tests.engine._precision_contracts import contract_value
from tests.engine.test_scheme_furniture import _validation_document


@pytest.mark.parametrize("start,end,expected", [
    ((0, 2, 2), (4, 2, 2), 2.0),
    ((4, 2, 2), (0, 2, 2), 2.0),
    ((0, 0, 0), (4, 4, 4), 2.0 * math.sqrt(3.0)),
    ((2, 2, 2), (4, 2, 2), 1.0),
    ((0, 4, 2), (4, 4, 2), 0.0),
    ((0, 2, 2), (1, 2, 2), 0.0),
    ((2, 2, 2), (2, 2, 2), 0.0),
])
def test_slab_length_has_hand_calculated_answers(
    start: tuple[float, float, float], end: tuple[float, float, float], expected: float,
) -> None:
    from aosr.search.furniture_prefilter import segment_box_length_m

    # 盒子三軸皆為 [1,3]；斜穿長度為 (3-1) × sqrt(3)，半段由 x=2 到 x=3。
    box = FurnitureBox(kind="desk", width_m=2, depth_m=2, height_m=2,
                       bottom_center_m=(2, 2, 1), margin_m=0)
    assert segment_box_length_m(start, end, box) == pytest.approx(expected)


def test_blocked_length_takes_maximum_across_all_pairs_and_pieces() -> None:
    from aosr.search.furniture_prefilter import check

    # 主位線向量 (1,-2,0)，周圍點 (1.1,-2,0)；y=[3.85,4.15] 切出 t=0.425～0.575。
    # 主位線兩件各 0.15 × sqrt(5)；周圍點線 z-first 0.15 × sqrt(5.21)，a-second 先從 +x 面 x=2.9 出盒，
    # 只有約 0.1432 × sqrt(5.21)。取最大（z-first 周圍點那一條），不相加。
    for case in ("both", "two_block"):
        violation, = check(Scheme.model_validate(_validation_document(case)),
                           contact_rel=contract_value("furniture_geometry_contact"))
        assert violation.reason.value == "direct_path_blocked"
        assert violation.amount_m == pytest.approx(0.15 * math.sqrt(5.21))


def test_room_overrun_is_the_largest_axis_distance() -> None:
    from aosr.search.furniture_prefilter import check

    # 主位 (3,3)，左偏 4 使盒底中心 x=-1，寬 1 的最小 x=-1.5，越界 1.5 m。
    violation, = check(Scheme.model_validate(_validation_document("outside")),
                       contact_rel=contract_value("furniture_geometry_contact"))
    assert violation.reason.value == "furniture_placement_invalid"
    assert violation.amount_m == 1.5


@pytest.mark.parametrize("left,expected_depth", [(0.2, 0.2), (1.0, 1.0)])
def test_gap_deficit_includes_overlap_depth(left: float, expected_depth: float) -> None:
    from aosr.search.furniture_prefilter import check

    # 第一盒 x=[1.5,2.5]，第二盒左偏 0.2 時 x=[2.3,3.3]，最大分離=-0.2。
    # 同中心時三軸交集最小為高度 0.6；改高 1 使三軸均重疊 1，缺口 = 界線 + 1。
    def seat(name: str, offset: float) -> dict[str, object]:
        return relative_item(furniture_id=name, depth_m=1.0, height_m=1.0, placement={
            "forward_m": -1.0, "left_m": offset, "bottom_height_m": 0, "yaw_deg": 0})

    violation, = check(Scheme.model_validate(document(seat("a", 1.0), seat("b", left))),
                       contact_rel=contract_value("furniture_geometry_contact"))
    margin = 8.0 * contract_value("furniture_geometry_contact")
    assert violation.reason.value == "furniture_placement_invalid"
    assert violation.amount_m == pytest.approx(margin + expected_depth, rel=0.0, abs=margin / 16.0)


def test_contained_boxes_use_common_interval_depth() -> None:
    from aosr.search.furniture_prefilter import check

    # 同中心 (2,2)，外盒寬深 2、高 1，內盒寬 1、深 0.5、高 0.6。
    # 三軸交集為 1、0.5、0.6，最大分離=-0.5，缺口為界線加 0.5。
    placement = {"forward_m": -1, "left_m": 1, "bottom_height_m": 0, "yaw_deg": 0}
    outer = relative_item(furniture_id="outer", width_m=2, depth_m=2, height_m=1, placement=placement)
    inner = relative_item(furniture_id="inner", placement=placement)
    violation, = check(Scheme.model_validate(document(outer, inner)),
                       contact_rel=contract_value("furniture_geometry_contact"))
    margin = 8.0 * contract_value("furniture_geometry_contact")
    assert violation.amount_m == pytest.approx(margin + 0.5, rel=0.0, abs=margin / 16.0)


def test_placement_amount_takes_maximum_and_precedes_real_blockage(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.furniture_prefilter import check
    from aosr.reporting import furniture_layout

    outside = Scheme.model_validate(_validation_document("outside"))
    blocked = Scheme.model_validate(_validation_document("both"))
    assert outside.furniture and blocked.furniture
    inside = relative_item(furniture_id="inside", height_m=2, placement={
        "forward_m": 0, "left_m": 0, "bottom_height_m": 0, "yaw_deg": 0})
    # 越界 1.5 m、耳朵深度 0.25 m，且原先那塊桌板擋左直達；只回擺放錯與最大 1.5。
    scheme = Scheme.model_validate(document(outside.furniture[0].model_dump(), inside,
                                            blocked.furniture[0].model_dump()))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("擺放錯不得接著判直達")

    monkeypatch.setattr(furniture_layout, "direct_blockers", forbidden)
    violation, = check(scheme, contact_rel=contract_value("furniture_geometry_contact"))
    assert violation.reason.value == "furniture_placement_invalid" and violation.amount_m == 1.5


def test_point_inside_box_uses_distance_to_nearest_face() -> None:
    from aosr.search.furniture_prefilter import check

    # 耳朵 (3,3,1.2)，盒子 x=[2.5,3.5]、y=[2.75,3.25]、z=[0,2]，最近面距離 0.25。
    item = relative_item(height_m=2, placement={
        "forward_m": 0, "left_m": 0, "bottom_height_m": 0, "yaw_deg": 0})
    violation, = check(Scheme.model_validate(document(item)),
                       contact_rel=contract_value("furniture_geometry_contact"))
    assert violation.reason.value == "furniture_placement_invalid"
    assert violation.amount_m == 0.25


def test_speaker_inside_box_uses_distance_to_nearest_face() -> None:
    from aosr.search.furniture_prefilter import check

    # 主位 (3,3) 面向 +y：前方 2、左方 1 是 (2,5)；桌板 x=[1.85,2.15]、y=[4.85,5.15]、z=[1.0,1.5]。
    # 左喇叭 (2,5,1.2) 在盒中心，到最近面（四個側面）0.15。書桌跟著主位走、聆聽距離短時搜尋真的會遇到。
    item = relative_item(furniture_id="desk", kind="desk", material="wood", width_m=0.3, depth_m=0.3, height_m=0.5,
                         placement={"forward_m": 2, "left_m": 1, "bottom_height_m": 1.0, "yaw_deg": 0})
    violation, = check(Scheme.model_validate(document(item)),
                       contact_rel=contract_value("furniture_geometry_contact"))
    assert violation.reason.value == "furniture_placement_invalid"
    assert violation.amount_m == pytest.approx(0.15, abs=1e-12)


def test_ceiling_overrun_is_measured_on_the_vertical_axis() -> None:
    from aosr.search.furniture_prefilter import check

    # 房高 4 m；天雲底 3.95、厚 0.1，頂 4.05，超出天花板 0.05。
    item = cloud_item(placement={"bottom_center_m": [2.0, 2.0, 3.95], "yaw_deg": 90})
    violation, = check(Scheme.model_validate(document(item)),
                       contact_rel=contract_value("furniture_geometry_contact"))
    assert violation.reason.value == "furniture_placement_invalid"
    assert violation.amount_m == pytest.approx(0.05, abs=1e-12)


def test_gap_boundary_uses_contact_margin_as_positive_floor() -> None:
    from aosr.search.furniture_prefilter import check

    margin = 8.0 * contract_value("furniture_geometry_contact")
    first = relative_item(furniture_id="a", placement={
        "forward_m": -1, "left_m": 1, "bottom_height_m": 0, "yaw_deg": 0})
    second = relative_item(furniture_id="b", placement={
        "forward_m": -1, "left_m": -margin, "bottom_height_m": 0, "yaw_deg": 0})
    violation, = check(Scheme.model_validate(document(first, second)),
                       contact_rel=contract_value("furniture_geometry_contact"))
    assert violation.reason.value == "furniture_placement_invalid"
    assert violation.amount_m == margin


@pytest.mark.parametrize("depth,blocked", [(0.5, False), (2.0, True)])
def test_grazing_uses_the_layout_contact_boundary(depth: float, blocked: bool) -> None:
    from aosr.search.furniture_prefilter import check

    margin = 8.0 * contract_value("furniture_geometry_contact")
    content = _validation_document("both")
    scheme = Scheme.model_validate(content)
    assert scheme.furniture
    item = scheme.furniture[0].model_dump() | {"placement": {
        "forward_m": 1, "left_m": 0.5, "bottom_height_m": 1.2 - depth * margin, "yaw_deg": 0}}
    violations = check(Scheme.model_validate(content | {"furniture": [item]}),
                       contact_rel=contract_value("furniture_geometry_contact"))
    assert {item.reason.value for item in violations} == ({"direct_path_blocked"} if blocked else set())


def test_placement_failure_wins_without_inspecting_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import furniture_prefilter
    from aosr.reporting import furniture_layout

    def placement_error(*args: object, **kwargs: object) -> None:
        raise ValueError("direct_path_blocked：刻意誤導的任意訊息")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("擺放錯時不得接著判直達")

    monkeypatch.setattr(furniture_layout, "furniture_boxes", placement_error)
    monkeypatch.setattr(furniture_layout, "direct_blockers", forbidden)
    violation, = furniture_prefilter.check(Scheme.model_validate(_validation_document("outside")),
                                          contact_rel=contract_value("furniture_geometry_contact"))
    assert violation.reason.value == "furniture_placement_invalid"


def test_unmeasurable_layout_failure_raises_instead_of_inventing_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.search import furniture_prefilter
    from aosr.reporting import furniture_layout

    def unknown_failure(*args: object, **kwargs: object) -> None:
        raise ValueError("任意訊息")

    monkeypatch.setattr(furniture_layout, "furniture_boxes", unknown_failure)
    with pytest.raises(ValueError):
        furniture_prefilter.check(Scheme.model_validate(document(cloud_item())),
                                  contact_rel=contract_value("furniture_geometry_contact"))


def test_invalid_bottom_raises_instead_of_inventing_amount() -> None:
    from aosr.search.furniture_prefilter import check

    item = relative_item(placement={
        "forward_m": -1, "left_m": 1, "bottom_height_m": 0.1, "yaw_deg": 0})
    with pytest.raises(ValueError):
        check(Scheme.model_validate(document(item)), contact_rel=contract_value("furniture_geometry_contact"))
