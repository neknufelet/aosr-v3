"""伺服器圖面：手算四角、高度、舊契約、錯誤仍畫與絕對盒子改動。"""
import re
from math import sqrt
from pathlib import Path
from typing import cast

import pytest

from aosr.gui.plan_view import plan_for
from aosr.reporting.scheme import Scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import cloud_item, document, relative_item
from tests.engine._gui_furniture_drawings import CABINET, listening_document, working_document
from tests.engine.test_gui_app import _app
from tests.engine.test_scheme_furniture import _validation_document


def test_plan_furniture_hand_calculated_corners_and_heights() -> None:
    plan = plan_for(Scheme.model_validate(listening_document()), DIRECTIVITY)
    items = {str(item["id"]): item for item in cast(list[dict[str, object]], plan["furniture"])}
    # 主位 (3.2,1.9) 朝 −x，沙發後退 0.2；90° 時沿 x 的尺寸是深 0.75。
    expected = {
        "sofa": ([(3.025, 1.0), (3.775, 1.0), (3.775, 2.8), (3.025, 2.8)], (0.0, 0.65)),
        "table": ([(2.05, 1.4), (2.55, 1.4), (2.55, 2.4), (2.05, 2.4)], (0.4, 0.44)),
        "cloud": ([(1.5, 1.3), (2.7, 1.3), (2.7, 2.5), (1.5, 2.5)], (2.55, 2.65)),
    }
    assert items.keys() == expected.keys()
    for name, (corners, heights) in expected.items():
        for actual, expected_corner in zip(cast(tuple[tuple[float, float], ...], items[name]["polygon"]), corners, strict=True):
            assert actual == pytest.approx(expected_corner)
        assert (items[name]["bottom_m"], items[name]["top_m"]) == pytest.approx(heights)
        side = [
            (corners[0][0], heights[0]), (corners[1][0], heights[0]),
            (corners[1][0], heights[1]), (corners[0][0], heights[1])]
        for actual, expected_corner in zip(cast(tuple[tuple[float, float], ...], items[name]["side_polygon"]), side, strict=True):
            assert actual == pytest.approx(expected_corner)


@pytest.mark.parametrize("yaw", [90, 270])
def test_plan_quarter_turn_uses_shared_extents_and_keeps_invalid_bottom_drawable(yaw: int) -> None:
    content = document(relative_item(placement={
        "forward_m": 1.0, "left_m": 0.5, "bottom_height_m": 0.5, "yaw_deg": yaw}))
    plan = plan_for(Scheme.model_validate(content), DIRECTIVITY)
    item = cast(list[dict[str, object]], plan["furniture"])[0]
    # 朝 +y；兩個直角轉向的全長都為 x=0.5、y=1，底面雖填錯仍須照畫。
    for actual, expected in zip(cast(tuple[tuple[float, float], ...], item["polygon"]),
                                [(2.25, 3.5), (2.75, 3.5), (2.75, 4.5), (2.25, 4.5)], strict=True):
        assert actual == pytest.approx(expected)


def test_plan_cabinet_faces_primary_with_hand_calculated_three_four_five_triangle() -> None:
    content = document(relative_item(), cloud_item(), facing="west", primary=(4.0, 5.0, 1.2))
    content["speakers"] = {"left": {"x": 1.0, "y": 1.0, "z": 1.2},
                            "right": {"x": 1.0, "y": 7.0, "z": 1.2}}
    content["speaker_setup"] = {"kind": "bookshelf", "mount": "stand", "cabinet": CABINET,
                                "representative": True}
    content["source_model"] = "omnidirectional"
    plan = plan_for(Scheme.model_validate(content), DIRECTIVITY)
    left = next(item for item in cast(list[dict[str, object]], plan["cabinets"]) if item["id"] == "left")
    # 朝向 (3/5,4/5)，右邊向量 (4/5,−3/5)，箱深 0.28，半寬 0.105。
    corners = [(0.916, 0.713), (1.084, 0.937), (0.916, 1.063), (0.748, 0.839)]
    for actual, expected_corner in zip(cast(tuple[tuple[float, float], ...], left["polygon"]), corners, strict=True):
        assert actual == pytest.approx(expected_corner)
    assert (left["bottom_m"], left["top_m"]) == pytest.approx((0.995, 1.35))
    assert "代表模型，非實際型號" in str(left["detail_text"])
    assert "箱體只做碰撞檢查，反射暫不計" in str(left["detail_text"])


def test_plan_desk_and_floor_bottom_use_support_even_when_source_height_is_wrong() -> None:
    for mount, bottom in (("desk", 0.75), ("floor", 0.0)):
        content = working_document()
        # 故意填錯高度：1.2−0.205＝0.995，不能冒充桌面頂 0.75 或地面 0。
        content["speakers"] = {"left": {"x": 1.7, "y": 1.5, "z": 1.2},
                               "right": {"x": 1.7, "y": 2.3, "z": 1.2}}
        content["speaker_setup"] = {"kind": "floorstanding" if mount == "floor" else "bookshelf",
                                    "mount": mount, "cabinet": CABINET, "representative": False}
        plan = plan_for(Scheme.model_validate(content), DIRECTIVITY)
        for cabinet in cast(list[dict[str, object]], plan["cabinets"]):
            assert cabinet["bottom_m"] == bottom
            assert cabinet["top_m"] == pytest.approx(bottom + 0.355)
            # 水平朝向 (7,±4)/√65；兩支沿 x 的投影相同，答案獨立手算。
            low = 1.7 - (0.28 * 7 + 0.105 * 4) / sqrt(65)
            high = 1.7 + 0.105 * 4 / sqrt(65)
            side = [(low, bottom), (high, bottom), (high, bottom + 0.355), (low, bottom + 0.355)]
            for actual, expected in zip(cast(tuple[tuple[float, float], ...], cabinet["side_polygon"]), side, strict=True):
                assert actual == pytest.approx(expected)


@pytest.mark.parametrize("case", ["outside", "both"])
def test_plan_invalid_layout_returns_drawing_and_problems_without_allowing_save(tmp_path: Path, case: str) -> None:
    content = _validation_document(case) | {"scheme_id": "invalid"}
    if case == "both":
        content["furniture"] = [*cast(list[dict[str, object]], content["furniture"]), cloud_item()]
    with _app(tmp_path) as client:
        response = client.post("/api/plan", json=content)
        assert response.status_code == 422
        payload = response.json()
        assert payload["problems"]
        assert payload["plan"]["furniture"]
        assert client.put("/api/schemes/invalid", json=content).status_code == 422
    if case == "both":
        assert {(path["speaker_id"], path["receiver_id"], tuple(path["furniture_ids"]))
                for path in payload["plan"]["blocked_paths"]} == {
                    ("left", "main", ("desk",)), ("left", "side", ("desk",))}
        assert {item["id"] for item in payload["plan"]["furniture"] if item["blocked"]} == {"desk"}
        assert next(item for item in payload["plan"]["furniture"] if item["id"] == "cloud")["blocked"] is False


def test_plan_only_undefined_cabinet_is_missing_and_note_is_chinese(tmp_path: Path) -> None:
    content = listening_document()
    content.pop("furniture")
    content["speakers"] = {"left": {"x": 3.2, "y": 1.9, "z": 0.5},
                           "right": {"x": 1.0, "y": 2.5, "z": 1.2}}
    with _app(tmp_path) as client:
        response = client.post("/api/plan", json=content)
    assert response.status_code == 200
    plan = response.json()
    assert plan["furniture"] == []
    assert {item["id"] for item in plan["cabinets"]} == {"right"}
    assert plan["drawing_notes"] == ["左聲道喇叭箱體沒畫：喇叭跟主位在同一個水平位置，定不出朝向"]
    assert not re.search(r"[A-Za-z]|讀不到", "".join(plan["drawing_notes"]))


def test_plan_object_details_use_two_decimal_places() -> None:
    plan = plan_for(Scheme.model_validate(listening_document()), DIRECTIVITY)
    for item in cast(list[dict[str, object]], plan["furniture"]) + cast(list[dict[str, object]], plan["cabinets"]):
        detail = str(item["detail_text"])
        values = re.findall(r"(?:寬|深|高|底面|頂面) (\d+\.\d+)", detail)
        assert values
        assert all(re.fullmatch(r"\d+\.\d{2}", value) for value in values), detail


def test_plan_outside_and_blocked_lists_both_furniture_problems(tmp_path: Path) -> None:
    content = _validation_document("both") | {"scheme_id": "mix"}
    outside = relative_item(placement={"forward_m": 1, "left_m": 4, "bottom_height_m": 0, "yaw_deg": 0})
    content["furniture"] = [*cast(list[dict[str, object]], content["furniture"]), outside]
    with _app(tmp_path) as client:
        response = client.post("/api/plan", json=content)
        assert client.put("/api/schemes/mix", json=content).status_code == 422
    assert response.status_code == 422
    payload = response.json()
    assert {problem["text"] for problem in payload["problems"]} == {
        "家具：家具 seat 超出房間接觸界線",
        "左聲道喇叭 → 主位：不符合擺位要求：直達路徑被家具 desk 擋住",
        "左聲道喇叭 → 座位 side：不符合擺位要求：直達路徑被家具 desk 擋住"}
    assert {item["id"] for item in payload["plan"]["furniture"] if item["blocked"]} == {"desk"}


def test_plan_unconvertible_shape_returns_only_problems(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        payload = client.post("/api/plan", json={"scheme_id": "bad"}).json()
    assert payload["problems"] and "plan" not in payload


def test_changed_furniture_uses_absolute_boxes_after_primary_moves() -> None:
    from aosr.gui.plan_view import changed_furniture_keys

    first = Scheme.model_validate(document(relative_item(), cloud_item()))
    moved = Scheme.model_validate(document(relative_item(), cloud_item(), primary=(3.2, 3.0, 1.2)))
    assert first.furniture == moved.furniture
    assert changed_furniture_keys(first, moved) == ("furniture:seat",)
    assert changed_furniture_keys(first, first) == ()


def test_changed_furniture_ignores_material_and_equivalent_half_turn() -> None:
    from aosr.gui.plan_view import changed_furniture_keys

    first = Scheme.model_validate(document(relative_item(), cloud_item()))
    second = Scheme.model_validate(document(
        relative_item(material="leather", placement={"forward_m": 1.0, "left_m": 0.5,
                                                     "bottom_height_m": 0.0, "yaw_deg": 180}),
        cloud_item(placement={"bottom_center_m": [2.0, 2.0, 2.5], "yaw_deg": 270})))
    assert first.furniture != second.furniture
    assert changed_furniture_keys(first, second) == ()


@pytest.mark.parametrize("change", [{"height_m": 0.8}, {"placement": {
    "bottom_center_m": [2.0, 2.0, 2.6], "yaw_deg": 90}}])
def test_changed_furniture_marks_height_or_bottom_without_footprint_change(change: dict[str, object]) -> None:
    from aosr.gui.plan_view import changed_furniture_keys

    first = Scheme.model_validate(document(relative_item(), cloud_item()))
    second = Scheme.model_validate(document(relative_item(), cloud_item(**change)))
    first_plan, second_plan = plan_for(first, DIRECTIVITY), plan_for(second, DIRECTIVITY)
    assert [item["polygon"] for item in cast(list[dict[str, object]], first_plan["furniture"])] == [
        item["polygon"] for item in cast(list[dict[str, object]], second_plan["furniture"])]
    assert changed_furniture_keys(first, second) == ("furniture:cloud",)


def test_saved_plan_drawing_does_not_read_files(monkeypatch: pytest.MonkeyPatch) -> None:
    scheme = Scheme.model_validate(listening_document())

    def forbid_read(*args: object, **kwargs: object) -> None:
        pytest.fail("已存方案畫圖不應讀檔")

    monkeypatch.setattr(Path, "open", forbid_read)
    plan = plan_for(scheme, DIRECTIVITY)
    assert plan["furniture"] and plan["cabinets"]
