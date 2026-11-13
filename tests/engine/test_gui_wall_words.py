"""牆名跟著方案走（#1134）：前牆照 /search-start 前牆預設那一條推（喇叭中點最近的牆），左右是面向前牆時的左右。

109 part2d 的擺位當考卷（左喇叭 y 0.94、主位面向 x0）：x0＝前牆、xL＝後牆、y0＝左牆、yL＝右牆。
同一間房轉 90、180、270 度再考一次——只考 109 分不出「推出來的」與「寫死 x0＝前牆」。
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, expect
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.labels import WALLS
from aosr.gui.wall_words import front_wall, wall_words
from aosr.reporting.scheme import Scheme
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser

# 109 part2d 存檔裡的喇叭與主位（公尺）。
LEFT_109 = (0.36810808858895405, 0.937648486381631)
RIGHT_109 = (0.36810808858895405, 3.4822329683114788)
PRIMARY_109 = (2.3865101734352736, 2.209940727346555)
Doc = dict[str, object]


def _part2d_layout() -> Doc:
    """109 號房、part2d 的喇叭與主位；周圍點跟著主位平移。"""
    document = room109_document("part2d-layout")
    speakers = cast(dict[str, dict[str, float]], document["speakers"])
    for name, (x, y) in (("left", LEFT_109), ("right", RIGHT_109)):
        speakers[name]["x"], speakers[name]["y"] = x, y
    points = cast(list[dict[str, object]], cast(Doc, document["receiver_set"])["points"])
    primary = next(cast(list[float], point["position_m"]) for point in points if point["role"] == "primary")
    dx, dy = PRIMARY_109[0] - primary[0], PRIMARY_109[1] - primary[1]
    for point in points:
        position = cast(list[float], point["position_m"])
        point["position_m"] = [position[0] + dx, position[1] + dy, position[2]]
    return document


def _turned(document: Doc, quarter_turns: int) -> Scheme:
    """整間房連喇叭、座位逆時針轉 90 度 × 次數（左右手不變，不是鏡像）：(x, y) → (Ly − y, x)，房間長寬互換。"""
    for _ in range(quarter_turns % 4):
        room = cast(dict[str, float], cast(Doc, document["scene"])["room_m"])
        ly = room["Ly"]

        def turn(x: float, y: float, ly: float = ly) -> tuple[float, float]:
            return ly - y, x

        for point in cast(dict[str, dict[str, float]], document["speakers"]).values():
            point["x"], point["y"] = turn(point["x"], point["y"])
        for receiver in cast(list[dict[str, object]], cast(Doc, document["receiver_set"])["points"]):
            position = cast(list[float], receiver["position_m"])
            receiver["position_m"] = [*turn(position[0], position[1]), position[2]]
        room["Lx"], room["Ly"] = room["Ly"], room["Lx"]
    return Scheme.model_validate(document)


def test_part2d_layout_names_x0_front_y0_left() -> None:
    names = wall_words(_turned(_part2d_layout(), 0))
    assert names == {"floor": "地板", "ceiling": "天花", "x0": "前牆", "xL": "後牆", "y0": "左牆", "yL": "右牆"}


@pytest.mark.parametrize(("turns", "front", "back", "left", "right"), [
    (1, "y0", "yL", "xL", "x0"), (2, "xL", "x0", "yL", "y0"), (3, "yL", "y0", "x0", "xL")])
def test_turned_room_moves_every_name_with_the_layout(turns: int, front: str, back: str, left: str, right: str) -> None:
    scheme = _turned(_part2d_layout(), turns)
    names = wall_words(scheme)
    assert (names[front], names[back], names[left], names[right]) == ("前牆", "後牆", "左牆", "右牆")
    # 獨立核一次「左」：左聲道喇叭離左牆比離右牆近（不經過 LEFT_OF 那張表）。
    room = scheme.scene.room_m
    speaker = scheme.speakers["left"]
    distance = {"x0": speaker.x, "xL": room.Lx - speaker.x, "y0": speaker.y, "yL": room.Ly - speaker.y}
    assert distance[left] < distance[right]


def test_names_fall_back_to_coordinates_when_the_two_rules_disagree_or_tie() -> None:
    # 主位坐在喇叭和最近那面牆之間：喇叭靠 x0，人卻面向 +x，兩套說法不同 → 不挑一套。
    document = _part2d_layout()
    for receiver in cast(list[dict[str, object]], cast(Doc, document["receiver_set"])["points"]):
        position = cast(list[float], receiver["position_m"])
        receiver["position_m"] = [position[0] - 2.2, position[1], position[2]]
    behind = Scheme.model_validate(document)
    assert front_wall(behind) is None and wall_words(behind) == WALLS
    # 喇叭中點離 x0 與 y0 一樣近：平手不猜。
    tied = _part2d_layout()
    speakers = cast(dict[str, dict[str, float]], tied["speakers"])
    middle_y = (speakers["left"]["y"] + speakers["right"]["y"]) / 2
    for point in speakers.values():
        point["x"] = middle_y
    room = cast(dict[str, float], cast(Doc, tied["scene"])["room_m"])
    assert room == ROOM_109
    assert wall_words(Scheme.model_validate(tied)) == WALLS


def test_input_preview_hands_the_page_the_names(tmp_path: Path) -> None:
    """首頁表單每改一格就問的那一支帶牆名（圖面資料 /api/plan 是凍結的答案，不加欄位）。"""
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost")
    document = _turned(_part2d_layout(), 2).model_dump(mode="json")
    preview = client.post("/api/input-preview", json=document)
    assert preview.status_code == 200, preview.text
    names = preview.json()["wall_names"]
    assert (names["xL"], names["x0"], names["yL"], names["y0"]) == ("前牆", "後牆", "左牆", "右牆")


def test_home_form_renames_walls_when_the_speakers_move_to_the_other_end(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        # 範本（參考房）喇叭靠 x0、主位面向 x0。
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        expect(page.locator("#wall-editor-y0")).to_contain_text("左牆材料")
        expect(page.locator("#elevation-wall-xL")).to_have_text("後牆")
        # 兩支喇叭搬到主位另一邊、靠 xL：前後左右整組換過來，表單與立面圖按鈕都跟著改。
        for name in ("left", "right"):
            page.locator(f"#speaker-{name}-x").fill("5")
        expect(page.locator("#wall-editor-xL")).to_contain_text("前牆材料")
        expect(page.locator("#wall-editor-x0")).to_contain_text("後牆材料")
        expect(page.locator("#wall-editor-yL")).to_contain_text("左牆材料")
        expect(page.locator("#elevation-wall-y0")).to_have_text("右牆")
        expect(page.locator("#scatter-xL").locator("xpath=..")).to_contain_text("前牆散射")
        _assert_quiet(watched)
