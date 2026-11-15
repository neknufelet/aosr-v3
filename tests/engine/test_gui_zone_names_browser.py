"""真瀏覽器考區名與號碼（#1176 第 B 支，老闆 11-15「自動 auto3」看不出是牆上哪一塊）：立面圖與做法表每一區同一個號碼
①②③…和一句位置名，表上名字底下小字代號（伺服器的訊息還寫代號）。

假房間六面（``_zone_naming_room`` 的 zone-naming-sample，數字是考卷自己編的）每一區的號碼與名字釘死：前牆（yL）5 區、
後牆 5 區（鏡像：畫面左緣是右牆）、右牆 8 區含門與窗（鏡像）、左牆整面一區、地板 4 區含地毯、天花 6 區含雲板與吊櫃。
材料庫是同一份檔的 5 筆假材料（門、窗、地毯、木地板、天花吊頂）。
另外幾個手造的直接考純函式 ``nameZones``：喇叭在左邊那塊、整面寬的長條、兩塊中間的直條、牆名定不出前後時不用喇叭與主位。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME, write_zone_naming_library
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write

SAMPLE = "zone-naming-sample"
# 每一面照號碼排：（區代號、號碼、位置名）。左右照立面圖上看到的（站在房間裡看那一面），「靠○牆」是那一邊印的牆名。
SAMPLE_NAMES: dict[str, list[tuple[str, int, str]]] = {
    "yL": [("z1", 1, "左聲道喇叭後面那塊"), ("z2", 2, "右聲道喇叭後面那塊"), ("z3", 3, "主位正前方那塊"),
           ("auto2", 4, "兩支喇叭中間那塊"), ("auto1", 5, "兩支喇叭中間那塊")],
    "y0": [("z1", 1, "主位正後方那塊"), ("auto3", 2, "靠右牆那條"), ("auto4", 3, "靠天花那條"), ("auto2", 4, "靠左牆那條"),
           ("auto1", 5, "靠地板那條")],
    "xL": [("window", 1, "窗"), ("door", 2, "門"), ("auto4", 3, "靠前牆那條"), ("auto6", 4, "左上那塊"), ("auto3", 5, "窗和門中間那塊"),
           ("auto5", 6, "右上那塊"), ("auto1", 7, "靠後牆那條"), ("auto2", 8, "靠地板那條")],
    "x0": [("whole_wall", 1, "整面")],
    "floor": [("rug", 1, "地毯"), ("auto2", 2, "靠左牆那條"), ("auto3", 3, "靠右牆那條"), ("auto1", 4, "主位底下那塊")],
    "ceiling": [("z1", 1, "主位上面那塊"), ("z2", 2, "兩支喇叭上面那條"), ("auto1", 3, "靠後牆那塊"), ("auto2", 4, "左邊中間那塊"),
                ("auto3", 5, "右邊中間那塊"), ("auto4", 6, "主位和兩支喇叭中間那塊")],
}
CIRCLED = "①②③④⑤⑥⑦⑧"
# 表上每一列（照畫出來的順序）：區代號、「區」那一格的號碼與名字、小字代號。
_TABLE_ROWS = """wall => [...document.querySelectorAll(`#zone-table-${wall} tr[id^=zone-row-]`)].map(row => {
    const key = row.id.replace(`zone-row-${wall}-`, '');
    return {key, label: document.getElementById(`zone-label-${wall}-${key}`).textContent,
            code: row.querySelector('.zone-code')?.textContent ?? null}; })"""
# 立面圖上每一個號碼圓：區代號 → {號碼, 圓心離畫面左緣幾 px}；另外畫面上的字（不准再出現「自動」「z1」）。
_BADGES = """() => Object.fromEntries([...document.querySelectorAll('#elevation .zone-number')].map(g => [g.dataset.zoneKey,
    {number: Number(g.dataset.number), text: g.textContent, x: g.querySelector('circle').getBoundingClientRect().x}]))"""


def test_every_zone_of_the_sample_room_has_one_number_and_a_place_name_on_both_the_elevation_and_the_table(
        tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    write_zone_naming_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        for wall, expected in SAMPLE_NAMES.items():
            page.locator(f"#elevation-wall-{wall}").click()
            rows = cast(list[dict[str, object]], page.evaluate(_TABLE_ROWS, wall))
            # 表照號碼排，「區」那一格寫「① 位置名」，底下小字代號（整面一區伺服器也叫整面，不另寫）。
            assert [(row["key"], row["label"]) for row in rows] == [
                (key, f"{CIRCLED[number - 1]} {name}") for key, number, name in expected], wall
            assert [row["code"] for row in rows] == [None if key == "whole_wall" else f"代號 {key}" for key, _, _ in expected]
            # 立面圖上同一個號碼；不再畫「z1」「自動」。
            badges = cast(dict[str, dict[str, object]], page.evaluate(_BADGES))
            assert {key: badge["number"] for key, badge in badges.items()} == {key: number for key, number, _ in expected}, wall
            assert all(badge["text"] == str(badge["number"]) for badge in badges.values())
            words = page.locator("#elevation-svg text").all_text_contents()
            assert not [word for word in words if "自動" in word or word.startswith(("z1", "auto"))], words
            # 滑過去看得到號碼、位置名與代號（伺服器的限制訊息寫代號，要對得起來）。
            key, number, name = expected[0]
            title = page.locator(f"#elevation [data-zone-id={key}] title" if key != "whole_wall" else "#elevation-frame title")
            expect(title).to_contain_text(f"{CIRCLED[number - 1]} {name}")
            if key != "whole_wall":
                expect(title).to_contain_text(f"（代號 {key}）")
            page.locator("#elevation-panel").screenshot(path=str(tmp_path / f"names-{wall}.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_left_and_right_follow_what_the_elevation_shows_on_the_mirrored_back_wall(tmp_path: Path, browser: Browser) -> None:
    """後牆是鏡像（站在房間裡看後牆，左手邊是右牆）：「靠右牆那條」的號碼畫在左緣印著「右牆」的那一邊，不准跟畫面相反。"""
    _write(tmp_path, ZONE_NAMING_SCHEME)
    write_zone_naming_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        for wall, toward_left, toward_right in (("y0", "auto3", "auto2"), ("xL", "auto4", "auto1"), ("floor", "auto2", "auto3")):
            page.locator(f"#elevation-wall-{wall}").click()
            badges = cast(dict[str, dict[str, float]], page.evaluate(_BADGES))
            left_word = page.locator("#elevation-edge-left").text_content()
            right_word = page.locator("#elevation-edge-right").text_content()
            assert badges[toward_left]["x"] < badges[toward_right]["x"], wall
            expect(page.locator(f"#zone-label-{wall}-{toward_left}")).to_contain_text(f"靠{left_word}")
            expect(page.locator(f"#zone-label-{wall}-{toward_right}")).to_contain_text(f"靠{right_word}")
        assert watched.page_errors == [] and watched.console_errors == []


def _names(page: Page, face: dict[str, object]) -> dict[str, str]:
    return cast(dict[str, str], page.evaluate(
        "face => Object.fromEntries([...nameZones(face)].map(([key, label]) => [key, `${label.numberText} ${label.name}`]))", face))


FRONT_EDGES = {"left": "左牆", "right": "右牆", "top": "天花", "bottom": "地板"}


def _face(zones: list[tuple[str, bool, list[float]]], **extra: object) -> dict[str, object]:
    return {"surface": "wall", "role": "front", "width": 4.0, "height": 3.0, "tolerance": 1e-6, "edges": FRONT_EDGES,
            "front": None, "left": None, "speakers": [], "primary": None,
            "zones": [{"key": key, "drawn": drawn, "whole": False, "box": box, "entry": None} for key, drawn, box in zones]} | extra


def test_hand_made_faces_name_speaker_blocks_strips_and_the_strip_between_two_speakers(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        speakers = [{"name": "左聲道喇叭", "at": [0.8, 1.0]}, {"name": "右聲道喇叭", "at": [3.2, 1.0]}]
        # 喇叭在左邊那塊；中間那條直條夾在兩支喇叭那兩塊中間；整面寬的長條貼著地板。號碼：畫的先（左到右），再自動補的（上到下）。
        face = _face([("auto1", False, [0.0, 4.0, 0.0, 0.3]), ("z2", True, [2.6, 4.0, 0.3, 3.0]),
                      ("z1", True, [0.0, 1.4, 0.3, 3.0]), ("auto2", False, [1.4, 2.6, 0.3, 3.0])], speakers=speakers)
        assert _names(page, face) == {"z1": "① 左聲道喇叭後面那塊", "z2": "② 右聲道喇叭後面那塊", "auto2": "③ 兩支喇叭中間那塊",
                                      "auto1": "④ 靠地板那條"}
        # 夾在中間、窄成一條（長邊／短邊 3 以上）才叫「條」；主位在中間那條上就叫主位正前方。
        narrow = _face([("z1", True, [0.0, 1.8, 0.0, 3.0]), ("z2", True, [2.2, 4.0, 0.0, 3.0]), ("auto1", False, [1.8, 2.2, 0.0, 3.0])],
                       speakers=speakers)
        assert _names(page, narrow)["auto1"] == "③ 兩支喇叭中間那條"
        assert _names(page, narrow | {"primary": [2.0, 1.0]})["auto1"] == "③ 主位正前方那條"
        # 整面寬：貼上緣靠天花、貼下緣靠地板，不貼邊、上下沒有有名字的叫中間那條。
        strips = _face([("auto3", False, [0.0, 4.0, 0.0, 1.4]), ("auto1", False, [0.0, 4.0, 1.4, 1.6]),
                        ("auto4", False, [0.0, 4.0, 1.6, 3.0])])
        assert _names(page, strips) == {"auto4": "① 靠天花那塊", "auto1": "② 中間那條", "auto3": "③ 靠地板那塊"}
        # 其他照中心落在三等分；牆名定不出前後（座標名）時不用喇叭與主位（側牆也不用：那裡不是喇叭「後面」）。
        grid = _face([("z1", True, [0.0, 1.0, 2.0, 3.0]), ("z2", True, [1.5, 2.5, 1.0, 2.0]), ("z3", True, [3.0, 4.0, 0.0, 1.0])],
                     speakers=speakers, primary=[2.0, 1.5])
        assert _names(page, grid | {"role": None}) == {"z1": "① 左上那塊", "z2": "② 正中間那塊", "z3": "③ 右下那塊"}
        assert _names(page, grid | {"role": "side"})["z2"] == "② 正中間那塊"
        assert _names(page, grid | {"role": "back"})["z2"] == "② 主位正後方那塊"
        # 地板：前＝前牆那一側、左＝左牆那一側（地板的立面圖前牆在畫面左緣、左牆在下緣）。
        floor = _face([("z1", True, [0.0, 1.0, 0.0, 1.0]), ("z2", True, [3.0, 4.0, 2.0, 3.0])], surface="floor", role=None,
                      edges={"left": "前牆", "right": "後牆", "top": "右牆", "bottom": "左牆"}, front="left", left="bottom")
        assert _names(page, floor) == {"z2": "① 後右那塊", "z1": "② 前左那塊"}
        assert _names(page, floor | {"speakers": speakers[:1], "zones": [
            {"key": "z1", "drawn": True, "whole": False, "box": [0.0, 1.0, 0.5, 1.5], "entry": None}]}) == {"z1": "① 左聲道喇叭底下那塊"}
        # 門、窗、地毯照材料庫分組與顯示名認；號碼超過 20 寫「(21)」。
        door = {"id": "sample_door_solid_26kg", "label": "實心門", "placement": {"group": "door_window"}}
        window = {"id": "sample_window_glass_5mm", "label": "玻璃窗 5 mm", "placement": {"group": "door_window"}}
        carpet = {"id": "carpet_on_concrete", "label": "地毯直接鋪混凝土", "placement": {"group": "floor"}}
        wood = {"id": "sample_floor_wood_on_joists", "label": "木地板", "placement": {"group": "surface_itself"}}
        assert [page.evaluate("entry => zoneSpecialName(entry)", entry) for entry in (door, window, carpet, wood)] == ["門", "窗", "地毯", None]
        assert page.evaluate("[20, 21].map(zoneNumberText)") == ["⑳", "(21)"]
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_drawn_box_over_the_whole_face_is_whole_and_exactly_three_to_one_is_a_strip(tmp_path: Path, browser: Browser) -> None:
    """畫的框四邊都貼到牆邊，跟整面一區同叫「整面」（不叫「靠天花那塊」）；長邊剛好是短邊 3 倍（60 × 20 公分，
    浮點算出來差一點點）兩區都叫「條」，不是一個條一個塊。"""
    _write(tmp_path, ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        assert _names(page, _face([("full", True, [0.0, 4.0, 0.0, 3.0])])) == {"full": "① 整面"}
        exact = _names(page, _face([("a", True, [0.0, 0.6, 0.0, 0.2]), ("b", True, [1.0, 1.6, 1.0, 1.2])]))
        assert sorted(exact) == ["a", "b"] and all(name.endswith("那條") for name in exact.values()), exact
        assert watched.page_errors == [] and watched.console_errors == []
