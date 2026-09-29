"""輸入頁在老闆的螢幕寬（1440）上看起來對不對：量瀏覽器排出來的位置、字級與顏色。

#516 視覺一輪第 8、9、10、13、14 項。量的是排版結果（誰跟誰同一行、字幾像素、按鈕長得一不一樣），
不釘個數；難不難看仍要靠截圖親眼看。瀏覽器缺席就紅、不跳過（跟 test_gui_browser 同一個瀏覽器）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page

from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser

WIDTH = 1440
# 同一行：兩格的垂直中線差不到幾個像素（按鈕比輸入格高一點，中線仍對齊）。
SAME_LINE_PX = 4
# 「對齊」「一樣寬」容許的小數像素誤差。
EDGE_PX = 1


def _box(page: Page, selector: str) -> dict[str, float]:
    box = page.locator(selector).bounding_box()
    assert box is not None, selector
    return {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}


def _middle(box: dict[str, float]) -> float:
    return box["y"] + box["height"] / 2


def _same(first: float, second: float) -> bool:
    return abs(first - second) < EDGE_PX


def test_scheme_rows_keep_each_button_beside_its_field(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        opened, listing, open_button = (_box(page, name) for name in ("#save-id", "#scheme-list", "#open-scheme"))
        new_name, save_as, source = (_box(page, name) for name in ("#save-as-id", "#save-as", "#source-model"))
        # 第一列：方案代號、開舊方案、打開。
        assert abs(_middle(opened) - _middle(listing)) < SAME_LINE_PX
        assert abs(_middle(listing) - _middle(open_button)) < SAME_LINE_PX
        # 第二列：「另存新名字」按鈕緊跟在新名字那一格右邊，不掉到下一行。
        assert abs(_middle(new_name) - _middle(save_as)) < SAME_LINE_PX
        assert 0 <= save_as["x"] - (new_name["x"] + new_name["width"]) < 24
        # 三列上下排開，每列第一格左邊對齊。
        assert _middle(opened) < _middle(new_name) < _middle(source)
        assert abs(opened["x"] - new_name["x"]) < EDGE_PX and abs(new_name["x"] - source["x"]) < EDGE_PX
        # 峰谷配對容差那句講的是評分怎麼配對峰谷，不是聲源：不准擺在方案區（聲源模型底下）。
        note = page.locator("#feature-note")
        assert "峰谷配對容差" in note.inner_text()
        assert note.evaluate("node => node.closest('section').querySelector('h2').textContent") == "喇叭與座位"
        _assert_quiet(watched)


PANELS = ("#plan-xy", "#plan-xz", "#zoom-xy", "#zoom-xz")
# 每一個圖上的字實際畫出來幾像素：字級（畫框單位）乘上畫框換到螢幕的倍率。
LABEL_PX_JS = """svg => [...svg.querySelectorAll('text')].map(text =>
  Number(text.getAttribute('font-size')) * text.getScreenCTM().a)"""
# 畫出來的字有沒有跑出畫框（viewBox）：躲字往下挪的量寫錯，字會被推到圖外看不到。
LABELS_OUTSIDE_JS = """svg => {
  const [left, top, width, height] = svg.getAttribute('viewBox').split(' ').map(Number);
  return [...svg.querySelectorAll('text')].map(text => [text.textContent, text.getBBox()])
    .filter(([, box]) => box.x < left || box.y < top || box.x + box.width > left + width ||
                         box.y + box.height > top + height)
    .map(([words, box]) => [words, box.x, box.y, box.width, box.height]);
}"""


def test_plan_panels_are_a_two_by_two_grid_with_readable_labels(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#zoom-xz circle").first.wait_for()
        plan, side, zoom_plan, zoom_side = (_box(page, name) for name in PANELS)
        # 兩欄兩列：上排整間房、下排聆聽區放大；同一欄左緣對齊、同一列上緣對齊，四格一樣寬、同列一樣高。
        assert _same(plan["x"], zoom_plan["x"]) and _same(side["x"], zoom_side["x"]) and plan["x"] < side["x"]
        assert _same(plan["y"], side["y"]) and _same(zoom_plan["y"], zoom_side["y"]) and plan["y"] < zoom_plan["y"]
        assert all(_same(box["width"], plan["width"]) for box in (side, zoom_plan, zoom_side))
        assert _same(plan["height"], side["height"]) and _same(zoom_plan["height"], zoom_side["height"])
        # 兩欄合起來佔滿整個區塊的內寬（沒有落單的第四格旁邊空一塊）。
        section = page.locator("#plan-drawings").bounding_box()
        assert section is not None
        assert _same(side["x"] + side["width"], section["x"] + section["width"])
        every_size: list[float] = []
        for name in PANELS:
            sizes = cast(list[float], page.locator(name).evaluate(LABEL_PX_JS))
            assert sizes, name
            every_size += sizes
            # 老闆的螢幕上字約 12 像素以上（原本三欄時約 8 像素），也不會大到比內文還大。
            assert all(12 <= size <= 16 for size in sizes), (name, sizes)
            # 字躲開別的字與點之後，仍然整個在畫框裡。
            assert page.locator(name).evaluate(LABELS_OUTSIDE_JS) == [], name
            # 房間（或聆聽區）的外框置中、塞滿格子：寬或高至少佔四分之三，左右留白一樣。
            frame = page.locator(f"{name} rect").first.bounding_box()
            panel = _box(page, name)
            assert frame is not None
            assert max(frame["width"] / panel["width"], frame["height"] / panel["height"]) >= 0.75, name
            left_gap = frame["x"] - panel["x"]
            right_gap = panel["x"] + panel["width"] - (frame["x"] + frame["width"])
            assert abs(left_gap - right_gap) < 2, (name, left_gap, right_gap)
        # 四張圖的畫框大小不同，字照樣一樣大（字級跟著畫框換算），不會放大圖的字特別大。
        assert max(every_size) - min(every_size) < 0.5, every_size
        _assert_quiet(watched)


def _message_style(page: Page) -> dict[str, str]:
    return cast(dict[str, str], page.locator("#messages").evaluate("""node => {
      const own = getComputedStyle(node), body = getComputedStyle(document.body);
      return {kind: node.className, color: own.color, size: own.fontSize, bodySize: body.fontSize,
              family: own.fontFamily, bodyFamily: body.fontFamily};
    }"""))


def test_check_passed_is_green_and_problems_use_notice(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#messages").evaluate("node => { node.textContent = ''; }")
        page.locator("#check").click()
        page.wait_for_function("() => document.getElementById('messages').textContent === '檢查通過'")
        passed = _message_style(page)
        # 成功用共用的 .ok（綠），字級與字型跟內文一樣（原本是等寬小字的紅褐色，看起來像錯誤）。
        assert passed["kind"] == "ok"
        red, green, blue = (int(part) for part in re.findall(r"\d+", passed["color"])[:3])
        assert green > red and green > blue, passed["color"]
        assert passed["size"] == passed["bodySize"] and passed["family"] == passed["bodyFamily"]
        page.locator("#room-Lx").fill("")
        page.locator("#check").click()
        page.wait_for_function("() => document.getElementById('messages').textContent.includes('必填')")
        problem = _message_style(page)
        assert problem["kind"] == "notice"
        assert problem["color"] != passed["color"]
        assert problem["size"] == problem["bodySize"]
        # 長度清空時伺服器照設計回 422，瀏覽器會記一筆「載入失敗」，那是預期的；JS 自己不准拋錯。
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_rows_and_legend_use_chinese_names_from_label_table(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        tables = page.request.get(f"{base}/api/labels").json()
        scheme = page.request.get(f"{base}/api/example").json()["scheme"]
        roles = {channel["speaker_id"]: channel["role"] for channel in scheme["channel_group"]["channels"]}
        expected = {**{f"speaker-{code}": tables["speakers"][role] for code, role in roles.items()},
                    **{f"receiver-{code}": tables["listening_points"][code]
                       for code in (point["receiver_id"] for point in scheme["receiver_set"]["points"])}}
        for field_prefix, name in expected.items():
            code = field_prefix.split("-", 1)[1]
            title = page.locator(f"#{field_prefix}-x").locator("xpath=ancestor::div[@class='row']/strong")
            # 每列：中文名在前，代號用小字放括號（輸入框的編號仍是代號）。
            assert title.inner_text() == f"{name}（{code}）"
            assert title.locator("small").inner_text() == f"（{code}）"
            sizes = title.evaluate(
                "node => [getComputedStyle(node).fontSize, getComputedStyle(node.querySelector('small')).fontSize]")
            assert float(sizes[1].removesuffix("px")) < float(sizes[0].removesuffix("px"))
        # 喇叭 left 與座位 left 看起來要不一樣。
        assert expected["speaker-left"] != expected["receiver-left"]
        # 平面圖底下的點清單用同一套名字：「標記－中文名（代號）：座標」。
        legend = dict(page.locator("#plan-legend li").evaluate_all(
            "nodes => nodes.map(node => [node.dataset.id, node.textContent])"))
        for field_prefix, name in expected.items():
            kind, code = field_prefix.split("-", 1)
            key = f"{'speaker' if kind == 'speaker' else 'receiver'}:{code}"
            assert legend[key].split("－", 1)[1].startswith(f"{name}（{code}）："), legend[key]
        assert legend["speaker:left"].split("（")[0] != legend["receiver:left"].split("（")[0]
        _assert_quiet(watched)
