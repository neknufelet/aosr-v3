"""輸入頁在老闆的螢幕寬（1440）上看起來對不對：量瀏覽器排出來的位置、字級與顏色。

#516 視覺一輪第 8、9、10、13、14 項。量的是排版結果（誰跟誰同一行、字幾像素、按鈕長得一不一樣），
不釘個數；難不難看仍要靠截圖親眼看。瀏覽器缺席就紅、不跳過（跟 test_gui_browser 同一個瀏覽器）。
"""
from __future__ import annotations

import json
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
        page.wait_for_function("() => document.getElementById('messages').textContent.includes('空著沒填')")
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


def _blank_and_press(page: Page, button: str, fields: tuple[str, ...]) -> None:
    for field in fields:
        page.locator(field).fill("")
    page.locator("#messages").evaluate("node => { node.textContent = ''; }")
    page.locator(button).click()
    page.wait_for_function("() => document.getElementById('messages').textContent !== ''")


def test_save_as_answers_beside_its_row_in_its_own_words(tmp_path: Path, browser: Browser) -> None:
    # 「另存新名字」在頁面最上面；以前空名字的提示、檢查沒過的問題都只寫在最底下的訊息列，按了像沒反應，
    # 空名字的提示還寫「新代號」，跟這一列的「新名字」對不上。
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        note, button = page.locator("#save-as-note"), _box(page, "#save-as")
        page.locator("#save-as").click()
        page.wait_for_function("() => document.getElementById('save-as-note').textContent !== ''")
        # 空名字：提示貼在按鈕右邊、同一行，用這一列的字（新名字、另存新名字），游標回到新名字那一格。
        hint = _box(page, "#save-as-note")
        assert abs(_middle(hint) - _middle(button)) < SAME_LINE_PX
        assert 0 <= hint["x"] - (button["x"] + button["width"]) < 24
        assert "「新名字」" in note.inner_text() and "「另存新名字」" in note.inner_text() and "代號" not in note.inner_text()
        assert note.get_attribute("class") == "notice"
        assert page.evaluate("() => document.activeElement.id") == "save-as-id"
        # 開始打字，提示就收掉。
        page.locator("#save-as-id").fill("next")
        assert note.inner_text() == ""
        # 檢查沒過：問題（伺服器寫好的白話，同一句只一行、列出哪幾格）寫在按鈕旁邊，也寫在底下的訊息列。
        before = {field: page.locator(field).input_value() for field in ("#room-Ly", "#wall-floor")}
        _blank_and_press(page, "#save-as", tuple(before))
        below = page.locator("#messages").inner_text()
        assert below.split("\n") == ["寬 Ly（公尺）、地板阻抗：空著沒填"], below
        assert note.inner_text() == f"沒有另存：{below}"
        # 按鈕旁邊那句不用捲動就看得到：整句落在視窗裡（bounding_box 以目前視窗的左上角為原點）。
        hint, viewport = _box(page, "#save-as-note"), page.viewport_size
        assert note.is_visible() and viewport is not None
        assert hint["y"] >= 0 and hint["y"] + hint["height"] <= viewport["height"]
        assert not (tmp_path / "schemes" / "next.json").exists()
        # 之後底下換了新的一句（例如按檢查），旁邊那句就過時了，一起清掉。
        page.locator("#check").click()
        page.wait_for_function("() => document.getElementById('save-as-note').textContent === ''")
        for field, value in before.items():
            page.locator(field).fill(value)
        page.locator("#save-as").click()
        page.wait_for_function("() => document.getElementById('save-id').value === 'next'")
        assert note.inner_text() == "「next」：方案已儲存" and note.get_attribute("class") == "ok"
        assert (tmp_path / "schemes" / "next.json").exists()
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_save_problems_show_right_under_the_buttons(tmp_path: Path, browser: Browser) -> None:
    # 按「儲存」與「檢查」：伺服器寫好的白話問題（表單上的中文名、同一句只一行）印在按鈕正下方，不存檔。
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        name = page.locator("#save-id").input_value()
        _blank_and_press(page, "#save", ("#room-Ly", "#speaker-right-y"))
        shown = page.locator("#messages").inner_text()
        assert shown.split("\n") == ["寬 Ly（公尺）、右聲道喇叭 y 座標：空著沒填"], shown
        assert page.locator("#messages").get_attribute("class") == "notice"
        assert not (tmp_path / "schemes" / f"{name}.json").exists()
        page.locator("#messages").evaluate("node => { node.textContent = ''; }")
        page.locator("#check").click()
        page.wait_for_function("() => document.getElementById('messages').textContent !== ''")
        assert page.locator("#messages").inner_text() == shown
        # 訊息列緊貼在那一排按鈕底下：按的人眼睛就在那裡。
        buttons, messages = _box(page, ".actions"), _box(page, "#messages")
        assert 0 <= messages["y"] - (buttons["y"] + buttons["height"]) < 40
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_browser_errors_are_said_in_plain_chinese(tmp_path: Path, browser: Browser) -> None:
    # 連不上伺服器、或伺服器回的不是 JSON 時，瀏覽器的錯是英文（Failed to fetch、Unexpected token …）；
    # 訊息列要寫白話，不印英文原文。
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        for answer, words in ((lambda route: route.abort(), "連不上本機的網頁伺服器"),
                              (lambda route: route.fulfill(status=500, content_type="text/html",
                                                           body="<html>Internal Server Error</html>"),
                               "伺服器的回覆讀不懂")):
            page.route("**/api/plan", answer)
            page.locator("#messages").evaluate("node => { node.textContent = ''; }")
            page.locator("#check").click()
            page.wait_for_function("() => document.getElementById('messages').textContent !== ''")
            shown = page.locator("#messages").inner_text()
            assert words in shown and not re.search(r"[A-Za-z]{3,}", shown), shown
            page.unroute("**/api/plan")
        assert all("500" in text or "ERR_FAILED" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


# 伺服器收不下的名字（方案代號的規則：英文字母、數字、_、-，第一個字是英文字母或數字）。
REFUSED_NAMES = ("客廳一號", "my room", "a/b", "..", "-x")


def test_save_as_refuses_unusable_names_beside_its_row(tmp_path: Path, browser: Browser) -> None:
    # 老闆第一個會打的是中文名字：以前送到伺服器才被擋，旁邊寫的卻是上面那一列「方案代號」的規則；
    # 有 / 或 .. 的名字更只拿到英文的「Not Found」。現在打字時就在旁邊說，按下去也不送出。
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url) if request.method == "PUT" else None)
        note, button = page.locator("#save-as-note"), _box(page, "#save-as")
        for name in REFUSED_NAMES:
            page.locator("#save-as-id").fill(name)
            # 一邊打就說：用這一列的字（新名字），講得出收哪些字、中文不收，不提別列的「方案代號」。
            live = note.inner_text()
            assert "「新名字」" in live and "中文" in live and "代號" not in live, live
            assert not re.search(r"[A-Za-z]{3,}", live), live
            assert note.get_attribute("class") == "notice"
            page.locator("#save-as").click()
            page.wait_for_function("() => document.getElementById('save-as-note').textContent.startsWith('沒有另存')")
            assert note.inner_text() == f"沒有另存：{live}"
            assert page.evaluate("() => document.activeElement.id") == "save-as-id"
            hint = _box(page, "#save-as-note")
            assert hint["y"] < _middle(button) < hint["y"] + hint["height"]
            assert 0 <= hint["x"] - (button["x"] + button["width"]) < 24
        # 收得下的名字打出來，旁邊的提示就收掉；底下的訊息列沒被這些提示蓋掉。
        page.locator("#save-as-id").fill("living-room_2")
        assert note.inner_text() == ""
        assert page.locator("#messages").inner_text() == "檢查通過"
        assert not sent, sent
        assert not list((tmp_path / "schemes").glob("*.json"))
        _assert_quiet(watched)


def test_open_failure_is_said_right_at_the_open_button(tmp_path: Path, browser: Browser) -> None:
    # 存著的方案現在檢查不過時按「打開」：以前原因只寫在頁面最底下的訊息列，最上面看起來像沒反應。
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        scheme = page.request.get(f"{base}/api/example").json()["scheme"]
        opened = page.locator("#save-id").input_value()
        speaker = next(iter(scheme["speakers"]))
        scheme["speakers"][speaker]["x"] = scheme["scene"]["room_m"]["Lx"] + 3
        (tmp_path / "schemes").mkdir(parents=True, exist_ok=True)
        (tmp_path / "schemes" / "outside.json").write_text(json.dumps({**scheme, "scheme_id": "outside"}))
        page.reload(wait_until="networkidle")
        page.locator("#plan-legend li").first.wait_for()
        room = page.locator("#room-Lx").input_value()
        page.locator("#scheme-list").select_option("outside")
        page.locator("#open-scheme").click()
        page.wait_for_function("() => document.getElementById('open-note').textContent !== ''")
        below = page.locator("#messages").inner_text()
        note = page.locator("#open-note")
        # 旁邊那句跟訊息列同一段白話，前面說清楚是哪一份沒打開；表單維持原來那一份。
        assert below and note.inner_text() == f"沒有打開「outside」：{below}"
        assert note.get_attribute("class") == "notice"
        assert page.locator("#save-id").input_value() == opened
        assert page.locator("#room-Lx").input_value() == room
        # 那句緊跟著「打開」：在按鈕旁邊或正下方，整句落在視窗裡（bounding_box 以目前視窗的左上角為原點）。
        button, hint, viewport = _box(page, "#open-scheme"), _box(page, "#open-note"), page.viewport_size
        assert viewport is not None
        assert button["y"] <= hint["y"] + hint["height"] and hint["y"] < button["y"] + button["height"] + 24
        assert hint["y"] >= 0 and hint["y"] + hint["height"] <= viewport["height"]
        # 之後底下換了新的一句（按檢查），旁邊那句就過時了，一起清掉。
        page.locator("#check").click()
        page.wait_for_function("() => document.getElementById('open-note').textContent === ''")
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_only_the_calculation_cell_carries_the_technical_tooltip(tmp_path: Path, browser: Browser) -> None:
    # 讀不出的結果檔那一列，四格都寫「讀不出」：滑鼠說明只能掛在「計算版本」那一格，不跟著字一樣的格子跑。
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "results" / f"{'d' * 32}.json").write_text(json.dumps(
        {"scheme": {"scheme_id": "unreadable"}, "schema_version": "aosr.scheme_result.v3"}))
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        row = page.locator("#results-list tr", has_text="unreadable")
        row.wait_for()
        headers = page.locator("table:has(#results-list) th").all_inner_texts()
        cells = row.locator("td").evaluate_all("cells => cells.map(cell => [cell.textContent, cell.title])")
        assert {text for text, _ in cells[1:5]} == {"讀不出"}
        assert {headers[index] for index, (_, title) in enumerate(cells) if title} == {"計算版本"}
        _assert_quiet(watched)


def _multiples(page: Page) -> dict[str, str]:
    return cast(dict[str, str], page.locator("#walls").evaluate(
        "node => Object.fromEntries([...node.querySelectorAll('span[id^=multiple-]')].map(s => [s.id, s.textContent]))"))


def test_impedance_multiple_never_sits_beside_a_changed_or_blank_box(tmp_path: Path, browser: Browser) -> None:
    # 清空一格阻抗時旁邊還寫「約 ρc 的 4.00 倍」：那是清空前的數字。表單有格子空著時伺服器一格倍數都不給，
    # 沒改過的牆仍是那個數字的倍數、照留；改過或清空的那一格清掉；表單補齊了就全部回來。
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        page.wait_for_function("() => document.getElementById('multiple-floor').textContent !== ''")
        before = _multiples(page)
        assert all(text.startswith("約 ρc 的") for text in before.values()), before
        floor = page.locator("#wall-floor").input_value()
        page.locator("#wall-floor").fill("")
        page.wait_for_function("() => document.getElementById('multiple-floor').textContent === ''")
        assert {key: text for key, text in _multiples(page).items() if key != "multiple-floor"} == \
            {key: text for key, text in before.items() if key != "multiple-floor"}
        page.locator("#wall-ceiling").fill("1000")
        page.wait_for_function("() => document.getElementById('multiple-ceiling').textContent === ''")
        assert _multiples(page)["multiple-x0"] == before["multiple-x0"]
        page.locator("#wall-floor").fill(floor)
        page.wait_for_function("() => document.getElementById('multiple-ceiling').textContent !== ''")
        after = _multiples(page)
        assert after["multiple-floor"] == before["multiple-floor"]
        assert after["multiple-ceiling"].startswith("約 ρc 的") and after["multiple-ceiling"] != before["multiple-ceiling"]
        _assert_quiet(watched)


def test_calculation_output_stays_in_technical_details() -> None:
    """算的過程中計算程式自己印的英文輸出（建網格的訊息等）不上主畫面，收在可展開的技術細節裡。"""
    static = Path(__file__).resolve().parents[2] / "src" / "aosr" / "gui" / "static"
    script = (static / "app.js").read_text(encoding="utf-8")
    page = (static / "index.html").read_text(encoding="utf-8")
    assert '$("run-state").textContent = state.display_text;' in script
    assert not re.search(r'\$\("run-state"\)[^;]*stderr_tail', script)
    assert '$("run-log-text").textContent = state.stderr_tail.join' in script
    assert re.search(r'<details id="run-log" hidden><summary>[^<]*技術細節[^<]*</summary><pre id="run-log-text">', page)
