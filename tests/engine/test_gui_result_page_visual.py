"""結果頁在老闆的螢幕寬（1440）上讀起來對不對：顯示名、圖例、摺疊區、判定欄與舊格式頁。

#516 視覺一輪結果頁那幾項（1、2、3、5、7、8、11）。量的是畫面上看得到的字、摺疊區開沒開、按鈕與選單，
不釘個數（個數都從伺服器資料算）；難不難看仍要靠截圖親眼看。瀏覽器缺席就紅、不跳過（跟 test_gui_browser
同一個瀏覽器）。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Locator, Route

from aosr.gui.result_view import FlutterGroupView, ResultView, _flutter_groups, build_result_view
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import QualityCategory
from aosr.scoring.review_alert import FlutterReviewAlert
from tests.engine.test_gui_browser import (RUN_ID, _assert_quiet, _assert_text_is_formatted,
                                           _legend_labels, _open, _save, _serve, _wait_for_lines,
                                           browser)
from tests.engine.test_scheme_pipeline import shared_control_result

WIDTH = 1440


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def _open_flags(locator: Locator) -> list[bool]:
    return [bool(flag) for flag in locator.evaluate_all("nodes => nodes.map((node) => node.open)")]


def _flutter(walls: tuple[str, str], nominal: int, duration: float) -> FlutterReviewAlert:
    return FlutterReviewAlert(category=QualityCategory.REFLECTIONS_AND_ECHO, walls=walls,
                              nominal_center_hz=nominal, center_frequency_hz=nominal * 0.992,
                              lower_hz=nominal * 0.89, upper_hz=nominal * 1.12,
                              decay_duration_s=duration, room_t20_s=0.0687, decay_db=60.0, note="警戒")


# 考卷資料沒有牆間顫動：用伺服器自己的合併函式做兩對牆、每對幾帶，塞進真的結果頁資料裡。
FLUTTER = _flutter_groups(tuple(_flutter(walls, nominal, duration)
                                for walls, duration in ((("floor", "ceiling"), 0.1072), (("x0", "xL"), 0.2144))
                                for nominal in (400, 500, 630, 800)))


def _with_flutter(real: Callable[..., ResultView], groups: tuple[FlutterGroupView, ...]
                  ) -> Callable[..., ResultView]:
    def build(result: SchemeResult, *, quality_targets_path: Path) -> ResultView:
        return real(result, quality_targets_path=quality_targets_path).model_copy(
            update={"flutter_groups": groups})
    return build


def test_legend_names_each_line_without_cursor_values(tmp_path: Path, browser: Browser,
                                                      result: SchemeResult) -> None:
    _save(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        _wait_for_lines(page)
        data = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        legend = page.locator("#chart .u-legend").inner_text()
        # uPlot 跟游標的圖例會多一格英文「Value」，每條線後面掛「: --」。
        assert "Value" not in legend and "--" not in legend
        responses = [item for item in data["frequency_responses"]
                     if item["role"] == data["frequency_responses"][0]["role"]
                     and item["receiver_role"] in {"primary", "surrounding"}]
        speakers, points = data["speaker_names"], data["point_names"]
        assert isinstance(speakers, dict) and isinstance(points, dict)
        assert set(_legend_labels(page)) == {
            f"{speakers[item['role']]} → {points[item['receiver_id']]}" for item in responses}
        # 每條線在圖例上有自己的顏色記號（記號框的邊色），不是只剩字。
        colours = page.locator("#chart .u-legend .u-marker").evaluate_all(
            "nodes => nodes.map((node) => getComputedStyle(node).borderColor)")
        assert colours and len(set(colours)) == len(colours)
        # 喇叭按鈕也是顯示名，目前選的那一顆標著。
        assert page.locator("#speaker-buttons button").all_inner_texts() == list(dict.fromkeys(
            speakers[item["role"]] for item in data["frequency_responses"]))
        assert page.locator("#speaker-buttons button").first.get_attribute("aria-pressed") == "true"
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_flutter_and_outside_window_paths_start_folded(tmp_path: Path, browser: Browser,
                                                       result: SchemeResult,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _save(tmp_path, result)
    monkeypatch.setattr("aosr.gui.app.build_result_view",
                        _with_flutter(build_result_view, FLUTTER))
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#content").wait_for(state="visible")
        blocks = page.locator("#alerts article.flutter")
        # 一對牆一筆：標題照伺服器合好的，逐帶明細一開始收著。
        assert blocks.locator("h3").all_inner_texts() == [group.heading_text for group in FLUTTER]
        assert _open_flags(blocks.locator("details")) == [False for _ in FLUTTER]
        band = FLUTTER[0].bands[0]
        assert band.center_text not in page.locator("#alerts").inner_text()
        blocks.first.locator("summary").click()
        assert band.center_text in blocks.first.inner_text()
        assert blocks.first.locator("tr").all_inner_texts()[1:] == [
            "\t".join((row.nominal_text, row.center_text, row.duration_text, row.room_t20_text,
                       row.excess_text)) for row in FLUTTER[0].bands]
        # 反射：窗內的直接列、窗外的收著，摘要行照伺服器那句說有幾條。
        reflections = page.request.get(f"{base}/api/results/{RUN_ID}").json()["reflections"]
        assert isinstance(reflections, list)
        channels = page.locator("#reflections article")
        assert channels.locator("h3").all_inner_texts() == [item["heading_text"] for item in reflections]
        for index, channel in enumerate(reflections):
            block = channels.nth(index)
            inside = [path for path in channel["paths"] if path["within_window"]]
            outside = [path for path in channel["paths"] if not path["within_window"]]
            # 注意事項自己一行，逐字印伺服器給的白話句子（不在網頁端把代號翻成短代稱）。
            notes = [text for text in block.locator(":scope > p").all_inner_texts()
                     if text.startswith("注意事項：")]
            assert channel["flags"] and notes == [f"注意事項：{channel['flags_text']}"]
            assert len(block.locator(":scope > table tr").all()) == len(inside) + 1
            assert block.locator("summary").inner_text() == channel["outside_summary_text"]
            assert _open_flags(block.locator("details")) == [False]
            assert outside[0]["delay_text"] not in block.inner_text()
            block.locator("summary").click()
            assert len(block.locator("details tr").all()) == len(outside) + 1
            assert outside[0]["delay_text"] in block.inner_text()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_reverberation_says_how_each_band_compares(tmp_path: Path, browser: Browser,
                                                   result: SchemeResult) -> None:
    _save(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#content").wait_for(state="visible")
        reverberation = page.request.get(f"{base}/api/results/{RUN_ID}").json()["reverberation"]
        assert isinstance(reverberation, dict)
        headings = page.locator("#reverb th").all_inner_texts()
        assert "跟目標比" in headings and "量得到嗎" in headings and "狀態與原因" not in headings
        column = headings.index("跟目標比")
        cells = page.locator(f"#reverb tr td:nth-child({column + 1})")
        assert cells.all_inner_texts() == [band["verdict_text"] for band in reverberation["bands"]]
        assert cells.evaluate_all("nodes => nodes.map((node) => node.dataset.verdict)") == [
            band["verdict"] for band in reverberation["bands"]]
        assert page.locator("#reverb-note").inner_text() == reverberation["caption_text"]
        assert page.locator("#reverb-compare-note").inner_text() == reverberation["compare_note"]
        assert "已量、已量" not in page.locator("#reverb").inner_text()
        _assert_quiet(watched)


def _with_commit(commit: str) -> Callable[..., ResultView]:
    """讀回核對會比對存檔裡的提交，改不得；只在建頁面資料那一步換成四十碼的提交。"""
    def build(result: SchemeResult, *, quality_targets_path: Path) -> ResultView:
        return build_result_view(result.model_copy(update={"engine_commit": commit}),
                                 quality_targets_path=quality_targets_path)
    return build


def test_categories_hide_technical_details_and_show_short_commit(tmp_path: Path, browser: Browser,
                                                                 result: SchemeResult,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "e53bfae" + "f" * 33
    _save(tmp_path, result)
    monkeypatch.setattr("aosr.gui.app.build_result_view", _with_commit(commit))
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#content").wait_for(state="visible")
        data = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        header = page.locator("header").inner_text()
        assert "程式版本：e53bfae" in header and commit not in header
        assert page.locator("#categories th").all_inner_texts() == ["類別", "狀態", "代價", "注意事項", "說明"]
        assert page.locator("#cost-note").inner_text() == data["cost_note"]
        categories = data["categories"]
        assert isinstance(categories, list)
        versions = [item["evaluator_version"] for item in categories if item["evaluator_version"]]
        assert versions
        # 評估器版本、原因碼收在「技術細節」：一開始收著，主畫面看不到模組名。
        assert _open_flags(page.locator("#category-details")) == [False]
        assert not any(version in page.locator("body").inner_text() for version in versions)
        assert all(item["flags_text"] in page.locator("#categories").inner_text() for item in categories)
        page.locator("#category-details summary").click()
        assert all(version in page.locator("#category-technical").inner_text() for version in versions)
        _assert_quiet(watched)


def test_old_format_result_says_so_and_offers_rerun(tmp_path: Path, browser: Browser,
                                                    result: SchemeResult) -> None:
    _save(tmp_path, result)
    path = tmp_path / "results" / f"{RUN_ID}.json"
    document = json.loads(path.read_text())
    document["schema_version"] = "aosr.scheme_result.v2"
    path.write_text(json.dumps(document))
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#rejection").wait_for(state="visible")
        rejected = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        assert rejected["reason_kind"] == "old_format"
        assert page.locator("#rejection-title").inner_text() == "舊格式的結果"
        assert page.locator("#reject-reason").inner_text() == rejected["reason_text"]
        # 技術原因（schema_version 那句）收進摺疊區，老闆先看到的是白話。
        assert _open_flags(page.locator("#reject-detail")) == [False]
        assert "schema_version" not in page.locator("#rejection").inner_text()
        page.locator("#reject-detail summary").click()
        assert page.locator("#reject-technical").inner_text() == rejected["reason"]
        requested: list[str] = []

        def capture(route: Route) -> None:
            requested.append(route.request.url)
            route.fulfill(status=200, content_type="application/json", body='{"run_id":"started"}')

        page.route("**/api/results/*/rerun", capture)
        page.get_by_role("button", name="用現在的程式重算", exact=True).click()
        page.locator("#rerun-state").filter(has_text="已開始重算").wait_for()
        assert requested == [f"{base}{rejected['rerun_url']}"]
        assert page.locator("#content").is_hidden()
        # 409 回應本身瀏覽器會記一筆「載入失敗」，那是預期的；JS 自己不准拋錯。
        assert all("409" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


# 考卷資料只有主位與一個周圍點，沒有周圍點彼此的配對：把主位那一對換成「周圍點彼此」塞進去，考下拉選單接得上。
ADD_PEER_PAIR_JS = """() => {
  const choice = view.listening_area.pair_choices.find((item) => item.role === selectedRole);
  const swap = (item) => ({...item, group: "surrounding_to_surrounding",
    reference_id: item.receiver_id, receiver_id: item.reference_id});
  view.listening_area.pairs.push(...view.listening_area.pairs.filter((item) => item.role === selectedRole &&
    item.receiver_id === choice.receiver_id && item.reference_id === choice.reference_id).map(swap));
  view.listening_area.pair_choices.push({...swap(choice), text: "考卷用的一對"});
  drawListening();
}"""


def test_peer_pairs_live_in_one_select_beside_the_buttons(tmp_path: Path, browser: Browser,
                                                          result: SchemeResult) -> None:
    _save(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        _wait_for_lines(page)
        full = set(_legend_labels(page))
        data = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        role = data["frequency_responses"][0]["role"]
        choices = [item for item in data["listening_area"]["pair_choices"]
                   if item["role"] == role and item["group"] == "primary_to_surrounding"]
        # 按鈕只有「全部位置」與主位對周圍點；沒有周圍點彼此的配對就不畫選單。
        assert page.locator("#pair-buttons button").all_inner_texts() == [
            "全部位置", *(item["text"] for item in choices)]
        assert not page.locator("#peer-pairs").all()
        page.evaluate(ADD_PEER_PAIR_JS)
        select = page.locator("#peer-pairs")
        assert "考卷用的一對" in select.locator("option").all_inner_texts()
        assert "考卷用的一對" not in page.locator("#pair-buttons button").all_inner_texts()
        select.select_option(label="考卷用的一對")
        page.locator("#pair-detail td").first.wait_for()
        _wait_for_lines(page)
        points = data["point_names"]
        assert isinstance(points, dict)
        shown = {label.split(" → ", 1)[1] for label in _legend_labels(page)}
        assert shown == {points[choices[0]["reference_id"]], points[choices[0]["receiver_id"]]}
        assert page.locator("#pair-detail h3").inner_text() == "考卷用的一對"
        # 選單選的那一對不是按鈕：按鈕全部不亮；再按一顆按鈕，選單回到提示那一格。
        assert set(page.locator("#pair-buttons button").evaluate_all(
            "nodes => nodes.map((node) => node.getAttribute('aria-pressed'))")) == {"false"}
        assert "chosen" in (select.get_attribute("class") or "")
        # 選回提示那一格＝不選任何一對：回到全部曲線，「全部位置」亮、選單不亮、明細清空。
        select.select_option(value="")
        _wait_for_lines(page)
        assert set(_legend_labels(page)) == full
        assert page.get_by_role("button", name="全部位置", exact=True).get_attribute("aria-pressed") == "true"
        assert "chosen" not in (select.get_attribute("class") or "")
        assert not page.locator("#pair-detail h3").all()
        # 再選一次那一對，再按一顆按鈕：選單回到提示那一格。
        select.select_option(label="考卷用的一對")
        page.locator("#pair-detail td").first.wait_for()
        page.get_by_role("button", name=choices[0]["text"], exact=True).click()
        assert select.input_value() == ""
        assert "chosen" not in (select.get_attribute("class") or "")
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
