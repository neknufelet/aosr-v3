"""結果頁在老闆的螢幕寬（1440）上讀起來對不對：顯示名、圖例、摺疊區、判定欄與舊格式頁。

#516 視覺一輪結果頁那幾項（1、2、3、5、7、8、11），第三輪加上重算按鈕、頁首計算版本、排名那一行、位置對同一列。量的是畫面上看得到的字、摺疊區開沒開、按鈕與選單，
不釘個數（個數都從伺服器資料算）；難不難看仍要靠截圖親眼看。瀏覽器缺席就紅、不跳過（跟 test_gui_browser
同一個瀏覽器）。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Locator, Route

from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from aosr.gui.labels import DIRECTIONS
from aosr.gui.result_view import (FlutterGroupView, PairView, ResultView, _flutter_groups, _pair_choices,
                                  build_result_view)
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import QualityCategory
from aosr.scoring.review_alert import FlutterReviewAlert
from tests.engine.test_gui_browser import (RUN_ID, _assert_quiet, _assert_text_is_formatted,
                                           _check_rerun_starts_once, _legend_labels, _open, _save, _serve,
                                           _wait_for_lines, browser)
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
        data = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        reflections, names = data["reflections"], data["labels"]
        assert isinstance(reflections, list) and isinstance(names, dict)
        channels = page.locator("#reflections article")
        assert channels.locator("h3").all_inner_texts() == [item["heading_text"] for item in reflections]
        for index, channel in enumerate(reflections):
            block = channels.nth(index)
            inside = [path for path in channel["paths"] if path["within_window"]]
            outside = [path for path in channel["paths"] if not path["within_window"]]
            # 第一行：結論、涵蓋、路徑數值驗證、原因，四段的名字照這裡釘的白話（值是伺服器的中文標籤）；
            # 「路徑數值驗證」那一段的值另外照這裡寫死的中文比，不拿伺服器的標籤表抄。
            reasons = "、".join(names[code] for code in channel["reason_codes"]) or "無"
            status = block.locator(":scope > p").first.inner_text()
            assert status == (f"結論：{names[channel['state']]}；涵蓋：{names[channel['coverage']]}；"
                              f"路徑數值驗證：{names[channel['validation']]}；原因：{reasons}")
            plain = {"validated": "已驗證", "unvalidated": "尚未驗證"}
            assert f"；路徑數值驗證：{plain[channel['validation']]}；" in status
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
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs: result.program_fingerprint)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#content").wait_for(state="visible")
        data = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        # 頁首主畫面講白話（計算版本跟現在的程式相同），提交碼與計算指紋收在「技術細節」，一開始收著。
        assert page.locator("#fingerprint-status").inner_text() == "物理與評分設定都跟現在相同"
        header = page.locator("header").inner_text()
        assert "e53bfae" not in header and data["fingerprint_text"] not in header and "指紋" not in header
        assert page.locator("#header-details").is_visible() and _open_flags(page.locator("#header-details")) == [False]
        page.locator("#header-details summary").click()
        technical = page.locator("#header-technical").inner_text()
        assert technical == f"程式版本碼：e53bfae；物理身分前 12 碼：{data['fingerprint_text']}"
        assert commit not in page.locator("body").inner_text()
        # 排名那一行照伺服器的白話；不再說「缺的類：無」，尚未評估的每一類都寫明不算進總代價。
        ranking = page.locator("#ranking-state").inner_text()
        assert ranking == data["ranking_text"] and "缺的類" not in ranking
        pending = [item["category"] for item in data["categories"] if item["state"] == "not_evaluated"]
        assert pending and all(data["labels"][code] in ranking.split("尚未評估、不算進總代價：", 1)[1]
                               for code in pending)
        # 「起伏 RMS 差」改成中文：摘要表的「量」那一欄印「起伏差（均方根）」，整頁沒有 RMS。
        assert "起伏差（均方根）" in page.locator("#summary").inner_text()
        assert "RMS" not in page.locator("body").inner_text()
        # 暫定線那一格是「線；」與「尚未正式校準」兩段，要換行只在兩段之間換（不會剩一個「準」字在下一行）。
        first = next(item for item in data["listening_area"]["summaries"]
                     if item["role"] == data["frequency_responses"][0]["role"])
        assert page.locator("#summary tr td:nth-child(7)").first.locator("span").all_inner_texts() == [
            f"{first['limit_text']} {first['unit']}；", first["baseline_note"]]
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
    _save_old_format(tmp_path, result)
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
        assert page.get_by_role("button", name="用現在的程式重算", exact=True).is_visible()
        # 頁首的「技術細節（版本碼）」只在讀到結果時才有東西：拒收頁不畫那個空的摺疊區。
        assert page.locator("#header-details").is_hidden()
        requested = _check_rerun_starts_once(page, "#rerun", "#rerun-state")
        assert requested == [f"{base}{rejected['rerun_url']}"]
        assert page.locator("#content").is_hidden()
        # 409 回應本身瀏覽器會記一筆「載入失敗」，那是預期的；JS 自己不准拋錯。
        assert all("409" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def _save_old_format(tmp_path: Path, result: SchemeResult) -> None:
    _save(tmp_path, result)
    path = tmp_path / "results" / f"{RUN_ID}.json"
    document = json.loads(path.read_text())
    document["schema_version"] = "aosr.scheme_result.v2"
    path.write_text(json.dumps(document))


# 重算擋下來時伺服器回的樣子（共用的白話問題格式）：頁面逐條印 text，一條一行。
PROBLEMS = {"problems": [
    {"text": "寬 Ly（公尺）、地板阻抗：空著沒填", "message": "空著沒填", "fields": ["寬 Ly（公尺）", "地板阻抗"],
     "paths": ["scene.room_m.Ly", "scene.impedance_pa_s_per_m_by_wall.floor"], "details": []},
    {"text": "左聲道喇叭：座標超出房間", "message": "座標超出房間", "fields": ["左聲道喇叭"],
     "paths": ["scheme"], "details": []}]}


def test_rerun_that_does_not_start_says_why_and_can_be_pressed_again(tmp_path: Path, browser: Browser,
                                                                     result: SchemeResult) -> None:
    _save_old_format(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        page.locator("#rejection").wait_for(state="visible")
        answers = iter(("problems", "offline", "error"))

        def answer(route: Route) -> None:
            kind = next(answers)
            if kind == "offline":
                route.abort()
            elif kind == "problems":
                route.fulfill(status=422, content_type="application/json", body=json.dumps(PROBLEMS))
            else:
                route.fulfill(status=409, content_type="application/json",
                              body=json.dumps({"error": "程式已更新，網頁伺服器要重開"}))

        page.route("**/api/results/*/rerun", answer)
        button, state = page.locator("#rerun"), page.locator("#rerun-state")
        # 方案過不了檢查：一句說明加每條問題一行，按鈕放回來（沒有開始任何計算）。
        button.click()
        state.filter(has_text="座標超出房間").wait_for()
        lines = state.inner_text().split("\n")
        assert lines[0].startswith("這份結果的方案過不了現在的檢查；請在方案輸入頁打開這個方案")
        assert lines[1:] == [item["text"] for item in PROBLEMS["problems"]]
        assert button.is_enabled() and not state.locator("a").all()
        # 連不上伺服器：說白話、按鈕放回來，不讓頁面拋錯。
        button.click()
        state.filter(has_text="連不上網頁伺服器").wait_for()
        assert button.is_enabled()
        # 伺服器自己說為什麼不能算：照印那一句。
        button.click()
        state.filter(has_text="網頁伺服器要重開").wait_for()
        assert state.inner_text() == "程式已更新，網頁伺服器要重開" and button.is_enabled()
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


def _six_point_pairs(role: str, primary: str) -> tuple[PairView, ...]:
    """範例方案的六個周圍點：主位對每一點一對、周圍點彼此兩兩一對，三種量共用同一對。"""
    ends = [("primary_to_surrounding", primary, point) for point in DIRECTIONS]
    ends += [("surrounding_to_surrounding", first, second) for first, second in combinations(DIRECTIONS, 2)]
    return tuple(PairView(role=role, speaker_id=role, metric=metric, group=group, receiver_id=receiver,
                          reference_id=reference, value=0.1, value_text="0.10", unit="dB", limit=3.0,
                          limit_text="3.00", excess_text="未超過", over_limit=False, baseline_note="",
                          status_text="未超過")
                 for metric in ("tilt", "ripple_rms", "overall_level") for group, reference, receiver in ends)


def _with_six_points(real: Callable[..., ResultView]) -> Callable[..., ResultView]:
    """考卷資料只有一個周圍點；老闆的方案有六個，按鈕排得滿一整列。位置對換成伺服器自己排的六點版本，
    摘要表的最差位置對也換成六點時最長的那種名字（主位上方 ↔ 主位下方），表格寬度才跟老闆看到的一樣。"""
    def build(result: SchemeResult, *, quality_targets_path: Path) -> ResultView:
        view = real(result, quality_targets_path=quality_targets_path)
        primary = result.scheme.receiver_set.primary.receiver_id
        pairs = tuple(pair for role in view.speaker_names for pair in _six_point_pairs(role, primary))
        widest = next(item.text for item in _pair_choices(pairs, primary)
                      if {item.reference_id, item.receiver_id} == {"up", "down"})
        summaries = tuple(item.model_copy(update={"worst_pair_text": widest})
                          for item in view.listening_area.summaries)
        area = view.listening_area.model_copy(update={"pairs": pairs, "summaries": summaries,
                                                      "pair_choices": _pair_choices(pairs, primary)})
        return view.model_copy(update={"listening_area": area})
    return build


# 表格裡字排成兩行以上的格子（同一格的字有兩種以上的行底高度），回那幾格的字。
WRAPPED_CELLS_JS = """rows => rows.flatMap((row) => [...row.cells].filter((cell) => {
  const range = document.createRange();
  range.selectNodeContents(cell);
  return new Set([...range.getClientRects()].map((rect) => Math.round(rect.bottom))).size > 1;
}).map((cell) => cell.textContent))"""


def _centre_y(locator: Locator) -> float:
    box = locator.bounding_box()
    assert box is not None
    return box["y"] + box["height"] / 2


def test_peer_select_shares_the_button_row_with_six_surrounding_points(
        tmp_path: Path, browser: Browser, result: SchemeResult, monkeypatch: pytest.MonkeyPatch) -> None:
    _save(tmp_path, result)
    monkeypatch.setattr("aosr.gui.app.build_result_view", _with_six_points(build_result_view))
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=WIDTH) as watched:
        page = watched.page
        select = page.locator("#peer-pairs")
        select.wait_for()
        buttons = page.locator("#pair-buttons button")
        data = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        role = data["frequency_responses"][0]["role"]
        choices = [item for item in data["listening_area"]["pair_choices"] if item["role"] == role]
        primary = [item for item in choices if item["group"] == "primary_to_surrounding"]
        assert {item["receiver_id"] for item in primary} == set(DIRECTIONS)
        # 按鈕只寫周圍點那一端，組名在按鈕前面寫一次。
        assert buttons.all_inner_texts() == ["全部位置", *(item["button_text"] for item in primary)]
        assert page.locator("#pair-buttons .pair-group").inner_text() == "主位對周圍點："
        # 1440 寬：全部位置、六個主位對周圍點與周圍點彼此的選單排在同一列（中線高度一樣），不是選單自己掉到下一列。
        centres = [_centre_y(buttons.nth(index)) for index in range(buttons.count())]
        assert max(centres) - min(centres) < 2, centres
        assert abs(_centre_y(select) - centres[0]) < 4, (_centre_y(select), centres)
        # 摘要表（量的名字「起伏差（均方根）」、位置對名字最長時）1440 寬每一格都只佔一行：
        # 沒有「左聲道喇／叭」「0.132／dB/oct」「尚未正式校／準」「重要性加權平／均」那種折行。
        assert "起伏差（均方根）" in page.locator("#summary").inner_text()
        assert page.locator("#summary tr").evaluate_all(WRAPPED_CELLS_JS) == []
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


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
        # 按鈕只有「全部位置」與主位對周圍點（按鈕只寫周圍點那一端）；沒有周圍點彼此的配對就不畫選單。
        assert page.locator("#pair-buttons button").all_inner_texts() == [
            "全部位置", *(item["button_text"] for item in choices)]
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
        page.get_by_role("button", name=choices[0]["button_text"], exact=True).click()
        assert select.input_value() == ""
        assert "chosen" not in (select.get_attribute("class") or "")
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
