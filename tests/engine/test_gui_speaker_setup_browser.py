"""真瀏覽器驗提示、存檔、差異列、CSV 及中文問題；畫面答案照 #559 施工單。"""
import csv
import io
import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page

from aosr.gui.compare_view import scheme_differences, summary_csv
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine import _speaker_setup_cases as cases
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_compare_routes import _files
from tests.engine.test_gui_compare_browser import _data
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.mark.parametrize("mount,representative,title", [
    ("stand", True, "書架喇叭、放腳架（代表模型，非實際型號）"),
    ("desk", False, "書架喇叭、放桌面（實際型號）"),
    ("floor", True, "落地喇叭、放地面（代表模型，非實際型號）"),
])
def test_input_notice_save_as_and_clearing(browser: Browser, tmp_path: Path,
                                          mount: str, representative: bool, title: str) -> None:
    document = cases.document(mount)
    cast(dict[str, object], document["speaker_setup"])["representative"] = representative
    scheme = Scheme.model_validate(document | {"scheme_id": "loaded"})
    (tmp_path / "schemes").mkdir()
    path = tmp_path / "schemes" / "loaded.json"
    path.write_text(scheme.model_dump_json())
    plain_path = tmp_path / "schemes" / "plain.json"
    plain_path.write_text(Scheme.model_validate(cases.document() | {"scheme_id": "plain", "speaker_setup": None}).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        notice = page.locator("#speaker-setup-notice")
        assert not notice.is_visible() and notice.evaluate("el => el.hidden")
        page.locator("#scheme-list").select_option("loaded")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'loaded'")
        assert notice.is_visible() and notice.evaluate("el => el.tagName") == "P"
        assert notice.inner_text() == (f"這份方案設了喇叭：{title}；這一頁還不能修改喇叭設定，平面圖與側面圖已畫出箱體；"
                                       "存檔與計算照方案檔裡的設定算。")
        page.locator("#speaker-left-x").fill("2.01")
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/loaded")
                                  and response.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok
        persisted = Scheme.model_validate_json(path.read_bytes())
        assert persisted.speakers["left"].x == 2.01
        assert persisted.speaker_setup == scheme.speaker_setup
        page.locator("#save-as-id").fill("copy")
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/copy")
                                  and response.request.method == "PUT") as copied:
            page.locator("#save-as").click()
        assert copied.value.ok
        assert Scheme.model_validate_json((tmp_path / "schemes" / "copy.json").read_bytes()).speaker_setup == scheme.speaker_setup
        page.locator("#scheme-section").screenshot(path=str(tmp_path / f"input-speaker-{mount}.png"))
        page.locator("#scheme-list").select_option("plain")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'plain'")
        assert not notice.is_visible() and notice.evaluate("el => el.hidden")
        assert notice.inner_text() == ""
        _assert_quiet(watched)


def _changed_result(original: SchemeResult, identifier: str, setup: dict[str, object] | None,
                    purpose: str | None = None) -> SchemeResult:
    # 只改不進逐對物理輸入的 metadata（方案描述）；沿用實跑結果與座標，不偽造求解答案。
    document = original.scheme.model_dump(mode="json") | {"scheme_id": identifier, "speaker_setup": setup}
    if purpose is not None:
        document["purpose"] = purpose
    scheme = Scheme.model_validate(document)
    return original.model_copy(update={"scheme": scheme, "candidate": original.candidate.model_copy(
        update={"candidate_id": identifier, "purpose": scheme.purpose})})


@pytest.mark.parametrize("scenario", ["both", "one", "purpose"])
def test_compare_speaker_rows_and_downloaded_csv(browser: Browser, tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory, worker_id: str, scenario: str) -> None:
    original = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    left_setup = cases.setup()
    right_setup = cases.setup("floorstanding", "floor", representative=False)
    cabinet = cast(dict[str, object], right_setup["cabinet"])
    cabinet.update(width_m=0.25, depth_m=0.38, height_m=1.605,
                   acoustic_center_behind_front_m=0.01, acoustic_center_above_bottom_m=1.25)
    left = _changed_result(original, "speaker-a", left_setup if scenario != "one" else None)
    right = _changed_result(original, "speaker-b", right_setup)
    expected = {
        "喇叭類型": ("書架喇叭", "落地喇叭"), "喇叭擺法": ("腳架", "地面"),
        "代表模型": ("代表模型，非實際型號", "實際型號"),
        "箱寬": ("0.21 公尺", "0.25 公尺"), "箱深": ("0.28 公尺", "0.38 公尺"),
        "箱高": ("0.355 公尺", "1.605 公尺"), "聲學中心離前面板": ("0 公尺", "0.01 公尺"),
        "聲學中心離箱底": ("0.205 公尺", "1.25 公尺"),
    }
    if scenario == "one":
        expected = {title: ("未設定", values[1]) for title, values in expected.items()}
    a_id, b_id = "b" * 32, "c" * 32
    with _serve(tmp_path) as url:
        _files(tmp_path, left, a_id)
        _files(tmp_path, right, b_id)
        with _open(browser, f"{url}/compare/{a_id}/{b_id}", response_path=f"/api/compare/{a_id}/{b_id}") as watched:
            page = watched.page
            assert watched.response is not None and watched.response.ok, page.locator("body").inner_text()
            if scenario == "purpose":
                _show_purpose_changes(page, url, left, right, a_id, b_id)
            page.locator("#changes table").wait_for()
            rows = {cells[0]: tuple(cells[1:]) for row in page.locator("#changes tr").all()
                    if (cells := row.locator("td").all_inner_texts())}
            for title, values in expected.items():
                assert rows[title] == values
            assert "其他設定（未逐項列出）" not in rows
            if scenario == "purpose":
                assert "方案用途" in rows
                assert rows["方案用途"] == ("聆聽方案", "工作方案")
            with page.expect_download() as downloaded:
                page.locator("#download-summary").click()
            path = tmp_path / "summary.csv"
            downloaded.value.save_as(path)
            csv_rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
            for title, values in expected.items():
                assert any(title in row and list(values) == row[-2:] for row in csv_rows)
            page.locator("#changes").screenshot(path=str(tmp_path / f"compare-speaker-{scenario}.png"))
            _assert_quiet(watched)


def _show_purpose_changes(page: Page, url: str, left: SchemeResult, right: SchemeResult,
                         a_id: str, b_id: str) -> None:
    """主線的結果比較拒收不同用途；隔離這道門，驗正式差異收集、CSV 與真頁面顯示。

    伺服器實際不會出現這種組合（比較端點用途不同回 409），這一情境的差異列與 CSV 由考卷呼叫正式函式產生、
    經攔截交給頁面，伺服器的 /export/summary 沒被打到；真伺服器那條路由 both、one 兩個情境證明。

    只注入比較頁的顯示資料，不偽造物理答案；預期每格文字仍由本題手寫表判。
    """
    first = Scheme.model_validate(left.scheme.model_dump() | {"purpose": "聆聽方案"})
    other = Scheme.model_validate(right.scheme.model_dump() | {"purpose": "工作方案"})
    changes = scheme_differences(first, other)
    view = _data(page, url).model_copy(update={"changes": changes})
    response = page.request.get(f"{url}/api/compare/{a_id}/{b_id}")
    assert response.ok
    body = response.json() | {"changes": [row.model_dump() for row in changes]}
    page.route(f"**/api/compare/{a_id}/{b_id}", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(body)))
    page.route(f"**/api/compare/{a_id}/{b_id}/export/summary", lambda route: route.fulfill(
        status=200, content_type="text/csv; charset=utf-8", body=summary_csv(view),
        headers={"Content-Disposition": 'attachment; filename="summary.csv"'}))
    page.reload(wait_until="networkidle")


def test_input_speaker_problem_messages_are_chinese(browser: Browser, tmp_path: Path) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        for document, field, choices in (
            (cases.document() | {"speaker_setup": cases.setup() | {"kind": "bad"}}, "喇叭類型", "書架喇叭"),
            (cases.document() | {"speaker_setup": cases.setup() | {"mount": "bad"}}, "喇叭擺法", "桌面"),
            (cases.document("desk") | {"furniture": []}, "喇叭類型與擺法", "現在有 0 件"),
            (cases.document("floor", left_z=0.81), "左聲道喇叭 z 座標", "聲學中心離地 0.8 m"),
        ):
            # 方案設定尚無編輯器，從正式驗證端點取得訊息，再經頁面既有顯示路徑印出。
            page.evaluate("async doc => { const result = await api('/api/validate', 'POST', doc); "
                          "document.getElementById('messages').textContent = problemLines(result.problems); }", document)
            text = page.locator("#messages").inner_text()
            assert field in text and choices in text
            for key in ("speaker_setup", "bookshelf", "floorstanding", "mount", "cabinet"):
                assert key not in text
        _assert_quiet(watched)


def test_notice_falls_back_to_one_sentence_when_labels_fail(browser: Browser, tmp_path: Path) -> None:
    # 名稱表載不到時，提示整句換成備用句：不逐格代入英文代號（bookshelf、stand），也不出現半句「未載入」。
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "loaded.json").write_text(
        Scheme.model_validate(cases.document() | {"scheme_id": "loaded"}).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, "about:blank") as watched:
        page = watched.page
        page.route("**/api/labels", lambda route: route.fulfill(
            status=500, content_type="application/json", body='{"error": "名稱表壞了"}'))
        page.goto(url, wait_until="networkidle")
        page.locator("#scheme-list").select_option("loaded")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'loaded'")
        notice = page.locator("#speaker-setup-notice")
        assert notice.is_visible()
        assert notice.inner_text() == ("這份方案設了喇叭類型與擺法（中文名沒載到）；這一頁還不能修改喇叭設定，平面圖與側面圖已畫出箱體；"
                                       "存檔與計算照方案檔裡的設定算。")
        # 故意讓名稱表回 500，主控台只准出現那一筆；頁面不准有錯。
        assert all("500" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []
