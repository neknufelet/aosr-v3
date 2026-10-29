"""真瀏覽器驗提示、存檔、差異列、CSV 及中文問題；畫面答案照 #559 施工單。"""
import csv
import io
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page

from aosr.reporting.scheme import Scheme
from tests.engine import _speaker_setup_cases as cases
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_compare_routes import _files
from tests.engine.test_gui_speaker_setup import SPEAKER_ROWS, changed_result, compare_setups
from tests.engine.test_scheme_pipeline import shared_control_result


MOUNTS = (
    ("stand", True, "書架喇叭、放腳架（代表模型，非實際型號）"),
    ("desk", False, "書架喇叭、放桌面（實際型號）"),
    ("floor", True, "落地喇叭、放地面（代表模型，非實際型號）"),
)


def test_input_notice_save_as_and_clearing(browser: Browser, tmp_path: Path) -> None:
    """三種擺設同一頁輪流開、存、另存，再開沒設喇叭的方案清掉提示（#972 由三格參數合成一題，每格斷言原樣保留）。"""
    (tmp_path / "schemes").mkdir()
    schemes = {}
    for mount, representative, _ in MOUNTS:
        document = cases.document(mount)
        cast(dict[str, object], document["speaker_setup"])["representative"] = representative
        schemes[mount] = Scheme.model_validate(document | {"scheme_id": f"loaded-{mount}"})
        (tmp_path / "schemes" / f"loaded-{mount}.json").write_text(schemes[mount].model_dump_json())
    plain_path = tmp_path / "schemes" / "plain.json"
    plain_path.write_text(Scheme.model_validate(cases.document() | {"scheme_id": "plain", "speaker_setup": None}).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        notice = page.locator("#speaker-setup-notice")
        assert not notice.is_visible() and notice.evaluate("el => el.hidden")
        for mount, _, title in MOUNTS:
            _check_notice_save_and_clear(page, tmp_path, mount, title, schemes[mount])
        _assert_quiet(watched)


def _check_notice_save_and_clear(page: Page, tmp_path: Path, mount: str, title: str, scheme: Scheme) -> None:
    name = scheme.scheme_id
    path = tmp_path / "schemes" / f"{name}.json"
    notice = page.locator("#speaker-setup-notice")
    page.locator("#scheme-list").select_option(name)
    page.locator("#open-scheme").click()
    page.wait_for_function(f"document.getElementById('save-id').value === '{name}'")
    assert notice.is_visible() and notice.evaluate("el => el.tagName") == "P", mount
    # 編輯器已能改設定；取消舊「不能修改」宣稱，仍守存檔和讀回原值。
    assert notice.inner_text() == (f"這份方案設了喇叭：{title}；下方可修改喇叭設定，平面圖與側面圖照方案檔畫箱體。")
    page.locator("#speaker-left-x").fill("2.01")
    with page.expect_response(lambda response: response.url.endswith(f"/api/schemes/{name}")
                              and response.request.method == "PUT") as saved:
        page.locator("#save").click()
    assert saved.value.ok
    persisted = Scheme.model_validate_json(path.read_bytes())
    assert persisted.speakers["left"].x == 2.01
    assert persisted.speaker_setup == scheme.speaker_setup
    copy = f"copy-{mount}"
    page.locator("#save-as-id").fill(copy)
    with page.expect_response(lambda response: response.url.endswith(f"/api/schemes/{copy}")
                              and response.request.method == "PUT") as copied:
        page.locator("#save-as").click()
    assert copied.value.ok
    assert Scheme.model_validate_json((tmp_path / "schemes" / f"{copy}.json").read_bytes()).speaker_setup == scheme.speaker_setup
    page.locator("#scheme-section").screenshot(path=str(tmp_path / f"input-speaker-{mount}.png"))
    page.locator("#scheme-list").select_option("plain")
    page.locator("#open-scheme").click()
    page.wait_for_function("document.getElementById('save-id').value === 'plain'")
    assert not notice.is_visible() and notice.evaluate("el => el.hidden"), mount
    assert notice.inner_text() == ""


def test_compare_speaker_rows_and_downloaded_csv(browser: Browser, tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    """兩邊都設了喇叭：真伺服器的差異列畫在頁面上、下載的摘要 CSV 逐格相同。

    #972：原本另有兩格。[one]（只有一邊設了喇叭，A 欄全是「未設定」）搬去路由考卷
    test_gui_speaker_setup::test_compare_route_marks_unset_speaker_side_in_rows_and_csv；網頁照印差異列由本題守。
    [purpose]（攔截注入伺服器不會產生的用途差異）刪掉：喇叭差異列與用途差異同時存在由
    test_gui_speaker_setup::test_speaker_difference_survives_simultaneous_purpose_change 守。
    """
    original = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    left_setup, right_setup = compare_setups()
    left = changed_result(original, "speaker-a", left_setup)
    right = changed_result(original, "speaker-b", right_setup)
    expected = SPEAKER_ROWS
    a_id, b_id = "b" * 32, "c" * 32
    with _serve(tmp_path) as url:
        _files(tmp_path, left, a_id)
        _files(tmp_path, right, b_id)
        with _open(browser, f"{url}/compare/{a_id}/{b_id}", response_path=f"/api/compare/{a_id}/{b_id}") as watched:
            page = watched.page
            assert watched.response is not None and watched.response.ok, page.locator("body").inner_text()
            page.locator("#changes table").wait_for()
            rows = {cells[0]: tuple(cells[1:]) for row in page.locator("#changes tr").all()
                    if (cells := row.locator("td").all_inner_texts())}
            for title, values in expected.items():
                assert rows[title] == values
            assert "其他設定（未逐項列出）" not in rows
            with page.expect_download() as downloaded:
                page.locator("#download-summary").click()
            path = tmp_path / "summary.csv"
            downloaded.value.save_as(path)
            csv_rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
            for title, values in expected.items():
                assert any(title in row and list(values) == row[-2:] for row in csv_rows)
            page.locator("#changes").screenshot(path=str(tmp_path / "compare-speaker-both.png"))
            _assert_quiet(watched)


def test_input_speaker_problem_messages_are_chinese(browser: Browser, tmp_path: Path) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        for document, field, choices in (
            (cases.document() | {"speaker_setup": cases.setup() | {"kind": "bad"}}, "喇叭類型", "書架喇叭"),
            (cases.document() | {"speaker_setup": cases.setup() | {"mount": "bad"}}, "喇叭擺法", "桌面"),
            (cases.document("desk") | {"furniture": []}, "喇叭類型與擺法", "現在有 0 件"),
            (cases.document("floor", left_z=0.81), "左聲道喇叭 z 座標", "聲學中心離地 0.8 m"),
        ):
            # 刻意不合法的選項由正式驗證端點送入，驗頁面既有中文顯示路徑。
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
        assert notice.inner_text() == ("這份方案設了喇叭類型與擺法（中文名沒載到）；下方可修改喇叭設定，平面圖與側面圖照方案檔畫箱體。")
        # 故意讓名稱表回 500，主控台只准出現那一筆；頁面不准有錯。
        assert all("500" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []
