"""結果頁四態實際瀏覽器、只查快取自動請求、補算與原结果保持可見。

#972：原本另有「計算行程被殺、結果頁寫出離開原因」一題。被殺的狀態、不能補算與原因原文（離開碼在錯誤輸出之前）
由路由考卷 test_gui_modal_jobs::test_stopped_and_killed_jobs_show_cause[kill] 守；網頁把失敗原因印進 #modal-reason、
藏補算鈕由本檔四態題的 FAILED 格守；輪詢途中換狀態由補算那一題（停止→未計算、放行→已診斷不計分）守。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.physics.fem_modal_check import FemModalCheck
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine._modal_cases import runner, sample
from tests.engine.test_gui_browser import RUN_ID, _assert_quiet, _assert_text_is_formatted, _open, _save, _serve, browser
from tests.engine.test_scheme_pipeline import shared_control_result
from tests.engine.test_gui_modal_jobs import wait_for_args
from tests.engine.test_gui_modal_view import checked_sample


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


@pytest.mark.parametrize("state", list(ModalDiagnosisState))
def test_result_modal_four_states_screenshot(tmp_path: Path, browser: Browser, result: SchemeResult,
                                            state: ModalDiagnosisState) -> None:
    _save(tmp_path, result)
    diagnosis = sample(result.scheme)[0] if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED else ModalDiagnosis(
        state=state, reason_text="ValueError('查位置原始錯誤')" if state is ModalDiagnosisState.FAILED else None,
        reason_code="unsupported_impedance" if state is ModalDiagnosisState.OUT_OF_SCOPE else None)
    if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
        check = sample(result.scheme)[1].spectrum.check
        assert check is not None
        default_weyl = FemModalCheck.__dataclass_fields__["weyl_terms"].default
        assert isinstance(default_weyl, str)
        diagnosis = checked_sample(replace(check, weyl_terms=default_weyl), result.scheme)
    with _serve(tmp_path, modal_runner=runner(tmp_path, diagnosis)) as base, _open(
            browser, f"{base}/results/{RUN_ID}", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("() => !['查快取中', '計算低頻模態診斷中'].includes(document.querySelector('#modal-state').textContent)")
        expected = {ModalDiagnosisState.DIAGNOSED_NOT_SCORED: "已診斷不計分", ModalDiagnosisState.NOT_COMPUTED: "未計算",
                    ModalDiagnosisState.FAILED: "失敗", ModalDiagnosisState.OUT_OF_SCOPE: "範圍外"}[state]
        assert page.locator("#modal-state").inner_text() == expected
        assert page.locator("#modal-calculate").is_visible() == (state is ModalDiagnosisState.NOT_COMPUTED)
        assert page.locator("#content").is_visible() and page.locator("#chart canvas").is_visible()
        if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
            assert "不是只有一個共振" in page.locator("#modal-report").inner_text()
            assert "不含稜邊項" in page.locator("#modal-report").inner_text()
            formula = page.locator("#modal-report details").filter(has_text="各段個數對 Weyl")
            assert not formula.locator("p").is_visible()
            assert "Neumann:" not in page.locator("#modal-report").inner_text()
            formula.locator("summary").click()
            assert diagnosis.room_layer is not None
            assert diagnosis.room_layer.check_summary.weyl_terms in formula.locator("p").inner_text()
            page.locator("#modal-report summary").first.click()
            assert "參考線範圍外，未評估" in page.locator("#modal-report").inner_text()
        if state is ModalDiagnosisState.FAILED:
            assert page.locator("#modal-reason").inner_text() == diagnosis.reason_text
        page.locator("#modal-diagnosis").scroll_into_view_if_needed()
        page.screenshot(path=str(tmp_path / f"modal-{state.value}.png"))
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_button_computes_without_hiding_original_result(tmp_path: Path, browser: Browser,
                                                       result: SchemeResult) -> None:
    import json
    _save(tmp_path, result)
    command = runner(tmp_path, ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED))
    with _serve(tmp_path, modal_runner=command) as base, _open(browser, f"{base}/results/{RUN_ID}") as watched:
        page = watched.page
        page.locator("#modal-calculate").wait_for(state="visible")
        assert "--cache-only" in json.loads((tmp_path / "modal-args.json").read_text())
        before = page.locator("#ranking-state").inner_text()
        runner(tmp_path, sample(result.scheme)[0], wait=True)
        with page.expect_response(lambda response: response.request.method == "POST" and
                                  response.url.endswith(f"/api/results/{RUN_ID}/modal")) as event:
            page.locator("#modal-calculate").click()
        job = event.value.json()["job"]
        page.locator("#modal-stop").wait_for(state="visible")
        arguments = wait_for_args(job)
        assert page.locator("#content").is_visible() and page.locator("#ranking-state").inner_text() == before
        assert "--cache-only" not in arguments
        page.locator("#modal-stop").click()
        page.locator("#modal-calculate").wait_for(state="visible")
        assert page.locator("#modal-state").inner_text() == "未計算"
        assert "停止" in page.locator("#modal-reason").inner_text()
        page.locator("#modal-diagnosis").scroll_into_view_if_needed()
        page.screenshot(path=str(tmp_path / "modal-stopped.png"))
        page.locator("#modal-calculate").click()
        page.locator("#modal-stop").wait_for(state="visible")
        (tmp_path / "modal-release").touch()
        page.wait_for_function("() => document.querySelector('#modal-state').textContent === '已診斷不計分'")
        assert page.locator("#ranking-state").inner_text() == before
        _assert_quiet(watched)
