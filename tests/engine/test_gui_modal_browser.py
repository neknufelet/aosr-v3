"""結果頁四態實際瀏覽器、只查快取自動請求、補算與原结果保持可見。"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine._modal_cases import runner, sample
from tests.engine.test_gui_browser import RUN_ID, _assert_quiet, _assert_text_is_formatted, _open, _save, _serve, browser
from tests.engine.test_scheme_pipeline import shared_control_result


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
            assert "不是一個共振" in page.locator("#modal-report").inner_text()
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
        page.locator("#modal-calculate").click()
        page.locator("#modal-stop").wait_for(state="visible")
        assert page.locator("#content").is_visible() and page.locator("#ranking-state").inner_text() == before
        assert "--cache-only" not in json.loads((tmp_path / "modal-args.json").read_text())
        (tmp_path / "modal-release").touch()
        page.wait_for_function("() => document.querySelector('#modal-state').textContent === '已診斷不計分'")
        assert page.locator("#ranking-state").inner_text() == before
        _assert_quiet(watched)
