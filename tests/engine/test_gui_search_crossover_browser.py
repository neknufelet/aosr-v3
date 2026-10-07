"""真瀏覽器核交接提醒、距離、輪詢、正常內容不標紅，並留本次暫存截圖。"""
from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.reporting.display import SPEAKERS
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity
from aosr.search import crossover_sensitivity as module
from aosr.search.crossover_record import INCOMPLETE, STALE, VERDICTS, summary_path, write_summary
from aosr.search.sampler import Excluded, RankingZone, Scored
from tests.engine._crossover_cases import evaluate, prepared
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser


def test_crossover_browser_verdicts_distances_and_screenshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            browser: Browser) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        block = page.locator("#crossover")
        block.wait_for()
        shown = block.inner_text()
        assert VERDICTS[summary.verdict] in shown
        assert "試算 9" in shown and all(f"{label}相距" in shown for label in SPEAKERS.values()) and "主位相距" in shown
        assert not block.locator("p.notice").all()
        block.screenshot(path=str(tmp_path / "688-crossover-sensitive.png"))
        for verdict in ("stable", "unverified"):
            changed = summary.model_copy(update={"verdict": verdict})
            write_summary(store.path, changed)
            page.wait_for_function("text => document.getElementById('crossover').textContent.includes(text)", arg=VERDICTS[verdict], timeout=12000)
            assert not block.locator("p.notice").all()
        write_summary(store.path, summary.model_copy(update={"completed": False}))
        page.wait_for_function("text => document.getElementById('crossover').textContent.includes(text)", arg=INCOMPLETE, timeout=12000)
        assert not block.locator("p.notice").all()
        block.screenshot(path=str(tmp_path / "688-crossover-incomplete.png"))
        store.status_path.write_text(status.model_copy(update={"asked": status.asked + 1}).model_dump_json())
        page.wait_for_function("text => document.getElementById('crossover').textContent.includes(text)", arg=STALE, timeout=12000)
        assert not block.locator("p.notice").all()
        block.screenshot(path=str(tmp_path / "688-crossover-stale.png"))
        summary_path(store.path).write_text("{")
        page.wait_for_function("() => document.querySelector('#crossover p.notice') !== null", timeout=12000)
        assert "讀不到" in block.inner_text()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


@pytest.mark.parametrize("scenario", ["sensitive", "excluded", "failed"])
def test_crossover_review_browser_states_and_screenshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        browser: Browser, scenario: str) -> None:
    store, registry, status = prepared(tmp_path, swap=scenario != "excluded")
    monkeypatch.setattr(module, "reevaluate", evaluate)
    if scenario == "excluded":
        original_score = module.Evaluator.score
        legacy = next(s for s in module.stitchings(200) if s.record.key == "legacy")
        def score(self: module.Evaluator, result: SchemeResult, candidate: CandidateEvaluation,
                  pinned: tuple[ComparisonIdentity, ...]) -> Scored | Excluded:
            points = result.pairs[0].report.points
            assert points
            if result.origin.trial_number == 7 and points[0].w_geo == legacy.geo_weight(points[0].frequency_hz):
                return Excluded(RankingZone.ELIMINATED)
            return original_score(self, result, candidate, pinned)
        monkeypatch.setattr(module.Evaluator, "score", score)
    if scenario == "failed":
        def broken(*args: object, **kwargs: object) -> object:
            raise KeyError("f_s_hz")
        monkeypatch.setattr(module, "reevaluate", broken)
        with pytest.raises(KeyError) as error:
            module.attach_crossover(store, status=status, quality_targets_path=registry)
        module.record_crossover_error(store, status, error.value)
    else:
        module.attach_crossover(store, status=status, quality_targets_path=registry)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        block = page.locator("#crossover")
        block.wait_for()
        shown = block.inner_text()
        if scenario == "excluded":
            assert VERDICTS["unverified"] in shown and "上一代接法" in shown
            assert "正式第一名在此接法被淘汰" in shown and "距離：" not in shown
        elif scenario == "sensitive":
            assert VERDICTS["sensitive"] in shown and all(f"{label}相距" in shown for label in SPEAKERS.values())
        else:
            assert "交接敏感度計算失敗：KeyError：" in shown
        assert bool(block.locator("p.notice").all()) is (scenario == "failed")
        block.screenshot(path=str(tmp_path / f"688-review-{scenario}.png"))
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
