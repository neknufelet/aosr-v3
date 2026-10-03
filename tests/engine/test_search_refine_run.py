"""細算計算與重排；禁止搜尋軸、錯序、錯批、同分換首與污染搜尋狀態。"""

from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.search.ledger import read_for
from aosr.search.refine import RefineLedger, refine_order
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore, refine_scheme_id
from tests.engine._search_refine_cases import RefineCompute, refine, stopped_store


def test_verification_baseline_order_batches_and_search_preserved(tmp_path: Path) -> None:
    store, registry = stopped_store(tmp_path)
    before = SearchStatus.model_validate_json(store.status_path.read_bytes())
    order = refine_order(read_for(store).rows)
    compute = RefineCompute(store, {None: 0.0})
    status = refine(store, registry, compute)
    assert status.refine.state == "stopped" and status.refine.stop_reason == "candidates_exhausted"
    assert status.model_dump(exclude={"refine"}) == before.model_dump(exclude={"refine"})
    assert status.refine.best == "baseline" and status.refine.streak == len(order)
    assert status.refine.refined == len(order)
    assert [job.trial_number for job in compute.jobs] == [None, *order]
    assert compute.batches == [(None,), *(order[i:i + store.settings.batch_size]
                                        for i in range(0, len(order), store.settings.batch_size))]
    for job in compute.jobs:
        assert job.scheme.scene.low_frequency_axis == LowFrequencyAxis.VERIFICATION
        assert job.scheme.scheme_id == refine_scheme_id(store.search_id, job.trial_number)
        assert job.scheme.scheme_id.endswith("-verification")
        assert job.result_path == store.refine_result_path(job.trial_number)
        assert job.result_path.parent == store.refine_dir
    header, rows = RefineLedger.read(store.refine_ledger_path)
    assert [row.trial_number for row in rows] == [None, *order]
    assert header.program_fingerprint == store.identity.program_fingerprint
    assert all(row.round == 1 and row.outcome == "scored" for row in rows)
    assert status.refine.message.startswith("細算已停：")
    assert "不代表細算完成——回饋還沒做" in status.refine.message


@pytest.mark.parametrize("reason,budget,convergence", [
    ("stable", 3, 2), ("refine_budget", 4, 50), ("candidates_exhausted", 50, 50),
])
def test_stop_priority_ties_and_streak(tmp_path: Path, reason: str, budget: int, convergence: int) -> None:
    store, registry = stopped_store(tmp_path, batch=3, budget=budget, convergence=convergence)
    order = refine_order(read_for(store).rows)
    # 第一個嚴格改善，第二個同分，第三個較差：連續數手算為 2；同批也到上限時 stable 優先。
    compute = RefineCompute(store, {None: 4.0, order[0]: 1.0, order[1]: 1.0, order[2]: 2.0})
    status = refine(store, registry, compute).refine
    assert status.stop_reason == reason
    expected = order[:3] if reason == "stable" else order[:budget] if reason == "refine_budget" else order
    assert status.refined == len(expected)
    assert status.best == order[0] and status.streak == len(expected) - 1
    # 音色只有殘差計價：殘差 1 dB / 較差參考 4 dB，兩聲道相同、類別權重 1。
    assert status.best_total_cost == 1.0 / 4.0


def test_missing_baseline_fails_and_excluded_baseline_pins(tmp_path: Path) -> None:
    store, registry = stopped_store(tmp_path / "missing")
    status = refine(store, registry, RefineCompute(store, {}, missing=frozenset({None})))
    assert status.refine.state == "failed" and "原方案缺類" in status.refine.message
    assert not RefineLedger.read(store.refine_ledger_path)[1]
    store, registry = stopped_store(tmp_path / "excluded")
    order = refine_order(read_for(store).rows)
    status = refine(store, registry, RefineCompute(store, {}, excluded=frozenset({None}), different=frozenset({order[0]})))
    rows = RefineLedger.read(store.refine_ledger_path)[1]
    assert rows[0].outcome == "excluded" and rows[0].total_cost is None
    assert rows[1].outcome == "not_comparable" and rows[1].total_cost is None
    assert status.refine.best != "baseline"


def test_report_search_paragraph_unchanged_refine_reason_visible(tmp_path: Path) -> None:
    from aosr.search.report import build_report, render_text
    from tests.engine._search_run_cases import RUN_DATE

    store, registry = stopped_store(tmp_path, budget=3)
    before = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    refine(store, registry, RefineCompute(store, {}))
    after = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert before.split("\n\n")[0] == after.split("\n\n")[0]
    paragraph = after.split("\n\n")[1]
    assert "細算做完沒" in paragraph and "已停（用完細算上限）" in paragraph
    assert "回饋還沒做" in paragraph


def test_missing_and_incomparable_candidates_have_no_cost(tmp_path: Path) -> None:
    store, registry = stopped_store(tmp_path, budget=3)
    order = refine_order(read_for(store).rows)
    compute = RefineCompute(store, {}, missing=frozenset({order[0]}), different=frozenset({order[1]}))
    status = refine(store, registry, compute)
    rows = RefineLedger.read(store.refine_ledger_path)[1]
    assert [row.outcome for row in rows] == ["scored", "not_evaluated", "not_comparable", "scored"]
    assert [row.total_cost for row in rows[1:3]] == [None, None]
    assert status.refine.streak == 3 and status.refine.best == "baseline"


def test_budget_has_priority_over_exhaustion(tmp_path: Path) -> None:
    store, registry = stopped_store(tmp_path)
    order = refine_order(read_for(store).rows)
    # 重新建搜尋把细算上限定在候選總數；最後一批同時耗盡兩者時，上限優先。
    store, registry = stopped_store(tmp_path / "bounded", budget=len(order))
    status = refine(store, registry, RefineCompute(store, {}))
    assert status.refine.stop_reason == "refine_budget"
    assert status.refine.refined == len(refine_order(read_for(store).rows))


def test_baseline_never_counts_towards_stable_streak(tmp_path: Path) -> None:
    store, registry = stopped_store(tmp_path, batch=1, budget=4, convergence=2)
    status = refine(store, registry, RefineCompute(store, {None: 0.0}))
    assert status.refine.stop_reason == "stable" and status.refine.refined == 2
    assert status.refine.streak == 2 and status.refine.best == "baseline"


def test_later_strict_winner_resets_streak_and_stopping_waits_for_batch(tmp_path: Path) -> None:
    store, registry = stopped_store(tmp_path, batch=3, convergence=1)
    order = refine_order(read_for(store).rows)
    # 第一個比原方案差會達到連續 1；仍須算第二個（嚴格改善歸零），第三個才重新累積到 1。
    compute = RefineCompute(store, {None: 2.0, order[0]: 3.0, order[1]: 1.0, order[2]: 2.0})
    status = refine(store, registry, compute).refine
    assert status.stop_reason == "stable" and status.best == order[1]
    assert status.streak == 1 and status.refined == store.settings.batch_size
    assert [job.trial_number for job in compute.jobs] == [None, *order[:store.settings.batch_size]]
    assert status.best_total_cost == 1.0 / 4.0
