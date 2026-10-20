"""第一階段搜尋考卷：只造有限元素能量，保存完整合法結果。"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from dataclasses import replace

import pytest

from aosr.config.paths import config_path
from aosr.reporting.result import ResultOrigin
from aosr.reporting.stage_one_result import run_stage_one, save_stage_one_result
from aosr.search.run import CandidateJob, SearchStatus
from aosr.search.stage_one_worker import StageOneComputed
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine import _stage_one_cases as cases
from tests.engine._search_store_cases import settings_document, purpose_settings


def make_store(tmp_path: Path, *, workers: int = 1, budget: int = 9) -> SearchStore:
    item = cases.scheme()
    settings = SearchSettings.model_validate(settings_document() | {
        "purpose": item.purpose, "stage": "stage_one_fem_only", "max_workers": workers,
        "budget": budget, "convergence_run": 100})
    kwargs = cases.run_kwargs()
    return SearchStore.create(tmp_path / "searches", project=item, settings=settings,
        identity=SearchIdentity(kwargs["physics_identity"], kwargs["program_fingerprint"], purpose_settings(item.purpose)),
        versions={"python": "test", "numpy": "test", "optuna": "test"})


class Killed(BaseException):
    """模擬沒有執行錯誤出口的強制結束。"""


class FakeCompute:
    def __init__(self, store: SearchStore, monkeypatch: pytest.MonkeyPatch, *,
                 unavailable: frozenset[int] = frozenset(), different: frozenset[int] = frozenset(),
                 kill_after: int | None = None) -> None:
        self.store, self.monkeypatch = store, monkeypatch
        self.unavailable, self.different = unavailable, different
        self.kill_after = kill_after
        self.calls: list[int] = []

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        from aosr.search.stage_one_worker import StageOneComputed
        for job in (reversed(jobs) if workers > 1 else jobs):
            if self.kill_after is not None and len(self.calls) >= self.kill_after:
                raise Killed()
            number = job.trial_number
            assert number is not None
            self.calls.append(number)
            parts = cases.shards(self.monkeypatch, job.scheme)
            result = run_stage_one(job.scheme, origin=ResultOrigin(kind="search_candidate", search_id=self.store.search_id,
                trial_number=number), shards=parts, **cases.run_kwargs())
            if number in self.unavailable:
                from aosr.scoring.stage_one_low_frequency_contract import StageOneReasonCode
                from aosr.scoring.stage_one_score import score_stage_one
                evaluation = result.evaluation.model_copy(update={"speakers": tuple(item.model_copy(update={
                    "state": "unavailable", "payload": None, "reason_codes": (StageOneReasonCode.NON_POSITIVE_ENERGY,)})
                    for item in result.evaluation.speakers)})
                result = result.model_copy(update={"evaluation": evaluation, "score": score_stage_one(evaluation, result.score_settings)})
            if number in self.different:
                from aosr.scoring.stage_one_score import score_stage_one
                settings = replace(result.score_settings, fingerprint="a" * 64)
                result = result.model_copy(update={"score_settings": settings, "score": score_stage_one(result.evaluation, settings)})
            SearchStore.scheme_path_for(job.result_path).write_text(job.scheme.model_dump_json())
            save_stage_one_result(result, job.result_path)
            yield StageOneComputed(job, result, .25)


def run(store: SearchStore, compute: FakeCompute) -> SearchStatus:
    from aosr.search.stage_one_run import start_stage_one_search
    return start_stage_one_search(store, compute=compute, probe=lambda: store.identity,
        registry_path=config_path("quality_targets.toml"), run_date=cases.run_kwargs()["run_date"],
        engine_version=store.identity.program_fingerprint)
