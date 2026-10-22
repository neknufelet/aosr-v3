"""第一階段搜尋考卷：只造有限元素能量，保存完整合法結果。"""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from dataclasses import replace

import pytest

from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.reporting.result import ResultOrigin
from aosr.reporting.stage_one_result import run_stage_one, save_stage_one_result
from aosr.search.run import CandidateJob, SearchStatus
from aosr.search.stage_one_worker import StageOneComputed
from aosr.search.layout_settings import LayoutSettings
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine import _stage_one_cases as cases
from tests.engine._search_store_cases import settings_document, purpose_settings
from tests.engine._fem_cli_cases import fake_slices


def placed_slices(*, room: Room, candidates: Mapping[str, tuple[Mapping[str, Point], Mapping[str, Point]]],
                  wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
                  density_kg_m3: float, sound_speed_m_s: float, frequency_indices: Sequence[int],
                  ) -> dict[str, dict[tuple[str, str], tuple[float, ...]]]:
    """替身能量跟著喇叭到座位的距離變：不同擺位合分不同，排序與逐候選傳分的考卷才分得出。"""
    energies = fake_slices(room=room, candidates=candidates, wall_impedances=wall_impedances,
        frequencies_hz=frequencies_hz, density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s,
        frequency_indices=frequency_indices)
    return {name: {(speaker, seat): tuple(
        value * (1.5 + math.sin(frequencies_hz[index] * math.dist(sources[speaker].as_tuple(), receivers[seat].as_tuple()) / 30.))
        for value, index in zip(values, frequency_indices, strict=True))
        for (speaker, seat), values in energies[name].items()}
        for name, (sources, receivers) in candidates.items()}


def make_store(tmp_path: Path, *, workers: int = 1, budget: int = 9, layout: LayoutSettings | None = None) -> SearchStore:
    item = cases.scheme()
    settings = SearchSettings.model_validate(settings_document() | {
        "purpose": item.purpose, "stage": "stage_one_fem_only", "max_workers": workers,
        "budget": budget, "convergence_run": 100} | ({} if layout is None else {"layout": layout}))
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
            parts = cases.shards(self.monkeypatch, job.scheme, solver=placed_slices)
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
