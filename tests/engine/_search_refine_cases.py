"""細算考卷只替換計算：真搜尋、真排名、真帳本與可控殘差。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.store import SearchStore
from tests.engine import _ranking_fixtures as fixtures
from tests.engine._search_review_cases import with_matching
from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, Killed, make_store, run


class SearchCompute(FakeCompute):
    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for job in jobs:
            SearchStore.scheme_path_for(job.result_path).write_text(job.scheme.model_dump_json())
        yield from super().__call__(jobs, workers)


class RefineCompute:
    def __init__(self, store: SearchStore, values: dict[int | None, float], *,
                 kill_after: int | None = None, missing: frozenset[int | None] = frozenset(),
                 different: frozenset[int | None] = frozenset(),
                 excluded: frozenset[int | None] = frozenset()) -> None:
        self.store, self.values = store, values
        self.kill_after, self.missing, self.different = kill_after, missing, different
        self.excluded = excluded
        self.jobs: list[CandidateJob] = []
        self.batches: list[tuple[int | None, ...]] = []

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        self.batches.append(tuple(job.trial_number for job in jobs))
        for job in jobs if workers == 1 else reversed(jobs):
            if self.kill_after is not None and len(self.jobs) >= self.kill_after:
                raise Killed("被砍")
            self.jobs.append(job)
            single = fixtures._single_timbre(
                job.scheme.scheme_id, tilt=0.0, residual=self.values.get(job.trial_number, 1.0),
                settings_fingerprint="different" if job.trial_number in self.different else "verification")
            timbre = evaluate_timbre_channels(
                job.scheme.channel_group, job.scheme.receiver_set.primary.receiver_id,
                {channel.role: single.model_copy(update={"provenance": single.provenance.model_copy(update={
                    "speaker_id": channel.speaker_id, "receiver_id": job.scheme.receiver_set.primary.receiver_id,
                })}) for channel in job.scheme.channel_group.channels},
                candidate_id=job.scheme.scheme_id, scene_fingerprint=single.scene_fingerprint,
                timbre_settings_fingerprint=single.settings_fingerprint)
            reverb = fixtures._reverberation(job.scheme.scheme_id)
            candidate = fixtures._candidate(reverb) if job.trial_number in self.missing else fixtures._candidate(timbre)
            if self.excluded:
                candidate = with_matching(candidate, breached=job.trial_number in self.excluded)
            identity = self.store.identity
            job.result_path.write_text(json.dumps({
                "candidate": candidate.model_dump(mode="json"), "physics_identity": identity.physics_identity,
                "program_fingerprint": identity.program_fingerprint,
                "purpose_settings": identity.purpose_settings.model_dump(mode="json"),
            }))
            yield ComputedCandidate(job, candidate, 0.25, identity)


def stopped_store(tmp_path: Path, *, batch: int = 3, workers: int = 1,
                  budget: int = 50, convergence: int = 50) -> tuple[SearchStore, Path]:
    store, registry = make_store(tmp_path, batch=batch, workers=workers, budget=12,
                                 refine={"budget": budget, "convergence_run": convergence})
    assert run(store, registry, SearchCompute(store)).state == "budget_exhausted"
    return store, registry


def refine(store: SearchStore, registry: Path, compute: RefineCompute) -> SearchStatus:
    from aosr.search.refine_run import refine_search

    return refine_search(store, compute=compute, probe=lambda: store.identity,
                         registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
