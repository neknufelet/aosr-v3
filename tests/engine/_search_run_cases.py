"""只替換計算的搜尋考卷工具；排名與取樣器都走產品入口。"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from aosr.reporting.evaluation import purpose_settings
from aosr.reporting.scheme import Scheme
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.search import constraints, layout
from aosr.search.ledger import Ledger, LedgerRow
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine import _ranking_fixtures as fixtures
from tests.engine._search_store_cases import reference_project, settings_document

if TYPE_CHECKING:
    from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus

RUN_DATE = date(2026, 10, 2)
ENGINE = "calc-v1:" + "a" * 64


def registry_copy(tmp_path: Path) -> Path:
    source = next((Path(__file__).resolve().parents[2] / "src" / "aosr" / "config" / "data").glob("quality_targets.*"))
    target = tmp_path / "registry"
    shutil.copyfile(source, target)
    return target


def make_store(tmp_path: Path, *, workers: int = 1, budget: int = 17,
               batch: int = 3, convergence: int = 100,
               layout_changes: dict[str, object] | None = None,
               scene_changes: dict[str, object] | None = None) -> tuple[SearchStore, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    project = reference_project(tmp_path)
    if scene_changes:
        project = Scheme.model_validate(project.model_dump() | {"scene": project.scene.model_dump() | scene_changes})
    registry = registry_copy(tmp_path)
    document = settings_document() | {
        "purpose": project.purpose, "max_workers": workers, "budget": budget,
        "batch_size": batch, "convergence_run": convergence,
    }
    original = SearchSettings.model_validate(document)
    changed = original.layout.model_dump() | (layout_changes or {})
    settings = SearchSettings.model_validate(document | {"layout": changed})
    identity = SearchIdentity("phys-v1:" + "b" * 64, ENGINE, purpose_settings(registry, project.purpose))
    store = SearchStore.create(tmp_path / "searches", project=project, settings=settings,
                               identity=identity, versions={"python": "test", "optuna": "test", "numpy": "test"})
    return store, registry


class Killed(BaseException):
    """模擬行程被砍；不屬於計算例外，不把搜尋宣判失敗。"""


class FakeCompute:
    """三個擺位量的平滑代價；種子固定而完成順序隨工作行程數變。"""

    def __init__(self, store: SearchStore, *, missing: frozenset[int | None] = frozenset(),
                 different: frozenset[int] = frozenset(), flat: bool = False,
                 fail_after: int | None = None, kill: bool = False,
                 persist_baseline: bool = False) -> None:
        self.store = store
        self.missing = missing
        self.different = different
        self.flat = flat
        self.fail_after = fail_after
        self.kill = kill
        self.persist_baseline = persist_baseline
        self.calls: list[int | None] = []
        self.batches: list[tuple[int, ...]] = []

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        from aosr.search.run import ComputedCandidate

        ordered = list(jobs)
        self.batches.append(tuple(job.trial_number for job in jobs if job.trial_number is not None))
        random.Random(self.store.settings.seed + workers).shuffle(ordered)
        for job in ordered:
            if job.trial_number is not None:
                if self.fail_after is not None and sum(number is not None for number in self.calls) >= self.fail_after:
                    if self.kill:
                        raise Killed("被砍")
                    raise RuntimeError("計算失敗")
                self._check_legal(job)
            self.calls.append(job.trial_number)
            left, right = job.scheme.speakers["left"], job.scheme.speakers["right"]
            primary = job.scheme.receiver_set.primary.position_m
            value = 1.0 if self.flat else 0.1 + 0.02 * (left.x**2 + (left.y - right.y)**2 + (primary[0] - left.x)**2)
            candidate_id = job.scheme.scheme_id
            single = fixtures._single_timbre(
                candidate_id, tilt=0.0, residual=value, target_deviation=value,
                settings_fingerprint="other" if job.trial_number in self.different else "fixed")
            timbre = evaluate_timbre_channels(
                job.scheme.channel_group, job.scheme.receiver_set.primary.receiver_id,
                {channel.role: single.model_copy(update={"provenance": single.provenance.model_copy(update={
                    "speaker_id": channel.speaker_id, "receiver_id": job.scheme.receiver_set.primary.receiver_id,
                })}) for channel in job.scheme.channel_group.channels},
                candidate_id=candidate_id, scene_fingerprint=single.scene_fingerprint,
                timbre_settings_fingerprint=single.settings_fingerprint)
            reverb = fixtures._reverberation(candidate_id)
            candidate = fixtures._candidate(reverb) if job.trial_number in self.missing else fixtures._candidate(timbre, reverb)
            # 接續考卷的原方案寫可讀候選包；其餘替身結果只寫空檔，沒落帳的一律重算。
            document = json.dumps({"candidate": candidate.model_dump(mode="json")})
            job.result_path.write_text(document if job.trial_number is None and self.persist_baseline else "", encoding="utf-8")
            yield ComputedCandidate(job, candidate, 0.25)

    def _check_legal(self, job: CandidateJob) -> None:
        left, right = job.scheme.speakers["left"], job.scheme.speakers["right"]
        primary = job.scheme.receiver_set.primary.position_m
        params = layout.LayoutParams(left.x, abs(left.y - right.y), primary[0] - left.x)
        placement = layout.place(self.store.project, self.store.settings.layout, params)
        assert not constraints.check(self.store.project, self.store.settings.layout, placement)


def run(store: SearchStore, registry: Path, compute: FakeCompute) -> SearchStatus:
    from aosr.search.run import start_search

    return start_search(store, compute=compute, probe=lambda: store.identity,
                        registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)


def rows(store: SearchStore) -> tuple[LedgerRow, ...]:
    return tuple(sorted(Ledger.read(store.ledger_path)[1], key=lambda row: row.trial_number))


def next_params(store: SearchStore) -> list[dict[str, str]]:
    from aosr.search.ledger import replay_history
    from aosr.search.sampler import SamplerAdapter

    header, recorded = Ledger.read(store.ledger_path)
    settings = store.settings
    sizes = {i: min(settings.batch_size, settings.budget - i * settings.batch_size)
             for i in range((settings.budget + settings.batch_size - 1) // settings.batch_size)}
    history, partial = replay_history(header, recorded, batch_sizes=sizes)
    assert not partial
    adapter = SamplerAdapter(layout.UNIT_SPACE, settings.sampler_settings())
    start = layout.standard_start(store.project, settings.layout)
    if start is not None:
        adapter.enqueue(layout.unit_from_params(start, settings.layout))
    adapter.replay(history)
    return [{key: value.hex() for key, value in item.params.items()} for item in adapter.ask_batch(settings.batch_size)]
