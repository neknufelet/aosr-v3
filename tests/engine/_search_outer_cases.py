"""外圈共用可控計算，保留真搜尋、細算與回饋帳本。"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.search import cli
from aosr.search.run import CandidateJob, ComputedCandidate, Compute
from aosr.search.store import SearchStore
from tests.engine._search_refine_cases import RefineCompute, SearchCompute
from tests.engine._search_run_cases import Killed
from tests.engine._stability_attach_cases import ShiftCompute
from tests.engine._modal_cases import runner
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState


class OuterCompute:
    def __init__(self, store: SearchStore, *, kill_after: int | None = None) -> None:
        self.search = SearchCompute(store, flat=True, persist_baseline=True)
        self.refine = RefineCompute(store, {None: 4.0, 1: 0.1})
        self.kill_after = kill_after
        self.completed = 0

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        compute = self.refine if jobs[0].result_path.parent == self.refine.store.refine_dir else self.search
        for result in compute(jobs, workers):
            if self.completed == self.kill_after:
                raise Killed("被砍")
            self.completed += 1
            yield result


class CompleteOuterCompute(OuterCompute):
    """保留外圈真流程，細算替身也寫齊供附件讀取的方案、出處與報表。"""

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        from aosr.reporting.result import ResultOrigin
        from tests.engine._crossover_cases import small_result
        from tests.engine._stability_attach_cases import pairs
        for computed in super().__call__(jobs, workers):
            job = computed.job
            if job.result_path.parent == self.refine.store.refine_dir:
                store = self.refine.store
                template = small_result(store, job.trial_number, 200, (1, 2), (220, 1000))
                result = template.model_copy(update={"scheme": job.scheme, "candidate": computed.candidate,
                    "pairs": pairs(job.scheme), "origin": ResultOrigin(kind="search_baseline" if job.trial_number is None else "search_candidate",
                        search_id=store.search_id, trial_number=job.trial_number)})
                job.result_path.write_text(result.model_dump_json())
                SearchStore.scheme_path_for(job.result_path).write_text(job.scheme.model_dump_json())
            yield computed


def invoke(store: SearchStore, registry: Path, monkeypatch: pytest.MonkeyPatch,
           compute: Compute | None = None) -> int:
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: store.identity)
    return cli.main(["auto", str(store.path), "--engine-commit", "requested", "--capabilities", str(registry),
                     "--modal-cache-dir", str(store.path.parent / "modal-cache")],
                    compute_factory=lambda opened, capabilities, commit: compute or OuterCompute(opened),
                    stability_compute_factory=lambda root: ShiftCompute(store),
                    modal_runner=runner(store.path.parent / "modal-runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))
