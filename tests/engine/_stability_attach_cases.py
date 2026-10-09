"""移位附件替身：保存完整合成文件，評分仍走細算端，不做物理。"""
from collections.abc import Iterator, Sequence
from pathlib import Path

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.quality_targets import load_quality_targets
from aosr.physics.report_path_output import PathTableSection
from aosr.physics.report_source import SourceModelKind
from aosr.reporting.result import PairResult, ResultOrigin, SchemeResult
from aosr.reporting.scheme import Scheme, expected_pairs
from aosr.scoring.contract import CandidateEvaluation
from aosr.search.refine import RefineLedger
from aosr.search.refine_run import _Refiner
from aosr.search.placement_stability_record import StabilitySummary
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.store import SearchStore
from aosr.search.timings import WallClock
from tests.engine._crossover_cases import prepared
from tests.engine._search_run_cases import RUN_DATE
from tests.engine.test_search_report import _pair


def pairs(scheme: Scheme) -> tuple[PairResult, ...]:
    table = PathTableSection(reflection_order_k=1, frequencies_hz=(100.0,), scattering_coefficient=(0.0,),
                            source_model_kind=SourceModelKind.OMNIDIRECTIONAL, rows=())
    return tuple(pair.model_copy(update={"report": pair.report.model_copy(update={"path_table": table,
        "top": pair.report.top.model_copy(update={"low_frequency_axis": LowFrequencyAxis.VERIFICATION})})})
        for speaker, receiver, role in expected_pairs(scheme) for pair in (_pair(scheme, speaker, receiver, role),))


def ready(folder: Path) -> tuple[SearchStore, Path, SearchStatus]:
    store, registry, status = prepared(folder)
    for _, row in enumerate(RefineLedger.read(store.refine_ledger_path)[1]):
        result = SchemeResult.model_validate_json(store.refine_result_path(row.trial_number).read_bytes())
        scheme = result.scheme.model_copy(update={"scene": result.scheme.scene.model_copy(update={
            "low_frequency_axis": LowFrequencyAxis.VERIFICATION})})
        result = result.model_copy(update={"scheme": scheme, "pairs": pairs(scheme)})
        store.refine_result_path(row.trial_number).write_text(result.model_dump_json())
        store.scheme_path_for(store.refine_result_path(row.trial_number)).write_text(scheme.model_dump_json())
    return store, registry, status


class ShiftCompute:
    def __init__(self, store: SearchStore, *, fail_after: int | None = None, stop: bool = False,
                 different: bool = False) -> None:
        self.store, self.fail_after, self.stop, self.different = store, fail_after, stop, different
        self.jobs: list[CandidateJob] = []
        self.batches: list[tuple[CandidateJob, ...]] = []
        self.written: dict[Path, SchemeResult] = {}

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        self.batches.append(tuple(jobs))
        for job in jobs:
            if self.fail_after is not None and len(self.jobs) >= self.fail_after:
                raise KeyboardInterrupt if self.stop else RuntimeError("某點計算失敗原文")
            self.jobs.append(job)
            base = SchemeResult.model_validate_json(self.store.refine_result_path(job.trial_number).read_bytes())
            document = base.candidate.model_dump_json().replace(base.scheme.scheme_id, job.scheme.scheme_id)
            if self.different:
                document = document.replace('"settings_fingerprint":"', '"settings_fingerprint":"other-')
            candidate = CandidateEvaluation.model_validate_json(document)
            origin = ResultOrigin.model_validate(dict(kind="search_placement_shift", search_id=self.store.search_id,
                trial_number=job.trial_number, shift_name=job.shift_name))
            result = base.model_copy(update={"scheme": job.scheme, "pairs": pairs(job.scheme),
                                            "candidate": candidate, "origin": origin})
            job.result_path.parent.mkdir(parents=True, exist_ok=True)
            job.result_path.write_text(result.model_dump_json())
            self.store.scheme_path_for(job.result_path).write_text(job.scheme.model_dump_json())
            self.written[job.result_path] = result
            yield ComputedCandidate(job, candidate, 0.25, self.store.identity)


def refiner(store: SearchStore, registry: Path, status: SearchStatus) -> _Refiner:
    return _Refiner(store, ShiftCompute(store), lambda: store.identity, load_quality_targets(registry),
                    RUN_DATE, store.identity.program_fingerprint, status, WallClock())


def attach(store: SearchStore, registry: Path, status: SearchStatus, compute: ShiftCompute) -> StabilitySummary:
    from aosr.search.placement_stability_attach import attach_stability
    return attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
                            probe=lambda: store.identity, compute_factory=lambda root: compute)
