"""選取考卷沿用真搜尋與細算流程，只用合成資料補齊 v4 結果。"""

from collections.abc import Iterator, Sequence
from pathlib import Path

from aosr.reporting.result import RESULT_SCHEMA_VERSION, ResultOrigin, SchemeResult, Timings
from aosr.reporting.scheme import expected_pairs
from aosr.search.run import CandidateJob, ComputedCandidate
from aosr.search.store import SearchStore
from tests.engine._search_refine_cases import RefineCompute, refine, stopped_store
from tests.engine._search_run_cases import RUN_DATE
from tests.engine.test_search_report import _pair


class ResultRefineCompute(RefineCompute):
    """計算與排名仍走細算替身；零件只建模型，不執行物理解算。"""

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for computed in super().__call__(jobs, workers):
            job = computed.job
            scheme = job.scheme.model_copy(update={"source_model": "omnidirectional"})
            result = SchemeResult(
                schema_version=RESULT_SCHEMA_VERSION, scheme=scheme, engine_commit="test",
                program_fingerprint=self.store.identity.program_fingerprint,
                physics_identity=self.store.identity.physics_identity,
                purpose_settings=self.store.identity.purpose_settings,
                origin=ResultOrigin(kind="search_baseline" if job.trial_number is None else "search_candidate",
                                    search_id=self.store.search_id, trial_number=job.trial_number),
                scope="stage_two_subset", run_date=RUN_DATE, quality_targets_fingerprint="a" * 64,
                timings=Timings(solve_s=0.0, output_s=0.0, evaluate_s=0.0, total_s=0.25),
                pairs=tuple(_pair(scheme, speaker, receiver, role)
                            for speaker, receiver, role in expected_pairs(scheme)), candidate=computed.candidate,
            )
            job.result_path.write_text(result.model_dump_json(), encoding="utf-8")
            yield computed


def refined_store(tmp_path: Path) -> SearchStore:
    store, registry = stopped_store(tmp_path, budget=1)
    refine(store, registry, ResultRefineCompute(store, {}))
    return store
