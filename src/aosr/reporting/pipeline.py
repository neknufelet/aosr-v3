"""把一份具體方案接到既有批次物理與評分入口。"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.scoring.contract import CONTRACT_SCHEMA_VERSION, CandidateEvaluation
from aosr.reporting.evaluation import (
    evaluate_parts, purpose_settings, quality_targets_fingerprint, read_registry_settings,
)
from aosr.reporting.physics_stage import solve_checked_physics
from aosr.reporting.result import RESULT_SCHEMA_VERSION, PairResult, ResultOrigin, SchemeResult, Timings
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import checked_inputs


def run_scheme(
    scheme: Scheme | object, *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
    quality_targets_path: Path, engine_commit: str, program_fingerprint: str, physics_identity: str,
    run_date: date, origin: ResultOrigin = ResultOrigin(kind="run"),
) -> SchemeResult:
    """驗每一對、批次求解一次、組零件並評估一份候選。"""
    start = time.perf_counter()
    # 先驗方案與每一對、再讀登記簿、最後求解：錯誤先後跟拆分前一樣。
    scheme, documents = checked_inputs(scheme, capabilities=capabilities,
                                       directivity=directivity)
    registry = read_registry_settings(quality_targets_path, scheme.purpose)
    physics = solve_checked_physics(scheme, documents, capabilities=capabilities)
    before_evaluate = time.perf_counter()
    pairs = tuple(PairResult.model_validate(pair.model_dump()) for pair in physics.pairs)
    temporary = SchemeResult(
        schema_version=RESULT_SCHEMA_VERSION,
        scheme=scheme, engine_commit=engine_commit,
        program_fingerprint=program_fingerprint, physics_identity=physics_identity,
        purpose_settings=purpose_settings(quality_targets_path, scheme.purpose),
        origin=origin, scope="stage_two_subset", run_date=run_date,
        quality_targets_fingerprint=quality_targets_fingerprint(quality_targets_path),
        timings=Timings(solve_s=physics.solve_s, output_s=physics.output_s,
                        evaluate_s=0.0, total_s=0.0),
        pairs=pairs,
        candidate=CandidateEvaluation(schema_version=CONTRACT_SCHEMA_VERSION,
                                      candidate_id=scheme.scheme_id,
                                      scene_fingerprint=physics.scene_fingerprint,
                                      evaluations=()),
    )
    candidate = evaluate_parts(temporary, quality_targets_path, registry,
                               capabilities=capabilities, directivity=directivity)
    end = time.perf_counter()
    return SchemeResult.model_validate(temporary.model_copy(update={
        "candidate": candidate,
        "timings": Timings(solve_s=physics.solve_s, output_s=physics.output_s,
                           evaluate_s=end - before_evaluate,
                           total_s=end - start),
    }))
