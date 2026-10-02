"""審查補題的聲道底線樣本；淘汰由正式排名層判定。"""

from aosr.scoring.contract import CandidateEvaluation, ChannelMatchingPayload
from tests.engine.test_channel_matching_cost import _safe_measured, _with_metric_summary


def with_matching(candidate: CandidateEvaluation, *, breached: bool) -> CandidateEvaluation:
    evaluation = _safe_measured(direct_time_cost_enabled=False)
    if breached:
        evaluation = _with_metric_summary(evaluation, metric="broadband_level_difference", mean=0.0, worst=100.0)
    payload = evaluation.payload
    assert isinstance(payload, ChannelMatchingPayload)
    evaluation = evaluation.model_copy(update={
        "candidate_id": candidate.candidate_id,
        "payload": payload.model_copy(update={"candidate_id": candidate.candidate_id}),
    })
    return CandidateEvaluation.model_validate(candidate.model_dump() | {
        "evaluations": (*candidate.evaluations, evaluation),
    })
