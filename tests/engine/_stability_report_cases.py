"""報告題目由第二步寫入端產生；替身交回可區分翻轉、缺分與家具事件的結果。"""
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.physics.report_path_output import PathTableSection
from aosr.reporting.result import ResultOrigin, SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.search import crossover_record, crossover_sensitivity, placement_stability_attach as writer
from aosr.search.placement_stability_record import StabilitySummary, read_summary
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.sampler import Excluded, RankingZone, Scored
from aosr.search.store import SearchStore
from tests.engine._crossover_cases import evaluate
from tests.engine._stability_attach_cases import ShiftCompute, attach, pairs as empty_pairs, ready
from tests.engine.test_search_placement_stability_events import path


def add_backfill_finalist(store: SearchStore, status: SearchStatus) -> int:
    """照細算帳的同分次序排第四，再由交接寫入端把它列作接法第一名。"""
    from aosr.scoring.contract import CandidateEvaluation
    from aosr.search.labels import STABILITY_CROSSOVERS
    from aosr.search.refine import RefineLedger, RefineRow
    from aosr.search.store import refine_scheme_id, refine_result_name
    number = 11
    result = SchemeResult.model_validate_json(store.refine_result_path(None).read_bytes())
    identifier = refine_scheme_id(store.search_id, number)
    scheme = result.scheme.model_copy(update={"scheme_id": identifier})
    candidate = CandidateEvaluation.model_validate_json(result.candidate.model_dump_json().replace(result.scheme.scheme_id, identifier))
    result = result.model_copy(update={"scheme": scheme, "candidate": candidate, "pairs": empty_pairs(scheme),
        "origin": ResultOrigin(kind="search_candidate", search_id=store.search_id, trial_number=number)})
    store.refine_result_path(number).write_text(result.model_dump_json())
    store.scheme_path_for(store.refine_result_path(number)).write_text(scheme.model_dump_json())
    baseline = next(row for row in RefineLedger.read(store.refine_ledger_path)[1] if row.trial_number is None)
    RefineLedger(store.refine_ledger_path).append(RefineRow(round=1, trial_number=number,
        result_file=refine_result_name(number), outcome="scored", total_cost=baseline.total_cost, seconds=0))
    fresh = crossover_record.fresh_summary(store, status).model_copy(update={"state": "done", "completed": True,
        "variants": tuple(crossover_record.VariantRecord(key=key, label=label, basis="題目",
            ranking=(crossover_record.CostRow(trial_number=number, total_cost=0),))
            for key, label in STABILITY_CROSSOVERS.items())})
    crossover_record.write_summary(store.path, fresh)
    return number


def furniture_table() -> PathTableSection:
    return PathTableSection.model_validate(dict(reflection_order_k=1, frequencies_hz=(100.0,),
        scattering_coefficient=(0.0,), source_model_kind="omnidirectional",
        rows=(path("cloud", "bottom", point=(2.0, 2.0, 2.5)),), furniture_ids=("cloud",),
        furniture_model="single_bounce_finite_size_v1", blocked_wall_paths=(),
        furniture_materials=(dict(furniture_id="cloud", material="wood", unknown_bands_hz=()),)))


class ReportCompute(ShiftCompute):
    def __init__(self, store: SearchStore, *, fail_after: int | None = None, stop: bool = False,
                 missing_on_flip: bool = False, flip: bool = True) -> None:
        super().__init__(store, fail_after=fail_after, stop=stop)
        self.missing_on_flip = missing_on_flip
        self.flip = flip
        self.scores: dict[str, Scored | Excluded] = {}
        self.snapshots: list[StabilitySummary] = []

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for computed in super().__call__(jobs, workers):
            snapshot = read_summary(self.store.path)
            assert snapshot is not None
            self.snapshots.append(snapshot)
            job = computed.job
            result = self.written[job.result_path]
            remove = job.trial_number == 7 and job.shift_name == "speakers_forward"
            pairs = tuple(p.model_copy(update={"report": p.report.model_copy(update={"path_table": furniture_table()})})
                          if (p.speaker_id == "left" and p.receiver_id == "main" and not remove)
                          or (job.trial_number == 9 and job.shift_name == "seat_left" and p.speaker_id == "right" and p.receiver_id == "up") else p
                          for p in result.pairs)
            result = result.model_copy(update={"pairs": pairs})
            job.result_path.write_text(result.model_dump_json())
            self.written[job.result_path] = result
            value = 10.0 if remove and self.flip else 0.1 if job.trial_number == 7 else 0.2 if job.trial_number == 9 else 0.3
            excluded = {"ear_down": RankingZone.ELIMINATED, "seat_right": RankingZone.UNASSESSED,
                        "speakers_backward": RankingZone.INCOMPARABLE}
            if self.missing_on_flip:
                excluded["speakers_forward"] = RankingZone.INCOMPARABLE
            self.scores[result.candidate.candidate_id] = (Excluded(excluded[job.shift_name])
                if job.trial_number is None and job.shift_name in excluded else Scored(value))
            yield computed


def add_baseline_furniture(store: SearchStore) -> None:
    from aosr.search.refine import RefineLedger
    cloud = dict(furniture_id="cloud", kind="ceiling_cloud", material="wood", width_m=1.0, depth_m=1.0,
        height_m=0.1, placement=dict(bottom_center_m=(2.0, 2.0, 2.5), yaw_deg=0.0))
    for row in RefineLedger.read(store.refine_ledger_path)[1]:
        result = SchemeResult.model_validate_json(store.refine_result_path(row.trial_number).read_bytes())
        scheme = Scheme.model_validate(result.scheme.model_dump() | {"furniture": (cloud,)})
        source = store.baseline_path if row.trial_number is None else store.candidate_path(row.trial_number)
        original = SchemeResult.model_validate_json(source.read_bytes())
        rebuilt = tuple(p.model_copy(update={"report": p.report.model_copy(update={"points": old.report.points})})
                        for p, old in zip(empty_pairs(scheme), original.pairs, strict=True))
        pairs = tuple(p.model_copy(update={"report": p.report.model_copy(update={"path_table": furniture_table()})})
                      if p.speaker_id == "left" and p.receiver_id == "main" else p for p in rebuilt)
        result = result.model_copy(update={"scheme": scheme, "pairs": pairs})
        store.refine_result_path(row.trial_number).write_text(result.model_dump_json())
        store.scheme_path_for(store.refine_result_path(row.trial_number)).write_text(scheme.model_dump_json())


def report_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_after: int | None = None,
                stop: bool = False, missing_on_flip: bool = False,
                flip: bool = True) -> tuple[SearchStore, Path, SearchStatus, StabilitySummary, ReportCompute]:
    store, registry, status = ready(tmp_path)
    add_baseline_furniture(store)
    monkeypatch.setattr(crossover_sensitivity, "reevaluate", evaluate)
    crossover_sensitivity.attach_crossover(store, status=status, quality_targets_path=registry)
    compute = ReportCompute(store, fail_after=fail_after, stop=stop, missing_on_flip=missing_on_flip, flip=flip)
    install_legality(store, monkeypatch)
    from aosr.search.scoring import screening_outcome
    original = screening_outcome
    from aosr.scoring.contract import CandidateEvaluation
    from aosr.scoring.ranking import ComparisonIdentity
    from aosr.config.quality_targets import QualityTargets
    from datetime import date
    def score(candidate: CandidateEvaluation, scheme: Scheme, *, registry: QualityTargets, run_date: date,
              engine_version: str, pinned: tuple[ComparisonIdentity, ...] | None) -> tuple[Scored | Excluded, tuple[ComparisonIdentity, ...] | None]:
        if candidate.candidate_id in compute.scores:
            return compute.scores[candidate.candidate_id], pinned
        outcome, identity = original(candidate, scheme, registry=registry, run_date=run_date, engine_version=engine_version, pinned=pinned)
        assert isinstance(outcome, (Scored, Excluded))
        return outcome, identity
    monkeypatch.setattr(writer, "screening_outcome", score)
    if fail_after is None:
        summary = attach(store, registry, status, compute)
    else:
        with pytest.raises(KeyboardInterrupt if stop else RuntimeError):
            attach(store, registry, status, compute)
        saved = read_summary(store.path)
        assert saved is not None
        summary = saved
    return store, registry, status, summary, compute


def install_legality(store: SearchStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from aosr.config.capabilities import CapabilityTable
    from aosr.config.directivity_defaults import DirectivityDefaults
    from aosr.reporting.validation import SchemeProblem
    from aosr.search.constraints import Reason, Violation
    from aosr.search.placement_stability_geometry import Legality, Shift, check_shift
    from aosr.search.store import refine_scheme_id
    original = check_shift
    def legal(project: Scheme, shift: Shift, *, contact_rel: float, capabilities: CapabilityTable,
              directivity: DirectivityDefaults) -> Legality:
        found = original(project, shift, contact_rel=contact_rel, capabilities=capabilities, directivity=directivity)
        if shift.template.scheme_id == refine_scheme_id(store.search_id, None):
            if shift.name == "ear_up":
                return Legality("unplaceable", violations=(Violation(Reason.SEAT_OUTSIDE_ROOM, 0.01),))
            if shift.name == "seat_forward":
                return Legality("placement_requirement_failed", violations=(Violation(Reason.DIRECT_PATH_BLOCKED, 0.005),))
            if shift.name == "acoustic_center_down":
                return Legality("unplaceable", problems=(SchemeProblem(path="speakers.left.z", message="高度不合"),))
        if shift.name == "speakers_forward":
            return replace(found, out_of_spec=(Violation(Reason.BASE_ANGLE_OUT_OF_RANGE, 0.002),))
        if shift.name == "seat_left":
            quantity = "spacing" if shift.template.scheme_id == refine_scheme_id(store.search_id, 9) else "front_distance"
            return replace(found, outside_search=(quantity,))
        if shift.name == "seat_right":
            return replace(found, outside_search=(), search_range_not_checked=True)
        return found
    monkeypatch.setattr(writer, "check_shift", legal)
