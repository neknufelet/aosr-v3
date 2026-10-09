"""附件契約：入圍、整批、細算身分、逐點重用與搜尋檔案隔離。"""
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from aosr.reporting.result import SchemeResult
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.search.outer_status import OUTER_MESSAGES, OuterStatus, attachment_skip_reason
from aosr.search.placement_stability import SHIFT_NAMES
from tests.engine._crossover_cases import protected
from tests.engine._search_run_cases import RUN_DATE
from tests.engine._stability_attach_cases import ShiftCompute, attach, ready, refiner


@pytest.mark.parametrize("conclusion", [None, *OUTER_MESSAGES])
def test_skip_uses_outer_conclusion(tmp_path: Path, conclusion: str | None) -> None:
    store, registry, status = ready(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate(dict(conclusion=conclusion))})
    compute = ShiftCompute(store)
    summary = attach(store, registry, status, compute)
    reason = attachment_skip_reason(status.outer.conclusion)
    if reason:
        assert summary.state == "skipped" and summary.reason_text == reason and not compute.batches
    else:
        assert summary.state == "done" and compute.batches


@pytest.mark.parametrize("name", ["physics_identity", "program_fingerprint", "purpose_settings"])
def test_identity_mismatch_never_dispatches(tmp_path: Path, name: str) -> None:
    from aosr.search.placement_stability_attach import attach_stability
    store, registry, status = ready(tmp_path)
    wrong = (replace(store.identity, purpose_settings=store.identity.purpose_settings.model_copy(update={"purpose": "different"}))
             if name == "purpose_settings" else replace(store.identity, physics_identity="changed")
             if name == "physics_identity" else replace(store.identity, program_fingerprint="changed"))
    def forbidden(root: Path) -> NoReturn:
        raise AssertionError("身分不同仍建立計算")
    summary = attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
        probe=lambda: wrong, compute_factory=forbidden)
    assert summary.state == "skipped" and "不補：身分跟這場搜尋不同" in summary.reason_text


def test_all_finalists_one_batch_original_included_and_pinned_by_refinement(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    before = protected(store)
    compute = ShiftCompute(store)
    summary = attach(store, registry, status, compute)
    assert summary.completed and summary.state == "done"
    assert [f.trial_number for f in summary.selection.finalists] == [7, 9, None]
    assert compute.batches == [tuple(compute.jobs)]
    assert {(j.trial_number, j.shift_name) for j in compute.jobs} == {(n, s) for n in (7, 9, None) for s in SHIFT_NAMES}
    assert all(j.scheme.scene.low_frequency_axis == LowFrequencyAxis.VERIFICATION for j in compute.jobs)
    assert protected(store) == before
    judge = refiner(store, registry, status)
    base = SchemeResult.model_validate_json(store.refine_result_path(None).read_bytes())
    judge.pin(base.candidate, base.scheme)
    for point in summary.points:
        job = next(j for j in compute.jobs if (j.trial_number, j.shift_name) == (point.trial_number, point.shift_name))
        written = compute.written[job.result_path]
        expected = judge.screen(written.candidate, job, written.timings.total_s)
        assert point.outcome == expected.outcome and point.total_cost == expected.total_cost
    assert summary.arithmetic is not None and summary.boundaries
    assert summary.computed_points == summary.total_points == len(compute.jobs)


def test_shift_own_identity_cannot_replace_refinement_pin(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    compute = ShiftCompute(store, different=True)
    summary = attach(store, registry, status, compute)
    assert {p.outcome for p in summary.points} == {"not_comparable"}
    assert all(p.total_cost is None for p in summary.points)


@pytest.mark.parametrize("stop", [False, True])
def test_failure_keeps_returned_points_and_marks_batch_aborted(tmp_path: Path, stop: bool) -> None:
    from aosr.search.placement_stability_record import read_summary
    store, registry, status = ready(tmp_path)
    compute = ShiftCompute(store, fail_after=1, stop=stop)
    with pytest.raises(KeyboardInterrupt if stop else RuntimeError):
        attach(store, registry, status, compute)
    summary = read_summary(store.path)
    assert summary is not None and not summary.completed
    assert summary.state == ("stopped" if stop else "failed")
    done = {(p.trial_number, p.shift_name) for p in summary.points if p.outcome is not None}
    assert done == {(j.trial_number, j.shift_name) for j in compute.jobs}
    assert all(p.reason_text == "同批被中止" for p in summary.points if p.outcome is None)
    assert summary.computed_points == len(done)
    assert any(p.result_file is not None and p.outcome is None for p in summary.points)
    resumed = ShiftCompute(store)
    second = attach(store, registry, status, resumed)
    assert second.completed
    assert {(j.trial_number, j.shift_name) for j in resumed.jobs}.isdisjoint(done)
    assert len(resumed.jobs) + len(compute.jobs) == second.total_points


def test_complete_reuses_whole_summary_and_broken_summary_is_local(tmp_path: Path) -> None:
    from aosr.search.placement_stability_record import summary_path
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    content = summary_path(store.path).read_bytes()
    compute = ShiftCompute(store)
    assert attach(store, registry, status, compute) == first and not compute.batches
    assert summary_path(store.path).read_bytes() == content
    summary_path(store.path).write_text("{broken")
    compute = ShiftCompute(store)
    assert attach(store, registry, status, compute).completed and compute.batches


def test_unplaceable_is_not_dispatched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import placement_stability_attach as module
    from aosr.search.placement_stability_geometry import Legality, Shift, check_shift
    from aosr.reporting.scheme import Scheme
    from aosr.config.capabilities import CapabilityTable
    from aosr.config.directivity_defaults import DirectivityDefaults
    from aosr.search.constraints import Reason, Violation
    store, registry, status = ready(tmp_path)
    def check(project: Scheme, shift: Shift, *, contact_rel: float, capabilities: CapabilityTable,
              directivity: DirectivityDefaults) -> Legality:
        return Legality("unplaceable", violations=(Violation(Reason.SEAT_OUTSIDE_ROOM, 0.01),)) if shift.name == "ear_up" else check_shift(
            project, shift, contact_rel=contact_rel, capabilities=capabilities, directivity=directivity)
    monkeypatch.setattr(module, "check_shift", check)
    compute = ShiftCompute(store)
    summary = attach(store, registry, status, compute)
    assert all(j.shift_name != "ear_up" for j in compute.jobs)
    assert summary.total_points == summary.computed_points == len(compute.jobs)
    for point in summary.points:
        if point.shift_name == "ear_up":
            assert point.outcome == "unplaceable" and point.violations[0].amount_m == 0.01
            assert point.total_cost is None and point.result_file is None


@pytest.mark.parametrize("primary", [True, False])
def test_furniture_row_disappearance_is_stored_with_primary_marker(tmp_path: Path, primary: bool) -> None:
    from aosr.physics.report_path_output import PathTableSection
    from aosr.physics.report_source import SourceModelKind
    from aosr.reporting.scheme import Scheme
    from aosr.search.refine import RefineLedger
    from tests.engine.test_search_placement_stability_events import path
    from tests.engine._stability_attach_cases import pairs
    store, registry, status = ready(tmp_path)
    cloud = dict(furniture_id="cloud", kind="ceiling_cloud", material="wood", width_m=1.0, depth_m=1.0,
        height_m=0.1, placement=dict(bottom_center_m=(2.0, 2.0, 2.5), yaw_deg=0.0))
    table = PathTableSection.model_validate(dict(reflection_order_k=1, frequencies_hz=(100.0,),
        scattering_coefficient=(0.0,), source_model_kind=SourceModelKind.OMNIDIRECTIONAL,
        rows=(path("cloud", "bottom", point=(2.0, 2.0, 2.5)),), furniture_ids=("cloud",),
        furniture_model="single_bounce_finite_size_v1", blocked_wall_paths=(),
        furniture_materials=(dict(furniture_id="cloud", material="wood", unknown_bands_hz=()),)))
    for row in RefineLedger.read(store.refine_ledger_path)[1]:
        result = SchemeResult.model_validate_json(store.refine_result_path(row.trial_number).read_bytes())
        scheme = Scheme.model_validate(result.scheme.model_dump() | {"furniture": (cloud,)})
        receiver = "main" if primary else next(p.receiver_id for p in pairs(scheme) if p.receiver_id != "main")
        base_pairs = tuple(p.model_copy(update={"report": p.report.model_copy(update={"path_table": table})})
                           if p.receiver_id == receiver else p for p in pairs(scheme))
        result = result.model_copy(update={"scheme": scheme, "pairs": base_pairs})
        store.refine_result_path(row.trial_number).write_text(result.model_dump_json())
        store.scheme_path_for(store.refine_result_path(row.trial_number)).write_text(scheme.model_dump_json())
    summary = attach(store, registry, status, ShiftCompute(store))
    assert all(p.model_discontinuity for p in summary.points)
    for point in summary.points:
        assert {(e.speaker_id, e.receiver_id, e.change, e.is_primary) for e in point.furniture_events} == {
            ("left", receiver, "disappeared", primary), ("right", receiver, "disappeared", primary)}
    if primary:
        assert all(d.edge_distance_m == 0.5 for b in summary.boundaries for d in b.distances)
    else:
        assert all(not b.distances for b in summary.boundaries)
    assert summary.arithmetic is not None
    assert all(r.continuous_best == r.continuous_worst == r.finalist.original_cost for r in summary.arithmetic.finalists)
