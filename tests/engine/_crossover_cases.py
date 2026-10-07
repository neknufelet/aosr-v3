"""交接考卷的合成小結果；能量、座標與正式權重由題目自行建造。"""
from __future__ import annotations

from pathlib import Path

from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point
from aosr.physics.crossover import crossover_weights
from aosr.physics.report_io import PointRow
from aosr.reporting.result import RESULT_SCHEMA_VERSION, ResultOrigin, SchemeResult, Timings
from aosr.reporting.scheme import expected_pairs
from aosr.scoring.contract import CONTRACT_SCHEMA_VERSION, CandidateEvaluation
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.refine_run import header_for
from aosr.search.ledger import LedgerRow, create_for
from aosr.search.run import RefineStatus, SearchStatus
from aosr.search.scoring import screening_outcome
from aosr.search.sampler import Scored
from aosr.search.store import SearchStore, candidate_name, refine_scheme_id, refine_result_name
from tests.engine import _ranking_fixtures as fixtures
from tests.engine._search_run_cases import RUN_DATE, make_store
from tests.engine.test_search_report import _pair


def evaluate(result: SchemeResult, **kwargs: object) -> CandidateEvaluation:
    """替身只量題目能量，排名與比較身分仍走真正入口。"""
    points = result.pairs[0].report.points
    assert points
    single = fixtures._single_timbre(result.scheme.scheme_id, tilt=0, residual=points[0].total_energy)
    group = result.scheme.channel_group
    primary = result.scheme.receiver_set.primary.receiver_id
    timbre = evaluate_timbre_channels(group, primary, {
        channel.role: single.model_copy(update={"provenance": single.provenance.model_copy(update={
            "speaker_id": channel.speaker_id, "receiver_id": primary})}) for channel in group.channels
    }, candidate_id=result.scheme.scheme_id, scene_fingerprint=single.scene_fingerprint,
        timbre_settings_fingerprint=single.settings_fingerprint)
    return fixtures._candidate(timbre)


def small_result(store: SearchStore, number: int | None, f_s: float,
                 energies: tuple[float, float], frequencies: tuple[float, ...]) -> SchemeResult:
    scheme = store.project.model_copy(update={"scheme_id": refine_scheme_id(store.search_id, number),
        "source_model": "omnidirectional"})
    if number is not None:
        offset = number / 1000
        scheme = scheme.model_copy(update={"speakers": {
            key: Point(p.x + offset, p.y - offset, p.z + offset) for key, p in scheme.speakers.items()},
            "receiver_set": scheme.receiver_set.model_copy(update={"points": tuple(
                p.model_copy(update={"position_m": tuple(v + offset for v in p.position_m)})
                for p in scheme.receiver_set.points)})})
    weights = crossover_weights(frequencies, f_s)
    points = tuple(PointRow(frequency_hz=f, fem_energy=energies[0] if f <= FEM_GEOMETRIC_CROSSOVER_CAP_HZ else None,
        direct_energy=energies[1], reflected_energy=0, interference_energy=0, late_energy=0, scattering=0,
        geometric_energy=energies[1], w_fem=wf, w_geo=wg, total_energy=wf * energies[0] + wg * energies[1])
        for f, wf, wg in zip(frequencies, weights.w_fem, weights.w_geo, strict=True))
    pairs = tuple(_pair(scheme, speaker, receiver, role) for speaker, receiver, role in expected_pairs(scheme))
    pairs = tuple(p.model_copy(update={"report": p.report.model_copy(update={"points": points,
        "top": p.report.top.model_copy(update={"f_s_hz": f_s})})}) for p in pairs)
    result = SchemeResult(schema_version=RESULT_SCHEMA_VERSION, scheme=scheme, engine_commit="test",
        program_fingerprint=store.identity.program_fingerprint, physics_identity=store.identity.physics_identity,
        purpose_settings=store.identity.purpose_settings, origin=ResultOrigin(kind="search_baseline" if number is None else "search_candidate",
            search_id=store.search_id, trial_number=number), scope="stage_two_subset", run_date=RUN_DATE,
        quality_targets_fingerprint="a" * 64, timings=Timings(solve_s=0, output_s=0, evaluate_s=0, total_s=0),
        pairs=pairs, candidate=CandidateEvaluation(schema_version=CONTRACT_SCHEMA_VERSION,
            candidate_id=scheme.scheme_id, scene_fingerprint="a" * 64, evaluations=()))
    return result.model_copy(update={"candidate": evaluate(result)})


def prepared(folder: Path, *, f_s: float = 200, swap: bool = True,
             frequencies: tuple[float, ...] = (220, 1000)) -> tuple[SearchStore, Path, SearchStatus]:
    store, registry = make_store(folder)
    store.ensure_refine_dir()
    book = RefineLedger.create(store.refine_ledger_path, header_for(store))
    search_book = create_for(store)
    results = [small_result(store, n, f_s, energies, frequencies) for n, energies in (
        (None, (5, 5)), (7, (1, 10 if f_s >= 300 and swap else 2)), (9, (2, 0.1) if swap else (3, 4)))]
    for result in results:
        number = result.origin.trial_number
        outcome, _ = screening_outcome(result.candidate, result.scheme, registry=load_quality_targets(registry),
            run_date=RUN_DATE, engine_version=store.identity.program_fingerprint, pinned=None)
        assert isinstance(outcome, Scored)
        store.refine_result_path(number).write_text(result.model_dump_json())
        book.append(RefineRow(round=1, trial_number=number, result_file=refine_result_name(number),
            outcome="scored", total_cost=outcome.value, seconds=0))
        source = store.baseline_path if number is None else store.candidate_path(number)
        source.write_text(result.model_dump_json())
        store.scheme_path_for(source).write_text(result.scheme.model_dump_json())
        store.scheme_path_for(store.refine_result_path(number)).write_text(result.scheme.model_dump_json())
        if number is not None:
            params = {"front_distance": 1.0, "spacing": 2.0, "listening_distance": 3.0}
            search_book.append(LedgerRow(batch_index=0, trial_number=number,
                unit_params_hex={key: (0.5).hex() for key in params}, params_m=params, outcome="scored",
                score=outcome.value, reason=None, violation_m=None, seconds=0, result_file=candidate_name(number)))
    from aosr.search.outer import conclude
    status = SearchStatus(state="budget_exhausted", refine=RefineStatus(state="stopped", stop_reason="refine_budget", refined=len(results)))
    return store, registry, conclude(store, status, "complete")


def protected(store: SearchStore) -> dict[str, bytes]:
    return {str(p.relative_to(store.path)): p.read_bytes() for p in store.path.rglob("*")
            if p.is_file() and "crossover-sensitivity" not in p.parts and "modal-diagnosis" not in p.parts}
