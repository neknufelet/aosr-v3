"""兩層真接線：單對物理、僅吞方案施工關的管線存讀與混合比較。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.furniture import FaceDirection
from aosr.physics import three_lane_report
from aosr.physics.report_io import solver_inputs
from aosr.physics.report_output import output_from_report
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.third_octave_decay import build_third_octave_decay
from aosr.reporting import validation
from aosr.reporting.compare import compare_results, identity_difference_groups
from aosr.reporting.evaluation import load_result, ResultStanding
from aosr.reporting.result import SchemeResult, save_result
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import Flag, QualityCategory
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from tests.engine import _furniture_cases as schemes, _furniture_energy_cases as case
from tests.engine import _scoring_source_model_control as control
from tests.engine._directivity import DIRECTIVITY
from tests.engine._report_cache import ControlPairPhysics
from tests.engine._scheme_cache import shared_json
from tests.engine.test_furniture_reflection_window_wiring import inputs
from tests.engine.test_gui_compare_view import _view
from tests.engine.test_scheme_pipeline import _run_control


def _single(furnished: bool) -> str:
    data = inputs(furnished=furnished)
    solved = solver_inputs(data)
    contact = case.CONTACT_REL if furnished else None
    with pytest.MonkeyPatch.context() as patch:
        for module, name, fake in control.STAND_INS:
            patch.setattr(module, name, fake)
        raw = three_lane_report.solve_three_lane_report(**solved._asdict(), contact_rel=contact)
        lane = raw.geometric_lane
        output = output_from_report(raw, inputs=data, with_points=True,
                                    path_table_inputs=solved, contact_rel=contact)
        window = build_reflection_window(data, frequencies_hz=lane.frequencies_hz,
            scattering_coefficient=lane.scattering, window_s=0.015, contact_rel=contact)
        return ControlPairPhysics(report=output, window=window,
            screen=build_reflection_screen(data, lane.frequencies_hz),
            third_octave_decay=build_third_octave_decay(raw, data)).model_dump_json()


@pytest.fixture(scope="module")
def single_pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[ControlPairPhysics, ...]:
    return tuple(ControlPairPhysics.model_validate_json(shared_json(tmp_path_factory, worker_id,
        f"step6-single-{furnished}", lambda: _single(furnished))) for furnished in (False, True))


def test_single_pair_changes_geometry_but_keeps_wall_screen_and_decay(single_pair: tuple[ControlPairPhysics, ...]) -> None:
    plain, furnished = single_pair
    assert furnished.report != plain.report and furnished.window != plain.window
    assert furnished.report.path_table != plain.report.path_table
    assert furnished.report.points is not None and plain.report.points is not None
    assert tuple(point.total_energy for point in furnished.report.points) != tuple(
        point.total_energy for point in plain.report.points)
    assert furnished.report.path_table is not None and plain.report.path_table is not None
    assert furnished.report.path_table.blocked_wall_paths is not None
    assert ("floor",) in {row.wall_sequence for row in plain.report.path_table.rows}
    assert ("floor",) in furnished.report.path_table.blocked_wall_paths
    assert ("desk", FaceDirection.TOP) in {
        (row.furniture_id, row.furniture_face) for row in furnished.report.path_table.rows}
    assert furnished.window.coverage == "approximate" and plain.window.coverage == "complete"
    assert furnished.window.rows != plain.window.rows
    assert furnished.screen.pairs == plain.screen.pairs
    assert furnished.screen.next_order_earliest_delay_s == plain.screen.next_order_earliest_delay_s
    assert furnished.screen.scene_fingerprint != plain.screen.scene_fingerprint
    assert furnished.third_octave_decay.rows == plain.third_octave_decay.rows
    assert furnished.third_octave_decay.scene_fingerprint != plain.third_octave_decay.scene_fingerprint
    assert tuple((band.t20_s, band.t30_s) for band in furnished.report.bands) == tuple(
        (band.t20_s, band.t30_s) for band in plain.report.bands)
    assert furnished.report.top.eyring_t60_by_band_s == plain.report.top.eyring_t60_by_band_s


def _open_only_construction_gate(patch: pytest.MonkeyPatch, swallowed: list[str]) -> None:
    original = validation._checked_furniture

    def checked(scheme: Scheme) -> None:
        try:
            original(scheme)
        except validation.SchemeValidationError as exc:
            if exc.problems != (validation.SchemeProblem("furniture", schemes.GATE_MESSAGE),):
                raise
            swallowed.append(scheme.scheme_id)

    patch.setattr(validation, "_checked_furniture", checked)


def _scheme_result(furnished: bool) -> str:
    content = schemes.document(schemes.relative_item()) if furnished else schemes.document()
    content["scheme_id"] = "step6-furnished" if furnished else "step6-plain"
    scheme = Scheme.model_validate(content)
    swallowed: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        _open_only_construction_gate(patch, swallowed)
        result = _run_control(scheme)
    assert swallowed == ([scheme.scheme_id] if furnished else []), "施工關必須真的被吞過一次，拆關後此題要紅"
    return result.model_dump_json()


@pytest.fixture(scope="module")
def scheme_pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, ...]:
    return tuple(SchemeResult.model_validate_json(shared_json(tmp_path_factory, worker_id,
        f"step6-scheme-{furnished}", lambda: _scheme_result(furnished))) for furnished in (False, True))


def test_scheme_wrapper_preserves_other_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    swallowed: list[str] = []
    _open_only_construction_gate(monkeypatch, swallowed)
    blocked = Scheme.model_validate(schemes.document(schemes.relative_item(height_m=2.0)))
    with pytest.raises(validation.SchemeValidationError, match="直達路徑被家具"):
        validation.checked_inputs(blocked, capabilities=schemes.CAPABILITIES, directivity=DIRECTIVITY)
    assert not swallowed


def test_scheme_save_reload_and_comparison_use_real_furniture_geometry(
    scheme_pair: tuple[SchemeResult, ...], tmp_path: Path,
) -> None:
    restored = []
    for result in scheme_pair:
        path = tmp_path / f"{result.scheme.scheme_id}.json"
        save_result(result, path)
        loaded = load_result(path, capabilities=schemes.CAPABILITIES, directivity=DIRECTIVITY,
            quality_targets_path=control.TARGETS, physics_identity=result.physics_identity)
        assert loaded.standing is ResultStanding.CURRENT
        assert loaded.result == result
        restored.append(loaded.result)
    plain, furnished = restored
    geometric = {QualityCategory.TIMBRE_BALANCE, QualityCategory.LISTENING_AREA_STABILITY,
                 QualityCategory.CHANNEL_MATCHING, QualityCategory.REFLECTIONS_AND_ECHO}
    assert {item.category for item in furnished.candidate.evaluations
            if Flag.FURNITURE_MODEL_APPROXIMATE in item.flags} == geometric
    assert all(Flag.FURNITURE_MODEL_APPROXIMATE not in item.flags for item in plain.candidate.evaluations)
    reflection = next(item for item in furnished.candidate.evaluations if item.category is QualityCategory.REFLECTIONS_AND_ECHO)
    assert isinstance(reflection.payload, ReflectionsAndEchoPayload)
    assert reflection.payload.furniture_model == "single_bounce_finite_size_v1"
    assert {channel.coverage for channel in reflection.payload.channels} == {"approximate"}
    left = next(pair for pair in furnished.pairs if (pair.speaker_id, pair.receiver_id) == ("left", "main"))
    assert left.report.path_table is not None
    assert left.report.path_table.blocked_wall_paths is not None
    assert ("floor",) in left.report.path_table.blocked_wall_paths
    assert ("seat", FaceDirection.TOP) in {(row.furniture_id, row.furniture_face) for row in left.report.path_table.rows}
    registry = load_quality_targets(config_path("quality_targets.toml"))
    ranking = compare_results(restored, quality_targets=registry, run_date=date(2026, 10, 8))
    assert ranking.not_comparable.rows
    assert {category for row in ranking.not_comparable.rows
            for _, categories in identity_difference_groups(ranking, row.identity)
            for category in categories} == {QualityCategory.REFLECTIONS_AND_ECHO}
    view = _view((plain, furnished))
    assert {row.category for row in view.categories if row.comparison_text} == {"reflections_and_echo"}
    assert next(row.comparison_text for row in view.categories if row.category == "reflections_and_echo") == (
        "評分條件不同，這一類代價不能直接比")
