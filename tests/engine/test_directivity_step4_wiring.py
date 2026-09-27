"""#505 第四步：預設模型從輸入一路進到評估與排名。"""

from __future__ import annotations

from copy import deepcopy
from datetime import date

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point
from aosr.physics import report_io, three_lane_report
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_output import output_from_report
from aosr.physics.report_source import default_source_model
from aosr.physics.third_octave_decay import build_third_octave_decay
from aosr.scoring.contract import Flag, QualityCategory
from aosr.scoring.ranking import RankingContext, rank_candidates
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from tests.engine import _scoring_source_model_control as pipeline
from tests.engine import _source_model_control as stand_ins
from tests.engine._directivity import DIRECTIVITY


def test_default_directivity_reaches_report_evaluators_and_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    for module, name, stand_in in pipeline.STAND_INS:
        monkeypatch.setattr(module, name, stand_in)
    omni_report, _omni_record = pipeline._solve("wall-1", "left", "main")
    omni_candidate = pipeline.candidate("wall-1")
    primary = Point(*pipeline.receivers().primary.position_m)

    def analytic_solve(candidate: str, speaker: str, receiver: str) -> tuple[report_io.ReportOutput, ReflectionInput]:
        scene = deepcopy(stand_ins.SCENE_WITHOUT_SOURCE_MODEL)
        walls = scene["impedance_pa_s_per_m_by_wall"]
        assert isinstance(walls, dict)
        scene["impedance_pa_s_per_m_by_wall"] = {
            wall: pipeline.IMPEDANCE_MULTIPLES[candidate] * value for wall, value in walls.items()
        }
        scene["source_m"] = pipeline.SPEAKERS[speaker]
        scene["receiver_m"] = pipeline.RECEIVERS[receiver]
        scene["source_model"] = default_source_model(primary, DIRECTIVITY).model_dump(mode="json")
        inputs = report_io.load_input_document(
            scene, load_capabilities(config_path("capabilities.toml")), DIRECTIVITY,
        )
        solved = report_io.solver_inputs(inputs)
        raw = three_lane_report.solve_three_lane_report(**solved._asdict())
        output = output_from_report(raw, inputs=inputs, with_points=True, path_table_inputs=solved)
        lane = raw.geometric_lane
        record = ReflectionInput(
            role=speaker, receiver_id=receiver, report=output,
            screen=build_reflection_screen(inputs, lane.frequencies_hz),
            window=build_reflection_window(
                inputs, frequencies_hz=lane.frequencies_hz,
                scattering_coefficient=lane.scattering, window_s=pipeline.WINDOW_S,
            ),
            third_octave_decay=build_third_octave_decay(raw, inputs),
            report_id=f"analytic-{speaker}-{receiver}", engine_commit="step4",
            speaker_id=speaker,
        )
        return output, record

    monkeypatch.setattr(pipeline, "_solve", analytic_solve)
    directed_report, _directed_record = analytic_solve("wall-1", "left", "main")
    assert directed_report.points is not None and omni_report.points is not None
    assert any(
        analytic.total_energy != omni.total_energy
        for analytic, omni in zip(directed_report.points, omni_report.points, strict=True)
    )
    candidate = pipeline.candidate("wall-1")
    omni_reflections = next(item for item in omni_candidate.evaluations
                            if item.category is QualityCategory.REFLECTIONS_AND_ECHO)
    analytic_reflections = next(item for item in candidate.evaluations
                                if item.category is QualityCategory.REFLECTIONS_AND_ECHO)
    assert isinstance(omni_reflections.payload, ReflectionsAndEchoPayload)
    assert isinstance(analytic_reflections.payload, ReflectionsAndEchoPayload)
    assert analytic_reflections.payload.wall_pairs == omni_reflections.payload.wall_pairs
    ranking = rank_candidates(
        (candidate,), load_quality_targets(pipeline.TARGETS),
        RankingContext(
            purpose=pipeline.PURPOSE,
            receiver_set_fingerprint=pipeline.receivers().fingerprint,
            channel_group_fingerprint=pipeline.group().fingerprint,
            run_date=date(2026, 9, 27), engine_version="step4-wiring",
        ),
    )
    assert any(Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in row.flags for row in ranking.rankable)
    assert all(Flag.NO_DIRECTIVITY not in row.flags for row in ranking.rankable)
    assert any(Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in item.flags for item in candidate.evaluations)
    assert all(Flag.NO_DIRECTIVITY not in item.flags for item in candidate.evaluations)
