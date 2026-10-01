"""#505 第四步：預設模型從輸入一路進到評估與排名。"""

from __future__ import annotations

from copy import deepcopy
from datetime import date

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point
from aosr.physics import report_io
from aosr.physics.report_source import default_source_model
from aosr.scoring.contract import Flag, QualityCategory
from aosr.scoring.ranking import RankingContext, rank_candidates
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from tests.engine import _scoring_source_model_control as pipeline
from tests.engine import _source_model_control as stand_ins
from tests.engine._directivity import DIRECTIVITY
from tests.engine._report_cache import control_pair_cache, shared_control_pair


def _shared_omni_report() -> report_io.ReportOutput:
    return pipeline._solve("wall-1", "left", "main")[0]


def test_default_directivity_reaches_report_evaluators_and_ranking(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    for module, name, stand_in in pipeline.STAND_INS:
        monkeypatch.setattr(module, name, stand_in)
    # 全向與解析單對上游各自在本跑真算一次；評估器與排名仍由本題執行。
    omni_report = _shared_omni_report()
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
        physical = shared_control_pair(inputs, pipeline.WINDOW_S)
        output = physical.report
        record = ReflectionInput(
            role=speaker, receiver_id=receiver, report=output,
            screen=physical.screen, window=physical.window,
            third_octave_decay=physical.third_octave_decay,
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
