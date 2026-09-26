"""#505 第三刀控制組的管線：兩個全向候選真解報表，六類評估、裝候選包、排名，再攤平成可比的形狀。

答案住 ``_scoring_source_model_answers.py``，是主對話在改動前的主線上用這一支實跑錄下的，不是被測函式現算的。
這一支只呼叫正式入口（報表、補算窗、牆對篩檢、三分之一八度衰減、六個評估器、候選包、排名），
自己不建任何一類評估結果，所以評估結果多一格之後這一支一字不改還能跑。
有限元素、晚期衰減、晚期混響逐階能量沿用第二刀控制組的三個替身（真的那兩支晚期解特徵值與線性方程組，
不同機器最後幾位會不同）。右喇叭放角落、周圍點偏一邊，五類代價都不是 0。

每一段攤平成兩半：非浮點（每個路徑、字串、旗標順序、狀態、原因碼、名次、分表）逐字比雜湊；
浮點照去掉索引的路徑分組、每組記個數與兩個加權和，另把分數本身（總代價、類代價、分項、原始量）逐一記。
浮點用相對誤差 1e-12 比，不逐位：評分層在陣列上用 numpy 的對數，雲端機器若走另一套指令集的實作，
最後幾位可能不同，本機測不到（照 test_timbre_listening_area_support.py 音色控制組的比法）。
按設計會變的格子明列在 ``EVALUATOR_VERSION``、``VERSION_FOLDING_CATEGORIES``、``SKIPPED_KEYS``。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator
from datetime import date
from typing import Final

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.physics import report_io, three_lane_report, three_lane_report_batch
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_io import ReportOutput
from aosr.physics.report_output import output_from_report
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.third_octave_decay import build_third_octave_decay
from aosr.scoring.channel_matching import (
    ChannelComparison, ChannelDefinition, ChannelGroup, ChannelPointInput, ChannelResponse,
    evaluate_channel_matching,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CandidateEvaluation, CategoryEvaluation, InputProvenance,
)
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.ranking import RankingContext, RankingResult, rank_candidates
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections
from aosr.scoring.reverberation import evaluate_reverberation
from aosr.scoring.timbre import evaluate_timbre, timbre_input_from_report
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from tests.engine import _source_model_control as stand_ins

PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
TARGETS = config_path("quality_targets.toml")
WINDOW_S: Final[float] = 15.0 / 1000.0
SPEAKERS: Final[dict[str, dict[str, float]]] = {
    "left": {"x": 1.15, "y": 0.95, "z": 1.2},
    "right": {"x": 0.6, "y": 3.6, "z": 1.25},
}
RECEIVERS: Final[dict[str, dict[str, float]]] = {
    "main": {"x": 4.35, "y": 2.05, "z": 1.05},
    "front": {"x": 3.3, "y": 1.6, "z": 1.05},
}
IMPEDANCE_MULTIPLES: Final[dict[str, float]] = {"wall-1": 1.0, "wall-2": 2.0}
STAND_INS = (
    (three_lane_report, "_solve_fem_energy", stand_ins.fake_fem_energy),
    (three_lane_report, "_solve_report_late_decay", stand_ins.fast_late_decay),
    (three_lane_report_batch, "solve_geometric_late_energy", stand_ins.fake_late_energy),
)
# 第三刀按設計會變的格子，逐一列；其餘每一格照舊比：
#   評估器版本字串：六個評估器都升版——值裡的「.v<數字>」換成「.v*」，出現在哪一格都一樣（含音色比較支撐
#     那段 JSON）；只認這六個評估器的名字，契約與排名的版本字串不在內、照舊逐字比。
#   聆聽區與聲道匹配的設定指紋：這兩支把上游評估器版本折進去（listening_area.py、channel_matching.py 的
#     _settings_fingerprint），值換成「<類名 settings>」，出現在哪一格都一樣；其他設定指紋不折版本，照舊比。
#   反射 payload 的 includes_speaker_directivity 換型成 source_model_kind、評估結果與比較身分新的
#     source_model_fingerprint：整格跳過（這幾格的值由新考卷守）。
EVALUATOR_VERSION = re.compile(
    r"(aosr\.scoring\.(?:timbre|timbre_channels|listening_area|channel_matching|reflections|reverberation))"
    r"\.v[0-9]+"
)
VERSION_FOLDING_CATEGORIES: Final[tuple[str, ...]] = ("listening_area_stability", "channel_matching")
SKIPPED_KEYS: Final[frozenset[str]] = frozenset({
    "includes_speaker_directivity", "source_model_kind", "source_model_fingerprint",
})


def _inputs(candidate: str, speaker: str, receiver: str) -> report_io.ReportInput:
    walls = stand_ins.SCENE_WITHOUT_SOURCE_MODEL["impedance_pa_s_per_m_by_wall"]
    assert isinstance(walls, dict)
    return report_io.load_input_document({
        **stand_ins.SCENE_WITHOUT_SOURCE_MODEL,
        "source_model": {"kind": "omnidirectional"},
        "source_m": SPEAKERS[speaker], "receiver_m": RECEIVERS[receiver],
        "impedance_pa_s_per_m_by_wall": {
            wall: IMPEDANCE_MULTIPLES[candidate] * value for wall, value in walls.items()
        },
    }, load_capabilities(config_path("capabilities.toml")))


def _provenance(speaker: str, receiver: str) -> InputProvenance:
    return InputProvenance(report_id=f"report-{speaker}-{receiver}", engine_commit="control",
                           speaker_id=speaker, receiver_id=receiver)


def _solve(candidate: str, speaker: str, receiver: str) -> tuple[ReportOutput, ReflectionInput]:
    inputs = _inputs(candidate, speaker, receiver)
    solved = report_io.solver_inputs(inputs)
    raw = three_lane_report.solve_three_lane_report(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s, density_kg_m3=solved.density_kg_m3,
        impedance_by_wall=solved.impedance_by_wall, scattering_by_wall=solved.scattering_by_wall,
        reflection_order_k=solved.reflection_order_k, low_frequency_axis=solved.low_frequency_axis,
    )
    report = output_from_report(raw, inputs=inputs, with_points=True, path_table_inputs=solved)
    lane = raw.geometric_lane
    record = ReflectionInput(
        role=speaker, receiver_id=receiver, report=report,
        screen=build_reflection_screen(inputs, lane.frequencies_hz),
        window=build_reflection_window(inputs, frequencies_hz=lane.frequencies_hz,
                                       scattering_coefficient=lane.scattering, window_s=WINDOW_S),
        third_octave_decay=build_third_octave_decay(raw, inputs),
        report_id=f"report-{speaker}-{receiver}", engine_commit="control", speaker_id=speaker,
    )
    return report, record


def receivers() -> ReceiverSet:
    main, front = RECEIVERS["main"], RECEIVERS["front"]
    return ReceiverSet(points=(
        ReceiverPoint(receiver_id="main", position_m=(main["x"], main["y"], main["z"]),
                      role=ReceiverRole.PRIMARY, importance=1.0),
        ReceiverPoint(receiver_id="front", position_m=(front["x"], front["y"], front["z"]),
                      role=ReceiverRole.SURROUNDING, importance=1.0,
                      direction_relative_to_primary="front"),
    ))


def group() -> ChannelGroup:
    return ChannelGroup(
        channels=(ChannelDefinition(role="left", speaker_id="left"),
                  ChannelDefinition(role="right", speaker_id="right")),
        comparisons=(ChannelComparison(left_role="left", right_role="right"),),
        feature_match_tolerance_hz=10.0,
    )


def _distance(report: ReportOutput) -> float:
    source, receiver = report.scene.source_m, report.scene.receiver_m
    return math.dist((source.x, source.y, source.z), (receiver.x, receiver.y, receiver.z))


def _curve(report: ReportOutput) -> tuple[tuple[float, ...], tuple[float, ...]]:
    assert report.points is not None
    return (tuple(point.frequency_hz for point in report.points),
            tuple(point.total_energy for point in report.points))


def _channel_points(reports: dict[tuple[str, str], ReportOutput],
                    timbres: dict[tuple[str, str], CategoryEvaluation],
                    listening: CategoryEvaluation) -> tuple[ChannelPointInput, ...]:
    return tuple(ChannelPointInput(
        receiver_id=receiver, receiver_set_fingerprint=receivers().fingerprint,
        timbre_settings_fingerprint=timbres[("left", receiver)].settings_fingerprint,
        listening_area_settings_fingerprint=listening.settings_fingerprint,
        channel_group_fingerprint=group().fingerprint,
        responses=tuple(ChannelResponse(
            role=role, timbre_evaluation=timbres[(role, receiver)],
            frequencies_hz=_curve(reports[(role, receiver)])[0],
            total_energy=_curve(reports[(role, receiver)])[1],
            direct_distance_m=_distance(reports[(role, receiver)]),
        ) for role in ("left", "right")),
    ) for receiver in RECEIVERS)


def candidate(name: str) -> CandidateEvaluation:
    """一個候選：四份真報表、六個評估器、裝成候選包。"""
    solved = {(speaker, receiver): _solve(name, speaker, receiver)
              for speaker in SPEAKERS for receiver in RECEIVERS}
    reports = {key: report for key, (report, _) in solved.items()}
    timbres = {key: evaluate_timbre(timbre_input_from_report(
        report, candidate_id=name, speaker_id=key[0], receiver_id=key[1],
        source_reference="三路接合報表共同能量基準", provenance=_provenance(*key),
    ), purpose=PURPOSE, quality_targets_path=TARGETS) for key, report in reports.items()}
    scene = timbres[("left", "main")].scene_fingerprint
    timbre_settings = timbres[("left", "main")].settings_fingerprint
    listening = evaluate_listening_area(
        receivers(), tuple(ReceiverPointResult(
            receiver_id=receiver, receiver_set_fingerprint=receivers().fingerprint,
            timbre_evaluation=timbres[("left", receiver)],
            frequencies_hz=_curve(reports[("left", receiver)])[0],
            total_energy=_curve(reports[("left", receiver)])[1],
        ) for receiver in RECEIVERS),
        candidate_id=name, speaker_id="left", timbre_settings_fingerprint=timbre_settings,
        scene_fingerprint=scene, feature_match_tolerance_hz=10.0, broadband_range_hz=(20.0, 8000.0),
    )
    reflections = evaluate_reflections(
        group(), tuple(record for _, record in solved.values()), primary_receiver_id="main",
        candidate_id=name, purpose=PURPOSE, quality_targets_path=TARGETS,
    )
    evaluations = (
        evaluate_timbre_channels(
            group(), "main", {role: timbres[(role, "main")] for role in SPEAKERS},
            candidate_id=name, scene_fingerprint=scene, timbre_settings_fingerprint=timbre_settings,
        ),
        listening,
        reflections,
        evaluate_channel_matching(
            receivers(), _channel_points(reports, timbres, listening), candidate_id=name,
            scene_fingerprint=scene, timbre_settings_fingerprint=timbre_settings,
            listening_area_settings_fingerprint=listening.settings_fingerprint,
            channel_group=group(), purpose=PURPOSE, quality_targets_path=TARGETS,
            reflections=reflections, sound_speed_m_s=343.0,
        ),
        evaluate_reverberation(reports[("left", "main")], candidate_id=name,
                               provenance=_provenance("left", "main"), logarithm_base=2.0),
    )
    return CandidateEvaluation(schema_version=CONTRACT_SCHEMA_VERSION, candidate_id=name,
                               scene_fingerprint=scene, evaluations=evaluations)


def run() -> tuple[tuple[CandidateEvaluation, ...], RankingResult]:
    """兩個候選加排名；呼叫端先掛好 ``STAND_INS``。"""
    candidates = tuple(candidate(name) for name in IMPEDANCE_MULTIPLES)
    ranking = rank_candidates(candidates, load_quality_targets(TARGETS), RankingContext(
        purpose=PURPOSE, receiver_set_fingerprint=receivers().fingerprint,
        channel_group_fingerprint=group().fingerprint, run_date=date(2026, 9, 27),
        engine_version="scoring-source-model-control",
    ))
    return candidates, ranking


def version_folding(candidates: tuple[CandidateEvaluation, ...]) -> dict[str, str]:
    """聆聽區與聲道匹配的設定指紋 → 佔位字；兩個候選的設定相同，指紋也相同。"""
    return {evaluation.settings_fingerprint: f"<{evaluation.category.value} settings>"
            for item in candidates for evaluation in item.evaluations
            if evaluation.category.value in VERSION_FOLDING_CATEGORIES}


def _leaves(node: object, path: str, folding: dict[str, str]) -> Iterator[tuple[str, object]]:
    if isinstance(node, dict):
        for key in sorted(node):
            if key not in SKIPPED_KEYS:
                yield from _leaves(node[key], f"{path}.{key}", folding)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _leaves(item, f"{path}[{index}]", folding)
    elif isinstance(node, str):
        yield path, EVALUATOR_VERSION.sub(r"\1.v*", folding.get(node, node))
    else:
        yield path, node


def _pattern(path: str) -> str:
    return re.sub(r"\[[0-9]+\]", "[]", path)


def flatten(document: object, folding: dict[str, str]) -> dict[str, object]:
    """一段攤平成兩半：非浮點（每個路徑與值，浮點只留路徑）的雜湊逐字比；
    浮點照去掉索引的路徑分組，每組記（個數, 加權絕對值和, 加權帶號和），用 fsum 算、順序固定。"""
    exact: list[tuple[str, object]] = []
    groups: dict[str, list[float]] = {}
    for path, value in _leaves(document, "", folding):
        if isinstance(value, float):
            exact.append((path, "<float>"))
            groups.setdefault(_pattern(path), []).append(value)
        else:
            exact.append((path, value))
    text = json.dumps(exact, ensure_ascii=False, separators=(",", ":"))
    return {
        "exact": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "floats": {pattern: (
            len(values),
            math.fsum(abs(value) * (1.0 + (index % 7) / 7.0) for index, value in enumerate(values)).hex(),
            math.fsum(value * (1.0 + (index % 5) / 5.0) for index, value in enumerate(values)).hex(),
        ) for pattern, values in sorted(groups.items())},
    }


def sections(candidates: tuple[CandidateEvaluation, ...],
             ranking: RankingResult) -> dict[str, dict[str, object]]:
    """每個候選每一類的評估一段；排名拆成表頭、每一列（不含逐類）、每一列每一類、其餘四塊。"""
    folding = version_folding(candidates)
    parts: dict[str, object] = {
        f"{item.candidate_id}/{evaluation.category.value}": evaluation.model_dump(mode="json")
        for item in candidates for evaluation in item.evaluations
    }
    document = ranking.model_dump(mode="json")
    parts["ranking/header"] = document.pop("header")
    for row in document.pop("rankable"):
        for line in row.pop("categories"):
            parts[f"ranking/{row['candidate_id']}/{line['evaluation']['category']}"] = line
        parts[f"ranking/{row['candidate_id']}/row"] = row
    parts["ranking/rest"] = document
    return {name: flatten(part, folding) for name, part in parts.items()}


def headline(ranking: RankingResult) -> dict[str, object]:
    """分數本身逐一記：總代價、每類的類代價、權重、加權代價、分項代價、原始量。"""
    return {row.candidate_id: {
        "total_cost": row.total_cost.hex(),
        "categories": {line.evaluation.category.value: {
            "category_cost": line.category_cost.hex(),
            "category_weight": line.category_weight.hex(),
            "weighted_cost": line.weighted_cost.hex(),
            "components": {item.name: item.cost.hex() for item in line.components},
            "raw_quantities": {item.name: item.value.hex() for item in line.evaluation.raw_quantities},
        } for line in row.categories},
    } for row in ranking.rankable}


def summary(candidates: tuple[CandidateEvaluation, ...], ranking: RankingResult) -> dict[str, object]:
    """給人看、也逐字比的一小段：每類的狀態、旗標、原因碼；每列的名次與合起來的旗標；其餘三塊有誰。"""
    return {
        "evaluations": {f"{item.candidate_id}/{evaluation.category.value}": [
            evaluation.state.value, [flag.value for flag in evaluation.flags],
            [code.value for code in evaluation.reason_codes],
        ] for item in candidates for evaluation in item.evaluations},
        "rows": {row.candidate_id: [row.rank, [flag.value for flag in row.flags]]
                 for row in ranking.rankable},
        "other_blocks": [[row.candidate_id for row in ranking.eliminated],
                         [row.candidate_id for row in ranking.not_evaluated],
                         [row.candidate_id for row in ranking.not_comparable.rows]],
    }
