"""保存零件的評估編排、品質登記簿設定與結果讀回驗證。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.quality_targets import QualityPurpose, SettingEntry, load_quality_targets
from aosr.physics.reflection_window import ReflectionWindow, build_reflection_window
from aosr.physics.report_io import ReportInput, ReportOutput, load_input_document
from aosr.scoring.channel_matching import ChannelPointInput, ChannelResponse, evaluate_channel_matching
from aosr.scoring.contract import (CONTRACT_SCHEMA_VERSION, CandidateEvaluation,
                                   CategoryEvaluation, QualityCategory)
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.listening_area_channels import evaluate_listening_area_channels
from aosr.scoring.reflections import evaluate_reflections
from aosr.scoring.reverberation import evaluate_reverberation
from aosr.scoring.timbre import evaluate_timbre, timbre_input_from_report
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.reporting.result import RESULT_SCHEMA_VERSION, PairResult, SchemeResult
from aosr.reporting.scheme import Scheme


SOURCE_REFERENCE = "三路接合報表共同能量基準"


def build_pair_window(inputs: ReportInput, report: ReportOutput,
                      window_s: float) -> ReflectionWindow:
    """只用保存的路徑表頻率與散射補算反射窗；缺表就拒收。"""
    table = report.path_table
    if table is None:
        raise ValueError("存下來的報表缺 path_table，無法建反射窗")
    return build_reflection_window(inputs, frequencies_hz=table.frequencies_hz,
                                   scattering_coefficient=table.scattering_coefficient,
                                   window_s=window_s)


class RegistrySettings(NamedTuple):
    window_s: float
    broadband_range_hz: tuple[float, float]
    logarithm_base: float


def _setting(purpose: QualityPurpose, key: str, unit: str) -> float | tuple[float, ...]:
    found = purpose.entry(key)
    if not isinstance(found, SettingEntry) or found.unit != unit:
        raise ValueError(f"{key} 必須是 {unit} 量法設定")
    return found.value


def read_registry_settings(path: Path, purpose_name: str) -> RegistrySettings:
    """同一次載入核反射窗、寬頻與殘響對數底的值及單位。"""
    purpose = load_quality_targets(path).purpose(purpose_name)
    window = _setting(purpose, "reflections_and_echo.window_upper_ms", "ms")
    broadband = _setting(purpose, "channel_matching.broadband_range_hz", "Hz")
    base = _setting(purpose, "reverberation.adjacent_t20_logarithm_base", "1")
    if isinstance(window, tuple) or isinstance(base, tuple):
        raise ValueError("反射窗與殘響對數底必須是單值")
    if not isinstance(broadband, tuple) or len(broadband) != 2:
        raise ValueError("寬頻範圍必須有兩端點")
    return RegistrySettings(float(window) / 1000.0,
                            (float(broadband[0]), float(broadband[1])), float(base))


def _curve(report: ReportOutput) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if report.points is None:
        raise ValueError("存下來的報表缺逐點資料")
    return (tuple(point.frequency_hz for point in report.points),
            tuple(point.total_energy for point in report.points))


def receiver_point_results(
    scheme: Scheme, pairs: dict[tuple[str, str], PairResult],
    timbres: dict[tuple[str, str], CategoryEvaluation], role: str,
) -> tuple[ReceiverPointResult, ...]:
    """重建單支喇叭逐座位輸入；重評與結果畫面共用。"""
    return tuple(ReceiverPointResult(
        receiver_id=point.receiver_id,
        receiver_set_fingerprint=scheme.receiver_set.fingerprint,
        timbre_evaluation=timbres[role, point.receiver_id],
        frequencies_hz=_curve(pairs[role, point.receiver_id].report)[0],
        total_energy=_curve(pairs[role, point.receiver_id].report)[1],
    ) for point in scheme.receiver_set.points)


def evaluate_point_timbres(
    result: SchemeResult, quality_targets_path: Path,
) -> dict[tuple[str, str], CategoryEvaluation]:
    """逐支逐座位音色；重評與結果畫面走同一條輸入路。"""
    return {(pair.role, pair.receiver_id): evaluate_timbre(timbre_input_from_report(
        pair.report, candidate_id=result.scheme.scheme_id, speaker_id=pair.speaker_id,
        receiver_id=pair.receiver_id, source_reference=SOURCE_REFERENCE,
        provenance=pair.provenance(result.engine_commit),
    ), purpose=result.scheme.purpose, quality_targets_path=quality_targets_path)
        for pair in result.pairs}


def _distance(report: ReportOutput) -> float:
    return math.dist(report.scene.source_m.as_tuple(), report.scene.receiver_m.as_tuple())


def _channel_points(
    result: SchemeResult, timbres: dict[tuple[str, str], CategoryEvaluation],
    listening: CategoryEvaluation,
) -> tuple[ChannelPointInput, ...]:
    pairs = {(pair.role, pair.receiver_id): pair for pair in result.pairs}
    group = result.scheme.channel_group
    first = group.channels[0].role
    return tuple(ChannelPointInput(
        receiver_id=point.receiver_id,
        receiver_set_fingerprint=result.scheme.receiver_set.fingerprint,
        timbre_settings_fingerprint=timbres[first, point.receiver_id].settings_fingerprint,
        listening_area_settings_fingerprint=listening.settings_fingerprint,
        channel_group_fingerprint=group.fingerprint,
        responses=tuple(ChannelResponse(
            role=channel.role,
            timbre_evaluation=timbres[channel.role, point.receiver_id],
            frequencies_hz=_curve(pairs[channel.role, point.receiver_id].report)[0],
            total_energy=_curve(pairs[channel.role, point.receiver_id].report)[1],
            direct_distance_m=_distance(pairs[channel.role, point.receiver_id].report),
        ) for channel in group.channels),
    ) for point in result.scheme.receiver_set.points)


def _listening_evaluation(
    scheme: Scheme, pairs: dict[tuple[str, str], PairResult],
    timbres: dict[tuple[str, str], CategoryEvaluation],
    settings_values: RegistrySettings, scene: str, settings: str,
) -> CategoryEvaluation:
    """逐聲道量聆聽區，再依聲道組收成唯一的候選類別。"""
    singles = {
        channel.role: evaluate_listening_area(
            scheme.receiver_set,
            receiver_point_results(scheme, pairs, timbres, channel.role),
            candidate_id=scheme.scheme_id, speaker_id=channel.speaker_id,
            timbre_settings_fingerprint=settings, scene_fingerprint=scene,
            feature_match_tolerance_hz=scheme.channel_group.feature_match_tolerance_hz,
            broadband_range_hz=settings_values.broadband_range_hz,
        ) for channel in scheme.channel_group.channels
    }
    first = scheme.channel_group.channels[0].role
    return evaluate_listening_area_channels(
        scheme.channel_group, scheme.receiver_set, singles,
        candidate_id=scheme.scheme_id, scene_fingerprint=scene,
        listening_area_settings_fingerprint=singles[first].settings_fingerprint,
    )


def evaluate_parts(result: SchemeResult, quality_targets_path: Path,
                    settings_values: RegistrySettings) -> CandidateEvaluation:
    scheme = result.scheme
    pairs = {(pair.role, pair.receiver_id): pair for pair in result.pairs}
    timbres = evaluate_point_timbres(result, quality_targets_path)
    first = scheme.channel_group.channels[0].role
    primary = scheme.receiver_set.primary.receiver_id
    scene = timbres[first, primary].scene_fingerprint
    settings = timbres[first, primary].settings_fingerprint
    # 聆聽區寬頻量與聲道匹配呼叫同一支寬頻能量計算，共用登記簿鍵。
    listening = _listening_evaluation(scheme, pairs, timbres, settings_values,
                                      scene, settings)
    reflections = evaluate_reflections(
        scheme.channel_group,
        tuple(pair.reflection_input(result.engine_commit) for pair in result.pairs),
        primary_receiver_id=primary, candidate_id=scheme.scheme_id,
        purpose=scheme.purpose, quality_targets_path=quality_targets_path,
    )
    evaluations = (
        evaluate_timbre_channels(
            scheme.channel_group, primary,
            {channel.role: timbres[channel.role, primary]
             for channel in scheme.channel_group.channels},
            candidate_id=scheme.scheme_id, scene_fingerprint=scene,
            timbre_settings_fingerprint=settings,
        ),
        listening, reflections,
        evaluate_channel_matching(
            scheme.receiver_set, _channel_points(result, timbres, listening),
            candidate_id=scheme.scheme_id, scene_fingerprint=scene,
            timbre_settings_fingerprint=settings,
            listening_area_settings_fingerprint=listening.settings_fingerprint,
            channel_group=scheme.channel_group, purpose=scheme.purpose,
            quality_targets_path=quality_targets_path, reflections=reflections,
            sound_speed_m_s=scheme.scene.sound_speed_m_s,
        ),
        evaluate_reverberation(
            pairs[first, primary].report, candidate_id=scheme.scheme_id,
            provenance=pairs[first, primary].provenance(result.engine_commit),
            logarithm_base=settings_values.logarithm_base,
        ),
    )
    return CandidateEvaluation(schema_version=CONTRACT_SCHEMA_VERSION,
                               candidate_id=scheme.scheme_id,
                               scene_fingerprint=scene, evaluations=evaluations)


def reevaluate(result: SchemeResult, *, quality_targets_path: Path) -> CandidateEvaluation:
    """只讀保存的報表與反射零件，重跑五類評估。"""
    settings_values = read_registry_settings(quality_targets_path, result.scheme.purpose)
    return evaluate_parts(result, quality_targets_path, settings_values)


def quality_targets_fingerprint(path: Path) -> str:
    """品質登記簿內容的指紋（排名表頭同一把尺）；結果檔記它，讀回時比對。"""
    return load_quality_targets(path).fingerprint


def load_result(path: Path, *, capabilities: CapabilityTable,
                directivity: DirectivityDefaults,
                quality_targets_path: Path) -> SchemeResult:
    """讀入 JSON，重驗報表輸入並由零件重評候選包。"""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if isinstance(document, dict) and document.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("結果檔格式是舊版（欄位 schema_version），請重算")
    result = SchemeResult.model_validate(document)
    for pair in result.pairs:
        load_input_document(pair.input_document, capabilities, directivity)
    current = quality_targets_fingerprint(quality_targets_path)
    if current != result.quality_targets_fingerprint:
        raise ValueError(
            f"品質登記簿的內容跟存檔時不同（存檔 {result.quality_targets_fingerprint[:12]}、"
            f"現在 {current[:12]}）：舊結果要用現在的登記簿重跑這個方案才能比較"
        )
    if reevaluate(result, quality_targets_path=quality_targets_path) != result.candidate:
        raise ValueError(
            "候選包跟存下來的零件對不上：檔案內容被改過，或評估程式跟存檔時不同版"
            f"（存檔 engine_commit={result.engine_commit}）"
        )
    return result
