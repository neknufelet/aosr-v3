"""保存零件的評估編排、品質登記簿設定與結果讀回驗證。"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from pydantic import ValidationError

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.config.quality_targets import QualityPurpose, SettingEntry, load_quality_targets
from aosr.physics.reflection_window import ReflectionWindow, build_reflection_window
from aosr.physics.report_io import ReportInput, ReportOutput, load_input_document
from aosr.scoring.channel_matching import ChannelPointInput, ChannelResponse, evaluate_channel_matching
from aosr.scoring.channel_matching_settings import BROADBAND_KEY
from aosr.scoring.contract import (CONTRACT_SCHEMA_VERSION, CandidateEvaluation,
                                   CategoryEvaluation, QualityCategory)
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.listening_area_channels import evaluate_listening_area_channels
from aosr.scoring.reflections import WINDOW_KEY, evaluate_reflections
from aosr.scoring.reverberation import evaluate_reverberation
from aosr.scoring.reverberation_cost import LOG_BASE_KEY
from aosr.scoring.timbre import evaluate_timbre, timbre_input_from_report
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.reporting.result import (RESULT_SCHEMA_VERSION, PairResult, PurposeSettings, SchemeResult,
                                  check_declared_axes)
from aosr.reporting.scheme import Scheme


SOURCE_REFERENCE = "三路接合報表共同能量基準"


def build_pair_window(inputs: ReportInput, report: ReportOutput,
                      window_s: float) -> ReflectionWindow:
    """核保存表頭與輸入家具，再用保存的頻率與散射建窗；有家具才讀接觸尺。"""
    table = report.path_table
    if table is None:
        raise ValueError("存下來的報表缺 path_table，無法建反射窗")
    ids = None if inputs.furniture is None else tuple(item.furniture_id for item in inputs.furniture)
    if ids is not None and table.furniture_ids is None:
        raise ValueError("輸入有家具，存下來的路徑表表頭沒有家具")
    if ids is None and table.furniture_ids is not None:
        raise ValueError("輸入沒有家具，存下來的路徑表表頭卻有家具")
    if table.furniture_ids != ids:
        raise ValueError("存下來的路徑表表頭與輸入的家具代號不一致")
    materials = None if inputs.furniture is None else tuple(
        (item.furniture_id, item.material) for item in inputs.furniture)
    stored_materials = None if table.furniture_materials is None else tuple(
        (item.furniture_id, item.material) for item in table.furniture_materials)
    if stored_materials != materials:
        raise ValueError("存下來的路徑表表頭與輸入的家具材質不一致")
    contact_rel = furniture_contact_rel(default_precision_contracts_path()) if inputs.furniture is not None else None
    return build_reflection_window(inputs, frequencies_hz=table.frequencies_hz,
                                   scattering_coefficient=table.scattering_coefficient,
                                   window_s=window_s, contact_rel=contact_rel)


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
    # 三條的鍵名用讀它的評分模組的常數：各類宣告「評估器那一層讀的尺」時才對得上（#633）。
    window = _setting(purpose, WINDOW_KEY, "ms")
    broadband = _setting(purpose, BROADBAND_KEY, "Hz")
    base = _setting(purpose, LOG_BASE_KEY, "1")
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
                    settings_values: RegistrySettings, *, capabilities: CapabilityTable,
                    directivity: DirectivityDefaults) -> CandidateEvaluation:
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
        tuple(pair.reflection_input(result.engine_commit, build_pair_window(
            load_input_document(pair.input_document, capabilities, directivity),
            pair.report, settings_values.window_s)) for pair in result.pairs),
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


def reevaluate(result: SchemeResult, *, quality_targets_path: Path,
               capabilities: CapabilityTable, directivity: DirectivityDefaults) -> CandidateEvaluation:
    """只讀保存的報表與反射零件，重跑五類評估。"""
    settings_values = read_registry_settings(quality_targets_path, result.scheme.purpose)
    return evaluate_parts(result, quality_targets_path, settings_values,
                          capabilities=capabilities, directivity=directivity)


def quality_targets_fingerprint(path: Path) -> str:
    """品質登記簿內容的指紋（排名表頭同一把尺）；結果檔記它，讀回時比對。"""
    return load_quality_targets(path).fingerprint


def purpose_settings(quality_targets_path: Path, purpose: str) -> PurposeSettings:
    """保存該用途的正規化完整快照，不把其他用途的設定牽進來。"""
    try:
        settings = load_quality_targets(quality_targets_path).purpose(purpose)
    except KeyError as exc:
        raise ValueError(f"品質登記簿沒有這個用途（{purpose}），要先把用途加回登記簿") from exc
    return PurposeSettings(purpose=purpose, fingerprint=settings.fingerprint,
                           content=settings.canonical())


class ResultStanding(StrEnum):
    """讀回結果相對於現在的物理與評分的等級。"""

    CURRENT = "current"
    RERANKED = "reranked"
    REMEASURED = "remeasured"
    NEEDS_PHYSICS = "needs_physics"


@dataclass(frozen=True)
class LoadedResult:
    """畫面與比較使用重新量過的候選包，存檔候選另留給查證。"""

    result: SchemeResult
    standing: ResultStanding
    stored_candidate: CandidateEvaluation | None


EVALUATOR_VERSION = re.compile(
    r"(aosr\.scoring\.(?:timbre|timbre_channels|listening_area|listening_area_channels|"
    r"channel_matching|reflections|reverberation))"
    r"\.v[0-9]+"
)
RESULT_VERSION = re.compile(r"aosr\.scheme_result\.v([0-9]+)\Z")


def _registry_labels(node: object) -> set[str]:
    """登記簿整份的身分保存在代價設定指紋；找齊各層 payload 的同一格。"""
    if isinstance(node, dict):
        labels = {value for key, value in node.items()
                  if key == "cost_settings_fingerprint" and isinstance(value, str)}
        return labels | set().union(*(_registry_labels(value) for value in node.values()))
    if isinstance(node, list):
        return set().union(*(_registry_labels(value) for value in node))
    return set()


def _fold_measurement(node: object, labels: dict[str, str]) -> object:
    if isinstance(node, dict):
        return {key: _fold_measurement(value, labels) for key, value in node.items()}
    if isinstance(node, list):
        return [_fold_measurement(value, labels) for value in node]
    if isinstance(node, str):
        for label, placeholder in labels.items():
            node = node.replace(label, placeholder)
        return EVALUATOR_VERSION.sub(r"\1.v*", node)
    return node


def _measurement(candidate: CandidateEvaluation) -> str:
    document = candidate.model_dump(mode="json")
    labels = {evaluation.settings_fingerprint: f"<{evaluation.category.value} settings>"
              for evaluation in candidate.evaluations}
    labels.update({label: "<registry>" for label in _registry_labels(document)})
    return json.dumps(_fold_measurement(document, labels), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _same_measurement(a: CandidateEvaluation, b: CandidateEvaluation) -> bool:
    """逐字比量出的內容，設定身分與評估器版尾另由讀回分級處理。

    每類 settings_fingerprint 的值在整份包（包含內嵌 JSON 字串）換成類名佔位字；
    登記簿整份指紋換成 registry 佔位字，七支評估器的 .v數字換成 .v*。
    聆聽區聲道彙總評估器也用同一規則摺疊版尾。
    只變權重、出處、未使用用途或評估器版尾而量出的內容相同，會判成相同；
    數字、狀態、旗標、原因碼與其他指紋任何一格改動都判成不同。
    """
    return _measurement(a) == _measurement(b)


def _check_result_version(document: object) -> None:
    version = document.get("schema_version") if isinstance(document, dict) else None
    if version == RESULT_SCHEMA_VERSION:
        return
    found = RESULT_VERSION.fullmatch(version) if isinstance(version, str) else None
    current = RESULT_VERSION.fullmatch(RESULT_SCHEMA_VERSION)
    if found is not None and current is not None and int(found[1]) < int(current[1]):
        raise ValueError("結果檔格式是舊版（欄位 schema_version），請重算")
    raise ValueError("結果檔格式認不得（欄位 schema_version），請用現在的程式重算")


def load_result(path: Path, *, capabilities: CapabilityTable,
                directivity: DirectivityDefaults, quality_targets_path: Path,
                physics_identity: str) -> LoadedResult:
    """讀入 JSON，重驗報表輸入並由零件重評候選包，再判物理、量法或排名是否改過。"""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    _check_result_version(document)
    result, stored_document = SchemeResult.validate_saved_parts(document)
    for pair in result.pairs:
        load_input_document(pair.input_document, capabilities, directivity)
    if result.physics_identity == physics_identity:
        check_declared_axes(result)
    current_settings = purpose_settings(quality_targets_path, result.scheme.purpose)
    remeasured = reevaluate(result, quality_targets_path=quality_targets_path,
                           capabilities=capabilities, directivity=directivity)
    result = SchemeResult.model_validate({**document, "candidate": remeasured})
    try:
        stored = CandidateEvaluation.model_validate(stored_document)
    except ValidationError:
        stored = None
    if result.physics_identity != physics_identity:
        standing = ResultStanding.NEEDS_PHYSICS
    elif stored is None or (stored != remeasured and not _same_measurement(stored, remeasured)):
        standing = ResultStanding.REMEASURED
    elif result.purpose_settings.fingerprint != current_settings.fingerprint:
        standing = ResultStanding.RERANKED
    else:
        standing = ResultStanding.CURRENT
    return LoadedResult(result, standing, stored)
