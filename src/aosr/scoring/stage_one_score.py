"""第一階段低音篩選分數（票 #558）：把每支喇叭的兩個分項合成一個越小越好的數。

照品質紙第 3 條：目標、較差參考與權重都住品質登記簿，不把有單位的量直接相加——每個分項先走
``cost_shapes`` 的 ``less_is_better``（代價＝分項 dB ÷ 較差參考），再乘權重相加。老闆 2026-10-20 定兩項
有效權重 1:1、而且「兩項各增加 1 dB，對代價的增量應相同」，所以兩個較差參考必須相等，不相等直接報錯
（否則權重填 1:1 也會變成別的比例）。分項名稱用具名集合比：登記簿多一項（例如拖尾）或少一項都拒收，
拖尾不進第一階段的分數。

左右兩支：每支喇叭各算一次，分項取逐聲道等權平均、最差那一支另列提醒、不淘汰（主對話判斷，援引
``docs/decisions/timbre-slot-is-per-channel-mean-ranks-worst-guards.md`` 與
``docs/decisions/listening-area-channels-mean-and-worst-deviations-are-review-alerts.md``）；任一支不可估整個候選
不可估，不拿另一支湊。左右相干疊加今天算不了（報表只存能量），結果標「未評估」。

登記簿條目住第一階段那一段（``QualityTargets.stage_one``），不另開用途、也不放進第二階段的用途：
音色設定指紋（``timbre.py`` 的 ``fingerprint=registry.fingerprint``）、排名表頭（``ranking.py`` 的 ``_read_rules``）
與聲道匹配（``channel_matching_settings.py`` 的 ``registry_fingerprint``）都抄整份登記簿指紋，另開一個用途照樣會
動到第二階段的比較身分；用途本身是房間的使用情境，拿來裝階段也不對題。這一支的設定指紋只雜湊自己讀的
那幾格（照 ``reflections._settings``），每次搜尋固定、不隨當批候選分布改。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from aosr.config.quality_targets import EntryStatus, QualityTargets, SettingEntry, TargetEntry
from aosr.scoring.contract_base import FrequencyRange
from aosr.scoring.contract_base import FrozenModel as _FrozenModel
from aosr.scoring.cost_shapes import shape_cost, target, weight_table
from aosr.scoring.recommendation import NotFinalReason, not_final_reasons
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import (
    IdealSourceAssumption,
    IdealSourceAssumptionText,
    StageOneLowFrequencyEvaluation,
    StageOneLowFrequencyInput,
    StageOneReasonCode,
    StageOneSpeakerPayload,
)


STAGE_ONE_SCORE_VERSION: Final = "aosr.scoring.stage_one_score.v1"
_PREFIX: Final[str] = "stage_one_low_frequency."
SCORED_RANGE_KEY: Final[str] = _PREFIX + "scored_range_hz"
WEIGHTS_KEY: Final[str] = _PREFIX + "component_weights"
# 進分數的兩個分項：名字就是 PR-1 payload 的欄位名，也是登記簿目標鍵的尾巴與權重表的列名。
ComponentName = Literal["own_line_ripple_rms_db", "seat_robustness_rms_db"]
PRINCIPAL: Final[ComponentName] = "own_line_ripple_rms_db"
ROBUSTNESS: Final[ComponentName] = "seat_robustness_rms_db"
COMPONENT_NAMES: Final[frozenset[str]] = frozenset({PRINCIPAL, ROBUSTNESS})
# 這一支讀的登記簿條目（第一階段那一段）；指紋只收這幾格。
SETTING_KEYS: Final[tuple[str, ...]] = (
    SCORED_RANGE_KEY, _PREFIX + PRINCIPAL, _PREFIX + ROBUSTNESS, WEIGHTS_KEY,
)
COHERENT_SUM_NOT_EVALUATED: Final = "left_right_coherent_sum_not_evaluated"
COHERENT_SUM_NOT_EVALUATED_TEXT: Final = "左右相干疊加未評估：報表只存能量，左右兩支各自計分後等權平均"
CoherentSumNotEvaluated = Literal["left_right_coherent_sum_not_evaluated"]
CoherentSumNotEvaluatedText = Literal["左右相干疊加未評估：報表只存能量，左右兩支各自計分後等權平均"]


@dataclass(frozen=True)
class StageOneScoreSettings:
    """從登記簿第一階段那一段讀出的計分範圍、兩個目標與權重；``status`` 是讀到的格子裡最低的狀態。

    形狀由這一型自己守，讀登記簿與讀回結果檔都經過這裡：目標與權重各剛好兩個具名分項、各一次，
    兩個目標都是 ``less_is_better`` 而且較差參考相等，計分範圍遞增為正。
    """

    purpose: str
    scored_range_hz: tuple[float, float]
    targets: tuple[tuple[ComponentName, TargetEntry], ...]
    weights: tuple[tuple[ComponentName, float], ...]
    status: EntryStatus
    fingerprint: str

    def __post_init__(self) -> None:
        for part, names in (("目標", [name for name, _ in self.targets]),
                            ("權重", [name for name, _ in self.weights])):
            if len(names) != len(COMPONENT_NAMES) or set(names) != COMPONENT_NAMES:
                raise ValueError(f"計分設定的{part}必須剛好是 {sorted(COMPONENT_NAMES)} 各一次，這裡是 {names}")
        for _, entry in self.targets:
            if entry.cost_shape != "less_is_better":
                raise ValueError(f"{entry.key} 的代價形狀必須是 less_is_better")
        if len({entry.worse_reference for _, entry in self.targets}) != 1:
            raise ValueError(
                "兩個分項的較差參考必須相等：較差參考只為無單位化，不相等時權重的比例就不是登記的那個比例"
            )
        lower, upper = self.scored_range_hz
        if not 0.0 < lower < upper:
            raise ValueError(f"計分範圍必須是遞增的正頻率範圍，這裡是 {self.scored_range_hz}")

    def target(self, name: ComponentName) -> TargetEntry:
        return dict(self.targets)[name]

    def weight(self, name: ComponentName) -> float:
        return dict(self.weights)[name]


def _scored_range(entry: SettingEntry) -> tuple[float, float]:
    if entry.unit != "Hz" or not isinstance(entry.value, tuple) or len(entry.value) != 2:
        raise ValueError(f"{entry.key} 必須是兩個端點的 Hz 範圍")
    return float(entry.value[0]), float(entry.value[1])


def stage_one_score_settings(registry: QualityTargets, purpose_name: str) -> StageOneScoreSettings:
    """讀第一階段那一段；權重表的分項不等於兩個具名分項報錯，其餘形狀由 ``StageOneScoreSettings`` 守。"""
    section = registry.stage_one(purpose_name)
    scored = section.entry(SCORED_RANGE_KEY)
    if not isinstance(scored, SettingEntry):
        raise TypeError(f"{SCORED_RANGE_KEY} 不是量法設定")
    names: tuple[ComponentName, ...] = (PRINCIPAL, ROBUSTNESS)
    targets = tuple((name, target(section, _PREFIX + name, "dB")) for name in names)
    table = weight_table(section, WEIGHTS_KEY)
    rows = {item.name: item for item in table.item}
    if set(rows) != COMPONENT_NAMES:
        raise ValueError(f"{WEIGHTS_KEY} 的分項必須剛好是 {sorted(COMPONENT_NAMES)}，登記簿寫 {sorted(rows)}")
    weights = tuple((name, rows[name].value) for name in names)
    statuses = (scored.status, *(entry.status for _, entry in targets), *(rows[name].status for name in names))
    bounds = _scored_range(scored)
    canonical = json.dumps({
        "score_version": STAGE_ONE_SCORE_VERSION,
        "scored_range_hz": bounds,
        "targets": {name: {"value": entry.value, "unit": entry.unit, "cost_shape": entry.cost_shape,
                           "worse_reference": entry.worse_reference, "status": entry.status}
                    for name, entry in targets},
        "weights": {name: {"value": rows[name].value, "status": rows[name].status} for name in names},
        "scored_range_status": scored.status,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return StageOneScoreSettings(
        purpose=purpose_name, scored_range_hz=bounds, targets=targets, weights=weights,
        status="baseline" if "baseline" in statuses else "calibrated",
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


class StageOneComponent(_FrozenModel):
    """一個分項：原始 dB、較差參考、權重與它在合計裡的那一份（權重 × dB ÷ 較差參考）。"""

    name: ComponentName
    value_db: Annotated[float, Field(ge=0.0)]
    worse_reference_db: Annotated[float, Field(gt=0.0)]
    weight: Annotated[float, Field(ge=0.0)]
    weighted_cost: Annotated[float, Field(ge=0.0)]


class StageOneChannelScore(_FrozenModel):
    """一支喇叭：scored（已計分）帶兩個分項與小計；unavailable（不可估）只帶原因。"""

    speaker_id: str = Field(min_length=1)
    state: Literal["scored", "unavailable"]
    components: tuple[StageOneComponent, ...]
    screening_subtotal: Annotated[float, Field(ge=0.0)] | None
    reason_codes: tuple[StageOneReasonCode, ...]

    @model_validator(mode="after")
    def _state_matches_content(self) -> Self:
        scored = self.state == "scored"
        if scored != (self.screening_subtotal is not None) or scored == bool(self.reason_codes):
            raise ValueError("scored 必帶小計且無原因；unavailable 無小計且必帶原因")
        if scored != bool(self.components):
            raise ValueError("scored 必帶分項；unavailable 不帶")
        return self


class WorstChannel(_FrozenModel):
    """小計最大的那一支；只提醒，不淘汰、不改合計。"""

    speaker_id: str = Field(min_length=1)
    screening_subtotal: Annotated[float, Field(ge=0.0)]


class StageOneScore(_FrozenModel):
    """一個候選的第一階段篩選分數（越小越好）；不是最終推薦，採用前要回到完整評估。

    ``components`` 是兩個分項的逐聲道平均（原始 dB）與它們在合計裡的份量；``principal_only_sort_key_db``
    是只看主項排序用的鍵（主項逐聲道平均，dB）。逐聲道結果照留，假設標示照評估器那一份抄。
    """

    score_version: Literal["aosr.scoring.stage_one_score.v1"]
    score_kind: Literal["screening"]
    candidate_id: str = Field(min_length=1)
    state: Literal["scored", "unavailable"]
    components: tuple[StageOneComponent, ...]
    screening_total: Annotated[float, Field(ge=0.0)] | None
    principal_only_sort_key_db: Annotated[float, Field(ge=0.0)] | None
    channels: tuple[StageOneChannelScore, ...] = Field(min_length=1)
    worst_channel: WorstChannel | None
    reason_codes: tuple[StageOneReasonCode, ...]
    scored_range_hz: FrequencyRange
    settings_status: EntryStatus
    settings_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    not_final_reasons: tuple[NotFinalReason, ...] = Field(min_length=1)
    channel_summation: CoherentSumNotEvaluated
    channel_summation_text: CoherentSumNotEvaluatedText
    assumption: IdealSourceAssumption
    assumption_text: IdealSourceAssumptionText

    @model_validator(mode="after")
    def _state_matches_content(self) -> Self:
        scored = self.state == "scored"
        filled = (self.screening_total is not None, self.principal_only_sort_key_db is not None,
                  self.worst_channel is not None, bool(self.components))
        if any(item != scored for item in filled) or scored == bool(self.reason_codes):
            raise ValueError("scored 必帶合計、主項鍵、最差聲道與分項且無原因；unavailable 反過來")
        if scored and {item.name for item in self.components} != COMPONENT_NAMES:
            raise ValueError("分項必須剛好是兩個具名分項")
        return self


def _components(values: dict[ComponentName, float], settings: StageOneScoreSettings) -> tuple[StageOneComponent, ...]:
    found = []
    for name, weight in settings.weights:
        entry = settings.target(name)
        found.append(StageOneComponent(
            name=name, value_db=values[name], worse_reference_db=entry.worse_reference, weight=weight,
            weighted_cost=weight * shape_cost(entry, (values[name],)),
        ))
    return tuple(found)


def _total(components: tuple[StageOneComponent, ...]) -> float:
    return math.fsum(item.weighted_cost for item in components)


def _values(payload: StageOneSpeakerPayload) -> dict[ComponentName, float]:
    return {PRINCIPAL: payload.own_line_ripple_rms_db, ROBUSTNESS: payload.seat_robustness_rms_db}


def score_stage_one(
    evaluation: StageOneLowFrequencyEvaluation, settings: StageOneScoreSettings
) -> StageOneScore:
    """把第一階段量測合成篩選分數；評估器用的計分範圍必須等於登記簿那一格。"""
    if tuple(evaluation.scoring_range_hz) != settings.scored_range_hz:
        raise ValueError(
            f"評估用的計分範圍 {evaluation.scoring_range_hz} 跟登記簿 {settings.scored_range_hz} 不同"
        )
    channels: list[StageOneChannelScore] = []
    payloads: list[StageOneSpeakerPayload] = []
    for speaker in evaluation.speakers:
        if speaker.payload is None:
            channels.append(StageOneChannelScore(
                speaker_id=speaker.speaker_id, state="unavailable", components=(),
                screening_subtotal=None, reason_codes=speaker.reason_codes,
            ))
            continue
        payloads.append(speaker.payload)
        own = _components(_values(speaker.payload), settings)
        channels.append(StageOneChannelScore(
            speaker_id=speaker.speaker_id, state="scored", components=own,
            screening_subtotal=_total(own), reason_codes=(),
        ))
    reasons = tuple(code for code in StageOneReasonCode
                    if any(code in channel.reason_codes for channel in channels))
    common = {
        "score_version": STAGE_ONE_SCORE_VERSION, "score_kind": "screening",
        "candidate_id": evaluation.candidate_id, "channels": tuple(channels),
        "scored_range_hz": settings.scored_range_hz, "settings_status": settings.status,
        "settings_fingerprint": settings.fingerprint,
        "not_final_reasons": not_final_reasons((), False, settings.status),
        "channel_summation": COHERENT_SUM_NOT_EVALUATED,
        "channel_summation_text": COHERENT_SUM_NOT_EVALUATED_TEXT,
        "assumption": evaluation.assumption, "assumption_text": evaluation.assumption_text,
    }
    if reasons:
        return StageOneScore.model_validate({
            **common, "state": "unavailable", "components": (), "screening_total": None,
            "principal_only_sort_key_db": None, "worst_channel": None, "reason_codes": reasons,
        })
    # 逐聲道等權平均：每支喇叭一樣重，不照聲道角色加權。
    mean = {name: math.fsum(_values(payload)[name] for payload in payloads) / len(payloads)
            for name in (PRINCIPAL, ROBUSTNESS)}
    components = _components(mean, settings)
    worst = max(channels, key=lambda channel: channel.screening_subtotal or 0.0)
    return StageOneScore.model_validate({
        **common, "state": "scored", "components": components, "screening_total": _total(components),
        "principal_only_sort_key_db": mean[PRINCIPAL],
        "worst_channel": WorstChannel(speaker_id=worst.speaker_id,
                                      screening_subtotal=worst.screening_subtotal or 0.0),
        "reason_codes": (),
    })


def score_stage_one_input(data: StageOneLowFrequencyInput, settings: StageOneScoreSettings) -> StageOneScore:
    """用登記簿的計分範圍跑第一階段量測，再合分；只讀輸入。"""
    return score_stage_one(
        evaluate_stage_one_low_frequency(data, scoring_range_hz=settings.scored_range_hz), settings
    )
