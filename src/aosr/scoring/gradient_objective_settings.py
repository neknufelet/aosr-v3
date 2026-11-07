"""第三階段梯度目標的登記簿讀法（#561 壬一）：第一階段兩分項的設定，加上專用計分範圍與大峰代理的溫度、目標、權重。

老闆 2026-10-30 拍 B：大峰的平滑代理進梯度目標當試驗；#1032 再加專用計分範圍。四格住品質登記簿第一階段那一段、鍵前綴
``stage_three_gradient.``；只有梯度目標讀它們——第一階段合分（``stage_one_score``）與第二階段低頻頻響類只讀
``stage_one_low_frequency.*``，兩邊的設定指紋都不含這四格，加這四格不動第一階段名次與第二階段的比較身分。
#1032：第三階段以專用範圍重建兩分項設定與指紋；第一階段原設定不變。

這一支只讀具名的四個鍵、不列舉整段；指紋只雜湊自己讀的四格與第一階段設定的指紋（照 ``stage_one_score``
「指紋只收自己讀的那幾格」）。代理的較差參考必須等於第一階段兩分項那個值：較差參考只為無單位化，
不相等時權重就不是登記的那個比例（同 ``StageOneScoreSettings`` 守兩分項相等的理由）。
程式裡不寫任何溫度、權重或較差參考的數字。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final

from aosr.config.quality_targets import EntryStatus, QualityPurpose, QualityTargets, SettingEntry, TargetEntry
from aosr.scoring.cost_shapes import target, weight_table
from aosr.scoring.stage_one_score import (
    PRINCIPAL, SCORED_RANGE_KEY as STAGE_ONE_RANGE_KEY, StageOneScoreSettings, stage_one_settings_from_section,
)

GRADIENT_OBJECTIVE_VERSION: Final = "aosr.scoring.gradient_objective.v1"
_PREFIX: Final[str] = "stage_three_gradient."
PROXY: Final = "big_peak_proxy_db"
TEMPERATURE_KEY: Final[str] = _PREFIX + "big_peak_proxy_temperature_db"
PROXY_TARGET_KEY: Final[str] = _PREFIX + PROXY
WEIGHTS_KEY: Final[str] = _PREFIX + "component_weights"
SCORED_RANGE_KEY: Final[str] = _PREFIX + "scored_range_hz"
# 這一支讀的登記簿條目（四格）；第一階段那幾格由 ``stage_one_score.SETTING_KEYS`` 管。
SETTING_KEYS: Final[tuple[str, ...]] = (TEMPERATURE_KEY, PROXY_TARGET_KEY, WEIGHTS_KEY, SCORED_RANGE_KEY)


@dataclass(frozen=True)
class GradientObjectiveSettings:
    """梯度目標的設定：第一階段兩分項那一份、代理的溫度（dB）、代理的目標條目、代理的權重；
    ``status`` 是四格與第一階段裡最低的狀態，``fingerprint`` 雜湊四格與第一階段設定的指紋。"""

    stage_one: StageOneScoreSettings
    temperature_db: float
    proxy_target: TargetEntry
    proxy_weight: float
    status: EntryStatus
    fingerprint: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature_db) and self.temperature_db > 0.0):
            raise ValueError(f"{TEMPERATURE_KEY} 必須是有限正數（軟最大值的溫度），這裡是 {self.temperature_db}")
        if not (math.isfinite(self.proxy_weight) and self.proxy_weight >= 0.0):
            raise ValueError(f"{WEIGHTS_KEY} 的 {PROXY} 必須是有限非負數，這裡是 {self.proxy_weight}")
        if self.proxy_target.cost_shape != "less_is_better":
            raise ValueError(f"{PROXY_TARGET_KEY} 的代價形狀必須是 less_is_better")
        if self.proxy_target.worse_reference != self.stage_one.target(PRINCIPAL).worse_reference:
            raise ValueError(
                f"{PROXY_TARGET_KEY} 的較差參考必須等於第一階段兩分項的較差參考：較差參考只為無單位化，"
                "不相等時權重的比例就不是登記的那個比例"
            )

    @property
    def worse_reference(self) -> float:
        return self.proxy_target.worse_reference


def _temperature(entry: object) -> tuple[SettingEntry, float]:
    if not isinstance(entry, SettingEntry):
        raise TypeError(f"{TEMPERATURE_KEY} 不是量法設定")
    if entry.unit != "dB" or isinstance(entry.value, tuple):
        raise ValueError(f"{TEMPERATURE_KEY} 必須是單一個 dB 數值")
    return entry, float(entry.value)


def gradient_objective_settings(registry: QualityTargets, purpose_name: str) -> GradientObjectiveSettings:
    """讀第一階段那一段：第一階段兩分項照 ``stage_one_settings_from_section``，再讀四個具名鍵。"""
    return gradient_objective_settings_from_section(registry.stage_one(purpose_name))


def gradient_objective_settings_from_section(section: QualityPurpose) -> GradientObjectiveSettings:
    """同一份讀法，從已取出的第一階段那一段讀；權重表的列不是剛好代理那一列就報錯。"""
    original = stage_one_settings_from_section(section)
    scored = section.entry(SCORED_RANGE_KEY)
    if not isinstance(scored, SettingEntry):
        raise TypeError(f"{SCORED_RANGE_KEY} 不是量法設定")
    if scored.unit != "Hz" or not isinstance(scored.value, tuple) or len(scored.value) != 2:
        raise ValueError(f"{SCORED_RANGE_KEY} 必須是兩個端點的 Hz 範圍")
    lower, upper = scored.value
    if not all(math.isfinite(value) for value in scored.value) or not 0.0 < lower < upper:
        raise ValueError("第三階段計分範圍必須是遞增的有限正頻率範圍")
    if lower < original.scored_range_hz[0]:
        raise ValueError("第三階段計分範圍下界不得低於第一階段下界")
    # 交既有讀法重建兩分項的設定與指紋，不把舊指紋貼在新範圍上；原 section 不變。
    effective = section.model_copy(update={"setting": tuple(
        scored.model_copy(update={"key": STAGE_ONE_RANGE_KEY}) if entry.key == STAGE_ONE_RANGE_KEY else entry
        for entry in section.setting)})
    stage_one = stage_one_settings_from_section(effective)
    temperature, temperature_db = _temperature(section.entry(TEMPERATURE_KEY))
    proxy = target(section, PROXY_TARGET_KEY, "dB")
    table = weight_table(section, WEIGHTS_KEY)
    names = [item.name for item in table.item]
    if names != [PROXY]:
        raise ValueError(f"{WEIGHTS_KEY} 的分項必須剛好是 [{PROXY!r}]，登記簿寫 {names}")
    row = table.item[0]
    statuses = (stage_one.status, temperature.status, proxy.status, row.status)
    canonical = json.dumps({
        "objective_version": GRADIENT_OBJECTIVE_VERSION,
        "stage_one_fingerprint": stage_one.fingerprint,
        "scored_range": {"key": scored.key, "value": scored.value, "unit": scored.unit, "status": scored.status},
        "temperature": {"value": temperature.value, "unit": temperature.unit, "status": temperature.status},
        "proxy_target": {"value": proxy.value, "unit": proxy.unit, "cost_shape": proxy.cost_shape,
                         "worse_reference": proxy.worse_reference, "status": proxy.status},
        "proxy_weight": {"value": row.value, "status": row.status},
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return GradientObjectiveSettings(
        stage_one=stage_one, temperature_db=temperature_db, proxy_target=proxy,
        proxy_weight=float(row.value), status="baseline" if "baseline" in statuses else "calibrated",
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
