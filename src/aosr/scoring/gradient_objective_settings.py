"""第三階段梯度目標的登記簿讀法（#561 壬一）：第一階段兩分項的設定，加上大峰代理的溫度、目標與權重三格。

老闆 2026-10-30 拍 B：大峰的平滑代理進梯度目標當試驗。三格住品質登記簿第一階段那一段、鍵前綴
``stage_three_gradient.``；只有梯度目標讀它們——第一階段合分（``stage_one_score``）與第二階段低頻頻響類只讀
``stage_one_low_frequency.*``，兩邊的設定指紋都不含這三格，加這三格不動第一階段名次與第二階段的比較身分。

這一支只讀具名的三個鍵、不列舉整段；指紋只雜湊自己讀的三格與第一階段設定的指紋（照 ``stage_one_score``
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
    PRINCIPAL, StageOneScoreSettings, stage_one_settings_from_section,
)

GRADIENT_OBJECTIVE_VERSION: Final = "aosr.scoring.gradient_objective.v1"
_PREFIX: Final[str] = "stage_three_gradient."
PROXY: Final = "big_peak_proxy_db"
TEMPERATURE_KEY: Final[str] = _PREFIX + "big_peak_proxy_temperature_db"
PROXY_TARGET_KEY: Final[str] = _PREFIX + PROXY
WEIGHTS_KEY: Final[str] = _PREFIX + "component_weights"
# 這一支讀的登記簿條目（三格）；第一階段那幾格由 ``stage_one_score.SETTING_KEYS`` 管。
SETTING_KEYS: Final[tuple[str, ...]] = (TEMPERATURE_KEY, PROXY_TARGET_KEY, WEIGHTS_KEY)


@dataclass(frozen=True)
class GradientObjectiveSettings:
    """梯度目標的設定：第一階段兩分項那一份、代理的溫度（dB）、代理的目標條目、代理的權重；
    ``status`` 是三格與第一階段裡最低的狀態，``fingerprint`` 雜湊三格與第一階段設定的指紋。"""

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
    """讀第一階段那一段：第一階段兩分項照 ``stage_one_settings_from_section``，再讀三個具名鍵。"""
    return gradient_objective_settings_from_section(registry.stage_one(purpose_name))


def gradient_objective_settings_from_section(section: QualityPurpose) -> GradientObjectiveSettings:
    """同一份讀法，從已取出的第一階段那一段讀；權重表的列不是剛好代理那一列就報錯。"""
    stage_one = stage_one_settings_from_section(section)
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
