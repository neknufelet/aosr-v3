"""聲道匹配量法設定讀取；由原評估器原文拆出。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aosr.config.quality_targets import QualityPurpose, SettingEntry, Unit, load_quality_targets

_PREFIX: Final[str] = "channel_matching."
_BROADBAND_KEY: Final[str] = _PREFIX + "broadband_range_hz"
_SMOOTHING_KEY: Final[str] = "timbre_balance.smoothing_width_octave_ripple"
_SWITCH_KEY: Final[str] = _PREFIX + "direct_time_cost_enabled"
_SETTING_UNITS: Final[dict[str, Unit]] = {
    _BROADBAND_KEY: "Hz",
    _SMOOTHING_KEY: "oct",
    _SWITCH_KEY: "1",
}
@dataclass(frozen=True)
class _Settings:
    broadband_range_hz: tuple[float, float]
    smoothing_width_octave: float
    direct_time_cost_enabled: bool
    any_baseline: bool
    registry_fingerprint: str


def _setting(
    purpose: QualityPurpose, key: str, expected_unit: Unit
) -> SettingEntry:
    entry = purpose.entry(key)
    if not isinstance(entry, SettingEntry):
        raise TypeError(f"{key} 不是量法設定")
    if entry.unit != expected_unit:
        raise ValueError(
            f"{key} 單位應為 {expected_unit}，登記簿寫 {entry.unit}"
        )
    return entry


def _number(entry: SettingEntry) -> float:
    if isinstance(entry.value, tuple):
        raise TypeError(f"{entry.key} 必須是單一數值")
    return float(entry.value)


def _range(entry: SettingEntry) -> tuple[float, float]:
    if not isinstance(entry.value, tuple) or len(entry.value) != 2:
        raise TypeError(f"{entry.key} 必須是兩個數的範圍")
    lower, upper = (float(value) for value in entry.value)
    if not 0.0 < lower < upper:
        raise ValueError(f"{entry.key} 必須是遞增正值範圍")
    return lower, upper


def _load_settings(path: str | Path, purpose_name: str) -> _Settings:
    registry = load_quality_targets(path)
    purpose = registry.purpose(purpose_name)
    broadband = _setting(purpose, _BROADBAND_KEY, _SETTING_UNITS[_BROADBAND_KEY])
    smoothing = _setting(purpose, _SMOOTHING_KEY, _SETTING_UNITS[_SMOOTHING_KEY])
    switch = _setting(purpose, _SWITCH_KEY, _SETTING_UNITS[_SWITCH_KEY])
    if switch.value not in (0, 1):
        raise ValueError(f"{switch.key} 必須是 0 或 1")
    # 左右差異曲線用的是音色那一格「起伏的平滑寬度」：0＝看原始曲線（票 #432，老闆拍「用原始檔」），負的紅。
    if _number(smoothing) < 0.0:
        raise ValueError(f"{smoothing.key} 不准是負的")
    used = (broadband, smoothing, switch)
    return _Settings(
        broadband_range_hz=_range(broadband),
        smoothing_width_octave=_number(smoothing),
        direct_time_cost_enabled=bool(switch.value),
        any_baseline=any(item.status == "baseline" for item in used),
        registry_fingerprint=registry.fingerprint,
    )


