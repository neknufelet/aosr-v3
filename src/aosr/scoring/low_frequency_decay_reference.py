"""低頻拖尾（第③類）的參考曲線：給一個共振頻率，回這個用途的參考 T60 與那個值從哪裡來。

票 #669（老闆 2026-10-06 回「A」）；決策紙 docs/decisions/low-frequency-decay-reference-curve.md。
數字只住品質登記簿，這裡只讀、驗形狀、內插。它只是「從哪裡開始算超出」的工程目標，不是單一共振的
絕對可聽界線：低於它不保證聽不出，高於它也不直接等於音質變差。T60 是單一共振衰減 60 dB 的秒數，
跟方法紙鎖的 3 ln 10／Im ω 同一個量。代價形狀另題，這支還沒接進任何評估器或排名；接的時候那一類要宣告
evaluator_keys（#633）。

- 目標線：登記的頻率點本身是文獻錨點；錨點之間 T60 對 log f 走直線（工程內插，原文沒有內插規則）；
  延伸範圍內用登記的延伸值（工程延伸；含下端、不含上端，上端就是最低錨點）；其餘頻率未評估，不外推。
- 參考線（人工刺激那條）：只有錨點與內插、沒有延伸；只印在模態表上，不進分數。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from aosr.config.quality_targets import QualityPurpose, SettingEntry, Unit

_PREFIX: Final[str] = "low_frequency_decay."
_ANCHOR_FREQUENCIES_KEY: Final[str] = _PREFIX + "reference_anchor_frequencies_hz"
_ANCHOR_T60_KEY: Final[str] = _PREFIX + "reference_anchor_t60_s"
_EXTENSION_RANGE_KEY: Final[str] = _PREFIX + "reference_extension_range_hz"
_EXTENSION_T60_KEY: Final[str] = _PREFIX + "reference_extension_t60_s"
_STRICT_FREQUENCIES_KEY: Final[str] = _PREFIX + "strict_reference_frequencies_hz"
_STRICT_T60_KEY: Final[str] = _PREFIX + "strict_reference_t60_s"
_SETTING_UNITS: Final[dict[str, Unit]] = {
    _ANCHOR_FREQUENCIES_KEY: "Hz",
    _ANCHOR_T60_KEY: "s",
    _EXTENSION_RANGE_KEY: "Hz",
    _EXTENSION_T60_KEY: "s",
    _STRICT_FREQUENCIES_KEY: "Hz",
    _STRICT_T60_KEY: "s",
}
# 這支讀的鍵；接進評分時由那一類的 evaluator_keys 宣告依賴（#633）。
SETTING_KEYS: Final[tuple[str, ...]] = tuple(_SETTING_UNITS)


class ReferenceOrigin(StrEnum):
    """參考值的來源種類；兩種工程規則不會因為求解器標了 validated 就變成聽感已驗證。"""

    LITERATURE_ANCHOR = "literature_anchor"
    ENGINEERING_INTERPOLATION = "engineering_interpolation"
    ENGINEERING_EXTENSION = "engineering_extension"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class ReferenceT60:
    """一個頻率的參考值；未評估時 t60_s 是 None——未評估不是通過。"""

    frequency_hz: float
    t60_s: float | None
    origin: ReferenceOrigin


@dataclass(frozen=True)
class ReferenceLine:
    """一條參考線：同索引的錨點頻率與 T60，加上可有可無的延伸段（下端 Hz、上端 Hz、T60 秒）。"""

    frequencies_hz: tuple[float, ...]
    t60_s: tuple[float, ...]
    extension: tuple[float, float, float] | None = None

    def at(self, frequency_hz: float) -> ReferenceT60:
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError(f"頻率必須是有限正數，收到 {frequency_hz!r}")
        if frequency_hz in self.frequencies_hz:
            index = self.frequencies_hz.index(frequency_hz)
            return ReferenceT60(frequency_hz, self.t60_s[index], ReferenceOrigin.LITERATURE_ANCHOR)
        if self.extension is not None and self.extension[0] <= frequency_hz < self.extension[1]:
            return ReferenceT60(frequency_hz, self.extension[2], ReferenceOrigin.ENGINEERING_EXTENSION)
        segments = zip(self.frequencies_hz, self.t60_s, self.frequencies_hz[1:], self.t60_s[1:])
        for low_hz, low_s, high_hz, high_s in segments:
            if low_hz < frequency_hz < high_hz:
                share = math.log(frequency_hz / low_hz) / math.log(high_hz / low_hz)
                t60 = low_s + share * (high_s - low_s)
                return ReferenceT60(frequency_hz, t60, ReferenceOrigin.ENGINEERING_INTERPOLATION)
        return ReferenceT60(frequency_hz, None, ReferenceOrigin.NOT_ASSESSED)


@dataclass(frozen=True)
class LowFrequencyDecayReference:
    """這個用途的目標線（音樂線＋延伸）與只當參考的人工線。"""

    target: ReferenceLine
    strict: ReferenceLine

    def target_at(self, frequency_hz: float) -> ReferenceT60:
        return self.target.at(frequency_hz)

    def strict_at(self, frequency_hz: float) -> ReferenceT60:
        return self.strict.at(frequency_hz)


def _setting(purpose: QualityPurpose, key: str) -> SettingEntry:
    entry = purpose.entry(key)
    if not isinstance(entry, SettingEntry):
        raise TypeError(f"{key} 不是量法設定")
    if entry.unit != _SETTING_UNITS[key]:
        raise ValueError(f"{key} 單位應為 {_SETTING_UNITS[key]}，登記簿寫 {entry.unit}")
    return entry


def _numbers(purpose: QualityPurpose, key: str) -> tuple[float, ...]:
    entry = _setting(purpose, key)
    if not isinstance(entry.value, tuple):
        raise TypeError(f"{key} 必須是數值清單")
    return tuple(float(value) for value in entry.value)


def _number(purpose: QualityPurpose, key: str) -> float:
    entry = _setting(purpose, key)
    if isinstance(entry.value, tuple):
        raise TypeError(f"{key} 必須是單一數值")
    return float(entry.value)


def _anchors(purpose: QualityPurpose, frequencies_key: str, t60_key: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    frequencies = _numbers(purpose, frequencies_key)
    t60 = _numbers(purpose, t60_key)
    if not frequencies or len(frequencies) != len(t60):
        raise ValueError(f"{frequencies_key} 與 {t60_key} 必須非空且等長")
    if any(value <= 0.0 for value in frequencies):
        raise ValueError(f"{frequencies_key} 必須為正")
    if any(high <= low for low, high in zip(frequencies, frequencies[1:])):
        raise ValueError(f"{frequencies_key} 必須嚴格遞增")
    if any(value <= 0.0 for value in t60):
        raise ValueError(f"{t60_key} 必須為正")
    return frequencies, t60


def load_reference(purpose: QualityPurpose) -> LowFrequencyDecayReference:
    """由用途設定讀兩條線；形狀不對就丟錯，不猜。"""
    frequencies, t60 = _anchors(purpose, _ANCHOR_FREQUENCIES_KEY, _ANCHOR_T60_KEY)
    extension_range = _numbers(purpose, _EXTENSION_RANGE_KEY)
    extension_t60 = _number(purpose, _EXTENSION_T60_KEY)
    if len(extension_range) != 2 or not 0.0 < extension_range[0] < extension_range[1]:
        raise ValueError(f"{_EXTENSION_RANGE_KEY} 必須是兩個遞增的正數")
    if extension_range[1] != frequencies[0]:
        raise ValueError(f"{_EXTENSION_RANGE_KEY} 的上端必須等於最低的錨點頻率 {frequencies[0]}")
    if extension_t60 <= 0.0:
        raise ValueError(f"{_EXTENSION_T60_KEY} 必須為正")
    strict_frequencies, strict_t60 = _anchors(purpose, _STRICT_FREQUENCIES_KEY, _STRICT_T60_KEY)
    return LowFrequencyDecayReference(
        target=ReferenceLine(frequencies, t60, (extension_range[0], extension_range[1], extension_t60)),
        strict=ReferenceLine(strict_frequencies, strict_t60),
    )
