"""排名類別註冊表：代價、資料資格與登記簿來源的唯一接線點。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType
from typing import Final, cast

from aosr.config.quality_targets import EntryStatus, QualityPurpose
from aosr.scoring import (
    channel_matching_cost,
    listening_area_cost,
    reverberation_cost,
    timbre_cost,
)
from aosr.scoring.contract import CategoryEvaluation, QualityCategory


CategoryCoster = Callable[[CategoryEvaluation, QualityPurpose, str], CategoryEvaluation]
EligibilityChecker = Callable[[CategoryEvaluation, QualityPurpose], tuple[StrEnum, ...]]
RegistrySourceCollector = Callable[
    [QualityPurpose], tuple[tuple[str, EntryStatus], ...]
]
FloorChecker = Callable[[CategoryEvaluation, QualityPurpose], tuple[str, ...]]


class EliminationReason(StrEnum):
    """踩到硬底線的受控原因代碼；一個候選踩幾條就列幾條。"""

    TIMBRE_PEAK_BEYOND_LIMIT = "timbre_peak_beyond_limit"
    TIMBRE_DIP_BEYOND_LIMIT = "timbre_dip_beyond_limit"
    LISTENING_AREA_TILT_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_tilt_primary_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_TILT_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_tilt_surrounding_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_RIPPLE_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_ripple_primary_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_RIPPLE_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_ripple_surrounding_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_LEVEL_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_level_primary_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_LEVEL_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_level_surrounding_to_surrounding_worst_beyond_limit"
    )
    CHANNEL_MATCHING_TILT_WORST_BEYOND_LIMIT = (
        "channel_matching_tilt_worst_beyond_limit"
    )
    CHANNEL_MATCHING_RIPPLE_WORST_BEYOND_LIMIT = (
        "channel_matching_ripple_worst_beyond_limit"
    )
    CHANNEL_MATCHING_LEVEL_WORST_BEYOND_LIMIT = (
        "channel_matching_level_worst_beyond_limit"
    )
    CHANNEL_MATCHING_DIRECT_TIME_WORST_BEYOND_LIMIT = (
        "channel_matching_direct_time_worst_beyond_limit"
    )
    EXTERNAL_FLOOR_FAILED = "external_floor_failed"


@dataclass(frozen=True)
class CategoryRegistration:
    """一類接到排名層的代價、資格、來源與底線行為。"""

    coster: CategoryCoster
    eligibility_reasons: EligibilityChecker
    eligibility_keys: tuple[str, ...]
    registry_sources: RegistrySourceCollector
    floor_reasons: FloorChecker


def _no_eligibility(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> tuple[StrEnum, ...]:
    del evaluation, purpose
    return ()


def _no_floor_reasons(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> tuple[str, ...]:
    del evaluation, purpose
    return ()


def _registration(module: ModuleType) -> CategoryRegistration:
    """依共同名字讀一個類別模組；沒有的資格規則與底線保護可以省略。"""
    return CategoryRegistration(
        coster=cast(CategoryCoster, module.cost_evaluation),
        eligibility_reasons=cast(
            EligibilityChecker,
            getattr(module, "eligibility_reasons", _no_eligibility),
        ),
        eligibility_keys=cast(tuple[str, ...], getattr(module, "eligibility_keys", ())),
        registry_sources=cast(RegistrySourceCollector, module.registry_sources),
        floor_reasons=cast(
            FloorChecker, getattr(module, "floor_reasons", _no_floor_reasons)
        ),
    )


# 新增一類只在這張表加一行；類別模組依上面的共同名字提供自己的行為。
# **直接 import，不用動態取模組**：字串式的 import 是分層規矩的逃生門
# （layers-import-downward-only 那張卡明文擋），靜態上看不出引到哪一層。
_CATEGORY_MODULES: Final[dict[QualityCategory, ModuleType]] = {
    QualityCategory.TIMBRE_BALANCE: timbre_cost,
    QualityCategory.LISTENING_AREA_STABILITY: listening_area_cost,
    QualityCategory.REVERBERATION: reverberation_cost,
    QualityCategory.CHANNEL_MATCHING: channel_matching_cost,
}
CATEGORY_REGISTRY: Final[dict[QualityCategory, CategoryRegistration]] = {
    category: _registration(module) for category, module in _CATEGORY_MODULES.items()
}
ELIGIBILITY_KEYS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(
        key
        for registration in CATEGORY_REGISTRY.values()
        for key in registration.eligibility_keys
    )
)
