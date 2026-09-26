"""評分層的聲源模型身分與兩條評估路徑各自的旗標。

對準點像擺位，是候選間可比較的變數；軸線由對準點計算；驗證狀態句由種類決定。
因此這三格不進模型指紋，改了模型種類、版本或六個參數才分表。
全向音色刻意沒有指向性旗標，維持既有全向報表的旗標與分數；反射仍發 NO_DIRECTIVITY。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Final

from aosr.physics.report_source import SourceModelKind, SourceModelSection
from aosr.scoring.contract_base import Flag


SOURCE_MODEL_FINGERPRINT_FIELDS: Final[frozenset[str]] = frozenset(
    {"kind", "model_version", "parameters"}
)
SOURCE_MODEL_EXCLUDED_FIELDS: Final[frozenset[str]] = frozenset(
    {"aim_m", "axis_unit_vector", "verification_status"}
)

REFLECTION_DIRECTIVITY_FLAGS: Final[dict[SourceModelKind, tuple[Flag, ...]]] = {
    SourceModelKind.OMNIDIRECTIONAL: (Flag.NO_DIRECTIVITY,),
    SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1: (
        Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED,
    ),
}
TIMBRE_DIRECTIVITY_FLAGS: Final[dict[SourceModelKind, tuple[Flag, ...]]] = {
    SourceModelKind.OMNIDIRECTIONAL: (),
    SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1: (
        Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED,
    ),
}


def source_model_fingerprint(section: SourceModelSection) -> str:
    """雜湊種類、模型版本、參數；全向的後兩格明確為 null。"""
    canonical = json.dumps(
        {
            "kind": section.kind.value,
            "model_version": section.model_version,
            "parameters": (
                section.parameters.model_dump(mode="json")
                if section.parameters is not None else None
            ),
        },
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def distinct_source_model_fingerprints(values: Iterable[str | None]) -> frozenset[str]:
    """彙總上游的非空身分；空值不冒充另一個模型。"""
    return frozenset(value for value in values if value is not None)


def unique_source_model_fingerprint(values: Iterable[str | None]) -> str | None:
    """上游恰有一種非空模型時照抄，衝突或沒有上游時留空。"""
    distinct = distinct_source_model_fingerprints(values)
    return next(iter(distinct)) if len(distinct) == 1 else None


def _flags_for(
    kind: SourceModelKind,
    table: dict[SourceModelKind, tuple[Flag, ...]],
) -> tuple[Flag, ...]:
    try:
        return table[kind]
    except KeyError as exc:
        raise ValueError(f"未知聲源模型種類：{kind}") from exc


def reflection_directivity_flags(kind: SourceModelKind) -> tuple[Flag, ...]:
    """反射評估的聲源模型旗標；表外種類直接拒收。"""
    return _flags_for(kind, REFLECTION_DIRECTIVITY_FLAGS)


def timbre_directivity_flags(kind: SourceModelKind) -> tuple[Flag, ...]:
    """音色評估的聲源模型旗標；全向刻意不新增旗標。"""
    return _flags_for(kind, TIMBRE_DIRECTIVITY_FLAGS)
