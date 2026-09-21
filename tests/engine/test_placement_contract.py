"""擺位契約自己的考卷：逐位元比對每一軸、正負零、表的形狀、合併與順序（票 #417 檢查席點的盲點）。"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from aosr.scoring.contract import Flag, ReasonCode, in_declared_order
from aosr.scoring.placement import (
    Placement,
    PlacementMismatchError,
    merge_or_empty,
    merge_placements,
)

_BASE = (1.2, 0.9, 1.1)


def _speaker_at(position: tuple[float, float, float]) -> Placement:
    return Placement(
        speaker_positions_m=(("left", position),),
        receiver_positions_m=(("main", (4.7, 2.8, 1.4)),),
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_axis_is_compared_down_to_the_last_bit(axis: int) -> None:
    """只有一軸差一個最小浮點刻度也算不同座標：三軸各驗一次，不是只比 x。"""
    moved = list(_BASE)
    moved[axis] = math.nextafter(_BASE[axis], math.inf)

    with pytest.raises(PlacementMismatchError, match="left"):
        merge_placements([_speaker_at(_BASE), _speaker_at((moved[0], moved[1], moved[2]))])


def test_positive_and_negative_zero_are_different_positions() -> None:
    """決策紙明寫逐位元比：正零與負零算不同座標，不先放寬。"""
    with pytest.raises(PlacementMismatchError, match="left"):
        merge_placements([_speaker_at((0.0, 0.9, 1.1)), _speaker_at((-0.0, 0.9, 1.1))])


def test_receiver_table_is_checked_as_well_as_speaker_table() -> None:
    one = Placement(speaker_positions_m=(), receiver_positions_m=(("main", (4.7, 2.8, 1.4)),))
    other = Placement(speaker_positions_m=(), receiver_positions_m=(("main", (4.7, 2.8, 1.5)),))

    with pytest.raises(PlacementMismatchError, match="main"):
        merge_placements([one, other])


def test_merge_or_empty_keeps_consistent_rows_and_empties_only_on_conflict() -> None:
    """沒有衝突就原樣合併；只有衝突才清成空表——不准不可估就一律丟掉擺位。"""
    right = Placement(
        speaker_positions_m=(("right", (1.2, 3.1, 1.1)),),
        receiver_positions_m=(("main", (4.7, 2.8, 1.4)),),
    )
    merged = merge_or_empty([right, _speaker_at(_BASE)])
    conflict = merge_or_empty([_speaker_at(_BASE), _speaker_at((9.0, 9.0, 9.0))])

    assert merged == merge_or_empty([_speaker_at(_BASE), right])
    assert dict(merged.speaker_positions_m) == {"left": _BASE, "right": (1.2, 3.1, 1.1)}
    assert dict(merged.receiver_positions_m) == {"main": (4.7, 2.8, 1.4)}
    assert conflict.speaker_positions_m == () and conflict.receiver_positions_m == ()


@pytest.mark.parametrize(
    "document",
    [
        {"speaker_positions_m": (("left", (math.nan, 0.0, 0.0)),), "receiver_positions_m": ()},
        {"speaker_positions_m": (("left", (math.inf, 0.0, 0.0)),), "receiver_positions_m": ()},
        {"speaker_positions_m": (("", (1.0, 0.0, 0.0)),), "receiver_positions_m": ()},
        {"speaker_positions_m": (("left", (1.0, 0.0)),), "receiver_positions_m": ()},
        {"speaker_positions_m": (("left", (1.0, 0.0, 0.0, 0.0)),), "receiver_positions_m": ()},
        {"speaker_positions_m": (), "receiver_positions_m": (), "note": "extra"},
    ],
    ids=["nan", "inf", "empty-id", "two-numbers", "four-numbers", "unknown-field"],
)
def test_placement_rejects_malformed_tables(document: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Placement.model_validate(document)


def test_placement_is_frozen() -> None:
    placement = _speaker_at(_BASE)
    field = "speaker_positions_m"

    with pytest.raises(ValidationError):
        setattr(placement, field, ())


def test_declared_order_ignores_input_order_and_duplicates() -> None:
    """彙總類的原因碼與旗標照列舉宣告的順序排，輸入順序怎麼換都一樣。"""
    reasons = [ReasonCode.SPEAKER_ID_MISMATCH, ReasonCode.CANDIDATE_ID_MISMATCH, ReasonCode.SPEAKER_ID_MISMATCH]
    flags = list(Flag)

    assert in_declared_order(reasons) == in_declared_order(list(reversed(reasons)))
    assert set(in_declared_order(reasons)) == set(reasons)
    assert in_declared_order(reversed(flags)) == tuple(flags)
    assert in_declared_order([]) == ()
