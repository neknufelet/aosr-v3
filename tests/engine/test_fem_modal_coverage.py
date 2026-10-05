"""漏模態自檢：圓聯集的解析下包絡，不採實軸覆蓋或採樣網格。"""
from __future__ import annotations

import math

import pytest

from aosr.physics.fem_modal import ModalShift
from aosr.physics.fem_modal_check import guaranteed_decay_height


def test_circle_envelope_minimum_is_at_internal_intersection() -> None:
    shifts = (ModalShift(1 / (2 * math.pi), 1, 2.0),
              ModalShift(3 / (2 * math.pi), 1, 2.0))
    assert guaranteed_decay_height(shifts, 4 / (2 * math.pi)) == pytest.approx(math.sqrt(3))


def test_circle_envelope_handles_containment_and_unsorted_shifts() -> None:
    shifts = (ModalShift(2 / (2 * math.pi), 1, 1.0),
              ModalShift(2 / (2 * math.pi), 1, 3.0))
    assert guaranteed_decay_height(shifts, 4 / (2 * math.pi)) == pytest.approx(math.sqrt(5))


def test_circle_envelope_gap_has_zero_guarantee() -> None:
    shifts = (ModalShift(1 / (2 * math.pi), 1, 0.5),
              ModalShift(3 / (2 * math.pi), 1, 0.5))
    assert guaranteed_decay_height(shifts, 4 / (2 * math.pi)) == 0.0


def test_circle_envelope_has_no_closed_boundary_claim() -> None:
    # 最遠返回根可能只佔重根的一部分名額；正式保證用嚴格小於高度。
    shifts = (ModalShift(1 / (2 * math.pi), 1, 1.0),)
    assert guaranteed_decay_height(shifts, 2 / (2 * math.pi)) == 0.0
