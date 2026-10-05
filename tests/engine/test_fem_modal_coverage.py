"""漏模態自檢：圓聯集的解析下包絡，不採實軸覆蓋或採樣網格。"""
from __future__ import annotations

import math

import pytest

from aosr.physics.fem_modal import ModalShift
from aosr.physics.fem_modal_check import guaranteed_decay_height


def test_circle_envelope_minimum_is_at_internal_intersection() -> None:
    # 兩圓心 0.5、3.5，半徑 2，範圍 0～4：端點高 √3.75，兩圓交點 x=2 只有 √1.75，最低點在交點。
    shifts = (ModalShift(0.5 / (2 * math.pi), 1, 2.0),
              ModalShift(3.5 / (2 * math.pi), 1, 2.0))
    assert guaranteed_decay_height(shifts, 4 / (2 * math.pi)) == pytest.approx(math.sqrt(1.75))


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


def test_circle_envelope_interior_gap_has_zero_guarantee() -> None:
    # 兩端都蓋到（圓心 0.25、3.75，半徑 1），中間 1.25～2.75 沒蓋到：保證高度必須是零。
    shifts = (ModalShift(0.25 / (2 * math.pi), 1, 1.0),
              ModalShift(3.75 / (2 * math.pi), 1, 1.0))
    assert guaranteed_decay_height(shifts, 4 / (2 * math.pi)) == 0.0


@pytest.mark.parametrize("edges", [(0.0, 50.0), (10.0, 165.0), (0.0, 100.0, 90.0, 165.0)])
def test_bad_band_edges_rejected_before_solving(edges: tuple[float, ...]) -> None:
    from aosr.physics import fem_modal
    from aosr.physics.fem_modal import ModalSolverOptions

    with pytest.raises(ValueError):
        fem_modal._band_edges(edges, 165.0, ModalSolverOptions())
