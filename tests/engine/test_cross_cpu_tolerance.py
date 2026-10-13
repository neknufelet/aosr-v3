"""跨處理器容差輔助檔的考卷：比法、hex、複數、NaN 與正負零、錯誤訊息。"""
from __future__ import annotations

import math

import pytest

from tests.engine._cross_cpu_tolerance import (GEOMETRIC_REL, LINEAR_SOLVE_REL, SCORING_ABS, SCORING_REL, assert_within,
                                               cell, ratios)

# 界線兩側推的比例：要比一個捨入步大得多，①級 5e-14 只有約兩百個末位，推 2^-10 會被捨入吃掉。
DELTA = 2.0**-3


@pytest.mark.parametrize("rel", (GEOMETRIC_REL, LINEAR_SOLVE_REL, SCORING_REL))
def test_relative_boundary_both_sides(rel: float) -> None:
    """抓比法寫成別的形狀：相對界線內推 δ 過、界線外推 δ 不過。"""
    answer = 0.37
    assert_within([cell("inside", answer + (1 - DELTA) * rel * answer, answer, rel=rel, abs_=0.0)])
    with pytest.raises(AssertionError):
        assert_within([cell("outside", answer + (1 + DELTA) * rel * answer, answer, rel=rel, abs_=0.0)])


def test_absolute_boundary_and_max_of_both() -> None:
    """抓只用相對或只用絕對：答案接近零時由絕對管，答案大時由相對管，取兩者較大。"""
    near_zero = 1e-20
    assert_within([cell("abs", near_zero + (1 - DELTA) * SCORING_ABS, near_zero, rel=SCORING_REL, abs_=SCORING_ABS)])
    with pytest.raises(AssertionError):
        assert_within([cell("abs", near_zero + (1 + DELTA) * SCORING_ABS, near_zero,
                            rel=SCORING_REL, abs_=SCORING_ABS)])
    equal = SCORING_ABS / SCORING_REL
    for factor in (1 - DELTA, 1 + DELTA):
        cells = [cell("equal", equal + factor * SCORING_ABS, equal, rel=SCORING_REL, abs_=SCORING_ABS)]
        if factor < 1:
            assert_within(cells)
        else:
            with pytest.raises(AssertionError):
                assert_within(cells)
    large = 1e4
    assert_within([cell("rel", large * (1 + (1 - DELTA) * SCORING_REL), large, rel=SCORING_REL, abs_=SCORING_ABS)])
    with pytest.raises(AssertionError):
        assert_within([cell("rel", large * (1 + (1 + DELTA) * SCORING_REL), large, rel=SCORING_REL, abs_=SCORING_ABS)])


def test_row_scale_absolute_comes_from_answer() -> None:
    """會相消的量給同列正量答案當絕對尺才過、不給就紅。絕對尺由答案算（不由算出值算）是呼叫端的寫法，這題守不到。"""
    row_total, tail = 0.27, 1e-6
    tolerance = LINEAR_SOLVE_REL * row_total
    assert_within([cell("tail", tail + (1 - DELTA) * tolerance, tail, rel=LINEAR_SOLVE_REL, abs_=tolerance)])
    with pytest.raises(AssertionError):
        assert_within([cell("tail", tail + (1 + DELTA) * tolerance, tail, rel=LINEAR_SOLVE_REL, abs_=tolerance)])
    with pytest.raises(AssertionError):
        assert_within([cell("tail", tail + (1 - DELTA) * tolerance, tail, rel=LINEAR_SOLVE_REL, abs_=0.0)])


def test_hex_strings_and_complex_modulus() -> None:
    """抓 hex 沒轉、或複數分開比實虛部：比的是差的模對答案的模。"""
    answer = complex(3.0, 4.0)
    edge = (1 - DELTA) * GEOMETRIC_REL * 5.0
    assert_within([cell("p", answer + complex(edge * 0.6, edge * 0.8), answer, rel=GEOMETRIC_REL, abs_=0.0)])
    with pytest.raises(AssertionError):
        over = (1 + DELTA) * GEOMETRIC_REL * 5.0
        assert_within([cell("p", answer + complex(over * 0.6, over * 0.8), answer, rel=GEOMETRIC_REL, abs_=0.0)])
    value = 0.1 * (1 + GEOMETRIC_REL / 4)
    assert_within([cell("hex", value.hex(), (0.1).hex(), rel=GEOMETRIC_REL, abs_=0.0)])


def test_last_bits_pass_and_small_real_error_fails() -> None:
    """末位差（1e-15）要放過；小但真的錯（1e-9）要擋。"""
    for rel in (GEOMETRIC_REL, LINEAR_SOLVE_REL, SCORING_REL):
        assert_within([cell("ulp", 0.42 * (1 + 1e-15), 0.42, rel=rel, abs_=0.0)])
        with pytest.raises(AssertionError):
            assert_within([cell("wrong", 0.42 * (1 + 1e-9), 0.42, rel=rel, abs_=0.0)])


def test_nan_and_signed_zero() -> None:
    """抓答案 NaN 被當成可比、算出 NaN 被放過；正負零算相等（正負零是契約的地方由同次對照守）。"""
    with pytest.raises(ValueError, match="答案不是有限數"):
        cell("nan", 1.0, math.nan, rel=GEOMETRIC_REL, abs_=0.0)
    for rel, abs_ in ((-GEOMETRIC_REL, 0.0), (GEOMETRIC_REL, -1e-20), (math.inf, 0.0), (GEOMETRIC_REL, math.inf),
                      (math.nan, 0.0)):
        with pytest.raises(ValueError, match="容差要是有限的非負數"):
            cell("bad", 1.0, 1.0, rel=rel, abs_=abs_)
    with pytest.raises(ValueError, match="不是 float.hex"):
        cell("decimal", "1.5", 1.5, rel=GEOMETRIC_REL, abs_=0.0)
    with pytest.raises(AssertionError):
        assert_within([cell("nan", math.nan, 1.0, rel=GEOMETRIC_REL, abs_=1.0)])
    assert_within([cell("zero", -0.0, 0.0, rel=GEOMETRIC_REL, abs_=0.0)])
    with pytest.raises(AssertionError):
        assert_within([cell("zero", 1e-300, 0.0, rel=GEOMETRIC_REL, abs_=0.0)])


def test_failure_lists_every_outside_cell_and_ratios() -> None:
    """抓錯誤訊息只列第一格或沒寫算出、答案、容差；ratios 給每一格的比值。"""
    cells = [cell("first", 2.0, 1.0, rel=GEOMETRIC_REL, abs_=0.0),
             cell("fine", 1.0, 1.0, rel=GEOMETRIC_REL, abs_=0.0),
             cell("second", 3.0, 1.0, rel=GEOMETRIC_REL, abs_=0.0)]
    with pytest.raises(AssertionError) as raised:
        assert_within(cells)
    message = str(raised.value)
    assert "first" in message and "second" in message and "fine" not in message
    assert "算出 2.0" in message and "答案 1.0" in message and f"容差 {GEOMETRIC_REL!r}" in message
    measured = ratios(cells)
    assert measured["fine"] == 0.0 and measured["first"] == pytest.approx(1 / GEOMETRIC_REL)
