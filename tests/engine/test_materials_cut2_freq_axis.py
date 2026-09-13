"""票 #218 第 1c 段：材料層的正式頻率表與純 Python 常用軸產生器。"""

from __future__ import annotations

import math
from typing import cast

import pytest

from aosr.materials import freq_axis
from tests.engine import _materials_answers as answers


def _answer_float_tuple(case_id: str, name: str) -> tuple[float, ...]:
    """從材料答案檔取出一串逐位浮點；形狀壞了就直接紅。"""
    result = answers.result_of(case_id)
    reported = answers.as_mapping(result.get("values"), f"{case_id}.values")
    raw = answers.as_mapping(reported.get(name), f"{case_id}.{name}")
    assert raw.get("kind") == "seq", f"{case_id}.{name} 不是一串值"
    assert raw.get("container") == "tuple", f"{case_id}.{name} 不是不可變 tuple"
    items = raw.get("items")
    assert isinstance(items, list), f"{case_id}.{name}.items 不是一串值"
    values: list[float] = []
    for item in items:
        encoded = answers.as_mapping(item, f"{case_id}.{name} 的一格")
        value = encoded.get("hex")
        assert encoded.get("kind") == "float" and isinstance(value, str), (
            f"{case_id}.{name} 裡有非浮點：{encoded!r}"
        )
        values.append(float.fromhex(value))
    return tuple(values)


def _answer_length() -> int:
    """從材料答案檔取出 donor 正式表的長度。"""
    case_id = "freq_axis.axis.length"
    result = answers.result_of(case_id)
    reported = answers.as_mapping(result.get("values"), f"{case_id}.values")
    encoded = answers.as_mapping(reported.get("length"), f"{case_id}.length")
    value = encoded.get("v")
    assert encoded.get("kind") == "int" and isinstance(value, int), (
        f"{case_id}.length 不是整數：{encoded!r}"
    )
    return value


def _independent_formula_axis(
    f_min_hz: float,
    f_max_hz: float,
    *,
    per_octave: int,
) -> tuple[float, ...]:
    """用 ``math.pow`` 找出滿足格點 ≤ f_max 的最大 n，不借對數取 floor。"""
    values: list[float] = []
    index = 0
    while (value := f_min_hz * math.pow(2.0, index / per_octave)) <= f_max_hz:
        values.append(value)
        index += 1
    return tuple(values)


def _nearest_formula_point(
    table_value: float,
    formula_axis: tuple[float, ...],
) -> tuple[int, float, float]:
    """回傳最近格點的索引、值、以及以公式值為分母的相對偏差。"""
    index, value = min(
        enumerate(formula_axis),
        key=lambda item: abs(table_value - item[1]) / item[1],
    )
    return index, value, abs(table_value - value) / value


def _maximum_table_deviation(
    formula_axis: tuple[float, ...],
    *,
    rounded_to_cents: bool,
) -> tuple[int, float, float, int, float]:
    """量 donor 手打表扣掉離格補點後，對指定公式基準的最大相對偏差。"""
    donor = _answer_float_tuple("freq_axis.FREQS_HZ.values", "FREQS_HZ")
    independent_axis = _independent_formula_axis(10.0, 20480.0, per_octave=6)
    off_grid_indexes = {
        table_index
        for table_index, table_value in enumerate(donor)
        if _nearest_formula_point(table_value, independent_axis)[2] > 2**-8
    }
    assert off_grid_indexes, "donor 表裡沒有任何由相對偏差辨認出的離格補點"

    deviations: list[tuple[int, float, float, int, float]] = []
    for table_index, table_value in enumerate(donor):
        if table_index in off_grid_indexes:
            continue
        formula_index, raw_formula_value, _ = _nearest_formula_point(table_value, formula_axis)
        formula_value = round(raw_formula_value, 2) if rounded_to_cents else raw_formula_value
        relative_deviation = abs(table_value - formula_value) / formula_value
        deviations.append(
            (formula_index, formula_value, relative_deviation, table_index, table_value)
        )
    return max(
        deviations,
        key=lambda item: item[2],
    )


def _assert_raw_formula_grid_deviation_at_most_2_to_minus_11(
    formula_axis: tuple[float, ...],
) -> None:
    """對原始公式格的最大相對偏差 ≤ 2^-11；描述 donor 手打表，不是 v3 契約。"""
    formula_index, formula_value, deviation, table_index, table_value = _maximum_table_deviation(
        formula_axis,
        rounded_to_cents=False,
    )
    assert deviation <= 2**-11, (
        f"對原始公式格的最大相對偏差 {deviation:.17g} > 2^-11，發生在 "
        f"FREQS_HZ[{table_index}]={table_value!r} Hz；最近的原始公式格 "
        f"formula_axis[{formula_index}]={formula_value!r} Hz"
    )


def _assert_rounded_formula_grid_deviation_at_most_2_to_minus_13(
    formula_axis: tuple[float, ...],
) -> None:
    """對 round(公式, 2) 的最大相對偏差 ≤ 2^-13；描述 donor 手打表，不是契約。"""
    formula_index, formula_value, deviation, table_index, table_value = _maximum_table_deviation(
        formula_axis,
        rounded_to_cents=True,
    )
    assert deviation <= 2**-13, (
        f"對 round(公式, 2) 的最大相對偏差 {deviation:.17g} > 2^-13，發生在 "
        f"FREQS_HZ[{table_index}]={table_value!r} Hz；最近的 round(公式, 2) 格 "
        f"formula_axis[{formula_index}]={formula_value!r} Hz"
    )


def _assert_axis_matches_independent_formula(actual: tuple[float, ...]) -> None:
    """逐位裁判正式軸是否等於用 ``math.pow`` 與「最大 n」獨立算出的軸。"""
    expected = _independent_formula_axis(10.0, 20480.0, per_octave=6)
    assert len(actual) == len(expected)
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        assert actual_value == expected_value, (
            f"formula_axis[{index}]：實際 {actual_value!r}，獨立公式 {expected_value!r}"
        )


@pytest.mark.parametrize(
    ("name", "case_id"),
    (
        ("FREQS_HZ", "freq_axis.FREQS_HZ.values"),
        ("FREQS_HZ_IDENTITY", "freq_axis.FREQS_HZ_IDENTITY.values"),
        ("CHORAS_BANDS", "freq_axis.CHORAS_BANDS.values"),
    ),
)
def test_public_frequency_sequence_matches_the_materials_donor_answer(
    name: str,
    case_id: str,
) -> None:
    """本段三串公開頻率值都逐位等於材料答案檔量到的 donor 值。"""
    actual = getattr(freq_axis, name)
    assert actual == _answer_float_tuple(case_id, name)


def test_frequency_axis_length_matches_the_donor_probe() -> None:
    """正式軸長度來自 donor 探針，不在考卷另抄一個數字。"""
    assert len(freq_axis.FREQS_HZ) == _answer_length()


def test_donor_table_max_relative_deviation_from_raw_formula_grid_is_at_most_2_to_minus_11() -> None:
    """對原始公式格的最大相對偏差 ≤ 2^-11；只描述上一代手打表，不是 v3 契約。"""
    actual = freq_axis.frequency_axis(
        "octave_fraction",
        10.0,
        20480.0,
        per_octave=6,
    )
    _assert_raw_formula_grid_deviation_at_most_2_to_minus_11(actual)


def test_donor_table_max_relative_deviation_from_round_formula_2_is_at_most_2_to_minus_13() -> None:
    """對 round(公式, 2) 的最大相對偏差 ≤ 2^-13；只描述上一代手打表，不是契約。"""
    actual = freq_axis.frequency_axis(
        "octave_fraction",
        10.0,
        20480.0,
        per_octave=6,
    )
    _assert_rounded_formula_grid_deviation_at_most_2_to_minus_13(actual)


def test_octave_fraction_axis_matches_an_independently_computed_formula() -> None:
    """公式軸的長度與每一格都等於用另一種標準庫寫法獨立算出的結果。"""
    actual = freq_axis.frequency_axis(
        "octave_fraction",
        10.0,
        20480.0,
        per_octave=6,
    )
    _assert_axis_matches_independent_formula(actual)


def test_octave_fraction_control_rejects_per_octave_plus_one_mutant() -> None:
    """控制組：把指數分母加一，兩種表偏差與公式獨立比對都必須紅。"""
    mutant: list[float] = []
    index = 0
    while (value := 10.0 * 2 ** (index / (6 + 1))) <= 20480.0:
        mutant.append(value)
        index += 1
    mutant_axis = tuple(mutant)

    with pytest.raises(AssertionError):
        _assert_raw_formula_grid_deviation_at_most_2_to_minus_11(mutant_axis)
    with pytest.raises(AssertionError):
        _assert_rounded_formula_grid_deviation_at_most_2_to_minus_13(mutant_axis)
    with pytest.raises(AssertionError):
        _assert_axis_matches_independent_formula(mutant_axis)


def test_octave_fraction_between_grid_f_max_keeps_last_point_inside_and_next_point_outside() -> None:
    """f_max 落在兩格之間時，最後一格 ≤ f_max，下一格則嚴格 > f_max。"""
    f_min_hz = 10.0
    per_octave = 3
    f_max_hz = f_min_hz * 2 ** (2 / per_octave) * 1.0001
    axis = freq_axis.frequency_axis(
        "octave_fraction",
        f_min_hz,
        f_max_hz,
        per_octave=per_octave,
    )
    next_point = f_min_hz * math.pow(2.0, len(axis) / per_octave)

    assert axis[-1] <= f_max_hz
    assert next_point > f_max_hz


def test_linear_axis_has_requested_endpoints_and_step() -> None:
    """1 Hz 細軸包含指定首尾，且每一格都是指定步長。"""
    axis = freq_axis.frequency_axis("linear", 10.0, 250.0, step_hz=1.0)
    assert axis[0] == 10.0
    assert axis[-1] == 250.0
    assert all(math.isclose(right - left, 1.0) for left, right in zip(axis, axis[1:]))
    assert axis == tuple(float(value) for value in range(10, 251))


def test_non_positive_minimum_names_f_min_hz() -> None:
    """非正下界要指名 f_min_hz。"""
    with pytest.raises(ValueError, match="f_min_hz"):
        freq_axis.frequency_axis("linear", 0.0, 250.0, step_hz=1.0)


def test_non_increasing_bounds_name_f_max_hz() -> None:
    """上界沒有高於下界要指名 f_max_hz。"""
    with pytest.raises(ValueError, match="f_max_hz"):
        freq_axis.frequency_axis("linear", 10.0, 10.0, step_hz=1.0)


@pytest.mark.parametrize("bad_per_octave", (None, 0, -1, 3.0, True))
def test_invalid_octave_fraction_names_per_octave(bad_per_octave: object) -> None:
    """倍頻格只能收正整數 per_octave。"""
    with pytest.raises(ValueError, match="per_octave"):
        freq_axis.frequency_axis(
            "octave_fraction",
            10.0,
            20.0,
            per_octave=cast(int | None, bad_per_octave),
        )


@pytest.mark.parametrize("bad_step_hz", (None, 0.0, -1.0, True))
def test_invalid_linear_step_names_step_hz(bad_step_hz: object) -> None:
    """線性格只能收正數 step_hz。"""
    with pytest.raises(ValueError, match="step_hz"):
        freq_axis.frequency_axis(
            "linear",
            10.0,
            20.0,
            step_hz=cast(float | None, bad_step_hz),
        )


def test_unknown_mode_names_mode() -> None:
    """不認得的模式要指名 mode。"""
    with pytest.raises(ValueError, match="mode"):
        freq_axis.frequency_axis("logarithmic", 10.0, 20.0)
