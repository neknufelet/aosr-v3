"""#608：打到的牆阻抗都不隨頻率變時乘積只算一次，並以主線舊寫法驗證複數逐位不變。"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from aosr.physics import amplitude as amp
from aosr.physics.amplitude import (
    CANONICAL_WALLS,
    Materials,
    _wall_axis,
    incidence_cos,
    reflection_coefficient,
)


# origin/main 的原函式逐字保留，僅改函式名稱；不跟隨新實作的重用策略。
def _legacy_reflection_product(
    materials: Materials,
    dist_m: float,
    receiver: tuple[float, float, float],
    image: tuple[float, float, float],
    counts: dict[str, int],
) -> tuple[complex, ...]:
    """一條路徑每個頻帶的反射乘積：對每一面牆，R 連乘 ``count`` 次（``r ** count``）。

    ``counts`` 是 :func:`aosr.geometry.shoebox.wall_count_signature` 的結果（每面牆打幾次）。
    ``r ** count`` 的前提是**整面牆同一個阻抗**（跟上一代逐次反彈取 cell 的做法在這個前提下
    等價；分格材料是以後的事）。直達路徑（全部 count 為 0）回 ``(1+0j, …)`` 恰好。入射 cos
    對同一面牆的每一次反彈都用「那一軸的 |receiver − image| / dist」——攤開直線的定義，
    不逐反彈求角。
    """
    out: list[complex] = []
    for f_index in range(len(materials.frequencies_hz)):
        product = complex(1.0, 0.0)
        for wall in CANONICAL_WALLS:
            count = counts.get(wall, 0)
            if count == 0:
                continue
            axis = _wall_axis(wall)
            cos = incidence_cos(axis, dist_m, receiver, image)
            r = reflection_coefficient(materials.impedance(wall, f_index), cos, materials.rho_c)
            product *= r if count == 1 else r ** count
        out.append(product)
    return tuple(out)


def _bits(values: tuple[complex, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((float.hex(value.real), float.hex(value.imag)) for value in values)


def _random_materials(rng: random.Random, varying: int, bands: int) -> Materials:
    walls: dict[str, tuple[complex, ...]] = {}
    for index, wall in enumerate(CANONICAL_WALLS):
        values = tuple(
            complex(10.0 ** rng.uniform(-3.0, 8.0), rng.uniform(-1e6, 1e6))
            for _ in range(bands)
        )
        walls[wall] = values if index < varying else (values[0],) * bands
    return Materials(rng.uniform(100.0, 800.0), tuple(50.0 * (i + 1) for i in range(bands)), walls)


@pytest.mark.parametrize("varying", [0, 1, len(CANONICAL_WALLS)], ids=["flat", "one", "all"])
def test_random_products_match_legacy_bits(tmp_path: Path, varying: int) -> None:
    """抓不同阻抗誤共用、牆序改動、次方改成逐次乘，以及浮點重排。"""
    rng = random.Random(608)
    for _ in range(500):
        scale = 10.0 ** rng.uniform(-6.0, 6.0)
        receiver = (rng.uniform(-scale, scale), rng.uniform(-scale, scale), rng.uniform(-scale, scale))
        image = (rng.uniform(-scale, scale), rng.uniform(-scale, scale), rng.uniform(-scale, scale))
        dist_m = 10.0 ** rng.uniform(-4.0, 4.0)
        counts = {wall: rng.randrange(5) for wall in CANONICAL_WALLS}
        materials = _random_materials(rng, varying, rng.randrange(3, 16))
        expected = _legacy_reflection_product(materials, dist_m, receiver, image, counts)
        actual = amp.reflection_product(materials, dist_m, receiver, image, counts)
        assert _bits(actual) == _bits(expected), (materials, dist_m, receiver, image, counts)


def _repeated_materials(varying: int) -> Materials:
    pattern = (0, 1, 0, 2, 1, 2)
    walls = {
        wall: tuple(
            complex(900.0 + index * 37.0 + (band * 53.0 if index < varying else 0.0), index * 19.0)
            for band in pattern
        )
        for index, wall in enumerate(CANONICAL_WALLS)
    }
    return Materials(400.0, tuple(100.0 * (i + 1) for i in range(len(pattern))), walls)


def _recording(monkeypatch: pytest.MonkeyPatch) -> tuple[list[int], list[tuple[str, str, str]]]:
    """換掉入射角與反射係數，記下呼叫；包裝器照算真公式。"""
    axes: list[int] = []
    inputs: list[tuple[str, str, str]] = []

    def record_cos(
        axis: int, dist: float, recv: tuple[float, float, float], img: tuple[float, float, float],
    ) -> float:
        axes.append(axis)
        return incidence_cos(axis, dist, recv, img)

    def record_coefficient(Z: complex, cos: float, rho_c: float) -> complex:
        inputs.append((Z.real.hex(), Z.imag.hex(), cos.hex()))
        return reflection_coefficient(Z, cos, rho_c)

    monkeypatch.setattr(amp, "incidence_cos", record_cos)
    monkeypatch.setattr(amp, "reflection_coefficient", record_coefficient)
    return axes, inputs


def test_constant_walls_compute_each_coefficient_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """打到的牆阻抗都不隨頻率變：每面牆只算一次入射角與反射係數，乘積排滿每個頻帶，數字跟逐頻帶逐位相同。"""
    materials = _repeated_materials(0)
    receiver, image, dist_m = (0.5, 0.25, 0.75), (0.0, 0.0, 0.0), 1.0
    counts = {wall: index % 4 + 1 for index, wall in enumerate(CANONICAL_WALLS)}
    expected = _legacy_reflection_product(materials, dist_m, receiver, image, counts)
    axes, inputs = _recording(monkeypatch)
    actual = amp.reflection_product(materials, dist_m, receiver, image, counts)
    assert _bits(actual) == _bits(expected)
    assert axes == [_wall_axis(wall) for wall in CANONICAL_WALLS]
    assert [item[:2] for item in inputs] == [
        (materials.impedance(wall, 0).real.hex(), materials.impedance(wall, 0).imag.hex()) for wall in CANONICAL_WALLS
    ]


@pytest.mark.parametrize("varying", [1, len(CANONICAL_WALLS)], ids=["one", "all"])
def test_any_varying_hit_wall_falls_back_to_every_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, varying: int,
) -> None:
    """只要一面打到的牆隨頻率變，就照原本逐頻帶算（不誤套頻率無關的省法）。"""
    materials = _repeated_materials(varying)
    receiver, image, dist_m = (0.5, 0.25, 0.75), (0.0, 0.0, 0.0), 1.0
    counts = {wall: 1 for wall in CANONICAL_WALLS}
    expected = _legacy_reflection_product(materials, dist_m, receiver, image, counts)
    _, inputs = _recording(monkeypatch)
    actual = amp.reflection_product(materials, dist_m, receiver, image, counts)
    assert _bits(actual) == _bits(expected)
    assert len(inputs) == len(materials.frequencies_hz) * len(CANONICAL_WALLS)


def test_varying_wall_that_is_not_hit_keeps_the_shortcut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """隨頻率變的牆沒被打到（次數 0）不影響：其餘打到的牆照樣只算一次。"""
    materials = _repeated_materials(1)
    receiver, image, dist_m = (0.5, 0.25, 0.75), (0.0, 0.0, 0.0), 1.0
    counts = {wall: 0 if index == 0 else 2 for index, wall in enumerate(CANONICAL_WALLS)}
    expected = _legacy_reflection_product(materials, dist_m, receiver, image, counts)
    _, inputs = _recording(monkeypatch)
    actual = amp.reflection_product(materials, dist_m, receiver, image, counts)
    assert _bits(actual) == _bits(expected)
    assert len(inputs) == len(CANONICAL_WALLS) - 1


@pytest.mark.parametrize("row", [
    (complex(800.0, 0.0), complex(800.0, -0.0), complex(800.0, 0.0)),
    (complex(math.nan, 0.0),) * 3,
], ids=["signed-zero", "nan"])
def test_signed_zero_or_nan_is_not_treated_as_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: tuple[complex, ...],
) -> None:
    """正零與負零數值相等但位元不同、NaN 不等於自己：都不算「不隨頻率變」，照逐頻帶算。"""
    materials = Materials(400.0, (100.0, 200.0, 300.0), {wall: row for wall in CANONICAL_WALLS})
    receiver, image = (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)
    counts = {wall: 1 for wall in CANONICAL_WALLS}
    expected = _legacy_reflection_product(materials, 1.0, receiver, image, counts)
    _, inputs = _recording(monkeypatch)
    actual = amp.reflection_product(materials, 1.0, receiver, image, counts)
    assert _bits(actual) == _bits(expected)
    assert len(inputs) == len(materials.frequencies_hz) * len(CANONICAL_WALLS)


def test_direct_path_is_exact_unity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """直達路徑連 dist=0 也不應碰角度與阻抗，仍回逐位的 1+0j。"""
    materials = _repeated_materials(len(CANONICAL_WALLS))

    def unexpected_cos(
        axis: int, dist: float, recv: tuple[float, float, float], img: tuple[float, float, float],
    ) -> float:
        raise AssertionError("直達路徑不應計算入射角")

    monkeypatch.setattr(amp, "incidence_cos", unexpected_cos)
    receiver = image = (0.0, 0.0, 0.0)
    for counts in ({}, {wall: 0 for wall in CANONICAL_WALLS}):
        expected = _legacy_reflection_product(materials, 0.0, receiver, image, counts)
        actual = amp.reflection_product(materials, 0.0, receiver, image, counts)
        assert _bits(actual) == _bits(expected)
        assert _bits(actual) == _bits((complex(1.0, 0.0),) * len(materials.frequencies_hz))


def test_empty_frequency_axis_does_not_evaluate_angles(tmp_path: Path) -> None:
    """頻率軸為空時，保持舊寫法的空輸出，dist=0 也不計算入射角。"""
    materials = Materials(400.0, (), {})
    receiver = image = (0.0, 0.0, 0.0)
    counts = {wall: 1 for wall in CANONICAL_WALLS}
    expected = _legacy_reflection_product(materials, 0.0, receiver, image, counts)
    assert amp.reflection_product(materials, 0.0, receiver, image, counts) == expected


@pytest.mark.parametrize("count", range(1, 5))
def test_signed_zero_impedance_matches_legacy_bits(tmp_path: Path, count: int) -> None:
    """數值相等的正零／負零也不得在重用時混淆。"""
    materials = Materials(
        400.0, (100.0, 200.0, 300.0, 400.0),
        {wall: (complex(200.0, 0.0), complex(200.0, -0.0)) * 2 for wall in CANONICAL_WALLS},
    )
    for wall in CANONICAL_WALLS:
        for cos in (0.0, 0.5, 1.0):
            receiver, image = (cos, cos, cos), (0.0, 0.0, 0.0)
            counts = {wall: count}
            expected = _legacy_reflection_product(materials, 1.0, receiver, image, counts)
            actual = amp.reflection_product(materials, 1.0, receiver, image, counts)
            assert _bits(actual) == _bits(expected)


@pytest.mark.parametrize("row", [
    (complex(900.0, 50.0), complex(900.0, 50.0), complex(2000.0, -300.0)),
    (complex(900.0, 50.0), complex(2000.0, -300.0), complex(900.0, 50.0)),
    (400.0, complex(400.0, 0.0), complex(400.0, 0.0)),
], ids=["same-first-two", "same-first-last", "mixed-types"])
def test_only_fully_identical_rows_take_the_shortcut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: tuple[complex, ...],
) -> None:
    """頭兩格相同、頭尾相同、或型別混用都不算「不隨頻率變」：照逐頻帶算，數字跟舊寫法逐位相同（複查）。"""
    walls: dict[str, tuple[complex, ...]] = {wall: (complex(500.0, 10.0),) * 3 for wall in CANONICAL_WALLS}
    walls[CANONICAL_WALLS[0]] = row
    materials = Materials(413.0, (100.0, 200.0, 300.0), walls)
    counts = {CANONICAL_WALLS[0]: 1}
    receiver, image = (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)
    expected = _legacy_reflection_product(materials, 1.0, receiver, image, counts)
    _, inputs = _recording(monkeypatch)
    actual = amp.reflection_product(materials, 1.0, receiver, image, counts)
    assert _bits(actual) == _bits(expected)
    assert len(inputs) == len(materials.frequencies_hz)


def test_short_row_still_raises_like_before(tmp_path: Path) -> None:
    """阻抗列比頻率軸短：舊寫法照頻帶 index 報錯，捷徑不准悄悄排滿（複查）。"""
    walls: dict[str, tuple[complex, ...]] = {wall: (complex(500.0, 10.0),) for wall in CANONICAL_WALLS}
    materials = Materials(413.0, (100.0, 200.0, 300.0), walls)
    counts = {CANONICAL_WALLS[0]: 1}
    receiver, image = (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)
    with pytest.raises(ValueError) as legacy:
        _legacy_reflection_product(materials, 1.0, receiver, image, counts)
    with pytest.raises(ValueError) as current:
        amp.reflection_product(materials, 1.0, receiver, image, counts)
    assert str(current.value) == str(legacy.value)


def test_patched_wall_never_takes_the_shortcut(tmp_path: Path) -> None:
    """分格牆不准拿第一格走頻率無關的捷徑：直接呼叫照舊報「這面牆有分格」（複查）。"""
    row: tuple[complex, ...] = (complex(500.0, 10.0),) * 3
    walls: dict[str, tuple[complex, ...]] = {wall: row for wall in CANONICAL_WALLS}
    grids: dict[str, tuple[int, int, tuple[tuple[complex, ...], ...]]] = {
        wall: (1, 1, (row,)) for wall in CANONICAL_WALLS}
    grids[CANONICAL_WALLS[0]] = (2, 1, (row, row))
    materials = Materials(413.0, (100.0, 200.0, 300.0), walls, grids)
    counts = {CANONICAL_WALLS[0]: 1}
    with pytest.raises(ValueError, match="分格"):
        amp.reflection_product(materials, 1.0, (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), counts)
