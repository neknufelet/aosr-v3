"""#608：重用實際相同的輸入，並以主線舊寫法驗證複數逐位不變。"""

from __future__ import annotations

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


@pytest.mark.parametrize("varying", [0, 1, len(CANONICAL_WALLS)], ids=["flat", "one", "all"])
@pytest.mark.parametrize("cos", [0.0, 0.5], ids=["grazing", "oblique"])
def test_reuses_identical_inputs_and_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, varying: int, cos: float,
) -> None:
    """抓角度／係數重算及未重用非相鄰的相同乘積；包裝器照算真公式。"""
    materials = _repeated_materials(varying)
    receiver, image, dist_m = (cos, cos, cos), (0.0, 0.0, 0.0), 1.0
    counts = {wall: index % 4 + 1 for index, wall in enumerate(CANONICAL_WALLS)}
    axes: list[int] = []
    inputs: list[tuple[str, str, str, str]] = []

    def record_cos(
        axis: int, dist: float, recv: tuple[float, float, float], img: tuple[float, float, float],
    ) -> float:
        axes.append(axis)
        return incidence_cos(axis, dist, recv, img)

    def record_coefficient(Z: complex, cos: float, rho_c: float) -> complex:
        inputs.append((Z.real.hex(), Z.imag.hex(), cos.hex(), rho_c.hex()))
        return reflection_coefficient(Z, cos, rho_c)

    expected_inputs = list(dict.fromkeys(
        (Z.real.hex(), Z.imag.hex(), cos.hex(), materials.rho_c.hex())
        for f_index in range(len(materials.frequencies_hz))
        for wall in CANONICAL_WALLS
        for Z in (materials.impedance(wall, f_index),)
    ))
    expected = _legacy_reflection_product(materials, dist_m, receiver, image, counts)
    monkeypatch.setattr(amp, "incidence_cos", record_cos)
    monkeypatch.setattr(amp, "reflection_coefficient", record_coefficient)
    actual = amp.reflection_product(materials, dist_m, receiver, image, counts)
    assert _bits(actual) == _bits(expected)
    assert axes == [_wall_axis(wall) for wall in CANONICAL_WALLS]
    assert inputs == expected_inputs
    for first, repeated in ((0, 2), (1, 4), (3, 5)):
        assert actual[first] is actual[repeated]
    if cos == 0.0:
        assert all(value is actual[0] for value in actual)


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


@pytest.mark.parametrize("receiver", [(0.5, 0.5, 0.5), (0.0, 0.25, 0.5)])
def test_coefficient_reuse_uses_values_across_walls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receiver: tuple[float, float, float],
) -> None:
    """同阻抗跨牆可共用，但入射 cos 不同必須分開；零次反彈的牆不取材料。"""
    counts = {wall: index % 5 for index, wall in enumerate(CANONICAL_WALLS)}
    row = (complex(800.0, 0.0), complex(800.0, -0.0), complex(800.0, 0.0))
    materials = Materials(
        400.0, (100.0, 200.0, 300.0),
        {wall: row for wall in CANONICAL_WALLS if counts[wall]},
    )
    inputs: list[tuple[str, str, str, str]] = []

    def record_coefficient(Z: complex, cos: float, rho_c: float) -> complex:
        inputs.append((Z.real.hex(), Z.imag.hex(), cos.hex(), rho_c.hex()))
        return reflection_coefficient(Z, cos, rho_c)

    expected_inputs = list(dict.fromkeys(
        (Z.real.hex(), Z.imag.hex(), receiver[_wall_axis(wall)].hex(), materials.rho_c.hex())
        for Z in row for wall in CANONICAL_WALLS if counts[wall]
    ))
    image = (0.0, 0.0, 0.0)
    expected = _legacy_reflection_product(materials, 1.0, receiver, image, counts)
    monkeypatch.setattr(amp, "reflection_coefficient", record_coefficient)
    actual = amp.reflection_product(materials, 1.0, receiver, image, counts)
    assert _bits(actual) == _bits(expected)
    assert inputs == expected_inputs


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
