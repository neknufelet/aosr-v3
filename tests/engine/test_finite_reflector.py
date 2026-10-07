"""有限矩形板的公式一致性；外部數字與考卷自寫的菲涅耳參照，不代表家具精度。

HL2009 第 307 頁；Cremer 1989 式 (5)；Meyer 第五版 §5.1.3；
Rindel 2005 簡報投影片 14、16、18。任意方位角是決策紙採用的推導。
"""
from __future__ import annotations

import math

import pytest
from scipy.special import fresnel

from aosr.geometry.furniture import FaceDirection, FurnitureFace, Vec3
from aosr.physics.finite_reflector import (
    cutoff_frequencies,
    effective_edge_lengths,
    finite_size_energy,
)

# HL2009 未公布聲速；343 m/s 是判定檔反推吻合的案例条件。
HL2009_INFERRED_C = 343.0
# 判定檔第 4.1 節的 Cremer／Meyer 對照條件，以及第 3 節自選算例的條件。
CREMER_M5_REFERENCE_C = 343.0
VERDICT_EXAMPLES_C = 343.0


def panel(lengths: tuple[float, float]) -> FurnitureFace:
    return FurnitureFace(FaceDirection.TOP, (0.0, 0.0, 0.0), lengths)


def ray(cos_theta: float, phi: float = 0.0) -> Vec3:
    sine = math.sqrt(1.0 - cos_theta * cos_theta)
    return sine * math.cos(phi), sine * math.sin(phi), -cos_theta


def exact_center_energy(f: float, lengths: tuple[float, float], cos_theta: float,
                        d_inc: float, d_refl: float) -> float:
    """R2005s 投影片 16：全邊兩端的 C、S 差；cosθ 只掛第一邊。

    這裡保留原文 a* 與兩端積分，故意不走產品的投影與截止頻率算法。
    """
    characteristic_distance = 2.0 * d_inc * d_refl / (d_inc + d_refl)
    wavelength = VERDICT_EXAMPLES_C / f
    scale = 2.0 / math.sqrt(wavelength * characteristic_distance)
    energy = 1.0
    for full_length, projection in zip(lengths, (cos_theta, 1.0), strict=True):
        endpoint = scale * full_length * projection / 2.0
        s_low, c_low = fresnel(-endpoint)
        s_high, c_high = fresnel(endpoint)
        energy *= ((float(c_high) - float(c_low)) ** 2 + (float(s_high) - float(s_low)) ** 2) / 2.0
    return energy


@pytest.mark.parametrize(("d_refl", "side", "published"), [
    (2.0, 2.0, 151), (2.0, 3.0, 67), (2.0, 4.0, 38), (2.0, 5.0, 24), (2.0, 6.0, 17),
    (1.0, 2.0, 80), (1.0, 3.0, 36), (1.0, 4.0, 20), (1.0, 5.0, 13), (1.0, 6.0, 9),
])
def test_hl2009_published_integer_cutoffs(d_refl: float, side: float, published: int) -> None:
    """÷4 的特徵距離或半寬都不能重現發表的十個整數。"""
    got = cutoff_frequencies(panel((side, side)), (0.0, 0.0, -1.0),
                             d_inc=14.5, d_refl=d_refl, c=HL2009_INFERRED_C)
    assert tuple(round(value) for value in got) == (published, published)


@pytest.mark.parametrize(("side", "d_inc", "d_refl", "cos_theta", "expected"), [
    (2.0, 30.0, 15.0, 1.0, 857.5),
    (1.5, 10.0, 20.0, 1.0, 1016.3),
    (1.5, 10.0, 20.0, math.sqrt(0.5), 2032.6),
])
def test_cremer_meyer_factor_two_controls(side: float, d_inc: float, d_refl: float,
                                        cos_theta: float, expected: float) -> None:
    got = cutoff_frequencies(panel((side, 3.0)), ray(cos_theta),
                             d_inc=d_inc, d_refl=d_refl, c=CREMER_M5_REFERENCE_C)
    assert round(got[0], 1) == expected


@pytest.mark.parametrize("frequency", [0.01, 0.1, 0.5])
def test_fresnel_low_frequency_limit(frequency: float) -> None:
    """獨立精確式咬住全長、係數二倍與兩方向能量相乘。"""
    got, = finite_size_energy((frequency,), panel((0.8, 1.6)), ray(0.6),
                              d_inc=0.75, d_refl=0.75, c=VERDICT_EXAMPLES_C)
    exact = exact_center_energy(frequency, (0.8, 1.6), 0.6, 0.75, 0.75)
    assert exact / got == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize(("phi", "expected_q"), [
    (0.0, (0.48, 1.6)), (math.pi / 2.0, (0.8, 0.96)),
])
def test_axis_aligned_azimuth_recovers_original(phi: float, expected_q: tuple[float, float]) -> None:
    assert effective_edge_lengths(panel((0.8, 1.6)), ray(0.6, phi)) == pytest.approx(expected_q)


def test_oblique_azimuth_preserves_projected_area() -> None:
    q = effective_edge_lengths(panel((0.8, 1.6)), ray(0.6, math.pi / 4.0))
    assert q == pytest.approx((0.8 * math.sqrt(0.6), 1.6 * math.sqrt(0.6)))
    got = finite_size_energy((1.0, 10.0, 31.25), panel((0.8, 1.6)), ray(0.6, math.pi / 4.0),
                             d_inc=0.75, d_refl=0.75, c=VERDICT_EXAMPLES_C)
    area_cutoff = VERDICT_EXAMPLES_C * 0.75 / (2.0 * 0.8 * 1.6 * 0.6)
    assert got == pytest.approx(tuple((f / area_cutoff) ** 2 for f in (1.0, 10.0, 31.25)))


def test_energy_is_monotone_bounded_and_saturates() -> None:
    frequencies = (1.0, 10.0, 125.0, 250.0, 500.0, 1000.0, 8000.0)
    got = finite_size_energy(frequencies, panel((0.8, 1.6)), ray(0.6),
                             d_inc=0.75, d_refl=0.75, c=VERDICT_EXAMPLES_C)
    assert all(first <= second for first, second in zip(got[:-1], got[1:], strict=True))
    assert all(0.0 <= value <= 1.0 for value in got)
    assert got[-2:] == (1.0, 1.0)


def test_exchanging_incident_and_reflected_distances_is_identical() -> None:
    args = ((10.0, 125.0, 500.0), panel((0.8, 1.6)), ray(0.6, math.pi / 4.0))
    assert finite_size_energy(*args, d_inc=1.25, d_refl=3.0, c=VERDICT_EXAMPLES_C) == (
        finite_size_energy(*args, d_inc=3.0, d_refl=1.25, c=VERDICT_EXAMPLES_C))


@pytest.mark.parametrize("phi", [0.0, math.pi / 4.0, math.pi / 2.0])
def test_normal_incidence_is_independent_of_azimuth(phi: float) -> None:
    got = finite_size_energy((1.0, 125.0, 500.0), panel((0.8, 1.6)), ray(1.0, phi),
                             d_inc=0.75, d_refl=0.75, c=VERDICT_EXAMPLES_C)
    # 垂直入射的原文式：a*=0.75，無 cosθ 項。
    expected = tuple(min(1.0, 2.0 * f * 0.8 ** 2 / (VERDICT_EXAMPLES_C * 0.75))
                     * min(1.0, 2.0 * f * 1.6 ** 2 / (VERDICT_EXAMPLES_C * 0.75))
                     for f in (1.0, 125.0, 500.0))
    assert got == pytest.approx(expected)


def test_sofa_incidence_plane_follows_long_edge() -> None:
    total_length = math.hypot(3.0, 0.55 + 0.75)
    incident = total_length * 0.55 / (0.55 + 0.75)
    reflected = total_length * 0.75 / (0.55 + 0.75)
    cosine = (0.55 + 0.75) / total_length
    got = cutoff_frequencies(panel((0.75, 1.8)), ray(cosine, math.pi / 2.0),
                             d_inc=incident, d_refl=reflected, c=VERDICT_EXAMPLES_C)
    assert tuple(round(value, 1) for value in got) == (486.6, 534.4)
    energy, = finite_size_energy((500.0,), panel((0.75, 1.8)), ray(cosine, math.pi / 2.0),
                                 d_inc=incident, d_refl=reflected, c=VERDICT_EXAMPLES_C)
    assert round(energy, 4) == 0.9356


@pytest.mark.parametrize(("lengths", "direction", "d", "cutoffs", "energy", "pressure"), [
    ((0.8, 1.6), (0.8, 0.0, -0.6), 0.75, (558.3, 50.2), 0.2239, 0.4732),
    ((1.2, 1.2), (1.25, 0.0, 1.2), math.hypot(1.25, 1.2), (430.3, 206.4), 0.1760, 0.4195),
])
def test_verdict_table_and_cloud_examples(lengths: tuple[float, float], direction: Vec3, d: float,
                                         cutoffs: tuple[float, float], energy: float, pressure: float) -> None:
    size = math.sqrt(sum(value * value for value in direction))
    unit: Vec3 = (direction[0] / size, direction[1] / size, direction[2] / size)
    actual_cutoffs = cutoff_frequencies(panel(lengths), unit, d_inc=d, d_refl=d, c=VERDICT_EXAMPLES_C)
    got, = finite_size_energy((125.0,), panel(lengths), unit, d_inc=d, d_refl=d, c=VERDICT_EXAMPLES_C)
    assert tuple(round(value, 1) for value in actual_cutoffs) == cutoffs
    assert round(got, 4) == energy
    assert round(math.sqrt(got), 4) == pressure


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_invalid_distances_and_speed_have_no_epsilon_rescue(bad: float) -> None:
    for d_inc, d_refl, c in ((bad, 1.0, VERDICT_EXAMPLES_C), (1.0, bad, VERDICT_EXAMPLES_C), (1.0, 1.0, bad)):
        with pytest.raises(ValueError):
            cutoff_frequencies(panel((0.8, 1.6)), ray(0.6), d_inc=d_inc, d_refl=d_refl, c=c)


def test_grazing_and_zero_direction_are_undefined_not_clipped() -> None:
    for direction in ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)):
        with pytest.raises(ValueError):
            effective_edge_lengths(panel((0.8, 1.6)), direction)


def test_empty_frequency_axis_returns_empty_energy() -> None:
    assert finite_size_energy((), panel((0.8, 1.6)), ray(0.6),
                              d_inc=0.75, d_refl=0.75, c=VERDICT_EXAMPLES_C) == ()
