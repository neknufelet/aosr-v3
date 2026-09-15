"""型錄無規入射吸音率換成細軸實數阻抗的獨立考卷。

積分真值直接從 Paris 角度積分式計算；票上數表與手算的對數頻率中點則是另一組
外部期望。這些題不拿產品反推函式替產品自己製造答案。
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import quad

from aosr.materials import catalog_absorption as subject


def _paris_integral(zeta: float) -> float:
    """直接積分逐角度能量吸音率，作為閉式公式的獨立真值。"""

    def weighted_absorption(theta: float) -> float:
        cosine = math.cos(theta)
        reflection = (zeta * cosine - 1.0) / (zeta * cosine + 1.0)
        absorption = 1.0 - reflection * reflection
        return 2.0 * absorption * math.sin(theta) * cosine

    value, _error = quad(
        weighted_absorption,
        0.0,
        math.pi / 2.0,
        epsabs=0.0,
        epsrel=subject.CATALOG_ABSORPTION_PROPERTY_REL,
    )
    return float(value)


def _relative_difference(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


@pytest.mark.parametrize(
    "zeta",
    (1.0, 1.567, 2.6, 19.8, 150.0, 389.0, 1.0e4, 1.0e6),
)
def test_closed_form_matches_independent_paris_integral(zeta: float) -> None:
    """抓閉式係數、對數項、角度權重或大阻抗數值穩定性寫錯。"""
    expected = _paris_integral(zeta)
    actual = subject.random_incidence_absorption(zeta)
    assert (
        _relative_difference(actual, expected)
        <= subject.CATALOG_ABSORPTION_PROPERTY_REL
    )


@pytest.mark.parametrize("zeta", (0.0, -1.0, math.nan, math.inf, -math.inf))
def test_forward_rejects_nonpositive_or_nonfinite_zeta(zeta: float) -> None:
    """抓正算接受非正值或非有限阻抗，然後吐出無意義數字。"""
    with pytest.raises(ValueError, match=r"zeta=.*finite.*> 0"):
        subject.random_incidence_absorption(zeta)


@pytest.mark.parametrize(
    "alpha",
    (0.001, 0.02, 0.05, 0.3, 0.6, 0.9, 0.95),
)
def test_hard_branch_inversion_round_trips(alpha: float) -> None:
    """抓反推換成垂直入射、55 度法，或二分根沒有回到原吸音率。"""
    zeta = subject.normalized_impedance_from_random_incidence_absorption(alpha)
    recovered = subject.random_incidence_absorption(zeta)
    assert (
        _relative_difference(recovered, alpha)
        <= subject.CATALOG_ABSORPTION_PROPERTY_REL
    )


def test_peak_absorption_inversion_returns_the_computed_peak() -> None:
    """抓頂點等號被當成無解，或反推頂點時走到錯的支線。"""
    actual = subject.normalized_impedance_from_random_incidence_absorption(
        subject.MAX_RANDOM_INCIDENCE_ABSORPTION
    )
    assert actual == subject.ZETA_AT_MAX_ABSORPTION


@pytest.mark.parametrize(
    ("alpha", "digits", "expected"),
    (
        (0.02, 0, 389.0),
        (0.05, 0, 150.0),
        (0.3, 1, 19.8),
        (0.6, 1, 7.1),
        (0.9, 1, 2.6),
    ),
)
def test_ticket_impedance_table(alpha: float, digits: int, expected: float) -> None:
    """抓 Paris 硬側反推偏離票 #298 的獨立數表。"""
    actual = subject.normalized_impedance_from_random_incidence_absorption(alpha)
    assert round(actual, digits) == expected


def _zeta_from_55_degree_method(alpha: float) -> float:
    root = math.sqrt(1.0 - alpha)
    return (1.0 + root) / ((1.0 - root) * math.cos(math.radians(55.0)))


@pytest.mark.parametrize("alpha", (0.02, 0.05, 0.3))
def test_55_degree_control_is_higher_at_low_absorption(alpha: float) -> None:
    """抓 55 度對照組的低吸音方向被倒置。"""
    zeta_55 = _zeta_from_55_degree_method(alpha)
    assert subject.random_incidence_absorption(zeta_55) > alpha


@pytest.mark.parametrize("alpha", (0.6, 0.9))
def test_55_degree_control_is_lower_at_high_absorption(alpha: float) -> None:
    """抓 55 度對照組的高吸音方向被倒置。"""
    zeta_55 = _zeta_from_55_degree_method(alpha)
    assert subject.random_incidence_absorption(zeta_55) < alpha


def test_program_computed_peak_dominates_dense_samples() -> None:
    """抓頂點求解停太早，導致 [1, 3] 內有更大的閉式值。"""
    samples = np.linspace(1.0, 3.0, 20_001, dtype=np.float64)
    sampled_maximum = max(
        subject.random_incidence_absorption(float(zeta)) for zeta in samples
    )
    assert subject.MAX_RANDOM_INCIDENCE_ABSORPTION >= sampled_maximum


def test_hard_branch_is_monotonically_decreasing_above_the_peak() -> None:
    """抓反推所依賴的硬側區間選錯，或閉式在該區間不是單調遞減。"""
    samples = np.geomspace(subject.ZETA_AT_MAX_ABSORPTION, 1.0e6)
    absorption = tuple(
        subject.random_incidence_absorption(float(zeta)) for zeta in samples
    )
    assert all(right < left for left, right in zip(absorption, absorption[1:]))


def test_absorption_between_zeta_one_and_peak_uses_hard_branch() -> None:
    """抓同一吸音率的兩根取成 ζ 小於頂點的軟側根。"""
    at_one = subject.random_incidence_absorption(1.0)
    alpha = (at_one + subject.MAX_RANDOM_INCIDENCE_ABSORPTION) / 2.0
    zeta = subject.normalized_impedance_from_random_incidence_absorption(alpha)
    assert zeta >= subject.ZETA_AT_MAX_ABSORPTION


@pytest.mark.parametrize("alpha", (0.0, -0.1, math.nan, math.inf, -math.inf))
def test_inverse_rejects_nonpositive_or_nonfinite_alpha(alpha: float) -> None:
    """抓反推接受契約外的 α，或用夾值偷偷救回來。"""
    with pytest.raises(ValueError, match=r"alpha=.*finite.*> 0"):
        subject.normalized_impedance_from_random_incidence_absorption(alpha)


def test_inverse_error_names_value_and_maximum() -> None:
    """抓超過實數阻抗上限時沒把輸入值與可達上限寫進訊息。"""
    alpha = math.nextafter(subject.MAX_RANDOM_INCIDENCE_ABSORPTION, math.inf)
    with pytest.raises(ValueError) as caught:
        subject.normalized_impedance_from_random_incidence_absorption(alpha)
    message = str(caught.value)
    assert f"alpha={alpha!r}" in message
    assert f"maximum={subject.MAX_RANDOM_INCIDENCE_ABSORPTION!r}" in message


def test_catalog_interpolates_alpha_on_log_frequency_before_inversion() -> None:
    """抓改成線性 Hz，或先反推各帶 ζ 再內插 ζ。"""
    catalog = subject.CatalogAbsorption(
        material_id="two-band",
        band_center_hz=(100.0, 1000.0),
        absorption=(0.2, 0.6),
    )
    log_midpoint_hz = math.sqrt(100.0 * 1000.0)
    rho_c = 400.0
    result = subject.impedance_on_axis(
        catalog,
        (100.0, log_midpoint_hz, 1000.0),
        rho_c,
    )

    assert result.frequencies_hz == (100.0, log_midpoint_hz, 1000.0)
    assert result.absorption == (0.2, 0.4, 0.6)
    recovered_midpoint = subject.random_incidence_absorption(
        result.impedance_pa_s_per_m[1] / rho_c
    )
    assert (
        _relative_difference(recovered_midpoint, 0.4)
        <= subject.CATALOG_ABSORPTION_PROPERTY_REL
    )
    assert result.extrapolated == (False, False, False)


def test_catalog_above_one_is_clamped_at_center_and_extended_points() -> None:
    """抓大於 1 的型錄帶在建構時被拒絕，或細軸沒有逐點夾到 Paris 頂點。"""
    catalog = subject.CatalogAbsorption("above-one", (500.0,), (1.05,))
    result = subject.impedance_on_axis(catalog, (100.0, 500.0, 2000.0), 1.0)

    assert result.catalog_absorption == (1.05, 1.05, 1.05)
    assert result.absorption == (
        subject.MAX_RANDOM_INCIDENCE_ABSORPTION,
        subject.MAX_RANDOM_INCIDENCE_ABSORPTION,
        subject.MAX_RANDOM_INCIDENCE_ABSORPTION,
    )
    assert result.impedance_pa_s_per_m == (
        subject.ZETA_AT_MAX_ABSORPTION,
        subject.ZETA_AT_MAX_ABSORPTION,
        subject.ZETA_AT_MAX_ABSORPTION,
    )
    assert result.extrapolated == (True, False, True)
    assert result.clamped == (True, True, True)


def test_catalog_between_peak_and_one_is_clamped() -> None:
    """抓小於 1 但超過 Paris 頂點的型錄值沒有被夾。"""
    catalog = subject.CatalogAbsorption("above-peak", (500.0,), (0.97,))
    result = subject.impedance_on_axis(catalog, (500.0,), 1.0)

    assert result.catalog_absorption == (0.97,)
    assert result.absorption == (subject.MAX_RANDOM_INCIDENCE_ABSORPTION,)
    assert result.impedance_pa_s_per_m == (subject.ZETA_AT_MAX_ABSORPTION,)
    assert result.clamped == (True,)


def test_catalog_interpolates_raw_alpha_before_pointwise_clamping() -> None:
    """抓各帶先夾再內插，或整段只因端帶超限就一律夾值。"""
    catalog = subject.CatalogAbsorption(
        "crosses-peak",
        (100.0, 1000.0),
        (0.9, 1.05),
    )
    frequencies_hz = (100.0, 200.0, 250.0, 1000.0)
    raw_alpha = (0.9, 0.9451544993495972, 0.9596910013008056, 1.05)
    result = subject.impedance_on_axis(catalog, frequencies_hz, 1.0)

    assert result.catalog_absorption == raw_alpha
    assert result.absorption == (
        raw_alpha[0],
        raw_alpha[1],
        subject.MAX_RANDOM_INCIDENCE_ABSORPTION,
        subject.MAX_RANDOM_INCIDENCE_ABSORPTION,
    )
    assert result.impedance_pa_s_per_m[:2] == tuple(
        subject.normalized_impedance_from_random_incidence_absorption(alpha)
        for alpha in raw_alpha[:2]
    )
    assert result.impedance_pa_s_per_m[2:] == (
        subject.ZETA_AT_MAX_ABSORPTION,
        subject.ZETA_AT_MAX_ABSORPTION,
    )
    assert result.clamped == (False, False, True, True)


def test_catalog_alpha_exactly_at_peak_is_not_marked_clamped() -> None:
    """抓頂點等號被誤判成超限夾值。"""
    peak = subject.MAX_RANDOM_INCIDENCE_ABSORPTION
    catalog = subject.CatalogAbsorption("at-peak", (500.0,), (peak,))
    result = subject.impedance_on_axis(catalog, (500.0,), 1.0)

    assert result.catalog_absorption == (peak,)
    assert result.absorption == (peak,)
    assert result.impedance_pa_s_per_m == (subject.ZETA_AT_MAX_ABSORPTION,)
    assert result.clamped == (False,)


def test_catalog_flatly_extends_end_bands_and_marks_only_outside_points() -> None:
    """抓量測範圍外改成報錯、不平坦延伸，或把中心點也標成延伸。"""
    catalog = subject.CatalogAbsorption(
        material_id="ends",
        band_center_hz=(100.0, 1000.0),
        absorption=(0.2, 0.6),
    )
    result = subject.impedance_on_axis(
        catalog,
        (50.0, 100.0, 200.0, 1000.0, 2000.0),
        400.0,
    )
    hand_interpolated = 0.2 + 0.4 * math.log10(2.0)

    assert result.absorption == (0.2, 0.2, hand_interpolated, 0.6, 0.6)
    assert result.extrapolated == (True, False, False, False, True)


def test_one_band_is_constant_and_only_its_center_is_not_extended() -> None:
    """抓單帶交給一般內插後崩潰，或延伸旗標沒有逐點判斷。"""
    catalog = subject.CatalogAbsorption(
        material_id="one-band",
        band_center_hz=(500.0,),
        absorption=(0.3,),
    )
    result = subject.impedance_on_axis(catalog, (100.0, 500.0, 2000.0), 400.0)

    assert result.absorption == (0.3, 0.3, 0.3)
    assert result.extrapolated == (True, False, True)


def test_result_uses_immutable_tuples_and_frozen_container() -> None:
    """抓回傳值暴露可變陣列或可重新指定的欄位。"""
    catalog = subject.CatalogAbsorption("immutable", (500.0,), (0.3,))
    result = subject.impedance_on_axis(catalog, (500.0,), 400.0)

    assert isinstance(result.frequencies_hz, tuple)
    assert isinstance(result.catalog_absorption, tuple)
    assert isinstance(result.absorption, tuple)
    assert isinstance(result.impedance_pa_s_per_m, tuple)
    assert isinstance(result.extrapolated, tuple)
    assert isinstance(result.clamped, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "absorption", (0.4,))


@pytest.mark.parametrize(
    ("band_center_hz", "absorption", "message"),
    (
        ((100.0,), (), "same length"),
        ((), (), "at least one"),
        ((100.0, 100.0), (0.2, 0.3), "strictly increasing"),
        ((200.0, 100.0), (0.2, 0.3), "strictly increasing"),
        ((0.0,), (0.2,), "band 1 frequency"),
        ((-100.0,), (0.2,), "band 1 frequency"),
        ((math.nan,), (0.2,), "band 1 frequency"),
        ((math.inf,), (0.2,), "band 1 frequency"),
    ),
)
def test_catalog_rejects_invalid_frequency_tables(
    band_center_hz: tuple[float, ...],
    absorption: tuple[float, ...],
    message: str,
) -> None:
    """抓型錄頻率表的空、錯長、非有限、非正或非遞增輸入。"""
    with pytest.raises(ValueError, match=message):
        subject.CatalogAbsorption("bad-frequency", band_center_hz, absorption)


@pytest.mark.parametrize("alpha", (0.0, -0.1, math.nan, math.inf, -math.inf))
def test_catalog_rejects_invalid_absorption_at_named_band(alpha: float) -> None:
    """抓型錄建構時沒有立即驗 α，或訊息沒指出壞在第幾帶。"""
    with pytest.raises(ValueError, match=r"band 2 alpha="):
        subject.CatalogAbsorption(
            "bad-alpha",
            (100.0, 200.0),
            (0.2, alpha),
        )


@pytest.mark.parametrize("rho_c", (0.0, -1.0, math.nan, math.inf, -math.inf))
def test_axis_conversion_rejects_invalid_rho_c(rho_c: float) -> None:
    """抓呼叫端注入的 ρc 非正或非有限仍被拿去乘。"""
    catalog = subject.CatalogAbsorption("valid", (500.0,), (0.3,))
    with pytest.raises(ValueError, match=r"rho_c=.*finite.*> 0"):
        subject.impedance_on_axis(catalog, (500.0,), rho_c)


@pytest.mark.parametrize("frequency_hz", (0.0, -1.0, math.nan, math.inf, -math.inf))
def test_axis_conversion_rejects_invalid_frequency(frequency_hz: float) -> None:
    """抓細軸含非正或非有限頻率仍進 log10 內插。"""
    catalog = subject.CatalogAbsorption("valid", (500.0,), (0.3,))
    with pytest.raises(ValueError, match=r"frequencies_hz point 2=.*finite.*> 0"):
        subject.impedance_on_axis(catalog, (500.0, frequency_hz), 400.0)
