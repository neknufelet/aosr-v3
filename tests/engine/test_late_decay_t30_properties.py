"""晚期衰減 T30 的獨立物理性質考卷。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aosr.config import art_lane
from aosr.geometry.shoebox import Wall
from aosr.physics import late_decay, late_energy
from tests.engine._precision_contracts import contract_value


_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_CASES = ("flat", "varied", "lowabs")
_TOLERANCE_REL = contract_value("late_decay_t30_property")


def _within_property_contract(actual: float, expected: float) -> bool:
    return abs(actual - expected) / abs(expected) <= _TOLERANCE_REL


def _uniform_impedance_inputs(multiple_of_rho_c: float) -> late_energy.LateEnergyInputs:
    base = late_energy.load_late_energy_inputs(
        _ROOT / "blueprint" / "reference_art_flat.json"
    )
    impedance = complex(multiple_of_rho_c * base.rho_c_pa_s_per_m)
    return late_energy.LateEnergyInputs(
        room=base.room,
        rho_c_pa_s_per_m=base.rho_c_pa_s_per_m,
        frequencies_hz=base.frequencies_hz,
        impedance_by_wall={
            wall: tuple(impedance for _frequency in base.frequencies_hz)
            for wall in Wall.wall_names()
        },
        n_per_wall=base.n_per_wall,
        domain_alpha_bar_max=math.inf,
    )


def _minimum_decay_levels_db(
    inputs: late_energy.LateEnergyInputs,
) -> tuple[float, ...]:
    problem = late_energy._reflection_problem(inputs)
    roots = late_decay._exact_roots(problem.transfer)
    levels = late_decay._decay_level(
        late_decay._order_decay(problem.transfer, problem.patches.areas),
        roots,
    )
    return tuple(float(value) for value in np.min(levels, axis=0))


@pytest.mark.parametrize("expected_t60_s", (0.1, 0.3, 1.0, 3.0))
@pytest.mark.parametrize("collision_frequency_hz", (128.625, 50.0, 400.0))
def test_shared_fit_recovers_hand_derived_linear_decay(
    expected_t60_s: float,
    collision_frequency_hz: float,
) -> None:
    """抓 T20、T30 未從同一支擬合回復單一指數衰減的已知 T60。"""
    order = np.arange(1, art_lane.ART_NEUMANN_K_MAX + 1, dtype=np.float64)
    level_db = (-60.0 * (order / collision_frequency_hz) / expected_t60_s)[:, None]

    t20 = late_decay._fit_decay(
        level_db,
        collision_frequency_hz,
        lower_db=art_lane.ART_WLS_T20_LO_DB,
    )
    t30 = late_decay._fit_decay(
        level_db,
        collision_frequency_hz,
        lower_db=art_lane.ART_WLS_T30_LO_DB,
    )

    assert _within_property_contract(float(t20.t60_s[0]), expected_t60_s)
    assert _within_property_contract(float(t30.t60_s[0]), expected_t60_s)


def test_mutant_beyond_tolerance_is_red() -> None:
    """同一個性質裁判對界線內真值判綠、界線外突變判紅。"""
    expected = 1.0

    assert _within_property_contract(expected * (1.0 + _TOLERANCE_REL / 2.0), expected)
    assert not _within_property_contract(expected * (1.0 + 2.0 * _TOLERANCE_REL), expected)


def test_t30_window_reaches_below_the_t20_window() -> None:
    """抓 T30 下緣誤改成 T20 的 −25 dB，漏掉曲線較慢的 −25～−35 dB 段。"""
    collision_frequency_hz = 128.625
    order = np.arange(1, art_lane.ART_NEUMANN_K_MAX + 1, dtype=np.float64)
    time_s = order / collision_frequency_hz
    early_slope_db_per_s = 200.0
    late_slope_db_per_s = 100.0
    crossover_s = 25.0 / early_slope_db_per_s
    level_db = np.where(
        time_s <= crossover_s,
        -early_slope_db_per_s * time_s,
        -25.0 - late_slope_db_per_s * (time_s - crossover_s),
    )[:, None]

    t20_s = float(
        late_decay._fit_decay(
            level_db,
            collision_frequency_hz,
            lower_db=art_lane.ART_WLS_T20_LO_DB,
        ).t60_s[0]
    )
    t30_s = float(
        late_decay._fit_decay(
            level_db,
            collision_frequency_hz,
            lower_db=art_lane.ART_WLS_T30_LO_DB,
        ).t60_s[0]
    )

    assert t30_s > 1.25 * t20_s


@pytest.mark.parametrize("case_name", _REFERENCE_CASES)
def test_combined_solver_wires_t30_to_the_accepted_window(case_name: str) -> None:
    """抓正式入口把 T30 誤接成 T20 下緣，或在入口內另寫一個下緣。"""
    inputs = late_energy.load_late_energy_inputs(
        _ROOT / "blueprint" / f"reference_art_decay_{case_name}.json"
    )
    sound_speed_m_s = 343.0
    problem = late_energy._reflection_problem(inputs)
    collision_frequency_hz = late_decay._collision_frequency(inputs, sound_speed_m_s)
    roots = late_decay._exact_roots(problem.transfer)
    level_db = late_decay._decay_level(
        late_decay._order_decay(problem.transfer, problem.patches.areas),
        roots,
    )
    expected_t30 = late_decay._fit_decay(
        level_db,
        collision_frequency_hz,
        lower_db=art_lane.ART_WLS_T30_LO_DB,
    )
    t20_window = late_decay._fit_decay(
        level_db,
        collision_frequency_hz,
        lower_db=art_lane.ART_WLS_T20_LO_DB,
    )

    combined = late_decay.solve_late_decay(inputs, sound_speed_m_s=sound_speed_m_s)
    actual_t30 = tuple(band.t30_s for band in combined.bands)
    assert actual_t30 == tuple(float(value) for value in expected_t30.t60_s)
    if case_name == "varied":
        assert actual_t30 != tuple(float(value) for value in t20_window.t60_s)


def test_t30_window_constants_match_accepted_decision() -> None:
    """產品視窗常數須等於 stage-nine-three-lane-stitch-and-report-with-interference 第 8 條。"""
    assert art_lane.ART_WLS_T30_LO_DB == -35.0
    assert art_lane.ART_WLS_T20_HI_DB == -5.0


def test_combined_solver_preserves_t20_and_reports_t30() -> None:
    """抓公開解只回一種殘響時間，或擴充時改壞既有 T20。"""
    inputs = late_energy.load_late_energy_inputs(
        _ROOT / "blueprint" / "reference_art_flat.json"
    )
    t20_only = late_decay.solve_late_decay_t20(inputs, sound_speed_m_s=343.0)
    combined = late_decay.solve_late_decay(inputs, sound_speed_m_s=343.0)

    assert combined.orders_used == t20_only.orders_used
    assert tuple(band.t20_s for band in combined.bands) == tuple(
        band.t20_s for band in t20_only.bands
    )
    for band in combined.bands:
        assert band.t30_s is not None
        assert band.t30_slope_db_per_s is not None
        assert band.t30_soft_weight_sum is not None
        assert band.t30_s > 0.0
        assert band.t30_s == -60.0 / band.t30_slope_db_per_s
        assert band.t30_soft_weight_sum > late_decay.ART_WLS_MIN_WEIGHT


def test_combined_solver_rejects_a_fully_absorbing_room() -> None:
    """抓全吸音的零斜率被 T30 公開入口靜靜換成備援值。"""
    inputs = late_energy.load_late_energy_inputs(
        _ROOT / "blueprint" / "reference_art_flat.json"
    )
    matched = {
        wall: tuple(complex(inputs.rho_c_pa_s_per_m) for _ in inputs.frequencies_hz)
        for wall in Wall.wall_names()
    }
    absorbing = late_energy.LateEnergyInputs(
        room=inputs.room,
        rho_c_pa_s_per_m=inputs.rho_c_pa_s_per_m,
        frequencies_hz=inputs.frequencies_hz,
        impedance_by_wall=matched,
        n_per_wall=inputs.n_per_wall,
        domain_alpha_bar_max=inputs.domain_alpha_bar_max,
    )

    with pytest.raises(ValueError, match="T20 擬合無效"):
        late_decay.solve_late_decay(absorbing, sound_speed_m_s=343.0)


def test_invalid_t30_fit_names_the_window() -> None:
    """抓 T30 無效時的例外只說擬合失敗，無法辨認是哪個視窗。"""
    level_db = np.zeros((art_lane.ART_NEUMANN_K_MAX, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="T30 擬合無效"):
        late_decay._fit_decay(
            level_db,
            128.625,
            lower_db=art_lane.ART_WLS_T30_LO_DB,
        )


def test_alpha_near_0026_reaches_t20_but_rejects_unreached_t30() -> None:
    """抓 256 階曲線沒到 −35 dB 時仍用軟視窗算出假的 T30。"""
    inputs = _uniform_impedance_inputs(150.0)
    minimums = _minimum_decay_levels_db(inputs)

    assert all(value <= art_lane.ART_WLS_T20_LO_DB for value in minimums)
    assert all(value > art_lane.ART_WLS_T30_LO_DB for value in minimums)
    late_decay.solve_late_decay_t20(inputs, sound_speed_m_s=343.0)
    with pytest.raises(late_decay.DecayRangeError, match=r"T30.*125 Hz") as caught:
        late_decay.solve_late_decay(inputs, sound_speed_m_s=343.0)
    assert caught.value.lower_db == art_lane.ART_WLS_T30_LO_DB
    assert "第 256 階最低" in str(caught.value)


def test_alpha_near_002_rejects_unreached_t20_before_t30() -> None:
    """抓 256 階曲線連 −25 dB 都沒到時仍回傳 T20 或 T30。"""
    inputs = _uniform_impedance_inputs(200.0)
    minimums = _minimum_decay_levels_db(inputs)

    assert all(value > art_lane.ART_WLS_T20_LO_DB for value in minimums)
    with pytest.raises(late_decay.DecayRangeError, match=r"T20.*125 Hz"):
        late_decay.solve_late_decay_t20(inputs, sound_speed_m_s=343.0)
    with pytest.raises(late_decay.DecayRangeError, match=r"T20.*125 Hz"):
        late_decay.solve_late_decay(inputs, sound_speed_m_s=343.0)
