"""晚期衰減 T30 的獨立物理性質考卷。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aosr.config import art_lane
from aosr.geometry.shoebox import Wall
from aosr.physics import late_decay, late_energy


_ROOT = Path(__file__).resolve().parents[2]


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

    t20_relative = abs(float(t20.t60_s[0]) - expected_t60_s) / expected_t60_s
    t30_relative = abs(float(t30.t60_s[0]) - expected_t60_s) / expected_t60_s
    assert t20_relative <= late_decay.LATE_DECAY_SYNTHETIC_PROPERTY_REL
    assert t30_relative <= late_decay.LATE_DECAY_SYNTHETIC_PROPERTY_REL


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

    with pytest.raises(ValueError, match="擬合無效"):
        late_decay.solve_late_decay(absorbing, sound_speed_m_s=343.0)
