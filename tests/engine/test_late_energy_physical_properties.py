"""晚期混響改用無規入射吸音率後的獨立物理性質考卷。"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aosr.config.source_reference import DIFFUSE_MONOPOLE_4PI
from aosr.geometry.shoebox import Room, Wall
from aosr.materials import catalog_absorption
from aosr.physics import late_decay, late_energy
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value


_ROOT = Path(__file__).resolve().parents[2]
_UNIFORM_CASES = ("flat", "lowabs")
# 五項物理性質的相對差界線（登記簿那一條）；T20／T30 對精確常數 Eyring 的界線，
# T20 那張紙寫「與晚期混響性質同一常數」，所以讀同一條，不讀合成衰減那一條。
_TOLERANCE_REL = contract_value("late_energy_physical_property")
_DECAY_TOLERANCE_REL = contract_value("late_energy_physical_property")
_INSIDE = (1.0 - MUTANT_MARGIN) * _TOLERANCE_REL
_OUTSIDE = (1.0 + MUTANT_MARGIN) * _TOLERANCE_REL


def _inputs(case_name: str) -> late_energy.LateEnergyInputs:
    return late_energy.load_late_energy_inputs(
        _ROOT / "blueprint" / f"reference_art_{case_name}.json"
    )


def _relative_difference(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


def _normal_incidence_absorption(zeta: float) -> float:
    """考卷自行算實數正規化阻抗的垂直入射吸音率。"""
    return 4.0 * zeta / (zeta + 1.0) ** 2


@pytest.mark.parametrize("n_per_wall", (2, 3, 6))
def test_form_factors_preserve_reciprocity_and_close_every_row(
    n_per_wall: int,
) -> None:
    """抓交換矩陣漏做雙邊對稱縮放，因而破壞面積加權互易或列和。"""
    patches = late_energy._patch_geometry(Room(7.0, 4.0, 2.5), n_per_wall)
    form_factors = late_energy._form_factors(patches)
    exchange = patches.areas[:, None] * form_factors
    reciprocity_scale = np.maximum(np.abs(exchange), np.abs(exchange.T))
    reciprocity = np.divide(
        np.abs(exchange - exchange.T),
        reciprocity_scale,
        out=np.zeros_like(exchange),
        where=reciprocity_scale > 0.0,
    )
    row_difference = np.abs(np.sum(form_factors, axis=1) - 1.0)

    assert float(np.max(reciprocity)) <= _TOLERANCE_REL
    assert float(np.max(row_difference)) <= _TOLERANCE_REL


@pytest.mark.parametrize("case_name", _UNIFORM_CASES)
def test_uniform_raw_energy_matches_sabine_diffuse_field(case_name: str) -> None:
    """抓無規入射吸音率、源項、正規化或 Eyring 前 raw 能量接錯。"""
    inputs = _inputs(case_name)
    result = late_energy.solve_late_energy(inputs)
    surface_area = 2.0 * (
        inputs.room.Lx * inputs.room.Ly
        + inputs.room.Ly * inputs.room.Lz
        + inputs.room.Lx * inputs.room.Lz
    )
    for band in result.bands:
        alpha = band.alpha_bar
        sabine = 4.0 * DIFFUSE_MONOPOLE_4PI * (1.0 - alpha) / (
            surface_area * alpha
        )
        assert (
            _relative_difference(band.raw_reverberant_energy, sabine)
            <= _TOLERANCE_REL
        )


def test_steady_state_patch_absorption_equals_source_power() -> None:
    """抓反射算子方向、互易、源項或 patch 吸收功率接錯。

    ``B=(I-diag(1-alpha)F)^-1 E`` 是 patch 出射度，``H=F B`` 是入射度；
    每格吸收功率因此是 ``A_i alpha_i H_i``，總和要等於 ``sum(A_i E_i)``。
    """
    inputs = _inputs("varied")
    problem = late_energy._reflection_problem(inputs)
    form_factors = late_energy._form_factors(problem.patches)
    alpha_patch = problem.alpha_by_wall[problem.patches.wall_index, :]
    source = np.full(
        problem.patches.areas.size,
        1.0 / np.sum(problem.patches.areas),
        dtype=np.float64,
    )
    identity = np.eye(problem.patches.areas.size, dtype=np.float64)
    source_power = float(np.dot(problem.patches.areas, source))
    for index in range(len(inputs.frequencies_hz)):
        outgoing = np.linalg.solve(identity - problem.transfer[:, :, index], source)
        incident = form_factors @ outgoing
        absorbed_by_patch = problem.patches.areas * alpha_patch[:, index] * incident
        absorbed_power = float(np.sum(absorbed_by_patch))
        assert (
            _relative_difference(absorbed_power, source_power)
            <= _TOLERANCE_REL
        )


def test_identical_floor_halves_equal_whole_wall_late_energy() -> None:
    """抓同阻抗地板分成兩區後，patch 指派或面積權重改變晚期能量。"""
    inputs = _inputs("varied")
    whole = late_energy.solve_late_energy(inputs)
    problem = late_energy._reflection_problem(inputs)
    form_factors = late_energy._form_factors(problem.patches)
    alpha_patch = problem.alpha_by_wall[problem.patches.wall_index, :].copy()
    floor = np.flatnonzero(problem.patches.wall_index == Wall.all().index(Wall.FLOOR))
    left = floor[problem.patches.centroids[floor, 0] < inputs.room.Lx / 2.0]
    right = floor[problem.patches.centroids[floor, 0] >= inputs.room.Lx / 2.0]
    for frequency_index, impedance in enumerate(
        inputs.impedance_by_wall[Wall.FLOOR.wall_name()]
    ):
        alpha = catalog_absorption.complex_random_incidence_absorption(
            impedance / inputs.rho_c_pa_s_per_m
        )
        alpha_patch[left, frequency_index] = alpha
        alpha_patch[right, frequency_index] = alpha
    transfer = np.asarray(
        (1.0 - alpha_patch)[:, None, :] * form_factors[:, :, None],
        dtype=np.float64,
    )
    split_raw = late_energy._exact_raw_energy(transfer, problem.patches.areas)
    split_alpha_bar = np.einsum(
        "p,pf->f", problem.patches.areas, alpha_patch
    ) / np.sum(problem.patches.areas)
    split = split_raw * np.asarray(
        [late_energy._eyring_ratio(float(alpha)) for alpha in split_alpha_bar]
    )
    for band, split_energy in zip(whole.bands, split, strict=True):
        assert (
            _relative_difference(float(split_energy), band.late_reverberant_energy)
            <= _TOLERANCE_REL
        )


@pytest.mark.parametrize("case_name", _UNIFORM_CASES)
def test_uniform_decay_matches_exact_constant_eyring(case_name: str) -> None:
    """抓 T20/T30 沒有回到同一無規入射吸音率的精確指數衰減時間。"""
    inputs = _inputs(case_name)
    sound_speed_m_s = 343.0
    energy = late_energy.solve_late_energy(inputs)
    decay = late_decay.solve_late_decay(inputs, sound_speed_m_s=sound_speed_m_s)
    volume = inputs.room.Lx * inputs.room.Ly * inputs.room.Lz
    surface_area = 2.0 * (
        inputs.room.Lx * inputs.room.Ly
        + inputs.room.Ly * inputs.room.Lz
        + inputs.room.Lx * inputs.room.Lz
    )
    for energy_band, decay_band in zip(energy.bands, decay.bands, strict=True):
        exact = (
            24.0
            * math.log(10.0)
            / sound_speed_m_s
            * volume
            / (-surface_area * math.log1p(-energy_band.alpha_bar))
        )
        assert (
            _relative_difference(decay_band.t20_s, exact)
            <= _DECAY_TOLERANCE_REL
        )
        assert decay_band.t30_s is not None
        assert (
            _relative_difference(decay_band.t30_s, exact)
            <= _DECAY_TOLERANCE_REL
        )


def test_late_energy_uses_random_incidence_for_complex_impedance() -> None:
    """抓晚期混響未依新紙使用無規入射，或複數阻抗沒有逐帶走 Paris 積分。"""
    base = _inputs("flat")
    impedance = complex(2.0, 0.3) * base.rho_c_pa_s_per_m
    inputs = replace(
        base,
        impedance_by_wall={
            wall: tuple(impedance for _frequency in base.frequencies_hz)
            for wall in Wall.wall_names()
        },
    )
    expected = catalog_absorption.complex_random_incidence_absorption(2.0 + 0.3j)
    result = late_energy.solve_late_energy(inputs)
    assert all(band.alpha_bar == expected for band in result.bands)


@pytest.mark.parametrize(
    "zeta",
    (1.6, 2.0, 4.0, 40.0),
)
def test_random_incidence_exceeds_normal_above_paris_peak(
    zeta: float,
) -> None:
    """換號點就是 Paris 頂點（ζ≈1.567；頂點上兩者只差捨入等級，所以不取頂點本身）；抓硬側 ``alpha_d > alpha_n`` 被倒置。"""
    diffuse = catalog_absorption.complex_random_incidence_absorption(zeta)
    normal = _normal_incidence_absorption(zeta)
    assert diffuse > normal


@pytest.mark.parametrize("zeta", (1.0, 1.4))
def test_random_incidence_is_below_normal_before_paris_peak(zeta: float) -> None:
    """換號點就是 Paris 頂點；抓軟側 ζ=1、1.4 的 ``alpha_d < alpha_n`` 被倒置。"""
    diffuse = catalog_absorption.complex_random_incidence_absorption(zeta)
    normal = _normal_incidence_absorption(zeta)
    assert diffuse < normal


def test_mutant_beyond_tolerance_is_red() -> None:
    """真值在界線內推 δ 判綠、界線外推 δ 判紅（兩側各 δ）。

    受驗的是這一條登記的其中一項性質、也是五項裡最咬得住互易的那一項：面積加權
    交換矩陣的互易失配 ``max|G_ij−G_ji| / max(|G_ij|,|G_ji|)``。控制組把最壞那一格
    往界線內推 δ、變異組往界線外推 δ，判準與產品考卷用的是同一條
    ``≤ _TOLERANCE_REL``；界線外那一組必須判紅（也就是誠實的判決欄記成不通過），
    而整支考卷本身不炸——它不 raise，只是斷言那一格落在界線外。
    """
    patches = late_energy._patch_geometry(Room(7.0, 4.0, 2.5), 3)
    exchange = patches.areas[:, None] * late_energy._form_factors(patches)
    reciprocity_scale = np.maximum(np.abs(exchange), np.abs(exchange.T))
    base = np.divide(
        np.abs(exchange - exchange.T),
        reciprocity_scale,
        out=np.zeros_like(exchange),
        where=reciprocity_scale > 0.0,
    )

    def worst_reciprocity(perturbation: float) -> float:
        """把互相對稱的那一對格之一推開，讓最壞失配落在指定的比值上。"""
        moved = exchange.copy()
        row, column = np.unravel_index(np.argmax(base), base.shape)
        moved[row, column] += perturbation * abs(exchange[row, column])
        scale = np.maximum(np.abs(moved), np.abs(moved.T))
        return float(
            np.max(
                np.divide(
                    np.abs(moved - moved.T),
                    scale,
                    out=np.zeros_like(moved),
                    where=scale > 0.0,
                )
            )
        )

    assert worst_reciprocity(_INSIDE) <= _TOLERANCE_REL
    assert worst_reciprocity(_OUTSIDE) > _TOLERANCE_REL

