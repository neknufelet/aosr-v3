"""替身對完整解：留出材料逐頻完整解，量每個座位的最大 dB 差與 1/3 八度帶能量差（#561）。

帶界用以 1000 Hz 為基準的正確二進位 1/3 八度：中心 ``1000·2^(n/3)``、帶界 ``1000·2^((2n±1)/6)``，相鄰兩帶共用同一個界值；
帶內頻點取閉區間（落在帶界上的點兩帶都算）。帶界整個在頻率軸裡的叫「完整覆蓋」，判尺只看這些帶；
軸端只覆蓋一部分的帶另列，不判。尺（精度契約）由呼叫端從登記簿讀進來傳給 :func:`within_reduced_contract`。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aosr.physics.fem_helmholtz import FrequencyWallImpedances
from aosr.physics.fem_reduced import (
    FullOperators, FullSolver, ReducedModel, evaluate_reduced, impedance_rows, system_coefficients, walls_at,
)

THIRD_OCTAVE_REFERENCE_HZ = 1000.0


@dataclass(frozen=True)
class Band:
    """一個 1/3 八度帶：帶界、帶內頻點索引、是否只部分覆蓋。"""

    lower_hz: float
    upper_hz: float
    indices: tuple[int, ...]
    partial: bool


@dataclass(frozen=True)
class ValidationReport:
    """每個座位（跨材料、喇叭、頻點取最大）的逐頻 dB 差與帶能量 dB 差，以及兩邊的原始複數聲壓。

    聲壓陣列形狀是 (材料, 頻點, 喇叭, 座位)。
    """

    frequency_db_by_seat: Mapping[str, float]
    band_db_by_seat: Mapping[str, float]
    partial_band_db_by_seat: Mapping[str, float]
    bands: tuple[Band, ...]
    full_solves: int
    full_pressure: NDArray[np.complex128]
    reduced_pressure: NDArray[np.complex128]


def third_octave_bands(frequencies_hz: Sequence[float]) -> tuple[Band, ...]:
    """頻率軸碰得到的每個 1/3 八度帶；沒有頻點落在裡面的帶不列。"""
    axis = np.asarray(frequencies_hz, dtype=np.float64)
    first = math.floor(3 * math.log2(float(axis.min()) / THIRD_OCTAVE_REFERENCE_HZ))
    last = math.ceil(3 * math.log2(float(axis.max()) / THIRD_OCTAVE_REFERENCE_HZ))
    # 先算整串帶界，帶 n 的上界與帶 n+1 的下界是同一個浮點數，落在界上的點兩帶都算、不會因捨入只進一帶。
    edges = [THIRD_OCTAVE_REFERENCE_HZ * 2.0 ** ((2 * number - 1) / 6) for number in range(first, last + 2)]
    bands = []
    for lower, upper in zip(edges, edges[1:]):
        inside = np.flatnonzero((axis >= lower) & (axis <= upper))
        if inside.size:
            bands.append(Band(lower_hz=lower, upper_hz=upper, indices=tuple(int(i) for i in inside),
                              partial=bool(lower < axis.min() or upper > axis.max())))
    return tuple(bands)


def _checked_pairing(model: ReducedModel, operators: FullOperators) -> None:
    if model.mesh_hash != operators.mesh_hash:
        raise ValueError("替身的網格雜湊跟完整模型不同")
    if model.source_names != tuple(operators.sources) or model.receiver_names != tuple(operators.receivers):
        raise ValueError("替身的喇叭或座位跟完整模型不同")
    if (model.density_kg_m3 != operators.density_kg_m3
            or model.sound_speed_m_s != operators.sound_speed_m_s):
        raise ValueError("替身的空氣條件跟完整模型不同")


def _band_db(bands: Sequence[Band], full: NDArray[np.complex128], reduced: NDArray[np.complex128],
             partial: bool) -> NDArray[np.float64] | None:
    """選定的帶裡，帶內 |p|² 平均的 dB 差，跨材料、喇叭、帶取絕對值最大，回每座位一格。"""
    chosen = [band for band in bands if band.partial == partial]
    if not chosen:
        return None
    rows = []
    for band in chosen:
        indices = list(band.indices)
        full_energy = np.mean(np.abs(full[:, indices]) ** 2, axis=1)
        reduced_energy = np.mean(np.abs(reduced[:, indices]) ** 2, axis=1)
        rows.append(np.abs(10.0 * np.log10(reduced_energy / full_energy)))
    return np.asarray(np.max(np.asarray(rows), axis=(0, 1, 2)), dtype=np.float64)


def validate_reduced(
    model: ReducedModel, operators: FullOperators, holdout_impedances: Sequence[FrequencyWallImpedances],
) -> ValidationReport:
    """每組留出材料、每個頻點都完整解一次（殘差上限沿用建模當次的設定），跟替身比。"""
    _checked_pairing(model, operators)
    axis = model.frequencies_hz
    holdouts = impedance_rows(holdout_impedances, axis)
    solver = FullSolver(operators, max_relative_residual=model.ledger.settings.full_solve_max_relative_residual)
    receivers = np.asarray(list(operators.receivers.values()), dtype=np.float64)
    full = np.empty((len(holdouts), len(axis), len(model.source_names), len(model.receiver_names)),
                    dtype=np.complex128)
    reduced = np.empty_like(full)
    for material, rows in enumerate(holdouts):
        for index, frequency in enumerate(axis):
            walls = walls_at(rows, index)
            full[material, index] = solver.solve(frequency, walls) @ receivers.T
            k_squared, coefficients = system_coefficients(
                frequency, walls, density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
            reduced[material, index] = evaluate_reduced(model.matrices, k_squared, coefficients)
    frequency_db = np.max(np.abs(20.0 * np.log10(np.abs(reduced) / np.abs(full))), axis=(0, 1, 2))
    bands = third_octave_bands(axis)
    complete = _band_db(bands, full, reduced, partial=False)
    if complete is None:
        raise ValueError("頻率軸沒有任何完整覆蓋的 1/3 八度帶，帶能量差無從判")
    partial = _band_db(bands, full, reduced, partial=True)
    names = model.receiver_names
    return ValidationReport(
        frequency_db_by_seat={name: float(frequency_db[i]) for i, name in enumerate(names)},
        band_db_by_seat={name: float(complete[i]) for i, name in enumerate(names)},
        partial_band_db_by_seat={} if partial is None else {
            name: float(partial[i]) for i, name in enumerate(names)},
        bands=bands, full_solves=solver.solves, full_pressure=full, reduced_pressure=reduced)


def within_reduced_contract(report: ValidationReport, *, frequency_db: float, band_db: float) -> bool:
    """每個座位的逐頻最大 dB 差 ≤ ``frequency_db``，而且完整覆蓋帶的最大帶能量 dB 差 ≤ ``band_db``。

    兩把尺都由呼叫端從精度契約登記簿讀進來；部分覆蓋的軸端帶只報不判。不是有限數一律判不過。
    """
    return seats_within_contract(report.frequency_db_by_seat, report.band_db_by_seat,
                                 frequency_db=frequency_db, band_db=band_db)


def seats_within_contract(frequency_db_by_seat: Mapping[str, float], band_db_by_seat: Mapping[str, float], *,
                          frequency_db: float, band_db: float) -> bool:
    """同 :func:`within_reduced_contract`，只吃每座位的兩格數字；驗證收據的准用閘門用它拿抄下的數字重判。"""
    values = (*frequency_db_by_seat.values(), *band_db_by_seat.values())
    return (all(math.isfinite(value) for value in values)
            and all(value <= frequency_db for value in frequency_db_by_seat.values())
            and all(value <= band_db for value in band_db_by_seat.values()))
