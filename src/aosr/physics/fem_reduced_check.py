"""替身對完整解：留出材料逐頻完整解，量每個座位的最大 dB 差與 1/3 八度帶能量差（#561）。

帶界用以 1000 Hz 為基準的正確二進位 1/3 八度：中心 ``1000·2^(n/3)``、帶界 ``1000·2^((2n±1)/6)``，相鄰兩帶共用同一個界值；
帶內頻點取閉區間（落在帶界上的點兩帶都算）。帶界整個在頻率軸裡的叫「完整覆蓋」，判尺只看這些帶；
軸端只覆蓋一部分的帶另列，不判。尺（精度契約）由呼叫端從登記簿讀進來傳給 :func:`within_reduced_contract`。

報告另記完整模型的喇叭與座位座標（每個座標一個 ``float.hex()``）：替身矩陣是那幾個點的點源與取值列，
網格雜湊與物理身分都不含位置，驗證收據靠這兩格把替身綁到建模那份方案的位置（#561 辛一審查 B1）。
"""
from __future__ import annotations

import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from aosr.physics.fem_full_batch import FullSolveBatch, checked_workers, collect_refinement_steps
from aosr.physics.dense_threads import checked_dense_threads, dense_blas_threads
from aosr.geometry.shoebox import Point
from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
from aosr.physics.fem_reduced import (
    FullOperators, ReducedModel, checked_axis, evaluate_reduced, impedance_rows, system_coefficients, walls_at,
)
from aosr.physics.fem_reduced_store import computation_digest, matrices_digest

THIRD_OCTAVE_REFERENCE_HZ = 1000.0
PointHex = tuple[str, str, str]


def points_hex(points: Mapping[str, Point]) -> dict[str, PointHex]:
    """名字 → 三個座標的 ``float.hex()``（逐位，不經十進位）。"""
    return {name: (float(point.x).hex(), float(point.y).hex(), float(point.z).hex()) for name, point in points.items()}


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

    聲壓陣列形狀是 (材料, 頻點, 喇叭, 座位)。``model_digest`` 是被驗那份替身的模型內容摘要
    （:func:`aosr.physics.fem_reduced_store.matrices_digest`），驗證收據靠它確認報告是對哪一份做的。
    ``computation_digest`` 是驗證當下被驗模型的計算條件指紋（:func:`aosr.physics.fem_reduced_store.computation_digest`：
    矩陣、頻率軸、空氣、喇叭與座位序、出身），驗證收據對載入的模型重算比對，矩陣相同但條件不同的替身寫不成收據（#983）。
    """

    frequency_db_by_seat: Mapping[str, float]
    band_db_by_seat: Mapping[str, float]
    partial_band_db_by_seat: Mapping[str, float]
    bands: tuple[Band, ...]
    full_solves: int
    full_pressure: NDArray[np.complex128]
    reduced_pressure: NDArray[np.complex128]
    model_digest: str
    computation_digest: str
    source_points: Mapping[str, PointHex]
    receiver_points: Mapping[str, PointHex]


@dataclass(frozen=True)
class SurrogateSpotCheck:
    """算式漂移抽查：頻點、跨座位喇叭的最大 dB 差、登記簿的逐頻尺與耗時。"""

    frequencies_hz: tuple[float, ...]
    max_db: float
    ruler_name: str
    ruler_value: float
    passed: bool
    seconds: float


def spot_check_reduced(model: ReducedModel, operators: FullOperators, impedances: FrequencyBoundaryImpedances,
                       frequencies_hz: Sequence[float], *, ruler_name: str, frequency_db: float,
                       workers: int = 1) -> SurrogateSpotCheck:
    """單組材料的指定頻點：中央處理器完整解對替身，只判呼叫端從登記簿讀到的逐頻尺。

    稀疏頻點不構成完整八度帶；獨立於留出驗證，不叫 ``validate_reduced``。
    """
    clock = time.perf_counter()
    checked_workers(workers)
    if (set(model.source_names) != set(operators.sources) or len(model.source_names) != len(operators.sources)
            or set(model.receiver_names) != set(operators.receivers) or len(model.receiver_names) != len(operators.receivers)):
        raise ValueError("替身的喇叭或座位跟完整模型不同")
    operators = replace(operators, sources={name: operators.sources[name] for name in model.source_names},
                        receivers={name: operators.receivers[name] for name in model.receiver_names})
    _checked_pairing(model, operators)
    axis = checked_axis(frequencies_hz)
    rows = impedance_rows([impedances], axis, patch_keys=operators.patch_keys)[0]
    jobs = [(frequency, walls_at(rows, index)) for index, frequency in enumerate(axis)]
    receivers = np.asarray(list(operators.receivers.values()), dtype=np.float64)
    with FullSolveBatch(operators, max_relative_residual=model.ledger.settings.full_solve_max_relative_residual,
                       workers=workers, device="cpu") as batch:
        solutions = (batch.map(jobs) if workers > 1 else
                     (batch.solver.solve(frequency, walls) for frequency, walls in jobs))
        full = np.asarray([solution @ receivers.T for solution in solutions], dtype=np.complex128)
    with dense_blas_threads(checked_dense_threads(None)):
        reduced = []
        for frequency, walls in jobs:
            k_squared, coefficients = system_coefficients(
                frequency, walls, density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s,
                patch_keys=model.patch_keys)
            reduced.append(evaluate_reduced(model.matrices, k_squared, coefficients))
    with np.errstate(divide="ignore", invalid="ignore"):
        difference = np.abs(20.0 * np.log10(np.abs(reduced) / np.abs(full)))
    maximum = float(np.max(difference))
    return SurrogateSpotCheck(axis, maximum, ruler_name, frequency_db,
                              math.isfinite(maximum) and maximum <= frequency_db, time.perf_counter() - clock)


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
    if model.patch_keys != operators.patch_keys:
        raise ValueError("替身的面片清單跟完整模型不同")
    if model.mesh_hash != operators.mesh_hash:
        raise ValueError("替身的網格雜湊跟完整模型不同")
    if model.source_names != tuple(operators.sources) or model.receiver_names != tuple(operators.receivers):
        raise ValueError("替身的喇叭或座位跟完整模型不同")
    if set(operators.source_points) != set(operators.sources) or set(operators.receiver_points) != set(operators.receivers):
        raise ValueError("完整模型的喇叭或座位座標跟它的點源、取值列名字不同")
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
    model: ReducedModel, operators: FullOperators, holdout_impedances: Sequence[FrequencyBoundaryImpedances],
    *, workers: int = 1, device: Literal["cpu", "gpu"] = "cpu",
    refinement_steps: list[int] | None = None,
    dense_threads: int | None = None,
) -> ValidationReport:
    """每組留出材料、每個頻點都完整解一次（殘差上限沿用建模當次的設定），跟替身比。"""
    checked_workers(workers)
    threads = checked_dense_threads(dense_threads)
    sys.stderr.write(f"完整解 {workers} 行程、裝置 {device}、稠密段 {threads} 緒\n")
    _checked_pairing(model, operators)
    axis = model.frequencies_hz
    holdouts = impedance_rows(holdout_impedances, axis, patch_keys=operators.patch_keys)
    with FullSolveBatch(operators, max_relative_residual=model.ledger.settings.full_solve_max_relative_residual,
                       workers=workers, device=device) as batch_solver:
        solver = batch_solver.solver
        parallel = (batch_solver.map([(frequency, walls_at(rows, index))
                                      for rows in holdouts for index, frequency in enumerate(axis)])
                    if workers > 1 else None)
        receivers = np.asarray(list(operators.receivers.values()), dtype=np.float64)
        full = np.empty((len(holdouts), len(axis), len(model.source_names), len(model.receiver_names)),
                        dtype=np.complex128)
        reduced = np.empty_like(full)
        for material, rows in enumerate(holdouts):
            for index, frequency in enumerate(axis):
                walls = walls_at(rows, index)
                solutions = solver.solve(frequency, walls) if parallel is None else next(parallel)
                full[material, index] = solutions @ receivers.T
        collect_refinement_steps(solver, refinement_steps)
    # 先收齊完整解聲壓，讓有待辦數上限的平行佇列持續前進；只保留報告原有的聲壓陣列。
    with dense_blas_threads(threads):
        for material, rows in enumerate(holdouts):
            for index, frequency in enumerate(axis):
                k_squared, coefficients = system_coefficients(
                    frequency, walls_at(rows, index), density_kg_m3=model.density_kg_m3,
                    sound_speed_m_s=model.sound_speed_m_s, patch_keys=model.patch_keys)
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
        bands=bands, full_solves=solver.solves, full_pressure=full, reduced_pressure=reduced,
        model_digest=matrices_digest(model.matrices),
        computation_digest=computation_digest(model),
        source_points=points_hex({name: operators.source_points[name] for name in model.source_names}),
        receiver_points=points_hex({name: operators.receiver_points[name] for name in names}))


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
