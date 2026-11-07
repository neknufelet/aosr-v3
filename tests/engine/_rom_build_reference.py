"""#1031 控制路徑：逐字保留 2b1b78f6 的交錯驗證，與同一份模型做逐位比較。

不隨產品排程重排；完整解、替身求值與報告算式仍呼叫原來的數值入口。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from aosr.physics.fem_full_batch import FullSolveBatch, checked_workers, collect_refinement_steps
from aosr.physics.fem_reduced import FullOperators, ReducedModel, evaluate_reduced, impedance_rows, system_coefficients, walls_at
from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
from aosr.physics.fem_reduced_check import ValidationReport, _band_db, _checked_pairing, points_hex, third_octave_bands
from aosr.physics.fem_reduced_store import computation_digest, matrices_digest


def legacy_validate(
    model: ReducedModel, operators: FullOperators, holdout_impedances: Sequence[FrequencyBoundaryImpedances],
    *, workers: int = 1, device: Literal["cpu", "gpu"] = "cpu",
    refinement_steps: list[int] | None = None,
) -> ValidationReport:
    """每組留出材料、每個頻點都完整解一次（殘差上限沿用建模當次的設定），跟替身比。"""
    checked_workers(workers)
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
                k_squared, coefficients = system_coefficients(
                    frequency, walls, density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s, patch_keys=model.patch_keys)
                reduced[material, index] = evaluate_reduced(model.matrices, k_squared, coefficients)
        collect_refinement_steps(solver, refinement_steps)
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
