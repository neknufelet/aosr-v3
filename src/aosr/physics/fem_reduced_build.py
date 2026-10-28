"""替身建模：起點快照、殘差貪婪補點、實數 SVD、投影（#561）。

照 #610 實驗那一版：起點是平均挑的幾組訓練材料 × 平均挑的幾個頻點；之後每輪用固定種子從還沒解過的
（材料、頻點）抽候選，以完整矩陣直接形成 ``b − A(Vq)`` 量替身殘差，取最大的一批完整解補進快照。
每輪補完快照、重做基底之後先抽候選量殘差，再決定停不停：候選用完記「候選用完」；抽到的最大殘差不超過停止值記
「抽到的候選殘差都在停止值內」（同時到預算也記這個，殘差是量到的事實）；否則完整解用滿預算記「預算停止、殘差目標未達」
（這時殘差真的量過、真的沒達，那一輪準備了下一批但不解，``solved`` 記 0）。停止值只是選點用的，不是驗證門檻；
替身準不準由 :mod:`aosr.physics.fem_reduced_check` 用留出材料對完整解判。
"""
from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aosr.config.fem_reduced import ReducedModelSettings
from aosr.physics.fem_helmholtz import FrequencyWallImpedances
from aosr.physics.fem_reduced import (
    BuildLedger, FullOperators, FullSolver, GreedyRound, ImpedanceRows, ReducedMatrices, ReducedModel,
    checked_axis, elapsed_since, even_indices, impedance_rows, input_hash, project, relative_residual,
    snapshot_basis, system_coefficients, walls_at,
)

Point2 = tuple[int, int]


@dataclass
class _Build:
    """建模途中的可變狀態；只活在 :func:`build_reduced_model` 一次呼叫裡。"""

    operators: FullOperators
    axis: tuple[float, ...]
    training: tuple[ImpedanceRows, ...]
    settings: ReducedModelSettings
    solver: FullSolver
    snapshots: list[NDArray[np.complex128]] = field(default_factory=list)
    seen: set[Point2] = field(default_factory=set)
    rounds: list[GreedyRound] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=lambda: dict.fromkeys(
        ("snapshot", "svd", "projection", "greedy"), 0.0))

    def solve(self, pending: Sequence[Point2]) -> None:
        """完整解（解次上限內）並補進快照；上一輪記下實際解了幾點。"""
        start = time.perf_counter()
        chosen = pending[:max(0, self.settings.max_full_solves - self.solver.solves)]
        for material, index in chosen:
            solutions = self.solver.solve(self.axis[index], walls_at(self.training[material], index))
            self.snapshots.extend(solutions)
            self.seen.add((material, index))
        if self.rounds:
            self.rounds[-1] = dataclasses.replace(self.rounds[-1], solved=len(chosen))
        self.seconds["snapshot"] += elapsed_since(start)

    def reduce(self) -> tuple[NDArray[np.float64], ReducedMatrices]:
        start = time.perf_counter()
        basis = snapshot_basis(self.snapshots, self.settings.svd_relative_cutoff)
        self.seconds["svd"] += elapsed_since(start)
        start = time.perf_counter()
        matrices = project(self.operators, basis)
        self.seconds["projection"] += elapsed_since(start)
        return basis, matrices

    def residual(self, basis: NDArray[np.float64], matrices: ReducedMatrices, point: Point2) -> float:
        """完整矩陣乘替身重建的場，逐支喇叭取相對殘差的最大值。"""
        material, index = point
        walls = walls_at(self.training[material], index)
        operators = self.operators
        k_squared, coefficients = system_coefficients(
            self.axis[index], walls, density_kg_m3=operators.density_kg_m3,
            sound_speed_m_s=operators.sound_speed_m_s)
        system = self.solver.system(self.axis[index], walls)
        coordinates = np.linalg.solve(
            matrices.stiffness - k_squared * matrices.mass + np.tensordot(coefficients, matrices.boundary, axes=1),
            matrices.sources.T)
        fields = basis @ coordinates
        return max(relative_residual(system, fields[:, column], load)
                   for column, load in enumerate(operators.sources.values()))

    def greedy_round(self, rng: np.random.Generator, basis: NDArray[np.float64],
                     matrices: ReducedMatrices) -> tuple[float, list[Point2]] | None:
        """抽候選、量殘差、記一輪；候選用完回 None。"""
        pool = [(material, index) for material in range(len(self.training))
                for index in range(len(self.axis)) if (material, index) not in self.seen]
        if not pool:
            return None
        start = time.perf_counter()
        chosen = rng.choice(len(pool), size=min(self.settings.candidates_per_round, len(pool)), replace=False)
        scored = sorted(((self.residual(basis, matrices, pool[int(i)]), pool[int(i)]) for i in chosen),
                        key=lambda item: item[0], reverse=True)
        self.seconds["greedy"] += elapsed_since(start)
        selected = [point for _, point in scored[:self.settings.batch]]
        values = [value for value, _ in scored]
        self.rounds.append(GreedyRound(
            round=len(self.rounds), snapshots=len(self.snapshots), rank=basis.shape[1],
            candidates=len(scored), max_residual=values[0], median_residual=float(np.median(values)),
            proposed=len(selected), solved=0))
        return values[0], selected


def build_reduced_model(
    operators: FullOperators, frequencies_hz: Sequence[float],
    training_impedances: Sequence[FrequencyWallImpedances], *, settings: ReducedModelSettings,
) -> ReducedModel:
    """完整解快照（每點重新分解、殘差超過上限就報錯）→ 殘差貪婪補點 → 實數 SVD → 投影的小矩陣與建模帳。

    ``training_impedances`` 是呼叫端算好的一批材料，每組六面各一條逐頻阻抗（``Wall`` 對整條頻率軸的列，
    或整軸固定值、``None`` 表剛性），全部先過產品阻抗閘門。
    """
    start = time.perf_counter()
    axis = checked_axis(frequencies_hz)
    training = impedance_rows(training_impedances, axis)
    build = _Build(operators=operators, axis=axis, training=training, settings=settings,
                   solver=FullSolver(operators, max_relative_residual=settings.full_solve_max_relative_residual))
    rng = np.random.default_rng(settings.greedy_seed)
    pending = [(material, index)
               for material in even_indices(settings.initial_impedance_count, len(training))
               for index in even_indices(settings.initial_frequency_count, len(axis))]
    while True:
        build.solve(pending)
        basis, matrices = build.reduce()
        # 先量這份基底的殘差再判預算：到上限時記「殘差目標未達」之前，殘差一定量過。
        outcome = build.greedy_round(rng, basis, matrices)
        if outcome is None:
            stop = "候選用完"
            break
        largest, pending = outcome
        if largest <= settings.stop_residual:
            stop = "抽到的候選殘差都在停止值內"
            break
        if build.solver.solves >= settings.max_full_solves:
            stop = "預算停止、殘差目標未達"
            break
    ledger = BuildLedger(
        full_solves=build.solver.solves, snapshots=len(build.snapshots), rank=basis.shape[1],
        rounds=tuple(build.rounds), stop_reason=stop,
        seconds={**build.seconds, "total": elapsed_since(start)}, settings=settings)
    return ReducedModel(
        matrices=matrices, source_names=tuple(operators.sources), receiver_names=tuple(operators.receivers),
        frequencies_hz=axis, density_kg_m3=operators.density_kg_m3, sound_speed_m_s=operators.sound_speed_m_s,
        mesh_hash=operators.mesh_hash, physics_identity=operators.physics_identity,
        input_hash=input_hash(operators, axis, training, settings), ledger=ledger)

