"""替身建模：起點快照、殘差貪婪補點、實數 SVD、投影（#561）。

照 #610 實驗那一版：起點是平均挑的幾組訓練材料 × 平均挑的幾個頻點；之後每輪用固定種子從還沒解過的
（材料、頻點）抽候選，以完整算子直接形成 ``b − A(Vq)`` 量替身殘差，取最大的一批完整解補進快照。
每輪補完快照、重做基底之後先抽候選量殘差，再決定停不停：候選用完記「候選用完」；抽到的最大殘差不超過停止值記
「抽到的候選殘差都在停止值內」（同時到預算也記這個，殘差是量到的事實）；否則完整解用滿預算記「預算停止、殘差目標未達」
（這時殘差真的量過、真的沒達，那一輪準備了下一批但不解，``solved`` 記 0）。停止值只是選點用的，不是驗證門檻；
替身準不準由 :mod:`aosr.physics.fem_reduced_check` 用留出材料對完整解判。

補訓練（#1082 段 2-8，:class:`ReducedModelBuilder`）：建完之後再交一批新的訓練材料，接著同一份快照、同一個抽樣器往下補點，
候選只從新的那一批抽（舊的已經量到停止值內）；停止規則與預算照舊，完整解次數跟建模那一段合計算預算。
只用 :func:`build_reduced_model` 不補訓練時，選點、快照、建模帳與替身逐位照舊。
"""
from __future__ import annotations

import dataclasses
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from aosr.config.fem_reduced import ReducedModelSettings
from aosr.physics.dense_threads import checked_dense_threads, dense_blas_threads
from aosr.physics.fem_full_batch import FullSolveBatch, checked_workers
from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
from aosr.physics.fem_reduced import (
    BuildLedger, FullOperators, FullSolver, GreedyRound, ImpedanceRows, ReducedMatrices, ReducedModel,
    checked_axis, elapsed_since, even_indices, impedance_rows, input_hash, project,
    snapshot_basis, system_coefficients, walls_at,
)

Point2 = tuple[int, int]
STOP_CANDIDATES_EXHAUSTED: Final = "候選用完"
STOP_RESIDUAL_MET: Final = "抽到的候選殘差都在停止值內"
STOP_BUDGET: Final = "預算停止、殘差目標未達"
# 補訓練（:meth:`ReducedModelBuilder.extend` 給了 ``max_rounds``）補滿這幾輪就先停，讓呼叫端核過再決定要不要接著補。
STOP_ROUND_LIMIT: Final = "補訓練補滿這幾輪先停、交回去核"


@dataclass(frozen=True)
class BuildProgress:
    """建模途中每一輪量完候選殘差之後交給呼叫端的一筆（只讀、不影響建模；#1098 網頁顯示進度用）。

    ``latest`` 是剛記下的那一輪（候選用完那一次沒有新的一輪，是 None）；這一刻那一輪的點還沒解，``latest.solved`` 一律是 0，
    實際解了幾次要等下一輪開解才寫進建模帳。``seconds`` 是到這一刻各段累計秒數的複本。
    """

    full_solves: int
    max_full_solves: int
    snapshots: int
    rank: int
    stop_residual: float
    latest: GreedyRound | None
    seconds: dict[str, float]
    elapsed_s: float


BuildProgressCallback = Callable[[BuildProgress], None]


@dataclass
class _Build:
    """建模途中的可變狀態；只活在一次建模（:class:`ReducedModelBuilder`）裡。``done`` 是換完整解行程池之前已經解了幾次
    （補訓練那一段另開一池，預算照合計算）。"""

    operators: FullOperators
    axis: tuple[float, ...]
    training: tuple[ImpedanceRows, ...]
    settings: ReducedModelSettings
    solver: FullSolver
    batch_solver: FullSolveBatch
    workers: int
    dense_threads: int | None = None
    done: int = 0
    snapshots: list[NDArray[np.complex128]] = field(default_factory=list)
    seen: set[Point2] = field(default_factory=set)
    rounds: list[GreedyRound] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=lambda: dict.fromkeys(
        ("snapshot", "svd", "projection", "greedy"), 0.0))

    @property
    def solves(self) -> int:
        """到這一刻合計完整解了幾次（含換池之前的）。"""
        return self.done + self.solver.solves

    def solve(self, pending: Sequence[Point2]) -> None:
        """完整解（解次上限內）並補進快照；上一輪記下實際解了幾點。"""
        start = time.perf_counter()
        chosen = pending[:max(0, self.settings.max_full_solves - self.solves)]
        parallel = (self.batch_solver.map([
            (self.axis[index], walls_at(self.training[material], index)) for material, index in chosen])
            if self.workers > 1 else None)
        for material, index in chosen:
            solutions = (self.solver.solve(self.axis[index], walls_at(self.training[material], index))
                         if parallel is None else next(parallel))
            self.snapshots.extend(solutions)
            self.seen.add((material, index))
        if self.rounds:
            self.rounds[-1] = dataclasses.replace(self.rounds[-1], solved=len(chosen))
        self.seconds["snapshot"] += elapsed_since(start)

    def reduce(self) -> tuple[NDArray[np.float64], ReducedMatrices]:
        with dense_blas_threads(self.dense_threads):
            start = time.perf_counter()
            basis = snapshot_basis(self.snapshots, self.settings.svd_relative_cutoff)
            self.seconds["svd"] += elapsed_since(start)
            start = time.perf_counter()
            matrices = project(self.operators, basis)
            self.seconds["projection"] += elapsed_since(start)
        return basis, matrices

    def residual(self, basis: NDArray[np.float64], matrices: ReducedMatrices, point: Point2) -> float:
        """完整算子乘替身重建的場，逐支喇叭取相對殘差的最大值；不組裝完整系統矩陣。"""
        material, index = point
        walls = walls_at(self.training[material], index)
        operators = self.operators
        k_squared, coefficients = system_coefficients(
            self.axis[index], walls, density_kg_m3=operators.density_kg_m3,
            sound_speed_m_s=operators.sound_speed_m_s, patch_keys=operators.patch_keys)
        coordinates = np.linalg.solve(
            matrices.stiffness - k_squared * matrices.mass + np.tensordot(coefficients, matrices.boundary, axes=1),
            matrices.sources.T)
        # 一次實數乘法重建實、虛部，避免 N×r 基底整份升成複數暫存。
        parts = basis @ np.concatenate((coordinates.real, coordinates.imag), axis=1)
        columns = coordinates.shape[1]
        fields = parts[:, :columns] + 1j * parts[:, columns:]
        product = operators.stiffness @ fields - k_squared * (operators.mass @ fields)
        for coefficient, boundary in zip(coefficients, operators.boundary, strict=True):
            if coefficient != 0:
                product += coefficient * (boundary @ fields)
        return max(float(np.linalg.norm(load - product[:, column]) / np.linalg.norm(load))
                   for column, load in enumerate(operators.sources.values()))

    def greedy_round(self, rng: np.random.Generator, basis: NDArray[np.float64],
                     matrices: ReducedMatrices, first: int = 0) -> tuple[float, list[Point2]] | None:
        """抽候選、量殘差、記一輪；候選用完回 None。``first`` 給了只從第幾組材料之後抽（補訓練只抽新的那一批）。"""
        pool = [(material, index) for material in range(first, len(self.training))
                for index in range(len(self.axis)) if (material, index) not in self.seen]
        if not pool:
            return None
        start = time.perf_counter()
        chosen = rng.choice(len(pool), size=min(self.settings.candidates_per_round, len(pool)), replace=False)
        with dense_blas_threads(self.dense_threads):
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


class ReducedModelBuilder:
    """一次建模的建模者：先 :meth:`build`（同 :func:`build_reduced_model`），需要的話再 :meth:`extend` 補訓練（#1082 段 2-8）。

    快照、已解過的點、每一輪紀錄、抽樣器都留在物件裡，補訓練接著往下補；完整解行程池每一段各開一個（段與段之間
    不佔著工作行程），解次照合計算預算。:meth:`release` 放掉快照與基底（不再補訓練時叫，省記憶體）。
    """

    def __init__(self, operators: FullOperators, frequencies_hz: Sequence[float],
                 training_impedances: Sequence[FrequencyBoundaryImpedances], *, settings: ReducedModelSettings,
                 workers: int = 1, dense_threads: int | None = None, progress: BuildProgressCallback | None = None) -> None:
        checked_workers(workers)
        self.threads = checked_dense_threads(dense_threads)
        sys.stderr.write(f"完整解 {workers} 行程、裝置 cpu、稠密段 {self.threads} 緒\n")
        self.started = time.perf_counter()
        self.operators, self.settings, self.workers, self.progress = operators, settings, workers, progress
        self.axis = checked_axis(frequencies_hz)
        self.training = impedance_rows(training_impedances, self.axis, patch_keys=operators.patch_keys)
        self.rng = np.random.default_rng(settings.greedy_seed)
        self._build: _Build | None = None
        self._basis: NDArray[np.float64] | None = None
        self._matrices: ReducedMatrices | None = None
        self._extension_first: int | None = None
        self.stop_reason: str | None = None

    def _report(self, build: _Build, basis: NDArray[np.float64], outcome: tuple[float, list[Point2]] | None) -> None:
        if self.progress is not None:
            self.progress(BuildProgress(
                full_solves=build.solves, max_full_solves=self.settings.max_full_solves,
                snapshots=len(build.snapshots), rank=basis.shape[1], stop_residual=self.settings.stop_residual,
                latest=None if outcome is None else build.rounds[-1], seconds=dict(build.seconds),
                elapsed_s=elapsed_since(self.started)))

    def _stop(self, build: _Build, outcome: tuple[float, list[Point2]] | None) -> str | None:
        """這一輪量完之後停不停（規則見模組說明）；不停回 None。"""
        if outcome is None:
            return STOP_CANDIDATES_EXHAUSTED
        if outcome[0] <= self.settings.stop_residual:
            return STOP_RESIDUAL_MET
        if build.solves >= self.settings.max_full_solves:
            return STOP_BUDGET
        return None

    def _batch(self) -> FullSolveBatch:
        return FullSolveBatch(self.operators, max_relative_residual=self.settings.full_solve_max_relative_residual,
                              workers=self.workers)

    def build(self) -> ReducedModel:
        """起點快照 → 殘差貪婪補點 → 實數 SVD → 投影；選點與結果跟 :func:`build_reduced_model` 逐位相同（它就是叫這一支）。"""
        if self._build is not None:
            raise RuntimeError("這一次建模已經建過了；補訓練用 extend")
        with self._batch() as batch_solver:
            build = _Build(operators=self.operators, axis=self.axis, training=self.training, settings=self.settings,
                           solver=batch_solver.solver, batch_solver=batch_solver, workers=self.workers,
                           dense_threads=self.threads)
            pending = [(material, index)
                       for material in even_indices(self.settings.initial_impedance_count, len(self.training))
                       for index in even_indices(self.settings.initial_frequency_count, len(self.axis))]
            while True:
                build.solve(pending)
                basis, matrices = build.reduce()
                # 先量這份基底的殘差再判預算：到上限時記「殘差目標未達」之前，殘差一定量過。
                outcome = build.greedy_round(self.rng, basis, matrices)
                self._report(build, basis, outcome)
                stop = self._stop(build, outcome)
                if stop is not None:
                    break
                assert outcome is not None
                pending = outcome[1]
        return self._finish(build, basis, matrices, stop)

    def extend(self, training_impedances: Sequence[FrequencyBoundaryImpedances] = (), *,
               max_rounds: int | None = None) -> ReducedModel:
        """補訓練：新的這批材料接在訓練材料後面，用目前的基底從補訓練的材料（第一次補起所有補進來的，不含原本的訓練材料）
        抽候選、量殘差、補點，停止規則與合計預算照舊。``max_rounds`` 給了就最多補這幾輪（每輪一次補點＋重算基底）先停
        （:data:`STOP_ROUND_LIMIT`；補到上限那一輪照舊再量一次殘差、由停止規則判），呼叫端核過再決定要不要接著補——再叫一次、不給新材料
        就從同一個候選池接著補，跟一次補到底選的點與結果逐位相同。

        回補完的替身（輸入雜湊含補進來的材料、建模帳接著記）；預算已經用完就一點都不補、照實記「預算停止」。"""
        build, basis, matrices = self._build, self._basis, self._matrices
        if build is None or basis is None or matrices is None:
            raise RuntimeError("還沒建模、或快照已經放掉了，不能補訓練")
        if max_rounds is not None and max_rounds < 1:
            raise ValueError("max_rounds 至少 1")
        if self._extension_first is None:
            self._extension_first = len(build.training)
        if training_impedances:
            build.training = (*build.training, *impedance_rows(training_impedances, self.axis,
                                                                patch_keys=self.operators.patch_keys))
        self.training = build.training
        rounds = 0
        with self._batch() as batch_solver:
            # 換池：舊池解過的次數記進 done，新池從 0 數，合計照舊。
            build.done, build.solver, build.batch_solver = build.solves, batch_solver.solver, batch_solver
            while True:
                outcome = build.greedy_round(self.rng, basis, matrices, self._extension_first)
                self._report(build, basis, outcome)
                stop = self._stop(build, outcome)
                if stop is not None:
                    break
                assert outcome is not None
                build.solve(outcome[1])
                basis, matrices = build.reduce()
                rounds += 1
                # 到上限那一輪不先停：照一次補到底，再量一次殘差由 _stop 判（殘差達標或預算停止），逐位相同。
                if max_rounds is not None and rounds >= max_rounds and build.solves < self.settings.max_full_solves:
                    stop = STOP_ROUND_LIMIT
                    break
        return self._finish(build, basis, matrices, stop)

    @property
    def full_solves(self) -> int:
        return 0 if self._build is None else self._build.solves

    def release(self) -> None:
        """不再補訓練：放掉快照、基底與已解過的點（替身本身不受影響）。"""
        self._build = self._basis = self._matrices = None

    def _finish(self, build: _Build, basis: NDArray[np.float64], matrices: ReducedMatrices, stop: str) -> ReducedModel:
        self._build, self._basis, self._matrices, self.stop_reason = build, basis, matrices, stop
        operators, settings = self.operators, self.settings
        ledger = BuildLedger(
            full_solves=build.solves, snapshots=len(build.snapshots), rank=basis.shape[1],
            rounds=tuple(build.rounds), stop_reason=stop,
            seconds={**build.seconds, "total": elapsed_since(self.started)}, settings=settings)
        return ReducedModel(
            matrices=matrices, source_names=tuple(operators.sources), receiver_names=tuple(operators.receivers),
            frequencies_hz=self.axis, density_kg_m3=operators.density_kg_m3, sound_speed_m_s=operators.sound_speed_m_s,
            mesh_hash=operators.mesh_hash, physics_identity=operators.physics_identity,
            input_hash=input_hash(operators, self.axis, build.training, settings), ledger=ledger,
            patch_keys=operators.patch_keys)


def build_reduced_model(
    operators: FullOperators, frequencies_hz: Sequence[float],
    training_impedances: Sequence[FrequencyBoundaryImpedances], *, settings: ReducedModelSettings, workers: int = 1,
    dense_threads: int | None = None, progress: BuildProgressCallback | None = None,
) -> ReducedModel:
    """完整解快照（每點重新分解、殘差超過上限就報錯）→ 殘差貪婪補點 → 實數 SVD → 投影的小矩陣與建模帳。

    ``training_impedances`` 是呼叫端算好的一批材料，每組每面片各一條逐頻阻抗（面片鍵對整條頻率軸的列；六個整面仍接受 ``Wall``，
    或整軸固定值、``None`` 表剛性），全部先過產品阻抗閘門。``progress`` 給了就在每一輪量完候選殘差後叫一次
    （:class:`BuildProgress`）；它只讀，選點、快照與結果跟不給逐位相同。
    """
    builder = ReducedModelBuilder(operators, frequencies_hz, training_impedances, settings=settings, workers=workers,
                                  dense_threads=dense_threads, progress=progress)
    model = builder.build()
    builder.release()
    return model
