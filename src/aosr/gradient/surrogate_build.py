"""替一份切法建替身（surrogate，縮減模型）的命令列（#1098）：網頁在獨立服務裡叫它，人也可以直接叫。

``python -m aosr.gradient.surrogate_build --scheme <方案檔> --work <建模資料夾> --slot <格子資料夾> --capabilities <能力表>``

照搜尋開跑那一段（``aosr.gradient.cli`` 的 ``start``）的同一組取法，免得建出來的替身開跑時被拒：
完整模型算子照 ``export_operators`` 的預設網格、頻率軸照方案場景的低頻軸、原牆照 ``run.scene_wall_materials``。
步驟：開跑前核對（方案有低頻軸、原牆能貼）→ 匯出完整模型 → 訓練與留出材料（:mod:`aosr.gradient.surrogate_training`）
→ 建模（停止規則照 ``fem_reduced.toml``：殘差達標或完整解預算用完。``--stop-residual``、``--max-full-solves`` 只換這一次建模的
停止值與預算，設定檔預設不動；換了照實寫進建模帳，建模帳在 manifest 與輸入雜湊裡，驗證收據對的就是它，等於進了替身身分）
→ 一定跑留出驗證、寫驗證收據 → 兩把尺都過才補完整模型指紋 → 用搜尋開跑那一道准用閘門核
（``run.gated_surrogate``、``check_surrogate_matches_scheme``、``check_surrogate_axis_covers_scoring``）→ 整個資料夾換名進格子。
驗證或閘門沒過：照實寫 ``result.json``、留在建模資料夾、不放進格子，離開碼 4。被停（SIGTERM／SIGINT）離開碼 3，不接續；
開始換名進格子之後才到的停止只記下、不打斷（見 :class:`_StopGate`）。
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Final

from aosr import runtime
from aosr.gradient.surrogate_files import (
    EXIT_FAILED, EXIT_INSTALLED, EXIT_NOT_USABLE, EXIT_STOPPED, MODEL_FILE, PROGRESS_FILE, RESULT_FILE, STAGE_DIR,
    TRAINING_FILE, Phase, RoundRecord, SurrogateProgress, SurrogateResult, Verdict,
)

if TYPE_CHECKING:
    from aosr.config.fem_reduced import ReducedModelSettings
    from aosr.config.surrogate_training import SurrogateTrainingSettings
    from aosr.physics.fem_reduced import FullOperators
    from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
    from aosr.physics.fem_reduced_build import BuildProgress
    from aosr.reporting.scheme import Scheme

SIGNALS: Final = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


class _Progress:
    """``progress.json`` 的寫手：每次換一份、同資料夾暫存檔換名（讀的人只看得到完整的舊檔或新檔）。"""

    def __init__(self, path: Path, first: SurrogateProgress) -> None:
        self.path = path
        self.current = first
        self._write()

    def _write(self) -> None:
        from aosr.gradient.run import write_json_atomic
        write_json_atomic(self.path, self.current)

    def update(self, **fields: object) -> None:
        self.current = self.current.model_copy(update={**fields, "updated_at": time.time()})
        self._write()

    def phase(self, phase: Phase, **fields: object) -> None:
        now = time.time()
        self.update(phase=phase, phase_started_at=now, **fields)

    def build_round(self, report: BuildProgress) -> None:
        rounds = self.current.rounds
        if report.latest is not None:
            rounds = (*rounds, RoundRecord(round=report.latest.round, full_solves=report.full_solves, rank=report.rank,
                                           max_residual=report.latest.max_residual, elapsed_s=report.elapsed_s))
        self.update(full_solves=report.full_solves, rank=report.rank, rounds=rounds,
                    snapshot_seconds=report.seconds.get("snapshot"))

    def validation(self, done: int, total: int) -> None:
        self.update(validation_done=done, validation_total=total)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aosr.gradient.surrogate_build", description=__doc__.splitlines()[0])
    parser.add_argument("--scheme", type=Path, required=True, help="方案檔（網頁交的是開跑那一刻的快照）")
    parser.add_argument("--work", type=Path, required=True, help="建模資料夾（進度、結果、先建在這裡的 stage/）")
    parser.add_argument("--slot", type=Path, required=True, help="這份切法的替身格子（建好、核過才換名進來）")
    parser.add_argument("--capabilities", type=Path, required=True, help="能力表（跟搜尋開跑同一份）")
    parser.add_argument("--replace", type=Path, default=None,
                        help="格子裡已經有一份（例如閘門拒收的舊替身）：新的核過之後把舊的整個搬到這裡，再放新的")
    parser.add_argument("--max-full-solves", type=int, default=None,
                        help="這一次建模的完整解預算；不給照 fem_reduced.toml（停止規則不變：殘差達標或預算用完就停）")
    parser.add_argument("--stop-residual", type=float, default=None,
                        help="這一次建模的停止殘差；不給照 fem_reduced.toml（只在這一次生效，照實記進建模帳）")
    parser.add_argument("--workers", type=int, default=None, help="完整解行程數；不給照 surrogate_training.toml")
    parser.add_argument("--dense-threads", type=int, default=None, help="稠密段 BLAS 緒數；不給＝實體核數")
    return parser


def _result(work: Path, result: SurrogateResult) -> None:
    from aosr.gradient.run import write_json_atomic
    write_json_atomic(work / RESULT_FILE, result)


def _rebuild_hint(stop_reason: str) -> str:
    """驗證沒過時重建有沒有用：預算用完才停的，加大預算可能會過；殘差先達標才停的，照同樣設定重建結果一模一樣
    （訓練材料與補點都是固定種子），加預算沒用。"""
    from aosr.physics.fem_reduced_build import STOP_BUDGET

    if stop_reason == STOP_BUDGET:
        return f"建模停止原因：{stop_reason}——加大完整解預算（max_full_solves）重建可能會過"
    return (f"建模停止原因：{stop_reason}——照同樣設定重建結果一模一樣（固定種子），加預算沒用；"
            "要過得給這一次建模更嚴的停止殘差（stop_residual）")


class _StopGate:
    """停止訊號（SIGTERM／SIGINT／SIGHUP）的處理函式：平常收到就打斷（離開碼 3）；``holding`` 打開之後收到只記下、不打斷，
    而且打開了就不再關——放進格子那一刻起這一次建模就是建好了，晚到的停止不准把它翻成「被停」，也不准把退舊、入新、
    寫判決三步打斷在中間（格子空掉或判決跟格子對不上）。用 Python 層的旗標、不用 ``pthread_sigmask``：完整解行程池、
    BLAS 都開緒，作業系統可能把訊號交給沒擋住的緒；Python 的訊號處理函式則一律在主緒跑，旗標擋得全。"""

    def __init__(self) -> None:
        self.holding = False
        self.deferred: list[int] = []

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        if self.holding:
            self.deferred.append(signum)
            return
        raise KeyboardInterrupt


STOPS: Final = _StopGate()


def _install(stage: Path, slot: Path, replace: Path | None) -> str | None:
    """整個資料夾換名進格子；格子已有東西時只在給了 ``--replace`` 才先把舊的搬走（同一顆磁碟的換名，不複製）。
    新的換不進去（例如跨磁碟換名失敗）就把舊的搬回格子再報原本的錯：格子不會因為一次失敗的安裝變空。
    連搬回都失敗時報一個錯、把兩次失敗與舊的現在在哪裡一起講出來（命令列只印最外層那一個錯，不印例外鏈）。"""
    retired = None
    if slot.exists():
        if replace is None:
            raise FileExistsError(f"替身格子 {slot} 已經有東西；要換掉舊的得給 --replace（舊的搬去哪）")
        replace.parent.mkdir(parents=True, exist_ok=True)
        slot.rename(replace)
        retired = replace
    try:
        slot.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(slot)
    except BaseException as error:
        if retired is not None:
            try:
                retired.rename(slot)
            except OSError as again:
                raise OSError(f"新的換不進格子 {slot}（{type(error).__name__}：{error}），舊的也搬不回去"
                              f"（{type(again).__name__}：{again}）：舊的留在 {retired}，要手動搬回 {slot}") from error
        raise
    return None if retired is None else str(retired)


def _commit(gate: _StopGate, stage: Path, slot: Path, replace: Path | None,
            record: Callable[[str | None], None]) -> None:
    """放進格子＋寫 installed 判決是一段不被停打斷的提交：先關住停止訊號（見 :class:`_StopGate`），再換名、再寫判決。"""
    gate.holding = True
    record(_install(stage, slot, replace))


@dataclass
class _Run:
    """一次建模的輸入與寫手：命令列參數解好、設定讀好、進度檔開好。"""

    work: Path
    stage: Path
    slot: Path
    replace: Path | None
    scheme: Scheme
    sampling: SurrogateTrainingSettings
    settings: ReducedModelSettings
    workers: int
    dense_threads: int | None
    progress: _Progress


def _open(args: argparse.Namespace) -> _Run:
    from aosr.config.fem_reduced import load_reduced_model_settings
    from aosr.config.paths import config_path
    from aosr.config.surrogate_training import load_surrogate_training_settings
    from aosr.gradient.boundary import scheme_patch_keys
    from aosr.reporting.scheme import load_scheme

    started = time.time()
    work, slot = args.work.resolve(), args.slot.resolve()
    replace = None if args.replace is None else args.replace.resolve()
    stage = work / STAGE_DIR
    if stage.exists() or (work / RESULT_FILE).exists():
        raise FileExistsError(f"建模資料夾 {work} 已經用過（有 {STAGE_DIR}/ 或 {RESULT_FILE}）；一次建模一個新資料夾")
    if slot.exists() and replace is None:
        raise FileExistsError(f"替身格子 {slot} 已經有東西；要換掉舊的得給 --replace")
    work.mkdir(parents=True, exist_ok=True)
    sampling = load_surrogate_training_settings(config_path("surrogate_training.toml"))
    settings = load_reduced_model_settings(config_path("fem_reduced.toml"))
    overrides = {name: value for name, value in (("max_full_solves", args.max_full_solves),
                                                 ("stop_residual", args.stop_residual)) if value is not None}
    if overrides:
        settings = type(settings).model_validate({**settings.model_dump(), **overrides})
    workers = sampling.compute.workers if args.workers is None else args.workers
    scheme = load_scheme(args.scheme.resolve())
    progress = _Progress(work / PROGRESS_FILE, SurrogateProgress(
        phase="preparing", started_at=started, phase_started_at=started, updated_at=started, workers=workers,
        dense_threads=args.dense_threads, patches=len(scheme_patch_keys(scheme)),
        max_full_solves=settings.max_full_solves, stop_residual=settings.stop_residual))
    return _Run(work, stage, slot, replace, scheme, sampling, settings, workers, args.dense_threads, progress)


def _prepare(job: _Run, capabilities_path: Path) -> tuple[FullOperators, tuple[float, ...], list[FrequencyBoundaryImpedances],
                                                          list[FrequencyBoundaryImpedances]]:
    """開跑前核對、匯出完整模型、算訓練與留出材料的阻抗，訓練說明寫進 stage/。"""
    from aosr.config.capabilities import load_capabilities
    from aosr.config.construction_environment import construction_environment
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.frequency_axis import low_frequency_axis_frequencies
    from aosr.config.material_gradient import load_material_gradient_settings
    from aosr.config.paths import config_path
    from aosr.gradient import run
    from aosr.gradient.cli import keep_signal_handlers
    from aosr.gradient.endpoint import checked_air
    from aosr.gradient.surrogate_training import TrainingSets
    from aosr.reporting.fem_reduced_scheme import export_operators

    scheme = job.scheme
    if scheme.scene.low_frequency_axis is None:
        raise ValueError("方案場景沒有 low_frequency_axis（低頻軸）：搜尋開跑會拒收，不建替身")
    capabilities = load_capabilities(capabilities_path.resolve())
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    saved = run.scene_wall_materials(scheme, capabilities=capabilities, directivity=directivity)
    run.require_supported_original_walls(saved.values())
    axis = low_frequency_axis_frequencies(scheme.scene.low_frequency_axis)[0]
    with keep_signal_handlers():
        operators = export_operators(scheme, capabilities=capabilities, directivity=directivity)
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    checked_air(air, density_kg_m3=operators.density_kg_m3, sound_speed_m_s=operators.sound_speed_m_s, source="完整模型算子")
    sets = TrainingSets(saved, load_material_gradient_settings(config_path("material_gradient.toml")), job.sampling,
                        air=air, ranges=ranges, frequencies_hz=axis)
    training_sets, holdout_sets = sets.training_sets(), sets.holdout_sets()
    training = [sets.impedances(item) for item in training_sets]
    holdout = [sets.impedances(item) for item in holdout_sets]
    job.stage.mkdir(parents=True)
    with (job.stage / TRAINING_FILE).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(sets.document(training_sets, holdout_sets), ensure_ascii=False, allow_nan=False) + "\n")
    return operators, axis, training, holdout


def build(args: argparse.Namespace) -> int:
    from aosr.gradient import run
    from aosr.physics.fem_reduced_build import build_reduced_model
    from aosr.physics.fem_reduced_check import points_hex, validate_reduced, within_reduced_contract
    from aosr.physics.fem_reduced_receipt import save_validation_receipt
    from aosr.physics.fem_reduced_store import save_operators_stamp, save_reduced

    job = _open(args)
    operators, axis, training, holdout = _prepare(job, args.capabilities)
    job.progress.phase("building", dofs=int(operators.p2.basis.N), axis_points=len(axis), training_sets=len(training),
                       holdout_sets=len(holdout))
    model = build_reduced_model(operators, axis, training, settings=job.settings, workers=job.workers,
                                dense_threads=job.dense_threads, progress=job.progress.build_round)
    npz = job.stage / MODEL_FILE
    save_reduced(model, npz)
    ledger = model.ledger
    job.progress.phase("validating", stop_reason=ledger.stop_reason, build_seconds=ledger.seconds["total"],
                       full_solves=ledger.full_solves, rank=ledger.rank, validation_done=0,
                       validation_total=len(holdout) * len(axis))
    rulers = run.holdout_rulers()
    clock = time.perf_counter()
    report = validate_reduced(model, operators, holdout, workers=job.workers, dense_threads=job.dense_threads,
                              progress=job.progress.validation)
    seconds = time.perf_counter() - clock
    save_validation_receipt(npz, report, rulers=rulers, seconds=seconds)
    within = within_reduced_contract(report, frequency_db=rulers.frequency.value, band_db=rulers.band.value)

    def verdict(kind: Verdict, message: str, *, slot_text: str | None = None, retired: str | None = None) -> None:
        _result(job.work, SurrogateResult(
            verdict=kind, message=message, stop_reason=ledger.stop_reason, full_solves=ledger.full_solves,
            rank=ledger.rank, build_seconds=ledger.seconds["total"], validation_seconds=seconds,
            frequency_db_by_seat=dict(report.frequency_db_by_seat), band_db_by_seat=dict(report.band_db_by_seat),
            rulers={"frequency": rulers.frequency.value, "band": rulers.band.value}, within_contract=within,
            slot=slot_text, retired=retired, finished_at=time.time()))

    if not within:
        verdict("not_validated", f"留出驗證沒過（逐頻最大差 {max(report.frequency_db_by_seat.values()):.4g} dB，尺 "
                f"{rulers.frequency.value:g} dB；帶能量最大差 {max(report.band_db_by_seat.values()):.4g} dB，尺 "
                f"{rulers.band.value:g} dB）：這份替身不能用，沒有放進格子。{_rebuild_hint(ledger.stop_reason)}")
        return EXIT_NOT_USABLE
    job.progress.phase("checking")
    save_operators_stamp(npz, operators, rulers=rulers)
    points = run.SchemePoints(points_hex(operators.source_points), points_hex(operators.receiver_points))
    del model
    try:
        gated = run.gated_surrogate(npz, mesh_hash=operators.mesh_hash, physics_identity=operators.physics_identity,
                                    points=points)
        run.check_surrogate_matches_scheme(gated, job.scheme, points)
        run.check_surrogate_axis_covers_scoring(gated, job.scheme,
                                                run.search_objective_settings(job.scheme.purpose).stage_one)
    except ValueError as error:
        verdict("gate_refused", f"搜尋開跑那一道准用閘門不收（{type(error).__name__}：{error}），沒有放進格子")
        return EXIT_NOT_USABLE
    job.progress.phase("installing")
    _commit(STOPS, job.stage, job.slot, job.replace, lambda retired: verdict(
        "installed", "建好、留出驗證兩把尺都過、搜尋的准用閘門核過，已放進這份切法的格子",
        slot_text=str(job.slot), retired=retired))
    return EXIT_INSTALLED


def main(argv: Sequence[str] | None = None) -> int:
    """參數錯由 argparse 以離開碼 2 結束；放進格子 0；留出驗證或閘門不過 4；被停 3；其餘錯誤原文寫標準錯誤、回 1。"""
    args = _parser().parse_args(None if argv is None else list(argv))
    runtime.configure_calculation_jax()
    try:
        return build(args)
    except KeyboardInterrupt:
        sys.stderr.write("建替身被停止（不接續，要重來）\n")
        return EXIT_STOPPED
    except Exception as error:
        sys.stderr.write(f"建替身失敗：{type(error).__name__}：{error}\n")
        return EXIT_FAILED


if __name__ == "__main__":
    for number in SIGNALS:
        signal.signal(number, STOPS)
    raise SystemExit(main())
