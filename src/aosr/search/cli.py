"""搜尋命令列：開始、接續、停止；正常結果只在搜尋資料夾的狀態檔與離開碼。

出錯時把錯誤原文寫到標準錯誤（主對話判斷：出錯就報錯看原文，搜尋資料夾還沒建好時也看得到）。
達到停止條件、因預算停止、使用者停止回 0；失敗回 1；參數錯誤回 2；中斷回 3。
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import date
from importlib.metadata import version
from pathlib import Path
from types import FrameType
from typing import TypeAlias

import optuna

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.reporting.calculation_fingerprint import calculation_fingerprint
from aosr.reporting.evaluation import purpose_settings
from aosr.reporting.physics_identity import physics_identity
from aosr.reporting.scheme import load_scheme
from aosr.search.report import build_report, render_text
from aosr.search.report_comparison import default_precision_contracts_path
from aosr.search.refine_run import refine_search, refinement_status
from aosr.search.run import Compute, SearchStatus, _write_status, resume_search, start_search
from aosr.search.settings import SearchSettings
from aosr.search.select import select_refined
from aosr.search.store import SearchIdentity, SearchStore
from aosr.search.worker import SubprocessCompute
from aosr.search.feedback import feedback_search
from aosr.search.outer import auto_search
from aosr.search.modal_attach import DEFAULT_RUNNER, STOPPED_NOTE, attach_modal, record_attachment_error
from aosr.search.outer_status import OUTER_MESSAGES, snapshot_of

ComputeFactory: TypeAlias = Callable[[SearchStore, Path, str], Compute]


class _SearchBusy(Exception):
    """拿不到資料夾鎖（尚有父或子行程持鎖，或資料夾打不開、拿鎖本身出錯），拒絕入口且保留所有狀態。"""


def _stderr(text: str) -> None:
    try:
        sys.stderr.write(text)
    except OSError:
        pass


def _stop_notice(store: SearchStore) -> None:
    try:
        status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    except (OSError, ValueError):
        return
    if status.outer.conclusion is not None and status.outer.snapshot == snapshot_of(status):
        _stderr(f"這個搜尋已有外圈結論（{OUTER_MESSAGES[status.outer.conclusion]}）；停止記號照樣建立，"
                "下一次 auto 會照停止記號判成使用者停止。只是要停正在補的低頻診斷，"
                "請改對 auto 行程送 SIGTERM（終止訊號）\n")


@contextmanager
def _search_lock(store: SearchStore) -> Iterator[int]:
    """鎖搜尋資料夾本身；只關父邊描述子，不主動解鎖，繼承的子行程仍持有同一把鎖。

    拿鎖的任何錯都照「拿不到」收：沒拿到鎖就不准寫狀態（複查：權限不足被當成搜尋失敗寫進狀態，搜尋接不回來）。
    """
    try:
        descriptor = store.open_folder_lock()
    except OSError as error:
        raise _SearchBusy(f"拿不到搜尋資料夾的鎖，沒有動任何檔：{error}") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise _SearchBusy("這個搜尋資料夾還有計算在跑（可能是上一次被強制結束後留下的子行程），等它結束再試") from error
        except OSError as error:
            raise _SearchBusy(f"拿不到搜尋資料夾的鎖，沒有動任何檔：{error}") from error
        yield descriptor
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.search.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="開始搜尋")
    start.add_argument("--project", type=Path, required=True)
    start.add_argument("--settings", type=Path, required=True)
    start.add_argument("--root", type=Path, required=True)
    resume = commands.add_parser("resume", help="接續進行中的搜尋")
    resume.add_argument("search", type=Path)
    refine = commands.add_parser("refine", help="接續細算與重排已停的搜尋")
    refine.add_argument("search", type=Path)
    auto = commands.add_parser("auto", help="自動接續搜尋、細算與回饋，寫外圈結論")
    auto.add_argument("search", type=Path)
    auto.add_argument("--modal-cache-dir", type=Path, required=True, help="網頁與搜尋共用的低頻模態快取資料夾")
    for command in (start, resume, refine, auto):
        command.add_argument("--engine-commit", required=True)
        command.add_argument("--capabilities", type=Path, default=config_path("capabilities.toml"))
    stop = commands.add_parser("stop", help="建立停止記號")
    stop.add_argument("search", type=Path)
    stop.add_argument("--refine", action="store_true", help="只停止細算")
    report = commands.add_parser("report", help="只讀搜尋報告")
    report.add_argument("--search", type=Path, required=True)
    report.add_argument("--contracts", type=Path, default=default_precision_contracts_path(), help="精度契約登記簿")
    select = commands.add_parser("select", help="把選中的細算結果放進結果清單")
    select.add_argument("search", type=Path)
    choice = select.add_mutually_exclusive_group(required=True)
    choice.add_argument("--trial", type=int)
    choice.add_argument("--baseline", action="store_true")
    select.add_argument("--data-dir", type=Path, required=True)
    feedback = commands.add_parser("feedback", help="只記事件與輪次，排入細算第一名附近的搜尋點")
    feedback.add_argument("search", type=Path)
    return parser


def _identity(purpose: str, capabilities: Path) -> SearchIdentity:
    """每批重新讀能力表、指向性與用途設定，身分不同就中斷。"""
    fingerprint = calculation_fingerprint(capabilities_path=capabilities)
    table = load_capabilities(capabilities)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    physics = physics_identity(capabilities=table, directivity=directivity)
    settings = purpose_settings(config_path("quality_targets.toml"), purpose)
    return SearchIdentity(physics, fingerprint, settings)


def _create(args: argparse.Namespace) -> SearchStore:
    project = load_scheme(args.project)
    settings = SearchSettings.model_validate_json(args.settings.read_bytes())
    identity = _identity(project.purpose, args.capabilities)
    versions = {"python": sys.version, "optuna": version("optuna"), "numpy": version("numpy")}
    return SearchStore.create(args.root, project=project, settings=settings,
                              identity=identity, versions=versions)


def _compute(store: SearchStore, capabilities: Path, commit: str, *, lock_fd: int | None = None) -> Compute:
    return SubprocessCompute(capabilities_path=capabilities, engine_commit=commit,
                             search_id=store.search_id, fem_root=store.fem_path, lock_fd=lock_fd)


def _failed(store: SearchStore | None, error: Exception) -> int:
    """錯誤原文寫到標準錯誤；運行前的錯誤也留狀態；拒接已停搜尋時保留原來停止原因。"""
    _stderr(f"搜尋失敗：{error}\n")
    if store is None:
        return 1
    try:
        status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    except (OSError, ValueError):
        return 1
    if status.state == "running":
        _write_status(store, status.model_copy(update={"state": "failed", "message": f"搜尋失敗：{error}"}))
    return 1


def main(argv: list[str] | None = None, *, compute_factory: ComputeFactory | None = None,
         modal_runner: tuple[str, ...] = DEFAULT_RUNNER) -> int:
    """日期只在命令列取今天；注入工廠只替換計算，搜尋與保存仍走產品入口。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = _parser().parse_args(argv)
    if args.command == "select":
        return _select_command(args)
    registry_path = config_path("quality_targets.toml")
    store: SearchStore | None = None
    with ExitStack() as held:
        # 這一段只接「打開資料夾、報告、停止記號、拿鎖」的錯；拿到鎖之後各命令照改動前自己的錯誤出口，
        # 錯誤出口本身再出錯時照舊往外丟，不在這裡再接一次（複查：重複寫失敗、蓋掉原本的錯誤追蹤）。
        try:
            store = _create(args) if args.command == "start" else SearchStore.open(args.search)
            if args.command == "report":
                report = build_report(store, quality_targets_path=registry_path, run_date=date.today(),
                                      precision_contracts_path=args.contracts)
                sys.stdout.write(render_text(report))
                return 0
            if args.command == "stop":
                (store.refine_stop_path if args.refine else store.stop_path).touch()
                _stop_notice(store)
                return 0
            lock_fd = held.enter_context(_search_lock(store))
        except _SearchBusy as error:
            _stderr(f"{error}\n")
            return 1
        except Exception as error:
            if args.command in ("report", "auto", "refine", "feedback"):
                prefix = {"report": "報告", "auto": "自動外圈", "refine": "細算", "feedback": "回饋"}[args.command]
                _stderr(f"{prefix}失敗：{error}\n")
                return 1
            return _failed(store, error)
        factory = compute_factory or (lambda opened, capabilities, commit:
                                      _compute(opened, capabilities, commit, lock_fd=lock_fd))
        return _mutating_command(args, store, factory, registry_path, lock_fd=lock_fd, modal_runner=modal_runner)


def _mutating_command(args: argparse.Namespace, store: SearchStore,
                      factory: ComputeFactory, registry_path: Path, *, lock_fd: int,
                      modal_runner: tuple[str, ...]) -> int:
    """在同一次持鎖範圍內完成計算及錯誤狀態寫入，外圈不重新拿鎖。"""
    if args.command == "auto":
        return _auto_command(args, factory, registry_path, lock_fd=lock_fd, modal_runner=modal_runner)
    if args.command == "refine":
        return _refine_command(args, factory, registry_path)
    if args.command == "feedback":
        return _feedback_command(args.search)
    try:
        if args.command == "start":
            _write_status(store, SearchStatus(timed_from_start=True))
        compute = factory(store, args.capabilities, args.engine_commit)
        purpose = store.project.purpose
        entry = start_search if args.command == "start" else resume_search
        status = entry(store, compute=compute, probe=lambda: _identity(purpose, args.capabilities),
                       registry_path=registry_path, run_date=date.today(),
                       engine_version=store.identity.program_fingerprint)
        return 3 if status.state == "interrupted" else 1 if status.state == "failed" else 0
    except Exception as error:
        return _failed(store, error)


def _auto_command(args: argparse.Namespace, factory: ComputeFactory, registry_path: Path, *,
                  lock_fd: int, modal_runner: tuple[str, ...]) -> int:
    """外圈拒絕只報錯；完整性錯誤不改搜尋或細算、不冒充一般結論。"""
    try:
        store = SearchStore.open(args.search)
        status = auto_search(store, compute=factory(store, args.capabilities, args.engine_commit),
                             probe=lambda: _identity(store.project.purpose, args.capabilities),
                             registry_path=registry_path, run_date=date.today(),
                             engine_version=store.identity.program_fingerprint)
        conclusion = status.outer.conclusion
        try:
            attach_modal(store, status=status, cache_dir=args.modal_cache_dir, lock_fd=lock_fd, runner=modal_runner)
        except (Exception, KeyboardInterrupt) as error:
            stopped = isinstance(error, KeyboardInterrupt)
            record_attachment_error(store, status, args.modal_cache_dir, error, stopped=stopped)
            _stderr(STOPPED_NOTE + "\n" if stopped else f"低頻診斷失敗，搜尋結果不受影響：{error}\n")
        if conclusion in ("search_interrupted", "refine_interrupted"):
            return 3
        return 1 if conclusion in ("search_failed", "refine_failed") else 0
    except Exception as error:
        _stderr(f"自動外圈失敗：{error}\n")
        return 1


def _select_command(args: argparse.Namespace) -> int:
    """選入拒絕只回原文，不走會改動搜尋或細算狀態的出口。"""
    try:
        store = SearchStore.open(args.search)
        outcome = select_refined(store, None if args.baseline else args.trial, args.data_dir)
        action = "已放進結果清單" if outcome.is_new else "已在結果清單"
        sys.stdout.write(f"{action}：代號 {outcome.run_id}\n")
        return 0
    except Exception as error:
        _stderr(f"{error}\n")
        return 1


def _feedback_command(path: Path) -> int:
    """獨立拒絕出口：前提失敗只寫標準錯誤，不能改搜尋或細算狀態。"""
    try:
        feedback_search(SearchStore.open(path))
        return 0
    except Exception as error:
        _stderr(f"回饋失敗：{error}\n")
        return 1


def _refine_command(args: argparse.Namespace, factory: ComputeFactory, registry_path: Path) -> int:
    """獨立錯誤出口；細算拒跑不能被搜尋的失敗出口改動搜尋狀態。"""
    store: SearchStore | None = None
    accepted: SearchStatus | None = None
    try:
        store = SearchStore.open(args.search)
        accepted = refinement_status(store)
        compute = factory(store, args.capabilities, args.engine_commit)
        status = refine_search(store, compute=compute, probe=lambda: _identity(store.project.purpose, args.capabilities),
                               registry_path=registry_path, run_date=date.today(),
                               engine_version=store.identity.program_fingerprint)
        return 3 if status.refine.state == "interrupted" else 1 if status.refine.state == "failed" else 0
    except Exception as error:
        _stderr(f"細算失敗：{error}\n")
        if store is not None and accepted is not None:
            current = SearchStatus.model_validate_json(store.status_path.read_bytes())
            if current.refine.state in ("not_started", "running"):
                failed = current.refine.model_copy(update={"state": "failed", "message": f"細算失敗：{error}"})
                _write_status(store, current.model_copy(update={"refine": failed}))
        return 1


def _interrupt_on_termination() -> None:
    """命令列行程將終止訊號轉為中斷，讓計算的 finally 清掉整群。"""
    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    for termination in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(termination, interrupt)


if __name__ == "__main__":
    _interrupt_on_termination()
    raise SystemExit(main())
