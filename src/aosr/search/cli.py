"""搜尋命令列：開始、接續、停止；正常結果只在搜尋資料夾的狀態檔與離開碼。

出錯時把錯誤原文寫到標準錯誤（主對話判斷：出錯就報錯看原文，搜尋資料夾還沒建好時也看得到）。
收斂、因預算停止、使用者停止回 0；失敗回 1；參數錯誤回 2；中斷回 3。
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Callable
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
from aosr.search.run import Compute, SearchStatus, _write_status, resume_search, start_search
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from aosr.search.worker import SubprocessCompute

ComputeFactory: TypeAlias = Callable[[SearchStore, Path, str], Compute]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.search.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="開始搜尋")
    start.add_argument("--project", type=Path, required=True)
    start.add_argument("--settings", type=Path, required=True)
    start.add_argument("--root", type=Path, required=True)
    resume = commands.add_parser("resume", help="接續進行中的搜尋")
    resume.add_argument("search", type=Path)
    for command in (start, resume):
        command.add_argument("--engine-commit", required=True)
        command.add_argument("--capabilities", type=Path, default=config_path("capabilities.toml"))
    stop = commands.add_parser("stop", help="建立停止記號")
    stop.add_argument("search", type=Path)
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


def _compute(store: SearchStore, capabilities: Path, commit: str) -> Compute:
    return SubprocessCompute(capabilities_path=capabilities, engine_commit=commit, search_id=store.search_id)


def _failed(store: SearchStore | None, error: Exception) -> int:
    """錯誤原文寫到標準錯誤；運行前的錯誤也留狀態；拒接已停搜尋時保留原來停止原因。"""
    sys.stderr.write(f"搜尋失敗：{error}\n")
    if store is None:
        return 1
    try:
        status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    except (OSError, ValueError):
        return 1
    if status.state == "running":
        _write_status(store, status.model_copy(update={"state": "failed", "message": f"搜尋失敗：{error}"}))
    return 1


def main(argv: list[str] | None = None, *, compute_factory: ComputeFactory | None = None) -> int:
    """日期只在命令列取今天；注入工廠只替換計算，搜尋與保存仍走產品入口。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = _parser().parse_args(argv)
    store: SearchStore | None = None
    try:
        store = _create(args) if args.command == "start" else SearchStore.open(args.search)
        if args.command == "start":
            _write_status(store, SearchStatus())
        if args.command == "stop":
            store.stop_path.touch()
            return 0
        compute = (compute_factory or _compute)(store, args.capabilities, args.engine_commit)
        purpose = store.project.purpose
        entry = start_search if args.command == "start" else resume_search
        status = entry(store, compute=compute, probe=lambda: _identity(purpose, args.capabilities),
                       registry_path=config_path("quality_targets.toml"), run_date=date.today(),
                       engine_version=store.identity.program_fingerprint)
        return 3 if status.state == "interrupted" else 1 if status.state == "failed" else 0
    except Exception as error:
        return _failed(store, error)


def _interrupt_on_termination() -> None:
    """命令列行程將終止訊號轉為中斷，讓計算的 finally 清掉整群。"""
    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    for termination in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(termination, interrupt)


if __name__ == "__main__":
    _interrupt_on_termination()
    raise SystemExit(main())
