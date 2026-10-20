"""產品預設的五條開行程路徑都交給會先設定精度的正式命令列入口。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest

from aosr import runtime
from aosr.config.paths import config_path
from aosr.reporting import scheme_cli
from aosr.gui import jobs
from aosr.gui.app import GuiHandlers, GuiSettings
from aosr.search import modal_attach, worker
from aosr.search.modal_record import ModalRole, RoleInput
from aosr.search.run import CandidateJob
from aosr.search.store import SearchStore
from tests.engine._modal_cases import scheme


class _PopenReached(Exception):
    """拿到真 argv 後立即停止；不開行程、不求解、不碰伺服器。"""


def _capture_popen(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> list[tuple[str, ...]]:
    captured: list[tuple[str, ...]] = []
    def capture(argv: list[str] | tuple[str, ...], **kwargs: object) -> NoReturn:
        captured.append(tuple(argv))
        raise _PopenReached
    monkeypatch.setattr(module.subprocess, 'Popen', capture)
    return captured


def _assert_cli(argv: tuple[str, ...], command: str, scheme_path: Path) -> None:
    assert argv[:4] == (sys.executable, '-m', 'aosr.reporting.scheme_cli', command)
    assert argv[4] == str(scheme_path)
    assert '--out' in argv


@pytest.mark.parametrize('command', ['run', 'modal'])
def test_gui_default_commands_reach_formal_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
) -> None:
    # 從網頁真正建物件的位置取預設命令；data_dir 明確只給考卷暫存目錄。
    handlers = GuiHandlers(GuiSettings(engine_commit='a' * 40, data_dir=tmp_path))
    captured = _capture_popen(monkeypatch, jobs)
    with pytest.raises(_PopenReached):
        if command == 'run':
            handlers.jobs.start(tmp_path / 'scheme.json')
        else:
            handlers.modal_jobs._start(scheme(), cache_only=False, result_id='test')
    argv = captured[0]
    path = tmp_path / 'scheme.json' if command == 'run' else next(
        folder / 'scheme.json' for folder in (tmp_path / 'modal-jobs' / 'runs').iterdir() if folder.is_dir())
    _assert_cli(argv, command, path)
    assert argv[argv.index('--capabilities') + 1] == str(config_path('capabilities.toml'))
    if command == 'modal':
        assert argv[argv.index('--cache-dir') + 1] == str(handlers.modal_jobs.cache_dir)


@pytest.mark.parametrize('command', ['run', 'fem-slice'])
def test_search_default_commands_reach_formal_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
) -> None:
    compute = worker.SubprocessCompute(capabilities_path=config_path('capabilities.toml'),
        engine_commit='test', search_id='test', fem_root=tmp_path / 'fem')
    job = CandidateJob(1, scheme(), tmp_path / 'result.json')
    captured = _capture_popen(monkeypatch, worker)
    with pytest.raises(_PopenReached):
        if command == 'run':
            compute._start(job)
        else:
            compute._slices((job,), max_workers=1)
    argv = captured[0]
    _assert_cli(argv, command, SearchStore.scheme_path_for(job.result_path.resolve()))
    assert argv[argv.index('--capabilities') + 1] == str(config_path('capabilities.toml').resolve())
    if command == 'fem-slice':
        assert argv[argv.index('--slice') + 1] == '0'
        assert argv[argv.index('--slices') + 1] == '1'
    else:
        assert argv[argv.index('--search-id') + 1] == 'test'


def test_stage_one_command_configures_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """第一階段尚未接搜尋，搜尋的子行程表不列它；直接驗正式入口先設定計算精度才分派。"""
    import argparse
    events: list[str] = []
    configure = runtime.configure_calculation_jax
    def configured() -> None:
        configure()
        events.append('configured')
    def reached(args: argparse.Namespace) -> int:
        assert events == ['configured']
        assert args.command == 'stage-one'
        assert args.scheme == tmp_path / 'scheme.json'
        return 0
    monkeypatch.setattr(runtime, 'configure_calculation_jax', configured)
    monkeypatch.setattr(scheme_cli, '_stage_one', reached)
    code = scheme_cli.main(['stage-one', str(tmp_path / 'scheme.json'), '--out', str(tmp_path / 'result.json'),
        '--capabilities', str(config_path('capabilities.toml')), '--engine-commit', 'test'])
    assert code == 0


def test_modal_attach_default_command_reaches_formal_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheme_path = tmp_path / 'scheme.json'
    item = RoleInput(ModalRole(role='baseline'), None, scheme_path, tmp_path / 'unused.json')
    captured = _capture_popen(monkeypatch, modal_attach)
    output = tmp_path / 'modal.json'
    with pytest.raises(_PopenReached):
        modal_attach._compute(item, output, cache_dir=tmp_path, lock_fd=None,
                              runner=modal_attach.DEFAULT_RUNNER)
    argv = captured[0]
    _assert_cli(argv, 'modal', scheme_path.resolve())
    assert argv[argv.index('--out') + 1] == str(output.resolve())
    assert argv[argv.index('--cache-dir') + 1] == str(tmp_path)
