"""網頁只准一份：真資料夾鎖與子行程，網路服務只用替身。"""
from __future__ import annotations

import errno
import fcntl
import os
import select
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn

from aosr.gui import __main__ as cli
from aosr.gui import app as gui_app
from aosr.gui.search_view import folder_process
from aosr.search.cli import _search_lock
from aosr.search.select import select_refined
from aosr.search.store import refine_result_name
from tests.engine._search_select_cases import refined_store


LOCK_OWNER = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
sys.stdout.write('ready\\n')
sys.stdout.flush()
sys.stdin.read()
# 刻意不關 fd；持鎖行程結束，由核心釋放。
"""


@contextmanager
def _locked_by_child(folder: Path) -> Iterator[subprocess.Popen[str]]:
    with subprocess.Popen([sys.executable, "-c", LOCK_OWNER, str(folder)], cwd=folder,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as owner:
        try:
            assert owner.stdout is not None
            readable, _, _ = select.select([owner.stdout], [], [], 5)
            assert readable, "考卷開的子行程未回報拿到鎖"
            assert owner.stdout.readline() == "ready\n"
            yield owner
        finally:
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=5)


def _assert_folder_held(folder: Path) -> None:
    # 對手拿共享鎖：只有排他鎖擋得住它，網頁的鎖被改成共享時這裡會紅。
    contender = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_SH | fcntl.LOCK_NB)
    finally:
        os.close(contender)


@pytest.fixture
def cli_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 這些考卷只驗啟動鎖；程式與物理身分的真算法由既有考卷守。
    monkeypatch.setattr(cli, "calculation_fingerprint", lambda **_: "calc-v1:" + "a" * 64)
    monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity",
                        lambda **_: "phys-v1:" + "b" * 64)
    monkeypatch.setattr(sys, "argv", ["gui", "--engine-commit", "a" * 40,
                                       "--data-dir", str(tmp_path)])


@pytest.mark.parametrize("alias", [False, True], ids=["real-path", "symlink"])
def test_cli_refuses_busy_data_folder_before_app_or_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    cli_arguments: None, alias: bool,
) -> None:
    called: list[str] = []

    def fake_create(settings: gui_app.GuiSettings) -> object:
        called.append("create_app")
        return object()

    def fake_run(*args: object, **kwargs: object) -> None:
        called.append("uvicorn.run")

    monkeypatch.setattr(gui_app, "create_app", fake_create)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    if alias:
        link = tmp_path / "alias"
        assert not link.exists()
        link.symlink_to(tmp_path, target_is_directory=True)
        monkeypatch.setattr(sys, "argv", ["gui", "--engine-commit", "a" * 40,
                                           "--data-dir", str(link), "--port", "8001"])
    with _locked_by_child(tmp_path):
        with pytest.raises(SystemExit) as refused:
            cli.main()
    assert refused.value.code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert str(tmp_path.resolve()) in err
    assert "已經有一份網頁服務在用" in err and "先停掉" in err
    assert err.endswith("\n") and "\n" not in err.rstrip("\n")
    assert called == []


def _track_lock(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    real_hold = cli.hold_data_folder
    descriptors: list[int] = []

    def tracked_hold(path: Path) -> int:
        descriptor = real_hold(path)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(cli, "hold_data_folder", tracked_hold)
    return descriptors


def _assert_closed(descriptors: list[int], folder: Path) -> None:
    assert descriptors, "入口沒有拿資料夾鎖"
    for descriptor in descriptors:
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF
    contender = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(contender)


@pytest.mark.parametrize("end", ["exit", "kill"])
def test_cli_starts_after_owner_ends_and_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_arguments: None, end: str,
) -> None:
    descriptors = _track_lock(monkeypatch)
    called: list[str] = []
    app = object()

    def fake_create(settings: gui_app.GuiSettings) -> object:
        _assert_folder_held(settings.data_dir)
        called.append("create_app")
        return app

    def fake_run(application: object, **kwargs: object) -> None:
        assert application is app
        _assert_folder_held(tmp_path)
        # 收尾要有上限，不然一條卡住的連線會讓舊服務一直拿著鎖。
        assert kwargs["timeout_graceful_shutdown"] == cli.SHUTDOWN_GRACE_SECONDS
        called.append("uvicorn.run")

    monkeypatch.setattr(gui_app, "create_app", fake_create)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    with _locked_by_child(tmp_path) as owner:
        if end == "kill":
            owner.kill()
        else:
            assert owner.stdin is not None
            owner.stdin.close()
        owner.wait(timeout=5)
        cli.main()
    assert called == ["create_app", "uvicorn.run"]
    _assert_closed(descriptors, tmp_path)


@pytest.mark.parametrize("failure", ["create_app", "uvicorn.run"])
def test_cli_closes_descriptor_when_startup_or_service_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_arguments: None, failure: str,
) -> None:
    descriptors = _track_lock(monkeypatch)

    def fake_create(settings: gui_app.GuiSettings) -> object:
        _assert_folder_held(settings.data_dir)
        if failure == "create_app":
            raise RuntimeError(failure)
        return object()

    def fake_run(application: object, **kwargs: object) -> None:
        _assert_folder_held(tmp_path)
        raise RuntimeError("uvicorn.run")

    monkeypatch.setattr(gui_app, "create_app", fake_create)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    with pytest.raises(RuntimeError, match=failure):
        cli.main()
    _assert_closed(descriptors, tmp_path)


def test_hold_data_folder_creates_directory_without_files_or_child_inheritance(tmp_path: Path) -> None:
    folder = tmp_path / "new" / "data"
    assert not folder.exists()
    descriptor = cli.hold_data_folder(folder)
    try:
        assert folder.is_dir() and list(folder.iterdir()) == []
        assert os.fstat(descriptor).st_ino == folder.stat().st_ino
        assert not os.get_inheritable(descriptor)
        _assert_folder_held(folder)
        probe = """
import errno, os, sys
try:
    os.fstat(int(sys.argv[1]))
except OSError as error:
    assert error.errno == errno.EBADF
else:
    raise AssertionError('網頁的鎖被子行程繼承')
"""
        # 故意不由 Popen 關掉多餘描述子，讓考卷真的能抓到「可繼承」的植錯。
        child = subprocess.run([sys.executable, "-c", probe, str(descriptor)], cwd=folder,
                               close_fds=False, capture_output=True, text=True, timeout=5)
        assert child.returncode == 0, child.stderr
    finally:
        os.close(descriptor)


def test_gui_lock_allows_search_lock_view_and_result_selection(tmp_path: Path) -> None:
    assert not (tmp_path / "searches").exists()
    store = refined_store(tmp_path)
    assert store.path.parent == tmp_path / "searches"
    descriptor = cli.hold_data_folder(tmp_path)
    try:
        _assert_folder_held(tmp_path)
        assert folder_process(tmp_path).held is True
        process = folder_process(store.path)
        assert process.held is False and process.text == "沒有計算行程拿著這個資料夾"
        with _search_lock(store) as search_descriptor:
            assert os.fstat(search_descriptor).st_ino == store.path.stat().st_ino
            assert folder_process(store.path).held is True
        assert folder_process(store.path).held is False
        selected = select_refined(store, None, tmp_path)
        assert selected.is_new
        assert selected.result_path.parent == tmp_path / "results"
        assert selected.result_path.read_bytes() == (store.path / refine_result_name(None)).read_bytes()
    finally:
        os.close(descriptor)


def test_second_hold_on_the_same_folder_is_refused_even_in_one_process(tmp_path: Path) -> None:
    """排他：同一個資料夾拿第二次一定拿不到（改成共享鎖時兩份網頁都起得來，這題會紅）。"""
    first = cli.hold_data_folder(tmp_path)
    try:
        with pytest.raises(BlockingIOError):
            cli.hold_data_folder(tmp_path)
    finally:
        os.close(first)
