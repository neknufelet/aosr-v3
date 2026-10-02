"""乾淨命令列行程驗靜默與終止訊號；計算只跑小替身腳本。"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aosr.runtime import child_process_env
from aosr.search.run import SearchStatus
from aosr.search.settings import SearchSettings
from tests.engine._search_worker_cases import CLI_SCRIPT, ENTRY_SCRIPT, prepared_result
from tests.engine.test_search_cli import opened, start_args
from tests.engine.test_search_worker import assert_dead, events, setup_worker


def launch(tmp_path: Path, options: dict[str, object]) -> subprocess.Popen[bytes]:
    setup_worker(tmp_path, options, ())
    launcher = tmp_path / "launcher"
    launcher.write_text(CLI_SCRIPT, encoding="utf-8")
    args = start_args(tmp_path)
    project_path = Path(args[args.index("--project") + 1])
    project_path.write_text(prepared_result(tmp_path).scheme.model_dump_json())
    settings_path = Path(args[args.index("--settings") + 1])
    settings = SearchSettings.model_validate_json(settings_path.read_bytes())
    settings_path.write_text(settings.model_copy(update={"batch_size": 2, "max_workers": 2}).model_dump_json())
    environment = child_process_env(threads=1)
    environment["PYTHONPATH"] = os.pathsep.join((environment["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    return subprocess.Popen([sys.executable, str(launcher), str(tmp_path), *args],
                            cwd=tmp_path, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def cleanup(process: subprocess.Popen[bytes], tmp_path: Path) -> None:
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=20)
    for record in events(tmp_path):
        for key in ("pid", "child"):
            if record[key] is not None:
                try:
                    os.kill(int(str(record[key])), signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize("termination", (signal.SIGTERM, signal.SIGHUP))
def test_termination_stops_workers_and_descendants(tmp_path: Path, termination: signal.Signals) -> None:
    process = launch(tmp_path, {str(number): {"sleep": 120, "child": True} for number in (0, 1)})
    try:
        deadline = time.monotonic() + 20
        expected = {"event-0", "event-1"}
        while not expected.issubset({path.name for path in tmp_path.glob("event-*")}):
            if process.poll() is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert expected.issubset({path.name for path in tmp_path.glob("event-*")})
        process.send_signal(termination)
        process.communicate(timeout=20)
        assert process.returncode != 0
        records = events(tmp_path)
        assert records
        for record in records:
            assert_dead(int(str(record["pid"])))
            if record["child"] is not None:
                assert_dead(int(str(record["child"])))
        assert SearchStatus.model_validate_json(opened(tmp_path).status_path.read_bytes()).state == "running"
    finally:
        cleanup(process, tmp_path)


def test_clean_process_success_has_empty_output(tmp_path: Path) -> None:
    process = launch(tmp_path, {})
    try:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr.decode()
        assert stdout == b"" and stderr == b""
        assert SearchStatus.model_validate_json(opened(tmp_path).status_path.read_bytes()).state == "budget_exhausted"
    finally:
        cleanup(process, tmp_path)


def test_main_does_not_change_signal_handlers(tmp_path: Path) -> None:
    from aosr.search.cli import main
    from tests.engine.test_search_cli import fake_factory

    before = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    code = main(start_args(tmp_path), compute_factory=fake_factory)
    assert code == 0
    assert {number: signal.getsignal(number) for number in before} == before


def test_module_entry_installs_both_termination_handlers(tmp_path: Path) -> None:
    launcher = tmp_path / "entry-launcher"
    launcher.write_text(ENTRY_SCRIPT, encoding="utf-8")
    completed = subprocess.run([sys.executable, str(launcher), str(tmp_path)], cwd=tmp_path,
                               env=child_process_env(threads=1), capture_output=True, timeout=20)
    observed = tmp_path / "entry-signals"
    assert observed.is_file(), completed.stderr.decode()
    assert set(observed.read_text().splitlines()) == {"SIGTERM", "SIGHUP"}
