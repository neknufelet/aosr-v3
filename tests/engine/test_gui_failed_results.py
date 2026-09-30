"""有結果產物的失敗、停止與計算中工作，不能冒充正常完成。"""
from __future__ import annotations

import csv
import io
import json
import sys
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.jobs import JobManager
from aosr.reporting.result import SchemeResult, save_result
from tests.engine.test_gui_compare_routes import pair
from tests.engine.test_gui_compare_view import moved_primary_result


def _runner(tmp_path: Path, outcomes: dict[str, tuple[SchemeResult, int, bool]]) -> tuple[str, ...]:
    """替身複製真的結果；需要等待時，以方案自己的旗標釋放。"""
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    release = tmp_path / "release"
    release.mkdir(exist_ok=True)
    codes = {}
    for name, (result, code, wait) in outcomes.items():
        save_result(result, sources / name)
        codes[name] = [code, wait]
    script = tmp_path / "runner"
    script.write_text(
        "import json,shutil,sys,time\nfrom pathlib import Path\n"
        f"sources = Path({str(sources)!r})\nrelease = Path({str(release)!r})\n"
        f"codes = {codes!r}\n"
        "name = json.loads(Path(sys.argv[1]).read_text())['scheme_id']\n"
        "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "shutil.copyfile(sources / name, out)\n"
        "code, wait = codes[name]\n"
        "while wait and not (release / name).is_file():\n    time.sleep(0.02)\n"
        "raise SystemExit(code)\n", encoding="utf-8")
    return sys.executable, str(script)


def _client(tmp_path: Path, runner: tuple[str, ...]) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path,
                                            runner=runner)), base_url="http://localhost")


def _start(client: TestClient, result: SchemeResult) -> str:
    name = result.scheme.scheme_id
    saved = client.put(f"/api/schemes/{name}", json=result.scheme.model_dump(mode="json"))
    assert saved.status_code == 200, saved.text
    response = client.post("/api/runs", json={"scheme_id": name})
    assert response.status_code == 200, response.text
    return str(response.json()["run_id"])


def _wait_state(client: TestClient, run_id: str, expected: str) -> dict[str, object]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state: dict[str, object] = client.get(f"/api/runs/{run_id}").json()
        if state["status"] == expected:
            return state
        time.sleep(0.02)
    raise AssertionError(f"計算沒有變成 {expected}：{state}")


def _wait_file(path: Path) -> None:
    deadline = time.monotonic() + 20
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise AssertionError("替身沒有寫出結果")
        time.sleep(0.02)


def _row(client: TestClient, run_id: str) -> dict[str, str]:
    return next(item for item in client.get("/api/results").json()["results"]
                if item["run_id"] == run_id)


def _path(tmp_path: Path, folder: str, run_id: str) -> Path:
    return (tmp_path / folder / run_id).with_suffix(".json")


def test_list_marks_failed_done_and_untracked(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a, b = pair
    runner = _runner(tmp_path, {a.scheme.scheme_id: (a, 3, False), b.scheme.scheme_id: (b, 0, False)})
    with _client(tmp_path, runner) as client:
        failed, done = _start(client, a), _start(client, b)
        _wait_state(client, failed, "failed")
        _wait_state(client, done, "done")
        untracked = "d" * 32
        save_result(a, _path(tmp_path, "results", untracked))
        assert _row(client, failed)["run_status"] == "failed"
        assert _row(client, failed)["status_text"] == "失敗：內容可能不完整"
        assert _row(client, done)["run_status"] == "done"
        assert _row(client, done)["status_text"] == "完成"
        assert _row(client, untracked)["run_status"] == "none"
        assert _row(client, untracked)["status_text"] == "完成（沒有計算紀錄）"


def test_cached_summary_refreshes_status_without_result_file_change(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a = pair[0]
    with _client(tmp_path, _runner(tmp_path, {a.scheme.scheme_id: (a, 3, True)})) as client:
        run_id = _start(client, a)
        path = _path(tmp_path, "results", run_id)
        try:
            _wait_file(path)
            before = path.stat()
            assert _row(client, run_id)["run_status"] == "running"
            (tmp_path / "release" / a.scheme.scheme_id).touch()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                row = _row(client, run_id)
                if row["run_status"] == "failed":
                    break
                time.sleep(0.02)
            assert row["run_status"] == "failed"
            assert "失敗" in row["status_text"]
            after = path.stat()
            assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)
        finally:
            client.post(f"/api/runs/{run_id}/stop", json={})


def test_stopped_artifact_has_notice_and_does_not_freeze_name(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a = pair[0]
    with _client(tmp_path, _runner(tmp_path, {a.scheme.scheme_id: (a, 3, True)})) as client:
        run_id = _start(client, a)
        try:
            _wait_file(_path(tmp_path, "results", run_id))
            changed = a.scheme.model_dump(mode="json")
            changed["scene"]["room_m"]["Lx"] += 0.1
            busy = client.put(f"/api/schemes/{a.scheme.scheme_id}", json=changed)
            assert busy.status_code == 409 and "正在計算" in busy.text
            stopped = client.post(f"/api/runs/{run_id}/stop", json={})
            assert stopped.status_code == 200, stopped.text
            assert stopped.json()["status"] == "stopped"
            assert _row(client, run_id)["run_status"] == "stopped"
            detail = client.get(f"/api/results/{run_id}").json()
            assert detail["run_status"] == "stopped" and "停止" in detail["run_notice"]
            assert client.put(f"/api/schemes/{a.scheme.scheme_id}", json=changed).status_code == 200
        finally:
            client.post(f"/api/runs/{run_id}/stop", json={})


@pytest.mark.parametrize("mode", ["failed", "done", "mixed", "none"])
def test_only_finished_artifacts_freeze_scheme_name(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], mode: str) -> None:
    a = pair[0]
    runner = _runner(tmp_path, {a.scheme.scheme_id: (a, 3 if mode in {"failed", "mixed"} else 0, False)})
    with _client(tmp_path, runner) as client:
        run_id = _start(client, a)
        _wait_state(client, run_id, "failed" if mode in {"failed", "mixed"} else "done")
        if mode == "mixed":
            _runner(tmp_path, {a.scheme.scheme_id: (a, 0, False)})
            _wait_state(client, _start(client, a), "done")
        if mode == "none":
            _path(tmp_path, "runs", run_id).unlink()
        changed = a.scheme.model_dump(mode="json")
        changed["scene"]["room_m"]["Lx"] += 0.1
        response = client.put(f"/api/schemes/{a.scheme.scheme_id}", json=changed)
        if mode == "failed":
            assert response.status_code == 200, response.text
        else:
            assert response.status_code == 409
            assert "已經有算好的結果" in response.text


def test_result_notice_only_for_unfinished_and_survives_rejection(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a, b = pair
    runner = _runner(tmp_path, {a.scheme.scheme_id: (a, 3, False), b.scheme.scheme_id: (b, 0, False)})
    with _client(tmp_path, runner) as client:
        failed, done = _start(client, a), _start(client, b)
        _wait_state(client, failed, "failed")
        _wait_state(client, done, "done")
        response = client.get(f"/api/results/{failed}")
        assert response.status_code == 200
        assert response.json()["run_status"] == "failed"
        notice = response.json()["run_notice"]
        assert "失敗" in notice and "離開碼 3" in notice
        normal = client.get(f"/api/results/{done}").json()
        assert "run_status" not in normal and "run_notice" not in normal
        _path(tmp_path, "results", failed).write_text("{}", encoding="utf-8")
        rejected = client.get(f"/api/results/{failed}")
        assert rejected.status_code == 409
        assert rejected.json()["run_notice"] == notice


def test_failed_comparison_and_exports_keep_values_without_winner(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a, b = pair
    runner = _runner(tmp_path, {a.scheme.scheme_id: (a, 0, False), b.scheme.scheme_id: (b, 0, False)})
    with _client(tmp_path, runner) as client:
        left, right = _start(client, a), _start(client, b)
        _wait_state(client, left, "done")
        _wait_state(client, right, "done")
        url = f"/api/compare/{left}/{right}"
        normal = client.get(url).json()
        assert normal["run_notices"] == []
        assert "比較好" in normal["table"]["verdict_text"]
        # 同一份產物改成真的非零結束工作：數字與曲線必須跟正常比較保持相同。
        _runner(tmp_path, {a.scheme.scheme_id: (a, 3, False)})
        failed = _start(client, a)
        _wait_state(client, failed, "failed")
        url = f"/api/compare/{failed}/{right}"
        response = client.get(url)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["table"]["better"] == ""
        assert "比較好：" not in data["table"]["verdict_text"]
        assert "A：失敗" in data["table"]["verdict_text"]
        assert all(row["better"] == row["better_text"] == "" for row in data["categories"])
        assert any(text.startswith("A：") and "離開碼 3" in text for text in data["run_notices"])
        assert data["overlay"] == normal["overlay"]
        assert [(row["a"], row["b"]) for row in data["categories"]] == [
            (row["a"], row["b"]) for row in normal["categories"]]
        assert (data["table"]["a_cell"], data["table"]["b_cell"]) == (
            normal["table"]["a_cell"], normal["table"]["b_cell"])
        summary = client.get(f"{url}/export/summary")
        assert summary.status_code == 200 and "比較好：" not in summary.text
        rows = list(csv.reader(io.StringIO(summary.content.decode("utf-8-sig"))))
        assert ["計算狀態", "失敗：內容可能不完整", "完成"] in rows
        curves = client.get(f"{url}/export/curves")
        assert curves.status_code == 200 and "A：" in curves.text and "失敗" in curves.text
        # 不能比及拒收，也要保留原來那一邊的失敗警語。
        problem = client.get(f"/api/compare/{failed}/{left}")
        assert problem.status_code == 409 and problem.json()["run_notices"] == data["run_notices"]
        _path(tmp_path, "results", failed).write_text("{}", encoding="utf-8")
        rejected = client.get(url)
        assert rejected.status_code == 409 and rejected.json()["run_notices"] == data["run_notices"]


@pytest.mark.parametrize("error", [FileNotFoundError, OSError, ValueError, KeyError, TypeError])
def test_unreadable_run_record_is_untracked(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch, error: type[Exception]) -> None:
    def unreadable(self: JobManager, run_id: str) -> dict[str, object]:
        raise error("計算紀錄讀不出")

    monkeypatch.setattr(JobManager, "get", unreadable)
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "d" * 32
        save_result(pair[0], _path(tmp_path, "results", run_id))
        assert _row(client, run_id)["run_status"] == "none"
        assert _row(client, run_id)["status_text"] == "完成（沒有計算紀錄）"
        detail = client.get(f"/api/results/{run_id}")
        assert detail.status_code == 200
        assert "run_status" not in detail.json() and "run_notice" not in detail.json()


def test_running_and_two_unfinished_sides_keep_diagnostic_values(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a, b = pair
    runner = _runner(tmp_path, {a.scheme.scheme_id: (a, 3, False), b.scheme.scheme_id: (b, 0, True)})
    with _client(tmp_path, runner) as client:
        failed, waiting = _start(client, a), _start(client, b)
        try:
            _wait_state(client, failed, "failed")
            _wait_file(_path(tmp_path, "results", waiting))
            detail = client.get(f"/api/results/{waiting}").json()
            assert detail["run_status"] == "running"
            assert detail["run_notice"] == "這一筆還在計算中，結果檔還可能再變；不是正常完成的結果"
            reversed_data = client.get(f"/api/compare/{waiting}/{failed}").json()
            assert "A：計算中；B：失敗" in reversed_data["table"]["verdict_text"]
            assert reversed_data["table"]["better"] == ""
            assert all(row["better"] == row["better_text"] == "" for row in reversed_data["categories"])
            assert any(text.startswith("A：") and "計算中" in text for text in reversed_data["run_notices"])
            assert any(text.startswith("B：") and "失敗" in text for text in reversed_data["run_notices"])
            client.post(f"/api/runs/{waiting}/stop", json={})
            stopped = client.get(f"/api/compare/{failed}/{waiting}").json()
            assert "A：失敗；B：已停止" in stopped["table"]["verdict_text"]
            assert any(text.startswith("B：") and "停止" in text for text in stopped["run_notices"])
            # 收不到離開碼時，失敗仍要標，但不能印空括號或捏造離開碼。
            state = client.get(f"/api/runs/{failed}").json()
            state["exit_code"] = None
            _path(tmp_path, "runs", failed).write_text(json.dumps(state), encoding="utf-8")
            notice = client.get(f"/api/results/{failed}").json()["run_notice"]
            assert "失敗，" in notice and "離開碼" not in notice
        finally:
            client.post(f"/api/runs/{waiting}/stop", json={})


def test_failed_split_table_keeps_existing_no_total_explanation(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    a, moved = pair[0], moved_primary_result(tmp_path_factory, worker_id)
    runner = _runner(tmp_path, {a.scheme.scheme_id: (a, 3, False),
                                moved.scheme.scheme_id: (moved, 0, False)})
    with _client(tmp_path, runner) as client:
        failed, done = _start(client, a), _start(client, moved)
        _wait_state(client, failed, "failed")
        _wait_state(client, done, "done")
        response = client.get(f"/api/compare/{failed}/{done}")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["table"]["a_cell"] == data["table"]["b_cell"] == "不列"
        assert "A：失敗" in data["table"]["verdict_text"]
        assert "尚未評估" in data["pending_text"] and "不算進總代價" not in data["pending_text"]
