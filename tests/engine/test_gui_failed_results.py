"""有結果產物的失敗、停止與計算中工作，不能冒充正常完成。"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.jobs import JobManager
from aosr.gui.labels import RESULT_RUN_LABELS, RUN_EXIT_TEXT
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
        "import json,os,shutil,sys,time\nfrom pathlib import Path\n"
        f"sources = Path({str(sources)!r})\nrelease = Path({str(release)!r})\n"
        f"codes = {codes!r}\n"
        "name = json.loads(Path(sys.argv[1]).read_text())['scheme_id']\n"
        "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "part = out.with_name(out.name + '.part')\n"
        "shutil.copyfile(sources / name, part)\nos.replace(part, out)\n"
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
            assert _row(client, run_id)["status_text"] == "計算中：結果檔還可能再變"
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
            assert _row(client, run_id)["status_text"] == "已停止：內容可能不完整"
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


def _assert_finished_control(data: dict[str, object]) -> None:
    # 主對話於改動前主線 8f99845 用同一組 pair 實跑取得；答案不從 build_compare_view 重算。
    table = data["table"]
    assert isinstance(table, dict)
    assert table["verdict_text"] == "A 比較好：總代價比 B 低 0.519"
    assert table["better"] == "a"
    categories = data["categories"]
    assert isinstance(categories, list)
    assert [(row["category"], row["better"], row["better_text"]) for row in categories] == [
        ("timbre_balance", "a", "A 較好"),
        ("listening_area_stability", "b", "B 較好"),
        ("low_frequency_decay", "", ""),
        ("reflections_and_echo", "a", "A 較好"),
        ("reverberation", "same", "相同"),
        ("channel_matching", "b", "B 較好"),
        ("spatial_impression", "", "")]


def _assert_failed_exports(client: TestClient, url: str) -> None:
    notice = "A：" + RESULT_RUN_LABELS["failed"][2].format(exit_text=RUN_EXIT_TEXT.format(code=3))
    summary = client.get(f"{url}/export/summary")
    assert summary.status_code == 200 and "比較好：" not in summary.text
    rows = list(csv.reader(io.StringIO(summary.content.decode("utf-8-sig"))))
    assert ["計算狀態", "失敗：內容可能不完整", "完成"] in rows
    assert [notice] in rows
    curves = client.get(f"{url}/export/curves")
    assert curves.status_code == 200
    curve_rows = list(csv.reader(io.StringIO(curves.content.decode("utf-8-sig"))))
    assert ["計算狀態", "A：失敗：內容可能不完整", "B：完成"] in curve_rows
    assert [notice] in curve_rows


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
        _assert_finished_control(normal)
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
        assert data["table"]["verdict_text"] == "有一份計算沒有正常完成（A：失敗），不下哪一份比較好的結論"
        assert all(row["better"] == row["better_text"] == "" for row in data["categories"])
        assert any(text.startswith("A：") and "離開碼 3" in text for text in data["run_notices"])
        assert data["overlay"] == normal["overlay"]
        assert [(row["a"], row["b"]) for row in data["categories"]] == [
            (row["a"], row["b"]) for row in normal["categories"]]
        assert (data["table"]["a_cell"], data["table"]["b_cell"]) == (
            normal["table"]["a_cell"], normal["table"]["b_cell"])
        _assert_failed_exports(client, url)
        # 不能比及拒收，也要保留原來那一邊的失敗警語。
        problem = client.get(f"/api/compare/{failed}/{left}")
        assert problem.status_code == 409 and problem.json()["run_notices"] == data["run_notices"]
        _path(tmp_path, "results", failed).write_text("{}", encoding="utf-8")
        rejected = client.get(url)
        assert rejected.status_code == 409 and rejected.json()["run_notices"] == data["run_notices"]


@pytest.mark.parametrize("record", [None, "不是 JSON", "[]", '{"status": "unknown"}'])
def test_unreadable_run_record_is_untracked(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        record: str | None) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "d" * 32
        save_result(pair[0], _path(tmp_path, "results", run_id))
        if record is not None:
            _path(tmp_path, "runs", run_id).write_text(record, encoding="utf-8")
        assert _row(client, run_id)["run_status"] == "none"
        assert _row(client, run_id)["status_text"] == "完成（沒有計算紀錄）"
        detail = client.get(f"/api/results/{run_id}")
        assert detail.status_code == 200
        assert "run_status" not in detail.json() and "run_notice" not in detail.json()


@pytest.mark.parametrize("status, code", [("failed", 3), ("done", 0)])
def test_missing_stderr_preserves_recorded_status(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], status: str, code: int) -> None:
    a, b = pair
    with _client(tmp_path, (sys.executable,)) as client:
        left, right = "a" * 32, "b" * 32
        for run_id, result, recorded, exit_code in (
                (left, a, status, code), (right, b, "done", 0)):
            save_result(result, _path(tmp_path, "results", run_id))
            _path(tmp_path, "runs", run_id).write_text(json.dumps({
                "status": recorded, "exit_code": exit_code, "started_at": 0,
                "scheme_id": result.scheme.scheme_id,
                "stderr_path": str(tmp_path / "missing.stderr")}), encoding="utf-8")
        assert _row(client, left)["run_status"] == status
        detail = client.get(f"/api/results/{left}")
        assert detail.status_code == 200, detail.text
        comparison = client.get(f"/api/compare/{left}/{right}")
        assert comparison.status_code == 200, comparison.text
        changed = a.scheme.model_dump(mode="json")
        changed["scene"]["room_m"]["Lx"] += 0.1
        edited = client.put(f"/api/schemes/{a.scheme.scheme_id}", json=changed)
        if status == "failed":
            assert "離開碼 3" in detail.json()["run_notice"]
            assert comparison.json()["table"]["better"] == ""
            assert "不下哪一份比較好的結論" in comparison.json()["table"]["verdict_text"]
            assert edited.status_code == 200, edited.text
        else:
            assert "run_notice" not in detail.json()
            _assert_finished_control(comparison.json())
            assert edited.status_code == 409, edited.text


@pytest.mark.parametrize("error", [FileNotFoundError, OSError, ValueError, KeyError, TypeError])
def test_get_error_preserves_raw_running_without_settling(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]) -> None:
    manager = JobManager(tmp_path, (sys.executable,), "a" * 40, tmp_path / "capabilities")
    run_id = "c" * 32
    path = _path(tmp_path, "runs", run_id)
    raw = json.dumps({"status": "running", "exit_code": 3})
    path.write_text(raw, encoding="utf-8")

    def unreadable(self: JobManager, run_id: str) -> dict[str, object]:
        raise error("計算紀錄讀不出")

    monkeypatch.setattr(JobManager, "get", unreadable)
    status = manager.result_status(run_id)
    assert status.status == "running" and status.exit_code == 3
    assert path.read_text(encoding="utf-8") == raw


def test_compare_build_result_view_rejection_keeps_failed_side_notice(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = pair
    runner = _runner(tmp_path, {a.scheme.scheme_id: (a, 3, False), b.scheme.scheme_id: (b, 0, False)})
    with _client(tmp_path, runner) as client:
        failed, done = _start(client, a), _start(client, b)
        _wait_state(client, failed, "failed")
        _wait_state(client, done, "done")

        def rejected(*args: object, **kwargs: object) -> None:
            raise ValueError("組結果頁資料時拒收")

        monkeypatch.setattr("aosr.gui.app.build_result_view", rejected)
        response = client.get(f"/api/compare/{failed}/{done}")
        assert response.status_code == 409, response.text
        notice = "A：" + RESULT_RUN_LABELS["failed"][2].format(exit_text=RUN_EXIT_TEXT.format(code=3))
        assert notice in response.json()["run_notices"]
        assert response.json()["side"] == "a"


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
            assert reversed_data["table"]["verdict_text"] == (
                "兩份計算都沒有正常完成（A：計算中；B：失敗），不下哪一份比較好的結論")
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


@pytest.mark.parametrize("result_written, expected", [(True, "done"), (False, "failed")])
def test_dead_run_with_missing_stderr_settles_after_restart(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], result_written: bool,
        expected: str) -> None:
    """伺服器重開（沒有行程把手）、錯誤輸出檔又不在：死掉的那一筆照樣結算，不永遠卡在計算中。"""
    dead = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    dead.wait()
    run_id = "e" * 32
    result = _path(tmp_path, "results", run_id)
    for folder in ("results", "runs"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    if result_written:
        save_result(pair[0], result)
    _path(tmp_path, "runs", run_id).write_text(json.dumps({
        "run_id": run_id, "scheme_id": pair[0].scheme.scheme_id, "status": "running",
        "started_at": time.time() - 5, "pid": dead.pid, "exit_code": None,
        "result_path": str(result), "stderr_path": str(tmp_path / "moved-away.stderr")}),
        encoding="utf-8")
    with _client(tmp_path, (sys.executable,)) as client:
        state = client.get(f"/api/runs/{run_id}")
        assert state.status_code == 200, state.text
        assert state.json()["status"] == expected and state.json()["stderr_tail"] == []
        recorded = json.loads(_path(tmp_path, "runs", run_id).read_text(encoding="utf-8"))
        assert recorded["status"] == expected
        if result_written:
            assert _row(client, run_id)["run_status"] == expected
