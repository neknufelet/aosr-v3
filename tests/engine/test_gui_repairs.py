"""GUI 審查修補的端點、文字與行程組回歸。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser
from starlette.testclient import TestClient
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tests.engine._gui_cache import gui_startup_identity_memo
from aosr.gui.app import (STATIC, GuiSettings, _is_json_media_type, _read_scheme,
                          _require_scheme_id, create_app, repo_root)
from aosr.gui.jobs import JobManager
from aosr.reporting.validation import SchemeValidationError
from tests.engine.test_gui_browser import _open, _serve, browser as browser


COMMIT = "a" * 40


def _client(tmp_path: Path, runner: tuple[str, ...] | None = None) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path,
                                             runner=runner)), base_url="http://localhost")


def _example(client: TestClient, scheme_id: str = "demo") -> dict[str, object]:
    document: dict[str, object] = client.get("/api/example").json()["scheme"]
    document["scheme_id"] = scheme_id
    return document


def _object_cell(document: dict[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], document[key])


def test_calculation_child_receives_only_safe_environment_and_repo_cwd(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Child:
        pid = os.getpid()

    def fake_popen(*args: object, **kwargs: object) -> Child:
        seen.update(kwargs)
        return Child()

    for name in ("MKL_CBWR", "OMP_NUM_THREADS", "KMP_SETTINGS", "OPENBLAS_NUM_THREADS",
                 "JAX_ENABLE_X64", "XLA_FLAGS", "PYTHONPATH"):
        monkeypatch.setenv(name, "inherited")
    outside = tmp_path / "foreign"
    monkeypatch.setenv("PYTHONPATH", str(outside))
    monkeypatch.setattr("aosr.gui.jobs.subprocess.Popen", fake_popen)
    monkeypatch.setattr(JobManager, "get", lambda self, run_id: {"run_id": run_id})
    manager = JobManager(tmp_path, ("runner",), COMMIT, tmp_path / "capabilities")
    manager.start(tmp_path / "scheme")
    child_env = cast(dict[str, str], seen["env"])
    assert seen["cwd"] == repo_root()
    threads = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
    # 三個執行緒變數由計算入口自己給 1，不繼承呼叫端的值；其餘數值庫開關一律不帶。
    assert all(child_env[name] == "1" for name in threads)
    assert all(not name.startswith(("MKL_", "OMP_", "KMP_", "OPENBLAS_", "JAX_", "XLA_"))
               for name in set(child_env) - threads)
    import aosr
    assert child_env["PYTHONPATH"] == str(Path(aosr.__file__).resolve().parent.parent)
    assert str(outside) not in child_env["PYTHONPATH"]


def test_js_uses_one_scale_for_both_axes(browser: Browser, tmp_path: Path) -> None:
    script = (STATIC / "plan.js").read_text()
    assert "innerHTML" not in script
    assert "Math.log" not in script and "Math.pow" not in script
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        room = cast(dict[str, float], page.request.get(f"{url}/api/example").json()["scheme"]["scene"]["room_m"])
        page.wait_for_selector("#plan-xy rect")
        for selector, axis in (("#plan-xy", "Ly"), ("#plan-xz", "Lz")):
            rectangle = page.locator(f"{selector} rect").first
            # 尺寸答案取自範例寫入端；量實際 SVG，兩軸的每公尺比例必須相同。
            width = float(rectangle.get_attribute("width") or "0")
            height = float(rectangle.get_attribute("height") or "0")
            assert width > 0 and height > 0
            assert width / room["Lx"] == pytest.approx(height / room[axis])
        assert not watched.page_errors


def test_primary_z_label_explains_surrounding_points() -> None:
    script = (STATIC / "app.js").read_text()
    page = (STATIC / "index.html").read_text()
    decision = Path("docs/decisions/gui-first-local-2d.md").read_text()
    assert "主位 z 座標" in script and "周圍點要逐點改" in page
    assert "耳高" not in script + page
    assert "主位 z 座標" in decision and "周圍點要逐點改" in decision


def test_js_blank_number_is_null() -> None:
    script = (STATIC / "app.js").read_text()
    assert "Number(input.value)" in script
    assert "input.value.trim() === \"\" ? null" in script


def test_js_saves_form_scheme_id() -> None:
    script = (STATIC / "app.js").read_text()
    assert "scheme.scheme_id = $(\"save-id\").value" in script


def test_scene_error_is_deduplicated_with_structured_path(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        _object_cell(_object_cell(document, "scene"), "impedance_pa_s_per_m_by_wall")["floor"] = -1
        problems = client.post("/api/validate", json=document).json()["problems"]
    # 場景問題每一對都驗到一次，回給網頁的只一條：表單中文欄名、白話，原路徑另外留著。
    assert [(item["fields"], item["paths"]) for item in problems] == [
        (["地板阻抗"], ["scene.impedance_pa_s_per_m_by_wall.floor"])]


def test_pair_error_keeps_pair_path(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        _object_cell(document, "speakers")["left"] = {"x": 3.2, "y": 1.9, "z": 1.2}
        problems = client.post("/api/validate", json=document).json()["problems"]
    assert any(path.startswith("pairs.left.") and path.endswith("source_model.aim_m")
               for item in problems for path in item["paths"])


@pytest.mark.parametrize("endpoint", ["/api/validate", "/api/runs"])
def test_malformed_json_is_structured(tmp_path: Path, endpoint: str) -> None:
    with _client(tmp_path) as client:
        response = client.post(endpoint, content="{", headers={"content-type": "application/json"})
    assert response.status_code == 400 and response.json()["error"]


def test_long_scheme_name_is_structured(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        name = "x" * 300
        for response in (client.get(f"/api/schemes/{name}"),
                         client.get(f"/api/plan/{name}"),
                         client.put(f"/api/schemes/{name}", json=document)):
            assert response.status_code == 400 and response.json()["error"]


def test_corrupt_scheme_is_structured_on_every_reader(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        _object_cell(document, "scene")["density_kg_m3"] = 0
        (tmp_path / "schemes" / "demo.json").write_text(json.dumps(document))
        for response in (client.get("/api/schemes/demo"), client.get("/api/plan/demo"),
                         client.post("/api/runs", json={"scheme_id": "demo"})):
            assert response.status_code == 422
            assert response.json()["problems"][0]["paths"] == ["scene.density_kg_m3"]
            assert response.json()["problems"][0]["fields"] == ["密度"]


def test_disk_scheme_uses_structured_validation(tmp_path: Path) -> None:
    document = json.loads(Path("blueprint/scheme_reference_room.json").read_text())
    document["scene"]["density_kg_m3"] = 0
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(document))
    with pytest.raises(SchemeValidationError) as error:
        _read_scheme(path)
    assert error.value.problems[0].path == "scene.density_kg_m3"


def test_corrupt_run_state_is_structured(tmp_path: Path) -> None:
    run_id = "a" * 32
    with _client(tmp_path) as client:
        (tmp_path / "runs" / f"{run_id}.json").write_text("{")
        response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 400 and response.json()["error"]


def test_unexpected_route_error_is_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_get(self: object, run_id: str) -> dict[str, object]:
        raise RuntimeError("替身異常")

    monkeypatch.setattr("aosr.gui.jobs.JobManager.get", broken_get)
    app = create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path))
    with TestClient(app, base_url="http://localhost", raise_server_exceptions=False) as client:
        response = client.get(f"/api/runs/{'a' * 32}")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]


def test_host_and_content_type_rejected(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/schemes", headers={"host": "attacker.example"}).status_code == 400
        response = client.post("/api/validate", content="{}", headers={"content-type": "text/plain"})
        assert response.status_code == 415 and response.json()["error"]
        response = client.put("/api/schemes/demo", content="{}", headers={"content-type": "text/plain"})
        assert response.status_code == 415 and response.json()["error"]


def test_host_allowlist_and_json_media_type_are_configured(tmp_path: Path) -> None:
    app = create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path))
    assert any(cast(object, item.cls) is TrustedHostMiddleware and
               item.kwargs["allowed_hosts"] == ["127.0.0.1", "localhost"]
               for item in app.user_middleware)
    assert _is_json_media_type("application/json; charset=utf-8")
    assert not _is_json_media_type("text/plain")


def test_null_number_cells_are_required(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        _object_cell(_object_cell(document, "speakers"), "left")["x"] = None
        response = client.post("/api/validate", json=document)
        assert [item["text"] for item in response.json()["problems"]] == ["左聲道喇叭 x 座標：空著沒填"]
        assert "必填" in str(response.json()["problems"][0]["details"])
        document = _example(client)
        _object_cell(document, "scene")["scattering_by_wall"] = {
            wall: None for wall in _object_cell(_object_cell(document, "scene"),
                                               "impedance_pa_s_per_m_by_wall")}
        response = client.post("/api/validate", json=document)
        assert response.json()["problems"] and "必填" in str(response.json()["problems"])


def test_example_uses_own_rho_c_and_formatted_label(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        example = client.get("/api/example").json()
        scene = example["scheme"]["scene"]
        assert example["rho_c"] == scene["density_kg_m3"] * scene["sound_speed_m_s"]
        checked = client.post("/api/validate", json=example["scheme"]).json()
        assert checked["impedance_labels"]["floor"] == "約 ρc 的 4.00 倍"


def test_symlink_children_into_repo_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("aosr.gui.app.repo_root", lambda: repo)
    for name in ("schemes", "runs", "results", "searches"):
        data = tmp_path / name
        data.mkdir()
        (data / name).symlink_to(repo, target_is_directory=True)
        with pytest.raises(ValueError, match="repo"):
            create_app(GuiSettings(engine_commit=COMMIT, data_dir=data))


def test_uppercase_commit_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="engine_commit"):
        create_app(GuiSettings(engine_commit="A" * 40, data_dir=tmp_path))


def test_save_path_must_match_document_id(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client, "other")
        response = client.put("/api/schemes/demo", json=document)
        assert response.status_code == 400 and "scheme_id" in response.json()["error"]
        assert not (tmp_path / "schemes" / "demo.json").exists()


def test_scheme_id_guard_rejects_name_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scheme_id"):
        _require_scheme_id(tmp_path / "demo.json", {"scheme_id": "other"})
    _require_scheme_id(tmp_path / "demo.json", {"scheme_id": "demo"})


@pytest.mark.parametrize("restart,parent_first", [(False, False), (True, False),
                                                    (False, True), (True, True)])
def test_stop_waits_for_term_ignoring_grandchild(
    tmp_path: Path, restart: bool, parent_first: bool,
) -> None:
    script = tmp_path / "family.py"
    ready = tmp_path / "ready"
    script.write_text("import os,signal,time,sys\n"
                      "if os.fork() == 0:\n"
                      " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                      " open(sys.argv[1], 'w').write(str(os.getpid()))\n"
                      " time.sleep(60)\n"
                      "else:\n"
                      + (" while not os.path.exists(sys.argv[1]): time.sleep(0.01)\n"
                         " open(sys.argv[1]+'.parent-exit', 'w').close()\n"
                         if parent_first else " time.sleep(60)\n"))
    runner = (sys.executable, str(script), str(ready))
    manager = JobManager(tmp_path, runner, COMMIT, tmp_path / "capabilities.toml")
    state = manager.start(tmp_path / "scheme.json")
    run_id = str(state["run_id"])
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.02)
        assert ready.exists()
        if parent_first:
            for _ in range(100):
                if (tmp_path / "ready.parent-exit").exists():
                    break
                time.sleep(0.02)
            time.sleep(0.05)
            if restart:
                manager.processes[run_id].wait(timeout=2)
                assert not (Path("/proc") / str(state["pid"])).exists()
            assert manager.get(run_id)["status"] == "running"
        target = (JobManager(tmp_path, runner, COMMIT, tmp_path / "capabilities.toml")
                  if restart else manager)
        stopped = target.stop(run_id)
        assert stopped["status"] == "stopped"
        grandchild = int(ready.read_text())
        grandchild_stat = Path(f"/proc/{grandchild}/stat")
        assert not grandchild_stat.exists() or grandchild_stat.read_text().split(") ", 1)[1][0] == "Z"
    finally:
        try:
            os.killpg(int(str(state["pid"])), signal.SIGKILL)
        except ProcessLookupError:
            pass
        manager.processes[run_id].wait(timeout=2)


def test_stop_racing_status_polls_is_recorded_as_stopped(tmp_path: Path) -> None:
    """停止在背景執行緒、查狀態同時在跑：整組死掉那一刻被查狀態那邊先看到，也要判已停止、不是失敗。"""
    import threading

    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(60)\n")
    manager = JobManager(tmp_path, (sys.executable, str(script)), COMMIT,
                         tmp_path / "capabilities.toml")
    for _ in range(8):
        run_id = str(manager.start(tmp_path / "scheme.json")["run_id"])
        seen: list[str] = []
        stopper = threading.Thread(target=manager.stop, args=(run_id,))
        stopper.start()
        while stopper.is_alive():
            seen.append(str(manager.get(run_id)["status"]))
        stopper.join()
        assert "failed" not in seen
        assert manager.get(run_id)["status"] == "stopped"
        assert json.loads((tmp_path / "runs" / f"{run_id}.json").read_text())["status"] == "stopped"


def test_process_name_with_parenthesis_does_not_break_group_check(tmp_path: Path) -> None:
    """本機任何行程的名字含「) 」，查行程組都不准拋錯（拆 /proc/<pid>/stat 要從最後一個右括號切）。"""
    import subprocess

    odd = subprocess.Popen([sys.executable, "-c",
                            "open('/proc/self/comm','w').write('x) y z'); import time; time.sleep(30)"],
                           start_new_session=True)
    try:
        for _ in range(100):
            if Path(f"/proc/{odd.pid}/comm").read_text().startswith("x) y"):
                break
            time.sleep(0.02)
        manager = JobManager(tmp_path, (sys.executable, "-c", "pass"), COMMIT,
                             tmp_path / "capabilities.toml")
        state: dict[str, object] = {"pid": odd.pid, "started_at": time.time(), **_identity_of(odd.pid)}
        assert manager._group_alive(state) is True
        state["pid"] = 2**22 + 7
        assert manager._group_alive(state) is False
    finally:
        odd.kill()
        odd.wait()


def test_rho_c_label_is_formatted_by_server(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        example = client.get("/api/example").json()
    assert example["rho_c_label"] == "ρc：411.6 帕·秒／公尺"
    assert "example.rho_c_label" in (STATIC / "app.js").read_text()


@pytest.mark.parametrize("endpoint", ["/api/validate", "/api/runs"])
def test_invalid_utf8_body_is_structured_400(tmp_path: Path, endpoint: str) -> None:
    with _client(tmp_path) as client:
        response = client.post(endpoint, content=b'{"a":"\xff"}',
                               headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"] == "內文不是有效 JSON"


def test_group_alive_rescans_members_born_during_the_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#601：領頭行程一派生孫行程就結束時，第一次拍下的 /proc 名單可能還沒有孫行程；判死前要再掃一次。

    把「拍名單時孫行程還沒出生」這個時序固定下來：第一次列 /proc 時把孫行程拿掉，之後照實列。
    """
    ready = tmp_path / "ready"
    script = tmp_path / "family.py"
    script.write_text("import os,sys,time\n"
                      "if os.fork() == 0:\n"
                      " open(sys.argv[1], 'w').write(str(os.getpid()))\n"
                      " time.sleep(60)\n")
    leader = subprocess.Popen([sys.executable, str(script), str(ready)], start_new_session=True)
    try:
        for _ in range(250):
            if ready.exists() and ready.read_text():
                break
            time.sleep(0.02)
        grandchild = ready.read_text()
        assert grandchild.isdecimal()
        leader_stat = Path("/proc") / str(leader.pid) / "stat"
        for _ in range(250):
            if leader_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(0.02)
        assert leader_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        original = Path.iterdir
        scans: list[Path] = []

        def first_listing_misses_grandchild(self: Path) -> Iterator[Path]:
            entries = list(original(self))
            if self == Path("/proc"):
                scans.append(self)
                if len(scans) == 1:
                    return iter([entry for entry in entries if entry.name != grandchild])
            return iter(entries)

        monkeypatch.setattr(Path, "iterdir", first_listing_misses_grandchild)
        manager = JobManager(tmp_path, (sys.executable,), COMMIT, tmp_path / "capabilities.toml")
        assert manager._group_alive({"pid": leader.pid, "started_at": time.time(), **_identity_of(leader.pid)})
        assert scans
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait()


def _identity_of(pid: int) -> dict[str, object]:
    """考卷自己讀核心的開始時脈（第 22 欄，從最後一個右括號切）與開機代號，不呼叫產品的解析器。"""
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {"proc_start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


@pytest.fixture
def unrelated_sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """每題只管理自己開的獨立行程組。"""
    sleeper = subprocess.Popen(["sleep", "60"], start_new_session=True)
    cleanup_wait = sleeper.wait
    try:
        yield sleeper
    finally:
        sleeper.kill()
        cleanup_wait(timeout=2)


def _running_record(tmp_path: Path, pid: int, **identity: object) -> tuple[JobManager, str]:
    manager = JobManager(tmp_path, (sys.executable,), COMMIT, tmp_path / "capabilities")
    run_id = "a" * 32
    (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "status": "running", "started_at": time.time(),
        "pid": pid, "exit_code": None,
        "result_path": str(tmp_path / "results" / f"{run_id}.json"),
        "stderr_path": str(tmp_path / "missing.stderr"), **identity}), encoding="utf-8")
    return manager, run_id


@pytest.mark.parametrize("identity", ["legacy", "legacy_started_now", "wrong_ticks"])
@pytest.mark.parametrize("first", ["get", "stop"])
def test_stale_pid_on_unrelated_group_leader_is_not_signalled(
        tmp_path: Path, unrelated_sleeper: subprocess.Popen[bytes], identity: str,
        first: str) -> None:
    extra: dict[str, object] = {"started_at": time.time() - 86400}
    if identity == "legacy_started_now":
        # 舊紀錄沒有身分欄位：連開始時間剛好對得上也不認領（不靠牆上時間猜）。
        extra = {"started_at": time.time()}
    if identity == "wrong_ticks":
        # sleep 的名稱沒有空白或括號；直接按核心欄號取答案，不呼叫產品的解析器。
        ticks = int((Path("/proc") / str(unrelated_sleeper.pid) / "stat").read_text().split()[21])
        extra = {"proc_start_ticks": ticks + 1,
                 "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    manager, run_id = _running_record(tmp_path, unrelated_sleeper.pid, **extra)
    state = getattr(manager, first)(run_id)
    assert state["status"] == "failed"
    assert "計算行程已不在" in str(state["display_text"])
    assert manager.stop(run_id)["status"] == "failed"
    assert unrelated_sleeper.poll() is None
    assert manager.read_state(run_id)["status"] == "failed"
    assert not manager.read_state(run_id).get("stop_requested")


@pytest.mark.parametrize("first", ["get", "stop"])
def test_boot_id_mismatch_counts_as_dead(
        tmp_path: Path, unrelated_sleeper: subprocess.Popen[bytes], first: str) -> None:
    ticks = int((Path("/proc") / str(unrelated_sleeper.pid) / "stat").read_text().split()[21])
    manager, run_id = _running_record(tmp_path, unrelated_sleeper.pid,
                                     proc_start_ticks=ticks, boot_id="another-boot")
    assert getattr(manager, first)(run_id)["status"] == "failed"
    assert manager.stop(run_id)["status"] == "failed"
    assert unrelated_sleeper.poll() is None


@pytest.mark.parametrize("marker", [None, "b" * 32])
def test_leader_gone_members_without_marker_not_signalled(tmp_path: Path, marker: str | None) -> None:
    ready = tmp_path / "ready"
    script = tmp_path / "orphan.py"
    script.write_text("import os,sys,time\nfrom pathlib import Path\n"
                      "if os.fork() == 0:\n"
                      " Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                      " time.sleep(60)\n")
    env = {"AOSR_GUI_RUN_ID": marker} if marker is not None else {}
    leader = subprocess.Popen([sys.executable, str(script), str(ready)],
                              start_new_session=True, env=env)
    try:
        leader.wait(timeout=2)
        deadline = time.monotonic() + 2
        while (not ready.exists() or not ready.read_text()) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists() and ready.read_text().isdecimal()
        assert not (Path("/proc") / str(leader.pid)).exists()
        # 開機代號對得上，擋下它的只能是「成員身上沒有自己的記號」那一關。
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        manager, run_id = _running_record(tmp_path, leader.pid, boot_id=boot)
        assert manager.stop(run_id)["status"] == "failed"
        member_stat = Path("/proc") / ready.read_text() / "stat"
        assert member_stat.read_text().split()[2] != "Z"
        assert manager.read_state(run_id)["status"] == "failed"
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait(timeout=2)


@pytest.mark.parametrize("first", ["get", "stop"])
@pytest.mark.parametrize("source", ["killpg", "stat"])
def test_permission_error_on_liveness_does_not_raise(
        tmp_path: Path, unrelated_sleeper: subprocess.Popen[bytes],
        monkeypatch: pytest.MonkeyPatch, first: str, source: str) -> None:
    # 身分本來對得上，擋下它的只能是權限錯誤那一關。
    manager, run_id = _running_record(tmp_path, unrelated_sleeper.pid, **_identity_of(unrelated_sleeper.pid))

    def denied(pid: int, sig: int) -> None:
        raise PermissionError("模擬無權探測行程組")

    if source == "killpg":
        monkeypatch.setattr("aosr.gui.jobs.os.killpg", denied)
    else:
        original_read = Path.read_text

        def denied_stat(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
            if path == Path("/proc") / str(unrelated_sleeper.pid) / "stat":
                raise PermissionError("模擬無權讀取核心身分")
            return original_read(path, encoding=encoding, errors=errors)

        monkeypatch.setattr(Path, "read_text", denied_stat)
    assert getattr(manager, first)(run_id)["status"] == "failed"
    assert manager.stop(run_id)["status"] == "failed"
    assert manager.read_state(run_id)["status"] == "failed"
    assert manager.result_status(run_id).status == "failed"
    assert manager.list_recent()["running"] == []
    assert unrelated_sleeper.poll() is None


@pytest.mark.parametrize("pid", [0, -1])
@pytest.mark.parametrize("assume_alive", [False, True])
def test_stop_rejects_nonpositive_pid_without_signalling(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid: int, assume_alive: bool) -> None:
    manager, run_id = _running_record(tmp_path, pid)

    def forbidden(group: int, sig: int) -> None:
        raise AssertionError("無效組號不准探測或送訊號")

    monkeypatch.setattr("aosr.gui.jobs.os.killpg", forbidden)
    if assume_alive:
        monkeypatch.setattr(JobManager, "_group_alive", lambda self, state: True)
    assert manager.stop(run_id)["status"] == "failed"
    assert not manager.read_state(run_id).get("stop_requested")


def test_start_records_boot_id_and_proc_start_ticks(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    script = tmp_path / "marked.py"
    script.write_text("import os,sys,time\nfrom pathlib import Path\n"
                      "Path(sys.argv[1]).write_text(os.environ['AOSR_GUI_RUN_ID'])\n"
                      "time.sleep(60)\n")
    manager = JobManager(tmp_path, (sys.executable, str(script), str(ready)), COMMIT,
                         tmp_path / "capabilities")
    state = manager.start(tmp_path / "scheme.json")
    run_id = str(state["run_id"])
    process = manager.processes[run_id]
    try:
        deadline = time.monotonic() + 2
        while (not ready.exists() or not ready.read_text()) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists() and ready.read_text() == run_id
        recorded = manager.read_state(run_id)
        assert recorded["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        assert recorded["proc_start_ticks"] == int(
            (Path("/proc") / str(process.pid) / "stat").read_text().split()[21])
        assert isinstance(recorded["proc_start_ticks"], int)
    finally:
        process.kill()
        process.wait(timeout=2)


def test_leader_thread_exiting_first_still_settles_done_with_real_exit_code(tmp_path: Path) -> None:
    """主執行緒先走、另一條執行緒寫完結果才以 0 結束：/proc 已把領頭標成殭屍、核心卻還不讓收離開碼那段，不能判失敗。"""
    script = tmp_path / "late.py"
    script.write_text("import ctypes, os, sys, threading, time\n"
                      "out = sys.argv[sys.argv.index('--out') + 1]\n"
                      "def finish():\n"
                      "    time.sleep(1.0)\n"
                      "    open(out, 'w').write('{}')\n"
                      "    os._exit(0)\n"
                      "threading.Thread(target=finish).start()\n"
                      "ctypes.CDLL(None).pthread_exit(None)\n")
    manager = JobManager(tmp_path, (sys.executable, str(script)), COMMIT, tmp_path / "capabilities.toml")
    state = manager.start(tmp_path / "scheme.json")
    run_id = str(state["run_id"])
    leader = Path("/proc") / str(state["pid"]) / "stat"
    seen_zombie = False
    current = state
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            if leader.exists() and leader.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                seen_zombie = True
            current = manager.get(run_id)
            if current["status"] != "running":
                break
            time.sleep(0.02)
    finally:
        try:
            os.killpg(int(str(state["pid"])), signal.SIGKILL)
        except ProcessLookupError:
            pass
        manager.processes[run_id].wait(timeout=5)
    assert seen_zombie, "考卷沒造出「領頭已是殭屍、計算還在收尾」那段"
    assert current["status"] == "done"
    assert current["exit_code"] == 0


@pytest.mark.parametrize("signal_step", ["term", "kill"])
def test_stop_checks_identity_again_right_before_signalling(
        tmp_path: Path, unrelated_sleeper: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch,
        signal_step: str) -> None:
    """就算開頭查狀態那一步誤回「計算中」，送 SIGTERM 與 SIGKILL 前那兩次身分確認都要擋住不相干的行程。"""
    identity = _identity_of(unrelated_sleeper.pid)
    identity["proc_start_ticks"] = int(str(identity["proc_start_ticks"])) + 1
    manager, run_id = _running_record(tmp_path, unrelated_sleeper.pid, **identity)

    def always_running(self: JobManager, rid: str) -> dict[str, object]:
        state = dict(self._load(rid))
        state["status"] = "running"
        return state

    monkeypatch.setattr(JobManager, "get", always_running)
    if signal_step == "kill":
        # 假裝 SIGTERM 之後等逾時，走到 SIGKILL 那一步；第二次等待照實放行。
        waits: list[bool] = []

        def first_wait_times_out(self: JobManager, state: dict[str, object], timeout: float) -> bool:
            waits.append(True)
            return len(waits) > 1

        monkeypatch.setattr(JobManager, "_wait_group", first_wait_times_out)
    manager.stop(run_id)
    assert unrelated_sleeper.poll() is None


@pytest.mark.parametrize("reap_first", [False, True])
def test_unmarked_grandchild_counts_while_the_leader_is_unreaped(tmp_path: Path, reap_first: bool) -> None:
    """伺服器沒重開、領頭已結束但還沒收：殭屍留著組號，沒帶記號的孫行程也算這筆計算的，停止要等到把它停掉。

    領頭一旦被收掉（reap_first），組號就可能被重用，這時只認帶記號的成員：沒帶記號的不認領、只結算。
    """
    script = tmp_path / "unmarked.py"
    ready = tmp_path / "ready"
    grandchild_code = ("import os, signal, sys, time\n"
                       "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                       "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
                       "time.sleep(60)\n")
    script.write_text("import os, subprocess, sys, time\n"
                      "env = {k: v for k, v in os.environ.items() if k != 'AOSR_GUI_RUN_ID'}\n"
                      f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}, sys.argv[1]], env=env)\n"
                      "while not os.path.exists(sys.argv[1]) or not open(sys.argv[1]).read():\n"
                      "    time.sleep(0.01)\n")
    manager = JobManager(tmp_path, (sys.executable, str(script), str(ready)), COMMIT,
                         tmp_path / "capabilities.toml")
    state = manager.start(tmp_path / "scheme.json")
    run_id = str(state["run_id"])
    leader = Path("/proc") / str(state["pid"]) / "stat"
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (
                leader.exists() and leader.read_text().rsplit(")", 1)[1].split()[0] == "Z"):
            time.sleep(0.02)
        assert leader.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        grandchild = int(ready.read_text())
        assert b"AOSR_GUI_RUN_ID=" not in (Path("/proc") / str(grandchild) / "environ").read_bytes()
        if reap_first:
            manager.processes[run_id].wait(timeout=5)
            assert manager.get(run_id)["status"] == "failed"
            assert (Path("/proc") / str(grandchild)).exists()
            return
        assert manager.get(run_id)["status"] == "running"
        assert manager.stop(run_id)["status"] == "stopped"
        grandchild_stat = Path(f"/proc/{grandchild}/stat")
        assert not grandchild_stat.exists() or grandchild_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        try:
            os.killpg(int(str(state["pid"])), signal.SIGKILL)
        except ProcessLookupError:
            pass
        manager.processes[run_id].wait(timeout=5)
