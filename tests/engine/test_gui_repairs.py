"""GUI 審查修補的端點、文字與行程組回歸。"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient
from starlette.middleware.trustedhost import TrustedHostMiddleware

from aosr.gui.app import (STATIC, GuiSettings, _is_json_media_type, _read_scheme,
                          _require_scheme_id, create_app, repo_root)
from aosr.gui.jobs import JobManager
from aosr.reporting.validation import SchemeValidationError


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
        pid = 12345

    def fake_popen(*args: object, **kwargs: object) -> Child:
        seen.update(kwargs)
        return Child()

    for name in ("MKL_CBWR", "OMP_NUM_THREADS", "KMP_SETTINGS", "OPENBLAS_NUM_THREADS",
                 "JAX_ENABLE_X64", "XLA_FLAGS", "PYTHONPATH"):
        monkeypatch.setenv(name, "inherited")
    monkeypatch.setattr("aosr.gui.jobs.subprocess.Popen", fake_popen)
    monkeypatch.setattr(JobManager, "get", lambda self, run_id: {"run_id": run_id})
    manager = JobManager(tmp_path, ("runner",), COMMIT, tmp_path / "capabilities")
    manager.start(tmp_path / "scheme")
    child_env = cast(dict[str, str], seen["env"])
    assert seen["cwd"] == repo_root()
    assert all(not name.startswith(("MKL_", "OMP_", "KMP_", "OPENBLAS_", "JAX_", "XLA_"))
               and name != "PYTHONPATH" for name in child_env)


def test_js_uses_one_scale_for_both_axes() -> None:
    script = (STATIC / "plan.js").read_text()
    assert "Math.min(520 / width, 320 / height)" in script
    assert "width: width * scale, height: height * scale" in script
    assert "innerHTML" not in script
    assert "Math.log" not in script and "Math.pow" not in script


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
    assert problems == [{"path": "scene.impedance_pa_s_per_m_by_wall.floor",
                         "message": problems[0]["message"]}]


def test_pair_error_keeps_pair_path(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        _object_cell(document, "speakers")["left"] = {"x": 3.2, "y": 1.9, "z": 1.2}
        problems = client.post("/api/validate", json=document).json()["problems"]
    assert any(item["path"].startswith("pairs.left.") and
               item["path"].endswith("source_model.aim_m") for item in problems)


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
            assert response.json()["problems"][0]["path"] == "scene.density_kg_m3"


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
        assert response.json()["problems"] and "必填" in str(response.json()["problems"])
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
    for name in ("schemes", "runs", "results"):
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
                                                    (False, True)])
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
        assert manager._group_alive(odd.pid) is True
        assert manager._group_alive(2**22 + 7) is False
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
