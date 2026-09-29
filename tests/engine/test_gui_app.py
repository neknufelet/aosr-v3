"""本機網頁入口與方案、平面圖、計算狀態。"""
from __future__ import annotations

import json
import os
import re
import socket
import signal
import subprocess
import sys
import time
import uvicorn
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.gui.app import STATIC, GuiSettings, create_app, repo_root
from aosr.gui.labels import label_tables, listening_point_label, speaker_label
from aosr.gui.plan_view import _plan_views, plan_for as _plan
from aosr.reporting.scheme import Scheme
from tests.engine._gui_plan_before_move import RESPONSES


COMMIT = "a" * 40


def test_socketpair_is_not_replaced_by_test_fixture() -> None:
    assert socket.socketpair.__module__ == "socket"


def _app(tmp_path: Path, runner: tuple[str, ...] | None = None) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path,
                                              runner=runner)), base_url="http://localhost")


def test_settings_reject_missing_commit_and_repository_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert repo_root() == Path.cwd()
    assert not GuiSettings(engine_commit=COMMIT).data_dir.is_relative_to(repo_root())
    with pytest.raises(ValueError, match="engine_commit"):
        create_app(GuiSettings(engine_commit="", data_dir=tmp_path))
    monkeypatch.setattr("aosr.gui.app.repo_root", lambda: tmp_path.parent)
    with pytest.raises(ValueError, match="repo"):
        create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path))


def test_page_example_validation_and_scheme_routes(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200 and "<html" in page.text
        assets = {name: client.get(f"/static/{name}")
                  for name in ("app.js", "style.css")}
        assert "api/validate" in assets["app.js"].text
        assert assets["style.css"].status_code == 200
        example = client.get("/api/example")
        assert example.status_code == 200
        document = example.json()["scheme"]
        assert document["source_model"] == "product_default"
        assert document["channel_group"]["feature_match_tolerance_hz"] == 10.0
        scene = document["scene"]
        assert example.json()["rho_c"] == scene["density_kg_m3"] * scene["sound_speed_m_s"]
        tolerance = document["channel_group"]["feature_match_tolerance_hz"]
        note = example.json()["feature_match_note"]
        # 註記講的是峰谷配對容差（不是聲源模型），數字照方案、不多說它查證過。
        assert f"峰谷配對容差 {tolerance:g} Hz" in note and "還沒查證" in note
        assert "考卷" not in note and "聲源" not in note
        checked = client.post("/api/validate", json=document).json()
        assert checked["problems"] == []
        assert checked["impedance_multiples"]["floor"] == pytest.approx(4.0)
        bad = json.loads(json.dumps(document))
        bad["scene"]["impedance_pa_s_per_m_by_wall"]["floor"] = -1
        assert client.post("/api/validate", json=bad).json()["problems"]
        document["scheme_id"] = "demo"
        assert client.put("/api/schemes/demo", json=document).status_code == 200
        assert client.get("/api/schemes/demo").json()["scheme"] == document
        assert "demo" in client.get("/api/schemes").json()["schemes"]
        assert client.put("/api/schemes/..", json=document).status_code != 200


def test_feature_match_note_follows_example_tolerance(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repo = tmp_path / "repo"
    (fake_repo / "blueprint").mkdir(parents=True)
    example = json.loads((repo_root() / "blueprint" / "scheme_reference_room.json").read_text(
        encoding="utf-8"))
    example["channel_group"]["feature_match_tolerance_hz"] = 12.5
    (fake_repo / "blueprint" / "scheme_reference_room.json").write_text(
        json.dumps(example), encoding="utf-8")
    monkeypatch.setattr("aosr.gui.app.repo_root", lambda: fake_repo)
    with _app(tmp_path / "data") as client:
        note = client.get("/api/example").json()["feature_match_note"]
    assert "12.5 Hz" in note and "10 Hz" not in note


def test_labels_route_names_speakers_and_points_apart(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        response = client.get("/api/labels")
        example = client.get("/api/example").json()["scheme"]
    assert response.status_code == 200
    assert response.json() == label_tables()
    assert response.json()["speakers"] == {"left": "左聲道喇叭", "right": "右聲道喇叭"}
    assert response.json()["listening_points"] == {
        "main": "主位", "front": "主位前方", "back": "主位後方", "left": "主位左方",
        "right": "主位右方", "up": "主位上方", "down": "主位下方"}
    # 喇叭的 left 跟座位的 left 不能同名；範例方案每一支喇叭、每一個座位都有中文名。
    assert speaker_label("left") != listening_point_label("left")
    assert all(speaker_label(channel["role"]) != channel["role"]
               for channel in example["channel_group"]["channels"])
    assert all(listening_point_label(point["receiver_id"]) != point["receiver_id"]
               for point in example["receiver_set"]["points"])
    # 表上沒有的代號照原樣回，不猜。
    assert speaker_label("center") == "center"
    assert listening_point_label("seat-3") == "seat-3"


def test_each_page_links_shared_then_own_stylesheet(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        pages = {"home": client.get("/"), "result": client.get("/results/" + "b" * 32),
                 "compare": client.get("/compare/a/b")}
        for page, response in pages.items():
            own = f"{page}.css"
            links = re.findall(r'<link rel="stylesheet" href="/static/(?:vendor/uplot/)?([^"]+)"',
                               response.text)
            assert links.index("style.css") < links.index(own)
            served = client.get(f"/static/{own}")
            assert served.status_code == 200 and served.text.startswith("/*")
        assert client.get(f"/static/{'other' + '.css'}").status_code == 404
    shared = (STATIC / "style.css").read_text(encoding="utf-8")
    # 成功用 .ok（綠、一般粗細），警示與錯誤仍用 .notice。
    assert re.search(r"\.ok\{color:#[0-9a-f]{6};font-weight:400\}", shared)
    assert ".notice{" in shared


def test_plan_includes_every_speaker_and_receiver(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        assert client.put("/api/schemes/demo", json=document).status_code == 200
        plan = client.get("/api/plan/demo").json()
        assert {speaker["id"] for speaker in plan["speakers"]} == set(document["speakers"])
        assert {point["id"] for point in plan["receivers"]} == {
            point["receiver_id"] for point in document["receiver_set"]["points"]}
        assert all(speaker["aim"] for speaker in plan["speakers"])
        assert all("聲道" in speaker["role_label"] for speaker in plan["speakers"])
        assert {point["role_label"] for point in plan["receivers"]} >= {"主位", "周圍點"}
        assert plan["room"] == document["scene"]["room_m"]
        assert all(point["marker"] and point["detail_text"]
                   for point in plan["speakers"] + plan["receivers"])
        assert plan["views"]["plan"] and plan["views"]["side"]
        assert plan["listening_zoom"]["plan"] and plan["listening_zoom"]["side"]


def _without_detail_text(node: object) -> object:
    """整份回應拿掉點的明細字（detail_text）與合併格的明細（detail_lines），其他欄位原樣。"""
    if isinstance(node, dict):
        return {key: _without_detail_text(value) for key, value in node.items()
                if key not in {"detail_text", "detail_lines"}}
    if isinstance(node, list):
        return [_without_detail_text(item) for item in node]
    return node


def test_plan_endpoint_unchanged_after_move_except_names(tmp_path: Path) -> None:
    # 答案是搬家前主線 594b1e6 實跑的原文（出處見 _gui_plan_before_move 的說明），不准重產。
    # #516 視覺一輪第 8 項只改名字：明細從「代號（中文）」改成「中文名（代號）」，中文名查顯示名稱表
    # （喇叭用聲道、座位用座位代號）。所以明細字的冒號後半（座標）與其他每一個欄位仍要跟搬家前一字不差。
    tables = label_tables()
    with _app(tmp_path) as client:
        example = client.get("/api/example").json()["scheme"]
        roles = {channel["speaker_id"]: channel["role"] for channel in example["channel_group"]["channels"]}
        assert set(RESPONSES) == {"product_default", "omnidirectional"}
        for source_model, expected in RESPONSES.items():
            document = {**example, "source_model": source_model}
            response = client.post("/api/plan", json=document)
            assert response.status_code == 200
            before, after = json.loads(expected), response.json()
            assert _without_detail_text(after) == _without_detail_text(before)
            for kind, names, code_of in (("speakers", tables["speakers"], roles),
                                         ("receivers", tables["listening_points"], None)):
                for old, new in zip(before[kind], after[kind], strict=True):
                    code = str(new["id"])
                    name = names[code_of[code] if code_of else code]
                    assert new["detail_text"] == f"{name}（{code}）：" + old["detail_text"].split("：", 1)[1]
            # 合併格的明細就是成員各自的明細字，順序跟搬家前一樣。
            texts = {point["key"]: point["detail_text"] for point in after["speakers"] + after["receivers"]}
            for view in after["views"].values():
                for item in view:
                    assert item["detail_lines"] == [texts[key] for key in item["keys"]]


def test_plan_markers_merge_projection_and_zoom_contains_listening_points() -> None:
    document = json.loads(Path("blueprint/scheme_reference_room.json").read_text())
    scheme = Scheme.model_validate(document)
    plan = _plan(scheme, load_directivity_defaults(config_path("directivity_defaults.toml")))
    speakers = cast(list[dict[str, object]], plan["speakers"])
    receivers = cast(list[dict[str, object]], plan["receivers"])
    points = speakers + receivers
    assert all(item["marker"] and item["detail_text"] for item in points)
    assert {item["marker"] for item in receivers} >= {"主", "前", "後", "左", "右", "上", "下"}
    primary = scheme.receiver_set.primary
    views = cast(dict[str, list[dict[str, object]]], plan["views"])
    zoom = cast(dict[str, dict[str, list[float]]], plan["listening_zoom"])
    for axis, name in ((1, "plan"), (2, "side")):
        projected = {point.receiver_id for point in scheme.receiver_set.points
                     if (point.position_m[0], point.position_m[axis]) ==
                     (primary.position_m[0], primary.position_m[axis])}
        merged = next(item for item in views[name]
                      if f"receiver:{primary.receiver_id}" in cast(list[str], item["keys"]))
        assert set(cast(list[str], merged["keys"])) == {
            f"receiver:{point}" for point in projected}
        bounds = zoom[name]
        for point in scheme.receiver_set.points:
            if point.role.value in {"primary", "surrounding"}:
                assert bounds["u"][0] <= point.position_m[0] <= bounds["u"][1]
                assert bounds["v"][0] <= point.position_m[axis] <= bounds["v"][1]


def test_unknown_surrounding_direction_uses_receiver_id() -> None:
    document = json.loads(Path("blueprint/scheme_reference_room.json").read_text())
    point = next(point for point in document["receiver_set"]["points"]
                 if point["role"] == "surrounding")
    point["direction_relative_to_primary"] = "diagonal"
    scheme = Scheme.model_validate(document)
    plan = _plan(scheme, load_directivity_defaults(config_path("directivity_defaults.toml")))
    receivers = cast(list[dict[str, object]], plan["receivers"])
    shown = next(item for item in receivers if item["id"] == point["receiver_id"])
    assert shown["marker"] == point["receiver_id"]
    # 方向認不得就照實說，不拿代號去名稱表裡猜一個方向。
    assert str(shown["detail_text"]).startswith(f"方向未標示的周圍點（{point['receiver_id']}）")


def test_plan_names_speaker_left_and_point_left_apart() -> None:
    # 平面圖下的點清單、點選明細都用中文名打頭：喇叭 left 與座位 left 以前都寫 left，看起來像同一個。
    document = json.loads(Path("blueprint/scheme_reference_room.json").read_text())
    plan = _plan(Scheme.model_validate(document),
                 load_directivity_defaults(config_path("directivity_defaults.toml")))
    speakers = cast(list[dict[str, object]], plan["speakers"])
    receivers = cast(list[dict[str, object]], plan["receivers"])
    heads = {str(item["key"]): str(item["detail_text"]).split("：", 1)[0] for item in speakers + receivers}
    assert heads["speaker:left"] == f"{speaker_label('left')}（left）"
    assert heads["receiver:left"] == f"{listening_point_label('left')}（left）"
    assert heads["speaker:left"].split("（")[0] != heads["receiver:left"].split("（")[0]
    # 範例方案每一點都查得到中文名：名字那半句不是代號本身。
    assert all(head.split("（")[0] != key.split(":")[1] for key, head in heads.items())


def test_plan_merges_speaker_and_receiver_at_same_projection() -> None:
    speaker: dict[str, object] = {"id": "source", "key": "speaker:source", "marker": "L", "detail_text": "來源",
                                  "point": {"x": 1.0, "y": 2.0, "z": 0.5}}
    receiver: dict[str, object] = {"id": "seat", "key": "receiver:seat", "marker": "主", "role": "primary", "detail_text": "主位",
                                   "point": {"x": 1.0, "y": 2.0, "z": 1.0}}
    plan = _plan_views([speaker], [receiver], False)
    assert {frozenset(cast(list[str], item["keys"])) for item in plan} == {
        frozenset({"speaker:source", "receiver:seat"})}


def test_plan_collision_keys_caption_and_other_seat() -> None:
    document = json.loads(Path("blueprint/scheme_reference_room.json").read_text())
    document["speakers"]["right"] = {"x": 3.2, "y": 1.8, "z": 0.5}
    document["receiver_set"]["points"].append({
        "receiver_id": "seat2", "position_m": [1.5, 1.2, 1.2],
        "role": "other_seat", "importance": 0.0,
        "direction_relative_to_primary": None})
    plan = _plan(Scheme.model_validate(document),
                 load_directivity_defaults(config_path("directivity_defaults.toml")))
    all_points = cast(list[dict[str, object]], plan["speakers"]) + cast(
        list[dict[str, object]], plan["receivers"])
    left = next(item for item in cast(list[dict[str, object]], plan["speakers"])
                if item["id"] == "left")
    assert str(left["detail_text"]).startswith("左聲道喇叭（left）")
    seat = next(item for item in cast(list[dict[str, object]], plan["receivers"]) if item["id"] == "seat2")
    # 名稱表沒有的座位：中文名照它的角色寫，代號放括號，不猜一個名字。
    assert str(seat["detail_text"]).startswith("其他座位（seat2）")
    assert len({item["key"] for item in all_points}) == len(all_points)
    assert {item["key"] for item in all_points} == {
        *(f"speaker:{key}" for key in document["speakers"]),
        *(f"receiver:{point['receiver_id']}" for point in document["receiver_set"]["points"])}
    views = cast(dict[str, list[dict[str, object]]], plan["views"])
    markers = {str(item["key"]): str(item["marker"]) for item in all_points}
    for name in ("plan", "side"):
        seat = next(item for item in views[name]
                    if "receiver:seat2" in cast(list[str], item["keys"]))
        # 其他座位只畫圓點不印字，名字在圖例與明細裡。
        assert seat["drawn"] is True and seat["caption"] == ""
        assert markers["receiver:seat2"] == "座1"
        primary = next(item for item in views[name] if "receiver:main" in cast(list[str], item["keys"]))
        assert primary["drawn"] is True and primary["caption"] == "主"
    # 放大圖每一格的字＝成員短標記用「／」串起來（伺服器給錯時瀏覽器考卷會跟著錯，所以在這裡釘）。
    for name in ("zoom_plan", "zoom_side"):
        for item in views[name]:
            assert item["caption"] == "／".join(markers[key] for key in cast(list[str], item["keys"]))
    collision = next(item for item in views["plan"]
                     if {"speaker:right", "receiver:left"} <= set(cast(list[str], item["keys"])))
    assert collision["caption"] == "R"


def test_plan_post_draws_unsaved_form_and_reports_field_problems(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "unsaved"
        plan = client.post("/api/plan", json=document)
        assert plan.status_code == 200 and plan.json()["message"] == "檢查通過"
        assert plan.json()["room"] == document["scene"]["room_m"]
        assert not (tmp_path / "schemes" / "unsaved.json").exists()
        document["scene"]["impedance_pa_s_per_m_by_wall"]["floor"] = -1
        rejected = client.post("/api/plan", json=document)
        assert rejected.status_code == 422
        assert all("path" in item and "message" in item for item in rejected.json()["problems"])


def test_run_done_status_and_failed_validation(tmp_path: Path) -> None:
    script = tmp_path / "finish.py"
    script.write_text("import sys\nfrom pathlib import Path\n"
                      "Path(sys.argv[sys.argv.index('--out') + 1]).write_text('ok')\n")
    with _app(tmp_path, (sys.executable, str(script))) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        assert client.put("/api/schemes/demo", json=document).status_code == 200
        assert client.post("/api/runs", json={"scheme_id": "missing"}).status_code != 200
        started = client.post("/api/runs", json={"scheme_id": "demo"})
        assert started.status_code == 200
        run_id = started.json()["run_id"]
        for _ in range(100):
            state = client.get(f"/api/runs/{run_id}").json()
            if state["status"] != "running":
                break
            time.sleep(0.02)
        assert state["status"] == "done" and state["exit_code"] == 0
        assert Path(state["result_path"]).read_text() == "ok"
        assert state["elapsed_s"] >= 0 and state["reference_s"] > 0
        assert "完成" in state["display_text"]
        assert state["stderr_tail"] == [] and state["next_step_note"]
    with _app(tmp_path) as client:
        assert client.get(f"/api/runs/{run_id}").json()["status"] == "done"


def test_run_reports_failure_and_rejects_tampered_scheme(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("import sys\nsys.stderr.write('替身失敗\\n')\nsys.exit(5)\n")
    with _app(tmp_path, (sys.executable, str(script))) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        assert client.put("/api/schemes/demo", json=document).status_code == 200
        started = client.post("/api/runs", json={"scheme_id": "demo"})
        assert started.status_code == 200
        run_id = started.json()["run_id"]
        for _ in range(100):
            state = client.get(f"/api/runs/{run_id}").json()
            if state["status"] != "running":
                break
            time.sleep(0.02)
        assert state["status"] == "failed" and state["exit_code"] == 5
        assert "替身失敗" in state["stderr_tail"][-1]
        document["scene"]["impedance_pa_s_per_m_by_wall"]["floor"] = -1
        (tmp_path / "schemes" / "demo.json").write_text(json.dumps(document))
        rejected = client.post("/api/runs", json={"scheme_id": "demo"})
        assert rejected.status_code == 422 and rejected.json()["problems"]


def test_restart_recovers_completed_run_without_invented_exit_code(tmp_path: Path) -> None:
    script = tmp_path / "finish-after-restart.py"
    script.write_text("import sys,time\nfrom pathlib import Path\n"
                      "time.sleep(0.1)\n"
                      "Path(sys.argv[sys.argv.index('--out') + 1]).write_text('ok')\n")
    runner = (sys.executable, str(script))
    with _app(tmp_path, runner) as first:
        document = first.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        assert first.put("/api/schemes/demo", json=document).status_code == 200
        run_id = first.post("/api/runs", json={"scheme_id": "demo"}).json()["run_id"]
        with _app(tmp_path, runner) as restarted:
            for _ in range(100):
                state = restarted.get(f"/api/runs/{run_id}").json()
                if Path(state["result_path"]).is_file():
                    break
                time.sleep(0.02)
            assert Path(state["result_path"]).is_file()
            for _ in range(100):
                state = restarted.get(f"/api/runs/{run_id}").json()
                if state["status"] != "running":
                    break
                time.sleep(0.02)
            assert state["status"] == "done"
            assert state["exit_code"] is None


def test_stop_terminates_process_group(tmp_path: Path) -> None:
    script = tmp_path / "sleep.py"
    marker = tmp_path / "child-finished"
    ready = tmp_path / "child-started"
    child = ("import time; from pathlib import Path; time.sleep(0.8); "
             f"Path({str(marker)!r}).write_text('survived')")
    script.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
                      f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                      f"Path({str(ready)!r}).write_text('ready')\n"
                      "time.sleep(60)\n")
    with _app(tmp_path, (sys.executable, str(script))) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        assert client.put("/api/schemes/demo", json=document).status_code == 200
        run_id = client.post("/api/runs", json={"scheme_id": "demo"}).json()["run_id"]
        state = client.get(f"/api/runs/{run_id}").json()
        try:
            for _ in range(100):
                if ready.is_file():
                    break
                time.sleep(0.01)
            assert ready.is_file()
            with _app(tmp_path, (sys.executable, str(script))) as restarted:
                assert restarted.get(f"/api/runs/{run_id}").json()["status"] == "running"
            stopped = client.post(f"/api/runs/{run_id}/stop", json={})
            assert stopped.json()["status"] == "stopped"
            assert client.get(f"/api/runs/{run_id}").json()["status"] == "stopped"
            time.sleep(0.9)
            assert not marker.exists(), "子行程未隨行程組停止"
        finally:
            try:
                os.killpg(state["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cli_binds_loopback_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.gui.__main__ import main

    called: dict[str, object] = {}

    def fake_run(app: object, *, host: str, port: int, proxy_headers: bool) -> None:
        called.update(host=host, port=port, proxy_headers=proxy_headers)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["aosr.gui", "--engine-commit", COMMIT,
                                       "--data-dir", str(tmp_path), "--port", "8765"])
    main()
    assert called == {"host": "127.0.0.1", "port": 8765, "proxy_headers": False}
    monkeypatch.setattr(sys, "argv", ["aosr.gui"])
    with pytest.raises(SystemExit):
        main()


def test_cli_measures_fingerprint_before_loading_gui_and_calculation(tmp_path: Path) -> None:
    probe = """import sys, uvicorn
from aosr.reporting import calculation_fingerprint as module
def first(**kwargs):
    # 允許清單而不是禁止清單：只禁幾個前綴，別的計算模組先載入就看不見（複查 09-29）。
    allowed = {'aosr', 'aosr.config', 'aosr.config.paths', 'aosr.gui', 'aosr.gui.__main__',
               'aosr.reporting', 'aosr.reporting.calculation_fingerprint'}
    print(sorted(name for name in sys.modules if name.split('.')[0] == 'aosr' and name not in allowed))
    raise RuntimeError('first fingerprint')
module.calculation_fingerprint = first
sys.argv = ['gui', '--engine-commit', 'a' * 40, '--data-dir', sys.argv[1]]
from aosr.gui.__main__ import main
try:
    main()
except RuntimeError as exc:
    assert str(exc) == 'first fingerprint'
else:
    raise AssertionError('fingerprint was not measured')
"""
    completed = subprocess.run([sys.executable, "-c", probe, str(tmp_path)],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", f"量啟動指紋前就載入了：{completed.stdout}"


@pytest.mark.parametrize("given", [False, True])
def test_cli_data_dir_default_lives_only_in_gui_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         given: bool) -> None:
    """命令列不另寫一份預設資料夾：沒給 --data-dir 就用 GuiSettings 的，給了就用給的。"""
    import aosr.gui.__main__ as cli
    import aosr.gui.app as gui_app

    seen: list[GuiSettings] = []

    def fake_create_app(settings: GuiSettings) -> object:
        seen.append(settings)
        return object()

    monkeypatch.setattr(cli, "calculation_fingerprint", lambda **_: "calc-v1:" + "0" * 64)
    monkeypatch.setattr(gui_app, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    argv = ["gui", "--engine-commit", "a" * 40] + (["--data-dir", str(tmp_path)] if given else [])
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert [item.data_dir for item in seen] == [tmp_path if given else GuiSettings.data_dir]
