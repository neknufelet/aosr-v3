"""網頁「自訂房間」那一頁（票 #1315）：上傳、檢查、選吸音、起背景計算、讀結果。

背景計算那支子行程在這裡換成一支假的（只把收到的快照抄進結果），真的計算另一題在行程內
用小網格跑：喇叭數、頻率軸、模態與網格圖都要接得上網頁要讀的欄位。
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.testclient import TestClient

from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.gui.app import GuiSettings, create_app
from aosr.config.paths import config_path
from aosr.config.room_shell_import import load_room_shell_import_settings
from aosr.geometry.rhino_room import read_rhino_room
from aosr.gui.material_editor import user_library_path
from aosr.gui.material_surfaces import GROUP_SHORT_WORDS, placement
from aosr.gui.room_model import HARD_KEY, RoomModels, room_model_routes
from aosr.materials.library import builtin_entries, dump_user_library
from aosr.reporting.room_shell_run import run
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins

TEST_ROOM = Path(__file__).parent / "data" / "room_shell_test_room.3dm"
FAKE_RUNNER = """
import json, sys
snapshot, out = sys.argv[1], sys.argv[sys.argv.index("--out") + 1]
data = json.loads(open(snapshot, encoding="utf-8").read())
open(out, "w", encoding="utf-8").write(json.dumps({"snapshot": data}))
"""


def _client(tmp_path: Path) -> TestClient:
    script = tmp_path / "fake-runner.py"
    script.write_text(FAKE_RUNNER, encoding="utf-8")
    data = tmp_path / "data"
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=data,
                                             room_model_runner=(sys.executable, str(script)))),
                      base_url="http://localhost")


def _upload(client: TestClient) -> dict[str, object]:
    response = client.post("/api/room-models",
                           json={"file_name": "room.3dm", "data_base64": base64.b64encode(TEST_ROOM.read_bytes()).decode()})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def _layer_names(model: dict[str, object]) -> list[str]:
    layers = model["layers"]
    assert isinstance(layers, list)
    return [str(layer["name"]) for layer in layers]


def test_upload_lists_checks_layers_and_a_picture(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        page = client.get("/room-model")
        assert page.status_code == 200
        # 頁面自己引用的每一支腳本與樣式表都要拿得到（白名單漏登記就 404）。
        assets = re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', page.text)
        assert any(asset.endswith("room_model.js") for asset in assets)
        assert all(client.get(asset).status_code == 200 for asset in assets)
        first = _upload(client)
        assert first["ok"] is True
        layers = first["layers"]
        assert isinstance(layers, list)
        names = [layer["name"] for layer in layers]
        assert names == sorted(names) and len(names) == 10 and "Cloud_木板" in names
        assert all(layer["area_m2"] > 0 for layer in layers)
        assert str(first["preview_svg"]).count("<polygon") > 50
        choices = first["choices"]
        assert isinstance(choices, list)
        # 硬牆在最前，其餘是整個材料庫（內建，含門、窗、木地板），照「能用在哪」分組、組的順序同材料庫那一頁。
        assert choices[0] == {"key": HARD_KEY, "label": "硬牆", "group": ""}
        builtin = {entry.id: entry for entry in builtin_entries()}
        assert sorted(c["key"] for c in choices[1:]) == sorted(builtin)
        assert {"door_wood_solid", "glass_double_gap30", "wood_floor_on_joists", "glass_wool_48k_50mm"} <= set(builtin)
        order = list(GROUP_SHORT_WORDS.values())
        groups = [c["group"] for c in choices[1:]]
        assert groups == sorted(groups, key=order.index)
        assert all(c["label"] == builtin[c["key"]].label and c["group"] == GROUP_SHORT_WORDS[placement(builtin[c["key"]]).group]
                   for c in choices[1:])
        assert _upload(client)["id"] == first["id"]
        bad = client.post("/api/room-models", json={"data_base64": "不是base64"})
        assert bad.status_code == 400
        # 讀不了的檔回 400，而且不留在磁碟上
        junk = client.post("/api/room-models", json={"data_base64": base64.b64encode(b"not a rhino file").decode()})
        assert junk.status_code == 400
        assert sorted(p.name for p in (tmp_path / "data" / "room-models").iterdir()) == [str(first["id"])]


def test_an_upload_over_the_limit_is_refused_while_reading(tmp_path: Path) -> None:
    models = RoomModels(tmp_path, (sys.executable, "-c", "pass"), "a" * 40, config_path("capabilities.toml"))
    models.body_limit = 1000
    app = Starlette(routes=room_model_routes(models, tmp_path))
    with TestClient(app, base_url="http://localhost") as client:
        big = client.post("/api/room-models", content=b'{"data_base64": "' + b"A" * 2000 + b'"}',
                          headers={"Content-Type": "application/json"})
        assert big.status_code == 400 and "太大" in big.json()["error"]
        # 沒帶長度（分段送）也要在讀到超過上限時就停
        chunked = client.post("/api/room-models", content=iter([b'{"data_base64": "', b"A" * 2000, b'"}']),
                              headers={"Content-Type": "application/json"})
        assert chunked.status_code == 400 and "太大" in chunked.json()["error"]
        small = client.post("/api/room-models", content=b"[", headers={"Content-Type": "application/json"})
        assert small.status_code == 400 and "JSON" in small.json()["error"]


def test_run_freezes_each_layers_library_material_and_returns_the_result(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        model = _upload(client)
        names = _layer_names(model)
        url = f"/api/room-models/{model['id']}/runs"
        assert client.post(url, json={"materials": {names[0]: HARD_KEY}}).status_code == 400
        assert client.post(url, json={"materials": {name: "no_such_material" for name in names}}).status_code == 400
        assert client.post(url, json={"materials": {name: [HARD_KEY] for name in names}}).status_code == 400
        assert client.post("/api/room-models/" + "0" * 64 + "/runs", json={"materials": {}}).status_code == 404
        choice = {name: ("glass_wool_48k_50mm" if name.startswith("Panel") else
                         "door_wood_solid" if name.startswith("Door") else HARD_KEY) for name in names}
        job = client.post(url, json={"materials": choice}).json()
        assert client.post(url, json={"materials": choice}).json()["run_id"] == job["run_id"]  # 重按不重算
        for _ in range(100):
            state = client.get(f"/api/room-model-jobs/{job['run_id']}").json()
            if state["status"] != "running":
                break
            time.sleep(0.1)
        assert state["status"] == "done", state
        # 快照封的是那一筆材料的整份內容（之後材料庫改了，這一跑算的還是當時那一筆）。
        builtin = {entry.id: entry.model_dump(mode="json") for entry in builtin_entries()}
        frozen = state["result"]["snapshot"]["material_by_layer"]
        assert frozen == {name: None if choice[name] == HARD_KEY else builtin[choice[name]] for name in names}
        assert client.get("/api/room-model-jobs/" + "z" * 32).status_code == 404


def test_user_materials_join_the_dropdown_and_a_broken_user_file_leaves_the_builtins(tmp_path: Path) -> None:
    models = RoomModels(tmp_path, (sys.executable, "-c", "pass"), "a" * 40, config_path("capabilities.toml"))
    template = next(entry for entry in builtin_entries() if entry.id == "glass_wool_48k_50mm")
    mine = template.model_copy(update={"id": "my_panel", "label": "我的吸音板"})
    user_library_path(tmp_path).write_text(dump_user_library((mine,)), encoding="utf-8")
    assert {"key": "my_panel", "label": "我的吸音板", "group": GROUP_SHORT_WORDS["wall"]} in models.choices()
    user_library_path(tmp_path).write_text("{", encoding="utf-8")
    keys = [c["key"] for c in models.choices()]
    assert "my_panel" not in keys and len(keys) == len(builtin_entries()) + 1


def test_background_run_gives_one_curve_per_speaker_on_the_official_axis(tmp_path: Path) -> None:
    """真的子行程（冷開機、雙精度）：一層多孔構造（逐頻算複數阻抗）、一層文獻硬面（阻抗表），其餘硬牆；
    同一份模型全部硬牆的結果在這個行程裡算（硬牆不碰算數函式庫），兩份要差得出來。"""
    builtin = {entry.id: entry.model_dump(mode="json") for entry in builtin_entries()}
    snapshot: dict[str, object] = {"model_path": str(TEST_ROOM), "material_by_layer": {
        "Panel_吸音棉": builtin["glass_wool_48k_50mm"], "Cloud_木板": builtin["wood_floor_on_joists"]}}
    source, out = tmp_path / "snapshot.json", tmp_path / "result.json"
    source.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    child = subprocess.run([sys.executable, "-m", "aosr.reporting.room_shell_run", str(source), "--out", str(out),
                            "--engine-commit", "a" * 40, "--capabilities", str(config_path("capabilities.toml")),
                            "--max-frequency-hz", "60"],
                           env=forward_cpu_pins(child_process_env(threads=1)), capture_output=True, text=True, timeout=600)
    assert child.returncode == 0, child.stderr[-2000:]
    result = json.loads(out.read_text(encoding="utf-8"))
    axis = [round(f, 3) for f in FEM_LANE_FREQUENCIES_HZ if f <= 60.0]
    assert result["frequencies_hz"] == axis
    curves = result["curves"]
    speakers = read_rhino_room(TEST_ROOM, load_room_shell_import_settings(
        config_path("room_shell_import.toml")).max_file_bytes).speakers
    assert isinstance(curves, list) and len(speakers) >= 1 and len(curves) == len(speakers)
    assert all(len(curve["spl_db"]) == len(axis) for curve in curves)
    modes = result["modes_hz"]
    assert isinstance(modes, list) and modes[0] > 20.0
    assert str(result["mesh_svg"]).count("<polygon") > 50
    hard = run({"model_path": str(TEST_ROOM), "material_by_layer": {}}, max_frequency_hz=60.0, engine_commit="a" * 40)
    assert hard["modes_hz"] == modes
    hard_curves = hard["curves"]
    assert isinstance(hard_curves, list)
    gaps = [abs(a - b) for curve, rigid in zip(curves, hard_curves, strict=True)
            for a, b in zip(curve["spl_db"], rigid["spl_db"], strict=True)]
    assert max(gaps) > 1.0  # 吸音棉那一層真的有吃到阻抗（全部硬牆時模態峰更尖）
