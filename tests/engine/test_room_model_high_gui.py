"""網頁「自訂房間」那一頁的高音那一段：起算、查工作、背景計算的欄位。

背景計算那支子行程在網頁那幾題換成一支假的（只把收到的快照抄進結果）；真的計算另一題在行程內
用粗一點的分格跑測試房：路徑數要對得上路徑表考卷釘住的數、每張圖要的欄位都在。
"""
from __future__ import annotations

import base64
import dataclasses
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.config.room_shell_import import load_room_shell_import_settings
from aosr.config.paths import config_path
from aosr.config.shell_lane import SHELL_GEOMETRY_TOL_M
from aosr.geometry.rhino_room import read_rhino_room
from aosr.geometry.room_shell import RoomShell
from aosr.geometry.shell_faces import ShellFaces, shell_faces
from aosr.gui.room_model import ABSORPTION_RATIOS
from aosr.physics.shell_paths import shell_path_table
from aosr.reporting import room_shell_high_run
from aosr.reporting.room_shell_high_run import SPEAKER_WALL_NUDGE_M, _off_the_wall, high_frequencies, run

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
    fake = (sys.executable, str(script))
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path / "data",
                                             room_model_runner=fake, room_model_high_runner=fake)),
                      base_url="http://localhost")


def _wait(client: TestClient, url: str) -> dict[str, object]:
    for _ in range(100):
        state: dict[str, object] = client.get(url).json()
        if state["status"] != "running":
            return state
        time.sleep(0.1)
    raise AssertionError(f"{url} 一直在算")


def test_page_has_the_high_button_and_its_script(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        page = client.get("/room-model").text
        assert 'id="run-high"' in page and 'id="high"' in page
        assets = re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', page)
        assert any(asset.endswith("room_model_high.js") for asset in assets)
        assert all(client.get(asset).status_code == 200 for asset in assets)


def test_high_run_uses_the_same_snapshot_as_the_low_run(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        model = client.post("/api/room-models", json={
            "file_name": "room.3dm", "data_base64": base64.b64encode(TEST_ROOM.read_bytes()).decode()}).json()
        names = [str(layer["name"]) for layer in model["layers"]]
        url = f"/api/room-models/{model['id']}/high-runs"
        assert client.post(url, json={"absorption": {names[0]: "hard"}}).status_code == 400
        assert client.post("/api/room-models/" + "0" * 64 + "/high-runs", json={"absorption": {}}).status_code == 404
        choice = {name: ("strong" if name.startswith("Panel") else "hard") for name in names}
        job = client.post(url, json={"absorption": choice}).json()
        assert client.post(url, json={"absorption": choice}).json()["run_id"] == job["run_id"]  # 重按不重算
        low = client.post(f"/api/room-models/{model['id']}/runs", json={"absorption": choice}).json()
        assert low["run_id"] != job["run_id"]  # 低頻、高音各一個工作
        high_state = _wait(client, f"/api/room-model-high-jobs/{job['run_id']}")
        low_state = _wait(client, f"/api/room-model-jobs/{low['run_id']}")
        assert high_state["status"] == low_state["status"] == "done", (high_state, low_state)
        high_result, low_result = high_state["result"], low_state["result"]
        assert isinstance(high_result, dict) and isinstance(low_result, dict)
        assert high_result["snapshot"] == low_result["snapshot"]
        ratios = high_result["snapshot"]["impedance_ratio_by_layer"]
        assert ratios == {name: (dict(ABSORPTION_RATIOS)["strong"] if name.startswith("Panel") else None)
                          for name in names}
        assert client.get("/api/room-model-high-jobs/" + "z" * 32).status_code == 404


def test_background_high_run_has_every_chart_field(tmp_path: Path) -> None:
    """測試房（粗分格省時間）：兩支喇叭的有效路徑逐階數目等於路徑表考卷釘住的數，被擋掉的路徑裡有天花板那條一階。"""
    names = ["Ceiling_石膏板", "Cloud_木板", "Door_金屬門", "Floor_木地板", "Panel_吸音棉", "Wall_乳膠漆",
             "Wall_吸音區A", "Wall_吸音區_B", "Wall_吸音區_C", "Window_玻璃窗"]
    ratios: dict[str, float | None] = {name: None for name in names} | {"Panel_吸音棉": 2.0, "Floor_木地板": 10.0}
    result = run({"model_path": str(TEST_ROOM), "impedance_ratio_by_layer": ratios}, engine_commit="a" * 40,
                 patch_edge_m=2.0)
    axis = [round(f, 3) for f in high_frequencies()]
    assert result["frequencies_hz"] == axis and axis[0] >= 300.0
    speakers = result["speakers"]
    assert isinstance(speakers, list) and len(speakers) == 2
    counts = []
    for speaker in speakers:
        paths, blocked = speaker["paths"], speaker["blocked"]
        counts.append([sum(1 for p in paths if p["order"] == k) for k in range(4)])
        assert paths[0]["order"] == 0 and paths[0]["db"] == 0.0
        assert all(p["t_ms"] > paths[0]["t_ms"] and len(p["points"]) == p["order"] + 2 for p in paths[1:])
        assert blocked and all(len(p["points"]) == p["order"] + 2 for p in blocked)
        for key in ("direct", "early", "late", "total"):
            values = speaker["energy_db"][key]
            assert len(values) == len(axis) and all(math.isfinite(v) for v in values)
    assert counts == [[1, 6, 18, 34], [1, 6, 17, 32]]
    assert any(p["order"] == 1 and p["via"] == ["Ceiling_石膏板"] for p in speakers[0]["blocked"])
    decay = result["decay"]
    assert isinstance(decay, dict)
    points = decay["points"]
    assert points[0]["db"] == 0.0 and -61.0 < points[-1]["db"] <= -60.0  # 算到降 60 dB 為止
    assert all(later["db"] < earlier["db"] and later["t_ms"] > earlier["t_ms"] for earlier, later in zip(points, points[1:]))
    assert isinstance(decay["t60_s"], float) and decay["t60_s"] > 0.0
    triangles, layers = result["triangles"], result["layers"]
    assert isinstance(triangles, dict) and isinstance(layers, list)
    assert sorted(triangles) == names and [layer["name"] for layer in layers] == names
    assert result["solid_layers"] == ["Cloud_木板"]  # 測試房的吸音板是外殼凸出來的，天雲是單獨的造型
    json.dumps(result, ensure_ascii=False)


def _test_room() -> tuple[RoomShell, ShellFaces]:
    settings = load_room_shell_import_settings(config_path("room_shell_import.toml"))
    room = read_rhino_room(TEST_ROOM, settings.max_file_bytes)
    return room, shell_faces(room, settings.coplanar_angle_deg, settings.coplanar_offset_m, SHELL_GEOMETRY_TOL_M)


def _on_back_wall(room: RoomShell, faces: ShellFaces) -> tuple[tuple[float, float, float], int]:
    """把喇叭 1 往它背後那面牆推到正好貼在牆上（嵌牆喇叭），回位置與那一面。"""
    point = np.asarray(room.speakers[0].position)
    planes = np.unique(faces.plane_of[:faces.shell_triangles])
    planes = planes[faces.supporting[planes]]
    heights = point @ faces.plane_normals[planes].T - faces.plane_offsets[planes]
    plane = int(planes[int(np.argmin(heights))])
    on_wall = point - float(np.min(heights)) * faces.plane_normals[plane]
    return (float(on_wall[0]), float(on_wall[1]), float(on_wall[2])), plane


def test_in_wall_speaker_is_moved_off_the_wall() -> None:
    """貼在牆上的喇叭原本被拒（不在房間裡面），往房內挪 5 毫米就算得出路徑表。"""
    room, faces = _test_room()
    on_wall, plane = _on_back_wall(room, faces)
    seat = room.seats[0].position
    with pytest.raises(ValueError, match="不在房間裡面"):
        shell_path_table(faces, on_wall, seat, max_order=1, tol_m=SHELL_GEOMETRY_TOL_M)
    moved = _off_the_wall(on_wall, faces)
    height = float(np.asarray(moved) @ faces.plane_normals[plane] - faces.plane_offsets[plane])
    assert math.isclose(height, SPEAKER_WALL_NUDGE_M, abs_tol=1e-9)
    assert shell_path_table(faces, moved, seat, max_order=1, tol_m=SHELL_GEOMETRY_TOL_M).path_count > 1
    inside = room.speakers[1].position
    assert _off_the_wall(inside, faces) == inside  # 不貼牆的不動


def test_speaker_on_the_seat_is_refused_in_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """喇叭跟聽位同一點原本算出 NaN、結果檔寫不出來；現在一開始就說人話。"""
    room, _ = _test_room()
    on_seat = dataclasses.replace(room.speakers[1], position=room.seats[0].position)
    moved = dataclasses.replace(room, speakers=(room.speakers[0], on_seat))
    monkeypatch.setattr(room_shell_high_run, "read_rhino_room", lambda *_: moved)
    names = sorted({t.layer for t in (*room.shell, *(t for s in room.solids for t in s.triangles))})
    with pytest.raises(ValueError, match="喇叭 2 跟聽位幾乎在同一點"):
        run({"model_path": str(TEST_ROOM), "impedance_ratio_by_layer": dict.fromkeys(names)},
            engine_commit="a" * 40, patch_edge_m=2.0)
