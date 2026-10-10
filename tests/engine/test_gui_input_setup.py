"""伺服器填值與保留考卷；逐欄答案手寫，存檔走正式端點。"""
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from tests.engine import _furniture_cases as cases
from tests.engine._gui_input_answers import CABINETS, FURNITURE
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_app import _app


def edit(client: TestClient, document: dict[str, object], action: str, value: object) -> dict[str, object]:
    response = client.post("/api/input-edit", json={"scheme": document, "action": action, "value": value})
    assert response.status_code == 200, response.text
    return cast(dict[str, object], response.json()["scheme"])


@pytest.mark.parametrize("kind", FURNITURE)
def test_each_furniture_default_every_field(tmp_path: Path, kind: str) -> None:
    with _app(tmp_path) as client:
        document = edit(client, cases.document(), "furniture", kind)
        assert document["furniture"] == [FURNITURE[kind]]
        defaults = client.get("/api/input-defaults").json()
        note = defaults["furniture"][kind]["description"]
        assert "老闆 2026-10-10" in note and "主對話估" in note
        example = client.get("/api/example").json()["scheme"]
        assert {key: point["z"] for key, point in example["speakers"].items()} == {"left": 1.2, "right": 1.2}


@pytest.mark.parametrize("facing,center,yaw", [
    ("north", [3.0, 4.0, 3.85], 0), ("west", [2.0, 3.0, 3.85], 90),
    ("south", [3.0, 2.0, 3.85], 0), ("east", [4.0, 3.0, 3.85], 90),
])
def test_cloud_center_and_depth_axis_then_stays_fixed(tmp_path: Path, facing: str,
                                                     center: list[float], yaw: int) -> None:
    with _app(tmp_path) as client:
        document = edit(client, cases.document(facing=facing), "furniture", "ceiling_cloud")
        items = cast(list[dict[str, object]], document["furniture"])
        assert items[0]["placement"] == {"bottom_center_m": center, "yaw_deg": yaw}
        before = deepcopy(items)
        document = edit(client, document, "shortcut", "listening")
        assert next(item for item in cast(list[dict[str, object]], document["furniture"])
                    if item["furniture_id"] == "cloud") == before[0]


@pytest.mark.parametrize("kind", CABINETS)
def test_representative_two_types_every_cabinet_field(tmp_path: Path, kind: str) -> None:
    with _app(tmp_path) as client:
        document = edit(client, cases.document(), "setup", True)
        document = edit(client, document, "speaker_kind", kind)
        setup = cast(dict[str, object], document["speaker_setup"])
        assert setup == {"kind": kind, "mount": "stand" if kind == "bookshelf" else "floor",
                         "representative": True, "cabinet": CABINETS[kind]}
        document = edit(client, document, "representative", False)
        assert cast(dict[str, object], document["speaker_setup"])["representative"] is False


def test_readonly_items_survive_edit_and_save(tmp_path: Path) -> None:
    extras = [cases.relative_item(furniture_id="chair-one", kind="chair"),
              cases.cloud_item(furniture_id="second-cloud"),
              cases.relative_item(furniture_id="other-sofa", placement={
                  "forward_m": -1, "left_m": 1, "bottom_height_m": 0, "yaw_deg": 90})]
    # 三件原樣保留：位置彼此分開，存檔不依靠假驗證。
    cast(dict[str, object], extras[0]["placement"])["forward_m"] = -1
    cast(dict[str, object], extras[0]["placement"])["left_m"] = -1
    with _app(tmp_path) as client:
        document = edit(client, cases.document(*extras), "furniture", "desk")
        assert cast(list[dict[str, object]], document["furniture"])[:3] == extras
        preview = client.post("/api/input-preview", json=document).json()
        assert [item["item"] for item in preview["readonly"]] == extras
        assert all("這一頁不能改" in item["text"] for item in preview["readonly"])
        document["scheme_id"] = "preserved"
        response = client.put("/api/schemes/preserved", json=document)
        assert response.status_code == 200, response.text
        saved = client.get("/api/schemes/preserved").json()["scheme"]
        # 方案模型本來按代號排序；以代號配對，逐欄核原件，不把清單順序當物理欄位。
        saved_items = {item["furniture_id"]: item for item in saved["furniture"]}
        assert {item["furniture_id"]: saved_items[item["furniture_id"]] for item in extras} == {
            item["furniture_id"]: item for item in extras}


def test_mount_options_and_switch_preserve_height(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        document = edit(client, cases.document(), "setup", True)
        assert client.post("/api/input-preview", json=document).json()["mounts"] == ["stand"]
        refused = client.post("/api/input-edit", json={"scheme": document, "action": "mount", "value": "desk"})
        assert refused.status_code == 400 and "前方桌面" in refused.json()["error"]
        document = edit(client, document, "furniture", "desk")
        assert client.post("/api/input-preview", json=document).json()["mounts"] == ["stand", "desk"]
        document = edit(client, document, "mount", "desk")
        assert document["speakers"] == cases.document()["speakers"]
        document = edit(client, document, "speaker_kind", "floorstanding")
        assert client.post("/api/input-preview", json=document).json()["mounts"] == ["floor"]
        assert document["speakers"] == cases.document()["speakers"]


def test_fixed_role_collision_and_nonzero_relative_angle_remain_readonly(tmp_path: Path) -> None:
    item = cases.relative_item(furniture_id="sofa", placement={
        "forward_m": 1, "left_m": 0.5, "bottom_height_m": 0, "yaw_deg": 90})
    with _app(tmp_path) as client:
        document = cases.document(item)
        readonly = client.post("/api/input-preview", json=document).json()["readonly"]
        assert [record["item"] for record in readonly] == [item]
        response = client.post("/api/input-edit", json={"scheme": document, "action": "furniture", "value": "sofa"})
        assert response.status_code == 400 and "這一頁不能改" in response.json()["error"]
        changed = edit(client, document, "furniture", "desk")
        assert cast(list[dict[str, object]], changed["furniture"])[0] == item


def test_coordinates_height_sentence_and_stand_hint(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        document = edit(client, cases.document(), "furniture", "desk")
        document = edit(client, document, "setup", True)
        preview = client.post("/api/input-preview", json=document).json()
        assert preview["coordinates"]["table"] == "房間底面中心：x 3、y 3.825、z 0.72 公尺"
        document = edit(client, document, "mount", "desk")
        preview = client.post("/api/input-preview", json=document).json()
        assert preview["height_problems"] == {
            "left": "左聲道喇叭 z 座標：高度 1.2 m 跟擺法推出值不同：桌面頂 0.75 m＋聲學中心離箱底 0.205 m＝0.955 m",
            "right": "右聲道喇叭 z 座標：高度 1.2 m 跟擺法推出值不同：桌面頂 0.75 m＋聲學中心離箱底 0.205 m＝0.955 m"}
        document = edit(client, document, "mount", "stand")
        cast(dict[str, dict[str, object]], document["speakers"])["left"]["z"] = 1.21
        assert client.post("/api/input-preview", json=document).json()["stand_hint"] == "搜尋只用一個高度，兩支要一樣"


@pytest.mark.parametrize("shortcut,kinds,heights", [
    ("working", ["desk"], {"main": 1.2, "front": 1.2, "back": 1.2, "left": 1.2,
                           "right": 1.2, "up": 1.3, "down": 1.1}),
    ("listening", ["sofa", "coffee_table"], {"main": 1.05, "front": 1.05, "back": 1.05,
                                           "left": 1.05, "right": 1.05, "up": 1.15, "down": 0.95}),
])
def test_shortcut_handwritten_heights_roundtrip_without_new_fields(tmp_path: Path, shortcut: str,
                                                                kinds: list[str], heights: dict[str, float]) -> None:
    with _app(tmp_path) as client:
        original = client.get("/api/example").json()["scheme"]
        original = edit(client, original, "setup", True)
        original = edit(client, original, "furniture", "ceiling_cloud")
        original_cloud = deepcopy(original["furniture"])
        document = edit(client, original, "shortcut", shortcut)
        items = cast(list[dict[str, object]], document["furniture"])
        assert [item for item in items if item["kind"] != "ceiling_cloud"] == [FURNITURE[kind] for kind in kinds]
        assert [item for item in items if item["kind"] == "ceiling_cloud"] == original_cloud
        points = cast(dict[str, list[dict[str, object]]], document["receiver_set"])["points"]
        assert {point["receiver_id"]: cast(list[float], point["position_m"])[2]
                for point in points} == pytest.approx(heights)
        assert document["speakers"] == original["speakers"] and document["speaker_setup"] == original["speaker_setup"]
        assert document.keys() == original.keys()
        document["scheme_id"] = shortcut
        response = client.put(f"/api/schemes/{shortcut}", json=document)
        assert response.status_code == 200, response.text
        assert client.get(f"/api/schemes/{shortcut}").json()["scheme"] == document
