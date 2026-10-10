"""伺服器填值與保留考卷；逐欄答案手寫，存檔走正式端點。"""
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import STATIC, GuiSettings, create_app
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


@pytest.mark.parametrize("identifier,yaw", [("table", 90), ("old-table", 0)])
@pytest.mark.parametrize("action,value", [
    ("furniture", "sofa"), ("furniture", "ceiling_cloud"),
    ("shortcut", "working"), ("shortcut", "listening"),
])
def test_readonly_table_mount_survives_unrelated_edits(tmp_path: Path, identifier: str,
                                                     yaw: int, action: str, value: str) -> None:
    table = deepcopy(FURNITURE["desk"])
    table["furniture_id"] = identifier
    cast(dict[str, object], table["placement"])["yaw_deg"] = yaw
    with _app(tmp_path) as client:
        original = cases.document(table)
        original["speaker_setup"] = {"kind": "bookshelf", "mount": "desk", "representative": True,
                                     "cabinet": CABINETS["bookshelf"]}
        document = edit(client, original, action, value)
        assert document["speaker_setup"] == original["speaker_setup"]
        assert next(item for item in cast(list[dict[str, object]], document["furniture"])
                    if item["furniture_id"] == identifier) == table
        assert client.post("/api/input-preview", json=document).json()["mounts"] == ["stand", "desk"]


def test_mount_changes_only_when_this_action_removes_last_table(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        original = edit(client, cases.document(), "setup", True)
        # 即使原方案沒有桌子而擺法不合法，勾沙發也不得偷偷修原設定。
        cast(dict[str, object], original["speaker_setup"])["mount"] = "desk"
        assert edit(client, original, "furniture", "sofa")["speaker_setup"] == original["speaker_setup"]
        document = edit(client, original, "furniture", "desk")
        removed = edit(client, document, "remove_furniture", "table")
        assert cast(dict[str, object], removed["speaker_setup"])["mount"] == "stand"
        readonly = deepcopy(FURNITURE["desk"]) | {"furniture_id": "old-table"}
        cast(list[dict[str, object]], document["furniture"]).append(readonly)
        retained = edit(client, document, "remove_furniture", "table")
        assert retained["speaker_setup"] == original["speaker_setup"]


@pytest.mark.parametrize("point_index", [0, 1], ids=["primary", "surrounding"])
@pytest.mark.parametrize("shortcut", ["working", "listening"])
def test_shortcut_empty_height_is_chinese_400_and_preserves_scheme(tmp_path: Path, point_index: int,
                                                                 shortcut: str) -> None:
    document = cases.document()
    points = cast(dict[str, list[dict[str, object]]], document["receiver_set"])["points"]
    points[point_index]["position_m"] = [3.0, 3.0, None]
    original = deepcopy(document)
    with _app(tmp_path) as client:
        response = client.post("/api/input-edit", json={"scheme": document, "action": "shortcut", "value": shortcut})
    assert response.status_code == 400
    assert response.json() == {"error": "請先填好主位與周圍點的高度，再按快捷"}
    assert document == original


@pytest.mark.parametrize("endpoint", ["input-preview", "input-edit"])
@pytest.mark.parametrize("malformed", ["scene", "mount", "speakers", "points", "primary"])
def test_malformed_input_scheme_is_chinese_400(tmp_path: Path, endpoint: str, malformed: str) -> None:
    document = cases.document()
    if malformed == "scene":
        document = {"scene": None}
    elif malformed == "mount":
        document["speaker_setup"] = {"kind": "bookshelf", "representative": True, "cabinet": CABINETS["bookshelf"]}
    elif malformed == "speakers":
        document["speakers"] = []
    elif malformed == "points":
        cast(dict[str, object], document["receiver_set"])["points"] = {}
    else:
        points = cast(dict[str, list[dict[str, object]]], document["receiver_set"])["points"]
        points[0]["role"] = "surrounding"
    body = document if endpoint == "input-preview" else {"scheme": document, "action": "furniture", "value": "sofa"}
    with _app(tmp_path) as client:
        response = client.post(f"/api/{endpoint}", json=body)
    assert response.status_code == 400
    assert response.json() == {"error": "方案格式不完整，請先填好房間、喇叭與座位"}


@pytest.mark.parametrize("endpoint", ["input-preview", "input-edit"])
def test_input_endpoints_bad_json_is_chinese_400(tmp_path: Path, endpoint: str) -> None:
    with _app(tmp_path) as client:
        response = client.post(f"/api/{endpoint}", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"error": "內文不是有效 JSON"}


@pytest.mark.parametrize("endpoint,function", [("input-edit", "edit_input"), ("input-preview", "input_preview")])
def test_input_endpoints_product_type_error_is_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 endpoint: str, function: str) -> None:
    def broken(*args: object, **kwargs: object) -> dict[str, object]:
        raise TypeError("產品函式的型別錯誤")

    monkeypatch.setattr(f"aosr.gui.app.{function}", broken)
    document = cases.document()
    body = document if endpoint == "input-preview" else {"scheme": document, "action": "furniture", "value": "sofa"}
    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path))
    with TestClient(app, base_url="http://localhost", raise_server_exceptions=False) as client:
        response = client.post(f"/api/{endpoint}", json=body)
    assert response.status_code == 500
    assert response.json() == {"error": "TypeError: 產品函式的型別錯誤"}
    assert "方案格式不完整" not in response.text


def test_preview_owns_furniture_editability_and_mount_choices(tmp_path: Path) -> None:
    table = deepcopy(FURNITURE["desk"])
    cast(dict[str, object], table["placement"])["yaw_deg"] = 90
    with _app(tmp_path) as client:
        document = cases.document(FURNITURE["sofa"], table)
        document["speaker_setup"] = {"kind": "bookshelf", "mount": "desk", "representative": True,
                                     "cabinet": CABINETS["bookshelf"]}
        preview = client.post("/api/input-preview", json=document).json()
        source = (STATIC / "app.js").read_text(encoding="utf-8")
    assert preview["furniture_items"] == [
        {"furniture_id": "sofa", "role": "sofa", "editable": True},
        {"furniture_id": "table", "role": None, "editable": False}]
    assert preview["furniture_controls"] == {"sofa": True, "table": False, "cloud": True}
    assert preview["mounts"] == ["stand", "desk"]
    assert "editableRole" not in source
    assert "relative_yaw" not in source
    assert "inputDefaults.mounts" not in source


@pytest.mark.parametrize("endpoint", ["preview", "edit"])
def test_non_finite_numbers_are_chinese_400(tmp_path: Path, endpoint: str) -> None:
    # 複查二：Lx 送無限大，正式驗證會列原因，但預覽把原值帶回時 JSON 寫不出來，變成英文 500。
    document = cases.document()
    cast(dict[str, dict[str, object]], document["scene"])["room_m"]["Lx"] = float("inf")
    body: object = document if endpoint == "preview" else {"scheme": document, "action": "furniture", "value": "sofa"}
    with _app(tmp_path) as client:
        # 測試客戶端的 json= 不肯送無限大；自己寫成 JSON 的 Infinity（Python json 預設會寫、也會讀）。
        response = client.post(f"/api/input-{endpoint}", content=json.dumps(body),
                               headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"error": "方案裡有不是有限的數字（無限大或非數），請改成一般數字"}
