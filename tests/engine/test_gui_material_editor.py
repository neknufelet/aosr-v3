"""材料編輯頁的後端（#1082 段 2-2a）：清單、存、刪都走材料庫的讀寫與檢查；失敗不動檔。"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.material_editor import MaterialStore, _ENTRY, plain_error, user_library_path
from aosr.materials.library import builtin_entries, load_user_library

COMMIT = "a" * 40
CREATE = {"If-None-Match": "*"}


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path)), base_url="http://localhost")


def _family() -> dict[str, object]:
    entry = next(entry for entry in builtin_entries() if entry.kind == "optimizable")
    document = entry.model_dump(mode="json")
    return document | {"id": "my_wool", "label": "我的棉", "preset_members": [], "template_preset": None}


def _assembly(thickness_m: float = 0.05) -> dict[str, object]:
    return {"kind": "assembly", "backing": "rigid", "layers": [
        {"kind": "porous", "thickness_m": thickness_m, "flow_resistivity_pa_s_m2": 12000.0, "material_class": "fibrous"},
        {"kind": "air", "thickness_m": 0.1}]}


def _fixed(material: dict[str, object], material_id: str = "my_panel", label: str = "我的板") -> dict[str, object]:
    return {"kind": "fixed", "id": material_id, "label": label, "provenance": "model", "source": "自己估的", "material": material}


def _measured(imaginary: bool) -> dict[str, object]:
    rows = [{"frequency_hz": 100.0, "real_pa_s_m": 900.0}, {"frequency_hz": 400.0, "real_pa_s_m": 600.0}]
    if imaginary:
        rows = [row | {"imag_pa_s_m": -300.0} for row in rows]
    return {"kind": "measured_impedance", "points": rows, "base_assembly": _assembly(), "source_and_conditions": "阻抗管、垂直入射、50 mm 樣品"}


def test_listing_puts_every_builtin_first_read_only_with_the_form_spec(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        listing = client.get("/api/materials").json()
    rows = listing["materials"]
    assert [row["entry"]["id"] for row in rows] == [entry.id for entry in builtin_entries()]
    assert all(row["builtin"] for row in rows)
    assert listing["user_file_problem"] is None
    assert {"thickness_m", "flow_resistivity_pa_s_m2", "material_class"} <= set(listing["required_fields"]["porous"])
    assert listing["required_fields"]["air"] == ["thickness_m"]
    assert set(listing["material_form"]) >= {"porous", "air", "assembly", "measured_impedance"}
    assert not user_library_path(tmp_path).exists()


def test_new_material_is_saved_then_listed_as_editable_and_reloads_unchanged(tmp_path: Path) -> None:
    document = _family()
    with _client(tmp_path) as client:
        saved = client.put("/api/materials/my_wool", json=document, headers=CREATE)
        assert saved.status_code == 200, saved.text
        rows = client.get("/api/materials").json()["materials"]
    mine = [row for row in rows if not row["builtin"]]
    assert [row["entry"]["id"] for row in mine] == ["my_wool"]
    assert mine[0]["basis"] == "模擬，之後要拿實測對照"
    assert load_user_library(user_library_path(tmp_path))[0].model_dump(mode="json") == mine[0]["entry"]
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == ["materials.json"]


def test_editing_an_existing_material_replaces_it_in_place(tmp_path: Path) -> None:
    first, second = _fixed(_assembly(), "panel_a", "板甲"), _fixed(_assembly(), "panel_b", "板乙")
    with _client(tmp_path) as client:
        for document in (first, second):
            assert client.put(f"/api/materials/{document['id']}", json=document, headers=CREATE).status_code == 200
        changed = _fixed(_assembly(0.08), "panel_a", "板甲改")
        assert client.put("/api/materials/panel_a", json=changed).status_code == 200
    entries = load_user_library(user_library_path(tmp_path))
    assert [entry.id for entry in entries] == ["panel_a", "panel_b"]
    assert entries[0].label == "板甲改"
    assert entries[0].model_dump(mode="json")["material"]["layers"][0]["thickness_m"] == 0.08


def _unchanged_after(tmp_path: Path, request: str, path: str, body: object, headers: dict[str, str] | None = None) -> tuple[int, str]:
    library = user_library_path(tmp_path)
    with _client(tmp_path) as client:
        assert client.put("/api/materials/keep", json=_fixed(_assembly(), "keep", "留著"), headers=CREATE).status_code == 200
        before = library.read_bytes()
        response = client.request(request, path, json=body, headers=headers)
    assert library.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir() if item.is_file()) == ["materials.json"]
    return response.status_code, response.json()["error"]


@pytest.mark.parametrize(("case", "request_kind", "path", "body", "headers", "status", "words"), [
    ("改內建", "PUT", f"/api/materials/{builtin_entries()[0].id}", _fixed(_assembly(), builtin_entries()[0].id, "改內建"), None, 403, "內建材料唯讀"),
    ("刪內建", "POST", f"/api/materials/{builtin_entries()[-1].id}/delete", {}, None, 403, "內建材料唯讀"),
    ("網址代號對不上", "PUT", "/api/materials/other", _fixed(_assembly(), "mine", "我的"), None, 400, "網址的代號 other"),
    ("新增撞到已有代號", "PUT", "/api/materials/keep", _fixed(_assembly(), "keep", "另一個"), CREATE, 409, "已經有代號 keep"),
    ("顯示名撞內建", "PUT", "/api/materials/mine", _fixed(_assembly(), "mine", builtin_entries()[0].label), None, 409, "已經被代號"),
    ("檢查不過", "PUT", "/api/materials/mine", _fixed(_assembly(-0.05), "mine", "我的"), None, 400, "多層構造›第 1 層›多孔層›層厚度：要大於 0"),
    ("刪不存在的", "POST", "/api/materials/nothing/delete", {}, None, 404, "沒有這筆使用者材料"),
])
def test_refused_saves_and_deletes_leave_the_file_byte_identical(
    tmp_path: Path, case: str, request_kind: str, path: str, body: object, headers: dict[str, str] | None,
    status: int, words: str,
) -> None:
    code, message = _unchanged_after(tmp_path, request_kind, path, body, headers)
    assert (code, words in message) == (status, True), (case, message)


def test_deleting_a_user_material_removes_only_that_one(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        for document in (_fixed(_assembly(), "panel_a", "板甲"), _fixed(_assembly(), "panel_b", "板乙")):
            assert client.put(f"/api/materials/{document['id']}", json=document, headers=CREATE).status_code == 200
        assert client.post("/api/materials/panel_a/delete", json={}).status_code == 200
    assert [entry.id for entry in load_user_library(user_library_path(tmp_path))] == ["panel_b"]


@pytest.mark.parametrize(("case", "content"), [
    ("不是 JSON", "{ broken"),
    ("跟內建撞代號", json.dumps({"schema_version": "aosr.material_library.v1",
                              "materials": [_fixed(_assembly(), builtin_entries()[0].id, "別名")]})),
])
def test_an_unreadable_user_file_is_reported_and_never_overwritten(tmp_path: Path, case: str, content: str) -> None:
    library = user_library_path(tmp_path)
    library.write_text(content, encoding="utf-8")
    with _client(tmp_path) as client:
        listing = client.get("/api/materials").json()
        saved = client.put("/api/materials/mine", json=_fixed(_assembly(), "mine", "我的"), headers=CREATE)
        deleted = client.post("/api/materials/mine/delete", json={})
    assert all(row["builtin"] for row in listing["materials"]), case
    assert "讀不進來" in listing["user_file_problem"]
    assert (saved.status_code, deleted.status_code) == (409, 409)
    assert "讀不進來" in saved.json()["error"]
    assert library.read_text(encoding="utf-8") == content


def test_measured_tables_with_and_without_imaginary_parts_round_trip_with_their_label(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        for imaginary, material_id in ((True, "tube_full"), (False, "tube_real")):
            document = _fixed(_measured(imaginary), material_id, material_id)
            assert client.put(f"/api/materials/{material_id}", json=document, headers=CREATE).status_code == 200
        rows = {row["entry"]["id"]: row for row in client.get("/api/materials").json()["materials"] if not row["builtin"]}
    assert (rows["tube_full"]["basis"], rows["tube_real"]["basis"]) == ("實測", "近似")
    assert [point["imag_pa_s_m"] for point in rows["tube_real"]["entry"]["material"]["points"]] == [None, None]


def test_the_page_and_its_two_static_files_are_served(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        page = client.get("/materials")
        assets = [client.get(f"/static/{name}") for name in (
            "materials.js", "materials.css", "vendor/uplot/uPlot.iife.min.js", "vendor/uplot/uPlot.min.css")]
        home = client.get("/")
    assert page.status_code == 200
    # 段 2-2b 起多載入畫吸音曲線的圖表庫（跟結果頁同一份內附檔）
    assert re.findall(r'<script src="/static/([^"]+)"', page.text) == ["vendor/uplot/uPlot.iife.min.js", "materials.js"]
    assert [asset.status_code for asset in assets] == [200, 200, 200, 200]
    assert 'href="/materials"' in home.text


def _errors(document: object) -> str:
    with pytest.raises(ValidationError) as caught:
        _ENTRY.validate_python(document)
    return plain_error(caught.value)


def test_plain_errors_name_the_cell_in_the_form_words() -> None:
    family = _family()
    cavity = family | {"cavity": cast(dict[str, object], family["cavity"]) | {"range_m": [0.0, "x"]}}
    assert _errors(cavity) == "空腔深度›範圍›上限：要填數字"
    measured = _measured(False) | {
        "points": [{"frequency_hz": 100.0, "real_pa_s_m": 900.0}, {"frequency_hz": 200.0, "real_pa_s_m": -1.0}],
        "base_assembly": {"kind": "assembly", "backing": "rigid", "layers": [{"kind": "air", "thickness_m": 0.0}]}}
    assert _errors(_fixed(measured)) == "實測阻抗表›第 2 列›阻抗實部：要大於 0；實測阻抗表›底構造›第 1 層›空氣層›空腔厚度：要大於 0"
    board = _fixed({"kind": "assembly", "backing": "rigid", "layers": [{"kind": "board"}]})
    assert _errors(board) == "多層構造›第 1 層›薄板層›面密度：必填；多層構造›第 1 層›薄板層›面阻力：必填"
    assert _errors(_fixed({"kind": "assembly", "backing": "rigid", "layers": []})) == "多層構造›層：至少要 1 筆"
    assert _errors(_fixed({"kind": "what"})) == "材料：種類不認得"
    assert _errors(_fixed({"kind": "constant_impedance", "impedance_pa_s_m": None}) | {"provenance": None}) == (
        "數字從哪來：必填；固定阻抗›阻抗：必填")
    porosity = {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 12000.0, "material_class": "fibrous",
                "model": "komatsu", "porosity": 0.9}
    message = _errors(_fixed({"kind": "assembly", "backing": "rigid", "layers": [porosity]}))
    assert message.startswith("多層構造›第 1 層›多孔層：小松纖維經驗式 多孔模型不讀孔隙率或曲折度")
    assert not re.search(r"komatsu|porosity", message)


def test_store_listing_reads_the_one_shared_path(tmp_path: Path) -> None:
    store = MaterialStore(tmp_path)
    assert store.path == tmp_path / "materials.json" == user_library_path(tmp_path)


def test_parallel_saves_both_land_and_a_failed_rename_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = MaterialStore(tmp_path)
    store.save("keep", _fixed(_assembly(), "keep", "留著"), create=True)
    barrier = Barrier(2)

    def save(material_id: str) -> None:
        barrier.wait(timeout=10)
        store.save(material_id, _fixed(_assembly(), material_id, material_id), create=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(save, ("panel_a", "panel_b")))
    assert {entry.id for entry in load_user_library(store.path)} == {"keep", "panel_a", "panel_b"}
    before = store.path.read_bytes()

    def broken(self: Path, target: object) -> Path:
        raise OSError("換名失敗")

    monkeypatch.setattr(Path, "replace", broken)
    with pytest.raises(OSError, match="換名失敗"):
        store.save("panel_c", _fixed(_assembly(), "panel_c", "panel_c"), create=True)
    assert store.path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["materials.json"]


def test_a_body_that_is_not_json_is_a_plain_400(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        reply = client.put("/api/materials/mine", content=b"{ broken", headers={"Content-Type": "application/json"})
    assert (reply.status_code, reply.json()) == (400, {"error": "內文不是有效 JSON"})
    assert not user_library_path(tmp_path).exists()
