"""每一區選做法時方案那一區換什麼（#1105，#1082 第 1-3 段）：固定的多層構造貼在原牆上、換一組時原牆不跟著變、
不貼與交給搜尋放回原牆、固定阻抗與實測表直接取代、看不出原牆照實說、指了原方案照原方案取；拒收的原因。

一律用 109 號房的長寬高；原牆是參考方案的固定阻抗（來自已驗輸入），不在考卷裡自己造阻抗數字當原牆。
"""
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.zone_edit import CutBox, cut_wall
from aosr.materials.construction_models import Assembly
from aosr.materials.library import FixedEntry, builtin_entries
from aosr.reporting.scheme import Scheme
from tests.engine._surface_zone_cases import NEW, room109_document

PRESET = "glass_wool_48k_50mm"
OTHER = "fabric_pet_a_gap100"
_TABLE: dict[str, object] = {
    "kind": "measured_impedance", "source_and_conditions": "阻抗管內徑 40 mm、垂直入射、樣品 50 mm",
    "points": [{"frequency_hz": f, "real_pa_s_m": r} for f, r in ((100.0, 900.0), (200.0, 700.0), (400.0, 500.0))],
    "base_assembly": {"kind": "assembly", "backing": "rigid", "layers": [
        {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 19100.0, "material_class": "fibrous"}]}}
USER_LIBRARY = {"schema_version": "aosr.material_library.v1", "materials": [
    {"kind": "fixed", "id": "panel_measured", "label": "實測吸音板（只有實部）", "provenance": "measured",
     "source": "阻抗管", "material": _TABLE}]}


def _zones(document: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    return cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])


def _cut_document() -> dict[str, object]:
    """x0 牆畫一個框，其餘自動補滿；三區都是原牆（參考方案的固定阻抗）。"""
    document = room109_document()
    result = cut_wall(document, "x0", [CutBox(id="z1", u_m=(0.0, 1.0), v_m=(0.0, 1.2))])
    _zones(document)["x0"] = cast(list[dict[str, object]], result["zones"])
    return document


def _preset_layers(material_id: str) -> list[object]:
    entry = next(item for item in builtin_entries() if item.id == material_id)
    assert isinstance(entry, FixedEntry) and isinstance(entry.material, Assembly)
    return cast(list[object], entry.material.model_dump(mode="json")["layers"])


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")


def _post(client: TestClient, document: dict[str, object], zone: str, **choice: object) -> tuple[int, dict[str, object]]:
    response = client.post("/api/zone-material", json={"scheme": document, "wall": "x0", "zone": zone, **choice})
    return response.status_code, cast(dict[str, object], response.json())


def _put(document: dict[str, object], zone: str, material: object) -> None:
    next(item for item in _zones(document)["x0"] if item.get("id") == zone)["material"] = material


def test_fixed_assembly_mounts_on_the_original_wall_and_switching_keeps_that_wall(client: TestClient) -> None:
    document = _cut_document()
    wall = deepcopy(_zones(document)["x0"][0]["material"])
    status, data = _post(client, document, "z1", choice="fixed", material_id=PRESET)
    assert status == 200, data
    mounted = cast(dict[str, object], data["material"])
    assert mounted == {"kind": "mounted_on_wall", "layers": _preset_layers(PRESET), "wall": wall}
    assert data["origin"] == "current" and "貼在原牆上" in str(data["note"]) and "看不出原牆" not in str(data["note"])
    # 換另一組：還是貼在同一面原牆上（不是貼在上一組外面）。
    _put(document, "z1", mounted)
    status, data = _post(client, document, "z1", choice="fixed", material_id=OTHER)
    assert status == 200 and data["origin"] == "unwrapped"
    assert data["material"] == {"kind": "mounted_on_wall", "layers": _preset_layers(OTHER), "wall": wall}
    # 不貼／交給搜尋：放回原牆。
    _put(document, "z1", data["material"])
    status, data = _post(client, document, "z1", choice="original")
    assert status == 200 and data["material"] == wall and data["origin"] == "unwrapped"
    Scheme.model_validate(document)
    # 剛切出來、沒貼過的區選「交給搜尋」：現在的材料就是原牆，不跳警告。
    status, data = _post(client, document, "auto1", choice="original")
    assert status == 200 and data["origin"] == "current" and data["note"] == "這一區放回原牆"


def test_measured_table_replaces_the_wall_and_cannot_be_unwrapped_without_an_original(client: TestClient,
                                                                                       tmp_path: Path) -> None:
    (tmp_path / "materials.json").write_text(json.dumps(USER_LIBRARY, ensure_ascii=False), encoding="utf-8")
    document = _cut_document()
    wall = deepcopy(_zones(document)["x0"][0]["material"])
    status, data = _post(client, document, "z1", choice="fixed", material_id="panel_measured")
    assert status == 200, data
    material = cast(dict[str, object], data["material"])
    assert material["kind"] == "measured_impedance" and "直接取代" in str(data["note"])
    assert [point["imag_pa_s_m"] for point in cast(list[dict[str, object]], material["points"])] == [0.0, 0.0, 0.0]
    # 沒指原方案：拆不出原牆，照現在的材料不動、照實說。
    _put(document, "z1", material)
    status, data = _post(client, document, "z1", choice="original")
    assert status == 200 and data["origin"] == "unknown" and data["material"] == material
    assert "看不出原牆" in str(data["note"])
    # 指了原方案：照原方案蓋住這一區的那一塊取原牆。
    original = room109_document("room109-original")
    (tmp_path / "schemes").mkdir(exist_ok=True)
    (tmp_path / "schemes" / "room109-original.json").write_text(Scheme.model_validate(original).model_dump_json())
    status, data = _post(client, document, "z1", choice="original", original_scheme_id="room109-original")
    assert status == 200 and data["origin"] == "original_scheme" and data["material"] == wall


@pytest.mark.parametrize(("choice", "status", "words"), [
    ({"choice": "fixed", "material_id": "glass_wool_48k"}, 400, "可最佳化材料"),
    ({"choice": "fixed", "material_id": "no_such_material"}, 404, "沒有「no_such_material」"),
    ({"choice": "fixed"}, 400, "要給材料代號"),
    ({"choice": "original", "original_scheme_id": "gone"}, 404, "不在了"),
])
def test_bad_choices_are_refused_in_plain_words(client: TestClient, choice: dict[str, object], status: int,
                                                words: str) -> None:
    code, data = _post(client, _cut_document(), "z1", **choice)
    assert code == status and words in str(data["error"])


def test_unknown_zone_unreadable_library_and_half_filled_form_are_refused(client: TestClient, tmp_path: Path) -> None:
    document = _cut_document()
    code, data = _post(client, document, "nope", choice="original")
    assert code == 400 and "沒有「nope」這一區" in str(data["error"])
    (tmp_path / "materials.json").write_text("{not json", encoding="utf-8")
    code, data = _post(client, document, "z1", choice="fixed", material_id=PRESET)
    assert code == 400 and "使用者材料庫讀不進來" in str(data["error"])
    broken = deepcopy(document)
    cast(dict[str, object], broken["scene"])["room_m"] = {"Lx": -1.0, "Ly": 1.0, "Lz": 1.0}
    code, data = _post(client, broken, "z1", choice="original")
    assert code == 400 and "方案表單還有格子沒填好" in str(data["error"])


def test_original_zone_that_is_itself_mounted_cannot_take_another_layer(client: TestClient, tmp_path: Path) -> None:
    original = room109_document("room109-treated")
    wall = deepcopy(_zones(original)["x0"][0]["material"])
    _zones(original)["x0"][0]["material"] = {"kind": "mounted_on_wall", "layers": _preset_layers(PRESET), "wall": wall}
    (tmp_path / "schemes").mkdir(exist_ok=True)
    (tmp_path / "schemes" / "room109-treated.json").write_text(Scheme.model_validate(original).model_dump_json())
    document = _cut_document()
    _put(document, "z1", {"kind": "measured_impedance", **{k: v for k, v in _TABLE.items() if k != "kind"},
                          "points": [{**point, "imag_pa_s_m": 0.0} for point in cast(list[dict[str, object]], _TABLE["points"])]})
    code, data = _post(client, document, "z1", choice="fixed", material_id=OTHER, original_scheme_id="room109-treated")
    assert code == 400 and "不能再貼一層" in str(data["error"])
