"""每一區選做法時方案那一區換什麼（#1105，#1082 第 1-3 段）：固定的多層構造貼在原牆上、換一組時原牆不跟著變、
不貼與交給搜尋放回原牆、固定阻抗與實測表直接取代、看不出原牆照實說、指了原方案照原方案取；拒收的原因。

一律用 109 號房的長寬高；原牆是參考方案的固定阻抗（來自已驗輸入），不在考卷裡自己造阻抗數字當原牆。
"""
import json
from collections.abc import Mapping
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
from tests.engine._construction_cases import construction
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
    # 牆照表單上的名字叫（這間 x0 是前牆），不寫座標代號、也不寫「x 起點牆」那種座標名。
    assert code == 400 and "前牆沒有「nope」這一區" in str(data["error"]) and "x0" not in str(data["error"]), data
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


def test_named_original_scheme_wins_over_unwrapping_like_the_improvement_baseline(client: TestClient,
                                                                                    tmp_path: Path) -> None:
    """已經貼過的區、又指了原方案：照原方案取原牆，跟 1-5 改善基準（derive_baseline）同一個先後，不拆貼上的層。"""
    original = room109_document("room109-carpeted")
    _zones(original)["x0"][0]["material"] = {"kind": "assembly", "layers": _preset_layers(OTHER), "backing": "rigid"}
    (tmp_path / "schemes").mkdir(exist_ok=True)
    (tmp_path / "schemes" / "room109-carpeted.json").write_text(Scheme.model_validate(original).model_dump_json())
    document = _cut_document()
    wall = deepcopy(_zones(document)["x0"][0]["material"])
    _put(document, "z1", {"kind": "mounted_on_wall", "layers": _preset_layers(PRESET), "wall": wall})
    code, data = _post(client, document, "z1", choice="original", original_scheme_id="room109-carpeted")
    assert code == 200 and data["origin"] == "original_scheme"
    assert data["material"] == _zones(original)["x0"][0]["material"]


def test_choosing_what_the_original_wall_already_is_keeps_the_wall_instead_of_doubling_it(client: TestClient) -> None:
    """#1175：原牆本身就是那一套多層構造（天花的吊頂），再選同一筆不准包成貼在原牆上（兩層疊在一起）：放原牆、回
    same_as_original 給網頁改成保留原牆；換別的照常貼上去；已經疊成兩層的那一區（老闆 11-15 存的：貼的層＝原牆的層）
    再選同一筆，也拆回原牆一份。"""
    document = _cut_document()
    built = {"kind": "assembly", "layers": _preset_layers(PRESET), "backing": "rigid"}
    _put(document, "z1", built)
    status, data = _post(client, document, "z1", choice="fixed", material_id=PRESET)
    assert status == 200, data
    assert data["material"] == built and data["same_as_original"] is True and "一模一樣" in str(data["note"])
    status, data = _post(client, document, "z1", choice="fixed", material_id=OTHER)
    assert status == 200 and data["same_as_original"] is False
    assert data["material"] == {"kind": "mounted_on_wall", "layers": _preset_layers(OTHER), "wall": built}
    _put(document, "z1", {"kind": "mounted_on_wall", "layers": _preset_layers(PRESET), "wall": built})
    status, data = _post(client, document, "z1", choice="fixed", material_id=PRESET)
    assert status == 200 and data["same_as_original"] is True and data["material"] == built


def test_whole_wall_of_a_six_number_scheme_is_found_through_the_validated_inputs(client: TestClient) -> None:
    """六面都是整面固定阻抗的方案驗證後折回六個數字：整面那一區照計算同一條驗證拿材料，照樣換得了。"""
    document = room109_document()
    wall = deepcopy(_zones(document)["y0"][0]["material"])
    response = client.post("/api/zone-material", json={"scheme": document, "wall": "y0", "zone": "whole_wall",
                                                       "choice": "fixed", "material_id": PRESET})
    data = cast(dict[str, object], response.json())
    assert response.status_code == 200, data
    assert data["material"] == {"kind": "mounted_on_wall", "layers": _preset_layers(PRESET), "wall": wall}


# 一次換好幾區（#1176 第 B 支，``/api/zone-material/batch``）：每一區照單筆同一套規則算、全有或全無、不寫檔。
def _batch(client: TestClient, document: dict[str, object], *changes: Mapping[str, object],
           **extra: object) -> tuple[int, dict[str, object]]:
    response = client.post("/api/zone-material/batch", json={"scheme": document, "changes": list(changes), **extra})
    return response.status_code, cast(dict[str, object], response.json())


def _files(root: Path) -> list[tuple[str, bytes]]:
    return sorted((str(path.relative_to(root)), path.read_bytes()) for path in root.rglob("*") if path.is_file())


def test_a_batch_gives_each_zone_what_the_one_zone_endpoint_gives_and_writes_nothing(client: TestClient, tmp_path: Path) -> None:
    document = _cut_document()
    changes = [{"wall": "x0", "zone": "z1", "choice": "fixed", "material_id": PRESET},
               {"wall": "x0", "zone": "auto1", "choice": "original"},
               {"wall": "x0", "zone": "auto2", "choice": "fixed", "material_id": OTHER}]
    before = _files(tmp_path)
    status, data = _batch(client, document, *changes)
    assert status == 200, data
    singles = [_post(client, document, str(change["zone"]), **{key: value for key, value in change.items()
                                                               if key not in ("wall", "zone")})[1] for change in changes]
    assert data == {"changes": singles}
    assert [cast(dict[str, object], item["material"])["kind"] for item in singles] == ["mounted_on_wall", "constant_impedance",
                                                                                      "mounted_on_wall"]
    assert _files(tmp_path) == before


def test_one_bad_zone_in_the_middle_refuses_the_whole_batch_and_says_which(client: TestClient, tmp_path: Path) -> None:
    """中間那一筆不成：整批不成、一區的結果都不回，寫明是哪一區、為什麼、一區都沒改；第幾筆與那一區自己的原因另外回給網頁。"""
    document = _cut_document()
    good = {"wall": "x0", "zone": "z1", "choice": "fixed", "material_id": PRESET}
    status, data = _batch(client, document, good, {"wall": "x0", "zone": "auto1", "choice": "fixed", "material_id": "no_such_material"},
                          {"wall": "x0", "zone": "auto2", "choice": "original"})
    assert status == 404 and "changes" not in data
    assert data["index"] == 1 and data["wall"] == "x0" and data["zone"] == "auto1"
    assert data["reason"] == "材料庫裡沒有「no_such_material」這一筆"
    assert str(data["error"]).endswith("「自動 auto1」：材料庫裡沒有「no_such_material」這一筆；一區都沒改"), data
    # 畫面跟方案對不上的區、可最佳化材料當固定、指定貼卻沒給材料、同一區出現兩次：照樣整批不成、點名那一區；
    # 給人看的那一句牆照表單上的名字叫，不出現座標代號 x0。
    for bad, status_code, words in (({"wall": "x0", "zone": "nope", "choice": "original"}, 400, "沒有「nope」這一區"),
                                    ({"wall": "x0", "zone": "auto2", "choice": "fixed", "material_id": "glass_wool_48k"}, 400, "可最佳化材料"),
                                    ({"wall": "x0", "zone": "auto2", "choice": "fixed"}, 400, "選固定材料要給材料代號"),
                                    ({"wall": "x0", "zone": "z1", "choice": "original"}, 400, "同一區在這一批裡出現兩次")):
        status, data = _batch(client, document, good, bad)
        assert status == status_code and data["index"] == 1 and words in str(data["reason"]), data
        assert str(data["error"]).endswith("；一區都沒改") and "x0" not in str(data["error"]), data
        if bad["zone"] == "nope":  # 那一區自己的原因也照表單牆名（這間 x0 是前牆），整句只有一種叫法
            assert str(data["reason"]).startswith("前牆沒有「nope」"), data
    # 不是哪一區的問題（材料庫讀不進來、原方案不在了、請求沒有任何一區）：一樣一區都沒改，沒有第幾筆。
    status, data = _batch(client, document, good, original_scheme_id="gone")
    assert status == 404 and data["index"] is None and str(data["error"]).endswith("不在了；在上面「原方案」重選；一區都沒改")
    status, data = _batch(client, document)
    assert status == 400 and "至少一區" in str(data["error"])
    (tmp_path / "materials.json").write_text("{not json", encoding="utf-8")
    status, data = _batch(client, document, good)
    assert status == 400 and data["index"] is None and "使用者材料庫讀不進來" in str(data["error"])
    assert str(data["error"]).endswith("；一區都沒改")


def test_a_zone_picked_as_what_it_already_is_is_flagged_inside_a_batch_too(client: TestClient) -> None:
    """#1175 那一條在一批裡照樣成立：原牆本身就是那一套多層構造的區選同一筆，回原牆一份、旗標 same_as_original；同一批別區照常貼上去。"""
    document = _cut_document()
    built = {"kind": "assembly", "layers": _preset_layers(PRESET), "backing": "rigid"}
    _put(document, "z1", built)
    status, data = _batch(client, document, {"wall": "x0", "zone": "z1", "choice": "fixed", "material_id": PRESET},
                          {"wall": "x0", "zone": "auto1", "choice": "fixed", "material_id": PRESET})
    assert status == 200, data
    first, second = cast(list[dict[str, object]], data["changes"])
    assert first["same_as_original"] is True and first["material"] == built and "一模一樣" in str(first["note"])
    assert second["same_as_original"] is False and cast(dict[str, object], second["material"])["kind"] == "mounted_on_wall"


@pytest.mark.parametrize("backing", ["rigid", "air", "zone_wall"])
def test_assign_construction_obeys_backing_and_remembers_direct_replacement(client: TestClient, backing: str) -> None:
    doc = construction(backing)
    assert client.put(f"/api/materials/{doc['id']}", json=doc).status_code == 200
    scheme = _cut_document()
    original = _zones(scheme)["x0"][0]["material"]
    code, data = _post(client, scheme, "z1", choice="fixed", material_id=doc["id"])
    assert code == 200, data
    if backing == "zone_wall":
        assert data["material"] == {"kind": "mounted_on_wall", "layers": doc["layers"], "wall": original}
    else:
        assert data["material"] == {"kind": "assembly", "layers": doc["layers"], "backing": backing}
        assert "直接取代" in str(data["note"])
    _put(scheme, "z1", data["material"])
    code, restored = _post(client, scheme, "z1", choice="original", previous_construction=data.get("picked_construction"))
    assert code == 200 and restored["origin"] == ("unwrapped" if backing == "zone_wall" else "unknown")


@pytest.mark.parametrize("backing", ["air", "rigid"])
def test_zone_wall_with_identical_original_layers_keeps_that_wall(client: TestClient, backing: str) -> None:
    doc = construction("zone_wall")
    assert client.put(f"/api/materials/{doc['id']}", json=doc).status_code == 200
    scheme = _cut_document()
    original = {"kind": "assembly", "layers": doc["layers"], "backing": backing}
    _put(scheme, "z1", original)
    code, data = _post(client, scheme, "z1", choice="fixed", material_id=doc["id"])
    assert code == 200 and data["same_as_original"] is True and data["material"] == original


def test_zone_wall_refuses_to_mount_over_an_already_mounted_original(client: TestClient, tmp_path: Path) -> None:
    doc = construction("zone_wall")
    assert client.put(f"/api/materials/{doc['id']}", json=doc).status_code == 200
    original = room109_document("built")
    zone = _zones(original)["x0"][0]
    zone["material"] = {"kind": "mounted_on_wall", "layers": doc["layers"], "wall": zone["material"]}
    (tmp_path / "schemes" / "built.json").write_text(Scheme.model_validate(original).model_dump_json())
    code, data = _post(client, _cut_document(), "z1", choice="fixed", material_id=doc["id"], original_scheme_id="built")
    assert code == 400 and "不能再貼一層" in str(data["error"])


def test_unknown_original_cannot_match_a_zone_wall_stack(client: TestClient) -> None:
    base = construction("rigid", id="base_panel", label="整塊面板")
    facing = construction("zone_wall")
    for doc in (base, facing):
        assert client.put(f"/api/materials/{doc['id']}", json=doc).status_code == 200
    scheme = _cut_document()
    code, replaced = _post(client, scheme, "z1", choice="fixed", material_id=base["id"])
    assert code == 200
    _put(scheme, "z1", replaced["material"])
    code, mounted = _post(client, scheme, "z1", choice="fixed", material_id=facing["id"],
                          previous_construction=replaced["picked_construction"])
    assert code == 200 and mounted["origin"] == "unknown"
    assert mounted["same_as_original"] is False and "看不出原牆" in str(mounted["note"])
    assert mounted["material"] == {"kind": "mounted_on_wall", "layers": facing["layers"], "wall": replaced["material"]}


def test_selecting_a_copy_of_this_original_face_keeps_it_original(client: TestClient) -> None:
    doc = construction("air")
    scheme = _cut_document()
    surface = {"kind": "assembly", "backing": "air", "layers": doc["layers"]}
    _put(scheme, "z1", surface)
    assert client.put(f"/api/materials/{doc['id']}", json=doc).status_code == 200
    code, original = _post(client, scheme, "z1", choice="original")
    assert code == 200 and original["origin"] == "current" and "看不出" not in str(original["note"])
    code, picked = _post(client, scheme, "z1", choice="fixed", material_id=doc["id"])
    assert code == 200 and picked["material"] == surface and picked["same_as_original"] is True
    assert "picked_construction" not in picked
