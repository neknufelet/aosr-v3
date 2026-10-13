"""審查重現的壞容器、換預設與材料白話；使用正式驗證端點，不建材料物件。"""
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.construction_input import material_field_values, material_input_problems, wall_materials
from aosr.gui.problem_text import plain_problems
from aosr.reporting.validation import SchemeProblem
from tests.engine._furniture_cases import reference_document
from tests.engine.test_construction_presets import preset_document
from tests.engine.test_gui_app import _app
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo


@pytest.mark.parametrize("scene", (None, "x", []))
@pytest.mark.parametrize("route", ("validate", "save"))
def test_non_object_scene_returns_formal_problems(tmp_path: Path, scene: object, route: str) -> None:
    document = reference_document() | {"scene": scene}
    document["scheme_id"] = "broken"
    with _app(tmp_path) as client:
        response = (client.post("/api/validate", json=document) if route == "validate"
                    else client.put("/api/schemes/broken", json=document))
    assert response.status_code == (200 if route == "validate" else 422), response.text
    problems = response.json()["problems"]
    assert problems and any("房間與材料" in str(row["text"]) for row in problems)
    assert not (tmp_path / "schemes" / "broken.json").exists()


@pytest.mark.parametrize("container", ("zones", "wall", "entry", "material", "layers", "layer"))
@pytest.mark.parametrize("bad", (None, "x", []))
def test_material_readers_check_every_container(container: str, bad: object) -> None:
    document = preset_document({"kind": "assembly", "backing": "rigid", "layers": [
        {"kind": "porous", "thickness_m": .05, "flow_resistivity_pa_s_m2": 19100.}]})
    scene = cast(dict[str, object], document["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
    material = cast(dict[str, object], zones["floor"][0]["material"])
    if container == "zones":
        scene["surface_zones_by_wall"] = bad
    elif container == "wall":
        cast(dict[str, object], zones)["floor"] = bad
    elif container == "entry":
        cast(list[object], zones["floor"])[0] = bad
    elif container == "material":
        zones["floor"][0]["material"] = bad
    elif container == "layers":
        material["layers"] = bad
    else:
        cast(list[object], material["layers"])[0] = bad
    assert not material_input_problems(document, ())
    fields = material_field_values(document)
    assert not any(name.startswith("layer-floor-") for name in fields)
    if container in ("zones", "wall", "entry", "material"):
        assert "floor" not in wall_materials(document)


@pytest.mark.parametrize("layers", ("missing", [], None))
@pytest.mark.parametrize("route", ("preview", "validate"))
def test_missing_layer_list_is_plain_refusal(tmp_path: Path, layers: object, route: str) -> None:
    with _app(tmp_path) as client:
        preset = client.get("/api/input-defaults").json()["construction_presets"][0]
        material = deepcopy(preset["material"])
        if layers == "missing":
            material.pop("layers")
        else:
            material["layers"] = layers
        document = preset_document(material)
        response = client.post("/api/input-preview" if route == "preview" else "/api/validate", json=document)
    assert response.status_code == (400 if route == "preview" else 200), response.text
    body = response.json()
    text = body["error"] if route == "preview" else "；".join(row["text"] for row in body["problems"])
    assert "地板材料構造" in text and "至少一層" in text
    assert "第 1 層" not in text and "layers" not in text


@pytest.mark.parametrize("case", ("half", "range"))
@pytest.mark.parametrize("other_wall", (False, True))
def test_preset_replaces_bad_target_but_keeps_other_wall_errors(tmp_path: Path, case: str, other_wall: bool) -> None:
    with _app(tmp_path) as client:
        presets = client.get("/api/input-defaults").json()["construction_presets"]
        material = deepcopy(presets[0]["material"])
        layer = material["layers"][0]
        ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
        layer["thickness_m" if case == "half" else "flow_resistivity_pa_s_m2"] = (
            None if case == "half" else ranges.sample_flow_resistivity_max_pa_s_m2 * 2)
        document = preset_document(material)
        if other_wall:
            scene = cast(dict[str, object], document["scene"])
            zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
            zones["ceiling"][0]["material"] = deepcopy(material)
        original = deepcopy(document)
        response = client.post("/api/input-edit", json={"scheme": document, "action": "wall_preset",
            "value": presets[1]["id"], "wall": "floor"})
        assert response.status_code == (400 if other_wall else 200), response.text
        if other_wall:
            text = response.json()["error"]
            assert "天花材料構造" in text and "第 1 層" in text
            assert "地板材料構造" not in text
        else:
            changed = response.json()["scheme"]
            assert changed["scene"]["surface_zones_by_wall"]["floor"][0]["material"] == presets[1]["material"]
            expected = deepcopy(original)
            cast(dict[str, object], cast(dict[str, object], expected["scene"])["surface_zones_by_wall"])["floor"] = [
                {"area": "whole_wall", "material": presets[1]["material"]}]
            assert changed == expected
            assert not client.post("/api/validate", json=changed).json()["problems"]
    assert document == original


def test_preset_leaves_half_filled_room_to_formal_validation(tmp_path: Path) -> None:
    """牆以外的半填數字跟其他填值動作一樣不擋換預設，留給正式驗證列原因。"""
    with _app(tmp_path) as client:
        presets = client.get("/api/input-defaults").json()["construction_presets"]
        document = preset_document(presets[0]["material"])
        cast(dict[str, object], cast(dict[str, object], document["scene"])["room_m"])["Lx"] = None
        response = client.post("/api/input-edit", json={"scheme": document, "action": "wall_preset",
            "value": presets[1]["id"], "wall": "floor"})
        assert response.status_code == 200, response.text
        changed = response.json()["scheme"]
        assert changed["scene"]["surface_zones_by_wall"]["floor"][0]["material"] == presets[1]["material"]
        assert changed["scene"]["room_m"]["Lx"] is None
        assert client.post("/api/validate", json=changed).json()["problems"]


@pytest.mark.parametrize("field,value,label", (("kind", "slab", "種類"),
    ("model", "foo", "多孔模型"), ("material_class", "wood", "材料類別")))
@pytest.mark.parametrize("layer_kind", ("porous", "air"))
def test_unknown_layer_codes_have_plain_chinese_display(
    tmp_path: Path, field: str, value: str, label: str, layer_kind: str,
) -> None:
    with _app(tmp_path) as client:
        material = client.get("/api/input-defaults").json()["construction_presets"][0]["material"]
        if layer_kind == "air":
            material["layers"] = [{"kind": "air", "thickness_m": .05}]
        material["layers"][0][field] = value
        response = client.post("/api/validate", json=preset_document(material))
    assert response.status_code == 200, response.text
    rows = response.json()["problems"]
    assert rows
    text = "；".join(row["text"] for row in rows)
    assert "地板材料構造" in text and label in text
    for code in ("kind", "model", "material_class", "komatsu", "miki", "wilson_2015", "mechel_1976", "fibrous", "foam"):
        assert code not in text, text
    assert "代號 層種類代號" not in text


@pytest.mark.parametrize("container", ("scene", "zones", "wall", "entry"))
@pytest.mark.parametrize("bad", (None, "x", []))
def test_preset_checks_target_containers_before_replacing(tmp_path: Path, container: str, bad: object) -> None:
    with _app(tmp_path) as client:
        preset = client.get("/api/input-defaults").json()["construction_presets"][0]
        document = preset_document(preset["material"])
        scene = cast(dict[str, object], document["scene"])
        zones = cast(dict[str, object], scene["surface_zones_by_wall"])
        if container == "scene":
            document["scene"] = bad
        elif container == "zones":
            scene["surface_zones_by_wall"] = bad
        elif container == "wall":
            zones["floor"] = bad
        else:
            cast(list[object], zones["floor"])[0] = bad
        response = client.post("/api/input-edit", json={"scheme": document, "action": "wall_preset",
            "value": preset["id"], "wall": "floor"})
    assert response.status_code == 400, response.text
    assert response.json()["error"]


def test_material_message_keeps_codes_inside_alphanumeric_names() -> None:
    problem = SchemeProblem("scene.surface_zones_by_wall.floor.0.material", "floor1 1floor floor_a afloor；floor")
    text = str(plain_problems((problem,), {})[0]["text"])
    assert "floor1 1floor floor_a afloor" in text
    assert text.endswith("；地板")
