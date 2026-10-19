"""施工二網頁入口：選填空值、幾何成對規則、伺服器換算與表單規格。"""
from copy import deepcopy
import math
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient
from httpx import Response as HttpxResponse

from aosr.config.paths import config_path
from aosr.gui.problem_text import MATERIAL_MODELS, material_form_spec
from tests.engine._construction_preset_answers import NEW_ROWS
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_construction_presets import preset_document
from tests.engine.test_gui_app import _app

P1 = "perforated_gypsum_8_18r_backed_gw30_ceiling"


def facing_document(*, measured: bool = False) -> dict[str, object]:
    layers = deepcopy(NEW_ROWS[P1][1])
    assembly: dict[str, object] = {"kind": "assembly", "backing": "rigid", "layers": layers}
    material = assembly
    if measured:
        material = {"kind": "measured_impedance", "base_assembly": assembly,
                    "source_and_conditions": "阻抗管垂直入射，測試資料",
                    "points": [{"frequency_hz": 100., "real_pa_s_m": 400., "imag_pa_s_m": -200.},
                               {"frequency_hz": 1000., "real_pa_s_m": 500., "imag_pa_s_m": -100.}]}
    return preset_document(material) | {"scheme_id": "options"}


def first_layer(document: dict[str, object]) -> dict[str, object]:
    scene = cast(dict[str, object], document["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
    material = cast(dict[str, object], zones["floor"][0]["material"])
    base = cast(dict[str, object], material.get("base_assembly", material))
    return cast(list[dict[str, object]], base["layers"])[0]


def send(client: TestClient, route: str, document: dict[str, object]) -> HttpxResponse:
    if route == "save":
        return cast(HttpxResponse, client.put("/api/schemes/options", json=document))
    if route == "edit":
        return cast(HttpxResponse, client.post("/api/input-edit", json={"scheme": document, "action": "wall_preset",
                           "value": "glass_wool_48k_50mm", "wall": "ceiling"}))
    if route == "plan":
        return cast(HttpxResponse, client.post("/api/plan", json=document))
    return cast(HttpxResponse, client.post("/api/validate" if route == "validate" else "/api/input-preview", json=document))


@pytest.mark.parametrize("route", ("validate", "preview", "save", "edit", "plan"))
@pytest.mark.parametrize("measured", (False, True))
def test_optional_null_is_removed_at_each_web_entry(tmp_path: Path, route: str, measured: bool) -> None:
    data = facing_document(measured=measured)
    first_layer(data).update(surface_mass_kg_m2=None, open_area=None, hole_spacing_2_m=None)
    with _app(tmp_path) as client:
        response = send(client, route, data)
        assert response.status_code == 200, response.text
        if route == "validate":
            assert response.json()["problems"] == []
        elif route == "edit":
            assert "surface_mass_kg_m2" not in first_layer(response.json()["scheme"])
        elif route == "plan":
            assert response.json()["message"] == "檢查通過"
        elif route == "save":
            saved = client.get("/api/schemes/options").json()["scheme"]
            assert "surface_mass_kg_m2" not in first_layer(saved)
            assert "open_area" not in first_layer(saved)


@pytest.mark.parametrize("route", ("validate", "preview", "save", "edit"))
def test_required_null_stays_a_plain_refusal(tmp_path: Path, route: str) -> None:
    data = facing_document()
    first_layer(data)["thickness_m"] = None
    with _app(tmp_path) as client:
        response = send(client, route, data)
    if route == "validate":
        assert response.json()["problems"]
    else:
        assert response.status_code == (422 if route == "save" else 400), response.text
    assert "地板" in response.text and "層厚度" in response.text


def test_web_cleanup_retains_required_null_in_document() -> None:
    from aosr.gui.input_setup import check_input_shape, InputShapeError
    from tests.engine._directivity import DIRECTIVITY
    from tests.engine._furniture_cases import CAPABILITIES
    data = facing_document()
    first_layer(data).update(thickness_m=None, surface_mass_kg_m2=None)
    with pytest.raises(InputShapeError):
        check_input_shape(data, CAPABILITIES, DIRECTIVITY)
    assert "thickness_m" in first_layer(data)
    assert first_layer(data)["thickness_m"] is None
    assert "surface_mass_kg_m2" not in first_layer(data)


@pytest.mark.parametrize("route", ("validate", "preview", "save", "edit"))
@pytest.mark.parametrize("case,words", (("both", "只能選一條"), ("rectangular", "排距"), ("backing", "必須一起填正數")))
def test_geometry_and_backing_rules_remain_formal_plain_refusals(tmp_path: Path, route: str, case: str, words: str) -> None:
    data = facing_document()
    layer = first_layer(data)
    if case == "both":
        layer["open_area"] = .155
    elif case == "rectangular":
        layer["hole_pattern"] = "rectangular"
    else:
        layer["backing_fabric_surface_mass_kg_m2"] = None
    with _app(tmp_path) as client:
        response = send(client, route, data)
    assert words in response.text, response.text
    assert "perforated" not in str(response.json().get("error", ""))
    if route == "validate":
        assert response.json()["problems"]
    else:
        assert response.status_code == (422 if route == "save" else 400)


@pytest.mark.parametrize("measured", (False, True))
def test_p1_preview_open_area_matches_independent_hand_calculation(tmp_path: Path, measured: bool) -> None:
    with _app(tmp_path) as client:
        preview = client.post("/api/input-preview", json=facing_document(measured=measured))
    assert preview.status_code == 200, preview.text
    expected = math.pi / 4 * (8 / 18) ** 2
    assert preview.json()["open_area_texts"] == {"open-area-floor-0": f"算出的開孔率約 {expected:.2%}"}


@pytest.mark.parametrize("pattern,area", (("square", math.pi / 4 * (8 / 18) ** 2),
    ("staggered_60", math.pi / (2 * math.sqrt(3)) * (8 / 18) ** 2),
    ("staggered_45", math.pi / 2 * (8 / 18) ** 2), ("rectangular", math.pi / 4 * 8 ** 2 / (18 * 20)),
    ("slotted", 5 / 20)))
def test_geometry_preview_routes_have_server_text(tmp_path: Path, pattern: str, area: float) -> None:
    data = facing_document()
    layer = first_layer(data)
    layer["hole_pattern"] = pattern
    if pattern == "rectangular":
        layer["hole_spacing_2_m"] = .020
    if pattern == "slotted":
        layer.clear()
        layer.update(kind="slotted", thickness_m=.0125, slot_width_m=.005, slot_spacing_m=.020)
    with _app(tmp_path) as client:
        preview = client.post("/api/input-preview", json=data)
    assert preview.status_code == 200, preview.text
    assert preview.json()["open_area_texts"]["open-area-floor-0"] == f"算出的開孔率約 {area:.2%}"


def test_direct_open_area_has_no_calculated_preview_key(tmp_path: Path) -> None:
    data = facing_document()
    layer = first_layer(data)
    layer.pop("hole_spacing_m")
    layer.pop("hole_pattern")
    layer["open_area"] = .155
    with _app(tmp_path) as client:
        response = client.post("/api/input-preview", json=data)
        assert response.status_code == 200, response.text
        assert "open_area_texts" not in response.json()
        layer["open_area"] = None
        assert client.post("/api/input-preview", json=data).status_code == 400


def test_only_new_layer_specs_add_required_flags() -> None:
    spec = material_form_spec()
    for kind in ("fabric", "micro_perforated", "perforated", "slotted"):
        fields = cast(dict[str, dict[str, object]], spec[kind]["fields"])
        for name, model_field in MATERIAL_MODELS[kind].model_fields.items():
            assert fields[name]["required"] is model_field.is_required()
    for kind in ("constant_impedance", "assembly", "porous", "air", "board"):
        assert all("required" not in field for field in cast(dict[str, dict[str, object]], spec[kind]["fields"]).values())


def test_input_scripts_only_render_server_open_area_values() -> None:
    static = config_path("construction_presets.toml").parents[2] / "gui" / "static"
    for name in ("walls.js", "app.js", "measured.js"):
        script = (static / name).read_text()
        for formula in ("Math.PI", "Math.pow", "Math.sqrt", "** 2", "**2", "3.14159", "0.785"):
            assert formula not in script
