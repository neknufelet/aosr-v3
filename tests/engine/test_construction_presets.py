"""預設只是填表資料：數字答案來自施工單核過的原表，不是載入器自己生成。"""
from copy import deepcopy
from pathlib import Path
from typing import cast
import subprocess
import sys

import pytest
from pydantic import ValidationError

from aosr.config.construction_presets import load_construction_presets
from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.input_setup import input_defaults
from aosr.reporting.physics_identity import physics_identity_parts
from aosr.reporting.validation import validate_scheme
from aosr.reporting.scheme import Scheme
from tests.engine._construction_preset_answers import NEW_ROWS
from aosr.runtime import child_process_env
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._surface_zone_cases import NEW, zone_document
from tests.engine.test_gui_app import _app

ANSWERS = {
    "glass_wool_48k_50mm": (.05, 19100., None, "09"),
    "glass_wool_48k_50mm_gap50": (.05, 19100., .05, "09"),
    "glass_wool_48k_50mm_gap100": (.05, 19100., .1, "09"),
    "glass_wool_48k_100mm": (.1, 19100., None, "09"),
    "glass_wool_24k_50mm": (.05, 6900., None, "03"),
    "glass_wool_96k_50mm": (.05, 53100., None, "11"),
}


def preset_document(material: dict[str, object]) -> dict[str, object]:
    document = zone_document()
    scene = cast(dict[str, object], document["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    zones["floor"][0]["material"] = deepcopy(material)
    return document


def test_presets_equal_decided_rows_and_validate() -> None:
    presets = load_construction_presets(config_path("construction_presets.toml"))
    assert {p.id for p in presets} == set(ANSWERS) | set(NEW_ROWS)
    assert {p.label for p in presets} == {"48K 玻璃棉 50 mm 貼牆", "48K 玻璃棉 50 mm＋50 mm 空腔",
        "48K 玻璃棉 50 mm＋100 mm 空腔", "48K 玻璃棉 100 mm 貼牆", "24K 玻璃棉 50 mm 貼牆", "96K 玻璃棉 50 mm 貼牆"} | {row[0] for row in NEW_ROWS.values()}
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
    for preset in presets:
        if preset.id in NEW_ROWS:
            label, layers = NEW_ROWS[preset.id]
            assert preset.label == label
            assert preset.material == {"kind": "assembly", "backing": "rigid", "layers": layers}
            assert preset.links and all(link.startswith("https://") for link in preset.links)
            assert preset.source and preset.notes and all(note.strip() for note in preset.notes)
            assert not validate_scheme(preset_document(preset.material), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
            continue
        thickness, sigma, gap, sample = ANSWERS[preset.id]
        layers = [{"kind": "porous", "model": "komatsu", "material_class": "fibrous",
                   "thickness_m": thickness, "flow_resistivity_pa_s_m2": sigma, "flow_resistivity_source": "density_table"}]
        if gap is not None:
            layers.append({"kind": "air", "thickness_m": gap})
        assert preset.material == {"kind": "assembly", "backing": "rigid", "layers": layers}
        assert ranges.sample_flow_resistivity_min_pa_s_m2 <= sigma <= ranges.sample_flow_resistivity_max_pa_s_m2
        assert "Table 3" in preset.source and f"glass wool {sample}" in preset.source
        assert "Komatsu, T. (2008)" in preset.source and "Sato, S., Fujimori, T., Miura, H. (1991)" in preset.source
        assert preset.links and all(link.startswith("https://") for link in preset.links)
        assert "實際產品流阻可能差數倍" in preset.notes
        assert not validate_scheme(preset_document(preset.material), capabilities=CAPABILITIES, directivity=DIRECTIVITY)


def test_presets_stay_outside_physics_identity_and_defaults_are_plain_data() -> None:
    parts = physics_identity_parts(capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert "construction_presets.toml" not in parts.data_files
    assert "aosr.config.construction_presets" not in parts.closure
    presets = cast(list[dict[str, object]], input_defaults()["construction_presets"])
    assert {p["id"] for p in presets} == set(ANSWERS) | set(NEW_ROWS)
    assert all(isinstance(p["material"], dict) for p in presets)


@pytest.mark.parametrize("value", (None, 0., "missing"))
def test_missing_layer_names_wall_layer_and_field_in_plain_text(tmp_path: Path, value: object) -> None:
    """輸入頁預覽回白話 400；存檔跟底版一樣回 422 問題清單，兩邊都寫牆、第幾層、哪一欄。"""
    preset = load_construction_presets(config_path("construction_presets.toml"))[0]
    material = deepcopy(preset.material)
    layer = cast(list[dict[str, object]], material["layers"])[0]
    if value == "missing":
        layer.pop("thickness_m")
    else:
        layer["thickness_m"] = value
    document = preset_document(material)
    document["scheme_id"] = "reference"
    with _app(tmp_path) as client:
        preview = client.post("/api/input-preview", json=document)
        save = client.put("/api/schemes/reference", json=document)
    assert preview.status_code == 400, preview.text
    assert save.status_code == 422, save.text
    for text in (preview.json()["error"], "；".join(row["text"] for row in save.json()["problems"])):
        assert "地板" in text and "第 1 層" in text and "層厚度" in text
        assert "thickness_m" not in text
    assert not (tmp_path / "schemes" / "reference.json").exists()


def test_komatsu_outside_registry_is_refused_before_save(tmp_path: Path) -> None:
    preset = load_construction_presets(config_path("construction_presets.toml"))[0]
    material = deepcopy(preset.material)
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
    cast(list[dict[str, object]], material["layers"])[0]["flow_resistivity_pa_s_m2"] = ranges.sample_flow_resistivity_min_pa_s_m2 / 2
    document = preset_document(material)
    document["scheme_id"] = "out-of-range"
    with _app(tmp_path) as client:
        response = client.put("/api/schemes/out-of-range", json=document)
        assert response.status_code == 422, response.text
        text = "；".join(row["text"] for row in response.json()["problems"])
        assert "地板" in text and "第 1 層" in text and "流阻" in text
        assert f"{ranges.manual_flow_resistivity_min_pa_s_m2:g}" in text
        assert f"{ranges.sample_flow_resistivity_max_pa_s_m2:g}" in text and "Pa·s/m²" in text
        assert "komatsu" not in text
    assert not (tmp_path / "schemes" / "out-of-range.json").exists()


def test_every_preset_runs_through_late_surface_core(tmp_path: Path) -> None:
    env = child_process_env(threads=1)
    env["TMPDIR"] = str(tmp_path)
    child = subprocess.run([sys.executable, "-B", "-m", "tests.engine._construction_presets_probe"],
        cwd=Path(__file__).resolve().parents[2], env=env, text=True, capture_output=True, timeout=120)
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize("case", ("blank-label", "blank-source", "bad-link", "duplicate-id", "duplicate-label", "extra", "non-ascii-id"))
def test_loader_refuses_broken_metadata(tmp_path: Path, case: str) -> None:
    source = config_path("construction_presets.toml").read_text()
    if case == "blank-label":
        source = source.replace('label = "48K 玻璃棉 50 mm 貼牆"', 'label = " "')
    elif case == "blank-source":
        start = source.index('source = """')
        end = source.index('"""', start + len('source = """')) + len('"""')
        source = source[:start] + 'source = " "' + source[end:]
    elif case == "bad-link":
        source = source.replace('https://', 'http://')
    elif case == "duplicate-id":
        source = source.replace('id = "glass_wool_24k_50mm"', 'id = "glass_wool_48k_50mm"')
    elif case == "duplicate-label":
        source = source.replace('label = "24K 玻璃棉 50 mm 貼牆"', 'label = "48K 玻璃棉 50 mm 貼牆"')
    elif case == "non-ascii-id":
        source = source.replace('id = "glass_wool_24k_50mm"', 'id = "玻璃棉"')
    else:
        source = 'unexpected = true\n' + source
    path = tmp_path / "broken.toml"
    path.write_text(source)
    with pytest.raises(ValidationError):
        load_construction_presets(path)


def test_preset_order_and_every_field_survives_disk_roundtrip(tmp_path: Path) -> None:
    presets = load_construction_presets(config_path("construction_presets.toml"))
    assert tuple(p.id for p in presets) == (*ANSWERS, *NEW_ROWS)
    for preset in presets:
        document = preset_document(preset.material)
        path = tmp_path / f"{preset.id}.json"
        path.write_text(Scheme.model_validate(document).model_dump_json())
        saved = Scheme.model_validate_json(path.read_text()).model_dump(mode="json")
        scene = cast(dict[str, object], saved["scene"])
        zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
        assert zones["floor"][0]["material"] == preset.material
