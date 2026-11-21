"""元件目錄的每一格都能回到樣板、設定或層的欄位約束。"""
from pathlib import Path

from httpx import Response
import pytest
from starlette.testclient import TestClient

from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.problem_text import kind_word, rounded_number
from aosr.materials.construction_models import Assembly, MicroPerforatedLayer, PerforatedLayer, PorousLayer
from aosr.materials.library import FixedEntry, builtin_entries


def _catalog(folder: Path) -> Response:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=folder)), base_url="http://localhost") as client:
        response: Response = client.get("/api/material-elements")
        assert response.is_success, response.text
        return response


def test_shared_wool_uses_template_labels_sources_and_only_matched_family_sizes(tmp_path: Path) -> None:
    entries = {e.id: e for e in builtin_entries() if isinstance(e, FixedEntry) and isinstance(e.material, Assembly)}
    rows = _catalog(tmp_path).json()["elements"]
    wool = entries["glass_wool_48k_50mm"]
    assert isinstance(wool.material, Assembly)
    layer = wool.material.layers[0]
    assert isinstance(layer, PorousLayer)
    expected_name = f"{kind_word('porous')}　流阻 {rounded_number(layer.flow_resistivity_pa_s_m2)} Pa·s/m²"
    row, = [r for r in rows if r["name"] == expected_name]
    assert {wool.label, entries["slotted_wood_typ9_absorber40_gap98"].label} <= row["templates"].keys()
    assert "流阻：照密度查論文表" in row["flow_sources"]
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    for family in settings.family:
        member, = [r for r in rows if r["kind"] == "porous" and entries[family.template_preset].label in r["templates"]]
        assert member["families"] == {family.id: list(family.thicknesses_m)}
    carpet, = [r for r in rows if entries["carpet_on_concrete"].label in r["templates"]]
    assert not carpet["families"] and carpet["spec_title"] == "還沒填規格"
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
    assert row["range_title"] == "原研究量過的範圍（不是準確性界線）"
    assert f"{rounded_number(ranges.sample_flow_resistivity_min_pa_s_m2)}–{rounded_number(ranges.sample_flow_resistivity_max_pa_s_m2)} Pa·s/m²" in row["ranges"][0]
    assert f"{rounded_number(ranges.measured_frequency_min_hz)}–{rounded_number(ranges.measured_frequency_max_hz)} Hz" in row["ranges"][1]


def test_plate_bounds_and_absent_board(tmp_path: Path) -> None:
    rows = _catalog(tmp_path).json()["elements"]
    for model, attr, sign in ((MicroPerforatedLayer, "le", "≤"), (PerforatedLayer, "ge", "≥")):
        kind = model.model_fields["kind"].default
        edge, = [getattr(m, attr) for m in model.model_fields["hole_diameter_m"].metadata if hasattr(m, attr)]
        plates = [r for r in rows if r["kind"] == kind]
        assert plates and all(f"孔徑 {sign} {rounded_number(edge * 1000)} mm" in r["ranges"][0] for r in plates)
    board, = [r for r in rows if r["kind"] == "board"]
    assert board["template_note"] == "內建樣板沒有這一種"
    assert board["parameters"] == ["面密度", "面阻力"]


def test_synthetic_templates_group_only_matching_identities_in_first_seen_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fabric: dict[str, object] = {"kind": "fabric", "surface_mass_kg_m2": .088, "flow_resistance_pa_s_m": 317.0}
    a: dict[str, object] = {"kind": "air", "thickness_m": .037}
    b: dict[str, object] = {"kind": "air", "thickness_m": .083}

    def sample(key: str, provenance: str, layers: list[dict[str, object]]) -> FixedEntry:
        return FixedEntry.model_validate({"kind": "fixed", "id": key, "label": key, "provenance": provenance, "source": "自編",
                                          "material": {"kind": "assembly", "backing": "rigid", "layers": layers}})

    first = sample("first", "model", [a, fabric, a, b])
    second = sample("second", "measured", [fabric | {"surface_mass_kg_m2": .021}, a, fabric])
    monkeypatch.setattr("aosr.gui.material_elements.builtin_entries", lambda: (first, second))
    rows = _catalog(tmp_path).json()["elements"]
    assert [r["kind"] for r in rows] == ["porous", "fabric", "fabric", "micro_perforated", "perforated", "slotted", "board", "air"]
    fabrics = [r for r in rows if r["kind"] == "fabric"]
    assert [list(r["templates"]) for r in fabrics] == [[first.label, second.label], [second.label]]
    assert fabrics[0]["name"] != fabrics[1]["name"]
    air = rows[-1]
    assert air["provenance"] == [first.provenance, second.provenance]
    assert isinstance(first.material, Assembly)
    sizes = [f"空腔厚度 {rounded_number(l.thickness_m * 1000)} mm" for l in first.material.layers if l.kind == "air"]
    assert air["templates"] == {first.label: list(dict.fromkeys(sizes)), second.label: sizes[:1]}
