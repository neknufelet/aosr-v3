"""元件目錄的每一格都能回到樣板、設定或層的欄位約束。"""
from typing import cast
from pathlib import Path

from httpx import Response
import pytest
from starlette.testclient import TestClient

from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.material_names import family_display_name
from aosr.gui.material_elements import Fields, _word
from aosr.gui.problem_text import kind_word, material_form_spec, rounded_number
from aosr.materials.construction_models import Assembly, BoardLayer, MicroPerforatedLayer, PerforatedLayer, PorousLayer
from aosr.materials.library import FixedEntry, builtin_entries
from tests.engine._construction_preset_answers import FITTED_VPR_IDS


def _catalog(folder: Path) -> Response:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=folder)), base_url="http://localhost") as client:
        response: Response = client.get("/api/material-elements")
        assert response.is_success, response.text
        return response


def test_shared_wool_uses_template_labels_sources_and_only_matched_family_sizes(tmp_path: Path) -> None:
    entries = {e.id: e for e in builtin_entries() if isinstance(e, FixedEntry) and isinstance(e.material, Assembly)}
    rows = _catalog(tmp_path).json()["elements"]
    wool = entries["glass_wool_48k_50mm"]
    row, = [r for r in rows if r["kind"] == "porous" and wool.label in r["templates"]]
    assert {wool.label, entries["slotted_wood_typ9_absorber40_gap98"].label} <= row["templates"].keys()
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    labels = {e.id: family_display_name(e.label) for e in builtin_entries()}
    for family in settings.family:
        member, = [r for r in rows if r["kind"] == "porous" and entries[family.template_preset].label in r["templates"]]
        template = entries[family.template_preset].material
        assert isinstance(template, Assembly)
        layer, = [layer for layer in template.layers if isinstance(layer, PorousLayer)]
        assert member["name"] == f"{labels[family.id]}　流阻 {rounded_number(layer.flow_resistivity_pa_s_m2)} Pa·s/m²"
        assert member["families"] == {family.id: list(family.thicknesses_m)}
    carpet, = [r for r in rows if entries["carpet_on_concrete"].label in r["templates"]]
    assert carpet["template_note"] == "用這組物性的內建樣板："
    assert carpet["flow_sources"] == [f"流阻：未註明，樣板：{entries['carpet_on_concrete'].label}"]
    assert carpet["name"].startswith(kind_word("porous") + "　")
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
    sample = next(e for e in builtin_entries() if e.id == "mpp_t1mm_d0p5_gap50")
    assert isinstance(sample, FixedEntry) and isinstance(sample.material, Assembly)
    layer, = [layer for layer in sample.material.layers if isinstance(layer, MicroPerforatedLayer)]
    assert layer.open_area is not None
    plate, = [r for r in rows if r["kind"] == layer.kind and sample.label in r["templates"]]
    assert any(f"開孔率 {rounded_number(layer.open_area * 100)}%" in size for size in plate["templates"][sample.label])
    board, = [r for r in rows if r["kind"] == "board"]
    # #1330 擬合 VPR 的鋼板是內建裡第一個「板」：板這一種從此有內建樣板，列得出它的名字。
    vpr, = [e for e in builtin_entries() if e.id in FITTED_VPR_IDS]
    assert board["template_note"] == "用這組物性的內建樣板："
    assert vpr.label in board["templates"]
    # 有樣板以後參數那一格寫這筆板層的值（從寫入端取：內建那筆的板層）。
    assert isinstance(vpr, FixedEntry) and isinstance(vpr.material, Assembly)
    plate_layer, = [layer for layer in vpr.material.layers if layer.kind == "board"]
    assert isinstance(plate_layer, BoardLayer)
    fields = cast(Fields, material_form_spec()["board"]["fields"])
    assert board["parameters"] == [_word(fields, "surface_mass_kg_m2", plate_layer.surface_mass_kg_m2),
                                   _word(fields, "resistance_pa_s_m", plate_layer.resistance_pa_s_m)]
    assert board["parameters"][0].startswith("面密度 ") and board["parameters"][1].startswith("面阻力 ")


def _sample(key: str, provenance: str, layers: list[dict[str, object]], label: str | None = None) -> FixedEntry:
    return FixedEntry.model_validate({"kind": "fixed", "id": key, "label": label or key, "provenance": provenance, "source": "自編",
                                      "material": {"kind": "assembly", "backing": "rigid", "layers": layers}})


def test_synthetic_templates_group_only_matching_identities_in_first_seen_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fabric: dict[str, object] = {"kind": "fabric", "surface_mass_kg_m2": .088, "flow_resistance_pa_s_m": 317.0}
    a: dict[str, object] = {"kind": "air", "thickness_m": .037}
    b: dict[str, object] = {"kind": "air", "thickness_m": .083}

    first = _sample("first", "model", [a, fabric, a, b])
    second = _sample("second", "measured", [fabric | {"surface_mass_kg_m2": .021}, a, fabric])
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


def test_flow_sources_list_their_own_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layer: dict[str, object] = {"kind": "porous", "thickness_m": .029, "flow_resistivity_pa_s_m2": 8600., "material_class": "fibrous"}
    sourced = layer | {"flow_resistivity_source": "density_table"}
    first = _sample("first", "model", [sourced, sourced], "棉樣板")
    second = _sample("second", "model", [layer], "板後棉樣板")
    monkeypatch.setattr("aosr.gui.material_elements.builtin_entries", lambda: (first, second))
    row, = [r for r in _catalog(tmp_path).json()["elements"] if r["kind"] == "porous"]
    assert row["flow_sources"] == [f"流阻：照密度查論文表，樣板：{first.label}", f"流阻：未註明，樣板：{second.label}"]


def test_komatsu_names_fitted_materials_only_for_unmatched_layers(tmp_path: Path) -> None:
    research = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
    description = type(research).model_fields["fitted_materials"].description
    assert description
    words = [description.split(f"{value} 是", 1)[1].split("，")[0].rstrip("）") for value in research.fitted_materials]
    note = f"原研究擬合的是{'與'.join(words)}，這一筆不在其中"
    carpet_label = next(e.label for e in builtin_entries() if e.id == "carpet_on_concrete")
    rows = _catalog(tmp_path).json()["elements"]
    carpet, = [r for r in rows if carpet_label in r["templates"] and r["kind"] == "porous"]
    assert carpet["ranges"][-1] == note
    assert all(note not in r["ranges"] for r in rows if r["families"])


def test_perforated_plates_differing_only_in_pitch_share_one_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plate: dict[str, object] = {"kind": "perforated", "thickness_m": .006, "hole_diameter_m": .003, "hole_pattern": "square"}
    entries = [_sample(key, "model", [plate | {"hole_spacing_m": pitch}]) for key, pitch in (("one", .015), ("two", .021))]
    monkeypatch.setattr("aosr.gui.material_elements.builtin_entries", lambda: entries)
    row, = [r for r in _catalog(tmp_path).json()["elements"] if r["kind"] == "perforated"]
    assert list(row["templates"]) == [e.label for e in entries]
    assert row["templates"]["one"] != row["templates"]["two"]


def test_family_label_keeps_parentheses_inside_the_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = builtin_entries()
    family = next(e for e in entries if e.kind == "optimizable")
    renamed = family.model_copy(update={"label": "自編（甲）棉（備註）"})
    monkeypatch.setattr("aosr.gui.material_elements.builtin_entries", lambda: tuple(
        renamed if e.id == family.id else e for e in entries))
    assert _catalog(tmp_path).json()["family_labels"][family.id] == "自編（甲）棉"
