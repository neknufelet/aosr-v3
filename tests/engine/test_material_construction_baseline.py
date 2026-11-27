"""#1243：e0cda0de 上錄製的舊三種條目基準；數字與材料均為考卷自編。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter
from starlette.applications import Starlette
from starlette.testclient import TestClient

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient.surrogate_coverage import family_coverage
from aosr.gui.material_editor import material_routes, user_library_path
from aosr.gui.material_surfaces import placement
from aosr.gui.zone_modes import Origin, zone_material
from aosr.materials import library, search_library
from aosr.materials.construction_models import Assembly, ConstantImpedance, MountedOnWall

from tests.engine._construction_preset_answers import FITTED_VPR_IDS

ANSWER = Path(__file__).parent / "data" / "material_construction_baseline.json"


def legacy_entries() -> tuple[library.LibraryEntry, ...]:
    """剛背、空氣背、固定阻抗、兩種實測表、兩種家族與可調 VPR。"""
    common: dict[str, object] = {"provenance": "model", "source": "考卷自編，非實物"}
    wool = {"kind": "porous", "thickness_m": 0.04, "flow_resistivity_pa_s_m2": 17000.0, "material_class": "fibrous"}
    docs: list[dict[str, object]] = []
    for backing in ("rigid", "air"):
        docs.append(common | {"kind": "fixed", "id": f"pin_{backing}", "label": f"自編層 {backing}",
                              "material": {"kind": "assembly", "layers": [wool, {"kind": "air", "thickness_m": 0.06}],
                                           "backing": backing}})
    docs.append(common | {"kind": "fixed", "id": "pin_impedance", "label": "自編阻抗",
                          "material": {"kind": "constant_impedance", "impedance_pa_s_m": 4321.0}})
    for imaginary in (False, True):
        points = [{"frequency_hz": f, "real_pa_s_m": r, **({"imag_pa_s_m": -125.0} if imaginary else {})}
                  for f, r in ((110.0, 830.0), (440.0, 610.0))]
        docs.append(common | {"kind": "fixed", "id": f"pin_table_{str(imaginary).lower()}", "label": f"自編表 {imaginary}",
                              "material": {"kind": "measured_impedance", "points": points, "source_and_conditions": "自編",
                                           "base_assembly": {"kind": "assembly", "layers": [wool], "backing": "rigid"}}})
    for cavity in (False, True):
        docs.append(common | {"kind": "optimizable", "id": f"pin_family_{str(cavity).lower()}", "label": f"自編家族 {cavity}",
                              "porous": wool, "porous_thickness": {"adjustable": True, "range_m": [0.02, 0.08],
                              "choices_m": [0.02, 0.04, 0.08], "initial_m": 0.04},
                              "cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.03} if cavity else None})
    docs.append(common | {"kind": "tunable_layers", "id": "pin_vpr", "label": "自編可調板", "structure": "vpr",
                          "layers": [{"kind": "board", "surface_mass_kg_m2": 9.0, "resistance_pa_s_m": 0.0},
                                     wool | {"frame_compression_modulus_pa": 51000.0, "frame_loss_factor": 0.12}],
                          "plate_density_kg_m3": 7500.0,
                          "variables": [{"layer": 0, "field": "surface_mass_kg_m2", "range": [6.0, 12.0], "initial": 9.0}]})
    adapter: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
    return tuple(adapter.validate_python(doc) for doc in docs)


def artifacts(directory: Path) -> dict[str, object]:
    entries = legacy_entries()
    user_library_path(directory).write_text(library.dump_user_library(entries), encoding="utf-8")
    presets = load_construction_presets(config_path("construction_presets.toml"))
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    joined = search_library.with_user_materials(settings, presets, entries)
    fixed = tuple(entry for entry in entries if isinstance(entry, library.FixedEntry))
    targets = [f"fixed:{entry.id}" for entry in fixed]
    base = ConstantImpedance(impedance_pa_s_m=7654.0)
    with TestClient(Starlette(routes=material_routes(directory))) as client:
        listing = client.get("/api/materials").json()
    return {
        "snapshot_family_sha256": hashlib.sha256(search_library.snapshot_bytes(joined.entries)).hexdigest(),
        "snapshot_fixed_sha256": hashlib.sha256(search_library.snapshot_bytes(joined.entries, fixed)).hexdigest(),
        "coverage": [row.model_dump(mode="json") for row in family_coverage(joined.settings, joined.presets)],
        "fingerprint": joined.settings.fingerprint,
        "candidates": [row.model_dump(mode="json") for row in search_library.fixed_candidates(targets, presets, fixed)],
        "listing": listing,
        "zones": {entry.id: {"material": change.material.model_dump(mode="json"), "note": change.note,
                               "same_as_original": change.same_as_original}
                  for entry in fixed for change in [zone_material(base, Origin(base, "current"), entry)]},
        "placement": {entry.id: placement(entry).to_json() for entry in entries},
        "basis": {entry.id: library.impedance_basis(entry) for entry in entries},
    }


def assert_old_fields(actual: object, expected: object) -> None:
    """API 可加鍵；原有的每一格仍逐值相同。"""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual
            assert_old_fields(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_old_fields(left, right)
    else:
        assert actual == expected


def test_legacy_materials_match_pre_construction_main(tmp_path: Path) -> None:
    expected = json.loads(ANSWER.read_text(encoding="utf-8"))
    actual = artifacts(tmp_path)
    # #1323 只在內建清單裡多了文獻硬面那幾行（門、窗、木地板…）；拿掉它們，其餘每一行仍逐值相同、順序不變。
    listing = actual.pop("listing")
    catalog = {entry.id for entry in library.catalog_entries()}
    assert isinstance(listing, dict) and isinstance(listing["materials"], list)
    listing["materials"] = [row for row in listing["materials"] if row["entry"]["id"] not in catalog | set(FITTED_VPR_IDS)]
    assert_old_fields(listing, expected.pop("listing"))
    assert actual == expected


def test_fitted_vpr_library_rows_apply_as_fixed_materials(tmp_path: Path) -> None:
    with TestClient(Starlette(routes=material_routes(tmp_path))) as client:
        response = client.get("/api/materials")
    assert response.status_code == 200
    rows = {row["entry"]["id"]: row["entry"] for row in response.json()["materials"]}
    wall = ConstantImpedance(impedance_pa_s_m=7654.)
    for key in FITTED_VPR_IDS:
        entry = library.FixedEntry.model_validate(rows[key])
        assert isinstance(entry.material, Assembly)
        applied = zone_material(wall, Origin(wall, "current"), entry)
        assert isinstance(applied.material, MountedOnWall)
        assert applied.material.layers == entry.material.layers and applied.material.wall == wall
