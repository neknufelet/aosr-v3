"""家具資料的答案取自家具決策紙 16–18 條與 #559 三份提案，不用實作算答案。"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from aosr.config.furniture_materials import (
    FurnitureKindCode, FurnitureMaterials, load_furniture_materials,
)
from aosr.config.physics_constants import load_physics_constants
from aosr.geometry.furniture import FurnitureKind
from aosr.materials.catalog_absorption import CatalogAbsorption, impedance_on_axis
from aosr.materials.furniture_materials import furniture_impedance_on_axis


DATA = Path(__file__).resolve().parents[2] / "src/aosr/config/data"
KNOWN = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
ALL_BANDS = (63.0, *KNOWN, 8000.0)


@pytest.mark.parametrize("name,frequencies,answer", [
    ("fabric", KNOWN, (0.30, 0.41, 0.51, 0.59, 0.68, 0.69)),
    ("leather", KNOWN, (0.25, 0.38, 0.40, 0.32, 0.22, 0.14)),
    ("wood", KNOWN, (0.04, 0.04, 0.05, 0.06, 0.06, 0.06)),
    ("glass", KNOWN[1:], (0.06, 0.04, 0.03, 0.02, 0.02)),
    ("absorptive_cloud", KNOWN, (0.13, 0.39, 0.70, 0.85, 0.88, 0.88)),
])
def test_defaults_equal_decision_answers(name: str, frequencies: tuple[float, ...],
                                        answer: tuple[float, ...]) -> None:
    material = getattr(load_furniture_materials(DATA / "furniture_materials.toml"), name)
    assert material.default.band_center_hz == frequencies
    assert material.default.absorption == answer
    assert material.band_center_hz == ALL_BANDS
    assert material.unknown_bands_hz == ((63.0, 125.0, 8000.0) if name == "glass" else (63.0, 8000.0))
    assert "估計，非本件實測" in material.label
    assert material.provenance.status == "estimated"
    assert material.provenance.source.strip()
    assert material.provenance.conditions.strip()


@pytest.mark.parametrize("name,kind,bands,lower,upper", [
    ("fabric", "sofa", KNOWN, (0.25, 0.28, 0.33, 0.32, 0.22, 0.14),
     (0.33, 0.47, 0.64, 0.70, 0.76, 0.88)),
    ("fabric", "chair", KNOWN, (0.28, 0.28, 0.33, 0.37, 0.47, 0.60),
     (0.63, 0.79, 0.75, 0.70, 0.76, 0.88)),
    ("leather", "sofa", KNOWN, (0.22, 0.31, 0.36, 0.30, 0.19, 0.10),
     (0.31, 0.48, 0.50, 0.40, 0.48, 0.49)),
    ("leather", "chair", KNOWN, (0.22, 0.31, 0.36, 0.30, 0.19, 0.10),
     (0.31, 0.48, 0.50, 0.40, 0.48, 0.49)),
    ("wood", "coffee_table", KNOWN, (0.04, 0.04, 0.05, 0.06, 0.06, 0.06),
     (0.18, 0.12, 0.10, 0.09, 0.08, 0.07)),
    ("wood", "desk", KNOWN, (0.04, 0.04, 0.05, 0.06, 0.06, 0.06),
     (0.18, 0.12, 0.10, 0.09, 0.08, 0.07)),
    ("wood", "ceiling_cloud", KNOWN, (0.04, 0.04, 0.05, 0.06, 0.06, 0.06),
     (0.18, 0.12, 0.10, 0.09, 0.08, 0.07)),
    ("glass", "coffee_table", KNOWN, (0.08, 0.04, 0.03, 0.03, 0.02, 0.02),
     (0.18, 0.06, 0.04, 0.03, 0.02, 0.02)),
    ("glass", "desk", KNOWN, (0.08, 0.04, 0.03, 0.03, 0.02, 0.02),
     (0.18, 0.06, 0.04, 0.03, 0.02, 0.02)),
    ("absorptive_cloud", "ceiling_cloud", ALL_BANDS,
     (0.01, 0.06, 0.32, 0.46, 0.67, 0.75, 0.75, 0.68),
     (0.17, 0.17, 0.46, 0.74, 1.01, 1.01, 1.01, 0.9512)),
])
def test_bounds_equal_proposal_answers(name: str, kind: str, bands: tuple[float, ...],
                                      lower: tuple[float, ...], upper: tuple[float, ...]) -> None:
    material = getattr(load_furniture_materials(DATA / "furniture_materials.toml"), name)
    bounds = material.for_kind(kind)
    assert bounds.lower.band_center_hz == bounds.upper.band_center_hz == bands
    assert bounds.lower.absorption == lower
    assert bounds.upper.absorption == upper


@pytest.mark.parametrize("kind,choices", [
    ("sofa", {"fabric", "leather"}), ("chair", {"fabric", "leather"}),
    ("coffee_table", {"wood", "glass"}), ("desk", {"wood", "glass"}),
    ("ceiling_cloud", {"wood", "absorptive_cloud"}),
])
def test_available_materials_match_furniture_kinds(kind: str, choices: set[str]) -> None:
    data = load_furniture_materials(DATA / "furniture_materials.toml")
    assert {name for name, material in data.materials() if kind in material.applicable_kinds} == choices
    assert set(get_args(FurnitureKindCode)) == {item.value for item in FurnitureKind}


REJECTION_REASONS = {
    "bands": "嚴格遞增", "nonpositive": "greater than 0", "nonfinite": "finite number",
    "boolean": "valid number", "length": "長度必須一致", "bounds": "下界不得大於上界",
    "default_above_upper": "下界 ≤ 預設 ≤ 上界", "default_below_lower": "下界 ≤ 預設 ≤ 上界",
    "overlap": "不重疊", "partition": "合起來等於全部頻帶", "interior_unknown": "兩端",
    "missing_bound": "涵蓋所有有預設值的頻帶", "provenance": "Field required",
    "blank_source": "at least 1 character", "kind": "Input should be", "extra": "Extra inputs",
    "unavailable": "沒有材質可選", "bounds_kind": "逐種提供上下界", "duplicate_kind": "不可重複",
    "foreign_bound_band": "全部頻帶清單",
}


@pytest.mark.parametrize("bad", sorted(REJECTION_REASONS))
def test_rejects_each_invalid_material_shape(bad: str) -> None:
    with (DATA / "furniture_materials.toml").open("rb") as file:
        data = tomllib.load(file)
    material = dict(data["fabric"])
    default = dict(material["default"])
    bounds = [dict(item) for item in material["bounds"]]
    lower, upper = dict(bounds[0]["lower"]), dict(bounds[0]["upper"])
    if bad == "bands":
        default["band_center_hz"] = list(reversed(KNOWN))
    elif bad in {"nonpositive", "nonfinite", "boolean", "length"}:
        default["absorption"] = {"nonpositive": [0.0] * 6, "nonfinite": [float("nan")] * 6,
                                 "boolean": [True] * 6, "length": [0.3]}[bad]
    elif bad == "bounds":
        lower["absorption"] = [0.9] * 6
    elif bad == "default_above_upper":
        upper["absorption"] = [0.28, 0.47, 0.64, 0.70, 0.76, 0.88]
    elif bad == "default_below_lower":
        lower["absorption"] = [0.32, 0.28, 0.33, 0.32, 0.22, 0.14]
    elif bad == "overlap":
        material["unknown_bands_hz"] = [63.0, 125.0, 8000.0]
    elif bad == "partition":
        material["unknown_bands_hz"] = [63.0]
    elif bad == "interior_unknown":
        default["band_center_hz"] = [125.0, 250.0, 1000.0, 2000.0, 4000.0]
        default["absorption"] = [0.30, 0.41, 0.59, 0.68, 0.69]
        material["unknown_bands_hz"] = [63.0, 500.0, 8000.0]
    elif bad == "missing_bound":
        lower["band_center_hz"], lower["absorption"] = [63.0], [0.25]
    elif bad == "provenance":
        del material["provenance"]
    elif bad == "blank_source":
        material["provenance"] = dict(material["provenance"], source="  ")
    elif bad == "kind":
        material["applicable_kinds"] = ["cupboard"]
    elif bad == "extra":
        default["extra"] = 0.3
    elif bad == "unavailable":
        material["applicable_kinds"], bounds = ["sofa"], bounds[:1]
        leather = dict(data["leather"])
        leather["applicable_kinds"], leather["bounds"] = ["sofa"], leather["bounds"][:1]
        data["leather"] = leather
    elif bad == "bounds_kind":
        bounds[0]["kind"] = "desk"
    elif bad == "duplicate_kind":
        material["applicable_kinds"] = ["sofa", "sofa", "chair"]
    elif bad == "foreign_bound_band":
        lower["band_center_hz"] = [31.5, *KNOWN[1:]]
    bounds[0].update(lower=lower, upper=upper)
    material.update(default=default, bounds=bounds)
    data["fabric"] = material
    with pytest.raises(ValidationError, match=REJECTION_REASONS[bad]):
        FurnitureMaterials.model_validate(data)


@pytest.mark.parametrize("field", ["status", "source_kind", "source", "conditions"])
def test_each_provenance_field_is_required(field: str) -> None:
    with (DATA / "furniture_materials.toml").open("rb") as file:
        data = tomllib.load(file)
    del data["fabric"]["provenance"][field]
    with pytest.raises(ValidationError):
        FurnitureMaterials.model_validate(data)


@pytest.mark.parametrize("field", ["status", "source_kind"])
def test_provenance_status_and_source_kind_are_controlled(field: str) -> None:
    with (DATA / "furniture_materials.toml").open("rb") as file:
        data = tomllib.load(file)
    data["fabric"]["provenance"][field] = "pretend_measured"
    with pytest.raises(ValidationError):
        FurnitureMaterials.model_validate(data)


def test_materials_are_frozen_and_loader_requires_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "materials.toml"
    path.write_bytes((DATA / "furniture_materials.toml").read_bytes())
    loaded = load_furniture_materials(path)
    with pytest.raises(ValidationError, match="frozen"):
        loaded.fabric.label = "改掉"
    with pytest.raises(ValidationError, match="frozen"):
        loaded.fabric.default.absorption = (0.1,)
    with pytest.raises(TypeError):
        load_furniture_materials(**{})
    with pytest.raises(ValueError, match="適用"):
        loaded.fabric.for_kind("desk")


@pytest.mark.parametrize("name,kind", [("fabric", "sofa"), ("fabric", "chair"), ("leather", "chair"),
                                      ("wood", "desk"), ("glass", "coffee_table"),
                                      ("absorptive_cloud", "ceiling_cloud")])
@pytest.mark.parametrize("curve", ["default", "lower", "upper"])
def test_pointwise_impedance_equals_catalog_conversion(name: str, kind: str, curve: str) -> None:
    material = getattr(load_furniture_materials(DATA / "furniture_materials.toml"), name)
    series = material.default if curve == "default" else getattr(material.for_kind(kind), curve)
    axis = (31.5, 63.0, 90.0, 125.0, 180.0, 250.0, 707.0, 1000.0, 4000.0, 5700.0, 8000.0, 12000.0)
    rho_c = load_physics_constants(DATA / "physics_constants.toml").rho_c
    expected = impedance_on_axis(CatalogAbsorption(name, series.band_center_hz, series.absorption), axis, rho_c)
    result = furniture_impedance_on_axis(material, axis, rho_c, curve=curve, kind=kind)
    for field in ("frequencies_hz", "catalog_absorption", "absorption", "impedance_pa_s_per_m",
                  "extrapolated", "clamped"):
        assert getattr(result, field) == getattr(expected, field)
    assert result.unknown_label == "未知（計算時用相鄰頻帶延伸代算）"


@pytest.mark.parametrize("name,expected", [
    ("fabric", (True, True, False, False, False, False, True, True)),
    ("leather", (True, True, False, False, False, False, True, True)),
    ("wood", (True, True, False, False, False, False, True, True)),
    ("absorptive_cloud", (True, True, False, False, False, False, True, True)),
    ("glass", (True, True, True, True, False, False, True, True)),
])
def test_unknown_extension_flags_cover_low_and_high_points(name: str, expected: tuple[bool, ...]) -> None:
    material = getattr(load_furniture_materials(DATA / "furniture_materials.toml"), name)
    axis = (63.0, 90.0, 125.0, 200.0, 250.0, 4000.0, 5700.0, 8000.0)
    result = furniture_impedance_on_axis(material, axis, load_physics_constants(DATA / "physics_constants.toml").rho_c)
    assert result.extrapolated == expected
    assert result.unknown_extrapolated == expected


def test_unknown_default_remains_unknown_when_glass_bound_has_value() -> None:
    material = load_furniture_materials(DATA / "furniture_materials.toml").glass
    result = furniture_impedance_on_axis(material, (125.0, 250.0),
                                        load_physics_constants(DATA / "physics_constants.toml").rho_c,
                                        kind="desk", curve="lower")
    assert result.catalog_absorption == (0.08, 0.04)
    assert result.extrapolated == (False, False)
    assert result.unknown_extrapolated == (True, False)


def test_cloud_upper_clamps_after_interpolation() -> None:
    material = load_furniture_materials(DATA / "furniture_materials.toml").absorptive_cloud
    result = furniture_impedance_on_axis(material, (500.0, 707.1067811865476, 1000.0, 4000.0),
                                        load_physics_constants(DATA / "physics_constants.toml").rho_c,
                                        kind="ceiling_cloud", curve="upper")
    assert result.catalog_absorption[1] == pytest.approx(0.875)
    assert result.catalog_absorption[2:] == (1.01, 1.01)
    assert result.clamped == (False, False, True, True)


def test_impedance_scales_with_callers_physical_condition() -> None:
    material = load_furniture_materials(DATA / "furniture_materials.toml").wood
    rho_c = load_physics_constants(DATA / "physics_constants.toml").rho_c
    first = furniture_impedance_on_axis(material, (63.0, 180.0, 8000.0), rho_c)
    second = furniture_impedance_on_axis(material, (63.0, 180.0, 8000.0), rho_c * 2.0)
    assert second.impedance_pa_s_per_m == tuple(value * 2.0 for value in first.impedance_pa_s_per_m)
    assert second.extrapolated == first.extrapolated
    assert second.clamped == first.clamped
    assert second.unknown_extrapolated == first.unknown_extrapolated


def test_bounds_require_applicable_kind_and_valid_curve() -> None:
    material = load_furniture_materials(DATA / "furniture_materials.toml").fabric
    rho_c = load_physics_constants(DATA / "physics_constants.toml").rho_c
    with pytest.raises(ValueError, match="種類"):
        furniture_impedance_on_axis(material, (125.0,), rho_c, curve="lower")
    with pytest.raises(ValueError, match="適用"):
        furniture_impedance_on_axis(material, (125.0,), rho_c, kind="desk")
    with pytest.raises(ValueError, match="曲線"):
        furniture_impedance_on_axis(material, (125.0,), rho_c, curve="other")


def test_new_data_modules_are_outside_physics_and_modal_import_closures() -> None:
    from aosr.reporting.physics_identity import physics_import_closure
    from aosr.reporting.modal_diagnosis import modal_import_closure

    modules = {"aosr.config.furniture_materials", "aosr.config.representative_speakers",
               "aosr.config._furniture_records", "aosr.materials.furniture_materials"}
    assert modules.isdisjoint(physics_import_closure().modules)
    assert modules.isdisjoint(modal_import_closure().modules)
