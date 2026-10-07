"""家具資料的答案取自家具決策紙 16–18 條與 #559 三份提案，不用實作算答案。"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from aosr.config.furniture_materials import (
    FurnitureKindCode, FurnitureMaterial, FurnitureMaterials, load_furniture_materials,
)
from aosr.config.physics_constants import load_physics_constants
from aosr.geometry.furniture import FurnitureKind
from aosr.materials.catalog_absorption import CatalogAbsorption, impedance_on_axis
from aosr.materials.furniture_materials import furniture_impedance_on_axis


DATA = Path(__file__).resolve().parents[2] / "src/aosr/config/data"
KNOWN = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
ALL_BANDS = (63.0, *KNOWN, 8000.0)


# 附註字串照決策紙第 16–18 條與施工單第 1 項原句，不從設定檔抄。
@pytest.mark.parametrize("name,frequencies,answer,note", [
    ("fabric", KNOWN, (0.30, 0.41, 0.51, 0.59, 0.68, 0.69), "可能偏高，尤其 2000、4000 Hz"),
    ("leather", KNOWN, (0.25, 0.38, 0.40, 0.32, 0.22, 0.14), "參考的是合成皮"),
    ("wood", KNOWN, (0.04, 0.04, 0.05, 0.06, 0.06, 0.06), "借用木地板的值"),
    ("glass", KNOWN[1:], (0.06, 0.04, 0.03, 0.02, 0.02), "估計，非本件實測"),
    ("absorptive_cloud", KNOWN, (0.13, 0.39, 0.70, 0.85, 0.88, 0.88), "玻璃棉或岩棉類約 40 mm"),
])
def test_defaults_equal_decision_answers(name: str, frequencies: tuple[float, ...],
                                        answer: tuple[float, ...], note: str) -> None:
    material = getattr(load_furniture_materials(DATA / "furniture_materials.toml"), name)
    assert material.default.band_center_hz == frequencies
    assert material.default.absorption == answer
    assert material.band_center_hz == ALL_BANDS
    assert material.unknown_bands_hz == ((63.0, 125.0, 8000.0) if name == "glass" else (63.0, 8000.0))
    assert "估計，非本件實測" in material.label
    assert note in material.label
    assert set(material.applicable_kinds) == SPEC_ALLOWED_KINDS[name]
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


def test_wood_lower_bound_is_default_by_main_agent_decision() -> None:
    wood = load_furniture_materials(DATA / "furniture_materials.toml").wood
    assert "不另給下界，下界取預設值" in wood.provenance.conditions
    for kind in ("coffee_table", "desk", "ceiling_cloud"):
        assert wood.for_kind(kind).lower == wood.default


@pytest.mark.parametrize("kind,choices", [
    ("sofa", {"fabric", "leather"}), ("chair", {"fabric", "leather"}),
    ("coffee_table", {"wood", "glass"}), ("desk", {"wood", "glass"}),
    ("ceiling_cloud", {"wood", "absorptive_cloud"}),
])
def test_available_materials_match_furniture_kinds(kind: str, choices: set[str]) -> None:
    data = load_furniture_materials(DATA / "furniture_materials.toml")
    assert {name for name, material in data.materials() if kind in material.applicable_kinds} == choices
    assert set(get_args(FurnitureKindCode)) == {item.value for item in FurnitureKind}


# 每種壞形狀：(被擋的理由片段, 改哪一層的哪幾格)；理由要對，免得因為別的錯碰巧被擋。
# 表格寫不下的四種（刪整格出處、出處空白、沙發沒得選、多掛一種家具）寫在考卷本體。
INVALID_SHAPES: dict[str, tuple[str, dict[str, dict[str, object]]]] = {
    "bands": ("嚴格遞增", {"default": {"band_center_hz": list(reversed(KNOWN))}}),
    "repeated_band": ("嚴格遞增", {"material": {"unknown_bands_hz": [63.0, 63.0, 8000.0]}}),
    "nonpositive": ("greater than 0", {"default": {"absorption": [0.0] * 6}}),
    "nonfinite": ("finite number", {"default": {"absorption": [float("nan")] * 6}}),
    "boolean": ("valid number", {"default": {"absorption": [True] * 6}}),
    "length": ("長度必須一致", {"default": {"absorption": [0.3]}}),
    "extra": ("Extra inputs", {"default": {"extra": 0.3}}),
    "bounds": ("下界不得大於上界", {"lower": {"absorption": [0.9] * 6}}),
    "default_above_upper": ("下界 ≤ 預設 ≤ 上界", {"upper": {"absorption": [0.28, 0.47, 0.64, 0.70, 0.76, 0.88]}}),
    "default_below_lower": ("下界 ≤ 預設 ≤ 上界", {"lower": {"absorption": [0.32, 0.28, 0.33, 0.32, 0.22, 0.14]}}),
    "missing_bound": ("涵蓋所有有預設值的頻帶", {
        "lower": {"band_center_hz": list(KNOWN[:-1]), "absorption": [0.25, 0.28, 0.33, 0.32, 0.22]},
        "upper": {"band_center_hz": list(KNOWN[:-1]), "absorption": [0.33, 0.47, 0.64, 0.70, 0.76]}}),
    "foreign_bound_band": ("全部頻帶清單", {
        "lower": {"band_center_hz": [31.5, *KNOWN], "absorption": [0.25, 0.25, 0.28, 0.33, 0.32, 0.22, 0.14]},
        "upper": {"band_center_hz": [31.5, *KNOWN], "absorption": [0.33, 0.33, 0.47, 0.64, 0.70, 0.76, 0.88]}}),
    "bound_bands_differ": ("頻帶清單必須相同", {
        "lower": {"band_center_hz": [63.0, *KNOWN], "absorption": [0.60, 0.25, 0.28, 0.33, 0.32, 0.22, 0.14]}}),
    "all_bands_order": ("嚴格遞增", {"material": {"band_center_hz": list(reversed(ALL_BANDS))}}),
    "overlap": ("不重疊", {"material": {"unknown_bands_hz": [63.0, 125.0, 8000.0]}}),
    "partition": ("合起來等於全部頻帶", {"material": {"unknown_bands_hz": [63.0]}}),
    "interior_unknown": ("兩端", {
        "default": {"band_center_hz": [125.0, 250.0, 1000.0, 2000.0, 4000.0],
                    "absorption": [0.30, 0.41, 0.59, 0.68, 0.69]},
        "material": {"unknown_bands_hz": [63.0, 500.0, 8000.0]}}),
    "kind": ("Input should be", {"material": {"applicable_kinds": ["cupboard"]}}),
    "duplicate_kind": ("不可重複", {"material": {"applicable_kinds": ["sofa", "sofa", "chair"]}}),
    "bounds_kind": ("逐種提供上下界", {"bounds": {"kind": "desk"}}),
    "foreign_kind": ("適用家具種類錯誤", {"material": {"applicable_kinds": ["sofa", "chair", "desk"]}}),
    "provenance": ("Field required", {}),
    "blank_source": ("at least 1 character", {}),
    "unavailable": ("沒有材質可選", {}),
}


@pytest.mark.parametrize("bad", sorted(INVALID_SHAPES))
def test_rejects_each_invalid_material_shape(bad: str) -> None:
    reason, edits = INVALID_SHAPES[bad]
    with (DATA / "furniture_materials.toml").open("rb") as file:
        data = tomllib.load(file)
    material = dict(data["fabric"])
    first = dict(material["bounds"][0])
    first.update(lower=dict(first["lower"], **edits.get("lower", {})),
                 upper=dict(first["upper"], **edits.get("upper", {})), **edits.get("bounds", {}))
    material.update(default=dict(material["default"], **edits.get("default", {})),
                    bounds=[first, *material["bounds"][1:]], **edits.get("material", {}))
    if bad == "provenance":
        del material["provenance"]
    elif bad == "blank_source":
        material["provenance"] = dict(material["provenance"], source="  ")
    elif bad == "unavailable":
        material.update(applicable_kinds=["sofa"], bounds=material["bounds"][:1])
        data["leather"] = dict(data["leather"], applicable_kinds=["sofa"], bounds=data["leather"]["bounds"][:1])
    elif bad == "foreign_kind":
        material["bounds"] = [*material["bounds"], dict(first, kind="desk")]
    data["fabric"] = material
    with pytest.raises(ValidationError, match=reason):
        FurnitureMaterials.model_validate(data)


@pytest.mark.parametrize("name,band", [("glass", 125.0), ("absorptive_cloud", 63.0), ("absorptive_cloud", 8000.0)])
def test_bound_order_is_checked_where_default_is_unknown(name: str, band: float) -> None:
    with (DATA / "furniture_materials.toml").open("rb") as file:
        data = tomllib.load(file)
    bounds = data[name]["bounds"][0]
    index = bounds["lower"]["band_center_hz"].index(band)
    bounds["lower"]["absorption"][index] = bounds["upper"]["absorption"][index] + 0.01
    with pytest.raises(ValidationError, match="下界不得大於上界"):
        FurnitureMaterials.model_validate(data)


# 可選的家具種類照決策紙第 16、17 條與施工單第 1 項，不從程式的允許表抄。
SPEC_ALLOWED_KINDS = {
    "fabric": {"sofa", "chair"}, "leather": {"sofa", "chair"},
    "wood": {"coffee_table", "desk", "ceiling_cloud"}, "glass": {"coffee_table", "desk"},
    "absorptive_cloud": {"ceiling_cloud"},
}


@pytest.mark.parametrize("name,foreign", [
    (name, kind) for name, allowed in SPEC_ALLOWED_KINDS.items()
    for kind in sorted({item.value for item in FurnitureKind} - allowed)])
def test_each_material_rejects_foreign_furniture_kind(name: str, foreign: str) -> None:
    with (DATA / "furniture_materials.toml").open("rb") as file:
        data = tomllib.load(file)
    material = data[name]
    material["applicable_kinds"].append(foreign)
    material["bounds"].append(dict(material["bounds"][0], kind=foreign))
    with pytest.raises(ValidationError, match="適用家具種類錯誤"):
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


# 八度帶照 frequency_axis 的半開區間 [中心÷√2, 中心×√2)：125÷√2≈88.388、250÷√2≈176.777、
# 4000×√2≈5656.854 Hz；兩側各取一個貼邊點，再取剛好落在 125÷√2 與 4000×√2 上的點。
EDGE_AXIS = (63.0, 88.38, 125.0 / math.sqrt(2.0), 88.39, 125.0, 176.77, 176.78, 250.0, 4000.0,
             5656.8, 4000.0 * math.sqrt(2.0), 5656.9, 8000.0)
SIX_BAND_EXTRAPOLATED = (True, True, True, True, False, False, False, False, False, True, True, True, True)
SIX_BAND_UNKNOWN = (True, True, False, False, False, False, False, False, False, False, True, True, True)


@pytest.mark.parametrize("name,extrapolated,unknown", [
    ("fabric", SIX_BAND_EXTRAPOLATED, SIX_BAND_UNKNOWN),
    ("leather", SIX_BAND_EXTRAPOLATED, SIX_BAND_UNKNOWN),
    ("wood", SIX_BAND_EXTRAPOLATED, SIX_BAND_UNKNOWN),
    ("absorptive_cloud", SIX_BAND_EXTRAPOLATED, SIX_BAND_UNKNOWN),
    ("glass", (True, True, True, True, True, True, True, False, False, True, True, True, True),
     (True, True, True, True, True, True, False, False, False, False, True, True, True)),
])
def test_unknown_flags_cover_only_unknown_octave_bands(name: str, extrapolated: tuple[bool, ...],
                                                     unknown: tuple[bool, ...]) -> None:
    material = getattr(load_furniture_materials(DATA / "furniture_materials.toml"), name)
    result = furniture_impedance_on_axis(material, EDGE_AXIS, load_physics_constants(DATA / "physics_constants.toml").rho_c)
    assert result.extrapolated == extrapolated
    assert result.unknown_extrapolated == unknown


def test_low_end_without_unknown_band_is_extrapolated_but_not_unknown() -> None:
    with (DATA / "furniture_materials.toml").open("rb") as file:
        wood = tomllib.load(file)["wood"]
    wood.update(band_center_hz=[*KNOWN, 8000.0], unknown_bands_hz=[8000.0])
    material = FurnitureMaterial.model_validate(wood)
    result = furniture_impedance_on_axis(material, (63.0, 90.0, 125.0, 5700.0),
                                        load_physics_constants(DATA / "physics_constants.toml").rho_c)
    assert result.extrapolated == (True, True, False, True)
    assert result.unknown_extrapolated == (False, False, False, True)


def test_high_end_without_unknown_band_is_extrapolated_but_not_unknown() -> None:
    with (DATA / "furniture_materials.toml").open("rb") as file:
        wood = tomllib.load(file)["wood"]
    wood.update(band_center_hz=[63.0, *KNOWN], unknown_bands_hz=[63.0])
    material = FurnitureMaterial.model_validate(wood)
    result = furniture_impedance_on_axis(material, (63.0, 4000.0, 5700.0, 12000.0),
                                        load_physics_constants(DATA / "physics_constants.toml").rho_c)
    assert result.extrapolated == (True, False, True, True)
    assert result.unknown_extrapolated == (True, False, False, False)


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


def test_scheme_material_choices_join_physics_but_stay_outside_modal_closure() -> None:
    from aosr.reporting.physics_identity import physics_import_closure
    from aosr.reporting.modal_diagnosis import modal_import_closure

    modules = {"aosr.config.furniture_materials", "aosr.config.representative_speakers",
               "aosr.config._furniture_records", "aosr.materials.furniture_materials"}
    assert modules.intersection(physics_import_closure().modules) == {
        "aosr.config.furniture_materials", "aosr.config._furniture_records"}
    assert modules.isdisjoint(modal_import_closure().modules)
