"""面層格式：拒收歧義、錯誤量法與未開放算法，綱要和驗證器同守。"""
from __future__ import annotations

import math
from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.engine._zone_schema_reader import accepts

from aosr.materials.construction_models import Assembly, Layer, PorousLayer

FACING_CASES = {
    "fabric": {"kind": "fabric", "surface_mass_kg_m2": .25, "flow_resistance_pa_s_m": 236.},
    "micro_perforated": {"kind": "micro_perforated", "thickness_m": .001, "hole_diameter_m": .0005, "open_area": .02, "surface_mass_kg_m2": 1.2},
    "perforated": {"kind": "perforated", "thickness_m": .0125, "hole_diameter_m": .008, "open_area": .16, "surface_mass_kg_m2": 8.},
    "slotted": {"kind": "slotted", "thickness_m": .0125, "slot_width_m": .006, "open_area": .18, "surface_mass_kg_m2": 8.},
}


@pytest.mark.parametrize("kind", FACING_CASES)
def test_facings_are_layers_and_measured_base_accepts_them(kind: str) -> None:
    layer: Layer = TypeAdapter(Layer).validate_python(FACING_CASES[kind])
    assert layer.kind == kind
    schema = TypeAdapter(Layer).json_schema()
    assert accepts(schema, FACING_CASES[kind], schema)
    from aosr.materials.construction_models import MeasuredImpedance
    table = MeasuredImpedance.model_validate({"kind": "measured_impedance", "points": [
        {"frequency_hz": 100., "real_pa_s_m": 400., "imag_pa_s_m": -50.},
        {"frequency_hz": 500., "real_pa_s_m": 500., "imag_pa_s_m": 10.}],
        "source_and_conditions": "阻抗管樣品", "base_assembly": {"kind": "assembly", "layers": [FACING_CASES[kind]], "backing": "air"}})
    assert table.base_assembly.layers[0] == layer


@pytest.mark.parametrize("kind", FACING_CASES)
@pytest.mark.parametrize("bad", [0., -1., float("inf"), float("nan"), True, "1"])
def test_facing_numbers_are_positive_finite_and_strict(kind: str, bad: object) -> None:
    for field, value in FACING_CASES[kind].items():
        if isinstance(value, float):
            with pytest.raises(ValidationError):
                TypeAdapter(Layer).validate_python(FACING_CASES[kind] | {field: bad})


@pytest.mark.parametrize("kind", FACING_CASES)
@pytest.mark.parametrize("field,value", [("method", "rebillard"), ("method", "pieren_fixed"), ("method", "bessel_ingard"), ("method", "eq_fluid"), ("end_correction", "jb"), ("model", "eq_fluid")])
def test_facing_algorithms_cannot_be_selected(kind: str, field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Layer).validate_python(FACING_CASES[kind] | {field: value})


@pytest.mark.parametrize("kind", ["micro_perforated", "perforated", "slotted"])
@pytest.mark.parametrize("area", [0., 1., 1.01])
def test_open_area_is_strictly_between_zero_and_one(kind: str, area: float) -> None:
    data = FACING_CASES[kind] | {"open_area": area}
    with pytest.raises(ValidationError):
        TypeAdapter(Layer).validate_python(data)
    schema = TypeAdapter(Layer).json_schema()
    assert not accepts(schema, data, schema)


@pytest.mark.parametrize("pattern,constant", [("square", math.pi / 4), ("staggered_60", math.pi / (2 * math.sqrt(3))),
    ("staggered_45", math.pi / 2), ("rectangular", math.pi / 4)])
def test_pattern_open_area_uses_independent_geometry(pattern: str, constant: float) -> None:
    from aosr.materials.construction_models import layer_open_area
    data = {key: value for key, value in FACING_CASES["perforated"].items() if key != "open_area"}
    data |= {"hole_pattern": pattern, "hole_spacing_m": .02}
    if pattern == "rectangular":
        data["hole_spacing_2_m"] = .03
    layer: Layer = TypeAdapter(Layer).validate_python(data)
    assert hasattr(layer, "hole_diameter_m")
    from aosr.materials.construction_models import PerforatedLayer
    assert isinstance(layer, PerforatedLayer)
    assert layer.hole_spacing_m is not None and layer.hole_pattern is not None
    expected = constant * .008 ** 2 / (.02 * (.03 if pattern == "rectangular" else .02))
    assert layer_open_area(layer.hole_diameter_m, layer.hole_spacing_m, layer.hole_pattern, layer.hole_spacing_2_m) == pytest.approx(expected, rel=1e-15)
    schema = TypeAdapter(Layer).json_schema()
    assert accepts(schema, data, schema)


@pytest.mark.parametrize(("pattern", "field"), [("square", "hole_spacing_m"), ("staggered_60", "hole_spacing_m"),
    ("staggered_45", "hole_spacing_m"), ("rectangular", "hole_spacing_m"), ("rectangular", "hole_spacing_2_m")])
def test_holes_cannot_touch_but_arbitrarily_small_gap_is_accepted(pattern: str, field: str) -> None:
    data = {key: value for key, value in FACING_CASES["perforated"].items() if key != "open_area"}
    p = .008 * math.sqrt(2) if pattern == "staggered_45" else .008
    data |= {"hole_pattern": pattern, "hole_spacing_m": .02, "hole_spacing_2_m": .02} if pattern == "rectangular" else {"hole_pattern": pattern}
    data[field] = p
    with pytest.raises(ValidationError, match="孔.*重疊|孔距"):
        TypeAdapter(Layer).validate_python(data)
    TypeAdapter(Layer).validate_python(data | {field: p * (1 + 1e-10)})


@pytest.mark.parametrize("kind", ["micro_perforated", "perforated", "slotted"])
def test_geometry_routes_and_pairing_match_schema(kind: str) -> None:
    base = deepcopy(FACING_CASES[kind])
    spacing = {"slot_spacing_m": .04} if kind == "slotted" else {"hole_spacing_m": .02, "hole_pattern": "square"}
    no_area = {key: value for key, value in base.items() if key != "open_area"}
    bad_rows = [no_area, base | spacing, base | {"open_area": None}, base | {"surface_mass_kg_m2": None}]
    if kind != "slotted":
        bad_rows += [no_area | {"hole_spacing_m": .02}, no_area | {"hole_pattern": "square"},
            no_area | spacing | {"hole_spacing_2_m": .03}, no_area | spacing | {"hole_pattern": "rectangular"}]
    if kind != "micro_perforated":
        for field in ("backing_fabric_surface_mass_kg_m2", "backing_fabric_flow_resistance_pa_s_m"):
            bad_rows += [base | {field: .2}, base | {field: None}]
        for field in ("backing_fabric_surface_mass_kg_m2", "backing_fabric_flow_resistance_pa_s_m"):
            for bad in (0., -1., None):
                bad_rows.append(base | {"backing_fabric_surface_mass_kg_m2": .063,
                    "backing_fabric_flow_resistance_pa_s_m": 236., field: bad})
    for data in bad_rows:
        with pytest.raises(ValidationError):
            TypeAdapter(Layer).validate_python(data)
        schema = TypeAdapter(Layer).json_schema()
        assert not accepts(schema, data, schema)
    TypeAdapter(Layer).validate_python(no_area | spacing)


def test_diameter_bounds_overlap_without_inventing_other_limits() -> None:
    for kind in ("micro_perforated", "perforated"):
        for diameter in (.001, .002):
            TypeAdapter(Layer).validate_python(FACING_CASES[kind] | {"hole_diameter_m": diameter})
    TypeAdapter(Layer).validate_python(FACING_CASES["micro_perforated"] | {"hole_diameter_m": .00001})
    TypeAdapter(Layer).validate_python(FACING_CASES["perforated"] | {"hole_diameter_m": .2})
    for kind, diameter, message in (("micro_perforated", .00201, "孔徑較大請改用一般圓孔板"),
                                    ("perforated", .00099, "孔徑較小請改用微孔板")):
        with pytest.raises(ValidationError, match=message):
            TypeAdapter(Layer).validate_python(FACING_CASES[kind] | {"hole_diameter_m": diameter})


def test_fabric_has_no_thickness_and_mpp_has_no_backing_fabric() -> None:
    for kind, extra in (("fabric", {"thickness_m": .001}),
            ("micro_perforated", {"backing_fabric_surface_mass_kg_m2": .063, "backing_fabric_flow_resistance_pa_s_m": 236.})):
        with pytest.raises(ValidationError):
            TypeAdapter(Layer).validate_python(FACING_CASES[kind] | extra)


def test_unspecified_flow_source_does_not_change_old_serialization() -> None:
    data = {"kind": "porous", "thickness_m": .05, "flow_resistivity_pa_s_m2": 19100., "material_class": "fibrous"}
    old = PorousLayer.model_validate(data).model_dump(mode="json")
    assert old == data | {"model": "komatsu"}
    assert PorousLayer.model_validate(data | {"flow_resistivity_source": "unspecified"}).model_dump(mode="json") == old
    for source in ("measured", "density_table", "catalog"):
        assert PorousLayer.model_validate(data | {"flow_resistivity_source": source}).model_dump()["flow_resistivity_source"] == source
    assembly = Assembly.model_validate({"kind": "assembly", "layers": [data], "backing": "rigid"})
    assert "flow_resistivity_source" not in assembly.model_dump_json()


def test_mainline_scheme_round_trip_and_unspecified_scene_fingerprint() -> None:
    """答案檔由 721c009c 的獨立環境輸出；舊方案逐位保存，預設來源不改場景指紋。"""
    import json
    from pathlib import Path
    from aosr.reporting.scheme import Scheme
    from aosr.physics.report_io import load_input_document, scene_fingerprint
    from tests.engine._directivity import DIRECTIVITY
    from tests.engine._furniture_cases import CAPABILITIES, pair
    path = Path(__file__).resolve().parents[2] / "blueprint" / "construction_legacy_scheme.json"
    original = path.read_text()
    scheme = Scheme.model_validate_json(original)
    assert json.dumps(scheme.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n" == original
    baseline = scene_fingerprint(load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY))
    assert baseline == "6bef52b01c613234a228f1bbe65ef5d0bf42eff564900614b6122ce975b2dfc8"  # 721c009c 主線實跑
    data = json.loads(original)
    data["scene"]["surface_zones_by_wall"]["floor"][0]["material"]["layers"][0]["flow_resistivity_source"] = "unspecified"
    changed = Scheme.model_validate(data)
    assert changed.model_dump_json() == scheme.model_dump_json()
    assert scene_fingerprint(load_input_document(pair(changed), CAPABILITIES, DIRECTIVITY)) == baseline


def test_unknown_layers_are_rejected_by_both_core_branches() -> None:
    from typing import cast
    import jax.numpy as jnp
    from aosr.config.construction_environment import construction_environment
    from aosr.materials.construction import _context, _layer_matrix
    air, _ranges = construction_environment(density_kg_m3=1.2, sound_speed_m_s=343.)
    unknown = cast(Layer, object())
    assembly = Assembly.model_construct(layers=(unknown,), backing="rigid")
    with pytest.raises(ValueError, match="不認識"):
        _context(assembly)
    with pytest.raises(ValueError, match="不認識"):
        _layer_matrix(unknown, jnp.asarray([100.]), air, None)


@pytest.mark.parametrize("kind,required", [("fabric", ("surface_mass_kg_m2", "flow_resistance_pa_s_m")),
    ("micro_perforated", ("thickness_m", "hole_diameter_m")), ("perforated", ("thickness_m", "hole_diameter_m")),
    ("slotted", ("thickness_m", "slot_width_m"))])
def test_facings_require_physical_inputs_and_stay_immutable(kind: str, required: tuple[str, ...]) -> None:
    for name in required:
        data = {field: value for field, value in FACING_CASES[kind].items() if field != name}
        with pytest.raises(ValidationError):
            TypeAdapter(Layer).validate_python(data)
    layer: Layer = TypeAdapter(Layer).validate_python(FACING_CASES[kind])
    with pytest.raises(ValidationError):
        setattr(layer, required[0], .1)


def test_slotted_spacing_cannot_touch_and_uses_independent_ratio() -> None:
    from aosr.materials.construction_models import layer_open_area
    data = {key: value for key, value in FACING_CASES["slotted"].items() if key != "open_area"}
    with pytest.raises(ValidationError, match="縫寬必須小於縫距"):
        TypeAdapter(Layer).validate_python(data | {"slot_spacing_m": .006})
    TypeAdapter(Layer).validate_python(data | {"slot_spacing_m": .006000001})
    assert layer_open_area(.006, .04, "slotted") == .15
