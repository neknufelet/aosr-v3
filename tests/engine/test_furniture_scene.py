"""家具共用零件只考公式接線；不宣稱真實家具或向下指向性已獨立驗證。"""
from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.furniture_materials import FurnitureMaterial, FurnitureMaterialName, load_furniture_materials
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.geometry.furniture import FurnitureBox, FurnitureKind, contact_margin_m
from aosr.geometry.shoebox import Point
from aosr.materials.furniture_materials import FurnitureImpedanceOnAxis, furniture_impedance_on_axis
from aosr.physics import furniture_scene as core
from aosr.physics.furniture_paths import furniture_path_amplitude
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.source_directivity import SourceModel, two_parameter_pressure_factor, v2_compat_pressure_factor
from tests.engine._precision_contracts import contract_value
from tests.engine.test_furniture_paths import PRESSURE_REFERENCE_RHO_C, TABLE_EXAMPLE_C, table_path


def item(identifier: str = "table", *, kind: FurnitureKind = FurnitureKind.DESK,
         material: FurnitureMaterialName = "wood", z: float = 0.5, yaw: float = 90.0) -> AbsoluteFurniture:
    return AbsoluteFurniture(furniture_id=identifier, kind=kind, material=material,
        width_m=0.8, depth_m=1.6, height_m=0.1, bottom_center_m=(2.0, 3.0, z), yaw_deg=yaw)


def test_absolute_furniture_becomes_sorted_boxes_and_contact_margin() -> None:
    room = (5.3, 3.7, 2.9)
    relative = contract_value("furniture_geometry_contact")
    records = (item("z"), item("a", kind=FurnitureKind.CHAIR, material="fabric", z=0.0, yaw=0.0))
    got, margin = core.furniture_scene(records, room, contact_rel=relative)
    assert margin == contact_margin_m(room, contact_rel=relative)
    assert tuple(furniture.furniture_id for furniture in got) == ("a", "z")
    expected = (
        FurnitureBox(kind="chair", width_m=0.8, depth_m=1.6, height_m=0.1,
                     bottom_center_m=(2.0, 3.0, 0.0), yaw_deg=0.0, margin_m=margin),
        FurnitureBox(kind="desk", width_m=0.8, depth_m=1.6, height_m=0.1,
                     bottom_center_m=(2.0, 3.0, 0.5), yaw_deg=90.0, margin_m=margin),
    )
    assert tuple(furniture.box for furniture in got) == expected
    assert core.furniture_scene(tuple(reversed(records)), room, contact_rel=relative) == (got, margin)


@pytest.mark.parametrize(("record", "message"), [
    (item(z=0.0), "懸空家具的底面必須高於地板並超過接觸界線"),
    (item(kind=FurnitureKind.SOFA, material="fabric", z=0.5), "貼地家具的底面必須在地板接觸界線內"),
    (item().model_copy(update={"width_m": 1e-320}), "家具尺寸在此座標無法表示有限的非退化盒子"),
])
def test_bad_box_errors_propagate_verbatim(record: AbsoluteFurniture, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        core.furniture_scene((record,), (5.3, 3.7, 2.9), contact_rel=contract_value("furniture_geometry_contact"))
    assert str(caught.value) == message


def test_boxes_are_built_with_the_scene_contact_margin() -> None:
    # 貼地沙發底面離地半份界線：用場景界線建盒子要收下；界線若沒傳進去（當 0）就會被拒。
    room = (5.3, 3.7, 2.9)
    relative = contract_value("furniture_geometry_contact")
    margin = contact_margin_m(room, contact_rel=relative)
    sofa = item("sofa", kind=FurnitureKind.SOFA, material="fabric", z=margin / 2.0, yaw=0.0)
    got, _ = core.furniture_scene((sofa,), room, contact_rel=relative)
    assert got[0].box.bottom_center_m[2] == margin / 2.0


def test_empty_scene_still_uses_the_given_contact_contract() -> None:
    room = (5.3, 3.7, 2.9)
    relative = contract_value("furniture_geometry_contact")
    assert core.furniture_scene((), room, contact_rel=relative) == ((), contact_margin_m(room, contact_rel=relative))
    assert core.furniture_impedances((), (125.0,), PRESSURE_REFERENCE_RHO_C) == {}


@pytest.mark.parametrize("operation", ["scene", "impedance"])
def test_duplicate_furniture_ids_are_rejected(operation: str) -> None:
    records = (item("same"), item("same", material="glass"))
    with pytest.raises(ValueError, match="furniture_id 不可重複：same"):
        if operation == "scene":
            core.furniture_scene(records, (5.3, 3.7, 2.9), contact_rel=contract_value("furniture_geometry_contact"))
        else:
            core.furniture_impedances(records, (125.0,), PRESSURE_REFERENCE_RHO_C)


@pytest.mark.parametrize(("material", "kind"), [
    ("fabric", FurnitureKind.SOFA), ("leather", FurnitureKind.CHAIR), ("wood", FurnitureKind.CEILING_CLOUD),
    ("glass", FurnitureKind.DESK), ("absorptive_cloud", FurnitureKind.CEILING_CLOUD),
])
@pytest.mark.parametrize("frequencies", [GEOMETRIC_LANE_FREQUENCIES_HZ, (63.0, 100.0, 300.0, 4000.0, 8000.0)])
def test_material_registry_ids_impedance_and_unknown_flags_match_direct_calculation(
    material: FurnitureMaterialName, kind: FurnitureKind, frequencies: tuple[float, ...],
) -> None:
    registry = dict(load_furniture_materials(config_path("furniture_materials.toml")).materials())
    record = item("named-" + material, kind=kind, material=material, z=0.0 if kind.grounded else 0.5)
    got = core.furniture_impedances((record,), frequencies, PRESSURE_REFERENCE_RHO_C)
    assert tuple(got) == (record.furniture_id,)
    expected = furniture_impedance_on_axis(registry[material], frequencies, PRESSURE_REFERENCE_RHO_C,
                                           curve="default", kind=kind.value)
    assert got[record.furniture_id] == expected
    assert got[record.furniture_id].unknown_extrapolated == expected.unknown_extrapolated
    assert all(math.isfinite(z) and z > 0.0 for z in got[record.furniture_id].impedance_pa_s_per_m)


def test_impedance_scales_bitwise_with_supplied_medium_and_cache_does_not_cross_calls() -> None:
    records = (item("wood"), item("glass", material="glass"))
    frequencies = (63.0, 125.0, 300.0, 4000.0, 8000.0)
    first = core.furniture_impedances(records, frequencies, PRESSURE_REFERENCE_RHO_C)
    doubled = core.furniture_impedances(records, frequencies, 2.0 * PRESSURE_REFERENCE_RHO_C)
    for record in records:
        before, after = first[record.furniture_id], doubled[record.furniture_id]
        assert tuple(z.hex() for z in after.impedance_pa_s_per_m) == tuple(
            (2.0 * z).hex() for z in before.impedance_pa_s_per_m)
        assert after.unknown_extrapolated == before.unknown_extrapolated


def test_same_material_is_computed_once_in_id_order_with_its_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """監看真實反推；快取按材質共用，但每件種類仍須有效。"""
    records = (item("z-cloud", kind=FurnitureKind.CEILING_CLOUD), item("a-desk"))
    axis = (63.0, 300.0, 8000.0)
    calls: list[tuple[FurnitureMaterial, tuple[float, ...], float, str, str | None]] = []
    def observe(material: FurnitureMaterial, frequencies: Sequence[float], medium: float, *,
                curve: str = "default", kind: str | None = None) -> FurnitureImpedanceOnAxis:
        calls.append((material, tuple(frequencies), medium, curve, kind))
        return furniture_impedance_on_axis(material, frequencies, medium, curve=curve, kind=kind)
    monkeypatch.setattr(core, "furniture_impedance_on_axis", observe)
    got = core.furniture_impedances(records, axis, PRESSURE_REFERENCE_RHO_C)
    wood = load_furniture_materials(config_path("furniture_materials.toml")).wood
    assert calls == [(wood, axis, PRESSURE_REFERENCE_RHO_C, "default", "desk")]
    assert tuple(got) == ("a-desk", "z-cloud")
    assert got["a-desk"] is got["z-cloud"]
    assert got["a-desk"] == furniture_impedance_on_axis(wood, axis, PRESSURE_REFERENCE_RHO_C, kind="desk")


def test_table_pressure_omnidirectional_is_bitwise_the_existing_amplitude() -> None:
    path = table_path()
    frequencies = (125.0, 500.0, 1000.0)
    impedances = tuple(ratio * PRESSURE_REFERENCE_RHO_C for ratio in (2.0, 4.0, 6.0))
    _, expected = furniture_path_amplitude(path, frequencies, impedances,
                                          rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    got = core.furniture_pressure_with_directivity(path, frequencies, impedances,
        rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C, model=SourceModel.OMNIDIRECTIONAL)
    assert tuple((z.real.hex(), z.imag.hex()) for z in got) == tuple(
        (z.real.hex(), z.imag.hex()) for z in expected)


@pytest.mark.parametrize("model", [SourceModel.TWO_PARAMETER, SourceModel.V2_COMPAT])
def test_table_pressure_multiplies_hand_downward_factor(model: SourceModel) -> None:
    """第五支 cosθ=.6 算例；方向 (.8,0,-.6) 對主位軸 (1,0,0)，手算 x=.2。"""
    path = table_path()
    frequencies = (125.0, 500.0, 1000.0)
    impedances = tuple(ratio * PRESSURE_REFERENCE_RHO_C for ratio in (2.0, 4.0, 6.0))
    params = load_directivity_defaults(config_path("directivity_defaults.toml"))
    _, pressure = furniture_path_amplitude(path, frequencies, impedances,
                                         rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    x = 1.0 - 0.8  # 向量手算，不取產品的 departure_direction、speaker_axis 或 one_minus_cos 當答案。
    factors = (two_parameter_pressure_factor(x, frequencies, params) if model == SourceModel.TWO_PARAMETER
               else v2_compat_pressure_factor(x, frequencies, 0.21, 0.08, TABLE_EXAMPLE_C))
    expected = tuple(p * float(d) for p, d in zip(pressure, factors, strict=True))
    got = core.furniture_pressure_with_directivity(path, frequencies, impedances,
        rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C, model=model, params=params,
        source=Point(-0.6, 0.0, 1.05), aim=Point(0.6, 0.0, 1.05), baffle_width_m=0.21, piston_radius_m=0.08)
    assert got == pytest.approx(expected, rel=1e-14, abs=0.0)
    assert path.cos_theta == pytest.approx(0.6)


def test_furniture_scene_stays_outside_physics_and_modal_closures() -> None:
    from aosr.reporting.modal_diagnosis import modal_identity, modal_import_closure
    from aosr.reporting.physics_identity import physics_import_closure

    modules = {"aosr.physics.furniture_scene"}
    assert modules.isdisjoint(physics_import_closure().modules)
    assert modules.isdisjoint(modal_import_closure().modules)
    assert modal_identity() == "modal-v1:5dcb7e682c04552785515d72567623f0bd8c42ae82a742ac4b7f5be015955e9e"
