"""#1084 分區＋實測表的雙精度子行程；不在父行程改 JAX 全域設定（底構造求值只收雙精度，#309）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aosr.runtime import configure_calculation_jax
configure_calculation_jax()

import numpy as np

from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import Assembly, MeasuredImpedance
from aosr.physics import geometric_lane, late_energy, report_io, report_output, wall_surfaces
from aosr.physics.report_measured import measured_seams
from aosr.physics.report_surfaces import RectangleZone, WholeWallZone, ZonesByWall
from aosr.physics.report_zone_seams import zone_measured_seams
from aosr.physics.room_paths import image_source_paths
from aosr.physics.three_lane_report import solve_three_lane_report
from aosr.physics.wall_reflections import zoned_reflection_sources
from aosr.reporting.scheme import Scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._precision_contracts import contract_value
from tests.engine._zoned_report_cases import small_scheme, split_x0, with_x0
from tests.engine.test_measured_impedance import measured

TABLE = MeasuredImpedance.model_validate(measured())
DENSITY, SOUND_SPEED = 1.2, 343.0


def _constant_zones() -> ZonesByWall:
    zones = small_scheme(zoned=True).scene.surface_zones_by_wall
    assert zones is not None
    return zones


def full_contract() -> None:
    """抓逐區另寫一套算法：一塊蓋滿整面的矩形區，接縫跟整面那一份逐位相同。"""
    full: ZonesByWall = {"x0": (RectangleZone(area="rectangle", id="all", u_m=(0.0, 2.0), v_m=(0.0, 1.5), material=TABLE),)}
    whole = measured_seams({"x0": (WholeWallZone(area="whole_wall", material=TABLE),)},
                           density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
    zoned = zone_measured_seams(full, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
    assert whole is not None and zoned is not None
    assert zoned == {"x0": {"all": whole["x0"]}}


def only_contract() -> None:
    """抓整面實測牆也進逐區表（會重複報），或常數分區也冒出空表。"""
    constant = _constant_zones()
    assert zone_measured_seams(constant, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED) is None
    whole: ZonesByWall = {"x0": (WholeWallZone(area="whole_wall", material=TABLE),)}
    assert zone_measured_seams(whole, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED) is None
    mixed = dict(constant) | {"x0": split_x0(TABLE, constant["x0"][0].material)}
    seams = zone_measured_seams(mixed, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
    assert seams is not None and set(seams) == {"x0"} and set(seams["x0"]) == {"a"}
    both = zone_measured_seams(dict(constant) | {"x0": split_x0(TABLE, TABLE)}, density_kg_m3=DENSITY,
                               sound_speed_m_s=SOUND_SPEED)
    assert both is not None and set(both["x0"]) == {"a", "b"} and both["x0"]["a"] == both["x0"]["b"]


def _measured_inputs() -> tuple[report_io.ReportInput, report_io.ReportInput]:
    """x0 切兩區都貼同一張實測表；對照組是整面同一張表。"""
    split = report_io.load_input_document(pair(with_x0(split_x0(TABLE, TABLE))), CAPABILITIES, DIRECTIVITY)
    whole = report_io.load_input_document(pair(with_x0((WholeWallZone(area="whole_wall", material=TABLE),))),
                                          CAPABILITIES, DIRECTIVITY)
    return split, whole


def image_contract() -> None:
    """抓分區實測表在鏡像法只吃首區或走錯一套內插；沿用 #1018 反射乘積絕對尺，鏡像路能量逐位相同。"""
    split, whole = _measured_inputs()
    axis = (63.0, 125.0, 503.0, 1000.0, 4000.0)
    sources = {name: zoned_reflection_sources(inputs.room_m, inputs.surface_zones_by_wall or {}, axis,
        density_kg_m3=inputs.density_kg_m3, sound_speed_m_s=inputs.sound_speed_m_s)
        for name, inputs in (("split", split), ("whole", whole))}

    def products(name: str) -> list[complex]:
        return [value for path in image_source_paths(room=split.room_m, source=split.source_m, receiver=split.receiver_m,
                c=split.sound_speed_m_s, max_order=3, reflection_sources=sources[name]) for value in path.reflection_product]

    reference_products, split_products = products("whole"), products("split")
    assert reference_products and len(reference_products) == len(split_products)
    difference = max(abs(a-b) for a, b in zip(reference_products, split_products, strict=True))
    assert difference <= contract_value("reflection_product_ulp"), difference

    def early(name: str) -> geometric_lane.GeometricEarlyResult:
        return geometric_lane.solve_geometric_early_lane(room=split.room_m, source=split.source_m,
            receiver=split.receiver_m, source_model=report_io.solver_inputs(whole).source_model,
            sound_speed_m_s=split.sound_speed_m_s, rho_c_pa_s_per_m=split.density_kg_m3*split.sound_speed_m_s,
            frequencies_hz=axis, impedance_by_wall=dict.fromkeys(Wall.wall_names(), 1646.4), reflection_order_k=3,
            reflection_sources=sources[name])

    reference, observed = early("whole"), early("split")
    np.testing.assert_array_equal(observed.reflected_energy, reference.reflected_energy)
    assert observed.scattering == reference.scattering


def late_contract() -> None:
    """抓分區實測表在晚期漏區或面積權重錯；對照是獨立算的整面實測表，沿用 #1021 同材尺。"""
    split, whole = _measured_inputs()
    axis = (63.0, 500.0, 1000.0, 4000.0)
    zoned = wall_surfaces.zoned_wall_surface_response(split.room_m, split.surface_zones_by_wall or {}, axis,
        density_kg_m3=split.density_kg_m3, sound_speed_m_s=split.sound_speed_m_s)
    single = wall_surfaces.zoned_wall_surface_response(whole.room_m, whole.surface_zones_by_wall or {}, axis,
        density_kg_m3=whole.density_kg_m3, sound_speed_m_s=whole.sound_speed_m_s)
    rho_c = split.density_kg_m3*split.sound_speed_m_s
    actual = late_energy.solve_late_energy(late_energy.LateEnergyInputs(split.room_m, rho_c, axis,
        zoned.named_rows(axis), 6, 0.9, absorption_by_zone=zoned.named_zone_rows(axis)))
    reference = late_energy.solve_late_energy(late_energy.LateEnergyInputs(whole.room_m, rho_c, axis,
        single.named_rows(axis), 6, 0.9))
    old = np.array([b.late_reverberant_energy for b in reference.bands])
    new = np.array([b.late_reverberant_energy for b in actual.bands])
    np.testing.assert_allclose(new, old, rtol=contract_value("late_energy_physical_property"), atol=0.0)


def routing_contract() -> None:
    """首區、第二區或其他整面貼實測表：有矩形的牆只進逐區接縫，整面一區的牆只進報表的整面接縫。"""
    base = report_io.load_input_document(pair(small_scheme()), CAPABILITIES, DIRECTIVITY)
    solved = report_io.solver_inputs(base)
    report = solve_three_lane_report(source_model=solved.source_model, room=solved.room,
        source=solved.source, receiver=solved.receiver, sound_speed_m_s=solved.sound_speed_m_s,
        density_kg_m3=solved.density_kg_m3, impedance_by_wall=solved.impedance_by_wall,
        materials_by_wall=solved.materials_by_wall, scattering_by_wall=solved.scattering_by_wall,
        reflection_order_k=solved.reflection_order_k, low_frequency_axis=solved.low_frequency_axis)
    zoned = report_io.load_input_document(pair(small_scheme(zoned=True)), CAPABILITIES, DIRECTIVITY)
    for wall, index in (("x0", 0), ("x0", 1), ("floor", 0)):
        zones = zoned.surface_zones_by_wall
        assert zones is not None
        updated = {name: [zone.model_dump(mode="json") for zone in cells] for name, cells in zones.items()}
        updated[wall][index]["material"] = measured()
        inputs = report_io.load_input_document(zoned.model_dump(mode="json") | {"surface_zones_by_wall": updated},
                                               CAPABILITIES, DIRECTIVITY)
        by_wall = report_output.output_from_report(report, inputs=inputs, with_points=False).measured_seams_by_wall
        by_zone = zone_measured_seams(inputs.surface_zones_by_wall, density_kg_m3=inputs.density_kg_m3,
                                      sound_speed_m_s=inputs.sound_speed_m_s)
        if wall == "x0":
            assert by_wall is None, (wall, index, by_wall)
            assert by_zone is not None and set(by_zone) == {"x0"} and set(by_zone["x0"]) == {"ab"[index]}, by_zone
        else:
            assert by_zone is None, by_zone
            assert by_wall is not None and set(by_wall) == {"floor"}, by_wall


def context_contract() -> None:
    """底構造在表兩端求不出值時，錯誤要說出是哪一區，後面接整面那一份的原文（牆名、表、底構造）。"""
    bad_base = Assembly.model_validate({"kind": "assembly", "layers": [{"kind": "porous", "thickness_m": .05,
        "flow_resistivity_pa_s_m2": 100000., "material_class": "fibrous"}], "backing": "rigid"})
    bad = TABLE.model_copy(update={"base_assembly": bad_base})
    constant = _constant_zones()
    try:
        zone_measured_seams(dict(constant) | {"x0": split_x0(constant["x0"][0].material, bad)},
                            density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
    except ValueError as exc:
        assert str(exc).startswith("分區 b：x0：實測阻抗表 2 列") and "100000.0" in str(exc), str(exc)
    else:
        raise AssertionError("底構造超出模型範圍卻沒拒收")


def seams_for_contract() -> None:
    """端到端的答案：用方案自己的空氣獨立算整面那一份接縫，寫成 JSON 檔（逐區那一份要逐位相同）。"""
    scene = Scheme.model_validate_json(Path(sys.argv[2]).read_bytes()).scene
    whole = measured_seams({"x0": (WholeWallZone(area="whole_wall", material=TABLE),)},
                           density_kg_m3=scene.density_kg_m3, sound_speed_m_s=scene.sound_speed_m_s)
    assert whole is not None
    Path(sys.argv[3]).write_text(json.dumps([seam.model_dump(mode="json") for seam in whole["x0"]], allow_nan=False))


if __name__ == "__main__":
    {"full": full_contract, "only": only_contract, "image": image_contract, "late": late_contract,
     "routing": routing_contract, "context": context_contract, "seams_for": seams_for_contract}[sys.argv[1]]()
