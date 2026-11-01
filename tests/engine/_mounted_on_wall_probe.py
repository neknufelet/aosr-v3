"""貼在原牆上的構造（#990）的隔離算數子行程；雙精度由正式開機入口開，父考卷只讀它寫的結果。"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

from aosr.runtime import configure_calculation_jax

configure_calculation_jax()

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from aosr.config.construction_environment import construction_environment
from aosr.config.construction_presets import load_construction_presets
from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.geometry.shoebox import Wall
from aosr.materials import construction
from aosr.materials import transfer_matrix as tmm
from aosr.materials.construction_diff import ConstructionTemplate, build_template, surface_impedance_from_parameters
from aosr.materials.construction_models import (
    AirLayer, Assembly, ConstantImpedance, MeasuredImpedance, MeasuredImpedancePoint, MountedOnWall, PorousLayer,
    SurfaceMaterial,
)
from aosr.physics import report_io, three_lane_report as report
from aosr.physics.fem_reduced_jax import rigid_mask_from_templates
from tests.engine._source_model_control import fake_fem_energy
from tests.engine._wall_reflections_probe import _input

# 20 Hz～2 kHz 對數取樣：實測表那一型同時有表內、表外兩段；角度用無規入射那一格（Paris）。
FREQUENCIES = np.geomspace(20., 2000., 25)
GRADIENT_FREQUENCIES = (80., 1000.)
OBLIQUE_DEG = 60.
STEP = 1e-5  # 中央差分相對步長，同 _construction_diff_probe。
# 剛性極限：原牆取三個相隔一千倍的有限大阻抗（Pa·s/m），父考卷看差距照同一個倍數縮小；不用無窮大。
RIGID_LOADS = (1e6, 1e9, 1e12)
# 換軸用的另一條軸：跟 FREQUENCIES 同長，夾進實測表原牆的三個表列頻點（50／200／500 Hz），表內外分界跟原軸不同。
OTHER_AXIS = tuple(sorted((*np.geomspace(21., 1900., 22).tolist(), 50., 200., 500.)))
# 控制組：改動前主線上量這兩筆預設的三個入口，父考卷對釘住的位元。
CONTROL_PRESETS = ("glass_wool_48k_50mm_gap50", "perforated_0p8mm_d10_gw24k_50mm")
CONTROL_FREQUENCIES = (31.5, 125., 1000.)
CONTROL_ANGLES = (0., 60.)

Parameters = dict[str, jax.Array]


def _bits(value: jax.Array) -> dict[str, object]:
    array = np.asarray(value)
    return {"real": [float(x).hex() for x in array.real.reshape(-1)],
            "imag": [float(x).hex() for x in array.imag.reshape(-1)],
            "dtype": str(value.dtype), "shape": list(array.shape)}


def _preset(name: str) -> SurfaceMaterial:
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    return next(adapter.validate_python(preset.material)
                for preset in load_construction_presets(config_path("construction_presets.toml")) if preset.id == name)


def _walls() -> dict[str, ConstantImpedance | MeasuredImpedance | Assembly]:
    """三種原牆：固定阻抗（30ρc 那一格的值）、實測表（50–500 Hz，表外底構造）、多層構造（剛背棉板）。"""
    base = Assembly(layers=(PorousLayer(thickness_m=.1, flow_resistivity_pa_s_m2=10000., material_class="fibrous"),),
                    backing="rigid")
    measured = MeasuredImpedance(points=(
        MeasuredImpedancePoint(frequency_hz=50., real_pa_s_m=3000., imag_pa_s_m=-2000.),
        MeasuredImpedancePoint(frequency_hz=200., real_pa_s_m=2500., imag_pa_s_m=-500.),
        MeasuredImpedancePoint(frequency_hz=500., real_pa_s_m=2000., imag_pa_s_m=-100.)),
        base_assembly=base, source_and_conditions="考卷假表：阻抗管、垂直入射、樣品 100 mm")
    wall = Assembly(layers=(PorousLayer(thickness_m=.03, flow_resistivity_pa_s_m2=15000., material_class="fibrous"),
                            AirLayer(thickness_m=.2)), backing="rigid")
    return {"constant": ConstantImpedance(impedance_pa_s_m=12348.), "measured": measured, "assembly": wall}


def _covering() -> tuple[PorousLayer | AirLayer, ...]:
    """新增層：48K 50 mm 棉板加 50 mm 空腔那一筆預設的兩層。"""
    material = _preset("glass_wool_48k_50mm_gap50")
    assert isinstance(material, Assembly)
    return cast(tuple[PorousLayer | AirLayer, ...], material.layers)


def _mounted() -> dict[str, MountedOnWall]:
    return {name: MountedOnWall(layers=_covering(), wall=wall) for name, wall in _walls().items()}


def control(air: PhysicsConstants) -> dict[str, object]:
    """舊多層構造（剛背）的三個入口；父考卷對改動前主線釘住的值逐位比。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    frequencies, angles = jnp.asarray(CONTROL_FREQUENCIES), jnp.asarray(CONTROL_ANGLES)
    rows: dict[str, object] = {}
    for name in CONTROL_PRESETS:
        material = _preset(name)
        rows[name] = {
            "normal": _bits(construction.surface_impedance(material, frequencies, air=air, ranges=ranges)[0]),
            "angles": _bits(construction.surface_impedance_at_angles(material, frequencies, angles, air=air, ranges=ranges)[0]),
            "diffuse": [float(x).hex() for x in np.asarray(
                construction.random_incidence_absorption(material, frequencies, air=air, ranges=ranges)[0])],
        }
    return rows


def zero(air: PhysicsConstants) -> dict[str, object]:
    """新增層厚度全設 0（梯度那邊的參數層）：表面阻抗對原牆自己的阻抗，垂直與 Paris 角度格。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    frequencies, angles = jnp.asarray(FREQUENCIES), tmm.diffuse_incidence_angles_deg(method="paris")
    rows: dict[str, object] = {}
    for name, material in _mounted().items():
        template = build_template(material, frequencies, air=air, ranges=ranges)
        thin = {key: (jnp.asarray(0.) if key.endswith(".thickness_m") else value)
                for key, value in template.baseline_parameters().items()}
        rows[name] = {
            "zeroed": sorted(key for key in thin if key.endswith(".thickness_m")),
            "normal": _bits(surface_impedance_from_parameters(template, thin, frequencies, air=air)),
            "normal_wall": _bits(construction.surface_impedance(material.wall, frequencies, air=air, ranges=ranges)[0]),
            "angles": _bits(surface_impedance_from_parameters(template, thin, frequencies, air=air, incidence_angles_deg=angles)),
            "angles_wall": _bits(construction.surface_impedance_at_angles(material.wall, frequencies, angles, air=air, ranges=ranges)[0]),
        }
    return rows


def rigid(air: PhysicsConstants) -> dict[str, object]:
    """原牆取越來越大的有限固定阻抗：跟同一組層剛背的多層構造的最大相對差，垂直與 Paris 角度格。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    frequencies, angles = jnp.asarray(FREQUENCIES), tmm.diffuse_incidence_angles_deg(method="paris")
    layers = _covering()
    backed = Assembly(layers=layers, backing="rigid")
    rows: dict[str, object] = {}

    def normal(material: SurfaceMaterial) -> jax.Array:
        return construction.surface_impedance(material, frequencies, air=air, ranges=ranges)[0]

    def oblique(material: SurfaceMaterial) -> jax.Array:
        return construction.surface_impedance_at_angles(material, frequencies, angles, air=air, ranges=ranges)[0]
    for label, evaluate in (("normal", normal), ("angles", oblique)):
        reference = np.asarray(evaluate(backed))
        rows[label] = [float(np.max(np.abs(np.asarray(evaluate(MountedOnWall(
            layers=layers, wall=ConstantImpedance(impedance_pa_s_m=load)))) - reference) / np.abs(reference)))
            for load in RIGID_LOADS]
    rows["loads"] = list(RIGID_LOADS)
    return rows


def _paris(impedance: np.ndarray, angles_deg: np.ndarray, rho_c: float) -> list[float]:
    """另寫一份 numpy 的 Paris 無規吸音（α=1−|r|²、r=(Z cosθ−ρc)/(Z cosθ+ρc)、sinθ cosθ 加權梯形平均），不經產品積分。"""
    theta = np.radians(angles_deg)
    z = impedance * np.cos(theta)
    alpha = 1 - np.abs((z - rho_c) / (z + rho_c)) ** 2
    weights = np.sin(theta) * np.cos(theta)
    def trapezoid(values: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(np.diff(theta) * (values[..., 1:] + values[..., :-1]) / 2, axis=-1))
    return [float(value) for value in trapezoid(alpha * weights) / trapezoid(weights)]


def paths(air: PhysicsConstants) -> dict[str, object]:
    """基準參數上可微入口對非可微入口；三種原牆、垂直與 Paris 角度格，另帶無規吸音的值與兩個對照、註記。

    對照一：同一份角度阻抗另用 numpy 積 Paris；對照二（原牆是多層構造）：新增層＋原牆各層直接串成一個剛背構造。
    """
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    frequencies, angles = jnp.asarray(FREQUENCIES), tmm.diffuse_incidence_angles_deg(method="paris")
    rows: dict[str, object] = {}
    for name, material in _mounted().items():
        template = build_template(material, frequencies, air=air, ranges=ranges)
        base = template.baseline_parameters()
        alpha, notes = construction.random_incidence_absorption(material, frequencies, air=air, ranges=ranges)
        rows[name] = {
            "variables": list(template.names),
            "normal": _bits(surface_impedance_from_parameters(template, base, frequencies, air=air)),
            "normal_existing": _bits(construction.surface_impedance(material, frequencies, air=air, ranges=ranges)[0]),
            "angles": _bits(surface_impedance_from_parameters(template, base, frequencies, air=air, incidence_angles_deg=angles)),
            "angles_existing": _bits(construction.surface_impedance_at_angles(material, frequencies, angles, air=air, ranges=ranges)[0]),
            "rigid_normal": _bits(construction.surface_impedance(Assembly(layers=material.layers, backing="rigid"),
                                                                 frequencies, air=air, ranges=ranges)[0]),
            "diffuse": [float(value) for value in np.asarray(alpha)],
            "diffuse_reference": _paris(np.asarray(construction.surface_impedance_at_angles(
                material, frequencies, angles, air=air, ranges=ranges)[0]), np.asarray(angles), air.rho_c),
            "diffuse_concatenated": None if not isinstance(material.wall, Assembly) else [
                float(value) for value in np.asarray(construction.random_incidence_absorption(
                    Assembly(layers=(*material.layers, *material.wall.layers), backing=material.wall.backing),
                    frequencies, air=air, ranges=ranges)[0])],
            "notes": [note.message for note in notes],
            "template_notes": [note.message for note in template.notes],
        }
    return rows


def _loss(template: ConstructionTemplate, air: PhysicsConstants, frequency: float, oblique: bool) -> Callable[[Parameters], jax.Array]:
    """實數純量：阻抗實虛部（除 ρc）加吸音；斜入射那支用 60 度。同 _construction_diff_probe._loss。"""
    def loss(parameters: Parameters) -> jax.Array:
        angles = jnp.asarray([OBLIQUE_DEG]) if oblique else None
        z = surface_impedance_from_parameters(template, parameters, jnp.asarray([frequency]), air=air,
                                              incidence_angles_deg=angles)
        alpha = tmm.reflection_and_absorption_coefficient(z, air=air, incidence_angles_deg=angles)[1]
        return jnp.sum((jnp.real(z) + 0.3 * jnp.imag(z)) / air.rho_c + 0.7 * alpha)
    return loss


def gradients(air: PhysicsConstants) -> list[dict[str, object]]:
    """三種原牆的每個變數：jit(grad) 對中央差分，比無量綱 p·dL/dp；樣板建在那一個頻點的軸上。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    rows: list[dict[str, object]] = []
    for name, material in _mounted().items():
        for frequency in GRADIENT_FREQUENCIES:
            template = build_template(material, jnp.asarray([frequency]), air=air, ranges=ranges)
            base = template.baseline_parameters()
            for oblique in (False, True):
                loss = jax.jit(_loss(template, air, frequency, oblique))
                automatic = jax.jit(jax.grad(_loss(template, air, frequency, oblique)))(base)
                for variable in template.names:
                    value = base[variable]
                    delta = value * STEP
                    central = (loss({**base, variable: value + delta}) - loss({**base, variable: value - delta})) / (2 * delta)
                    rows.append({"wall": name, "variable": variable, "frequency": frequency, "oblique": oblique,
                                 "automatic_scaled": float(automatic[variable] * value),
                                 "central_scaled": float(central * value)})
    return rows


def _error(action: Callable[[], object]) -> str:
    try:
        action()
    except ValueError as exc:
        return str(exc)
    return ""


def rejections(air: PhysicsConstants) -> dict[str, object]:
    """樣板不是剛性面；原牆的拒收訊息帶「原牆」前綴。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    frequencies = jnp.asarray(FREQUENCIES)
    template = build_template(_mounted()["measured"], frequencies, air=air, ranges=ranges)
    bad_wall = MountedOnWall(layers=_covering(), wall=Assembly(layers=(PorousLayer(
        thickness_m=.05, flow_resistivity_pa_s_m2=100., material_class="fibrous"),), backing="rigid"))
    mask = rigid_mask_from_templates([template, None, None, None, None, None])
    return {
        "bad_wall": _error(lambda: construction.surface_impedance(bad_wall, frequencies, air=air, ranges=ranges)),
        "rigid_mask": [bool(value) for value in np.asarray(mask)],
    }


def _at_axis(template: ConstructionTemplate, parameters: Parameters, air: PhysicsConstants,
             angles: jax.Array | None = None) -> Callable[[jax.Array], jax.Array]:
    """頻率軸當 jit 的輸入（追蹤值），其他固定。"""
    def evaluate(axis: jax.Array) -> jax.Array:
        return surface_impedance_from_parameters(template, parameters, axis, air=air, incidence_angles_deg=angles)
    return evaluate


def live_wall(air: PhysicsConstants) -> dict[str, object]:
    """原牆負載在求值當下現算：換頻率軸（jit 下，軸是追蹤值）、換空氣條件，可微入口對非可微入口在新條件下的值。

    三種原牆都比：換一條同長的軸（含實測表的三個表列頻點，表內外分界跟建樣板那條不同）在 jit 下照新軸算，
    垂直與 Paris 角度格；換空氣後垂直、0 度、既有入口三者一致。另帶審查的合成表（100／1000 Hz 兩列，
    100 Hz 建樣板、jit 裡求 1000 Hz）。
    """
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    frequencies, angles = jnp.asarray(FREQUENCIES), tmm.diffuse_incidence_angles_deg(method="paris")
    other = jnp.asarray(OTHER_AXIS)
    heavy = air.model_copy(update={"air_density_kg_m3": 2 * air.air_density_kg_m3})
    rows: dict[str, object] = {}
    for name, material in _mounted().items():
        template = build_template(material, frequencies, air=air, ranges=ranges)
        base = template.baseline_parameters()
        rows[name] = {
            "air_normal": _bits(surface_impedance_from_parameters(template, base, frequencies, air=heavy)),
            "air_angle_zero": _bits(surface_impedance_from_parameters(template, base, frequencies, air=heavy,
                                                                      incidence_angles_deg=jnp.asarray([0.]))[:, 0]),
            "air_existing": _bits(construction.surface_impedance(material, frequencies, air=heavy, ranges=ranges)[0]),
            "jit_other_axis": _bits(jax.jit(_at_axis(template, base, air))(other)),
            "other_axis_existing": _bits(construction.surface_impedance(material, other, air=air, ranges=ranges)[0]),
            "jit_other_axis_angles": _bits(jax.jit(_at_axis(template, base, air, angles))(other)),
            "other_axis_angles_existing": _bits(construction.surface_impedance_at_angles(
                material, other, angles, air=air, ranges=ranges)[0]),
        }
    table = MeasuredImpedance(points=(
        MeasuredImpedancePoint(frequency_hz=100., real_pa_s_m=1000., imag_pa_s_m=0.),
        MeasuredImpedancePoint(frequency_hz=1000., real_pa_s_m=2000., imag_pa_s_m=0.)),
        base_assembly=Assembly(layers=(AirLayer(thickness_m=.1),), backing="air"), source_and_conditions="考卷假表")
    synthetic = MountedOnWall(layers=(AirLayer(thickness_m=.01),), wall=table)
    template = build_template(synthetic, jnp.asarray([100.]), air=air, ranges=ranges)
    zero = {"layer1.thickness_m": jnp.asarray(0.)}
    rows["review_synthetic"] = {
        "jit_other_axis": _bits(jax.jit(_at_axis(template, zero, air))(jnp.asarray([1000.]))),
        "eager_other_axis": _bits(surface_impedance_from_parameters(template, zero, jnp.asarray([1000.]), air=air)),
        "wall_at_other_axis": _bits(construction.surface_impedance(table, jnp.asarray([1000.]), air=air, ranges=ranges)[0]),
    }
    return rows


def close_knots(air: PhysicsConstants) -> dict[str, object]:
    """單精度下很近的表列（審查修補二的擋項）四案：review 表列 1000 與 1000.0001220703125 Hz（對數後在單精度相等）；
    collide 1000 與 1000.00001 Hz（單精度下兩列捨入成同一個值，1000 存得下、另一列存不下）；
    straddle 999.99999 與 1000.00001 Hz（兩列都捨入成 1000，求 1000 Hz 落在兩列之間：分母是零那一支）；
    middle 999、1000.00001、1000.00002、1001 Hz（中間兩列重合，求 1000 Hz 要挑對段）；
    edge 下端 50.000001 Hz 在單精度存不下（捨入成 50），求 50 Hz 照真值是表外（走底構造）。新增層厚度 0（等於原牆本身）與 1 cm 各求一次，
    單精度頻率、角度與參數，直接與 jit、垂直與角度；回值、有限與否、對新增層厚度的梯度，與既有入口（雙精度）同一頻點的值。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    base = Assembly(layers=(AirLayer(thickness_m=.1),), backing="air")
    cases: dict[str, tuple[tuple[tuple[float, float], ...], tuple[float, ...]]] = {
        "review": (((1000., 1000.), (1000.0001220703125, 2000.)), (1000.00006103515625,)),
        "collide": (((1000., 1000.), (1000.00001, 2000.)), (1000., 1000.00006103515625)),
        "straddle": (((999.99999, 1000.), (1000.00001, 2000.)), (1000.,)),
        "middle": (((999., 1000.), (1000.00001, 2000.), (1000.00002, 3000.), (1001., 4000.)), (1000.,)),
        "edge": (((50.000001, 1000.), (500., 2000.)), (50., 499.99997))}
    angles = jnp.asarray([0., 60.], dtype=jnp.float32)
    rows: dict[str, object] = {}
    for name, (points, axis) in cases.items():
        wall = MeasuredImpedance(points=tuple(MeasuredImpedancePoint(frequency_hz=f, real_pa_s_m=z, imag_pa_s_m=0.)
                                              for f, z in points), base_assembly=base, source_and_conditions="考卷假表")
        frequencies = jnp.asarray(axis, dtype=jnp.float32)
        exact = jnp.asarray(np.asarray(frequencies, dtype=np.float64))
        template = build_template(MountedOnWall(layers=(AirLayer(thickness_m=.01),), wall=wall), exact, air=air, ranges=ranges)
        results: list[dict[str, object]] = []
        for thickness in (0., .01):
            parameters = {"layer1.thickness_m": jnp.asarray(thickness, dtype=jnp.float32)}
            for incidence in (None, angles):
                def evaluate(values: Parameters, incidence: jax.Array | None = incidence) -> jax.Array:
                    return surface_impedance_from_parameters(template, values, frequencies, air=air, incidence_angles_deg=incidence)
                for mode, function in (("eager", evaluate), ("jit", jax.jit(evaluate))):
                    value = function(parameters)
                    gradient = jax.grad(lambda values, function=function: jnp.sum(jnp.real(function(values))))(parameters)
                    results.append({"thickness": thickness, "angles": incidence is not None, "mode": mode, "value": _bits(value),
                                    "gradient": float(gradient["layer1.thickness_m"])})
        rows[name] = {"results": results,
                      "existing": _bits(construction.surface_impedance(wall, exact, air=air, ranges=ranges)[0]),
                      "existing_angles": _bits(construction.surface_impedance_at_angles(
                          wall, exact, jnp.asarray(np.asarray(angles, dtype=np.float64)), air=air, ranges=ranges)[0])}
    return rows


def notes(air: PhysicsConstants) -> dict[str, object]:
    """註記兩案（10 Hz）：新增第 1 層 foam 與原牆第 1 層 fibrous 同樣小松多孔層（同號註記）；
    實測表原牆 50–500 Hz 的補值，底構造第 1 層同時是材料模型外推。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    def porous(material_class: Literal["fibrous", "foam"]) -> PorousLayer:
        return PorousLayer(thickness_m=.05, flow_resistivity_pa_s_m2=19100., material_class=material_class)
    cases = {"collide": MountedOnWall(layers=(porous("foam"),), wall=Assembly(layers=(porous("fibrous"),), backing="rigid")),
             "fill": MountedOnWall(layers=(AirLayer(thickness_m=.05),), wall=_walls()["measured"])}
    rows: dict[str, object] = {}
    for name, material in cases.items():
        _, found = construction.surface_impedance(material, jnp.asarray([10.]), air=air, ranges=ranges)
        rows[name] = {"material": material.model_dump(mode="json"), "notes": [note.model_dump(mode="json") for note in found]}
    return rows


def three_lane(air: PhysicsConstants) -> dict[str, object]:
    """三路報表：x0 換成貼在原牆上的構造（原牆是那一面原本的固定阻抗），有限元素用替身但收下每面阻抗。

    回：有限元素收到的 x0 阻抗是否逐位等於非可微入口在有限元素頻率軸上的值、每個細軸頻點的幾何路與晚期能量。
    """
    del air
    fixed = report_io.solver_inputs(_input(order=1))
    scene_air, ranges = construction_environment(density_kg_m3=fixed.density_kg_m3, sound_speed_m_s=fixed.sound_speed_m_s)
    materials = dict(fixed.materials_by_wall)
    wall = materials[Wall.X0]
    assert isinstance(wall, ConstantImpedance)
    mounted = MountedOnWall(layers=_covering(), wall=wall)
    materials[Wall.X0] = mounted
    fake = cast(Callable[..., tuple[float, ...]], fake_fem_energy)
    rows: dict[str, object] = {}
    for label, solved in (("fixed", fixed), ("mounted", fixed._replace(materials_by_wall=materials))):
        seen: list[dict[Wall, tuple[complex, ...] | None]] = []

        def fem(**kwargs: object) -> tuple[float, ...]:
            seen.append(cast(dict[Wall, tuple[complex, ...] | None], kwargs["wall_impedances"]))
            return fake(**kwargs)
        with patch.object(report, "_solve_fem_energy", fem):
            raw = report.solve_three_lane_report(**solved._asdict())
        expected = construction.surface_impedance(mounted, jnp.asarray(raw.fem_frequencies_hz), air=scene_air, ranges=ranges)[0]
        rows[label] = {"capability": raw.capability.materials,
                       "x0_is_mounted": seen[0][Wall.X0] == tuple(complex(z) for z in np.asarray(expected)),
                       "geometric": [point.geometric_energy for point in raw.points],
                       "late": [point.late_energy for point in raw.points]}
    return rows


def main() -> None:
    mode, output = sys.argv[1:]
    air = load_physics_constants(config_path("physics_constants.toml"))
    actions: dict[str, Callable[[PhysicsConstants], object]] = {
        "control": control, "zero": zero, "rigid": rigid, "paths": paths, "gradients": gradients, "live_wall": live_wall, "notes": notes, "close_knots": close_knots,
        "rejections": rejections, "three_lane": three_lane}
    result = {"dtype": str(jnp.asarray(0.).dtype), "rows": actions[mode](air)}
    Path(output).write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
