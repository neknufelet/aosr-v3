"""可微構造入口的隔離算數子行程（#561）；雙精度由正式開機入口開，父考卷只讀它寫的結果。"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

from aosr.runtime import configure_calculation_jax

configure_calculation_jax()

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from aosr.config.construction_presets import load_construction_presets
from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.materials import construction
from aosr.materials import transfer_matrix as tmm
from aosr.materials.construction_diff import ConstructionTemplate, build_template, surface_impedance_from_parameters
from aosr.materials.construction_models import SurfaceMaterial

# 20 Hz～8 kHz 對數取樣，涵蓋空腔共振與多孔材外推區；角度用無規入射那一格（Paris，0～90 度不含 90）。
FREQUENCIES = np.geomspace(20., 8000., 61)
GRADIENT_FREQUENCIES = (80., 1000.)
OBLIQUE_DEG = 60.
STEP = 1e-5  # 中央差分相對步長，同 _transfer_matrix_probe.gradients。
# 單參數擾動倍數：夠小讓預設仍過既有入口的格式與範圍檢查（縫寬小於縫距、孔徑分界、Komatsu 流阻範圍），
# 又遠大於捨入，參數沒接進公式時阻抗一定會變。不是容差。
PERTURB = 1.01

Parameters = dict[str, jax.Array]


def _bits(value: jax.Array) -> dict[str, object]:
    array = np.asarray(value)
    return {"real": [float(x).hex() for x in array.real.reshape(-1)],
            "imag": [float(x).hex() for x in array.imag.reshape(-1)],
            "dtype": str(value.dtype), "shape": list(array.shape)}


def _templates(air: PhysicsConstants) -> dict[str, ConstructionTemplate]:
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    return {preset.id: build_template(adapter.validate_python(preset.material), jnp.asarray(FREQUENCIES), air=air, ranges=ranges)
            for preset in load_construction_presets(config_path("construction_presets.toml"))}


def identity(templates: dict[str, ConstructionTemplate], air: PhysicsConstants) -> dict[str, object]:
    """基準值上可微入口對既有兩個入口；父考卷逐位比。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    frequencies, angles = jnp.asarray(FREQUENCIES), tmm.diffuse_incidence_angles_deg(method="paris")
    rows: dict[str, object] = {}
    for name, template in templates.items():
        base = template.baseline_parameters()
        rows[name] = {
            "variables": list(template.names),
            "normal": _bits(surface_impedance_from_parameters(template, base, frequencies, air=air)),
            "normal_existing": _bits(construction.surface_impedance(template.material, frequencies, air=air, ranges=ranges)[0]),
            "angles": _bits(surface_impedance_from_parameters(template, base, frequencies, air=air, incidence_angles_deg=angles)),
            "angles_existing": _bits(construction.surface_impedance_at_angles(
                template.material, frequencies, angles, air=air, ranges=ranges)[0]),
        }
    return rows


def _loss(template: ConstructionTemplate, air: PhysicsConstants, frequency: float, oblique: bool,
          dtype: type = jnp.float64) -> Callable[[Parameters], jax.Array]:
    """實數純量：阻抗實虛部（除 ρc）加吸音；斜入射那支用 60 度。頻率與角度照 dtype 建，單精度路才不被升回雙精度。"""
    def loss(parameters: Parameters) -> jax.Array:
        angles = jnp.asarray([OBLIQUE_DEG], dtype=dtype) if oblique else None
        z = surface_impedance_from_parameters(template, parameters, jnp.asarray([frequency], dtype=dtype), air=air,
                                              incidence_angles_deg=angles)
        alpha = tmm.reflection_and_absorption_coefficient(z, air=air, incidence_angles_deg=angles)[1]
        return jnp.sum((jnp.real(z) + 0.3 * jnp.imag(z)) / air.rho_c + 0.7 * alpha)
    return loss


def gradients(templates: dict[str, ConstructionTemplate], air: PhysicsConstants) -> list[dict[str, object]]:
    """每筆預設的每個變數：jit(grad) 對中央差分，比無量綱 p·dL/dp。"""
    rows: list[dict[str, object]] = []
    for name, template in templates.items():
        base = template.baseline_parameters()
        for frequency in GRADIENT_FREQUENCIES:
            for oblique in (False, True):
                loss = jax.jit(_loss(template, air, frequency, oblique))
                automatic = jax.jit(jax.grad(_loss(template, air, frequency, oblique)))(base)
                for variable in template.names:
                    value = base[variable]
                    delta = value * STEP
                    central = (loss({**base, variable: value + delta}) - loss({**base, variable: value - delta})) / (2 * delta)
                    rows.append({"preset": name, "variable": variable, "frequency": frequency, "oblique": oblique,
                                 "automatic_scaled": float(automatic[variable] * value),
                                 "central_scaled": float(central * value)})
    return rows


def transforms(templates: dict[str, ConstructionTemplate], air: PhysicsConstants) -> dict[str, object]:
    """jit 對直接呼叫；vmap 兩組參數（基準、基準乘 1.07）對逐組直接呼叫。"""
    frequencies, angles = jnp.asarray(FREQUENCIES), jnp.asarray([0., 45., 80.])
    rows: dict[str, object] = {}
    for name, template in templates.items():
        base = template.baseline_parameters()
        batch = {key: jnp.stack([value, value * 1.07]) for key, value in base.items()}
        points = [{key: value[i] for key, value in batch.items()} for i in range(2)]
        row: dict[str, object] = {}
        for label, theta in (("normal", None), ("angles", angles)):
            def evaluate(parameters: Parameters, theta: jax.Array | None = theta, template: ConstructionTemplate = template) -> jax.Array:
                return surface_impedance_from_parameters(template, parameters, frequencies, air=air, incidence_angles_deg=theta)
            row[label] = {"jit": _bits(jax.jit(evaluate)(base)), "direct": _bits(evaluate(base)),
                          "vmap": _bits(jax.vmap(evaluate)(batch)),
                          "pointwise": _bits(jnp.stack([evaluate(point) for point in points]))}
        rows[name] = row
    return rows


def single(templates: dict[str, ConstructionTemplate], air: PhysicsConstants) -> dict[str, object]:
    """單精度路：同一個雙精度行程裡交 float32 的參數、頻率、角度；量跟雙精度差多少，不判。"""
    f64, f32 = jnp.asarray(FREQUENCIES), jnp.asarray(FREQUENCIES, dtype=jnp.float32)
    # Paris 那一格最後一點是 90 度往下一個雙精度位，轉單精度會捨成 90 被拒收；這裡用兩種精度都精確的整數度。
    a64, a32 = jnp.arange(0., 90., 1.), jnp.arange(0., 90., 1., dtype=jnp.float32)
    rows: dict[str, object] = {}
    for name, template in templates.items():
        base = template.baseline_parameters()
        low = {key: value.astype(jnp.float32) for key, value in base.items()}
        row: dict[str, object] = {}
        for label, (t64, t32) in (("normal", (None, None)), ("angles", (a64, a32))):
            exact = np.asarray(surface_impedance_from_parameters(template, base, f64, air=air, incidence_angles_deg=t64))
            value = surface_impedance_from_parameters(template, low, f32, air=air, incidence_angles_deg=t32)
            approx = np.asarray(value).astype(np.complex128)
            row[label] = {"dtype": str(value.dtype), "finite": bool(np.all(np.isfinite(approx))),
                          "max_rel": float(np.max(np.abs(approx - exact) / np.abs(exact)))}
        grad = jax.jit(jax.grad(_loss(template, air, GRADIENT_FREQUENCIES[1], True, jnp.float32)))(low)
        row["gradient_dtypes"] = sorted({str(value.dtype) for value in grad.values()})
        row["gradient_finite"] = all(bool(jnp.all(jnp.isfinite(value))) for value in grad.values())
        rows[name] = row
    return rows


def perturb(templates: dict[str, ConstructionTemplate], air: PhysicsConstants) -> dict[str, object]:
    """每個變數單獨乘 PERTURB：可微入口對「那一層那一格改成同一個值、重新驗證、走既有入口」；換錯層、換錯欄位、參數被忽略都對不上。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    frequencies, angles = jnp.asarray(FREQUENCIES), tmm.diffuse_incidence_angles_deg(method="paris")
    rows: dict[str, object] = {}
    for name, template in templates.items():
        base = template.baseline_parameters()
        baseline = surface_impedance_from_parameters(template, base, frequencies, air=air)
        row: dict[str, object] = {}
        for variable in template.variables:
            value = variable.baseline * PERTURB
            document = template.material.model_dump()
            document["layers"][variable.layer_index - 1][variable.field] = value
            rebuilt = adapter.validate_python(document)
            parameters = {**base, variable.name: jnp.asarray(value)}
            normal = surface_impedance_from_parameters(template, parameters, frequencies, air=air)
            row[variable.name] = {
                "normal": _bits(normal),
                "normal_existing": _bits(construction.surface_impedance(rebuilt, frequencies, air=air, ranges=ranges)[0]),
                "angles": _bits(surface_impedance_from_parameters(template, parameters, frequencies, air=air, incidence_angles_deg=angles)),
                "angles_existing": _bits(construction.surface_impedance_at_angles(rebuilt, frequencies, angles, air=air, ranges=ranges)[0]),
                "changed": not bool(jnp.array_equal(normal, baseline)),
            }
        rows[name] = row
    return rows


def _error(action: Callable[[], object]) -> str:
    try:
        action()
    except ValueError as exc:
        return str(exc)
    return ""


def rejections(templates: dict[str, ConstructionTemplate], air: PhysicsConstants) -> dict[str, object]:
    """建樣板照舊走既有檢查；求值只認樣板列的鍵、零維值；固定阻抗與實測表不收。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    frequencies = jnp.asarray(FREQUENCIES)
    porous = {"kind": "porous", "thickness_m": .05, "flow_resistivity_pa_s_m2": 19100., "material_class": "fibrous"}
    low_sigma = adapter.validate_python({"kind": "assembly", "backing": "rigid", "layers": [{**porous, "flow_resistivity_pa_s_m2": 100.}]})
    constant = adapter.validate_python({"kind": "constant_impedance", "impedance_pa_s_m": 2000.})
    template = next(iter(templates.values()))
    base = template.baseline_parameters()
    first = template.names[0]
    subset = build_template(template.material, frequencies, air=air, ranges=ranges, variables=(first,))
    return {
        "existing_check": _error(lambda: build_template(low_sigma, frequencies, air=air, ranges=ranges)),
        "constant": _error(lambda: build_template(constant, frequencies, air=air, ranges=ranges)),
        "unknown_variable": _error(lambda: build_template(template.material, frequencies, air=air, ranges=ranges, variables=("layer9.thickness_m",))),
        "missing_key": _error(lambda: surface_impedance_from_parameters(template, {}, frequencies, air=air)),
        "extra_key": _error(lambda: surface_impedance_from_parameters(template, {**base, "layer9.thickness_m": base[first]}, frequencies, air=air)),
        "batched_value": _error(lambda: surface_impedance_from_parameters(template, {**base, first: jnp.stack([base[first]] * 2)}, frequencies, air=air)),
        "subset_names": list(subset.names),
        "first": first,
        # 只放一個變數時其他欄位固定在基準值；基準上跟全部當變數逐位相同。
        "subset_bitwise": bool(jnp.array_equal(surface_impedance_from_parameters(subset, {first: base[first]}, frequencies, air=air),
                                               surface_impedance_from_parameters(template, base, frequencies, air=air))),
    }


def main() -> None:
    mode, output = sys.argv[1:]
    air = load_physics_constants(config_path("physics_constants.toml"))
    templates = _templates(air)
    actions = {"identity": identity, "gradients": gradients, "transforms": transforms, "single": single,
               "rejections": rejections, "perturb": perturb}
    result = {"dtype": str(jnp.asarray(0.).dtype), "rows": actions[mode](templates, air)}
    Path(output).write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
