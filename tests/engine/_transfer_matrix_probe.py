"""雙精度子行程探針；父考卷只讀它寫在 tmp_path 的結果。"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.materials import transfer_matrix as tmm


class Layer(TypedDict):
    kind: str
    thickness_m: float
    model: NotRequired[str]
    flow_resistivity_pa_s_m2: NotRequired[float]
    porosity: NotRequired[float]
    tortuosity: NotRequired[float]
    fibre_type: NotRequired[int]


class Case(TypedDict):
    layers: list[Layer]
    frequencies_hz: list[float]


class Property(TypedDict):
    value: float
    unit: str


class Problem(TypedDict):
    cases: dict[str, Case]
    air_properties: dict[str, Property]


def matrices(case: Case, frequencies: jax.Array, air: PhysicsConstants) -> list[jax.Array]:
    """只將 SI 題目欄位送進產品公開 API，不在考卷重新實作公式。"""
    result = []
    for layer in case['layers']:
        thickness = jnp.asarray(layer['thickness_m'])
        if layer['kind'] == 'air':
            matrix = tmm.air_layer(frequencies, thickness, air=air)
        else:
            matrix = tmm.porous_layer(
                frequencies, thickness, jnp.asarray(layer['flow_resistivity_pa_s_m2']),
                air=air, model=layer['model'], fibre_type=layer['fibre_type'],
                porosity=jnp.asarray(layer['porosity']), tortuosity=jnp.asarray(layer['tortuosity']),
            )
        result.append(matrix)
    return result


def spectrum(case: Case, frequencies: jax.Array, air: PhysicsConstants) -> tuple[jax.Array, jax.Array, jax.Array]:
    return tmm.compute(matrices(case, frequencies, air), air=air)


def encode(values: Sequence[jax.Array]) -> dict[str, list[str]]:
    z, reflection, absorption = (np.asarray(x) for x in values)
    return {
        'surface_impedance_real_hex': [float(x).hex() for x in z.real],
        'surface_impedance_imag_hex': [float(x).hex() for x in z.imag],
        'reflection_real_hex': [float(x).hex() for x in reflection.real],
        'reflection_imag_hex': [float(x).hex() for x in reflection.imag],
        'absorption_hex': [float(x).hex() for x in absorption],
    }


def reference(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    cases = {name: encode(spectrum(case, jnp.asarray(case['frequencies_hz']), air))
             for name, case in problem['cases'].items()}
    explicit_none, rigid = {}, {}
    for name, case in problem['cases'].items():
        layers = matrices(case, jnp.asarray(case['frequencies_hz']), air)
        explicit_none[name] = encode(tmm.compute(layers, air=air, load_impedance_pa_s_m=None))
        zs = tmm.rigid_backing_impedance(tmm.global_matrix(layers))
        r, alpha = tmm.reflection_and_absorption_coefficient(zs, air=air)
        rigid[name] = encode((zs, r, alpha))
    return {'cases': cases, 'dtype': str(jnp.asarray(0.0).dtype),
            'air': {name: getattr(air, name) for name in problem['air_properties']},
            'explicit_none_cases': explicit_none, 'rigid_cases': rigid}


def determinant_row(layer: jax.Array, name: str) -> dict[str, object]:
    """直接考 AD−BC 的物理不變量，不以產品的背牆比值作準答案。"""
    m = np.asarray(layer)
    ad, bc = m[..., 0, 0] * m[..., 1, 1], m[..., 0, 1] * m[..., 1, 0]
    determinant = ad - bc
    return {'layer': name,
            'real_hex': [float(x).hex() for x in determinant.real.reshape(-1)],
            'imag_hex': [float(x).hex() for x in determinant.imag.reshape(-1)],
            'cancellation_scale_hex': [float(x).hex() for x in (np.abs(ad) + np.abs(bc)).reshape(-1)]}


def determinants(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    rows, conditioned = [], []
    for name, case in problem['cases'].items():
        for index, layer in enumerate(matrices(case, jnp.asarray(case['frequencies_hz']), air)):
            rows.append(determinant_row(layer, f'{name}::{index}'))
        # 厚層高衰減會讓 AD 和 BC 都巨大；同一模型與其他凍結參數以 1 mm
        # 再驗一次，直接使用 det=1 的絕對尺，不讓相消條件數遮住整體縮放。
        thin_case = cast(Case, {**case, 'layers': [{**layer, 'thickness_m': .001} for layer in case['layers']]})
        for index, layer in enumerate(matrices(thin_case, jnp.asarray(case['frequencies_hz']), air)):
            conditioned.append(determinant_row(layer, f'{name}::{index}::1mm'))
    return {'rows': rows, 'conditioned_rows': conditioned}


def transforms(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    """每個模型與每種層疊都穿過 jit 和 vmap，連續參數仍是追蹤中的陣列。"""
    rows: dict[str, object] = {}
    for name, case in problem['cases'].items():
        frequencies = jnp.asarray([20.0, 100.0, 1000.0, 8000.0])
        compiled = jax.jit(lambda f: spectrum(case, f, air))(frequencies)
        mapped = jax.vmap(lambda f: spectrum(case, f, air))(frequencies)
        pointwise = tuple(jnp.stack([spectrum(case, f, air)[i] for f in frequencies]) for i in range(3))
        rows[name] = {'jit': encode(compiled), 'vmap': encode(mapped), 'pointwise': encode(pointwise)}
    # 再把厚度、流阻、孔隙率、曲折度一起當 jit/vmap 的動態輸入。
    def dynamic(parameters: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        layer = tmm.porous_layer(jnp.asarray(700.0), parameters[0], parameters[1],
                                air=air, model='wilson_2015', porosity=parameters[2], tortuosity=parameters[3])
        return tmm.compute([layer], air=air)
    parameters = jnp.asarray([[0.03, 12000.0, 0.7, 1.2], [0.08, 31000.0, 0.9, 2.0]])
    rows['dynamic_parameters'] = {
        'jit': encode(jax.jit(jax.vmap(dynamic))(parameters)),
        'vmap': encode(jax.vmap(dynamic)(parameters)),
        'pointwise': encode(tuple(jnp.stack([dynamic(p)[i] for p in parameters]) for i in range(3))),
    }
    return {'rows': rows}


def gradients(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    """中央差分在光滑支內比四個 SI 參數；C=60 不可微的等號不拿中央差分跨支。"""
    options = sorted({(layer['model'], layer['fibre_type']) for case in problem['cases'].values()
                      for layer in case['layers'] if layer['kind'] == 'porous'})
    rows: list[dict[str, object]] = []
    for model, fibre in options:
        for frequency in (80.0, 1000.0):
            parameters = jnp.asarray([0.05, 10000.0, 0.72, 1.8])
            def loss(p: jax.Array) -> jax.Array:
                matrix = tmm.porous_layer(jnp.asarray(frequency), p[0], p[1], air=air,
                                         model=model, fibre_type=fibre, porosity=p[2], tortuosity=p[3])
                z, _, alpha = tmm.compute([matrix], air=air)
                return (jnp.real(z) + 0.3 * jnp.imag(z)) / air.rho_c + 0.7 * alpha
            automatic = jax.jit(jax.grad(loss))(parameters)
            for index, parameter in enumerate(('thickness', 'flow_resistivity', 'porosity', 'tortuosity')):
                delta = parameters[index] * 1e-5
                direction = jnp.zeros_like(parameters).at[index].set(delta)
                central = (loss(parameters + direction) - loss(parameters - direction)) / (2 * delta)
                rows.append({'model': model, 'fibre_type': fibre, 'frequency': frequency, 'parameter': parameter,
                             'automatic_scaled': float(automatic[index] * parameters[index]),
                             'central_scaled': float(central * parameters[index])})
    # C=60 兩側的梯度必須有限，包含未選支也被計算的 where（分段選擇）。
    boundary = []
    for ratio in (60.0 - 1e-5, 60.0, 60.0 + 1e-5):
        def component_loss(p: jax.Array) -> jax.Array:
            k, z = tmm.equivalent_fluid_model(jnp.asarray(100.0), p[0], air=air,
                                             model='mechel_1976', porosity=p[1])
            return jnp.real(k + z) + jnp.imag(k + z)
        boundary.append(np.asarray(jax.grad(component_loss)(jnp.asarray([ratio * air.air_density_kg_m3 * 100.0, 0.72]))).tolist())
    def air_loss(thickness: jax.Array) -> jax.Array:
        z, _, _ = tmm.compute([tmm.air_layer(jnp.asarray(500.0), thickness, air=air)], air=air)
        return jnp.imag(z) / air.rho_c
    thickness = jnp.asarray(0.08)
    delta = thickness * 1e-5
    rows.append({'model': 'air', 'parameter': 'thickness',
                 'automatic_scaled': float(jax.grad(air_loss)(thickness) * thickness),
                 'central_scaled': float((air_loss(thickness + delta) - air_loss(thickness - delta)) / (2 * delta) * thickness)})
    return {'rows': rows, 'boundary_gradients': boundary}


def options(air: PhysicsConstants) -> dict[str, object]:
    failures = []
    for model, fibre in (('unknown', 1), ('mechel_grundmann', 0)):
        try:
            tmm.equivalent_fluid_model(jnp.asarray(100.0), jnp.asarray(10000.0), air=air, model=model, fibre_type=fibre)
        except ValueError:
            failures.append(model)
    try:
        tmm.global_matrix([])
    except ValueError:
        failures.append('empty_stack')
    return {'rejected': failures}


def main() -> None:
    mode, problem_file, output_file = sys.argv[1:]
    problem = cast(Problem, json.loads(Path(problem_file).read_text()))
    air = load_physics_constants(config_path('physics_constants.toml'))
    actions = {'reference': reference, 'transforms': transforms, 'gradients': gradients, 'determinants': determinants}
    result = options(air) if mode == 'options' else actions[mode](problem, air)
    Path(output_file).write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
