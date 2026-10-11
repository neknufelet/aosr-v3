"""第二步雙精度子行程探針；只呼叫 JAX 產品，原套件不進考卷。"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jve

from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.materials import transfer_matrix as tmm


class Layer(TypedDict):
    kind: str
    thickness_m: NotRequired[float]
    surface_mass_kg_m2: NotRequired[float]
    impedance_real_pa_s_m: NotRequired[float]
    impedance_imag_pa_s_m: NotRequired[float]
    resistance_pa_s_m: NotRequired[float]
    density_kg_m3: NotRequired[float]
    hole_diameter_m: NotRequired[float]
    hole_spacing_m: NotRequired[float]
    slot_width_m: NotRequired[float]
    slot_spacing_m: NotRequired[float]
    open_area: NotRequired[float]
    porosity: NotRequired[float]
    tortuosity: NotRequired[float]
    flow_resistivity_pa_s_m2: NotRequired[float]
    method: NotRequired[str]
    model: NotRequired[str]
    fibre_type: NotRequired[int]
    free: NotRequired[bool]
    end_correction: NotRequired[str]


class Case(TypedDict):
    layers: list[Layer]
    frequencies_hz: list[float]
    quantity: NotRequired[str]


class Property(TypedDict):
    value: float
    unit: str


class Problem(TypedDict):
    cases: dict[str, Case]
    air_properties: dict[str, Property]


def optional(layer: Layer, key: str) -> jax.Array | None:
    value = cast(dict[str, object], layer).get(key)
    return None if value is None else jnp.asarray(value)


def matrix(layer: Layer, f: jax.Array, air: PhysicsConstants) -> jax.Array:
    """SI 欄位直接送公開函式，連續參數不轉 Python float。"""
    kind = layer['kind']
    if kind == 'constant_z':
        return tmm.constant_z(f, jnp.asarray(layer['impedance_real_pa_s_m']) + 1j * layer['impedance_imag_pa_s_m'])
    if kind == 'membrane':
        return tmm.membrane_layer(f, jnp.asarray(layer['surface_mass_kg_m2']))
    thickness = jnp.asarray(layer['thickness_m'])
    if kind == 'air':
        return tmm.air_layer(f, thickness, air=air)
    if kind == 'porous':
        return tmm.porous_layer(f, thickness, jnp.asarray(layer['flow_resistivity_pa_s_m2']),
                                air=air, model=layer['model'])
    if kind == 'facing':
        return tmm.porous_facing_layer(f, thickness, air=air, method=layer['method'],
            surface_mass_kg_m2=optional(layer, 'surface_mass_kg_m2'),
            resistance_pa_s_m=optional(layer, 'resistance_pa_s_m'),
            porosity=jnp.asarray(layer['porosity']), tortuosity=jnp.asarray(layer['tortuosity']),
            free=layer.get('free', True), density_kg_m3=optional(layer, 'density_kg_m3'),
            hole_diameter_m=optional(layer, 'hole_diameter_m'),
            hole_spacing_m=optional(layer, 'hole_spacing_m'), open_area=optional(layer, 'open_area'))
    if kind == 'perforated':
        return tmm.perforated_panel_layer(f, thickness, jnp.asarray(layer['hole_diameter_m']),
            air=air, method=layer['method'], hole_spacing_m=optional(layer, 'hole_spacing_m'),
            open_area=optional(layer, 'open_area'), surface_mass_kg_m2=optional(layer, 'surface_mass_kg_m2'),
            end_correction=layer.get('end_correction'))
    if kind == 'slotted':
        return tmm.slotted_panel_layer(f, thickness, jnp.asarray(layer['slot_width_m']),
            air=air, method=layer['method'], slot_spacing_m=optional(layer, 'slot_spacing_m'),
            open_area=optional(layer, 'open_area'), surface_mass_kg_m2=optional(layer, 'surface_mass_kg_m2'))
    raise ValueError(kind)


def spectrum(case: Case, f: jax.Array, air: PhysicsConstants) -> tuple[jax.Array, ...]:
    if case.get('quantity') == 'pore_properties':
        layer = case['layers'][0]
        return tmm.viscothermal_circular(f, jnp.asarray(layer['hole_diameter_m']), air=air,
                                        open_area=optional(layer, 'open_area'))
    return tmm.compute([matrix(layer, f, air) for layer in case['layers']], air=air)


def hexes(values: jax.Array | np.ndarray) -> list[str]:
    return [float(x).hex() for x in np.asarray(values).reshape(-1)]


def encode(values: tuple[jax.Array, ...]) -> dict[str, list[str]]:
    if len(values) == 2:
        k, z = values
        return {'pore_wavenumber_real_hex': hexes(k.real), 'pore_wavenumber_imag_hex': hexes(k.imag),
                'pore_impedance_real_hex': hexes(z.real), 'pore_impedance_imag_hex': hexes(z.imag)}
    z, r, alpha = values
    return {'surface_impedance_real_hex': hexes(z.real), 'surface_impedance_imag_hex': hexes(z.imag),
            'reflection_real_hex': hexes(r.real), 'reflection_imag_hex': hexes(r.imag), 'absorption_hex': hexes(alpha)}


def reference(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    return {'cases': {name: encode(spectrum(case, jnp.asarray(case['frequencies_hz']), air))
                      for name, case in problem['cases'].items()},
            'dtype': str(jnp.asarray(0.0).dtype),
            'air': {name: getattr(air, name) for name in problem['air_properties']}}


GRADIENT_CASES = ('constant_complex', 'membrane_0.02', 'facing_rebillard_measured',
    'facing_rebillard_geometry', 'facing_rebillard_open_area', 'facing_rebillard_zero_resistance',
    'facing_pieren_free', 'facing_pieren_fixed', 'facing_pieren_flag_free', 'facing_pieren_flag_fixed',
    'facing_pieren_geometry', 'barrier_mass_0.05', 'barrier_pitch', 'barrier_mpp_mass_0.05',
    'bessel_ingard_mass_0.05', 'eq_fluid_mass_0.05', 'eq_fluid_pitch', 'slotted_mass_low',
    'slotted_pitch', 'pore_0.0005', 'pore_scaled')
CONTINUOUS = ('thickness_m', 'surface_mass_kg_m2', 'resistance_pa_s_m', 'density_kg_m3',
    'hole_diameter_m', 'hole_spacing_m', 'slot_width_m', 'slot_spacing_m', 'open_area',
    'porosity', 'tortuosity', 'impedance_real_pa_s_m', 'impedance_imag_pa_s_m')


def parameterized(case: Case, keys: tuple[str, ...], p: jax.Array) -> Case:
    layer = dict(case['layers'][0])
    for index, key in enumerate(keys):
        layer[key] = p[index]
    return cast(Case, {**case, 'layers': [layer, *case['layers'][1:]]})


def loss(case: Case, f: jax.Array, air: PhysicsConstants) -> jax.Array:
    values = spectrum(case, f, air)
    if len(values) == 2:
        k, z = values
        return (k.real + .3 * k.imag) / 100 + (z.real + .4 * z.imag) / air.rho_c
    z, _, alpha = values
    return (z.real + .3 * z.imag) / air.rho_c + .7 * alpha


def gradients(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for name in GRADIENT_CASES:
        case = problem['cases'][name]
        keys = tuple(key for key in CONTINUOUS if key in case['layers'][0])
        parameters = jnp.asarray([cast(dict[str, float], case['layers'][0])[key] for key in keys], dtype=float)
        for frequency in (20.0, 730.0, 8000.0):
            def objective(p: jax.Array) -> jax.Array:
                return loss(parameterized(case, keys, p), jnp.asarray(frequency), air)
            automatic = jax.jit(jax.grad(objective))(parameters)
            for index, key in enumerate(keys):
                scale = jnp.maximum(jnp.abs(parameters[index]), 1e-3)
                delta = scale * 1e-5
                direction = jnp.zeros_like(parameters).at[index].set(delta)
                central = (objective(parameters + direction) - objective(parameters - direction)) / (2 * delta)
                rows.append({'case': name, 'frequency': frequency, 'parameter': key,
                    'automatic_scaled': float(automatic[index] * scale), 'central_scaled': float(central * scale)})
    return {'rows': rows}


def transforms(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    rows: dict[str, object] = {}
    for name in GRADIENT_CASES:
        case = problem['cases'][name]
        keys = tuple(key for key in CONTINUOUS if key in case['layers'][0])
        p = jnp.asarray([cast(dict[str, float], case['layers'][0])[key] for key in keys], dtype=float)
        def evaluate(f: jax.Array, parameters: jax.Array) -> tuple[jax.Array, ...]:
            return spectrum(parameterized(case, keys, parameters), f, air)
        f = jnp.asarray([20.0, 730.0, 8000.0])
        # Both frequency and every supplied continuous parameter are dynamic inputs.
        batch = jnp.stack((p * .98, p, p * 1.02))
        pointwise = tuple(jnp.stack([evaluate(f[i], batch[i])[j] for i in range(f.size)])
                          for j in range(len(evaluate(f[0], batch[0]))))
        rows[name] = {'jit': encode(jax.jit(jax.vmap(evaluate))(f, batch)),
                      'vmap': encode(jax.vmap(evaluate)(f, batch)), 'pointwise': encode(pointwise)}
    return {'rows': rows}


def bessel(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    """SciPy 的指數縮放在比值中消掉，β>1000 也不溢位。"""
    beta = jnp.asarray(np.geomspace(1e-3, 1024.0, 1501))
    rows: list[dict[str, object]] = []
    for ray in (1.0, problem['air_properties']['prandtl_number']['value']):
        direction = jnp.sqrt(jnp.asarray(-1j * ray))
        z = np.asarray(beta * direction)
        expected = jve(1, z) / jve(0, z)
        actual = jax.jit(tmm._bessel_j1_over_z_j0)(beta * direction)
        for kind, value, want in (('ratio', beta * direction * actual, expected), ('ratio_over_z', actual, expected / z)):
            rows.append({'ray': ray, 'kind': kind, 'actual_real': hexes(value.real),
                'actual_imag': hexes(value.imag), 'expected_real': hexes(want.real), 'expected_imag': hexes(want.imag)})
    derivative_rows: list[dict[str, object]] = []
    for ray in (1.0, problem['air_properties']['prandtl_number']['value']):
        direction = jnp.sqrt(jnp.asarray(-1j * ray))
        for divided in (False, True):
            for component in ('real', 'imag'):
                def objective(b: jax.Array) -> jax.Array:
                    z = b * direction
                    q = tmm._bessel_j1_over_z_j0(z)
                    value = q if divided else z * q
                    return value.real if component == 'real' else value.imag
                automatic = jax.jit(jax.vmap(jax.grad(objective)))(beta)
                # An independent SciPy central difference also checks the derivative's sign and branch.
                def scipy_value(b: np.ndarray) -> np.ndarray:
                    z = b * complex(direction)
                    q = jve(1, z) / jve(0, z)
                    value = q / z if divided else q
                    return value.real if component == 'real' else value.imag
                b = np.asarray(beta)
                h = b * 1e-4
                central = (scipy_value(b + h) - scipy_value(b - h)) / (2 * h)
                derivative_rows.append({'ray': ray, 'divided': divided, 'component': component,
                    'automatic_scaled': hexes(automatic * beta), 'central_scaled': hexes(central * b)})
    max_diameter = max(layer.get('hole_diameter_m', 0.0) for case in problem['cases'].values() for layer in case['layers'])
    required_beta = max_diameter / 2 * np.sqrt(2 * np.pi * 8000 * air.air_density_kg_m3 / air.air_viscosity_pa_s)
    return {'rows': rows, 'derivatives': derivative_rows, 'dtype': str(beta.dtype),
            'required_beta': float(required_beta), 'tested_max_beta': float(beta[-1])}


def options(air: PhysicsConstants) -> dict[str, object]:
    f, t, d = jnp.asarray(500.0), jnp.asarray(.002), jnp.asarray(.001)
    calls: dict[str, Callable[[], object]] = {
        'perforated_method': lambda: tmm.perforated_panel_layer(f, t, d, air=air, method='bad', open_area=jnp.asarray(.1)),
        'end_correction': lambda: tmm.perforated_panel_layer(f, t, d, air=air, end_correction='bad', open_area=jnp.asarray(.1)),
        'slotted_method': lambda: tmm.slotted_panel_layer(f, t, d, air=air, method='bad', open_area=jnp.asarray(.1)),
        'facing_method': lambda: tmm.porous_facing_layer(f, t, air=air, method='bad'),
        'missing_pitch': lambda: tmm.perforated_panel_layer(f, t, d, air=air),
        'missing_facing_mass': lambda: tmm.porous_facing_layer(f, t, air=air, method='pieren_free', resistance_pa_s_m=jnp.asarray(100.)),
        'missing_facing_resistance': lambda: tmm.porous_facing_layer(f, t, air=air, method='pieren_fixed'),
        'missing_pressure': lambda: tmm.viscothermal_circular(f, d, air=air.model_copy(update={'atmospheric_pressure_pa': None})),
        'missing_prandtl': lambda: tmm.viscothermal_circular(f, d, air=air.model_copy(update={'prandtl_number': None})),
    }
    rejected = []
    for name, call in calls.items():
        try:
            call()
        except ValueError:
            rejected.append(name)
    # A fixed termination behind a membrane retains the source's absolute surface impedance.
    terminal = tmm.constant_z(f, jnp.asarray(180 + 240j))
    z, _, _ = tmm.compute([tmm.membrane_layer(f, jnp.asarray(.2)), terminal], air=air)
    return {'rejected': rejected, 'terminal_z': [float(z.real), float(z.imag)]}


def main() -> None:
    mode, problem_file, output_file = sys.argv[1:]
    problem = cast(Problem, json.loads(Path(problem_file).read_text()))
    air = load_physics_constants(config_path('physics_constants.toml'))
    air = PhysicsConstants.model_validate({**air.model_dump(),
        **{name: prop['value'] for name, prop in problem['air_properties'].items()}})
    actions = {'reference': reference, 'transforms': transforms, 'gradients': gradients, 'bessel': bessel}
    result = options(air) if mode == 'options' else actions[mode](problem, air)
    Path(output_file).write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
