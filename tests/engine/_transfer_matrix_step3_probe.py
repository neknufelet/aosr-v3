"""第三步雙精度探針：角度與積分只呼叫產品，答案由釘住的原套件另存。"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np

from tests.engine import _transfer_matrix_probe as step1
from tests.engine import _transfer_matrix_step2_probe as step2
from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.materials import transfer_matrix as tmm


class Case(step2.Case):
    incidence: NotRequired[str]
    incidence_angle: NotRequired[list[float] | None]
    diffuse_method: NotRequired[str]
    radiation_area_m2: NotRequired[float]
    x: NotRequired[list[float]]
    y_real: NotRequired[list[float]]
    y_imag: NotRequired[list[float]]


class Problem(TypedDict):
    cases: dict[str, Case]
    air_properties: dict[str, step2.Property]


def angles(case: Case) -> jax.Array:
    if case["incidence"] == "diffuse":
        return tmm.diffuse_incidence_angles_deg(
            case.get("incidence_angle"), method=case["diffuse_method"]
        )
    return jnp.asarray(case["incidence_angle"])


def matrix(
    layer: step2.Layer, f: jax.Array, air: PhysicsConstants, theta: jax.Array | None
) -> jax.Array:
    """照題目的SI與靜態選項呼叫；不在考卷重造角度物理。"""
    kind = layer["kind"]
    if kind == "membrane":
        return tmm.membrane_layer(
            f, jnp.asarray(layer["surface_mass_kg_m2"]), incidence_angles_deg=theta
        )
    t = jnp.asarray(layer["thickness_m"])
    if kind == "air":
        return tmm.air_layer(f, t, air=air, incidence_angles_deg=theta)
    if kind == "porous":
        return tmm.porous_layer(
            f,
            t,
            jnp.asarray(layer["flow_resistivity_pa_s_m2"]),
            air=air,
            model=layer["model"],
            fibre_type=layer.get("fibre_type", 1),
            porosity=jnp.asarray(layer.get("porosity", 0.95)),
            tortuosity=jnp.asarray(layer.get("tortuosity", 1.0)),
            incidence_angles_deg=theta,
        )
    if kind == "facing":
        return tmm.porous_facing_layer(
            f,
            t,
            air=air,
            method=layer["method"],
            surface_mass_kg_m2=step2.optional(layer, "surface_mass_kg_m2"),
            resistance_pa_s_m=step2.optional(layer, "resistance_pa_s_m"),
            porosity=jnp.asarray(layer["porosity"]),
            tortuosity=jnp.asarray(layer["tortuosity"]),
            free=layer.get("free", True),
            incidence_angles_deg=theta,
        )
    if kind == "perforated":
        return tmm.perforated_panel_layer(
            f,
            t,
            jnp.asarray(layer["hole_diameter_m"]),
            air=air,
            method=layer["method"],
            hole_spacing_m=step2.optional(layer, "hole_spacing_m"),
            open_area=step2.optional(layer, "open_area"),
            surface_mass_kg_m2=step2.optional(layer, "surface_mass_kg_m2"),
            end_correction=layer.get("end_correction"),
            incidence_angles_deg=theta,
        )
    return tmm.slotted_panel_layer(
        f,
        t,
        jnp.asarray(layer["slot_width_m"]),
        air=air,
        method=layer["method"],
        slot_spacing_m=step2.optional(layer, "slot_spacing_m"),
        open_area=step2.optional(layer, "open_area"),
        surface_mass_kg_m2=step2.optional(layer, "surface_mass_kg_m2"),
        incidence_angles_deg=theta,
    )


def spectrum(
    case: Case, f: jax.Array, air: PhysicsConstants, theta: jax.Array
) -> tuple[jax.Array, ...]:
    if case["layers"][0]["kind"] == "constant_z":
        l = case["layers"][0]
        z = tmm.constant_z(
            f,
            jnp.asarray(l["impedance_real_pa_s_m"] + 1j * l["impedance_imag_pa_s_m"]),
            incidence_angles_deg=theta,
        )
        r, a = tmm.reflection_and_absorption_coefficient(
            z, air=air, incidence_angles_deg=theta
        )
        return z, r, a
    return tmm.compute(
        [matrix(l, f, air, theta) for l in case["layers"]],
        air=air,
        incidence_angles_deg=theta,
        backing=case.get("backing", "rigid"),
        frequencies_hz=f,
        radiation_area_m2=jnp.asarray(case.get("radiation_area_m2", 1.0)),
    )


def encode(case: Case, air: PhysicsConstants) -> dict[str, list[str]]:
    if case.get("quantity") == "simpson":
        z = tmm.simpson(
            jnp.asarray(case["y_real"]) + 1j * jnp.asarray(case["y_imag"]),
            jnp.asarray(case["x"]),
        )
        return {
            "integral_real_hex": step2.hexes(z.real),
            "integral_imag_hex": step2.hexes(z.imag),
        }
    theta = angles(case)
    z, r, a = spectrum(case, jnp.asarray(case["frequencies_hz"]), air, theta)
    row = {
        "angles_deg_hex": step2.hexes(theta),
        "z_angle_real_hex": step2.hexes(z.real),
        "z_angle_imag_hex": step2.hexes(z.imag),
        "reflection_angle_real_hex": step2.hexes(r.real),
        "reflection_angle_imag_hex": step2.hexes(r.imag),
        "absorption_angle_hex": step2.hexes(a),
    }
    if case["incidence"] == "diffuse":
        field = tmm.field_impedance(z, theta)
        rf, af = tmm.reflection_and_absorption_coefficient(field, air=air)
        paris = tmm.diffuse_absorption_coefficient(z, theta, air=air)
        row.update(step2.encode((field, rf, af)))
        row.update(
            {
                "field_impedance_real_hex": step2.hexes(field.real),
                "field_impedance_imag_hex": step2.hexes(field.imag),
                "field_absorption_hex": step2.hexes(af),
                "paris_absorption_hex": step2.hexes(paris),
                "selected_absorption_hex": step2.hexes(
                    paris if case["diffuse_method"] == "paris" else af
                ),
            }
        )
    else:
        row.update(step2.encode((z[..., 0], r[..., 0], a[..., 0])))
    return row


def reference(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    return {
        "cases": {name: encode(c, air) for name, c in problem["cases"].items()},
        "dtype": str(jnp.asarray(0.0).dtype),
        "air": {name: getattr(air, name) for name in problem["air_properties"]},
    }


def derivative_rows(
    objective: Callable[[jax.Array], jax.Array],
    p: jax.Array,
    name: str,
    *,
    scale_floor: float = 1.0,
    gradient: Callable[[jax.Array], jax.Array] | None = None,
) -> list[dict[str, object]]:
    """``gradient`` 給了就用它（已編譯、可跨呼叫共用的自動微分）；沒給就就地編譯 ``objective`` 的梯度。"""
    automatic = (gradient or jax.jit(jax.grad(objective)))(p)
    rows = []
    for index in range(p.size):
        scale = jnp.maximum(jnp.abs(p[index]), scale_floor)
        delta = scale * 1e-5
        direction = jnp.zeros_like(p).at[index].set(delta)
        central = (objective(p + direction) - objective(p - direction)) / (2 * delta)
        rows.append(
            {
                "case": name,
                "parameter": index,
                "automatic_scaled": float(automatic[index] * scale),
                "central_scaled": float(central * scale),
            }
        )
    return rows


def gradients(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    rows = []
    for name, c in problem["cases"].items():
        if c.get("quantity") == "simpson" or c["incidence"] == "diffuse":
            continue

        def loss(p: jax.Array, f: jax.Array, case: Case = c) -> jax.Array:
            z, _, a = spectrum(case, f, air, p)
            return jnp.sum((z.real + 0.3 * z.imag) / air.rho_c + 0.7 * a)

        # 頻率是編譯後的動態參數：每個案例只編譯一次梯度（#972），三個頻點共用；中央差分照舊逐點直接算。
        compiled = jax.jit(jax.grad(loss))
        for frequency in (20.0, 730.0, 8000.0):
            f = jnp.asarray(frequency)
            rows.extend(
                derivative_rows(
                    partial(loss, f=f),
                    jnp.asarray([37.0]),
                    f"{name}::{frequency}",
                    gradient=partial(compiled, f=f),
                )
            )
    for count in (5, 6):
        p = jnp.asarray(np.linspace(7.0, 77.0, count))
        y = jnp.asarray(np.linspace(0.2, 1.7, count)) + 0.3j * jnp.asarray(
            np.linspace(1.2, 0.1, count)
        )

        def integral(q: jax.Array) -> jax.Array:
            v = tmm.simpson(y, jnp.deg2rad(q))
            return v.real + 0.4 * v.imag

        def trapezoid(q: jax.Array) -> jax.Array:
            return jnp.real(tmm.trapezoidal_integral(y, jnp.deg2rad(q)))

        def field(q: jax.Array) -> jax.Array:
            z = air.rho_c * (1.2 + y)[None, :]
            v = tmm.field_impedance(z, q)
            return jnp.sum((v.real + 0.3 * v.imag) / air.rho_c)

        def paris(q: jax.Array) -> jax.Array:
            return jnp.sum(
                tmm.diffuse_absorption_coefficient(
                    air.rho_c * (1.2 + y)[None, :], q, air=air
                )
            )

        for label, fn in [
            ("simpson", integral),
            ("trapezoid", trapezoid),
            ("field", field),
            ("paris", paris),
        ]:
            rows.extend(derivative_rows(fn, p, f"{label}::{count}"))
    rows.extend(continuous_oblique_gradients(air))
    return {"rows": rows}


def continuous_oblique_gradients(air: PhysicsConstants) -> list[dict[str, object]]:
    """角度新增後不能切斷原物理參數；亦驗輻射面積和Mechel未選支。"""
    rows = []
    p = jnp.asarray([0.04, 18000.0, 0.72, 1.8, 37.0])
    for model in ("wilson_2015", "mechel_1976"):

        def porous_loss(q: jax.Array) -> jax.Array:
            theta = q[4:]
            m = tmm.porous_layer(
                jnp.asarray(730.0),
                q[0],
                q[1],
                air=air,
                model=model,
                porosity=q[2],
                tortuosity=q[3],
                incidence_angles_deg=theta,
            )
            z, _, a = tmm.compute([m], air=air, incidence_angles_deg=theta)
            return jnp.sum((z.real + 0.3 * z.imag) / air.rho_c + 0.7 * a)

        rows.extend(
            derivative_rows(porous_loss, p, model + "::continuous", scale_floor=1e-8)
        )

    def radiation_loss(q: jax.Array) -> jax.Array:
        theta = q[2:]
        m = tmm.air_layer(jnp.asarray(730.0), q[0], air=air, incidence_angles_deg=theta)
        z, _, a = tmm.compute(
            [m],
            air=air,
            incidence_angles_deg=theta,
            backing="radiation",
            frequencies_hz=jnp.asarray(730.0),
            radiation_area_m2=q[1],
        )
        return jnp.sum((z.real + 0.3 * z.imag) / air.rho_c + 0.7 * a)

    rows.extend(
        derivative_rows(
            radiation_loss,
            jnp.asarray([0.03, 1e-6, 37.0]),
            "radiation::continuous",
            scale_floor=1e-8,
        )
    )
    for ratio in (60.0 - 1e-5, 60.0, 60.0 + 1e-5):

        def boundary(q: jax.Array) -> jax.Array:
            m = tmm.porous_layer(
                jnp.asarray(100.0),
                q[0],
                jnp.asarray(ratio * air.air_density_kg_m3 * 100.0),
                air=air,
                model="mechel_1976",
                porosity=q[1],
                incidence_angles_deg=q[2:],
            )
            return jnp.sum(m.real + 0.3 * m.imag)

        derivative = jax.jit(jax.grad(boundary))(jnp.asarray([0.04, 0.72, 89.99]))
        if not bool(jnp.all(jnp.isfinite(derivative))):
            raise AssertionError("Mechel斜入射分段或未選支污染梯度")
    return rows


def encode_integrated(values: tuple[jax.Array, ...]) -> dict[str, list[str]]:
    """三個場入射量沿用編碼，Paris另給吸音欄，不能冒用反射欄逃過絕對尺。"""
    return {**step2.encode(values[:3]), "paris_absorption_hex": step2.hexes(values[3])}


def transforms(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    c = problem["cases"]["stack_panel_porous_air"]
    f = jnp.asarray([20.0, 730.0, 8000.0])
    theta = jnp.asarray([0.0, 30.0, 60.0, 75.0])

    def fn(q: jax.Array) -> tuple[jax.Array, ...]:
        return spectrum(c, f, air, q)

    direct = fn(theta)
    compiled = jax.jit(fn)(theta)
    mapped = tuple(
        jnp.moveaxis(v, 0, -1)
        for v in jax.vmap(lambda q: tuple(v[..., 0] for v in fn(q)))(theta)
    )
    rows = {
        "angles": {
            "pointwise": step2.encode(direct),
            "jit": step2.encode(compiled),
            "vmap": step2.encode(mapped),
        }
    }
    for name in ("field_custom_even", "paris_custom_even"):
        case = problem["cases"][name]
        q = angles(case)

        def integrate(theta: jax.Array) -> tuple[jax.Array, ...]:
            z = spectrum(case, f, air, theta)[0]
            field = tmm.field_impedance(z, theta)
            reflection, alpha = tmm.reflection_and_absorption_coefficient(
                field, air=air
            )
            return (
                field,
                reflection,
                alpha,
                tmm.diffuse_absorption_coefficient(z, theta, air=air),
            )

        batch = jnp.stack((q, q + 0.05))
        pointwise = tuple(jnp.stack([integrate(a)[i] for a in batch]) for i in range(4))
        rows[name] = {
            "pointwise": encode_integrated(pointwise),
            "jit": encode_integrated(jax.jit(jax.vmap(integrate))(batch)),
            "vmap": encode_integrated(jax.vmap(integrate)(batch)),
        }
    return {
        "rows": rows,
        "direct_shapes": [list(v.shape) for v in direct],
        "mapped_shapes": [list(v.shape) for v in mapped],
        "expected_shape": [f.size, theta.size],
    }


def determinants(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    rows = []
    thin = []
    for name, c in problem["cases"].items():
        if c.get("quantity") == "simpson" or c["layers"][0]["kind"] == "constant_z":
            continue
        for i, l in enumerate(c["layers"]):
            rows.append(
                step2.determinant_row(
                    matrix(l, jnp.asarray(c["frequencies_hz"]), air, angles(c)),
                    f"{name}::{i}",
                )
            )
            small = cast(step2.Layer, {**l, "thickness_m": 0.001})
            thin.append(
                step2.determinant_row(
                    matrix(small, jnp.asarray(c["frequencies_hz"]), air, angles(c)),
                    f"{name}::{i}::1mm",
                )
            )
    return {"rows": rows, "conditioned_rows": thin}


def validation(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    rejected = []
    for label, values in [
        ("negative", [-1.0]),
        ("grazing", [90.0]),
        ("over", [91.0]),
        ("nan", [np.nan]),
        ("infinity", [np.inf]),
        ("empty", []),
    ]:
        try:
            tmm.air_layer(
                jnp.asarray(500.0),
                jnp.asarray(0.03),
                air=air,
                incidence_angles_deg=jnp.asarray(values),
            )
        except ValueError:
            rejected.append(label)
    ranges = [
        [],
        [0.0, 78.0],
        [0.0, 78.0, 0.0],
        [0.0, 78.0, -1.0],
        [40.0, 30.0, 1.0],
        [-1.0, 78.0, 1.0],
        [0.0, 91.0, 1.0],
        [0.0, np.nan, 1.0],
    ]
    for values in ranges:
        try:
            tmm.diffuse_incidence_angles_deg(values)
        except ValueError:
            rejected.append(str(values))
    # 非法動態角度必須在編譯後也拒收；子行程捕捉 JAX 包裝後的執行錯誤。
    compiled = jax.jit(
        lambda q: tmm.air_layer(
            jnp.asarray(500.0), jnp.asarray(0.03), air=air, incidence_angles_deg=q
        )
    )
    try:
        compiled(jnp.asarray([90.0])).block_until_ready()
    except (ValueError, RuntimeError):
        rejected.append("jit_grazing")
    return {
        "rejected": rejected,
        "expected_rejected": [
            "negative",
            "grazing",
            "over",
            "nan",
            "infinity",
            "empty",
            *map(str, ranges),
            "jit_grazing",
        ],
        **grazing_and_zero_spacing(problem, air),
    }


def grazing_and_zero_spacing(
    problem: Problem, air: PhysicsConstants
) -> dict[str, object]:
    close = jnp.asarray([np.nextafter(90.0, 0.0)])
    finite = spectrum(
        problem["cases"]["oblique_air"], jnp.asarray([20.0, 8000.0]), air, close
    )

    def air_loss(q: jax.Array) -> jax.Array:
        z, _, _ = spectrum(problem["cases"]["oblique_air"], jnp.asarray(500.0), air, q)
        return jnp.sum(z.imag / air.rho_c)

    slope = jax.grad(air_loss)(jnp.asarray([89.99]))
    # 未選分支的0分母也不能污染Simpson導數。
    zero_grad = jax.grad(lambda q: tmm.simpson(jnp.ones(4), q))(jnp.zeros(4))
    return {
        "near_grazing_finite": all(bool(jnp.all(jnp.isfinite(v))) for v in finite),
        "near_grazing_gradient": np.asarray(slope).tolist(),
        "zero_spacing_gradient": np.asarray(zero_grad).tolist(),
    }


def boundaries(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    """形狀、互斥背牆與非數值範圍皆拒收，不能靠廣播無聲算另一題。"""
    f = jnp.asarray([20.0, 730.0, 8000.0])
    theta = jnp.asarray([30.0, 60.0])
    layer = tmm.air_layer(f, jnp.asarray(0.03), air=air, incidence_angles_deg=theta)
    normal = tmm.air_layer(f, jnp.asarray(0.03), air=air)
    calls: dict[str, Callable[[], object]] = {
        "missing_angle_axis": lambda: tmm.compute(
            [normal], air=air, incidence_angles_deg=theta
        ),
        "mismatched_angle_axis": lambda: tmm.compute(
            [layer], air=air, incidence_angles_deg=jnp.asarray([30.0, 60.0, 75.0])
        ),
        "bad_method": lambda: tmm.diffuse_incidence_angles_deg(method="bad"),
        "nonnumeric_range": lambda: tmm.diffuse_incidence_angles_deg(
            cast(list[float], ["bad", 78.0, 1.0])
        ),
        "bad_backing": lambda: tmm.compute(
            [layer], air=air, incidence_angles_deg=theta, backing="bad"
        ),
        "conflicting_load": lambda: tmm.compute(
            [layer],
            air=air,
            incidence_angles_deg=theta,
            backing="air",
            load_impedance_pa_s_m=200.0,
        ),
        "radiation_without_frequency": lambda: tmm.compute(
            [layer], air=air, incidence_angles_deg=theta, backing="radiation"
        ),
        "zero_area": lambda: tmm.compute(
            [layer],
            air=air,
            incidence_angles_deg=theta,
            backing="air",
            radiation_area_m2=0.0,
        ),
        "negative_area": lambda: tmm.unflanged_pipe_radiation_load(
            f, air=air, radiation_area_m2=-1.0
        ),
        "field_singleton": lambda: tmm.field_impedance(
            jnp.asarray([300.0]), jnp.asarray([30.0])
        ),
        "field_shape": lambda: tmm.field_impedance(jnp.ones((3, 3)), theta),
        "field_repeated": lambda: tmm.field_impedance(
            jnp.ones((3, 2)), jnp.asarray([30.0, 30.0])
        ),
        "paris_decreasing": lambda: tmm.diffuse_absorption_coefficient(
            jnp.ones((3, 2)), jnp.asarray([60.0, 30.0]), air=air
        ),
        "simpson_shape": lambda: tmm.simpson(jnp.ones((3, 3)), theta),
        "trapezoid_shape": lambda: tmm.trapezoidal_integral(jnp.ones((3, 3)), theta),
    }
    rejected = []
    for label, fn in calls.items():
        try:
            fn()
        except ValueError:
            rejected.append(label)
    return {
        "rejected": rejected,
        "expected_rejected": list(calls),
        **matched_air_backing(air),
    }


def matched_air_backing(air: PhysicsConstants) -> dict[str, object]:
    f = jnp.asarray([20.0, 730.0, 8000.0])
    # 純空氣接匹配負載的獨立真值，包含近掠角，亦驗自訂隨角度負載。
    close = jnp.asarray([0.0, 30.0, 60.0, 75.0, 89.99, np.nextafter(90.0, 0.0)])
    m = tmm.air_layer(f, jnp.asarray(0.03), air=air, incidence_angles_deg=close)
    z, r, a = tmm.compute([m], air=air, incidence_angles_deg=close, backing="air")
    load = air.rho_c / jnp.cos(jnp.deg2rad(close))
    explicit = tmm.compute(
        [m], air=air, incidence_angles_deg=close, load_impedance_pa_s_m=load
    )
    return {
        "actual": step2.encode((z, r, a)),
        "explicit": step2.encode(explicit),
        "expected": step2.encode(
            (jnp.broadcast_to(load, z.shape), jnp.zeros_like(r), jnp.ones_like(a))
        ),
        "matrix_shape": list(m.shape),
        "expected_matrix_shape": [*f.shape, close.size, 2, 2],
    }


def normal_path(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    """堵住角度入口後跑原考卷；只比較數字不能證明沒換算法。"""

    def forbidden(incidence_angles_deg: jax.Array | float) -> jax.Array:
        raise AssertionError("不給角度卻進入斜入射算法")

    tmm._angles_radians = forbidden
    root = Path(sys.argv[2]).parent
    p1 = cast(
        step1.Problem,
        json.loads((root / "tmm_reference_problem_step1.json").read_text()),
    )
    p2 = cast(
        step2.Problem,
        json.loads((root / "tmm_reference_problem_step2.json").read_text()),
    )
    return {"step1": step1.reference(p1, air), "step2": step2.reference(p2, air)}


def trapezoid(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    """公開的梯形積分直接對 NumPy 的 trapezoid；Paris 是比值，少除 2 會被分子分母約掉，只靠它看不出來。"""
    x = np.asarray([0.0, 0.1, 0.35, 0.4, 0.9, 1.3, 1.5708])
    rows = []
    for name, y in (("real", np.cos(3 * x) + x**2), ("complex", np.exp(1j * 2 * x) / (1 + x)),
                    ("batched", np.stack((np.sin(x), x * np.cos(x))))):
        actual = np.asarray(tmm.trapezoidal_integral(jnp.asarray(y), jnp.asarray(x)))
        expected = np.trapezoid(y, x, axis=-1)
        rows.append({"name": name, "actual": [complex(v).real.hex() for v in np.ravel(actual)]
                     + [complex(v).imag.hex() for v in np.ravel(actual)],
                     "expected": [complex(v).real.hex() for v in np.ravel(expected)]
                     + [complex(v).imag.hex() for v in np.ravel(expected)]})
    return {"rows": rows, "dtype": str(jnp.asarray(0.0).dtype)}


def matrix_values(problem: Problem, air: PhysicsConstants) -> dict[str, object]:
    """直接看矩陣元素：整層乘−1會被A/C、(AZ+B)/(CZ+D)與行列式一起約掉，只有元素本身看得到。"""
    f = jnp.asarray([20.0, 730.0, 8000.0])

    def row(name: str, actual: jax.Array, expected: np.ndarray) -> dict[str, object]:
        a, e = np.ravel(np.asarray(actual)), np.ravel(np.broadcast_to(expected, np.shape(actual)))
        return {"name": name, "actual_real": step2.hexes(a.real), "actual_imag": step2.hexes(a.imag),
                "expected_real": step2.hexes(e.real), "expected_imag": step2.hexes(e.imag)}

    identity, sheets, air_rows, pore_rows = [], [], [], []
    for label, theta in (("normal", None), ("oblique", jnp.asarray([0.0, 37.0, 75.0]))):
        for name, c in problem["cases"].items():
            if c.get("quantity") == "simpson" or c["layers"][0]["kind"] == "constant_z":
                continue
            for i, l in enumerate(c["layers"]):
                where = f"{name}::{i}::{label}"
                if l["kind"] in ("air", "porous"):
                    # 零厚度流體層：cos 0＝1、sin 0＝0，必須正好是單位矩陣。
                    zero = cast(step2.Layer, {**l, "thickness_m": 0.0})
                    identity.append(row(where, matrix(zero, f, air, theta), np.eye(2)))
                elif l["kind"] == "perforated" and l["method"] == "eq_fluid":
                    # 孔道含末端修正，零厚度不是單位矩陣；改看 1 Hz 的零頻極限：相位→0，A、D→1。
                    m = matrix(l, jnp.asarray([1.0]), air, theta)
                    pore_rows.append(row(where, m[..., (0, 1), (0, 1)], np.asarray([1.0, 1.0])))
                else:
                    # 薄片 [[1,Z],[0,1]]：Z 已由對答案考卷咬，這裡看另外三格。
                    m = matrix(l, f, air, theta)
                    sheets.append(row(where, m[..., (0, 1, 1), (0, 0, 1)], np.asarray([1.0, 0.0, 1.0])))
        cosine = np.ones(1) if theta is None else np.cos(np.deg2rad(np.asarray(theta)))
        kz = 2 * np.pi * np.asarray(f)[:, None] / air.sound_speed_m_s * cosine
        zc = air.rho_c / cosine
        phase = kz * 0.03
        expected = np.stack((np.stack((np.cos(phase), 1j * zc * np.sin(phase)), axis=-1),
                             np.stack((1j * np.sin(phase) / zc, np.cos(phase)), axis=-1)), axis=-2)
        actual = tmm.air_layer(f, jnp.asarray(0.03), air=air, incidence_angles_deg=theta)
        air_rows.append(row(f"air::0.03::{label}", actual, expected if theta is not None else expected[:, 0]))
    return {"identity_rows": identity, "sheet_rows": sheets, "air_rows": air_rows,
            "pore_limit_rows": pore_rows, "dtype": str(jnp.asarray(0.0).dtype)}


def main() -> None:
    mode, problem_file, output_file = sys.argv[1:]
    p = cast(Problem, json.loads(Path(problem_file).read_text()))
    base = load_physics_constants(config_path("physics_constants.toml"))
    air = PhysicsConstants.model_validate(
        {
            **base.model_dump(),
            **{name: prop["value"] for name, prop in p["air_properties"].items()},
        }
    )
    actions = {
        "reference": reference,
        "trapezoid": trapezoid,
        "matrix_values": matrix_values,
        "gradients": gradients,
        "transforms": transforms,
        "determinants": determinants,
        "validation": validation,
        "normal_path": normal_path,
        "boundaries": boundaries,
    }
    Path(output_file).write_text(json.dumps(actions[mode](p, air), allow_nan=False))


if __name__ == "__main__":
    main()
