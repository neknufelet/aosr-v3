"""JAX 鏡像振幅的隔離雙精度探針（#562）：父行程開一個雙精度子行程跑這支，結果寫成 JSON 交回父行程判。

同程式的考卷跑在 JAX 預設精度（雙精度關），雙精度輸入會被悄悄降成單精度，所以「JAX 雙精度對現有實作」
只能在這裡量。這支只量、只回原始數字（十六進位浮點字串）與幾個是非，契約裁判住父行程
（``tests/engine/test_amplitude_jax.py``），門檻只從登記簿讀。
"""
from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

import aosr
import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, PorousLayer, SurfaceMaterial
from aosr.physics import amplitude as amp
from aosr.physics import amplitude_jax as aj
from aosr.physics.room_paths import RoomPath, image_source_paths, load_room_input
from aosr.physics.totals import totals_from_paths
from aosr.physics.wall_reflections import wall_reflection_sources

ROOM, SOURCE, RECEIVER = Room(6.0, 4.0, 3.0), Point(1.1, 0.9, 1.2), Point(3.7, 2.1, 1.45)
RHO, C = 1.19, 347.0
AXIS = (125.0, 251.0, 503.0, 1009.0, 2017.0, 4031.0)
BOARD = Assembly(layers=(BoardLayer(surface_mass_kg_m2=0.15, resistance_pa_s_m=80.0), AirLayer(thickness_m=0.43)),
                 backing="rigid")
POROUS = Assembly(layers=(PorousLayer(thickness_m=0.05, flow_resistivity_pa_s_m2=12000.0, material_class="fibrous"),
                          AirLayer(thickness_m=0.17)), backing="rigid")



def _hex_complex(values: np.ndarray) -> list[object]:
    """任意形狀的複數陣列 → 同形巢狀串列，每格 ``[實部 hex, 虛部 hex]``（雙精度，單精度先無損升格）。"""
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 0:
        return [float(array.real).hex(), float(array.imag).hex()]
    return [_hex_complex(row) for row in array]


def _hex_real(values: np.ndarray) -> list[object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array).hex()  # type: ignore[return-value]  # expires=2027-01-31 reason=遞迴底層回單一字串
    return [_hex_real(row) for row in array]


def _result(values: aj.JaxAmplitudes) -> dict[str, object]:
    return {"refl": _hex_complex(np.asarray(values.reflection_product)),
            "pressure": _hex_complex(np.asarray(values.path_pressure)),
            "total": _hex_complex(np.asarray(values.total_pressure)),
            "direct_energy": _hex_real(np.asarray(values.direct_energy)),
            "reflected_energy": _hex_real(np.asarray(values.reflected_energy)),
            "dtypes": [str(np.asarray(v).dtype) for v in values]}


def _python(paths: list[RoomPath]) -> dict[str, object]:
    totals = totals_from_paths(paths)
    return {"refl": _hex_complex(np.array([p.reflection_product for p in paths])),
            "pressure": _hex_complex(np.array([p.path_pressure for p in paths])),
            "total": _hex_complex(np.array(totals.pressure)),
            "direct_energy": _hex_real(np.array(totals.direct_energy)),
            "reflected_energy": _hex_real(np.array(totals.reflected_energy))}


Case = tuple[list[RoomPath], tuple[float, float, float], np.ndarray, float, tuple[float, ...]]


def _reference_case(case: str, directory: Path) -> Case:
    """三階 63 條的兩組振幅答案輸入（flat／varied），走現有載入器與路徑程式。"""
    from tests.engine.test_amplitude import _input_case

    path = directory / f"room-{case}.json"
    path.write_text(json.dumps(_input_case(case), sort_keys=True), encoding="utf-8")
    inp = load_room_input(path)
    assert inp.materials is not None
    paths = image_source_paths(inp.room, inp.source, inp.receiver, inp.sound_speed, inp.max_order, inp.materials)
    z = np.array([inp.materials.walls[w] for w in amp.CANONICAL_WALLS], dtype=np.complex128)
    return paths, inp.receiver.as_tuple(), z, inp.materials.rho_c, inp.materials.frequencies_hz


def _dense_case() -> Case:
    """幾何路那條 220 點的軸，六面牆各一列隨頻率變的複數阻抗；ωτ 最大到幾百弧度，量單精度相位誤差。"""
    rho_c = RHO * C
    freqs = np.asarray(GEOMETRIC_LANE_FREQUENCIES_HZ)
    walls = {wall: tuple(complex(rho_c * (2.0 + 0.7 * i), rho_c * (0.3 * i - 1.0) * 500.0 / f) for f in freqs)
             for i, wall in enumerate(amp.CANONICAL_WALLS)}
    materials = amp.Materials(rho_c, GEOMETRIC_LANE_FREQUENCIES_HZ, walls)
    paths = image_source_paths(ROOM, SOURCE, RECEIVER, C, 3, materials)
    z = np.array([walls[w] for w in amp.CANONICAL_WALLS], dtype=np.complex128)
    return paths, RECEIVER.as_tuple(), z, rho_c, GEOMETRIC_LANE_FREQUENCIES_HZ


def _construction_case() -> tuple[list[RoomPath], aj.PathTable, np.ndarray, float]:
    """兩面構造牆（板＋空腔、多孔＋空腔）的 Z(f, θ)，其餘固定；對照現有 WallReflectionSources.apply。"""
    materials: dict[Wall, SurfaceMaterial] = {w: ConstantImpedance(impedance_pa_s_m=900.0 + 173.0 * i)
                                             for i, w in enumerate(Wall.all())}
    materials[Wall.X0], materials[Wall.YL] = BOARD, POROUS
    sources = wall_reflection_sources(materials, AXIS, density_kg_m3=RHO, sound_speed_m_s=C)
    paths = image_source_paths(ROOM, SOURCE, RECEIVER, C, 3, reflection_sources=sources)
    table = aj.build_path_table(paths, RECEIVER.as_tuple(), AXIS)
    z_bounce = aj.bounce_impedances_from_sources(table, paths, RECEIVER.as_tuple(), sources)
    return paths, table, z_bounce, sources.fixed_materials.rho_c


def _same_bytes(a: aj.JaxAmplitudes, b: aj.JaxAmplitudes) -> bool:
    return all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b, strict=True))


def _transforms(table: aj.PathTable, z_bounce: jax.Array, rho_c: float) -> dict[str, bool]:
    """eager（逐運算）、jit、vmap 三種跑法逐位比；vmap 一批三組阻抗，逐組對 eager。"""
    eager = aj.amplitudes(table, z_bounce, rho_c)
    jitted = jax.jit(lambda z: aj.amplitudes(table, z, rho_c))(z_bounce)
    batch = jnp.stack((z_bounce, z_bounce * 1.1, z_bounce * (1.0 - 0.2j)))
    mapped = jax.vmap(lambda z: aj.amplitudes(table, z, rho_c))(batch)
    eager_each = [aj.amplitudes(table, batch[i], rho_c) for i in range(3)]
    jit_mapped = jax.jit(jax.vmap(lambda z: aj.amplitudes(table, z, rho_c)))(batch)
    return {
        "jit_equals_eager": _same_bytes(eager, jitted),
        "vmap_equals_eager": all(_same_bytes(aj.JaxAmplitudes(*(v[i] for v in mapped)), eager_each[i]) for i in range(3)),
        "jit_vmap_equals_eager": all(_same_bytes(aj.JaxAmplitudes(*(v[i] for v in jit_mapped)), eager_each[i])
                                     for i in range(3)),
    }


def _gradient(objective: Callable[[jax.Array], jax.Array], z: np.ndarray, picks: Sequence[tuple[int, ...]]) -> list[list[float]]:
    """``jax.grad`` 對複數 Z 給的是 ∂f/∂x − i·∂f/∂y；逐格回 [梯度實部, −梯度虛部, 中央差分 x, 中央差分 y]。"""
    grad = np.asarray(jax.grad(objective)(jnp.asarray(z)))
    rows = []
    for index in picks:
        step = 1e-6 * abs(z[index])
        diffs = []
        for direction in (1.0, 1j):
            plus, minus = z.copy(), z.copy()
            plus[index] += step * direction
            minus[index] -= step * direction
            diffs.append(float((objective(jnp.asarray(plus)) - objective(jnp.asarray(minus))) / (2.0 * step)))
        rows.append([float(grad[index].real), float(-grad[index].imag), *diffs])
    return rows


def _measure(directory: Path) -> dict[str, object]:
    out: dict[str, object] = {"x64": bool(getattr(jax.config, "jax_enable_x64"))}
    for name in ("flat", "varied", "dense"):
        paths, receiver, z_wall, rho_c, axis = _reference_case(name, directory) if name != "dense" else _dense_case()
        table = aj.build_path_table(paths, receiver, axis)
        z_bounce = aj.bounce_impedances(table, jnp.asarray(z_wall))
        record: dict[str, object] = {
            "frequencies": _hex_real(table.frequencies_hz), "delay": _hex_real(table.delay_s),
            "order": table.order.tolist(), "python": _python(paths),
            "jax64": _result(aj.amplitudes(table, z_bounce, rho_c)),
            "jax32": _result(aj.amplitudes(table, z_bounce.astype(jnp.complex64), rho_c)),
            "transforms64": _transforms(table, z_bounce, rho_c),
            "transforms32": _transforms(table, z_bounce.astype(jnp.complex64), rho_c),
        }
        if name == "varied":
            def energy(z: jax.Array) -> jax.Array:
                result = aj.amplitudes(table, aj.bounce_impedances(table, z), rho_c)
                return jnp.sum(jnp.abs(result.total_pressure) ** 2) + jnp.sum(result.reflected_energy)
            picks = [(w, k) for w in range(z_wall.shape[0]) for k in range(z_wall.shape[1])]
            record["gradient"] = _gradient(energy, z_wall, picks)
        out[name] = record
    paths, table, z_bounce_np, rho_c = _construction_case()
    z_bounce = jnp.asarray(z_bounce_np)

    def construction_energy(z: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(aj.amplitudes(table, z, rho_c).total_pressure) ** 2)

    hit = [(int(i), int(j)) for i, j in np.argwhere(table.bounce_mask)]
    bounce_picks = [(i, j, k) for i, j in hit[:: max(1, len(hit) // 8)] for k in (0, len(AXIS) - 1)]
    out["construction"] = {
        "frequencies": _hex_real(table.frequencies_hz), "delay": _hex_real(table.delay_s),
        "order": table.order.tolist(), "python": _python(paths),
        "jax64": _result(aj.amplitudes(table, z_bounce, rho_c)),
        "jax32": _result(aj.amplitudes(table, z_bounce.astype(jnp.complex64), rho_c)),
        "transforms64": _transforms(table, z_bounce, rho_c),
        "transforms32": _transforms(table, z_bounce.astype(jnp.complex64), rho_c),
        "gradient": _gradient(construction_energy, z_bounce_np, bounce_picks),
    }
    return out


if __name__ == "__main__":
    assert Path(str(aosr.__file__)).resolve() == Path(sys.argv[1]).resolve(), aosr.__file__
    target = Path(sys.argv[2])
    try:
        payload: dict[str, object] = {"error": "", "data": _measure(target.parent)}
    except Exception:
        payload = {"error": traceback.format_exc(), "data": {}}
    target.write_text(json.dumps(payload), encoding="utf-8")
