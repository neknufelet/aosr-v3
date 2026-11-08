"""鏡像法早期三欄的可微陣列核心；幾何、指向性、家具都由中央處理器雙精度先算。

面片代號沿用 gradient.boundary 的「牆名@區名」，順序為 Wall.all、同牆輸入區序；物理層不引用梯度層。
build_geometric_early_table 只建立常數，不求材料。solve_geometric_early 收逐片 (F,A) 斜入射阻抗、
(N,F) 垂直入射阻抗與 (N,) 或 (N,F) 常數散射；輸出各 (聲源,座位,F)。晚期能量留給下一段。
幾何輸入錯誤在編譯前拒收；編譯中材料反射分母為零、非有限或散射越界回旗標及 NaN。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import amplitude_jax as aj
from aosr.physics.furniture_scene import FurnitureLaneInputs
from aosr.physics.geometric_lane import _add_furniture_pressure, _unblocked_room_paths, _wall_areas
from aosr.physics.report_source import SourceModelSpec, directivity_to_apply
from aosr.physics.report_surfaces import RectangleZone, validate_surface_partition
from aosr.physics.room_paths import image_source_paths
from aosr.physics.source_directivity import SourceModel, apply_pressure_factor


@dataclass(frozen=True, eq=False)
class EarlyPairTable:
    """一組聲源座位的路徑、逐頻指向倍率及不乘房間散射的家具聲壓。"""

    paths: aj.PathTable
    directivity: NDArray[np.float64]
    furniture_pressure: NDArray[np.complex128]


@dataclass(frozen=True, eq=False)
class GeometricEarlyTable:
    """按聲源、座位排的常數；每組路徑共享全房的面片角清單，呼叫端只須求一次逐角阻抗。"""

    pairs: tuple[EarlyPairTable, ...]
    shape: tuple[int, int, int]
    patch_areas_m2: NDArray[np.float64]
    rho_c: float
    reflection_order_k: int

    @property
    def patch_ids(self) -> tuple[str, ...]:
        return self.pairs[0].paths.patch_ids

    @property
    def incidence_cosines(self) -> tuple[NDArray[np.float64], ...]:
        return self.pairs[0].paths.incidence_cosines

    @property
    def incidence_angles_deg(self) -> tuple[NDArray[np.float64], ...]:
        return self.pairs[0].paths.incidence_angles_deg


class GeometricEarlyJaxResult(NamedTuple):
    """早期三欄、s(f) 與有效旗標，各 (聲源,座位,F)；有效項不被無效頻點的反向傳播污染。"""

    direct_energy: jax.Array
    reflected_energy: jax.Array
    interference_energy: jax.Array
    scattering: jax.Array
    reflection_valid: jax.Array
    scattering_valid: jax.Array
    valid: jax.Array


def _pair_table(
    room: Room, source: Point, receiver: Point, model: SourceModelSpec, c: float, rho_c: float,
    axis: tuple[float, ...], order: int, zones: Mapping[Wall, tuple[RectangleZone, ...]] | None,
    furniture: FurnitureLaneInputs | None,
) -> EarlyPairTable:
    paths = image_source_paths(room, source, receiver, c, max_order=order)
    if furniture is not None:
        paths = list(_unblocked_room_paths(tuple(paths), source, receiver, furniture))
    table = aj.build_path_table(paths, receiver.as_tuple(), axis, rectangle_zones_by_wall=zones)
    table.direct_index()
    directivity = np.ones((len(paths), len(axis)), dtype=np.float64)
    directed = directivity_to_apply(model)
    if directed is not None:
        units = [replace(p, path_pressure=(1.+0j,)*len(axis)) for p in paths]
        scaled = apply_pressure_factor(units, receiver, axis, SourceModel.TWO_PARAMETER,
            source=source, aim=directed[1], params=directed[0])
        assert all(z.imag == 0. for p in scaled for z in p.path_pressure), "指向性倍率須為實數"
        directivity = np.asarray([[z.real for z in p.path_pressure] for p in scaled], dtype=np.float64)
    pressure = (0j,)*len(axis)
    if furniture is not None:
        pressure = _add_furniture_pressure(pressure, furniture, source, receiver, axis, rho_c, c, model)
    return EarlyPairTable(table, directivity, np.asarray(pressure, dtype=np.complex128))


def _shared_angles(pairs: Sequence[EarlyPairTable]) -> tuple[EarlyPairTable, ...]:
    """各對按首現去重後合併，cos 本身與其 Python 雙精度角度不重新計算。"""
    cosines, angles = [], []
    for patch in range(len(pairs[0].paths.patch_ids)):
        lookup = {float(c): float(a) for pair in pairs for c, a in zip(
            pair.paths.incidence_cosines[patch], pair.paths.incidence_angles_deg[patch], strict=True)}
        cosines.append(np.asarray(tuple(lookup), dtype=np.float64))
        angles.append(np.asarray(tuple(lookup.values()), dtype=np.float64))
    result = []
    for pair in pairs:
        table = pair.paths
        indices = np.zeros_like(table.bounce_angle)
        for patch, cos in enumerate(cosines):
            lookup_index = {float(c): i for i, c in enumerate(cos)}
            hits = table.bounce_mask & (table.bounce_patch == patch)
            indices[hits] = [lookup_index[float(c)] for c in table.bounce_cos[hits]]
        result.append(replace(pair, paths=replace(table, bounce_angle=indices,
            incidence_cosines=tuple(cosines), incidence_angles_deg=tuple(angles))))
    return tuple(result)


def build_geometric_early_table(
    *, room: Room, sources: Sequence[Point], receivers: Sequence[Point], source_models: Sequence[SourceModelSpec],
    sound_speed_m_s: float, rho_c_pa_s_per_m: float,
    frequencies_hz: tuple[float, ...] = GEOMETRIC_LANE_FREQUENCIES_HZ,
    reflection_order_k: int = REFLECTION_ORDER_K,
    rectangle_zones_by_wall: Mapping[Wall, tuple[RectangleZone, ...]] | None = None,
    furniture: FurnitureLaneInputs | None = None,
) -> GeometricEarlyTable:
    """只算幾何常數；保留既有被擋直達拒收與矩形覆蓋檢查，不連接材料鏈。"""
    if not sources or not receivers or len(source_models) != len(sources):
        raise ValueError("聲源、座位不可空，聲源模型須逐聲源提供")
    if not np.isfinite(rho_c_pa_s_per_m) or rho_c_pa_s_per_m <= 0:
        raise ValueError("rho_c 必須是有限正數")
    zones = rectangle_zones_by_wall or {}
    validate_surface_partition(room, {w.wall_name(): cells for w, cells in zones.items()})
    areas = _wall_areas(room)
    patch_areas = np.asarray([((z.u_m[1]-z.u_m[0])*(z.v_m[1]-z.v_m[0]) if z else areas[w.wall_name()])
        for w in Wall.all() for z in (zones.get(w) or (None,))], dtype=np.float64)
    pairs = [_pair_table(room, source, receiver, model, sound_speed_m_s, rho_c_pa_s_per_m,
        frequencies_hz, reflection_order_k, rectangle_zones_by_wall, furniture)
        for source, model in zip(sources, source_models, strict=True) for receiver in receivers]
    return GeometricEarlyTable(_shared_angles(pairs), (len(sources), len(receivers), len(frequencies_hz)),
        patch_areas, rho_c_pa_s_per_m, reflection_order_k)


def room_scattering(
    table: GeometricEarlyTable, normal_impedance: jax.Array, scattering_by_patch: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """面積×|R(f,0)|² 加權；全吸收時恰零，所有散射相同時保留輸入。"""
    z = aj._complex(jnp.asarray(normal_impedance))
    dtype = aj._real_dtype(z)
    s = jnp.asarray(scattering_by_patch, dtype=dtype)
    if s.ndim == 1:
        s = s[:, None]
    valid_z = jnp.isfinite(z) & (z + table.rho_c != 0)
    safe_z = jnp.where(valid_z, z, table.rho_c)
    r = (safe_z-table.rho_c)/(safe_z+table.rho_c)
    valid_s = jnp.isfinite(s) & (s >= 0) & (s <= 1)
    safe_s = jnp.where(valid_s, s, 0.)
    weights = jnp.asarray(table.patch_areas_m2, dtype=dtype)[:, None] * jnp.abs(r)**2
    denominator = jnp.sum(weights, axis=0)
    safe_denominator = jnp.where(denominator != 0, denominator, 1.)
    weighted = jnp.sum(weights*safe_s, axis=0)/safe_denominator
    value = jnp.where(denominator == 0, 0., jnp.where(jnp.all(safe_s == safe_s[:1], axis=0), safe_s[0], weighted))
    valid = jnp.all(valid_z & valid_s, axis=0) & jnp.isfinite(value)
    return jnp.where(valid, value, jnp.nan), valid


def _pair_pressure(pair: EarlyPairTable, rows: Sequence[jax.Array], rho_c: float) -> tuple[jax.Array, jax.Array]:
    table = pair.paths
    z = aj.bounce_impedances_by_patch(table, rows)
    mask = jnp.asarray(table.bounce_mask)[:, :, None]
    cos = jnp.asarray(table.bounce_cos, dtype=aj._real_dtype(z))[:, :, None]
    valid = jnp.isfinite(z) & (z*cos+rho_c != 0)
    safe_z = jnp.where(mask & valid, z, rho_c)
    pressure = aj.path_pressures(table, aj.reflection_products(table, safe_z, rho_c))
    return pressure*jnp.asarray(pair.directivity, dtype=aj._real_dtype(z)), jnp.all(~mask | valid, axis=(0, 1))


def solve_geometric_early(
    table: GeometricEarlyTable, impedance_by_patch: Sequence[jax.Array], normal_impedance: jax.Array,
    scattering_by_patch: jax.Array,
) -> GeometricEarlyJaxResult:
    """陣列入口：逐反彈連乘、逐階相干和與散射縮放；可編譯及對阻抗微分。"""
    scattering, scattering_valid = room_scattering(table, normal_impedance, scattering_by_patch)
    safe_s = jnp.where(scattering_valid, scattering, 0.)
    retained = 1.-safe_s
    # 零保留邊界取零切線；內層避開零底數的開根號，外層保留原來的零值。
    zero_retained = retained == 0.
    safe_retained = jnp.where(zero_retained, 1., retained)
    direct, reflected, interference, reflection_valid = [], [], [], []
    for pair in table.pairs:
        pressure, valid = _pair_pressure(pair, impedance_by_patch, table.rho_c)
        pd = pressure[pair.paths.direct_index()]
        pr = jnp.zeros_like(pd)
        for order in range(1, table.reflection_order_k+1):
            selected = jnp.asarray(pair.paths.order == order)[:, None]
            pk = jnp.sum(jnp.where(selected, pressure, jnp.zeros((), dtype=pressure.dtype)), axis=0)
            scale = jnp.where(zero_retained, 0., safe_retained**(order/2.))
            pr = pr + scale*pk
        pr = pr+jnp.asarray(pair.furniture_pressure, dtype=pressure.dtype)
        reflection_valid.append(valid)
        valid = valid & scattering_valid
        direct.append(jnp.abs(pd)**2)
        reflected.append(jnp.where(valid, jnp.abs(pr)**2, jnp.nan))
        interference.append(jnp.where(valid, 2.*jnp.real(pd*jnp.conj(pr)), jnp.nan))
    def pack(rows: list[jax.Array]) -> jax.Array:
        return jnp.stack(rows).reshape(table.shape)
    s_valid = jnp.broadcast_to(scattering_valid, table.shape)
    r_valid = pack(reflection_valid)
    return GeometricEarlyJaxResult(pack(direct), pack(reflected), pack(interference),
        jnp.where(r_valid, jnp.broadcast_to(scattering, table.shape), jnp.nan), r_valid, s_valid, r_valid & s_valid)


# 固定幾何的組裝器也需要建逐對表與合併入射角；原私名保留相容。
build_early_pair_table = _pair_table
shared_incidence_angles = _shared_angles
