"""鏡像法反射乘積、路徑壓力與總量的 JAX（可微分的陣列算數函式庫）版，給第三階段改牆面阻抗用（#562）。

**幾何是常數，只有阻抗會動。** 決策紙 ``optimization-stages-room-with-positions-then-layout-then-zoned-materials.md``
表二：第三階段房間幾何與擺位不變，只改表面阻抗，所以路徑表（每條路徑的距離、到達時間、每次反彈打哪面牆與
入射角餘弦）由 :func:`build_path_table` 用 CPU 雙精度從現有路徑程式算好、打包成 numpy 陣列常數；相位因子
``exp(−i·2πf·τ)`` 與 ``1/dist`` 也只跟幾何有關，同樣在這裡用跟 :func:`aosr.physics.amplitude.path_pressure`
同一種寫法（``cmath.exp``）先算好。JAX 那一段只做「Z → 反射係數 → 逐反彈連乘 → 乘上幾何常數 → 加總」，
對 Z 可以 ``jax.grad``、``jax.jit``、``jax.vmap``。

**公式照現有實作，不另發明。** 反射係數 ``R = (Z·cosθ − ρc)/(Z·cosθ + ρc)``，cosθ 照攤開直線的定義
（:func:`aosr.physics.amplitude.incidence_cos`）；同一面牆打 ``count`` 次就展開成 ``count`` 次反彈、各乘一次
（等於現有的 ``r ** count``，差在捨入）；直達路徑沒有反彈，乘積恰是 ``1+0j``。壓力
``(1/dist)·refl·exp(−iωτ)``；總壓力是全部路徑相加、直達能量是直達那一條的模平方、反射能量是其餘路徑
先複數相加再取模平方（:func:`aosr.physics.totals.totals_from_paths` 那一層）。

**精度跟著輸入走。** 阻抗是 ``complex64`` 就整段單精度（幾何常數在進場時轉成同一精度），``complex128`` 就
雙精度；這一支不碰 JAX 的全域設定（那件事只住 ``aosr.runtime``，規矩卡 ``layers-import-downward-only``）。
行程沒開雙精度時 JAX 會把雙精度輸入降成單精度，那是行程設定的事，這一支照收到的 dtype 算。

**阻抗從哪來。** 每面牆一列與角度無關的 Z(f)：用 :func:`bounce_impedances` 從 ``(…, 6, F)`` 收成逐反彈
``(…, P, B, F)``（這一步在 JAX 裡，梯度穿得過）。構造牆的 Z(f, θ)：用 :func:`bounce_impedances_from_sources`
在 CPU 上照 :class:`aosr.physics.wall_reflections.WallReflectionSources` 同一條路（同一組去重後的 cos、同一個
求值函式）求好再交進來。分格材料（逐反彈查格）不在這一支的範圍，路徑表遇到就拒收。

**這一支沒有任何現有程式引用。** 它不在物理身分的引用閉包裡，固定牆那條路也照舊不載入 JAX。
"""
from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.geometry.shoebox import Wall, wall_count_signature
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics import amplitude as amp
from aosr.physics.wall_surfaces import wall_path_impedances

if TYPE_CHECKING:
    from aosr.physics.room_paths import RoomPath
    from aosr.physics.wall_reflections import WallReflectionSources


@dataclass(frozen=True, eq=False)
class PathTable:
    """固定路徑表：全是 CPU 雙精度算好的 numpy 常數，``P`` 條路徑、每條最多 ``B`` 次反彈、``F`` 個頻點。

    ``bounce_wall`` 是牆在 :data:`aosr.physics.amplitude.CANONICAL_WALLS` 裡的序號；沒用到的反彈格
    ``bounce_mask`` 為 False，牆序號填 0、cos 填 1；那幾格的 Z 在 :func:`reflection_products` 裡先換成 ρc 再算，
    所以不管呼叫端在那裡填什麼，前向與梯度都有限、梯度恰為 0。
    ``inverse_distance`` 是 ``1/dist``、``phase`` 是 ``exp(−i·2πf·τ)``，跟現有壓力算式同一種寫法先算好。
    要 ``jax.jit`` 就用閉包把整張表帶進去（陣列成為編譯常數），不要當成要追蹤的參數。
    """

    frequencies_hz: NDArray[np.float64]
    order: NDArray[np.int64]
    dist_m: NDArray[np.float64]
    delay_s: NDArray[np.float64]
    bounce_wall: NDArray[np.int64]
    bounce_cos: NDArray[np.float64]
    bounce_mask: NDArray[np.bool_]
    inverse_distance: NDArray[np.float64]
    phase: NDArray[np.complex128]

    @property
    def path_count(self) -> int:
        return int(self.dist_m.shape[0])

    @property
    def bounce_count(self) -> int:
        return int(self.bounce_wall.shape[1])

    def direct_index(self) -> int:
        """恰好一條 order 0 的直達路徑；不是一條就 ValueError（跟總量那一層同一條規矩）。"""
        direct = np.flatnonzero(self.order == 0)
        if direct.size != 1:
            raise ValueError(f"order 0 的直達路徑不是恰好一條，是 {direct.size} 條")
        return int(direct[0])


class JaxAmplitudes(NamedTuple):
    """一次算完的結果：逐路徑反射乘積與壓力 ``(…, P, F)``，以及每頻點的總壓力與兩種能量 ``(…, F)``。"""

    reflection_product: jax.Array
    path_pressure: jax.Array
    total_pressure: jax.Array
    direct_energy: jax.Array
    reflected_energy: jax.Array


def build_path_table(
    paths: Sequence[RoomPath],
    receiver: tuple[float, float, float],
    frequencies_hz: Sequence[float],
) -> PathTable:
    """從現有路徑程式的結果打包路徑表；cos 與相位都照現有寫法用 Python 雙精度算。"""
    if not paths:
        raise ValueError("沒有路徑，打包不出路徑表")
    freqs = tuple(float(f) for f in frequencies_hz)
    if not freqs or any(not math.isfinite(f) or f <= 0 for f in freqs):
        raise ValueError("頻率軸必須非空且全為有限正數")
    per_path: list[list[tuple[int, float]]] = []
    for path in paths:
        if path.has_patch_materials:
            raise ValueError(f"路徑 {path.identity!r} 帶分格材料；逐反彈查格不在這一支的範圍")
        if path.dist_m == 0.0:
            raise ValueError("接收點落在鏡像聲源上（dist_m 為 0），壓力沒定義")
        counts = wall_count_signature(path.identity)
        bounces: list[tuple[int, float]] = []
        for wall_index, wall in enumerate(amp.CANONICAL_WALLS):
            count = counts.get(wall, 0)
            if count:
                cos = amp.incidence_cos(amp._wall_axis(wall), path.dist_m, receiver, path.image)
                bounces.extend([(wall_index, cos)] * count)
        per_path.append(bounces)
    n_paths, n_bounces = len(paths), max(len(b) for b in per_path)
    bounce_wall = np.zeros((n_paths, n_bounces), dtype=np.int64)
    bounce_cos = np.ones((n_paths, n_bounces), dtype=np.float64)
    bounce_mask = np.zeros((n_paths, n_bounces), dtype=np.bool_)
    phase = np.empty((n_paths, len(freqs)), dtype=np.complex128)
    for i, (path, bounces) in enumerate(zip(paths, per_path, strict=True)):
        for j, (wall_index, cos) in enumerate(bounces):
            bounce_wall[i, j], bounce_cos[i, j], bounce_mask[i, j] = wall_index, cos, True
        for k, f in enumerate(freqs):
            omega = 2.0 * math.pi * f
            phase[i, k] = cmath.exp(complex(0.0, -omega * path.delay_s))
    dist = np.array([p.dist_m for p in paths], dtype=np.float64)
    return PathTable(
        frequencies_hz=np.array(freqs, dtype=np.float64),
        order=np.array([p.order for p in paths], dtype=np.int64),
        dist_m=dist,
        delay_s=np.array([p.delay_s for p in paths], dtype=np.float64),
        bounce_wall=bounce_wall,
        bounce_cos=bounce_cos,
        bounce_mask=bounce_mask,
        inverse_distance=np.array([1.0 / p.dist_m for p in paths], dtype=np.float64),
        phase=phase,
    )


def _complex(values: jax.Array) -> jax.Array:
    """實數阻抗升成同精度的複數；複數照原 dtype。"""
    return values.astype(jnp.result_type(values.dtype, jnp.complex64))


def _real_dtype(values: jax.Array) -> np.dtype[np.floating]:
    return np.dtype(np.finfo(values.dtype).dtype)


def bounce_impedances(table: PathTable, impedance_by_wall: jax.Array) -> jax.Array:
    """每面牆一列 Z(f)（``(…, 6, F)``，牆序照 CANONICAL_WALLS）收成逐反彈 ``(…, P, B, F)``；梯度穿得過。"""
    z = _complex(jnp.asarray(impedance_by_wall))
    if z.shape[-2:] != (len(amp.CANONICAL_WALLS), table.frequencies_hz.shape[0]):
        raise ValueError(f"每面牆阻抗的形狀要是 (…, 6, {table.frequencies_hz.shape[0]})，收到 {z.shape}")
    return z[..., table.bounce_wall, :]


def bounce_impedances_from_sources(
    table: PathTable,
    paths: Sequence[RoomPath],
    receiver: tuple[float, float, float],
    sources: WallReflectionSources,
) -> NDArray[np.complex128]:
    """構造牆在每條路徑的入射角直接求 Z(f, θ)，固定牆照常數；回 numpy 雙精度 ``(P, B, F)``。

    跟 :meth:`WallReflectionSources.impedances_for_paths` 同一條路：每面構造牆收真的打到它的 cos、照路徑
    順序去重，交給同一支 :func:`aosr.physics.wall_surfaces.wall_path_impedances`。``paths`` 與 ``receiver``
    要跟打包 ``table`` 時同一份。
    """
    axis = tuple(float(f) for f in table.frequencies_hz)
    if axis != sources.fixed_materials.frequencies_hz:
        raise ValueError("反射來源的頻率軸跟路徑表不同")
    if len(paths) != table.path_count:
        raise ValueError("路徑數跟路徑表不同")
    out = np.ones((table.path_count, table.bounce_count, len(axis)), dtype=np.complex128)
    for wall_index, wall in enumerate(amp.CANONICAL_WALLS):
        material = sources.materials_by_wall[Wall.from_name(wall)]
        hits = table.bounce_mask & (table.bounce_wall == wall_index)
        if not hits.any():
            continue
        if isinstance(material, ConstantImpedance):
            out[hits] = complex(material.impedance_pa_s_m)
            continue
        cosines = tuple(dict.fromkeys(
            amp.incidence_cos(amp._wall_axis(wall), p.dist_m, receiver, p.image)
            for p in paths if wall_count_signature(p.identity)[wall]))
        rows = wall_path_impedances(material, axis, cosines, wall_name=wall,
                                    density_kg_m3=sources.density_kg_m3, sound_speed_m_s=sources.sound_speed_m_s,
                                    notes_by_wall=sources.notes_by_wall)
        for i, j in zip(*np.nonzero(hits), strict=True):
            out[i, j] = rows[float(table.bounce_cos[i, j])]
    return out


def reflection_products(table: PathTable, z_bounce: jax.Array, rho_c: float) -> jax.Array:
    """逐反彈 Z ``(…, P, B, F)`` → 每條路徑的反射乘積 ``(…, P, F)``；dtype 跟 Z 走。"""
    z = _complex(jnp.asarray(z_bounce))
    expected = (table.path_count, table.bounce_count, table.frequencies_hz.shape[0])
    if z.shape[-3:] != expected:
        raise ValueError(f"逐反彈阻抗的形狀要是 (…, {expected[0]}, {expected[1]}, {expected[2]})，收到 {z.shape}")
    cos = jnp.asarray(table.bounce_cos, dtype=_real_dtype(z))[:, :, None]
    mask = jnp.asarray(table.bounce_mask)[:, :, None]
    # 沒用到的反彈格先換成 ρc 再除：除法若在 where 之後才遮，那一格的 Z 剛好是 −ρc 時前向有限、梯度卻是 NaN，
    # 而且 NaN 會經過 where 的反向傳遍整個梯度。換成 ρc 後那一格的 R 有限，梯度恰為 0。
    z_safe = jnp.where(mask, z, jnp.asarray(rho_c, dtype=z.dtype))
    z_cos = z_safe * cos
    r = jnp.where(mask, (z_cos - rho_c) / (z_cos + rho_c), jnp.ones((), dtype=z.dtype))
    product = jnp.ones(z.shape[:-3] + (expected[0], expected[2]), dtype=z.dtype)
    for bounce in range(expected[1]):
        product = product * r[..., bounce, :]
    return product


def path_pressures(table: PathTable, reflection_product: jax.Array) -> jax.Array:
    """``(1/dist)·refl·exp(−iωτ)``，``1/dist`` 與相位是路徑表的雙精度常數、進場轉成乘積的精度。"""
    refl = _complex(jnp.asarray(reflection_product))
    inverse_distance = jnp.asarray(table.inverse_distance, dtype=_real_dtype(refl))[:, None]
    phase = jnp.asarray(table.phase, dtype=refl.dtype)
    return (inverse_distance * refl) * phase


def totals(table: PathTable, path_pressure: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """總壓力、直達能量（直達壓力模平方）、反射能量（其餘路徑先複數相加再取模平方），各 ``(…, F)``。"""
    direct = table.direct_index()
    reflected_mask = jnp.asarray(table.order > 0)[:, None]
    total = jnp.sum(path_pressure, axis=-2)
    reflected = jnp.sum(jnp.where(reflected_mask, path_pressure, jnp.zeros((), path_pressure.dtype)), axis=-2)
    return total, jnp.abs(path_pressure[..., direct, :]) ** 2, jnp.abs(reflected) ** 2


def amplitudes(table: PathTable, z_bounce: jax.Array, rho_c: float) -> JaxAmplitudes:
    """一次算完反射乘積、路徑壓力與總量。"""
    refl = reflection_products(table, z_bounce, rho_c)
    pressure = path_pressures(table, refl)
    total, direct_energy, reflected_energy = totals(table, pressure)
    return JaxAmplitudes(refl, pressure, total, direct_energy, reflected_energy)
