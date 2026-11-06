"""有限元素替身的 JAX（可微陣列算數）求值與面片構造接線（#561）。

同一份投影矩陣、同一個 ``K_r − k²M_r + Σ_w (jωρ/Z_w) B_r,w`` 算式；
頻率常數借 numpy 版 :func:`system_coefficients` 雙精度算好，不另抄係數公式。
常數是 NamedTuple（JAX 樹狀資料），可以整包交進 ``jit``；逐頻用 ``vmap`` 解小系統。
精度跟阻抗輸入走，矩陣乘法指定最高精度；本模組不改全域設定。

面片序由 ``boundary_patches``（共用的方案面片序）指定。剛性面由 ``rigid_mask`` 明確標示，除法前先換成安全分母，
所以那面的係數與梯度恰為零，即使該列阻抗填零或非有限數也不污染其他面。遮罩是**必填**：
沒有「不給就當全部不剛性」的預設——面片樣板入口把零佔位的剛性面與遮罩一包
（:class:`TemplateImpedances`）交出，漏接遮罩會在呼叫時就被擋，不會靜默算出 jωρ/0。
具體遮罩拒收分區剛性；追蹤中的遮罩一律忽略分區的剛性位元。
其餘只檢查靜態形狀與型別；頻率、空氣、材料參數與未遮阻抗的合法值由呼叫端守。
決策紙 ``docs/decisions/fem-reduced-model-jax-evaluation-for-stage-three.md``。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.physics_constants import PhysicsConstants
from aosr.materials.construction_diff import ConstructionTemplate, surface_impedance_from_parameters
from aosr.physics.fem_reduced import ReducedModel, checked_axis, system_coefficients
from aosr.physics.fem_reduced_boundary import PatchKeys, boundary_patches, checked_patch_keys


_HIGHEST = jax.lax.Precision.HIGHEST


class ReducedConstants(NamedTuple):
    """材料無關的 numpy 常數；partition_mask 是分區片布林列，sources 為（喇叭,r）、receivers 為（座位,r）。"""

    stiffness: NDArray[np.float64] | jax.Array
    mass: NDArray[np.float64] | jax.Array
    boundary: NDArray[np.float64] | jax.Array
    sources: NDArray[np.complex128] | jax.Array
    receivers: NDArray[np.float64] | jax.Array
    frequencies_hz: NDArray[np.float64] | jax.Array
    density_kg_m3: NDArray[np.float64] | jax.Array
    sound_speed_m_s: NDArray[np.float64] | jax.Array
    k_squared: NDArray[np.float64] | jax.Array
    wall_numerator: NDArray[np.complex128] | jax.Array
    partition_mask: NDArray[np.bool_] | jax.Array


class TemplateImpedances(NamedTuple):
    """面片樣板算出的阻抗 ``(P,F)`` 與同一組剛性遮罩 ``(P,)``：一包交出，求值入口兩個一起收。"""

    values: jax.Array
    rigid_mask: jax.Array


def reduced_constants(model: ReducedModel) -> ReducedConstants:
    """打包已建替身；k² 與 jωρ 用 numpy 唯一那份係數入口，單位阻抗得到分子。"""
    keys = checked_patch_keys(model.patch_keys)
    axis = checked_axis(model.frequencies_hz)
    m = model.matrices
    rank = m.stiffness.shape[0]
    if (rank == 0 or m.stiffness.shape != (rank, rank) or m.mass.shape != (rank, rank)
            or m.boundary.shape != (len(keys), rank, rank)
            or m.sources.shape != (len(model.source_names), rank)
            or m.receivers.shape != (len(model.receiver_names), rank)):
        raise ValueError("替身矩陣形狀與基底、面片、喇叭或座位數不同")
    coefficients = [system_coefficients(f, dict.fromkeys(keys, 1.0),
        density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s, patch_keys=keys) for f in axis]
    return ReducedConstants(
        stiffness=np.asarray(m.stiffness, dtype=np.float64), mass=np.asarray(m.mass, dtype=np.float64),
        boundary=np.asarray(m.boundary, dtype=np.float64), sources=np.asarray(m.sources, dtype=np.complex128),
        receivers=np.asarray(m.receivers, dtype=np.float64), frequencies_hz=np.asarray(axis, dtype=np.float64),
        density_kg_m3=np.asarray(model.density_kg_m3, dtype=np.float64),
        sound_speed_m_s=np.asarray(model.sound_speed_m_s, dtype=np.float64),
        k_squared=np.asarray([pair[0] for pair in coefficients], dtype=np.float64),
        wall_numerator=np.asarray([pair[1][0] for pair in coefficients], dtype=np.complex128),
        partition_mask=np.asarray([bool(zone) for _, zone in keys], dtype=np.bool_))


def _checked_impedances(constants: ReducedConstants, impedance_by_wall: jax.Array) -> jax.Array:
    values = jnp.asarray(impedance_by_wall)
    expected = (constants.boundary.shape[0], constants.frequencies_hz.shape[0])
    if values.shape != expected:
        raise ValueError(f"每面片阻抗形狀必須是 {expected}，收到 {values.shape}")
    if not jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError(f"阻抗必須是複數陣列，收到 {values.dtype}")
    return values


def wall_coefficients(constants: ReducedConstants, impedance_by_wall: jax.Array, *,
                      rigid_mask: jax.Array) -> jax.Array:
    """``(P,F)`` 阻抗 → ``(F,P)`` 的 jωρ/Z；``(P,)`` 布林遮罩必填，True 是剛性。

    具體遮罩若將分區片設成剛性就拒收；jit（編譯）追蹤時無法檢查值，
    遮罩一律與「不是分區」取交集，分區片的剛性位元不生效，仍用該片阻抗。
    """
    z = _checked_impedances(constants, impedance_by_wall)
    mask = jnp.asarray(rigid_mask)
    if mask.shape != (constants.boundary.shape[0],) or mask.dtype != jnp.bool_:
        raise ValueError("剛性遮罩必須是 (面片數,) 布林陣列")
    if not isinstance(mask, jax.core.Tracer) and not isinstance(constants.partition_mask, jax.core.Tracer):
        if np.any(np.asarray(mask) & constants.partition_mask):
            raise ValueError("分區面片一定要有材料，不能是剛性")
    mask = mask & ~jnp.asarray(constants.partition_mask)
    safe = jnp.where(mask[:, None], jnp.ones((), dtype=z.dtype), z)
    numerator = jnp.asarray(constants.wall_numerator, dtype=z.dtype)
    return jnp.where(mask[:, None], jnp.zeros((), dtype=z.dtype), numerator[None, :] / safe).T


def reduced_pressures(constants: ReducedConstants, impedance_by_wall: jax.Array, *,
                      rigid_mask: jax.Array, frequency_batch: int | None = None) -> jax.Array:
    """逐頻解小系統，回（頻率,喇叭,座位）複數聲壓；可 jit、grad 與外層 vmap。遮罩必填。

    具體遮罩拒收分區剛性；jit（編譯）追蹤時分區剛性一律不生效，仍用材料阻抗，見 wall_coefficients。
    frequency_batch 指定正整數時逐批解；checkpoint（反傳時重算）不保存每頻的系統矩陣與 LU 分解。
    lax.map（逐批映射）處理尾批；批量大於等於頻數仍走重算路徑，視為一批。
    批量是靜態參數：jit（編譯）時關入閉包，或用 static_argnames 指定 frequency_batch。
    """
    if frequency_batch is not None and (type(frequency_batch) is not int or frequency_batch <= 0):
        raise ValueError("frequency_batch 必須是正整數或 None")
    z = _checked_impedances(constants, impedance_by_wall)
    coefficients = wall_coefficients(constants, z, rigid_mask=rigid_mask)
    real_dtype = z.real.dtype
    stiffness = jnp.asarray(constants.stiffness, dtype=real_dtype)
    mass = jnp.asarray(constants.mass, dtype=real_dtype)
    boundary = jnp.asarray(constants.boundary, dtype=real_dtype)
    sources = jnp.asarray(constants.sources, dtype=z.dtype)
    receivers = jnp.asarray(constants.receivers, dtype=real_dtype)
    k_squared = jnp.asarray(constants.k_squared, dtype=real_dtype)
    def solve(k2: jax.Array, walls: jax.Array) -> jax.Array:
        system = stiffness - k2 * mass + jnp.tensordot(walls, boundary, axes=1, precision=_HIGHEST)
        coordinates = jnp.linalg.solve(system, sources.T)
        return jnp.matmul(receivers, coordinates, precision=_HIGHEST).T
    if frequency_batch is None:
        return jax.vmap(solve)(k_squared, coefficients)
    return cast(jax.Array, jax.lax.map(lambda pair: jax.checkpoint(solve)(*pair), (k_squared, coefficients),
                                     batch_size=min(frequency_batch, k_squared.shape[0])))


def rigid_mask_from_templates(templates: Sequence[ConstructionTemplate | None], *,
                              patch_keys: PatchKeys = boundary_patches()) -> jax.Array:
    """整面面片樣板的 None 對應剛性；與阻抗一起交進求值入口。"""
    checked_patch_keys(patch_keys)
    if len(templates) != len(patch_keys):
        raise ValueError("樣板必須照 Wall.all() 給齊六面" if patch_keys == boundary_patches() else "樣板必須照面片清單給齊")
    if any(zone and template is None for (_, zone), template in zip(patch_keys, templates, strict=True)):
        raise ValueError("分區面片一定要有材料樣板，不能給 None")
    return jnp.asarray([template is None for template in templates], dtype=bool)


def wall_impedances_from_templates(
    templates: Sequence[ConstructionTemplate | None], parameters: Sequence[Mapping[str, jax.Array]],
    frequencies: jax.Array, *, air: PhysicsConstants, patch_keys: PatchKeys = boundary_patches(),
) -> TemplateImpedances:
    """面片構造參數 → 垂直入射阻抗 ``(P,F)`` 與剛性遮罩一包；None 面填零、參數必須空字典。

    零只是剛性面的佔位值，遮罩跟阻抗同一包交出、求值時兩個一起給；
    不以無窮大代表剛性，避免除法反向傳播產生非有限梯度。
    """
    mask = rigid_mask_from_templates(templates, patch_keys=patch_keys)
    if len(parameters) != len(patch_keys):
        raise ValueError("參數必須照 Wall.all() 給齊六面" if patch_keys == boundary_patches() else "參數必須照面片清單給齊")
    axis = jnp.asarray(frequencies)
    if axis.ndim != 1 or axis.shape[0] == 0:
        raise ValueError("頻率軸必須是一維非空陣列")
    if not jnp.issubdtype(axis.dtype, jnp.floating):
        raise TypeError("頻率軸必須是浮點陣列")
    rows = []
    for template, values in zip(templates, parameters, strict=True):
        if template is None:
            if values:
                raise ValueError("剛性面不收構造參數")
            rows.append(jnp.zeros(axis.shape, dtype=jnp.result_type(axis.dtype, jnp.complex64)))
        else:
            rows.append(surface_impedance_from_parameters(template, values, axis, air=air))
    return TemplateImpedances(jnp.stack(rows), mask)
