"""θ → 材料四路阻抗 → 替身聲壓 → 六類；新增工廠，現行搜尋仍走原鏈。

呼叫端先建一份 SixCategoryConstants（六類幾何與評分常數），所有候選共用。
工廠只建材料樣板與保留牆常數，不重建幾何；精度跟 θ 走。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.gradient.material_gradient import (
    SurfaceDesign, _patch_impedances, design_template_material, design_templates, designs_mount,
    kept_impedance_rows, surface_parameters,
)
from aosr.gradient.surrogate_guard import require_matching_patches
from aosr.materials.construction_diff import ConstructionTemplate, build_template, surface_impedance_from_parameters
from aosr.materials.construction_models import ConstantImpedance, MeasuredImpedance, SurfaceMaterial
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_jax import ReducedConstants, reduced_constants, reduced_pressures, wall_impedances_from_templates
from aosr.scoring._six_category_build import constant_material_flags
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays, GradientObjectiveConstants, gradient_objective_from_pressures
from aosr.scoring.six_category_jax import SixCategoryArrays, SixCategoryConstants, six_category_costs


class SixCategoryChainAux(NamedTuple):
    """六類代價與旗標、兩個總分，以及舊低頻的 screening／proxy／objective_total（篩選／代理／目標）。"""

    categories: SixCategoryArrays
    low_frequency: GradientObjectiveArrays
    objective_total: jax.Array
    product_total: jax.Array


type SixCategoryChainObjective = Callable[[jax.Array], tuple[jax.Array, SixCategoryChainAux]]
type ImpedanceFunction = Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]


def _materials(designs: Sequence[SurfaceDesign]) -> tuple[SurfaceMaterial, ...]:
    materials = []
    for design in designs:
        material = design_template_material(design) if design.material is not None else design.kept_material
        if material is None:
            raise ValueError(f"六類面片 {design.surface_id} 必須有材料；無材料的合成剛性面沒有六類阻抗")
        if isinstance(material, MeasuredImpedance):
            raise ValueError(f"保留牆 {design.surface_id} 是 MeasuredImpedance：numpy 的表內無規入射吸音公式尚無對應 JAX 分支，六類工廠拒收")
        materials.append(material)
    return tuple(materials)


def _angle_impedance(material: SurfaceMaterial, template: ConstructionTemplate | None,
                     parameters: dict[str, jax.Array], axis: jax.Array, angles: jax.Array | None,
                     *, air: PhysicsConstants) -> jax.Array:
    if isinstance(material, ConstantImpedance):
        shape = axis.shape if angles is None else (axis.size, angles.size)
        return jnp.full(shape, complex(material.impedance_pa_s_m), dtype=jnp.result_type(axis.dtype, jnp.complex64))
    assert template is not None
    return surface_impedance_from_parameters(template, parameters, axis, air=air, incidence_angles_deg=angles)


def _angles(values: NDArray[np.float64], dtype: np.dtype) -> jax.Array:
    """沿用 Paris 角生成器護欄；90 度前一格轉單精度時不可變成 90 度。"""
    edge = jnp.nextafter(jnp.asarray(90., dtype=dtype), jnp.asarray(0., dtype=dtype))
    return jnp.minimum(jnp.asarray(values, dtype=dtype), edge)


def _impedance_function(designs: tuple[SurfaceDesign, ...], c: SixCategoryConstants, *,
                        air: PhysicsConstants, ranges: PorousModelRanges) -> ImpedanceFunction:
    materials = _materials(designs)
    templates = design_templates(designs, jnp.asarray(c.fem_frequencies_hz), air=air, ranges=ranges)
    angle_templates = tuple(template if template is not None or isinstance(material, ConstantImpedance) else
        build_template(material, jnp.asarray(c.union_frequencies_hz), air=air, ranges=ranges, variables=())
        for material, template in zip(materials, templates, strict=True))
    kept = tuple((i, row) for i, row in enumerate(kept_impedance_rows(designs, c.fem_frequencies_hz, air=air, ranges=ranges)) if row is not None)
    zoned = any('@' in d.surface_id for d in designs)

    def impedances(theta: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        parameters = surface_parameters(designs, theta)
        fem_axis = jnp.asarray(c.fem_frequencies_hz, dtype=theta.dtype)
        fem = (_patch_impedances(templates, parameters, fem_axis, air=air) if zoned else
               wall_impedances_from_templates(templates, parameters, fem_axis, air=air)).values
        for index, row in kept:
            fem = fem.at[index].set(jnp.asarray(row, dtype=fem.dtype))
        report_axis = jnp.asarray(c.frequencies_hz, dtype=theta.dtype)
        union_axis = jnp.asarray(c.union_frequencies_hz, dtype=theta.dtype)
        normal, oblique, paris = [], [], []
        for i, (material, template, values) in enumerate(zip(materials, angle_templates, parameters, strict=True)):
            a, b = (int(x) for x in c.angle_offsets[i:i+2])
            normal.append(_angle_impedance(material, template, values, report_axis, None, air=air))
            oblique.append(_angle_impedance(material, template, values, report_axis, _angles(c.angles_deg[a:b], theta.dtype), air=air))
            paris.append(_angle_impedance(material, template, values, union_axis, _angles(c.paris_angles_deg, theta.dtype), air=air))
        return fem, jnp.stack(normal), jnp.concatenate(oblique, axis=1), jnp.stack(paris)

    return impedances


def _check_inputs(model: ReducedModel, designs: tuple[SurfaceDesign, ...], objective: GradientObjectiveConstants,
                  c: SixCategoryConstants, frequencies: Sequence[float] | NDArray[np.float64]) -> None:
    ids = tuple(d.surface_id for d in designs)
    require_matching_patches(model, ids)
    if ids != tuple(str(x) for x in c.patch_ids):
        raise ValueError('六類常數與設計的有序面片清單不同')
    designs_mount(designs)
    axis = np.asarray(frequencies, dtype=np.float64)
    if not np.array_equal(axis, model.frequencies_hz) or not np.array_equal(axis, c.fem_frequencies_hz):
        raise ValueError('頻率軸必須等於替身與六類常數的 FEM 頻率軸')
    if any(not np.array_equal(a, b) for a, b in zip(objective, c.objective(), strict=True)):
        raise ValueError('低頻目標常數必須等於六類常數的搜尋低頻定義')
    if c.direct_distances_m.shape != (len(model.source_names), len(model.receiver_names)):
        raise ValueError('六類常數的聲源／座位數與替身不同')


def six_category_chain_objective(model: ReducedModel, designs: Sequence[SurfaceDesign],
                                 objective_constants: GradientObjectiveConstants, six_category_constants: SixCategoryConstants,
                                 *, air: PhysicsConstants, ranges: PorousModelRanges,
                                 frequencies: Sequence[float] | NDArray[np.float64], frequency_batch: int | None = None,
                                 single_precision: bool = False, device_constants: ReducedConstants | None = None) -> SixCategoryChainObjective:
    """回 θ → (六類搜尋目標, aux 分項)；可 jit／grad（編譯／微分），精度跟 θ 走。

    六類常數必須由呼叫端事先 build_constants 建一次；候選只改樣板與 constant_material（常數材料旗標）。
    保留構造用無變數樣板算角度阻抗，固定阻抗廣播；保留 MeasuredImpedance 明確拒收。
    舊低頻直接呼叫原評分入口，aux.low_frequency 保留篩選、代理與目標原值。
    這一支不接搜尋與顯卡包裝；single_precision／device_constants 顯卡選項留待後續支線。
    float32 θ 仍可在 CPU（中央處理器）求值，五個數值輸入進六類前統一成對應複數精度。
    """
    if single_precision or device_constants is not None:
        raise ValueError('六類工廠尚未接顯卡包裝；請直接以 θ 的精度求值')
    frozen_designs = tuple(designs)
    c = six_category_constants
    _check_inputs(model, frozen_designs, objective_constants, c, frequencies)
    impedance_for = _impedance_function(frozen_designs, c, air=air, ranges=ranges)
    flags = constant_material_flags(_materials(frozen_designs))
    constants = reduced_constants(model)

    def objective(theta: jax.Array, reduced: ReducedConstants = constants) -> tuple[jax.Array, SixCategoryChainAux]:
        values = jnp.asarray(theta)
        fem, normal, early, paris = impedance_for(values)
        pressures = reduced_pressures(reduced, fem, rigid_mask=jnp.zeros(len(frozen_designs), dtype=bool), frequency_batch=frequency_batch)
        dtype = jnp.result_type(values.dtype, jnp.complex64)
        z_fem, z_normal, z_early, z_paris, p = (jnp.asarray(x, dtype=dtype)
            for x in (fem, normal, early, paris, jnp.transpose(pressures, (1, 2, 0))))
        categories = six_category_costs(c, z_fem, z_normal, z_early, z_paris, p, jnp.asarray(flags))
        low = gradient_objective_from_pressures(objective_constants, pressures)
        aux = SixCategoryChainAux(categories, low, categories.objective_total, categories.product_total)
        return aux.objective_total, aux

    return objective
