"""CPU 雙精度冷行程（#1082 段 2-3d）：實測阻抗表當固定材料進第三階段六類鏈，對產品（非搜尋）那套。

同一間小房、同一份替身與六類常數（沿用 ``_six_category_chain_probe``）；兩張合成實測表（不是真實量測）：
一張有虛部、整面保留（方案原牆或固定面那一條），一張沒有虛部、經材料庫 ``to_physics`` 補 0，照准用清單
``fixed:<代號>`` 整塊換上一個分區（換樣板那一條）。兩張表都只蓋聯集軸中間一段、兩端剛好落在軸點上，
表內、表外與接縫都考到。答案是產品材料入口（``construction.surface_impedance`` 等）與正式 numpy 管線。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.construction_environment import construction_environment
from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.gradient import endpoint, material_gradient as old, six_category_chain as chain
from aosr.materials import construction
from aosr.materials.construction_models import Assembly, ConstantImpedance, MeasuredImpedance, SurfaceMaterial
from aosr.materials.library import MeasuredTable
from aosr.materials.search_library import FixedCandidate
from aosr.physics.late_family_jax import late_family_arrays
from aosr.reporting.scheme import Scheme
from aosr.scoring import six_category_jax as scoring
from aosr.scoring.six_category_jax import SixCategoryConstants, six_category_costs
from tests.engine import _six_category_jax_cases as cases
from tests.engine._six_category_chain_probe import candidate_designs, model_for, single_measure
from tests.engine.test_material_gradient import _settings

WHOLE, ZONE, TARGET = 'y0', 'x0@a', 'fixed:measured_no_imag'


def _scheme() -> Scheme:
    """跟 ``_six_category_chain_probe.main`` 同一間：分區的 x0、xL 換成構造。"""
    scheme = cases.scheme_case('zoned')
    zones = dict(scheme.scene.surface_zones_by_wall or {})
    assembly = cases.scheme_case('changed_wall').scene.surface_zones_by_wall
    assert assembly is not None
    zones['xL'] = assembly['x0']
    return Scheme.model_validate({**scheme.model_dump(), 'scene': {**scheme.scene.model_dump(), 'surface_zones_by_wall': zones}})


def _knots(c: SixCategoryConstants) -> tuple[float, ...]:
    """表的頻點：兩端是聯集軸上的點（表內外的接縫剛好落在求值點），中間兩點不在軸上（考內插）。"""
    union = np.asarray(c.union_frequencies_hz)
    low, high = float(union[len(union) // 5]), float(union[3 * len(union) // 5])
    middle = np.geomspace(low, high, 4)[1:3] * 1.013
    assert not np.isin(middle, union).any()
    return (low, float(middle[0]), float(middle[1]), high)


def tables(c: SixCategoryConstants, base: Assembly) -> tuple[MeasuredImpedance, MeasuredImpedance]:
    knots = _knots(c)
    with_imag = MeasuredImpedance.model_validate({'kind': 'measured_impedance', 'base_assembly': base.model_dump(),
        'source_and_conditions': '合成考卷用表（非真實量測），有虛部',
        'points': [{'frequency_hz': f, 'real_pa_s_m': 411.6 * r, 'imag_pa_s_m': 411.6 * i}
                   for f, r, i in zip(knots, (3.0, 5.5, 8.0, 4.0), (-2.0, -0.7, 0.4, 1.1), strict=True)]})
    library = MeasuredTable.model_validate({'kind': 'measured_impedance', 'base_assembly': base.model_dump(),
        'source_and_conditions': '合成考卷用表（非真實量測），沒有虛部',
        'points': [{'frequency_hz': f, 'real_pa_s_m': 411.6 * r} for f, r in zip(knots, (2.5, 6.0, 9.0, 3.5), strict=True)]})
    assert library.imaginary_missing
    return with_imag, library.to_physics()


def designs_with_tables(scheme: Scheme, c: SixCategoryConstants) -> tuple[tuple[old.SurfaceDesign, ...], Assembly]:
    designs = candidate_designs(scheme, c, 'rigid')
    base = next(d.material for d in designs if d.material is not None)
    assert isinstance(base, Assembly)
    kept, listed = tables(c, base)
    presets = {TARGET: FixedCandidate(id=TARGET, material=listed.model_dump(mode='json'))}
    result = []
    for design in designs:
        if design.surface_id == WHOLE:
            result.append(dataclasses.replace(design, kept_material=kept))
        elif design.surface_id == ZONE:
            result.append(old.fixed_candidate_design(ZONE, TARGET, presets, 'rigid'))
        else:
            result.append(design)
    return tuple(result), base


def _product_streams(materials: tuple[SurfaceMaterial, ...], c: SixCategoryConstants, air: PhysicsConstants,
                     ranges: PorousModelRanges) -> tuple[np.ndarray, ...]:
    """產品材料入口（numpy 那套：表內對數內插、表外底構造）求四路；常數與構造一樣走這裡當答案。"""
    def z(material: SurfaceMaterial, axis: np.ndarray, angles: np.ndarray | None = None) -> np.ndarray:
        if angles is None:
            return np.asarray(construction.surface_impedance(material, jnp.asarray(axis), air=air, ranges=ranges)[0])
        if not len(angles):
            return np.empty((len(axis), 0), dtype=np.complex128)
        return np.asarray(construction.surface_impedance_at_angles(material, jnp.asarray(axis), jnp.asarray(angles), air=air, ranges=ranges)[0])
    return (np.stack([z(m, c.fem_frequencies_hz) for m in materials]), np.stack([z(m, c.frequencies_hz) for m in materials]),
            np.concatenate([z(m, c.frequencies_hz, c.angles_deg[int(a):int(b)])
                            for m, a, b in zip(materials, c.angle_offsets[:-1], c.angle_offsets[1:], strict=True)], axis=1),
            np.stack([z(m, c.union_frequencies_hz, c.paris_angles_deg) for m in materials]))


def _errors(actual: tuple[jax.Array, ...], expected: tuple[np.ndarray, ...]) -> list[list[float]]:
    return [[float(np.max(np.abs(part(np.asarray(a)) - part(b)), initial=0.)) for part in (np.real, np.imag)]
            for a, b in zip(actual, expected, strict=True)]


def _rounds(values: np.ndarray, below: bool) -> list[float]:
    """轉單精度後往表外捨入的軸點：下端要往下捨、上端要往上捨。"""
    return [float(x) for x in values if (float(np.float32(x)) < x if below else float(np.float32(x)) > x)]


def seam_single(designs: tuple[old.SurfaceDesign, ...], c: SixCategoryConstants, air: PhysicsConstants,
                ranges: PorousModelRanges) -> dict[str, object]:
    """單精度端點接縫（審查 #1102）：表的兩端換成「轉 float32 後落到表外」的軸點，走真的 single_precision=True 工廠、
    三條頻率軸都當 jit 動態入參；阻抗要選表那一支（離表值比離底構造近），晚期吸音遮罩同判表內、吸音有限。"""
    union = np.asarray(c.union_frequencies_hz)
    row = [str(x) for x in c.patch_ids].index(WHOLE)
    kept = designs[row].kept_material
    assert isinstance(kept, MeasuredImpedance)
    knots = [p.frequency_hz for p in kept.points]
    low = _rounds(union[(union < knots[1]) & (union > union[0])], below=True)[-1]
    high = _rounds(union[(union > knots[-2]) & (union < union[-1])], below=False)[0]
    points = [p.model_dump() for p in kept.points]
    points[0]['frequency_hz'], points[-1]['frequency_hz'] = low, high
    table = MeasuredImpedance.model_validate({**kept.model_dump(), 'points': points})
    seam = tuple(dataclasses.replace(d, kept_material=table) if i == row else d for i, d in enumerate(designs))
    factory = chain.impedance_function(seam, c, air=air, ranges=ranges, single_precision=True)
    names = ('fem_frequencies_hz', 'frequencies_hz', 'union_frequencies_hz', 'angles_deg', 'paris_angles_deg')

    def evaluate(theta: jax.Array, axes: dict[str, jax.Array]) -> tuple[jax.Array, ...]:
        # jit 裡是追蹤值；標成 numpy 只為了讓 _replace 收（工廠只讀這五條軸）。
        moved = {n: cast(NDArray[np.float64], axes[n]) for n in names}
        return factory(theta, c._replace(fem_frequencies_hz=moved['fem_frequencies_hz'], frequencies_hz=moved['frequencies_hz'],
                                         union_frequencies_hz=moved['union_frequencies_hz'], angles_deg=moved['angles_deg'],
                                         paris_angles_deg=moved['paris_angles_deg']))
    theta = jnp.asarray(old.baseline_theta(seam), dtype=jnp.float32)
    streams = jax.jit(evaluate)(theta, {n: jnp.asarray(getattr(c, n), dtype=jnp.float32) for n in names})
    materials = chain.design_materials(seam)
    mask = np.asarray(scoring.constant_absorption_mask(materials, union))
    columns = [int(np.flatnonzero(union == f)[0]) for f in (low, high)]
    paris = np.asarray(streams[3])[row]
    table_z = [complex(points[0]['real_pa_s_m'], points[0]['imag_pa_s_m']), complex(points[-1]['real_pa_s_m'], points[-1]['imag_pa_s_m'])]
    base_z = np.asarray(construction.surface_impedance_at_angles(table.base_assembly, jnp.asarray(union), jnp.asarray(c.paris_angles_deg),
                                                                 air=air, ranges=ranges)[0])
    with patch.object(scoring, 'late_family_arrays', wraps=late_family_arrays) as spy:
        scoring._late(c, jnp.asarray(streams[3], dtype=jnp.complex128), jnp.zeros(len(c.frequencies_hz)), jnp.asarray(mask))
    alpha = np.asarray(spy.call_args.args[1])[row]
    return dict(
        rounds_outside=[float(np.float32(low)) < low, float(np.float32(high)) > high],
        mask_inside=[bool(mask[row, j]) for j in columns],
        nearer_table=[bool(np.all(np.abs(paris[j] - z) < np.abs(paris[j] - base_z[j]))) for j, z in zip(columns, table_z, strict=True)],
        alpha_finite=[bool(np.isfinite(alpha[j])) for j in columns])


def main(output: Path) -> None:
    scheme = _scheme()
    model = model_for(scheme)
    c = cases.constants(scheme)
    designs, _ = designs_with_tables(scheme, c)
    theta = jnp.asarray(old.baseline_theta(designs))
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    objective = chain.six_category_chain_objective(model, designs, c.objective(), c, air=air, ranges=ranges,
                                                   frequencies=model.frequencies_hz)
    calls: list[tuple[jax.Array, ...]] = []
    def observe(constants: SixCategoryConstants, *args: jax.Array) -> object:
        calls.append(args)
        return six_category_costs(constants, *args)
    with patch.object(chain, 'six_category_costs', observe):
        eager = objective(theta)[1]
    inputs = calls[0]
    compiled = jax.jit(objective)(theta)[1]
    # 答案：匯出方案（終點寫的那一份）的材料，走產品材料入口與正式 numpy 管線。
    exported = Scheme.model_validate({**scheme.model_dump(), 'scene': endpoint.scene_with_materials(
        scheme.scene, endpoint.materials_from_surfaces(designs, np.asarray(theta), _settings())).model_dump()})
    zones = exported.scene.surface_zones_by_wall or {}
    materials = tuple(next(z.material for z in zones[str(name).partition('@')[0]]
                           if getattr(z, 'id', '') == str(name).partition('@')[2]) for name in c.patch_ids)
    expected = _product_streams(materials, c, air, ranges)
    impedances = chain.impedance_function(designs, c, air=air, ranges=ranges)
    union = np.asarray(c.union_frequencies_hz)
    tables_by_row = {i: m for i, m in enumerate(materials) if isinstance(m, MeasuredImpedance)}
    measured_rows = list(tables_by_row)
    inside = {i: (union >= m.points[0].frequency_hz) & (union <= m.points[-1].frequency_hz) for i, m in tables_by_row.items()}
    mask = np.asarray(inputs[5])
    # 晚期吸音直接對產品的逐頻無規入射吸音（表內固定阻抗換算、表外底構造 Paris），用工廠實際交給評分的那一份阻抗與遮罩。
    # 攔晚期家族入口收到的逐面片吸音 (N, F_union)（_late 回傳的是展開到晚期格子的那一份）。
    with patch.object(scoring, 'late_family_arrays', wraps=late_family_arrays) as spy:
        scoring._late(c, inputs[3], jnp.zeros(len(c.frequencies_hz)), inputs[5])
    alpha = spy.call_args.args[1]
    alpha_expected = np.stack([np.asarray(construction.random_incidence_absorption(m, jnp.asarray(union), air=air, ranges=ranges)[0])
                               for m in materials])
    numpy_result = cases.numpy_result(exported, np.asarray(inputs[4]))
    costs, total = cases.numpy_costs(numpy_result)
    record = dict(
        measured_patches=[str(c.patch_ids[i]) for i in measured_rows],
        exported_kinds=[m.kind for m in materials],
        imag_zero=[all(p.imag_pa_s_m == 0.0 for p in m.points) for m in tables_by_row.values()],
        inside_counts=[int(inside[i].sum()) for i in measured_rows], outside_counts=[int((~inside[i]).sum()) for i in measured_rows],
        mask_shape=list(mask.shape), mask_rows=[mask[i].tolist() == inside[i].tolist() for i in measured_rows],
        mask_others=[bool(np.all(mask[i] == isinstance(materials[i], ConstantImpedance))) for i in range(len(materials)) if i not in measured_rows],
        scales=[[float(np.max(np.abs(part(x)), initial=0.)) for part in (np.real, np.imag)] for x in expected],
        eager_errors=_errors(inputs[:4], expected), jit_errors=_errors(jax.jit(impedances)(theta), expected),
        alpha=np.asarray(alpha).tolist(), alpha_expected=alpha_expected.tolist(),
        reference=costs.tolist(), reference_product=total, weights=c.category_weights.tolist(),
        eager=np.asarray(eager.categories[:6]).tolist(), jit=np.asarray(compiled.categories[:6]).tolist(),
        eager_product=float(eager.product_total), jit_product=float(compiled.product_total),
        flags=np.asarray(eager.categories[6:12]).tolist(), single=single_measure(objective, theta),
        seam_single=seam_single(designs, c, air, ranges))
    output.write_text(json.dumps(record, allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
