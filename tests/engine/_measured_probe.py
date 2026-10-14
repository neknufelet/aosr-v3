"""實測表的雙精度子行程；不在父行程改 JAX 全域設定。"""
from __future__ import annotations

import sys
import json
from aosr.runtime import configure_calculation_jax
configure_calculation_jax()

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter
from aosr.config.construction_environment import construction_environment
from aosr.materials import construction as core
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.materials.construction_models import Assembly, ConstructionNote, MeasuredImpedance, SurfaceMaterial
from aosr.geometry.shoebox import Wall
from aosr.physics.wall_surfaces import merge_construction_notes



air, ranges = construction_environment(density_kg_m3=1.2, sound_speed_m_s=343.)


def material() -> MeasuredImpedance:
    result: SurfaceMaterial = TypeAdapter(SurfaceMaterial).validate_python(json.loads(sys.argv[2]))
    assert isinstance(result, MeasuredImpedance)
    return result


def core_contract() -> None:
    m = material()
    axis = jnp.asarray([50., 100., np.sqrt(100. * 1000.), 1000., 2000., 100.])
    z, _ = core.surface_impedance(m, axis, air=air, ranges=ranges)
    np.testing.assert_array_equal(np.asarray(z)[[1, 3, 5]], [400.-200.j, 800.+100.j, 400.-200.j])
    np.testing.assert_allclose(z[2], 600.-50.j, rtol=1e-14)
    angles = jnp.asarray([0., 37., 80.])
    angled, _ = core.surface_impedance_at_angles(m, axis, angles, air=air, ranges=ranges)
    np.testing.assert_array_equal(np.asarray(angled)[[1, 2, 3, 5]], np.broadcast_to(np.asarray(z)[[1, 2, 3, 5], None], (4, 3)))
    expected, _ = core.surface_impedance_at_angles(m.base_assembly, axis[jnp.asarray([0, 4])], angles, air=air, ranges=ranges)
    np.testing.assert_array_equal(np.asarray(angled)[[0, 4]], expected)
    alpha, _ = core.random_incidence_absorption(m, axis, air=air, ranges=ranges)
    assert isinstance(alpha, jax.Array) and str(alpha.dtype) == 'float64'
    assert str(z.dtype) == 'complex128'
    for index in (1, 2, 3, 5):
        assert float(alpha[index]) == complex_random_incidence_absorption(complex(z[index]) / air.rho_c)
    expected_alpha, _ = core.random_incidence_absorption(m.base_assembly, axis[jnp.asarray([0, 4])], air=air, ranges=ranges)
    np.testing.assert_array_equal(np.asarray(alpha)[[0, 4]], expected_alpha)
    flipped = m.model_copy(update={'points': tuple(p.model_copy(update={'imag_pa_s_m': -p.imag_pa_s_m}) for p in m.points)})
    changed, _ = core.surface_impedance(flipped, axis, air=air, ranges=ranges)
    assert np.any(np.asarray(z)[[1, 2, 3]] != np.asarray(changed)[[1, 2, 3]])
    context = core._context(m)
    assert '2 列' in context and '100–1000 Hz' in context and '底構造' in context
    # 嚴格遞增但 log10 相同的相鄰列（100 與它的下一個浮點數）：命中的列仍逐位回自己的值，首尾兩端都試。
    for knots in ((100., float(np.nextafter(100., np.inf)), 1000.), (100., 1000., float(np.nextafter(1000., np.inf)))):
        rows = tuple(type(m.points[0])(frequency_hz=f, real_pa_s_m=r, imag_pa_s_m=i)
                     for f, (r, i) in zip(knots, ((400., -200.), (800., 100.), (900., 50.)), strict=True))
        colliding = m.model_copy(update={'points': rows})
        hit, _ = core.surface_impedance(colliding, jnp.asarray(knots), air=air, ranges=ranges)
        np.testing.assert_array_equal(np.asarray(hit), [400.-200.j, 800.+100.j, 900.+50.j])


def notes_contract() -> None:
    m = material().model_copy(update={'base_assembly': Assembly.model_validate({'kind': 'assembly', 'layers': [
        {'kind': 'porous', 'thickness_m': .05, 'flow_resistivity_pa_s_m2': 60000., 'material_class': 'foam'}], 'backing': 'rigid'})})
    for evaluate in (core.surface_impedance, core.random_incidence_absorption):
        _, inside = evaluate(m, jnp.asarray([100., 200., 1000.]), air=air, ranges=ranges)
        assert inside == ()
        _, notes = evaluate(m, jnp.asarray([20., 100., 200., 1000., 2000.]), air=air, ranges=ranges)
        by_code = {n.code: n for n in notes}
        assert by_code['measured_table_model_fill'].frequencies_hz == (20., 2000.)
        assert by_code['measured_table_model_fill'].layer_index is None
        assert 'layer_index' not in by_code['measured_table_model_fill'].model_dump(mode='json')
        assert by_code['komatsu_frequency_extrapolation'].frequencies_hz == (20.,)
        assert by_code['komatsu_material_class_outside_fit'].frequencies_hz == ()
        assert all(n.message.startswith('底構造第 1 層') for n in notes if n.layer_index is not None)
    # 表從 50 Hz 起：底構造在 50、60 Hz 也會外推，但那兩點在表內、不准出底構造註記。
    low = m.model_copy(update={'points': (m.points[0].model_copy(update={'frequency_hz': 50.}), m.points[1])})
    axis = jnp.asarray([20., 50., 60., 100., 2000.])
    base_only = {n.code: n for n in core.surface_impedance(low.base_assembly, axis, air=air, ranges=ranges)[1]}
    assert {50., 60.} <= set(base_only['komatsu_frequency_extrapolation'].frequencies_hz), base_only
    for notes in (core.surface_impedance(low, axis, air=air, ranges=ranges)[1],
                  core.random_incidence_absorption(low, axis, air=air, ranges=ranges)[1],
                  core.surface_impedance_at_angles(low, axis, jnp.asarray([0., 45.]), air=air, ranges=ranges)[1]):
        by_code = {n.code: n for n in notes}
        assert by_code['komatsu_frequency_extrapolation'].frequencies_hz == (20.,), by_code
        assert by_code['measured_table_model_fill'].frequencies_hz == (20., 2000.), by_code
    for evaluate in (core.surface_impedance, core.random_incidence_absorption):
        _, notes = evaluate(m, jnp.asarray([20., 100., 200., 1000., 2000.]), air=air, ranges=ranges)
        merged: dict[Wall, tuple[ConstructionNote, ...]] = {}
        merge_construction_notes(merged, Wall.X0, tuple(reversed(notes)))
        merge_construction_notes(merged, Wall.X0, notes)
        assert tuple(n.code for n in merged[Wall.X0]) == ('measured_table_model_fill',
            'komatsu_frequency_extrapolation', 'komatsu_material_class_outside_fit')


def seams_contract() -> None:
    from aosr.physics.report_measured import measured_seams
    from aosr.physics.report_surfaces import WholeWallZone
    m = material()
    axis = jnp.asarray([100., 300., 1000.])
    base_z, _ = core.surface_impedance(m.base_assembly, axis, air=air, ranges=ranges)
    m = m.model_copy(update={'points': tuple(type(m.points[0])(frequency_hz=float(f), real_pa_s_m=complex(z).real,
        imag_pa_s_m=complex(z).imag) for f, z in zip(axis, base_z, strict=True))})
    assert m.points[0].imag_pa_s_m < 0
    values, _ = core.surface_impedance(m, axis, air=air, ranges=ranges)
    np.testing.assert_array_equal(values, base_z)
    seams = measured_seams({'x0': (WholeWallZone(area='whole_wall', material=m),)}, density_kg_m3=1.2, sound_speed_m_s=343.)
    assert seams is not None
    base_alpha, _ = core.random_incidence_absorption(m.base_assembly, jnp.asarray([100., 1000.]), air=air, ranges=ranges)
    assert tuple(s.frequency_hz for s in seams['x0']) == (100., 1000.)
    for index, seam in enumerate(seams['x0']):
        # 1e-12 相對容差只涵蓋複數反射與浮點舍入；不宣稱材料模型精度。
        for field in ('real_pa_s_m', 'imag_pa_s_m', 'normal_absorption'):
            np.testing.assert_allclose(getattr(seam.measured, field), getattr(seam.base_assembly, field), rtol=1e-12, atol=0.)
        z = complex(seam.measured.real_pa_s_m, seam.measured.imag_pa_s_m)
        assert seam.measured.random_absorption == complex_random_incidence_absorption(z / air.rho_c)
        assert seam.base_assembly.random_absorption == float(base_alpha[index])
        assert seam.measured.random_absorption != seam.base_assembly.random_absorption
    assert measured_seams(None, density_kg_m3=1.2, sound_speed_m_s=343.) is None


def rejection_context_contract() -> None:
    import pytest
    bad_base = Assembly.model_validate({'kind': 'assembly', 'layers': [{'kind': 'porous',
        'thickness_m': .05, 'flow_resistivity_pa_s_m2': 100000., 'material_class': 'fibrous'}], 'backing': 'rigid'})
    m = material().model_copy(update={'base_assembly': bad_base})
    for evaluate in (core.surface_impedance, core.random_incidence_absorption):
        with pytest.raises(ValueError) as caught:
            evaluate(m, jnp.asarray([20.]), air=air, ranges=ranges)
        assert '實測阻抗表 2 列' in str(caught.value) and '100–1000 Hz' in str(caught.value)
        assert '底構造' in str(caught.value) and '100000.0' in str(caught.value)
        # 底構造只說一次：不再先抄一遍底構造逐層描述、再接底構造自己的錯誤。
        assert str(caught.value).startswith('實測阻抗表 2 列，涵蓋 100–1000 Hz；底構造第 1 層：'), str(caught.value)
        assert core._context(bad_base) not in str(caught.value), str(caught.value)
    from aosr.materials.construction_models import ConstantImpedance
    from aosr.physics.report_measured import measured_seams
    from aosr.physics.report_surfaces import WholeWallZone
    from aosr.physics.wall_surfaces import wall_surface_response
    with pytest.raises(ValueError) as caught:
        measured_seams({'x0': (WholeWallZone(area='whole_wall', material=m),)}, density_kg_m3=1.2, sound_speed_m_s=343.)
    assert str(caught.value).startswith('x0：實測阻抗表 2 列') and '100000.0' in str(caught.value), str(caught.value)
    walls = {wall: ConstantImpedance(impedance_pa_s_m=1646.4) for wall in Wall.all()} | {Wall.X0: m}
    with pytest.raises(ValueError) as caught:
        wall_surface_response(walls, (20., 200.), density_kg_m3=1.2, sound_speed_m_s=343.)
    assert str(caught.value).startswith('x0：實測阻抗表 2 列'), str(caught.value)


if __name__ == '__main__':
    {'core': core_contract, 'notes': notes_contract, 'seams': seams_contract, 'context': rejection_context_contract}[sys.argv[1]]()
