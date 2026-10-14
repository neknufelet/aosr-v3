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


if __name__ == '__main__':
    {'core': core_contract, 'notes': notes_contract, 'seams': seams_contract, 'context': rejection_context_contract}[sys.argv[1]]()
