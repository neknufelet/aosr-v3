"""外部審查兩個入口控制：只改聲源鍵序，以及真正沒有鏡像入射角的構造面片。"""
from __future__ import annotations

import dataclasses
from datetime import date
import json
from pathlib import Path
import sys
from typing import cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint, endpoint_six, material_gradient as g, run
from aosr.gradient.material_gradient_ledger import objective_fingerprint
from aosr.gradient.six_category_chain import SixCategoryChainAux, impedance_function
from aosr.gradient.impedance_angles import impedance_at_angles_or_empty
from aosr.materials.construction_models import Assembly
from aosr.physics.fem_reduced_jax import reduced_constants, reduced_pressures
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.compare import compare_results
from tests.engine import _six_category_jax_cases as cases
from tests.engine._directivity import DIRECTIVITY
from tests.engine._six_category_gpu_probe import factory, problem_case
from tests.engine.test_six_category_jax import _tolerances


def sources() -> dict[str, object]:
    scheme, room, settings, project, _ = problem_case()
    # 固定同一份反序 ROM（含 RHS），只換方案 speakers 字典的插入序。
    room = dataclasses.replace(room, source_names=room.source_names[::-1],
        matrices=dataclasses.replace(room.matrices, sources=room.matrices.sources[::-1]))
    project = project.model_copy(update={'device': 'cpu'})
    variants = (scheme, scheme.model_copy(update={'speakers': dict(reversed(tuple(scheme.speakers.items())))}))
    outputs = []
    metadata = []
    for variant in variants:
        with patch.object(run, 'check_inputs', return_value=(variant, room, settings)):
            problem = run.load_problem(project)
        theta = jnp.asarray(g.baseline_theta(problem.designs), dtype=jnp.float64)
        (value, aux), gradient = g.compiled_evaluator(problem.objective)(theta)
        aux = cast(SixCategoryChainAux, aux)
        outputs.append([np.asarray(x).tobytes().hex() for x in (value, aux.product_total, *aux.categories[:6], gradient)])
        context = endpoint_six.scoring_context(variant, room, project.purpose,
            capabilities=load_capabilities(project.capabilities_path), directivity=DIRECTIVITY,
            capabilities_path=Path(project.capabilities_path), engine_commit='test')
        metadata.append(tuple(context.constants.source_ids) == room.source_names)
    return dict(bitwise=outputs[0] == outputs[1], values=outputs, metadata=metadata,
                keys=[list(s.speakers) for s in variants], rom_sources=list(room.source_names))


def rejection() -> dict[str, object]:
    scheme, room, settings, project, c = problem_case()
    kept = run.scene_wall_materials(scheme, capabilities=load_capabilities(project.capabilities_path), directivity=DIRECTIVITY)
    designs = run.designs_for(project.presets, settings, kept=kept)
    wrong = dataclasses.replace(room, source_names=room.source_names[::-1],
        matrices=dataclasses.replace(room.matrices, sources=room.matrices.sources[::-1]))
    errors: list[str | None] = []
    for single in (False, True):
        try:
            factory(wrong, designs, c, single)
        except ValueError as error:
            errors.append(str(error))
        else:
            errors.append(None)
    try:
        with patch.object(endpoint_six, 'build_constants', return_value=c):
            endpoint_six.scoring_context(scheme, wrong, project.purpose,
                capabilities=load_capabilities(project.capabilities_path), directivity=DIRECTIVITY,
                capabilities_path=Path(project.capabilities_path), engine_commit='test')
    except ValueError as error:
        errors.append(str(error))
    else:
        errors.append(None)
    changed = False
    if hasattr(c, 'source_ids'):
        changed = objective_fingerprint(c) != objective_fingerprint(c._replace(source_ids=c.source_ids[::-1]))
    return dict(errors=errors, fingerprint_changed=changed)


def empty() -> dict[str, object]:
    scheme, room, settings, project, _ = problem_case()
    zones = dict(scheme.scene.surface_zones_by_wall or {})
    material = zones['x0'][0].material
    assert isinstance(material, Assembly)
    spans = (('a', (0., 1.123), (0., scheme.scene.room_m.Lz)),
             ('b', (1.123, 1.173), (.123, .173)),
             ('c', (1.173, scheme.scene.room_m.Ly), (0., scheme.scene.room_m.Lz)),
             ('d', (1.123, 1.173), (0., .123)),
             ('e', (1.123, 1.173), (.173, scheme.scene.room_m.Lz)))
    zones['x0'] = tuple(RectangleZone(area='rectangle', id=name, u_m=u, v_m=v, material=material)
                        for name, u, v in spans)
    scheme = scheme.model_copy(update={'scene': scheme.scene.model_copy(update={'surface_zones_by_wall': zones})})
    c = cases.constants(scheme)
    index = tuple(c.patch_ids).index('x0@b')
    assert c.angle_offsets[index] == c.angle_offsets[index + 1], c.angles_deg[c.angle_offsets[index]:c.angle_offsets[index + 1]]
    def no_tmm() -> jax.Array:
        raise AssertionError('空角度不得呼叫 TMM（傳遞矩陣）')
    empty_dtypes = []
    for dtype in (jnp.float32, jnp.float64):
        value = impedance_at_angles_or_empty(jnp.asarray(c.frequencies_hz, dtype=dtype), jnp.empty(0, dtype=dtype), no_tmm)
        assert value.shape == (len(c.frequencies_hz), 0)
        empty_dtypes.append(str(value.dtype))
    keys = tuple((Wall.from_name(str(name).partition('@')[0]), str(name).partition('@')[2]) for name in c.patch_ids)
    wall_indices = [tuple(Wall.all()).index(wall) for wall, _ in keys]
    boundary = np.stack([room.matrices.boundary[w] * c.patch_areas_m2[i] / c.wall_areas[w]
                         for i, w in enumerate(wall_indices)])
    room = dataclasses.replace(room, patch_keys=keys, matrices=dataclasses.replace(room.matrices, boundary=boundary))
    kept = run.scene_wall_materials(scheme, capabilities=load_capabilities(project.capabilities_path), directivity=DIRECTIVITY)
    air, ranges = construction_environment(density_kg_m3=room.density_kg_m3, sound_speed_m_s=room.sound_speed_m_s)
    rows = []
    for adjustable in (False, True):
        presets = tuple('glass_wool_48k_50mm_gap50' if name == 'xL' or (adjustable and name == 'x0@b') else None for name in c.patch_ids)
        designs = run.designs_for(presets, settings, kept=kept)
        assert isinstance(designs[index].material if adjustable else designs[index].kept_material, Assembly)
        theta = jnp.asarray(g.baseline_theta(designs), dtype=jnp.float64)
        for single in (False, True):
            (value, aux), gradient = g.compiled_evaluator(factory(room, designs, c, single))(theta)
            aux = cast(SixCategoryChainAux, aux)
            rows.append(dict(adjustable=adjustable, single=single,
                finite=bool(np.all(np.isfinite([float(value), float(aux.product_total), *np.asarray(gradient)])))))
        if not adjustable:
            z = impedance_function(designs, c, air=air, ranges=ranges)(theta, c)
            pressure = reduced_pressures(reduced_constants(room), z[0], rigid_mask=jnp.zeros(len(designs), dtype=bool))
            aux = factory(room, designs, c, False)(theta)[1]
            materials = endpoint.materials_from_surfaces(designs, np.asarray(theta), settings)
            exported = scheme.model_copy(update={'scene': endpoint.scene_with_materials(scheme.scene, materials)})
            result = cases.numpy_result(exported, np.asarray(pressure).transpose(1, 2, 0))
            ranked = compare_results((result,), quality_targets=load_quality_targets(config_path('quality_targets.toml')), run_date=date.today())
            # 沿支 A 的正式評估／總分 oracle（獨立答案）；合成 ROM 不要求通過產品排名底線。
            expected, product = cases.numpy_costs(result)
            assert np.isfinite(expected).all()
            bound = float(np.dot(c.category_weights, _tolerances(expected)))
            numpy = dict(actual=float(aux.product_total), expected=float(product), bound=bound,
                         ranking_status=ranked.status_of(result.scheme.scheme_id).value)
            original = run.designs_for((None,) * len(designs), settings, kept=kept)
            baseline, _ = endpoint_six.baseline_impedances(original, c, model=room)
            baseline_finite = all(np.isfinite(np.asarray(part)).all() for part in baseline)
    return dict(rows=rows, empty_dtypes=empty_dtypes, zero_angle_patch='x0@b', zero_angle_count=int(c.angle_offsets[index + 1] - c.angle_offsets[index]),
                numpy=numpy, baseline_finite=bool(baseline_finite))


if __name__ == '__main__':
    functions = {'sources': sources, 'rejection': rejection, 'empty': empty}
    Path(sys.argv[2]).write_text(json.dumps(functions[sys.argv[1]](), allow_nan=False))
