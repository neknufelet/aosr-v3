"""終點接線的獨立小矩陣控制；不建 FEM，也不沿用前次測試答案。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient import endpoint_six, run, material_gradient
from aosr.geometry.shoebox import Wall
from aosr.gradient.six_category_chain import impedance_function, design_materials
from aosr.physics.fem_reduced import BuildLedger, ReducedMatrices, ReducedModel
from aosr.reporting.compare import compare_results
from aosr.scoring.six_category_jax import constant_material_flags, six_category_costs
from tests.engine._frequency_batch_probe import problem
from tests.engine._six_category_endpoint_probe import _baseline_inputs
from tests.engine._six_category_jax_cases import array_inputs, numpy_result, scheme_case
from tests.engine.test_fem_reduced import _settings
from tests.engine.test_material_gradient import PURPOSE


def main(output: Path) -> None:
    scheme = scheme_case('zoned').model_copy(update={'purpose': PURPOSE})
    scheme = scheme.model_copy(update={'scene': scheme.scene.model_copy(update={'low_frequency_axis': LowFrequencyAxis.SEARCH})})
    channels = scheme.channel_group.channels[::-1]
    scheme = scheme.model_copy(update={'channel_group': scheme.channel_group.model_copy(update={'channels': channels})})
    names = tuple(channel.speaker_id for channel in channels)
    caps = load_capabilities(config_path('capabilities.toml'))
    directivity = load_directivity_defaults(config_path('directivity_defaults.toml'))
    # 幾何與正式材料入口沿用既有案例；小矩陣只供原牆替身，不冒充經驗證 ROM。
    from tests.engine._six_category_jax_cases import constants
    c = constants(scheme)
    small, _, _ = problem(len(c.fem_frequencies_hz))
    receivers = np.asarray(small.receivers)[np.arange(len(scheme.receiver_set.points)) % len(small.receivers)]
    matrices = ReducedMatrices(np.asarray(small.stiffness), np.asarray(small.mass),
        np.concatenate((small.boundary, small.boundary[:1])), np.asarray(small.sources), receivers)
    ledger = BuildLedger(0, 0, len(small.mass), (), 'synthetic', {}, _settings())
    keys = tuple((Wall.from_name(str(name).partition('@')[0]), str(name).partition('@')[2]) for name in c.patch_ids)
    model = ReducedModel(matrices, names, tuple(p.receiver_id for p in scheme.receiver_set.points),
        tuple(c.fem_frequencies_hz), scheme.scene.density_kg_m3, scheme.scene.sound_speed_m_s,
        '0' * 64, 'phys-v1:' + '0' * 64, '0' * 64, ledger, keys)
    settings = load_material_gradient_settings(config_path('material_gradient.toml'))
    kept = run.scene_wall_materials(scheme, capabilities=caps, directivity=directivity)
    designs = run.designs_for((None, None, 'glass_wool_48k_50mm_gap50', None, None, None, None), settings, kept=kept)
    originals = run.designs_for((None,) * len(designs), settings, kept=kept)
    theta = np.asarray(material_gradient.baseline_theta(designs), dtype=np.float64)
    full = np.asarray(array_inputs(scheme, c)[4]).transpose(2, 0, 1)
    reduced = full[::-1].copy()
    baseline = full.conjugate()[::-1].copy()
    context = endpoint_six.ScoringContext(scheme, c, caps, directivity, 'calc-v1:' + '0' * 64, 'phys-v1:' + '0' * 64, 'test')
    from aosr.gradient.endpoint import materials_from_surfaces, scene_with_materials
    exported = scheme.model_copy(update={'scene': scene_with_materials(scheme.scene, materials_from_surfaces(designs, theta, settings))})
    six = endpoint_six.score_endpoint(context, model, designs, theta, full, reduced, originals, baseline, exported)
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    actual = []
    for items, values, pressure, saved in ((designs, theta, full, six.endpoint), (originals, np.empty(0), baseline, six.baseline)):
        z = (impedance_function(items, c, air=air, ranges=ranges)(jnp.asarray(values), c) if items is designs else
             _baseline_inputs(items, c, air, ranges)[1])
        expected = six_category_costs(c, *z, jnp.asarray(pressure.transpose(1, 2, 0)), jnp.asarray(constant_material_flags(design_materials(items))))
        actual.append(np.asarray([*saved.full.categories, saved.full.objective_total, saved.full.product_total]).tobytes() ==
            np.asarray([*(float(v) for v in expected[:6]), float(expected.objective_total), float(expected.product_total)]).tobytes())
    reference = numpy_result(exported, full.transpose(1, 2, 0))
    ranking = compare_results((reference,), quality_targets=load_quality_targets(config_path('quality_targets.toml')), run_date=date.today())
    numpy_equal = six.endpoint.numpy_total == ranking.rankable[0].total_cost
    gain = six.improvement.full
    improved = [*gain.categories, gain.objective_total, gain.product_total]
    expected_gain = np.asarray([*six.baseline.full.categories, six.baseline.full.objective_total, six.baseline.full.product_total]) - np.asarray([*six.endpoint.full.categories, six.endpoint.full.objective_total, six.endpoint.full.product_total])
    improvement = np.asarray(improved).tobytes() == expected_gain.tobytes()
    output.write_text(json.dumps({'jax_bitwise': actual, 'numpy_bitwise': numpy_equal, 'improvement_bitwise': improvement}))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
