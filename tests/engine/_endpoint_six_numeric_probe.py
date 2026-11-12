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
from aosr.materials.construction_models import Assembly, MeasuredImpedance
from aosr.physics.report_surfaces import WholeWallZone
from aosr.reporting.scheme import Scheme
from aosr.config.material_gradient import MaterialGradientSettings
from aosr.gradient.material_gradient import SurfaceDesign
from aosr.scoring.six_category_jax import SixCategoryConstants
from aosr.scoring.six_category_jax import constant_absorption_mask, six_category_costs
from tests.engine._frequency_batch_probe import problem
from tests.engine._six_category_endpoint_probe import _baseline_inputs
from tests.engine._six_category_jax_cases import array_inputs, numpy_result, scheme_case
from tests.engine._six_category_measured_probe import tables
from tests.engine.test_six_category_jax import _tolerances
from tests.engine.test_fem_reduced import _settings
from tests.engine.test_material_gradient import PURPOSE


def _with_measured_floor(scheme: Scheme) -> Scheme:
    """地板整面換成合成實測表（有虛部、表外用 x0 那份構造補值；``_six_category_measured_probe.tables``）。"""
    from tests.engine._six_category_jax_cases import constants

    zones = dict(scheme.scene.surface_zones_by_wall or {})
    base = next(zone.material for cells in zones.values() for zone in cells if isinstance(zone.material, Assembly))
    table, _ = tables(constants(scheme), base)
    zones['floor'] = (WholeWallZone(area='whole_wall', material=table),)
    return Scheme.model_validate(scheme.model_copy(update={'scene': scheme.scene.model_copy(
        update={'surface_zones_by_wall': zones})}).model_dump())


def _within_bound(scheme: Scheme, c: SixCategoryConstants, saved: endpoint_six.SixComparison, items: tuple[SurfaceDesign, ...],
                  values: np.ndarray, pressure: np.ndarray, settings: MaterialGradientSettings) -> bool:
    """JAX 終點分數對正式 numpy 排名總分，差在登記的六類容差加權和內（沿用 slow_six_category_endpoint 那一把）。"""
    from aosr.gradient.endpoint import materials_from_surfaces, scene_with_materials

    exported = scheme.model_copy(update={'scene': scene_with_materials(scheme.scene, materials_from_surfaces(items, values, settings))})
    ranked = compare_results((numpy_result(exported, pressure.transpose(1, 2, 0)),),
        quality_targets=load_quality_targets(config_path('quality_targets.toml')), run_date=date.today())
    costs = np.asarray([next(line.category_cost for line in ranked.rankable[0].categories
        if line.identity.category.value == name) for name in endpoint_six.CATEGORY_NAMES])
    assert saved.full.product_total is not None
    return abs(saved.full.product_total - ranked.rankable[0].total_cost) <= float(np.dot(c.category_weights, _tolerances(costs)))


def main(output: Path, measured: bool = False) -> None:
    scheme = scheme_case('zoned').model_copy(update={'purpose': PURPOSE})
    # 合成聲壓只看頻率軸與座位（array_inputs 自己組材料阻抗、收不了實測表），換地板之前先取。
    pressure_scheme = scheme
    if measured:
        scheme = _with_measured_floor(scheme)
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
    full = np.asarray(array_inputs(pressure_scheme, c)[4]).transpose(2, 0, 1)
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
        expected = six_category_costs(c, *z, jnp.asarray(pressure.transpose(1, 2, 0)),
                                     jnp.asarray(constant_absorption_mask(design_materials(items), c.union_frequencies_hz)))
        actual.append(np.asarray([*saved.full.categories, saved.full.objective_total, saved.full.product_total]).tobytes() ==
            np.asarray([*(float(v) for v in expected[:6]), float(expected.objective_total), float(expected.product_total)]).tobytes())
    reference = numpy_result(exported, full.transpose(1, 2, 0))
    ranking = compare_results((reference,), quality_targets=load_quality_targets(config_path('quality_targets.toml')), run_date=date.today())
    numpy_equal = six.endpoint.numpy_total == ranking.rankable[0].total_cost
    gain = six.improvement.full
    improved = [*gain.categories, gain.objective_total, gain.product_total]
    expected_gain = np.asarray([*six.baseline.full.categories, six.baseline.full.objective_total, six.baseline.full.product_total]) - np.asarray([*six.endpoint.full.categories, six.endpoint.full.objective_total, six.endpoint.full.product_total])
    improvement = np.asarray(improved).tobytes() == expected_gain.tobytes()
    bounds = [_within_bound(scheme, c, saved, items, values, pressure, settings)
              for saved, items, values, pressure in ((six.endpoint, designs, theta, full), (six.baseline, originals, np.empty(0), baseline))]
    kinds = sorted({type(material).__name__ for material in design_materials(originals)})
    output.write_text(json.dumps({'jax_bitwise': actual, 'numpy_bitwise': numpy_equal, 'improvement_bitwise': improvement,
                                  'within_registered_sum_bound': bounds, 'measured_kept': MeasuredImpedance.__name__ in kinds}))


if __name__ == '__main__':
    main(Path(sys.argv[1]), measured=sys.argv[2:] == ['measured'])
