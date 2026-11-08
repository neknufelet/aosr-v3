"""小分區房真搜尋與終點：完整解只兩套，另從保留聲壓獨立求答案。"""
from __future__ import annotations

from collections.abc import Callable
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ, LowFrequencyAxis
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.gradient import endpoint, endpoint_run, endpoint_six, run
from aosr.gradient.report import render_report
from aosr.gradient.six_category_chain import _impedance_function, _materials
from aosr.physics.fem_reduced import FullSolver, evaluate_reduced, system_coefficients
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics.fem_reduced_check import validate_reduced
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, save_validation_receipt
from aosr.physics.fem_reduced_store import save_reduced
from aosr.physics import three_lane_report, three_lane_report_batch
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.reporting.scheme import Scheme
from aosr.reporting.result import SchemeResult
from aosr.scoring.six_category_jax import SixCategoryArrays, build_constants, constant_material_flags, six_category_costs
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, ReasonCode
from aosr.reporting.validation import checked_inputs
from tests.engine._six_category_jax_cases import numpy_costs, numpy_result, scheme_case
from tests.engine.test_fem_reduced import _settings
from tests.engine.test_material_gradient_cli import (
    BAND_NAME, FREQUENCY_NAME, NARROWED_STEPS, _run, _settings_text, _start_args,
)
from tests.engine._precision_contracts import contract_value
from tests.engine.test_six_category_jax import _tolerances


def _small_scheme() -> Scheme:
    """支 A 已驗案例等比例縮半（二進位精確縮放），避免對稱小房的反彈邊界退化。"""
    scheme = scheme_case('zoned')
    scene = scheme.scene
    assert scene.surface_zones_by_wall is not None
    zones = {name: tuple(zone.model_copy(update={'u_m': tuple(v / 2 for v in zone.u_m),
        'v_m': tuple(v / 2 for v in zone.v_m)}) if isinstance(zone, RectangleZone) else zone for zone in rows)
        for name, rows in scene.surface_zones_by_wall.items()}
    room = scene.room_m
    receivers = scheme.receiver_set.model_copy(update={'points': tuple(point.model_copy(update={
        'position_m': tuple(v / 2 for v in point.position_m)}) for point in scheme.receiver_set.points)})
    return Scheme.model_validate(scheme.model_copy(update={'scene': scene.model_copy(update={
        'low_frequency_axis': LowFrequencyAxis.SEARCH, 'room_m': Room(room.Lx / 2, room.Ly / 2, room.Lz / 2),
        'surface_zones_by_wall': zones}), 'receiver_set': receivers,
        'speakers': {name: Point(point.x / 2, point.y / 2, point.z / 2) for name, point in scheme.speakers.items()}}).model_dump())


def _build(directory: Path) -> None:
    scheme = _small_scheme()
    (directory / 'scheme.json').write_text(scheme.model_dump_json())
    (directory / 'settings.toml').write_text(_settings_text(NARROWED_STEPS))
    capabilities = load_capabilities(config_path('capabilities.toml'))
    directivity = load_directivity_defaults(config_path('directivity_defaults.toml'))
    operators = export_operators(scheme, capabilities=capabilities, directivity=directivity)
    axis = np.asarray(FEM_LANE_FREQUENCIES_HZ)
    def materials(seed: int, count: int) -> list[dict[tuple[Wall, str], tuple[complex, ...]]]:
        rng = np.random.default_rng(seed)
        return [{key: tuple(complex(z) for z in 411.6 * (rng.uniform(2, 12) - 1j * rng.uniform(.5, 2) * 100 / axis))
            for key in operators.patch_keys} for _ in range(count)]
    model = build_reduced_model(operators, FEM_LANE_FREQUENCIES_HZ, materials(561, 8), settings=_settings())
    (directory / 'good').mkdir()
    save_reduced(model, directory / 'good' / 'rom.npz')
    validation = validate_reduced(model, operators, materials(562, 2))
    save_validation_receipt(directory / 'good' / 'rom.npz', validation, rulers=HoldoutRulers(
        frequency=Ruler(FREQUENCY_NAME, contract_value(FREQUENCY_NAME)), band=Ruler(BAND_NAME, contract_value(BAND_NAME))), seconds=1.)
    # ROM 保持原聲源序，JSON 字典反序；搜尋准收的同一輸入必須也能完成終點。
    scheme = scheme.model_copy(update={'speakers': dict(reversed(tuple(scheme.speakers.items())))})
    (directory / 'scheme.json').write_text(scheme.model_dump_json())


def _unavailable_controls(context: endpoint_six.ScoringContext, scheme: Scheme, pressure: np.ndarray,
                          reference: SchemeResult, arrays: SixCategoryArrays) -> tuple[bool, bool, bool]:
    with patch.object(endpoint_six, 'run_scheme', side_effect=ValueError('正式 numpy 路徑不可估測試')):
        unavailable = endpoint_six._numpy_scores(context, scheme, pressure)
    raised = unavailable == ((None,) * 6, None, '正式 numpy 路徑不可估測試')
    original, *rest = reference.candidate.evaluations
    missing = CategoryEvaluation.model_validate({**original.model_dump(), 'state': EvaluationState.UNAVAILABLE,
        'category_cost': None, 'payload': None, 'raw_quantities': (), 'flags': (),
        'reason_codes': (ReasonCode.SOLVER_UNAVAILABLE,)})
    candidate = reference.candidate.model_copy(update={'evaluations': (missing, *rest)})
    incomplete = reference.model_copy(update={'candidate': candidate})
    with patch.object(endpoint_six, 'run_scheme', return_value=incomplete):
        costs, total, reason = endpoint_six._numpy_scores(context, scheme, pressure)
    missing_recorded = total is None and costs[0] is None and all(v is not None for v in costs[1:]) and reason is not None and 'solver_unavailable' in reason
    # 精確考不可估診斷的序列化入口，不假定哪一種有效材料必定讓 f_s 失效。
    notes = endpoint_six._notes(arrays._replace(schroeder_frequency_hz=jnp.asarray(jnp.nan)))
    return raised, missing_recorded, notes.schroeder_frequency_hz is None


def _oracle(folder: Path, checked: endpoint.EndpointResult, baseline: np.ndarray) -> tuple[list[bool], list[bool], list[bool], bool, tuple[bool, bool, bool]]:
    project = run.read_project(folder)
    scheme, model, settings = run.check_inputs(project)
    assert tuple(scheme.speakers) != model.source_names
    result = endpoint_run.read_result(folder)
    caps = load_capabilities(project.capabilities_path)
    directivity = load_directivity_defaults(config_path('directivity_defaults.toml'))
    _, pairs = checked_inputs(scheme, capabilities=caps, directivity=directivity)
    c = build_constants(scheme, pair_inputs={key: value[1] for key, value in pairs.items()})
    kept = run.scene_wall_materials(scheme, capabilities=caps, directivity=directivity)
    designs = run.final_designs([(s.surface_id, s.preset_id) for s in result.surfaces], settings, kept)
    originals = run.designs_for((None,) * len(designs), settings, kept=kept)
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    record = endpoint_run.read_endpoint(folder)
    assert record.six_category is not None
    direct, numpy_equal, bound_equal = [], [], []
    axes_distinct = False
    for items, theta, full, reduced, saved in (
        (designs, np.asarray(checked.report.exported_theta), checked.full_pressure, checked.reduced_pressure, record.six_category.endpoint),
        (originals, np.empty(0), baseline, None, record.six_category.baseline)):
        impedances = _impedance_function(items, c, air=air, ranges=ranges)(jnp.asarray(theta), c)
        if reduced is None:
            rows = np.asarray(impedances[0])
            values = []
            for i, frequency in enumerate(model.frequencies_hz):
                k2, coeff = system_coefficients(frequency, dict(zip(model.patch_keys, rows[:, i], strict=True)),
                    density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s, patch_keys=model.patch_keys)
                values.append(evaluate_reduced(model.matrices, k2, coeff))
            reduced = np.asarray(values)
        for pressure, scores, notes in ((reduced, saved.reduced, saved.reduced_crossover), (full, saved.full, saved.full_crossover)):
            arrays = six_category_costs(c, *impedances, jnp.asarray(pressure.transpose(1, 2, 0)), jnp.asarray(constant_material_flags(_materials(items))))
            expected = [*(float(v) for v in arrays[:6]), float(arrays.objective_total), float(arrays.product_total), float(arrays.low_frequency_objective)]
            actual = [*scores.categories, scores.objective_total, scores.product_total, scores.low_frequency_objective]
            equal = np.asarray(actual, dtype=float).tobytes() == np.asarray(expected).tobytes()
            equal &= notes.schroeder_frequency_hz is not None and notes.schroeder_frequency_hz.hex() == float(arrays.schroeder_frequency_hz).hex()
            equal &= notes.capped_by_upper_limit == bool(arrays.capped_by_upper_limit)
            equal &= notes.reverberation_band_count == int(arrays.reverberation_band_count)
            direct.append(bool(equal))
            axes_distinct |= float(arrays.objective_total) != float(arrays.product_total)
        materials = endpoint.materials_from_surfaces(items, theta, settings)
        material_scheme = scheme.model_copy(update={'scene': endpoint.scene_with_materials(scheme.scene, materials)})
        reference = numpy_result(material_scheme, full.transpose(1, 2, 0))
        costs, total = numpy_costs(reference)
        assert saved.numpy_total is not None
        numpy_equal.append(np.asarray(saved.numpy_categories).tobytes() == costs.tobytes() and saved.numpy_total.hex() == total.hex())
        bound = float(np.dot(c.category_weights, _tolerances(costs)))
        bound_equal.append(saved.implementation_bound == bound)
    context = endpoint_six.ScoringContext(scheme, c, caps, directivity, 'calc-v1:' + '0' * 64, 'phys-v1:' + '0' * 64, 'test')
    failures = _unavailable_controls(context, scheme, baseline, reference, arrays)
    return direct, numpy_equal, bound_equal, axes_distinct, failures


def _capture[**P, R](function: Callable[P, R], saved: list[R]) -> Callable[P, R]:
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        value = function(*args, **kwargs)
        saved.append(value)
        return value
    return wrapped


def _same_subtraction(base: float | None, end: float | None, gain: float | None) -> bool:
    assert base is not None and end is not None and gain is not None
    return gain.hex() == (base - end).hex()


def _over_bound_entry(folder: Path, checked: endpoint.EndpointResult, baseline: np.ndarray) -> bool:
    """重用已解聲壓，讓正式終點入口碰到超界；只擾動模擬 numpy 答案，不加完整解。"""
    target = folder.parent.parent / 'overbound' / folder.name
    shutil.copytree(folder, target)
    original = endpoint_six._numpy_scores
    def shifted(context: endpoint_six.ScoringContext, scheme: Scheme, pressure: np.ndarray,
                ) -> tuple[endpoint_six.Categories, float | None, str | None]:
        costs, total, reason = original(context, scheme, pressure)
        bound = endpoint_six.implementation_bound(context.constants, costs)
        assert total is not None and bound is not None and costs[0] is not None and reason is None
        delta = bound + bound
        categories = endpoint_six._categories((costs[0] + delta / float(context.constants.category_weights[0]), *costs[1:]))
        return categories, total + delta, None
    with (patch.object(endpoint_run, 'endpoint_check', return_value=checked),
          patch.object(endpoint_run, 'full_model_pressures', return_value=baseline),
          patch.object(endpoint_six, '_numpy_scores', shifted)):
        record = endpoint_run.run_endpoint(target)
    assert record.six_category is not None
    return (record.six_category.endpoint.implementation_over_bound is True
        and record.six_category.baseline.implementation_over_bound is True
        and '超界（只標示、不阻擋）' in render_report(target))


def main(directory: Path) -> None:
    _build(directory)
    start = _run(directory, _start_args(directory, directory / 'root', presets='none,none,glass_wool_48k_50mm_gap50,none,none,none,none'))
    assert start['code'] == 0, start
    (folder,) = (directory / 'root').iterdir()
    solves: list[np.ndarray] = []
    captured: list[endpoint.EndpointResult] = []
    baseline: list[np.ndarray] = []
    with (
        patch.object(FullSolver, 'solve', _capture(FullSolver.solve, solves)),
        patch.object(endpoint_run, 'endpoint_check', _capture(endpoint.endpoint_check, captured)),
        patch.object(endpoint_run, 'full_model_pressures', _capture(endpoint.full_model_pressures, baseline)),
        patch.object(three_lane_report_batch, 'solve_fem_energies_many', side_effect=AssertionError('不准多解 FEM')),
        patch.object(three_lane_report, '_solve_fem_energies', side_effect=AssertionError('不准多解 FEM')),
        patch.object(three_lane_report, '_solve_fem_energy', side_effect=AssertionError('不准多解 FEM')),
    ):
        record = endpoint_run.run_endpoint(folder)
        direct, numpy_equal, bound_equal, distinct, failures = _oracle(folder, captured[0], baseline[0])
        over_bound_entry = _over_bound_entry(folder, captured[0], baseline[0])
    six = record.six_category
    assert six is not None
    equal = True
    for name in ('reduced', 'full'):
        b, e, gain = getattr(six.baseline, name), getattr(six.endpoint, name), getattr(six.improvement, name)
        for key in ('objective_total', 'product_total', 'proxy_db', 'low_frequency_objective'):
            equal &= getattr(gain, key).hex() == (getattr(b, key) - getattr(e, key)).hex()
        equal &= all(_same_subtraction(b, e, g) for b, e, g in zip(b.categories, e.categories, gain.categories, strict=True))
    equal &= _same_subtraction(six.baseline.numpy_total, six.endpoint.numpy_total, six.improvement.numpy_total)
    equal &= all(_same_subtraction(b, e, g) for b, e, g in zip(six.baseline.numpy_categories, six.endpoint.numpy_categories, six.improvement.numpy_categories, strict=True))
    output = dict(version=record.format_version, record=record.model_dump(mode='json'), report=render_report(folder),
        solves=len(solves), expected_solves=2 * len(captured[0].full_pressure), jax_bitwise=direct, numpy_bitwise=numpy_equal,
        bound_bitwise=bound_equal, axes_distinct=distinct, improvement_bitwise=bool(equal), numpy_failure_recorded=failures[0],
        numpy_missing_category_recorded=failures[1], invalid_crossover_recorded=failures[2], source_order_reversed=True,
        over_bound_entry_allowed=over_bound_entry,
        diagnostics=all(v.schroeder_frequency_hz is not None and v.schroeder_frequency_hz > 0 and isinstance(v.capped_by_upper_limit, bool) and v.reverberation_band_count > 0
            for item in (six.baseline, six.endpoint) for v in (item.reduced_crossover, item.full_crossover)),
        all_scores_finite=all(v is not None for item in (six.baseline, six.endpoint) for score in (item.reduced, item.full)
            for v in (*score.categories, score.objective_total, score.product_total)))
    (directory / 'measure.json').write_text(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
