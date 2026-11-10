"""小分區房真搜尋與終點：完整解只兩套，另從保留聲壓獨立求答案。"""
from __future__ import annotations

from collections.abc import Callable
import json
import shutil
import sys
import traceback
from pathlib import Path
from collections.abc import Mapping, Sequence
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.capabilities import load_capabilities, CapabilityTable
from aosr.config.construction_environment import construction_environment
from aosr.config.directivity_defaults import load_directivity_defaults, DirectivityDefaults
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ, LowFrequencyAxis
from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.gradient.material_gradient import SurfaceDesign
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.gradient import endpoint, endpoint_run, endpoint_six, run
from aosr.gradient.report import render_report
from aosr.gradient.six_category_chain import impedance_function, design_materials
from aosr.materials.construction_models import MeasuredImpedance, SurfaceMaterial
from aosr.physics.report_surfaces import WholeWallZone
from aosr.materials.construction import surface_impedance, surface_impedance_at_angles
from aosr.reporting.compare import compare_results
from aosr.config.quality_targets import load_quality_targets
from datetime import date
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
from aosr.reporting.pipeline import run_scheme
from aosr.scoring.six_category_jax import SixCategoryArrays, SixCategoryConstants, build_constants, constant_material_flags, six_category_costs
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, ReasonCode, TimbreChannelsPayload
from aosr.reporting.validation import checked_inputs
from tests.engine._six_category_jax_cases import numpy_result, scheme_case
from tests.engine.test_fem_reduced import _settings
from tests.engine.test_material_gradient_cli import (
    BAND_NAME, FREQUENCY_NAME, NARROWED_STEPS, _run, _settings_text, _start_args,
)
from tests.engine._precision_contracts import contract_value
from tests.engine.test_six_category_jax import _tolerances


def _small_scheme(measured: bool = False) -> Scheme:
    """支 A 已驗案例等比例縮半（二進位精確縮放），避免對稱小房的反彈邊界退化。"""
    scheme = scheme_case('changed_wall' if measured else 'zoned')
    scene = scheme.scene
    assert scene.surface_zones_by_wall is not None
    zones = {name: tuple(zone.model_copy(update={'u_m': tuple(v / 2 for v in zone.u_m),
        'v_m': tuple(v / 2 for v in zone.v_m)}) if isinstance(zone, RectangleZone) else zone for zone in rows)
        for name, rows in scene.surface_zones_by_wall.items()}
    if measured:
        assembly = zones['x0'][0].material
        # 搜尋用新地板；終點的原牆基準保留裸實測表，專考基準拒收入口。
        zones['floor'] = (WholeWallZone(area='whole_wall', material=MeasuredImpedance.model_validate({
            'kind': 'measured_impedance', 'source_and_conditions': 'endpoint fixture impedance tube',
            'points': [{'frequency_hz': 1000., 'real_pa_s_m': 12000., 'imag_pa_s_m': -300.},
                       {'frequency_hz': 4000., 'real_pa_s_m': 20000., 'imag_pa_s_m': -100.}],
            'base_assembly': assembly.model_dump()})),)
    room = scene.room_m
    receivers = scheme.receiver_set.model_copy(update={'points': tuple(point.model_copy(update={
        'position_m': tuple(v / 2 for v in point.position_m)}) for point in scheme.receiver_set.points)})
    return Scheme.model_validate(scheme.model_copy(update={'scene': scene.model_copy(update={
        'low_frequency_axis': LowFrequencyAxis.SEARCH, 'room_m': Room(room.Lx / 2, room.Ly / 2, room.Lz / 2),
        'surface_zones_by_wall': zones}), 'receiver_set': receivers,
        'purpose': 'endpoint_scheme_purpose_only',
        'speakers': {name: Point(point.x / 2, point.y / 2, point.z / 2) for name, point in scheme.speakers.items()}}).model_dump())


def _build(directory: Path, measured: bool) -> None:
    scheme = _small_scheme(measured)
    scheme = scheme.model_copy(update={'speakers': dict(reversed(tuple(scheme.speakers.items())))})
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
    raised = False
    with patch.object(endpoint_six, 'run_scheme', side_effect=ValueError('能量鍵錯誤')):
        try:
            endpoint_six.numpy_scores(context, scheme, pressure)
        except ValueError as error:
            raised = str(error) == '能量鍵錯誤'
    original = next(item for item in reference.candidate.evaluations if item.category.value == endpoint_six.CATEGORY_NAMES[0])
    rest = tuple(item for item in reference.candidate.evaluations if item.category != original.category)
    missing = CategoryEvaluation.model_validate({**original.model_dump(), 'state': EvaluationState.UNAVAILABLE,
        'category_cost': None, 'payload': None, 'raw_quantities': (), 'flags': (),
        'reason_codes': (ReasonCode.SOLVER_UNAVAILABLE,)})
    candidate = reference.candidate.model_copy(update={'evaluations': (missing, *rest)})
    incomplete = reference.model_copy(update={'candidate': candidate})
    with patch.object(endpoint_six, 'run_scheme', return_value=incomplete):
        costs, total, reason = endpoint_six.numpy_scores(context, scheme, pressure)
    missing_recorded = total is None and costs[0] is None and all(v is not None for v in costs[1:]) and reason is not None and 'solver_unavailable' in reason
    timbre = next(e for e in reference.candidate.evaluations if isinstance(e.payload, TimbreChannelsPayload))
    assert isinstance(timbre.payload, TimbreChannelsPayload)
    wrong = timbre.model_copy(update={'payload': timbre.payload.model_copy(update={'channel_group_fingerprint': 'f' * 64})})
    mismatched = reference.model_copy(update={'candidate': reference.candidate.model_copy(update={
        'evaluations': tuple(wrong if e.category == timbre.category else e for e in reference.candidate.evaluations)})})
    official = compare_results((mismatched,), quality_targets=load_quality_targets(config_path('quality_targets.toml')), run_date=date.today())
    with patch.object(endpoint_six, 'run_scheme', return_value=mismatched):
        costs, total, reason = endpoint_six.numpy_scores(context, scheme, pressure)
    missing_recorded &= (not official.rankable and all(v is not None for v in costs) and total is None
        and reason == official.status_of(scheme.scheme_id).value)
    # 精確考不可估診斷的序列化入口，不假定哪一種有效材料必定讓 f_s 失效。
    notes = endpoint_six._notes(arrays._replace(schroeder_frequency_hz=jnp.asarray(jnp.nan)))
    return raised, missing_recorded, notes.schroeder_frequency_hz is None


def _baseline_inputs(items: tuple[SurfaceDesign, ...], c: SixCategoryConstants, air: PhysicsConstants,
                     ranges: PorousModelRanges) -> tuple[tuple[SurfaceMaterial, ...], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    materials = tuple(item.kept_material for item in items if item.kept_material is not None)
    assert len(materials) == len(items)
    def z(material: SurfaceMaterial, axis: np.ndarray, angles: np.ndarray | None = None) -> jax.Array:
        if angles is None:
            return surface_impedance(material, jnp.asarray(axis), air=air, ranges=ranges)[0]
        return surface_impedance_at_angles(material, jnp.asarray(axis), jnp.asarray(angles), air=air, ranges=ranges)[0]
    impedances = (jnp.stack([z(m, c.fem_frequencies_hz) for m in materials]),
        jnp.stack([z(m, c.frequencies_hz) for m in materials]),
        jnp.concatenate([z(m, c.frequencies_hz, c.angles_deg[int(a):int(b)]) for m,a,b in zip(materials, c.angle_offsets[:-1], c.angle_offsets[1:], strict=True)], axis=1),
        jnp.stack([z(m, c.union_frequencies_hz, c.paris_angles_deg) for m in materials]))
    return materials, impedances


def _numpy_reference(scheme: Scheme, pressure: np.ndarray) -> tuple[SchemeResult, np.ndarray, float]:
    reference = numpy_result(scheme, pressure.transpose(1, 2, 0))
    ranked = compare_results((reference,), quality_targets=load_quality_targets(config_path('quality_targets.toml')), run_date=date.today())
    costs = np.asarray([next(line.category_cost for line in ranked.rankable[0].categories
        if line.identity.category.value == name) for name in endpoint_six.CATEGORY_NAMES])
    return reference, costs, ranked.rankable[0].total_cost


def _oracle(folder: Path, checked: endpoint.EndpointResult, baseline: np.ndarray) -> tuple[list[bool], list[bool], list[bool], bool, tuple[bool, bool, bool]]:
    project = run.read_project(folder)
    scheme, model, settings = run.check_inputs(project)
    assert tuple(channel.speaker_id for channel in scheme.channel_group.channels) != model.source_names
    assert scheme.purpose != project.purpose
    channels = {channel.speaker_id: channel for channel in scheme.channel_group.channels}
    scheme = scheme.model_copy(update={'purpose': project.purpose, 'channel_group': scheme.channel_group.model_copy(
        update={'channels': tuple(channels[name] for name in model.source_names)})})
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
        if reduced is None:
            materials, impedances = _baseline_inputs(items, c, air, ranges)
        else:
            materials = design_materials(items)
            impedances = impedance_function(items, c, air=air, ranges=ranges)(jnp.asarray(theta), c)
        if reduced is None:
            rows = np.asarray(impedances[0])
            values = []
            for i, frequency in enumerate(model.frequencies_hz):
                k2, coeff = system_coefficients(frequency, dict(zip(model.patch_keys, rows[:, i], strict=True)),
                    density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s, patch_keys=model.patch_keys)
                values.append(evaluate_reduced(model.matrices, k2, coeff))
            reduced = np.asarray(values)
        for pressure, scores, notes in ((reduced, saved.reduced, saved.crossover), (full, saved.full, saved.crossover)):
            arrays = six_category_costs(c, *impedances, jnp.asarray(pressure.transpose(1, 2, 0)), jnp.asarray(constant_material_flags(materials)))
            expected = [*(float(v) for v in arrays[:6]), float(arrays.objective_total), float(arrays.product_total), float(arrays.low_frequency_objective)]
            actual = [*scores.categories, scores.objective_total, scores.product_total, scores.low_frequency_objective]
            equal = np.asarray(actual, dtype=float).tobytes() == np.asarray(expected).tobytes()
            equal &= notes.schroeder_frequency_hz is not None and notes.schroeder_frequency_hz.hex() == float(arrays.schroeder_frequency_hz).hex()
            equal &= notes.capped_by_upper_limit == bool(arrays.capped_by_upper_limit)
            equal &= notes.reverberation_band_count == int(arrays.reverberation_band_count)
            direct.append(bool(equal))
            axes_distinct |= float(arrays.objective_total) != float(arrays.product_total)
        surface_materials = endpoint.materials_from_surfaces(items, theta, settings)
        material_scheme = scheme.model_copy(update={'scene': endpoint.scene_with_materials(scheme.scene, surface_materials)})
        reference, costs, total = _numpy_reference(material_scheme, full)
        assert saved.numpy_total is not None
        numpy_equal.append(np.asarray(saved.numpy_categories).tobytes() == costs.tobytes() and saved.numpy_total.hex() == total.hex())
        bound = float(np.dot(c.category_weights, _tolerances(costs)))
        assert saved.full.product_total is not None
        bound_equal.append(abs(saved.full.product_total - total) <= bound)
    context = endpoint_six.ScoringContext(scheme, c, caps, directivity, 'calc-v1:' + '0' * 64, 'phys-v1:' + '0' * 64, 'test')
    failures = _unavailable_controls(context, scheme, baseline, reference, arrays)
    return direct, numpy_equal, bound_equal, axes_distinct, failures


def _bad_energy_key_entry(folder: Path, checked: endpoint.EndpointResult, baseline: np.ndarray) -> bool:
    target = folder.parent.parent / 'bad-energy-key' / folder.name
    shutil.copytree(folder, target, ignore=shutil.ignore_patterns('endpoint.json'))
    original = run_scheme
    def wrong(scheme: Scheme | object, *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
              quality_targets_path: Path, engine_commit: str, program_fingerprint: str, physics_identity: str,
              run_date: date, fem_energies: Mapping[tuple[str, str], Sequence[float]] | None = None) -> SchemeResult:
        assert fem_energies is not None
        injected = dict(fem_energies)
        key = next(iter(injected))
        injected[('wrong-source', key[1])] = injected.pop(key)
        return original(scheme, capabilities=capabilities, directivity=directivity,
            quality_targets_path=quality_targets_path, engine_commit=engine_commit,
            program_fingerprint=program_fingerprint, physics_identity=physics_identity,
            run_date=run_date, fem_energies=injected)
    with (patch.object(endpoint_run, 'endpoint_check', return_value=checked),
          patch.object(endpoint_run, 'full_model_pressures', return_value=baseline),
          patch.object(endpoint_six, 'run_scheme', wrong)):
        try:
            endpoint_run.run_endpoint(target)
        except ValueError:
            return not (target / 'endpoint.json').exists()
    return False


def _capture[**P, R](function: Callable[P, R], saved: list[R]) -> Callable[P, R]:
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        value = function(*args, **kwargs)
        saved.append(value)
        return value
    return wrapped


def _same_subtraction(base: float | None, end: float | None, gain: float | None) -> bool:
    assert base is not None and end is not None and gain is not None
    return gain.hex() == (base - end).hex()


def _start(directory: Path, *, measured: bool = False) -> Path:
    _build(directory, measured)
    presets = ('glass_wool_48k_50mm_gap50,none,glass_wool_48k_50mm_gap50,none,none,none' if measured else
               'none,none,glass_wool_48k_50mm_gap50,none,none,none,none')
    start: dict[str, object]
    if measured:
        from aosr.gradient import cli, worker
        def search(folder: Path, mode: str, lock_fd: int, *, device: str = 'cpu') -> int:
            # 同程序跑真 worker（搜尋工人），讓新檢查的暫時替換涵蓋 start 的父子入口。
            return worker.run_worker(folder, 'start', should_stop=run.stop_requested(folder))
        with (patch.object(run, 'require_supported_original_walls', return_value=None),
              patch.object(cli, '_spawn_and_wait', side_effect=search)):
            start = {'code': cli.main(_start_args(directory, directory / 'root', presets=presets)[3:])}
    else:
        start = _run(directory, _start_args(directory, directory / 'root', presets=presets))
    assert start['code'] == 0, start
    (folder,) = (directory / 'root').iterdir()
    return folder


def measured_rejection(directory: Path) -> None:
    """模擬新檢查之前的真搜尋結果；恢復檢查後，在完整求解之前拒收且無殘檔。"""
    folder = _start(directory, measured=True)
    before = {path.name for path in folder.iterdir()}
    error_type = error_message = error_origin = None
    try:
        with (patch.object(endpoint_run, 'endpoint_check', side_effect=AssertionError('不准求解終點')),
              patch.object(endpoint_run, 'full_model_pressures', side_effect=AssertionError('不准求解原牆')),
              patch.object(FullSolver, 'solve', side_effect=AssertionError('不准完整求解'))):
            endpoint_run.run_endpoint(folder)
    except ValueError as error:
        error_type, error_message = type(error).__name__, str(error)
        error_origin = traceback.extract_tb(error.__traceback__)[-1].name
    after = {path.name for path in folder.iterdir()}
    output = dict(error_type=error_type, error_message=error_message, error_origin=error_origin,
        endpoint_exists=(folder / endpoint_run.ENDPOINT_FILE).exists(), new_files=sorted(after - before),
        endpoint_temporary_files=sorted(name for name in after if name.startswith('.endpoint-')))
    (directory / 'rejection.json').write_text(json.dumps(output, ensure_ascii=False))


def main(directory: Path) -> None:
    folder = _start(directory)
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
        bad_key_failed = _bad_energy_key_entry(folder, captured[0], baseline[0])

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
        within_registered_sum_bound=bound_equal, axes_distinct=distinct, improvement_bitwise=bool(equal), numpy_failure_recorded=failures[0],
        numpy_missing_category_recorded=failures[1], invalid_crossover_recorded=failures[2], source_order_reversed=True,
        purpose_overridden=True, bad_energy_key_failed=bad_key_failed,
        diagnostics=all(v.schroeder_frequency_hz is not None and v.schroeder_frequency_hz > 0 and isinstance(v.capped_by_upper_limit, bool) and v.reverberation_band_count > 0
            for item in (six.baseline, six.endpoint) for v in (item.crossover,)),
        all_scores_finite=all(v is not None for item in (six.baseline, six.endpoint) for score in (item.reduced, item.full)
            for v in (*score.categories, score.objective_total, score.product_total)))
    (directory / 'measure.json').write_text(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    (measured_rejection if len(sys.argv) > 2 else main)(Path(sys.argv[1]))
