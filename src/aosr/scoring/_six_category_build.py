"""六類常數的 CPU（中央處理器）準備；只依賴物理層的已驗報表輸入。

Scheme（方案）只用結構介面，呼叫端交 checked_inputs 的逐對 ReportInput，避免評分層反向引用編排層。
面片順序用 gradient.boundary 同一個 boundary_patches 核心；考卷另與 scheme_patch_keys 直接對照。
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.config.frequency_axis import GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ, LATE_DECAY_FREQUENCIES_HZ, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.config.quality_targets import load_quality_targets
from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ
from aosr.geometry.shoebox import Wall
from aosr.materials.scattering_defaults import MATERIAL_SCATTERING_DEFAULT_S
from aosr.materials.transfer_matrix import diffuse_incidence_angles_deg
from aosr.physics import geometric_early_jax as early
from aosr.physics.geometric_lane import _unblocked_room_paths
from aosr.physics.room_paths import image_source_paths
from aosr.physics.fem_reduced_boundary import boundary_patches
from aosr.physics.furniture_scene import FurnitureLaneInputs, furniture_lane_inputs, furniture_absorption_area_m2
from aosr.physics.late_decay_jax import late_decay_geometry
from aosr.physics.late_energy_jax import zoned_late_energy_geometry
from aosr.physics.reflection_window import _coverage_from_geometry
from aosr.physics.report_io import ReportInput, solver_inputs
from aosr.physics.report_path_table import _furniture_rows
from aosr.physics.report_surfaces import RectangleZone, ZonesByWall
from aosr.physics.source_directivity import two_parameter_power_ratio
from aosr.physics.wall_surfaces import RectangleAbsorption
from aosr.scoring._six_category_constants import SixCategoryConstants
from aosr.scoring.channel_group import ChannelGroup
from aosr.scoring.direction_zones import DirectionZone, classify, listening_angles, listening_axis, zone_limits
from aosr.scoring.gradient_objective_jax import gradient_objective_constants
from aosr.scoring.gradient_objective_settings import gradient_objective_settings
from aosr.scoring.ranking import read_ranking_rules
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.reflections_jax import reflection_geometry, reflection_setting
from aosr.scoring.reverberation_cost import target_intervals
from aosr.scoring.reverberation_cost_jax import reverberation_cost_settings
from aosr.scoring.stage_one_objective_jax import objective_constants
from aosr.scoring.stage_one_score import stage_one_score_settings
from aosr.scoring.timbre_cost_jax import cost_constants


class SceneInputs(Protocol):
    @property
    def surface_zones_by_wall(self) -> ZonesByWall | None: ...


class SchemeInputs(Protocol):
    """已驗方案中評分需要的三欄；不載入上層 Scheme 類別。"""

    @property
    def scene(self) -> SceneInputs: ...
    @property
    def purpose(self) -> str: ...
    @property
    def channel_group(self) -> ChannelGroup: ...
    @property
    def receiver_set(self) -> ReceiverSet: ...


def _paths(data: early.GeometricEarlyTable, pairs: tuple[early.EarlyPairTable, ...]) -> dict[str, np.ndarray]:
    """變長路徑串接；反彈欄補到全包最大階數，不改任何有效格。"""
    tables = [p.paths for p in pairs]
    width = max(t.bounce_count for t in tables)
    result = {name: np.concatenate([getattr(t, name) for t in tables]) for name in
              ('order', 'dist_m', 'delay_s', 'inverse_distance', 'phase')}
    for name in ('bounce_wall', 'bounce_cos', 'bounce_mask', 'bounce_patch', 'bounce_angle'):
        result[name] = np.concatenate([np.pad(getattr(t, name), ((0, 0), (0, width-t.bounce_count)),
            constant_values=1 if name == 'bounce_cos' else 0) for t in tables])
    result.update(path_offsets=np.cumsum([0]+[t.path_count for t in tables], dtype=np.int64),
        directivity=np.concatenate([p.directivity for p in pairs]),
        furniture_pressure=np.stack([p.furniture_pressure for p in pairs[:len(data.pairs)]]),
        patch_ids=np.asarray(data.patch_ids), angle_offsets=np.cumsum([0]+[len(a) for a in tables[0].incidence_angles_deg]),
        angles_deg=np.concatenate(tables[0].incidence_angles_deg), cosines=np.concatenate(tables[0].incidence_cosines))
    return result


def _geometry(inputs: ReportInput, patch_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    """輻射度的六面＋區列映到阻抗的逐面片列；沿用原交集權重。"""
    zones = {name: tuple(RectangleAbsorption(z, ()) for z in cells if isinstance(z, RectangleZone))
             for name, cells in (inputs.surface_zones_by_wall or {}).items()
             if any(isinstance(z, RectangleZone) for z in cells)}
    geometry = zoned_late_energy_geometry(inputs.room_m, ART_N_PER_WALL_DEFAULT, zones)
    names = [w.wall_name() for w in Wall.all()] + [name+'@'+cell.zone.id for name in Wall.wall_names() for cell in zones.get(name, ())]
    mapping = np.zeros((len(names), len(patch_ids)), dtype=np.float64)
    for row, name in enumerate(names):
        if name in patch_ids:
            mapping[row, patch_ids.index(name)] = 1.0
    decay = late_decay_geometry(inputs.room_m, ART_N_PER_WALL_DEFAULT, sound_speed_m_s=inputs.sound_speed_m_s)
    whole = geometry.whole
    return dict(form_factors=whole.form_factors, late_patch_areas=whole.patch_areas,
        late_wall_index=whole.wall_index, wall_areas=whole.wall_areas,
        wall_weights=geometry.wall_weights @ mapping, patch_weights=geometry.patch_weights @ mapping,
        collision_frequency_hz=np.asarray(decay.collision_frequency_hz))


def _costs(scheme: SchemeInputs, frequencies: np.ndarray, fem: tuple[float, ...], path: Path, sound_speed_m_s: float) -> dict[str, np.ndarray]:
    registry = load_quality_targets(path)
    timbre = cost_constants(frequencies, scheme.receiver_set, scheme.channel_group, quality_targets_path=path,
                           purpose=scheme.purpose, sound_speed_m_s=sound_speed_m_s)
    result = {'response_'+name: np.asarray(value) for name, value in zip(timbre.responses._fields, timbre.responses, strict=True)}
    for key in ('input_layout', 'primary', 'measured', 'channel_pairs', 'importance'):
        result['timbre_'+key] = np.asarray(getattr(timbre, key))
    for key in ('timbre_weights', 'listening_weights', 'matching_weights', 'sound_speed_m_s'):
        result[key] = np.asarray(getattr(timbre, key))
    for prefix, pair in (('primary_pairs', timbre.primary_pairs), ('peer_pairs', timbre.peer_pairs)):
        result['timbre_'+prefix+'_indices'] = pair.indices
        result['timbre_'+prefix+'_weights'] = pair.weights
    for prefix, shape in (('timbre_shape', timbre.timbre_shapes), ('listening_shape', timbre.listening_shapes), ('matching_shape', timbre.matching_shapes)):
        result.update({prefix+'_'+name: np.asarray(item) for name, item in zip(shape._fields, shape, strict=True)})
    product = objective_constants(fem, stage_one_score_settings(registry, scheme.purpose), scheme.receiver_set, len(scheme.channel_group.channels))
    objective = gradient_objective_constants(fem, gradient_objective_settings(registry, scheme.purpose), scheme.receiver_set, len(scheme.channel_group.channels))
    reverb = reverberation_cost_settings(registry.purpose(scheme.purpose))
    for prefix, constants in (('product', product), ('objective', objective), ('reverb', reverb)):
        result.update({prefix+'_'+name: np.asarray(value) for name, value in zip(constants._fields, constants, strict=True)})
    categories = ('timbre_balance', 'listening_area_stability', 'reflections_and_echo', 'reverberation', 'channel_matching', 'low_frequency_response')
    weights = read_ranking_rules(registry, scheme.purpose).category_weights
    result['category_weights'] = np.asarray([weights[name] for name in categories])
    return result


def _reflection_constants(pairs: tuple[early.EarlyPairTable, ...], inputs: tuple[ReportInput, ...], purpose: str,
                          path: Path, furniture: FurnitureLaneInputs | None) -> dict[str, np.ndarray]:
    """反射窗含家具固定列；家具不乘房間散射，和正式路徑表同義。"""
    settings = load_quality_targets(path).purpose(purpose)
    axis = listening_axis(tuple(item.source_m.as_tuple() for item in inputs), inputs[0].receiver_m.as_tuple())
    geometries, fixed = [], []
    limits, _ = zone_limits(settings)
    window_value = reflection_setting(settings, 'window_upper_ms').value
    assert not isinstance(window_value, tuple)
    window = float(window_value) / 1000.0
    for pair, item in zip(pairs, inputs, strict=True):
        solved, table = solver_inputs(item), pair.paths
        raw = image_source_paths(item.room_m, item.source_m, item.receiver_m, item.sound_speed_m_s, max_order=int(max(table.order)))
        if furniture is not None:
            raw = list(_unblocked_room_paths(tuple(raw), item.source_m, item.receiver_m, furniture))
        directions = np.asarray([[(p.image[i]-item.receiver_m.as_tuple()[i])/p.dist_m for i in range(3)] for p in raw])
        geometries.append(reflection_geometry(table, directions, axis, settings))
        maxima = np.zeros((len(DirectionZone), len(table.frequencies_hz)))
        if furniture is not None:
            direct = table.direct_index()
            basis = tuple((table.inverse_distance[direct]*pair.directivity[direct])**2)
            rows = _furniture_rows(furniture, item.source_m, item.receiver_m, basis, solved.source_model,
                tuple(table.frequencies_hz), item.density_kg_m3*item.sound_speed_m_s, item.sound_speed_m_s)
            for row in rows:
                if row.delay_s-table.delay_s[direct] <= window:
                    zone = tuple(DirectionZone).index(classify(*listening_angles(row.direction_vector, axis), limits))
                    maxima[zone] = np.maximum(maxima[zone], row.relative_direct_energy)
        fixed.append(maxima)
    first = geometries[0]
    return dict(reflection_zone_mask=np.concatenate([g.zone_mask for g in geometries], axis=1),
        reflection_frequency_weights=first.frequency_weights, reflection_thresholds_db=first.thresholds_db,
        reflection_zone_weights=first.zone_weights, reflection_worse_reference_db=np.asarray(first.worse_reference_db),
        reflection_furniture_energy=np.stack(fixed), reflection_coverage=np.asarray([g.frequency_coverage for g in geometries]))



def _paris_angles_deg() -> NDArray[np.float64]:
    """numpy 正式路徑那一組 Paris 積分角；建常數要在雙精度已開的行程裡呼叫，否則角度格會先落成單精度。"""
    angles = diffuse_incidence_angles_deg(method='paris')
    if angles.dtype != np.float64:
        raise ValueError("建六類常數要在雙精度已開的行程裡呼叫")
    return np.asarray(angles, dtype=np.float64)

def build_constants(scheme: SchemeInputs, *, pair_inputs: Mapping[tuple[str, str], ReportInput],
                    quality_targets_path: Path = config_path('quality_targets.toml')) -> SixCategoryConstants:
    """從 Scheme 與 checked_inputs 的逐對輸入建一次 CPU 雙精度常數；不從參數求阻抗。"""
    points, channels = scheme.receiver_set.points, scheme.channel_group.channels
    inputs = tuple(pair_inputs[c.speaker_id, p.receiver_id] for c in channels for p in points)
    first, primary = inputs[0], points.index(scheme.receiver_set.primary)
    solved = [solver_inputs(item) for item in inputs]
    fem, report = low_frequency_axis_frequencies(first.low_frequency_axis)
    union = tuple(sorted(set(report) | set(LATE_DECAY_FREQUENCIES_HZ) | set(SCHROEDER_T60_BANDS_HZ)))
    zones = {Wall.from_name(name): tuple(z for z in cells if isinstance(z, RectangleZone))
        for name, cells in (first.surface_zones_by_wall or {}).items() if any(isinstance(z, RectangleZone) for z in cells)}
    rho = first.density_kg_m3*first.sound_speed_m_s
    furniture = furniture_lane_inputs(first.furniture, (first.room_m.Lx, first.room_m.Ly, first.room_m.Lz), report, rho,
        contact_rel=furniture_contact_rel(default_precision_contracts_path()) if first.furniture else None)
    data = early.build_geometric_early_table(room=first.room_m, sources=[solved[i*len(points)].source for i in range(len(channels))],
        receivers=[s.receiver for s in solved[:len(points)]], source_models=[solved[i*len(points)].source_model for i in range(len(channels))],
        sound_speed_m_s=first.sound_speed_m_s, rho_c_pa_s_per_m=rho, frequencies_hz=report,
        reflection_order_k=first.reflection_order_k, rectangle_zones_by_wall=zones, furniture=furniture)
    scheme_zones = {Wall.from_name(name): tuple(z.id for z in cells if isinstance(z, RectangleZone))
        for name, cells in (scheme.scene.surface_zones_by_wall or {}).items() if any(isinstance(z, RectangleZone) for z in cells)}
    expected = tuple(w.wall_name()+('@'+z if z else '') for w, z in boundary_patches(scheme_zones))
    if data.patch_ids != expected:
        raise ValueError('幾何早期表與方案面片順序不同')
    settings = load_quality_targets(quality_targets_path).purpose(scheme.purpose)
    window_value = reflection_setting(settings, 'window_upper_ms').value
    assert not isinstance(window_value, tuple)
    window = float(window_value)/1000.0
    primary_inputs = tuple(inputs[i*len(points)+primary] for i in range(len(channels)))
    coverage = [_coverage_from_geometry(item, window) for item in primary_inputs]
    reflection = tuple(early.build_early_pair_table(first.room_m, item.source_m, item.receiver_m, solver_inputs(item).source_model,
        first.sound_speed_m_s, rho, report, cov[0], zones, furniture) for item, cov in zip(primary_inputs, coverage, strict=True))
    pairs = early.shared_incidence_angles((*data.pairs, *reflection))
    values = _paths(data, pairs) | _geometry(first, data.patch_ids) | _costs(scheme, np.asarray(report), fem, quality_targets_path, first.sound_speed_m_s)
    values.update(_reflection_constants(pairs[len(data.pairs):], primary_inputs, scheme.purpose, quality_targets_path, furniture))
    ratios = [np.ones(len(report)) if s.source_model.parameters is None else two_parameter_power_ratio(report, s.source_model.parameters) for s in solved[::len(points)]]
    values.update(frequencies_hz=np.asarray(report), fem_frequencies_hz=np.asarray(fem), union_frequencies_hz=np.asarray(union),
        report_indices=np.asarray([union.index(f) for f in report]), decay_indices=np.asarray([union.index(f) for f in LATE_DECAY_FREQUENCIES_HZ]),
        midband_indices=np.asarray([union.index(f) for f in SCHROEDER_T60_BANDS_HZ]), paris_angles_deg=_paris_angles_deg(),
        patch_areas_m2=data.patch_areas_m2, rho_c=np.asarray(rho), reflection_order_k=np.asarray(first.reflection_order_k),
        primary_pair_indices=np.arange(len(data.pairs), len(pairs)), reflection_coverage=values['reflection_coverage'] & np.asarray([c[2] is not None for c in coverage]),
        direct_distances_m=np.asarray([p.paths.dist_m[p.paths.direct_index()] for p in data.pairs]).reshape(len(channels), len(points)),
        volume_m3=np.asarray(first.room_m.Lx*first.room_m.Ly*first.room_m.Lz), has_furniture=np.asarray(first.furniture is not None),
        furniture_absorption_area_m2=np.asarray(furniture_absorption_area_m2(first.furniture, union, rho)) if first.furniture else np.zeros(len(union)),
        scattering_by_patch=np.asarray([(first.scattering_by_wall or {}).get(name.partition('@')[0], MATERIAL_SCATTERING_DEFAULT_S) for name in data.patch_ids]),
        source_power_ratio=np.stack(ratios))
    centers = tuple(target_intervals(settings))
    if centers != GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ:
        raise ValueError('殘響目標帶中心與 numpy 報表帶中心不同')
    values['decay_band_mask'] = np.asarray([[center/np.sqrt(2) <= f < center*np.sqrt(2) for f in LATE_DECAY_FREQUENCIES_HZ] for center in centers])
    return SixCategoryConstants(**values)
