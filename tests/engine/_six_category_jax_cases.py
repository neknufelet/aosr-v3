"""六類組裝考卷的小房：正式方案、正式材料入口與正式 numpy 管線。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np

from aosr.config.capabilities import load_capabilities
from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.construction_diff import build_template, surface_impedance_from_parameters
from aosr.materials.construction_models import Assembly, ConstantImpedance, SurfaceMaterial
from aosr.physics.report_surfaces import RectangleZone, WholeWallZone, ZonesByWall
from aosr.reporting.pipeline import run_scheme
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import checked_inputs
from aosr.scoring import ranking
from aosr.scoring.receiver_set import primary_cross
from aosr.scoring.six_category_jax import SixCategoryConstants, build_constants
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import reference_document, relative_item
from tests.engine._geometric_early_jax_cases import scattering

TARGETS = config_path('quality_targets.toml')
NAMES = ('timbre_balance', 'listening_area_stability', 'reflections_and_echo', 'reverberation', 'channel_matching', 'low_frequency_response')


def scheme_case(name: str) -> Scheme:
    original = Scheme.model_validate(reference_document())
    tiny = name == 'unavailable'
    room = Room(1.6, 1.2, 1.0) if tiny else (Room(6., 4., 3.) if name in ('no_extension', 'late_furniture') else Room(4.5, 3.5, 2.6))
    sources = (Point(.3, .3, .4), Point(.3, .9, .4)) if tiny else (Point(1., 1.3, 1.2), Point(1., 2.5, 1.2))
    primary = (1.2, .7, .6) if tiny else (3.2, 1.9, 1.2)
    walls = {w.wall_name(): 411.6*(111.+7*i) for i, w in enumerate(Wall.all())}
    zones: ZonesByWall = {w: (WholeWallZone(area='whole_wall', material=ConstantImpedance(impedance_pa_s_m=z)),) for w, z in walls.items()}
    if name == 'changed_wall':
        zones['x0'] = (WholeWallZone(area='whole_wall', material=Assembly.model_validate({'kind':'assembly', 'layers':[
            {'kind':'porous', 'material_class':'fibrous', 'model':'komatsu', 'thickness_m':.05, 'flow_resistivity_pa_s_m2':16000.0},
            {'kind':'air', 'thickness_m':.05}], 'backing':'rigid'})),)
    if name in ('zoned', 'reordered'):
        zones['x0'] = tuple(RectangleZone(area='rectangle', id=key, u_m=span, v_m=(0., room.Lz),
            material=Assembly.model_validate({'kind':'assembly', 'layers':[
                {'kind':'porous', 'material_class':'fibrous', 'model':'komatsu', 'thickness_m':.05, 'flow_resistivity_pa_s_m2':16000.0},
                {'kind':'air', 'thickness_m':.05}], 'backing':'rigid'}) if key=='b' else zones['x0'][0].material)
            for key, span in (('b', (2*room.Ly/3, room.Ly)), ('a', (0., 2*room.Ly/3))))
        if name == 'reordered':
            zones['x0'] = zones['x0'][::-1]
    furniture = (relative_item(),) if name=='furniture' else None
    if name == 'late_furniture':
        furniture = (relative_item(furniture_id='late-panel', kind='desk', material='wood', width_m=1.8, depth_m=.05, height_m=2.,
            placement={'forward_m':-2.7, 'left_m':0., 'bottom_height_m':.1, 'yaw_deg':0}),)
    scene = original.scene.model_copy(update={'room_m':room, 'surface_zones_by_wall': zones,
        'impedance_pa_s_per_m_by_wall':None, 'low_frequency_axis':LowFrequencyAxis.SEARCH_48, 'reflection_order_k':3, 'scattering_by_wall':None if name=='default_scattering' else scattering()})
    return Scheme.model_validate(original.model_copy(update={'scene':scene, 'source_model':'product_default',
        'furniture':furniture, 'speaker_setup':None, 'receiver_set':primary_cross(primary_position_m=primary, offset_m=.1),
        'speakers':{ch.speaker_id:p for ch, p in zip(original.channel_group.channels, sources, strict=True)}}).model_dump())


def constants(scheme: Scheme) -> SixCategoryConstants:
    _, pairs = checked_inputs(scheme, capabilities=load_capabilities(config_path('capabilities.toml')), directivity=DIRECTIVITY)
    return build_constants(scheme, pair_inputs={key:value[1] for key,value in pairs.items()})


def array_inputs(scheme: Scheme, c: SixCategoryConstants) -> tuple[jnp.ndarray, ...]:
    air = load_physics_constants(config_path('physics_constants.toml'))
    ranges = load_porous_model_ranges(config_path('porous_model_ranges.toml'))
    zones = scheme.scene.surface_zones_by_wall or {}
    materials = []
    for name in c.patch_ids:
        wall, _, zone = str(name).partition('@')
        materials.append(next(item.material for item in zones[wall] if not isinstance(item, RectangleZone) or item.id==zone) if zones else ConstantImpedance(impedance_pa_s_m=cast(dict[str,float], scheme.scene.impedance_pa_s_per_m_by_wall)[wall]))
    def impedance(material: SurfaceMaterial, axis: np.ndarray, angles: np.ndarray | None = None) -> jnp.ndarray:
        if isinstance(material, ConstantImpedance):
            return jnp.full((len(axis),) if angles is None else (len(axis), len(angles)), complex(material.impedance_pa_s_m), dtype=jnp.complex128)
        template = build_template(material, jnp.asarray(axis), air=air, ranges=ranges, variables=())
        return surface_impedance_from_parameters(template, {}, jnp.asarray(axis), air=air, incidence_angles_deg=None if angles is None else jnp.asarray(angles))
    fem = jnp.stack([impedance(m, c.fem_frequencies_hz).reshape(len(c.fem_frequencies_hz), -1)[:, 0] for m in materials])
    normal = jnp.stack([impedance(m, c.frequencies_hz).reshape(len(c.frequencies_hz), -1)[:, 0] for m in materials])
    oblique = jnp.concatenate([impedance(m, c.frequencies_hz, c.angles_deg[int(a):int(b)]) for m,a,b in
                              zip(materials,c.angle_offsets[:-1],c.angle_offsets[1:],strict=True)], axis=1)
    paris = jnp.stack([impedance(m, c.union_frequencies_hz, c.paris_angles_deg) for m in materials])
    x = np.log2(c.fem_frequencies_hz)
    pressure = np.asarray([[(1.+.4*s+.25*r)*10**((2.4*np.sin((1.3+.07*r)*x+.23*s)+.3*r*x)/20.)
        *np.exp(1j*(.11*r*x+.7*s)) for r in range(len(scheme.receiver_set.points))] for s in range(2)])
    return fem, normal, oblique, paris, jnp.asarray(pressure), jnp.asarray([isinstance(m, ConstantImpedance) for m in materials])


def numpy_result(scheme: Scheme, pressure: np.ndarray) -> SchemeResult:
    energies = np.abs(pressure)**2
    injected = {(ch.speaker_id,p.receiver_id):tuple(float(v) for v in energies[s,r])
        for s,ch in enumerate(scheme.channel_group.channels) for r,p in enumerate(scheme.receiver_set.points)}
    return run_scheme(scheme, capabilities=load_capabilities(config_path('capabilities.toml')), directivity=DIRECTIVITY,
        quality_targets_path=TARGETS, engine_commit='test', program_fingerprint='calc-v1:'+'0'*64,
        physics_identity='phys-v1:'+'0'*64, run_date=date(2026,11,8), fem_energies=injected)


def numpy_costs(result: SchemeResult) -> tuple[np.ndarray, float]:
    rules = ranking.read_ranking_rules(load_quality_targets(TARGETS), result.scheme.purpose)
    settled = tuple(ranking._settle(e, rules) for e in result.candidate.evaluations)
    values = {e.category.value: np.nan if e.category_cost is None else e.category_cost.value for e in settled}
    lines, uncovered, missing = ranking._classify(settled, rules)
    assessment = ranking._Assessment(result.candidate, settled, tuple(lines), tuple(uncovered), tuple(missing), (), (), ranking.ExternalAcceptance.NOT_CHECKED)
    return np.asarray([values[name] for name in NAMES]), ranking._total_cost(assessment)
