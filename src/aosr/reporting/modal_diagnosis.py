"""#671 長方形與六面頻率無關正實阻抗的模態診斷；自己的快取、身分與四態。"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, generate_shoebox_mesh
from aosr.physics.fem_helmholtz import assemble_p2_operators
from aosr.physics.fem_modal import FemModalSpectrum, ModalSolverOptions, solve_fem_modes
from aosr.physics.fem_modal_participation import ModalExpansion, overlap_groups, prepare_modal_expansion
from aosr.physics.modal_convention import ModalKind
from aosr.reporting.import_closure import PhysicsImportClosure, import_code_digest, scan_import_closure
from aosr.reporting.modal_diagnosis_cache import CachedRoom, load_modal_cache, modal_cache_lock, write_modal_cache
from aosr.reporting.modal_diagnosis_model import (
    CheckSummary, ModalDiagnosis, ModalDiagnosisState, ModalKey, PlacementLayer, PlacementPair,
    RoomGroup, RoomLayer, RoomMode, hex_value,
)

MODAL_ENTRY_MODULE = "aosr.reporting.modal_diagnosis"


def _package_root(package_root: Path | None) -> Path:
    return package_root if package_root is not None else Path(__file__).resolve().parents[1]


def modal_import_closure(*, package_root: Path | None = None) -> PhysicsImportClosure:
    """模態入口的靜態閉包；不經 physics_identity（物理身分）或讀回目標線。"""
    return scan_import_closure(_package_root(package_root), MODAL_ENTRY_MODULE)[0]


def modal_identity(*, package_root: Path | None = None) -> str:
    """modal-v1 程式身分，不收品質登記簿；物理條件全由鑰匙帶出。"""
    return "modal-v1:" + import_code_digest(_package_root(package_root), MODAL_ENTRY_MODULE)


def mesh_from_key(key: ModalKey) -> ShoeboxMesh:
    """只照鑰匙建正式網格，聲速與網格三參數不另設預設值。"""
    return generate_shoebox_mesh(Room(*(hex_value(x) for x in key.room_hex)),
        max_frequency_hz=hex_value(key.mesh_frequency_max_hex),
        elements_per_wavelength=key.elements_per_wavelength,
        sound_speed_m_s=hex_value(key.speed_hex), random_seed=key.mesh_random_seed)


def room_layer_from_spectrum(spectrum: FemModalSpectrum, *, key: ModalKey,
                             identity: str, solve_seconds: float) -> RoomLayer:
    """保存全部解種類，只有共振進群；自檢不替強阻尼完備性背書。"""
    groups = overlap_groups(spectrum)
    group_of = {index: group_id for group_id, group in enumerate(groups) for index in group}
    if spectrum.check is None:
        raise ValueError("模態求解缺少 FemModalCheck 自檢摘要")
    summary = asdict(spectrum.check)
    if not math.isfinite(spectrum.check.guaranteed_min_t60_s):
        summary["guaranteed_min_t60_s"] = None
    return RoomLayer(key=key, modal_identity=identity, degrees_of_freedom=spectrum.degrees_of_freedom,
        mesh_frequency_max_hz=hex_value(key.mesh_frequency_max_hex), solve_seconds=solve_seconds,
        modes=tuple(RoomMode(mode_index=i, kind=m.kind, frequency_hz=m.frequency_hz,
            omega_real_rad_s=m.omega.real, omega_imag_rad_s=m.omega.imag,
            t60_s=m.t60_s, q=m.q, group_id=group_of.get(i)) for i, m in enumerate(spectrum.solutions)),
        groups=tuple(RoomGroup(group_id=i, member_indices=g,
            lower_hz=spectrum.solutions[g[0]].frequency_hz, upper_hz=spectrum.solutions[g[-1]].frequency_hz)
            for i, g in enumerate(groups)), check_summary=CheckSummary.model_validate(summary))


def placement_layer_from_expansion(expansion: ModalExpansion, *, sources: Mapping[str, Point],
                                   receivers: Mapping[str, Point]) -> PlacementLayer:
    """一次查所有配對；不保存物理零件回的相對 dB，也不把 None 寫成零。"""
    lookup = expansion.at_positions(tuple(sources.values()), tuple(receivers.values()))
    pairs = []
    for source_index, (source_id, source) in enumerate(sources.items()):
        for receiver_index, (receiver_id, receiver) in enumerate(receivers.items()):
            rows = tuple(row for row in lookup.entries if row.source_index == source_index
                         and row.receiver_index == receiver_index and row.mode.kind is ModalKind.RESONANCE)
            own, whole = [], []
            for row in rows:
                if row.resonance_magnitude is None or row.group_magnitude is None:
                    raise ArithmeticError(f"共振 {row.mode_index} 缺少絕對大小")
                if not math.isfinite(row.resonance_magnitude) or not math.isfinite(row.group_magnitude):
                    raise ArithmeticError(f"共振 {row.mode_index} 的絕對大小非有限：resonance_magnitude={row.resonance_magnitude!r}, group_magnitude={row.group_magnitude!r}")
                own.append(row.resonance_magnitude)
                whole.append(row.group_magnitude)
            pairs.append(PlacementPair(speaker_id=source_id, speaker_position_m=source.as_tuple(),
                receiver_id=receiver_id, receiver_position_m=receiver.as_tuple(),
                resonance_indices=tuple(row.mode_index for row in rows),
                resonance_magnitude=tuple(own), group_magnitude=tuple(whole)))
    return PlacementLayer(pairs=tuple(pairs))


def _supported(wall_impedances: Mapping[Wall, object]) -> bool:
    if set(wall_impedances) != set(Wall.all()):
        return False
    try:
        return all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                   for x in wall_impedances.values())
    except OverflowError:
        return False


def diagnose_modes(*, room: Room, wall_impedances: Mapping[Wall, object], density_kg_m3: float,
                   sound_speed_m_s: float, sources: Mapping[str, Point], receivers: Mapping[str, Point],
                   cache_dir: Path, options: ModalSolverOptions = ModalSolverOptions()) -> ModalDiagnosis:
    """同房同材料只解一次；求解或查位置錯誤保留 repr，無退回解法與重試。"""
    if not isinstance(room, Room):
        return ModalDiagnosis(state=ModalDiagnosisState.OUT_OF_SCOPE, reason_code="unsupported_room")
    if not _supported(wall_impedances):
        return ModalDiagnosis(state=ModalDiagnosisState.OUT_OF_SCOPE, reason_code="unsupported_impedance")
    key = None
    identity = None
    layer = None
    try:
        walls = {wall: float(value) for wall, value in wall_impedances.items() if isinstance(value, (int, float))}
        key = ModalKey.from_inputs(room=room, wall_impedances=walls, density_kg_m3=density_kg_m3,
                                   sound_speed_m_s=sound_speed_m_s, options=options)
        identity = modal_identity()
        with modal_cache_lock(cache_dir=cache_dir, key=key):
            cached = load_modal_cache(cache_dir=cache_dir, key=key, modal_identity=identity)
            mesh = mesh_from_key(key)
            if cached is None:
                started = perf_counter()
                spectrum = solve_fem_modes(mesh, wall_impedances=walls, density_kg_m3=hex_value(key.density_hex),
                    sound_speed_m_s=hex_value(key.speed_hex), frequency_max_hz=hex_value(key.mesh_frequency_max_hex),
                    options=key.solver_options, retain_shapes=True)
                layer = room_layer_from_spectrum(spectrum, key=key, identity=identity, solve_seconds=perf_counter() - started)
                cached = CachedRoom(layer, spectrum)
                expansion = prepare_modal_expansion(spectrum, assemble_p2_operators(mesh), wall_impedances=walls,
                    density_kg_m3=hex_value(key.density_hex), sound_speed_m_s=hex_value(key.speed_hex))
                write_modal_cache(cache_dir=cache_dir, cached=cached)
            else:
                expansion = prepare_modal_expansion(cached.spectrum, assemble_p2_operators(mesh), wall_impedances=walls,
                    density_kg_m3=hex_value(key.density_hex), sound_speed_m_s=hex_value(key.speed_hex))
            layer = cached.room_layer
        placement = placement_layer_from_expansion(expansion, sources=sources, receivers=receivers)
        return ModalDiagnosis(state=ModalDiagnosisState.DIAGNOSED_NOT_SCORED, key=key, modal_identity=identity,
                              room_layer=layer, placement_layer=placement)
    except Exception as exc:
        return ModalDiagnosis(state=ModalDiagnosisState.FAILED, reason_text=repr(exc), key=key,
                              modal_identity=identity, room_layer=layer)
