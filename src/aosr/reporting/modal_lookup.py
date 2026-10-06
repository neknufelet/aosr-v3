"""閉包外的共用模態入口：方案鑰匙、只查不解、原子保存擺位診斷。"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp

from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.fem_helmholtz import assemble_p2_operators
from aosr.physics.fem_modal_participation import prepare_modal_expansion
from aosr.reporting.modal_diagnosis import (
    diagnose_modes, mesh_fingerprint, mesh_from_key, modal_identity, placement_layer_from_expansion,
)
from aosr.reporting.modal_diagnosis_cache import load_modal_cache, modal_cache_lock
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState, ModalKey, hex_value
from aosr.reporting.scheme import Scheme


def scope_reason(room: object, wall_impedances: Mapping[Wall, object]) -> str | None:
    """保持與核心 _supported 相同的範圍；不更動核心及其模態身分。"""
    if not isinstance(room, Room):
        return "unsupported_room"
    if set(wall_impedances) != set(Wall.all()):
        return "unsupported_impedance"
    try:
        supported = all(not isinstance(x, bool) and isinstance(x, (int, float))
                        and math.isfinite(x) and x > 0 for x in wall_impedances.values())
    except OverflowError:
        supported = False
    return None if supported else "unsupported_impedance"


def _walls(scheme: Scheme) -> dict[Wall, float]:
    return {Wall.from_name(name): value for name, value in scheme.scene.impedance_pa_s_per_m_by_wall.items()}


def key_from_scheme(scheme: Scheme) -> ModalKey | None:
    """密度與聲速只讀方案 scene；求解選項使用核心預設，所有入口共用這一把鑰匙。"""
    if scheme_scope_reason(scheme) is not None:
        return None
    walls, scene = _walls(scheme), scheme.scene
    return ModalKey.from_inputs(room=scene.room_m, wall_impedances=walls,
        density_kg_m3=scene.density_kg_m3, sound_speed_m_s=scene.sound_speed_m_s)


def scheme_scope_reason(scheme: Scheme) -> str | None:
    if not isinstance(scheme.scene.room_m, Room):
        return "unsupported_room"
    try:
        return scope_reason(scheme.scene.room_m, _walls(scheme))
    except ValueError:
        return "unsupported_impedance"


def placement_digest(scheme: Scheme) -> str:
    """按代號排序的座標 hex；不收入口、方案名字、用途或評分設定。"""
    sources = {name: tuple(float(x).hex() for x in p.as_tuple()) for name, p in scheme.speakers.items()}
    receivers = {r.receiver_id: tuple(float(x).hex() for x in r.position_m) for r in scheme.receiver_set.points}
    content = json.dumps([sources, receivers], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def placement_path(scheme: Scheme, *, cache_dir: Path, identity: str | None = None) -> Path | None:
    key = key_from_scheme(scheme)
    if key is None:
        return None
    identity = identity or modal_identity()
    return cache_dir / f"placement-{key.digest}-{identity.removeprefix('modal-v1:')}-{placement_digest(scheme)}.json"


def read_placement(scheme: Scheme, *, cache_dir: Path, identity: str | None = None) -> ModalDiagnosis | None:
    """只讀小診斷檔，驗身分、房間鑰匙與完整擺位；不用網格或求解器。"""
    identity = identity or modal_identity()
    path = placement_path(scheme, cache_dir=cache_dir, identity=identity)
    if path is None:
        return None
    try:
        diagnosis = ModalDiagnosis.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return diagnosis if placement_matches(diagnosis, scheme, identity=identity) else None


def placement_matches(diagnosis: ModalDiagnosis, scheme: Scheme, *, identity: str) -> bool:
    """工作输出與擺位快取都驗同一份身分及位置契約。"""
    if (diagnosis.state is not ModalDiagnosisState.DIAGNOSED_NOT_SCORED or
            diagnosis.modal_identity != identity or diagnosis.key != key_from_scheme(scheme)):
        return False
    placement = diagnosis.placement_layer
    assert placement is not None
    expected = {(s, p.as_tuple(), r.receiver_id, r.position_m)
                for s, p in scheme.speakers.items() for r in scheme.receiver_set.points}
    actual = {(p.speaker_id, p.speaker_position_m, p.receiver_id, p.receiver_position_m) for p in placement.pairs}
    return actual == expected


def save_diagnosis(diagnosis: ModalDiagnosis, path: Path) -> None:
    """先驗完整文件，同目錄寫唯一暫存檔再改名；命令列與擺位快取共用。"""
    diagnosis = ModalDiagnosis.model_validate(diagnosis.model_dump())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = mkstemp(dir=path.parent, prefix="placement-write-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(diagnosis.model_dump_json())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _placement_lock(path: Path) -> Iterator[None]:
    """同擺位跨入口串行；房間層另由核心鎖防止重解。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _cache_only(scheme: Scheme, *, cache_dir: Path, key: ModalKey, identity: str) -> ModalDiagnosis:
    """房間鎖內讀快取、核網格、展開與查位置；整條路沒有求解呼叫。"""
    with modal_cache_lock(cache_dir=cache_dir, key=key):
        cached = load_modal_cache(cache_dir=cache_dir, key=key, modal_identity=identity)
        if cached is None:
            return ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED, key=key, modal_identity=identity)
        mesh = mesh_from_key(key)
        if mesh_fingerprint(mesh) != cached.room_layer.mesh_sha256:
            return ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED, key=key, modal_identity=identity)
        expansion = prepare_modal_expansion(cached.spectrum, assemble_p2_operators(mesh),
            wall_impedances=_walls(scheme), density_kg_m3=hex_value(key.density_hex),
            sound_speed_m_s=hex_value(key.speed_hex))
        placement = placement_layer_from_expansion(expansion, sources=scheme.speakers,
            receivers={r.receiver_id: Point(*r.position_m) for r in scheme.receiver_set.points})
        return ModalDiagnosis(state=ModalDiagnosisState.DIAGNOSED_NOT_SCORED, key=key,
            modal_identity=identity, room_layer=cached.room_layer, placement_layer=placement)


def _calculate(scheme: Scheme, *, cache_dir: Path) -> ModalDiagnosis:
    scene = scheme.scene
    return diagnose_modes(room=scene.room_m, wall_impedances=_walls(scheme),
        density_kg_m3=scene.density_kg_m3, sound_speed_m_s=scene.sound_speed_m_s,
        sources=scheme.speakers, receivers={r.receiver_id: Point(*r.position_m) for r in scheme.receiver_set.points},
        cache_dir=cache_dir)


def diagnose_scheme(scheme: Scheme, *, cache_dir: Path, cache_only: bool) -> ModalDiagnosis:
    """成功的擺位診斷共用；未計算不存擺位快取，之後房間快取到齊仍可自動沿用。"""
    key, identity = None, None
    try:
        key = key_from_scheme(scheme)
        if key is None:
            return ModalDiagnosis(state=ModalDiagnosisState.OUT_OF_SCOPE, reason_code=scheme_scope_reason(scheme))
        identity = modal_identity()
        path = placement_path(scheme, cache_dir=cache_dir, identity=identity)
        assert path is not None
        with _placement_lock(path):
            saved = read_placement(scheme, cache_dir=cache_dir, identity=identity)
            if saved is not None:
                return saved
            diagnosis = (_cache_only(scheme, cache_dir=cache_dir, key=key, identity=identity)
                         if cache_only else _calculate(scheme, cache_dir=cache_dir))
            if diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
                save_diagnosis(diagnosis, path)
            return diagnosis
    except Exception as exc:
        return ModalDiagnosis(state=ModalDiagnosisState.FAILED, reason_text=repr(exc), key=key,
                              modal_identity=identity)
