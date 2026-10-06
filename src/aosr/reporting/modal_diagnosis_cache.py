"""房間模態快取：唯一形狀檔先發布，標頭最後原子改名；不讀半份產物。"""
from __future__ import annotations

import fcntl
import hashlib
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from aosr.physics.fem_modal import FemModalSpectrum, FemMode, ModalShift
from aosr.physics.fem_modal_check import DecayOrigin, FemModalCheck, ModalCountBand
from aosr.reporting.modal_diagnosis_model import (
    FROZEN, SCHEMA_VERSION, Finite, Index, ModalIdentity, ModalKey, Nonnegative,
    Positive, RoomLayer,
)


@dataclass(frozen=True)
class CachedRoom:
    """完整房間層與原始求解帳，形狀保持唯讀，可重新準備擺位展開。"""
    room_layer: RoomLayer
    spectrum: FemModalSpectrum


class _Header(BaseModel):
    model_config = FROZEN
    schema_version: Literal["aosr.modal_diagnosis.v1"] = SCHEMA_VERSION
    key: ModalKey
    modal_identity: ModalIdentity
    payload_name: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ModeEvidence(BaseModel):
    model_config = FROZEN
    raw_omega_real_rad_s: Finite
    raw_omega_imag_rad_s: Finite
    residual: Nonnegative
    linearized_residual: Nonnegative
    shift_index: Index
    shift_hz: Positive
    above_guaranteed_decay: bool
    decay_origin: DecayOrigin | None
    continuation_index: tuple[Index, Index, Index] | None


class _Shift(BaseModel):
    model_config = FROZEN
    frequency_hz: Positive
    requested: Index
    radius_rad_s: Nonnegative


class _Ledger(BaseModel):
    model_config = FROZEN
    room_layer: RoomLayer
    mode_evidence: tuple[_ModeEvidence, ...]
    shifts: tuple[_Shift, ...]
    zero_rad_s: Nonnegative
    component_zero_rad_s: Nonnegative


def _header_path(cache_dir: Path, key: ModalKey) -> Path:
    return cache_dir / f"modal-{key.digest}.json"


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def modal_cache_lock(*, cache_dir: Path, key: ModalKey) -> Iterator[None]:
    """同鑰匙的跨行程首次求解串行；鎖外可以獨立查不同位置。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / f"modal-{key.digest}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _ledger(cached: CachedRoom) -> _Ledger:
    return _Ledger(room_layer=RoomLayer.model_validate(cached.room_layer.model_dump()),
        mode_evidence=tuple(_ModeEvidence(
            raw_omega_real_rad_s=m.raw_omega.real, raw_omega_imag_rad_s=m.raw_omega.imag,
            residual=m.residual, linearized_residual=m.linearized_residual,
            shift_index=m.shift_index, shift_hz=m.shift_hz,
            above_guaranteed_decay=m.above_guaranteed_decay, decay_origin=m.decay_origin,
            continuation_index=m.continuation_index) for m in cached.spectrum.solutions),
        shifts=tuple(_Shift(frequency_hz=s.frequency_hz, requested=s.requested,
                            radius_rad_s=s.radius_rad_s) for s in cached.spectrum.shifts),
        zero_rad_s=cached.spectrum.zero_rad_s, component_zero_rad_s=cached.spectrum.component_zero_rad_s)


def _restore(ledger: _Ledger, shapes: NDArray[np.complex128]) -> CachedRoom:
    room = ledger.room_layer
    if shapes.shape != (len(room.modes), room.degrees_of_freedom) or not np.all(np.isfinite(shapes)):
        raise ValueError("模態快取形狀尺寸或數值不合法")
    modes = []
    for row, evidence, shape in zip(room.modes, ledger.mode_evidence, shapes, strict=True):
        shape.flags.writeable = False
        modes.append(FemMode(omega=complex(row.omega_real_rad_s, row.omega_imag_rad_s),
            raw_omega=complex(evidence.raw_omega_real_rad_s, evidence.raw_omega_imag_rad_s),
            frequency_hz=row.frequency_hz, t60_s=row.t60_s, q=row.q, kind=row.kind,
            residual=evidence.residual, linearized_residual=evidence.linearized_residual,
            shift_index=evidence.shift_index, shift_hz=evidence.shift_hz, shape=shape,
            above_guaranteed_decay=evidence.above_guaranteed_decay,
            decay_origin=evidence.decay_origin, continuation_index=evidence.continuation_index))
    summary = room.check_summary
    check = FemModalCheck(
        guaranteed_decay_rate_rad_s=summary.guaranteed_decay_rate_rad_s,
        guaranteed_min_t60_s=math.inf if summary.guaranteed_min_t60_s is None else summary.guaranteed_min_t60_s,
        count_bands=tuple(ModalCountBand(**b.model_dump()) for b in summary.count_bands),
        static_count=summary.static_count, zero_mode_continuation_count=summary.zero_mode_continuation_count,
        overdamped_count=summary.overdamped_count, unconfirmed_decay_count=summary.unconfirmed_decay_count,
        returned_above_limit_count=summary.returned_above_limit_count, weyl_terms=summary.weyl_terms)
    spectrum = FemModalSpectrum(tuple(modes), tuple(ModalShift(**s.model_dump()) for s in ledger.shifts),
        ledger.zero_rad_s, ledger.component_zero_rad_s, room.degrees_of_freedom, check)
    return CachedRoom(room, spectrum)


def load_modal_cache(*, cache_dir: Path, key: ModalKey, modal_identity: str) -> CachedRoom | None:
    """綱要、鑰匙、身分或資料雜湊不同都視為無快取；禁止 pickle 與任意標頭路徑。"""
    try:
        header = _Header.model_validate_json(_header_path(cache_dir, key).read_text(encoding="utf-8"))
        if header.key != key or header.modal_identity != modal_identity:
            return None
        if re.fullmatch(rf"modal-{key.digest}-[0-9a-f]{{32}}\.npz", header.payload_name) is None:
            return None
        payload = cache_dir / header.payload_name
        if _file_digest(payload) != header.payload_sha256:
            return None
        with np.load(payload, allow_pickle=False) as archive:
            ledger = _Ledger.model_validate_json(str(archive["metadata"].item()))
            shapes = np.asarray(archive["shapes"], dtype=np.complex128)
        if ledger.room_layer.key != key or ledger.room_layer.modal_identity != modal_identity:
            return None
        return _restore(ledger, shapes)
    except (OSError, ValueError, KeyError, BadZipFile):
        return None


def write_modal_cache(*, cache_dir: Path, cached: CachedRoom) -> None:
    """發布成功前的例外不留下可讀半份，覆寫中斷仍保留舊版本；發布後清掉同鑰匙的舊形狀檔。

    產品路徑在 :func:`modal_cache_lock` 裡呼叫，清舊檔時沒有別的行程在讀同一把鑰匙。
    """
    ledger = _ledger(cached)
    if any(m.shape is None for m in cached.spectrum.solutions):
        raise ValueError("模態快取必須保存每個解的形狀")
    shapes = np.asarray([m.shape for m in cached.spectrum.solutions], dtype=np.complex128)
    _restore(ledger, shapes)  # 寫出前也驗尺寸、完整帳與有限數值。
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = cache_dir / f"modal-{cached.room_layer.key.digest}-{uuid4().hex}.npz"
    published = False
    try:
        with TemporaryDirectory(prefix="modal-write-", dir=cache_dir) as temporary:
            staging = Path(temporary)
            staged_payload = staging / payload.name
            np.savez_compressed(staged_payload, metadata=np.asarray(ledger.model_dump_json()), shapes=shapes)
            header = _Header(key=cached.room_layer.key, modal_identity=cached.room_layer.modal_identity,
                             payload_name=payload.name, payload_sha256=_file_digest(staged_payload))
            staged_header = staging / "header.json"
            staged_header.write_text(header.model_dump_json(), encoding="utf-8")
            staged_payload.replace(payload)
            staged_header.replace(_header_path(cache_dir, cached.room_layer.key))
            published = True
    finally:
        if not published:
            payload.unlink(missing_ok=True)
    # 新標頭已指向新形狀檔；同鑰匙的舊形狀檔沒有標頭指著，留著只佔磁碟（真房間一份約 0.3 GB）。
    for stale in cache_dir.glob(f"modal-{cached.room_layer.key.digest}-*.npz"):
        if stale.name != payload.name:
            stale.unlink(missing_ok=True)
