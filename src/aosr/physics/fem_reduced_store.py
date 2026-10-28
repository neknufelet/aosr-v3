"""替身存讀：小矩陣進 npz，出身與建模帳進同名 manifest（#561）。

``<名>.npz`` 存五個陣列；``<名>.manifest.json`` 存格式版本、喇叭與座位順序、牆序、頻率軸與空氣的 hex、
網格雜湊、物理身分、輸入雜湊、建模帳、每個陣列的 dtype／形狀／SHA-256、整份 npz 的 SHA-256，
最後一格是 manifest 其餘內容的 SHA-256。存之前與讀回時都核每個陣列的 dtype 與形狀（K、M、B 是 float64，
f_r 是 complex128，c_r 是 float64；r 取建模帳），不符就拒收、不轉型。讀回時雜湊三層都核：manifest 自己、整份 npz、逐陣列；
任何一位不同就拒收（這是防損壞與誤配，不是防有心竄改）。寫入前先確認兩個檔都不在，不覆寫既有檔。
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aosr.config.fem_reduced import ReducedModelSettings
from aosr.geometry.shoebox import Wall
from aosr.physics.fem_reduced import BuildLedger, GreedyRound, ReducedMatrices, ReducedModel

FORMAT_VERSION = 1
_ARRAYS = ("stiffness", "mass", "boundary", "sources", "receivers")
_DIGEST_FIELD = "manifest_sha256"
_FLOAT = np.dtype(np.float64)
_COMPLEX = np.dtype(np.complex128)


def _required(sources: int, receivers: int, rank: int) -> dict[str, tuple[np.dtype[np.generic], tuple[int, ...]]]:
    """每個陣列必須的 dtype 與形狀：S＝喇叭名數、R＝座位名數、r＝建模帳的基底大小、牆數照 ``Wall.all()``。"""
    walls = len(Wall.all())
    return {"stiffness": (_FLOAT, (rank, rank)), "mass": (_FLOAT, (rank, rank)),
            "boundary": (_FLOAT, (walls, rank, rank)), "sources": (_COMPLEX, (sources, rank)),
            "receivers": (_FLOAT, (receivers, rank))}


def _check_arrays(arrays: Mapping[str, NDArray[np.generic]], sources: int, receivers: int, rank: int) -> None:
    """dtype 與形狀逐格比，不符就 ValueError；這裡不轉型。"""
    for name, (dtype, shape) in _required(sources, receivers, rank).items():
        array = arrays[name]
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"替身陣列 {name} 必須是 {dtype} {shape}，收到 {array.dtype} {array.shape}")


def manifest_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + ".manifest.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(document: dict[str, object]) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def matrices_digest(matrices: ReducedMatrices) -> str:
    """模型內容摘要：五個小矩陣照固定順序，各取 C 順序位元組的 SHA-256（同 manifest 逐陣列那一格的算法），
    再把五個十六進位串接起來 SHA-256 一次。驗證報告與驗證收據靠它綁到被驗的那一份替身。"""
    digest = hashlib.sha256()
    for name in _ARRAYS:
        digest.update(_sha(np.ascontiguousarray(getattr(matrices, name)).tobytes()).encode())
    return digest.hexdigest()


def _ledger_document(ledger: BuildLedger) -> dict[str, object]:
    return {"full_solves": ledger.full_solves, "snapshots": ledger.snapshots, "rank": ledger.rank,
            "rounds": [vars(item) for item in ledger.rounds], "stop_reason": ledger.stop_reason,
            "seconds": dict(ledger.seconds), "settings": ledger.settings.model_dump(mode="json")}


def save_reduced(model: ReducedModel, npz_path: Path) -> Path:
    """寫 npz 與 manifest，回 manifest 路徑。

    陣列 dtype 或形狀不合（見 :func:`_required`）、或兩個檔有任何一個已經在，就在寫任何東西之前報錯，不覆寫。
    """
    target = manifest_path(npz_path)
    arrays: dict[str, NDArray[np.generic]] = {
        name: np.ascontiguousarray(getattr(model.matrices, name)) for name in _ARRAYS}
    _check_arrays(arrays, len(model.source_names), len(model.receiver_names), model.ledger.rank)
    for path in (npz_path, target):
        if path.exists():
            raise FileExistsError(f"替身檔已存在，不覆寫：{path}")
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    payload = buffer.getvalue()
    document: dict[str, object] = {
        "format_version": FORMAT_VERSION, "walls": [wall.wall_name() for wall in Wall.all()],
        "source_names": list(model.source_names), "receiver_names": list(model.receiver_names),
        "frequencies_hex": [value.hex() for value in model.frequencies_hz],
        "density_hex": float(model.density_kg_m3).hex(), "speed_hex": float(model.sound_speed_m_s).hex(),
        "mesh_hash": model.mesh_hash, "physics_identity": model.physics_identity,
        "input_hash": model.input_hash, "ledger": _ledger_document(model.ledger),
        "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": _sha(array.tobytes())}
                   for name, array in arrays.items()},
        "npz_sha256": _sha(payload),
    }
    document[_DIGEST_FIELD] = _sha(_canonical(document))
    text = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + "\n"
    with npz_path.open("xb") as handle:
        handle.write(payload)
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        npz_path.unlink()
        raise
    return target


def _checked_manifest(npz_path: Path) -> dict[str, object]:
    document = json.loads(manifest_path(npz_path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format_version") != FORMAT_VERSION:
        raise ValueError("替身 manifest 格式版本不符")
    recorded = document.pop(_DIGEST_FIELD, None)
    if recorded != _sha(_canonical(document)):
        raise ValueError("替身 manifest 內容雜湊不符")
    if document.get("walls") != [wall.wall_name() for wall in Wall.all()]:
        raise ValueError("替身 manifest 牆序不符")
    return document


def _arrays(npz_path: Path, document: dict[str, object]) -> dict[str, NDArray[np.generic]]:
    payload = npz_path.read_bytes()
    if document.get("npz_sha256") != _sha(payload):
        raise ValueError("替身 npz 整份雜湊不符")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: np.ascontiguousarray(archive[name]) for name in _ARRAYS}
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise ValueError("替身 npz 讀不開") from exc
    recorded = document.get("arrays")
    for name, array in arrays.items():
        entry = recorded.get(name) if isinstance(recorded, dict) else None
        if entry != {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": _sha(array.tobytes())}:
            raise ValueError(f"替身陣列 {name} 雜湊、型別或形狀不符")
    return arrays


def _ledger(document: dict[str, object]) -> BuildLedger:
    data = document["ledger"]
    if not isinstance(data, dict):
        raise ValueError("替身 manifest 建模帳格式不符")
    return BuildLedger(
        full_solves=int(data["full_solves"]), snapshots=int(data["snapshots"]), rank=int(data["rank"]),
        rounds=tuple(GreedyRound(**item) for item in data["rounds"]), stop_reason=str(data["stop_reason"]),
        seconds={str(key): float(value) for key, value in data["seconds"].items()},
        settings=ReducedModelSettings.model_validate(data["settings"]))


def load_reduced(npz_path: Path, *, expected_mesh_hash: str | None = None,
                 expected_physics_identity: str | None = None) -> ReducedModel:
    """核 manifest、整份 npz、逐陣列雜湊；給了網格雜湊或物理身分就再比，不同就拒收。"""
    document = _checked_manifest(npz_path)
    arrays = _arrays(npz_path, document)
    if expected_mesh_hash is not None and document["mesh_hash"] != expected_mesh_hash:
        raise ValueError("替身的網格雜湊跟這次的網格不同")
    if expected_physics_identity is not None and document["physics_identity"] != expected_physics_identity:
        raise ValueError("替身的物理身分跟這次的不同")
    frequencies = document["frequencies_hex"]
    if not isinstance(frequencies, list):
        raise ValueError("替身 manifest 頻率軸格式不符")
    source_names = tuple(str(name) for name in _list(document, "source_names"))
    receiver_names = tuple(str(name) for name in _list(document, "receiver_names"))
    ledger = _ledger(document)
    # 逐陣列雜湊已核過 manifest 記的 dtype／形狀等於實際陣列；這裡再比必須的那一套，不符就拒收、不轉型。
    _check_arrays(arrays, len(source_names), len(receiver_names), ledger.rank)
    return ReducedModel(
        matrices=ReducedMatrices(
            stiffness=_floats(arrays["stiffness"]), mass=_floats(arrays["mass"]),
            boundary=_floats(arrays["boundary"]), sources=_complexes(arrays["sources"]),
            receivers=_floats(arrays["receivers"])),
        source_names=source_names, receiver_names=receiver_names,
        frequencies_hz=tuple(float.fromhex(str(value)) for value in frequencies),
        density_kg_m3=float.fromhex(str(document["density_hex"])),
        sound_speed_m_s=float.fromhex(str(document["speed_hex"])),
        mesh_hash=str(document["mesh_hash"]), physics_identity=str(document["physics_identity"]),
        input_hash=str(document["input_hash"]), ledger=ledger)


def _floats(array: NDArray[np.generic]) -> NDArray[np.float64]:
    """型別已由 :func:`_check_arrays` 核過；這裡只給型別檢查器看，不是轉型。"""
    if array.dtype != _FLOAT:
        raise ValueError(f"替身陣列必須是 float64，收到 {array.dtype}")
    return np.asarray(array, dtype=np.float64)


def _complexes(array: NDArray[np.generic]) -> NDArray[np.complex128]:
    if array.dtype != _COMPLEX:
        raise ValueError(f"替身陣列必須是 complex128，收到 {array.dtype}")
    return np.asarray(array, dtype=np.complex128)


def _list(document: dict[str, object], key: str) -> list[object]:
    value = document[key]
    if not isinstance(value, list):
        raise ValueError(f"替身 manifest 的 {key} 格式不符")
    return value
