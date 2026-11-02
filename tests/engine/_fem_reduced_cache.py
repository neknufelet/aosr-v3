"""有限元素替身考卷共用的重計算（#996）：同一次 pytest 只建一次替身、同一份輸入的留出驗證只做一次，平行的工人經存檔共用。

照 ``_scheme_cache.shared_json``：檔案鎖、共用檔放在這一次 pytest 的共同暫存根；``-n 0``（master）每次都自己算。
共用的是存檔，每份考卷讀回時各自新建一份物件，兩份考卷不會拿到同一個可變物件。真的算的那個工人先比「剛算出的」
與「存檔再讀回的」逐位相同（``_scheme_cache.bitwise_equal``），不同就拋例外——存讀一致這件事不會因為共用而沒人考。只換算幾次，不換算法與輸入：
替身照呼叫端給的建法建、留出驗證照 :func:`validate_reduced` 做。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.physics.fem_helmholtz import FrequencyWallImpedances
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_check import Band, ValidationReport, validate_reduced
from aosr.physics.fem_reduced_store import computation_digest, load_reduced, manifest_path, save_reduced
from tests.engine._scheme_cache import bitwise_equal, shared_json

_REPORT_ARRAYS = ("full_pressure", "reduced_pressure")


def _directory(tmp_path_factory: pytest.TempPathFactory, worker_id: str, name: str) -> Path:
    """``-n 0`` 每次開一個新的暫存目錄；平行時放共同暫存根，每個工人都看得到同一個。"""
    if worker_id == "master":
        return tmp_path_factory.mktemp(name)
    directory = tmp_path_factory.getbasetemp().parent / f"{name}-shared"
    directory.mkdir(exist_ok=True)
    return directory


def _seconds_sorted(model: ReducedModel) -> ReducedModel:
    """讀回比較唯一的例外：建模帳 ``seconds`` 的鍵照產品存檔的順序（排序）重建，值不動、照舊逐位比。

    這一處的順序沒有語意：寫在 ``fem_reduced_store.save_reduced``（``_ledger_document`` 抄 ``dict(ledger.seconds)``，
    整份 manifest ``json.dumps(..., sort_keys=True)``），讀在 ``fem_reduced_store._ledger``（照 manifest 的順序、也就是排序後）。
    之後讀它的只有 ``fem_reduced_receipt.save_validation_receipt``（一樣 ``sort_keys=True`` 寫進收據）與
    ``test_fem_reduced::test_build_ledger_records_solves_rounds_and_rank``（比鍵的集合），沒有誰照順序讀。
    """
    seconds = dict(sorted(model.ledger.seconds.items()))
    return dataclasses.replace(model, ledger=dataclasses.replace(model.ledger, seconds=seconds))


def shared_model(tmp_path_factory: pytest.TempPathFactory, worker_id: str,
                 build: Callable[[], ReducedModel]) -> ReducedModel:
    """這一次 pytest 的那一份替身：第一個拿到鎖的工人建、用產品的 ``save_reduced`` 存，大家用 ``load_reduced`` 讀回。"""
    path = _directory(tmp_path_factory, worker_id, "fem-reduced-model") / "rom.npz"

    def produce() -> str:
        calculated = build()
        for stale in (path, manifest_path(path)):  # 前一個工人存完讀回不符就拋了，留下的檔不能擋住重算
            stale.unlink(missing_ok=True)
        save_reduced(calculated, path)
        assert bitwise_equal(load_reduced(path), _seconds_sorted(calculated))
        return str(path)

    return load_reduced(Path(shared_json(tmp_path_factory, worker_id, "fem-reduced-model", produce)))


def _hex(numbers: Mapping[str, float]) -> dict[str, str]:
    return {name: float(value).hex() for name, value in numbers.items()}


def _floats(document: dict[str, dict[str, str]], key: str) -> dict[str, float]:
    return {name: float.fromhex(value) for name, value in document[key].items()}


def _save_report(report: ValidationReport, directory: Path) -> None:
    """兩個複數聲壓陣列進 npz（逐位），其餘欄位進 JSON（浮點一律 ``float.hex()``）。"""
    np.savez(directory / "pressure.npz", **{name: getattr(report, name) for name in _REPORT_ARRAYS})
    document = {
        "frequency_db_by_seat": _hex(report.frequency_db_by_seat), "band_db_by_seat": _hex(report.band_db_by_seat),
        "partial_band_db_by_seat": _hex(report.partial_band_db_by_seat),
        "bands": [[band.lower_hz.hex(), band.upper_hz.hex(), list(band.indices), band.partial] for band in report.bands],
        "full_solves": report.full_solves, "model_digest": report.model_digest,
        "computation_digest": report.computation_digest,
        "source_points": {name: list(point) for name, point in report.source_points.items()},
        "receiver_points": {name: list(point) for name, point in report.receiver_points.items()}}
    (directory / "report.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def _load_report(directory: Path) -> ValidationReport:
    document = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    with np.load(directory / "pressure.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in _REPORT_ARRAYS}
    return ValidationReport(
        frequency_db_by_seat=_floats(document, "frequency_db_by_seat"),
        band_db_by_seat=_floats(document, "band_db_by_seat"),
        partial_band_db_by_seat=_floats(document, "partial_band_db_by_seat"),
        bands=tuple(Band(lower_hz=float.fromhex(lower), upper_hz=float.fromhex(upper), indices=tuple(indices),
                         partial=partial) for lower, upper, indices, partial in document["bands"]),
        full_solves=int(document["full_solves"]), full_pressure=arrays["full_pressure"],
        reduced_pressure=arrays["reduced_pressure"], model_digest=str(document["model_digest"]),
        computation_digest=str(document["computation_digest"]),
        source_points={name: (x, y, z) for name, (x, y, z) in document["source_points"].items()},
        receiver_points={name: (x, y, z) for name, (x, y, z) in document["receiver_points"].items()})


def _sha(array: NDArray[np.generic]) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _validation_inputs(model: ReducedModel, operators: FullOperators,
                       holdouts: Sequence[FrequencyWallImpedances]) -> str:
    """:func:`validate_reduced` 讀到的輸入的指紋：替身的計算條件指紋與完整解殘差上限；完整模型的網格雜湊、空氣、物理身分、
    P2 的剛度、質量與六面邊界質量（CSR 的形狀、dtype 與 data／indices／indptr 位元組）、點源與取值列的位元組、
    喇叭與座位座標；留出材料（``repr`` 對浮點逐位）。不在指紋裡的：P2 的基底物件（``validate_reduced`` 組系統時不讀它）。"""
    parts = [computation_digest(model), model.ledger.settings.model_dump_json(), operators.mesh_hash,
             operators.physics_identity, float(operators.density_kg_m3).hex(), float(operators.sound_speed_m_s).hex(),
             repr(dict(operators.source_points)), repr(dict(operators.receiver_points)), repr(list(holdouts))]
    for rows in (operators.sources, operators.receivers):
        parts += [f"{name}:{_sha(row)}" for name, row in rows.items()]
    for matrix in (operators.stiffness, operators.mass, *operators.boundary):
        parts += [str(matrix.shape), str(matrix.dtype), _sha(matrix.data), _sha(matrix.indices), _sha(matrix.indptr)]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def shared_holdout_report(tmp_path_factory: pytest.TempPathFactory, worker_id: str, model: ReducedModel,
                          operators: FullOperators, holdouts: Sequence[FrequencyWallImpedances]) -> ValidationReport:
    """``validate_reduced(model, operators, holdouts)``，指紋（見 :func:`_validation_inputs`）相同就這一次 pytest
    只做一次；指紋涵蓋的輸入差一位就是另一份，指紋沒涵蓋的不保證。"""
    name = f"fem-reduced-holdout-{_validation_inputs(model, operators, holdouts)[:32]}"
    directory = _directory(tmp_path_factory, worker_id, name)

    def produce() -> str:
        calculated = validate_reduced(model, operators, holdouts)
        _save_report(calculated, directory)
        assert bitwise_equal(_load_report(directory), calculated)
        return str(directory)

    return _load_report(Path(shared_json(tmp_path_factory, worker_id, name, produce)))
