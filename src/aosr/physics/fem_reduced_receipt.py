"""替身驗證收據與准用閘門（#561，老闆 2026-10-28 拍 A′）。

建模（:func:`aosr.physics.fem_reduced_build.build_reduced_model`）殘差達標或完整解預算用完就停；停在哪裡都還不能用，
建完必做留出驗證（:func:`aosr.physics.fem_reduced_check.validate_reduced`），兩把尺（精度契約登記簿那兩條）
都過才准用。這一支把驗證結果寫成 ``<名>.validation.json``，跟同名的 npz／manifest 放在一起：

- 收據對到那一份替身：manifest 整份摘要、``npz_sha256``、輸入雜湊、網格雜湊、物理身分從核過的 manifest 抄，
  模型內容摘要（``matrices_digest``）由載入的模型現算，不從呼叫端信；驗證報告帶的模型內容摘要必須等於它，
  別份替身的報告寫不成這份的收據。
- 驗證要整份做完才寫得成收據：留出至少一組、完整解次數＝組數×頻點數、耗時大於零、每座位兩格數字都是有限數。
- 建模帳摘要（``full_solves`` 只算建模）與驗證另列（留出組數、``validation_full_solves``、``validation_seconds``）
  分兩格寫，不相加；建模預算 ``max_full_solves`` 不含驗證。
- 每座位逐頻、完整覆蓋帶、部分覆蓋帶的 dB 差，hex 一份（權威，閘門重判用這份）、十進位一份給人看。
- 兩把尺的名字與當次讀到的值只是抄錄；數字由呼叫端從登記簿讀進來，這裡不寫任何尺的數字。

:func:`load_validated_reduced` 是准用閘門：替身讀不進來、收據缺（驗證未完成不能當成通過）、讀不開、版本不對、
對不到這份替身、驗證區塊缺或次數對不上、狀態不是「通過」、``within_contract`` 不是 True、數字兩份不一致、
或拿收據抄的每座位數字用這次讀進來的尺重判不過，一律拋 :class:`ReducedModelNotValidated`。
閘門不看建模的停止原因：預算停止、殘差目標未達但收據通過照樣准用；殘差達標但收據不通過照樣拒用。

版本 2（#561 辛一審查 B1）：``reduced_model`` 區多記喇叭與座位座標（名字 → 三個 ``float.hex()``，從驗證報告抄，
報告是從完整模型的算子取的）。網格雜湊與物理身分都不含位置，同一間房把喇叭移 0.1 m 的方案拿舊替身會照樣過、
目標靜默算錯；呼叫端給了 ``expected_source_points``／``expected_receiver_points`` 就逐名逐座標字串比，
不同就拒用並點名哪一支喇叭或哪一個座位。版本 1 收據沒有座標：呼叫端沒給預期座標時照舊放行，
給了就一律拒（收據沒記座標，驗證未完成不能當成通過）。
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_check import PointHex, ValidationReport, seats_within_contract, within_reduced_contract
from aosr.physics.fem_reduced_store import computation_digest, load_reduced, manifest_path, matrices_digest

RECEIPT_FORMAT_VERSION = 2
_READABLE_VERSIONS = (1, 2)
_POINT_KINDS = (("source_points", "喇叭"), ("receiver_points", "座位"))
RECEIPT_STATUSES = ("通過", "不通過")
_PASSED = RECEIPT_STATUSES[0]
_UNFINISHED = "驗證未完成不能當成通過"


class ReducedModelNotValidated(ValueError):
    """准用閘門拒用：訊息說是哪一條。"""


@dataclass(frozen=True)
class Ruler:
    """一把尺：精度契約登記簿裡的名字，與這次從登記簿讀到的值（呼叫端讀好傳進來）。"""

    name: str
    value: float


@dataclass(frozen=True)
class HoldoutRulers:
    """留出驗證的兩把尺：每座位逐頻最大 dB 差、完整覆蓋 1/3 八度帶的最大帶能量 dB 差。"""

    frequency: Ruler
    band: Ruler


def receipt_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + ".validation.json")


def _model_identity(npz_path: Path, model: ReducedModel) -> dict[str, str]:
    """對到這份替身的六格：兩個雜湊從 manifest 原文抄（:func:`load_reduced` 剛核過），三格取它讀出的模型，
    模型內容摘要現算。"""
    document = json.loads(manifest_path(npz_path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("替身 manifest 格式不符")
    return {"manifest_sha256": str(document["manifest_sha256"]), "npz_sha256": str(document["npz_sha256"]),
            "input_hash": model.input_hash, "mesh_hash": model.mesh_hash,
            "physics_identity": model.physics_identity, "matrices_digest": matrices_digest(model.matrices)}


def _number(value: float) -> dict[str, float | str | None]:
    """hex 一份（逐位）、十進位一份給人看；非有限數十進位記 null，hex 照記（``nan``／``inf``）。"""
    return {"decimal": value if math.isfinite(value) else None, "hex": value.hex()}


def _checked_report(report: ValidationReport, model: ReducedModel, seconds: float) -> None:
    """報告要對這份替身、而且整份做完；不然就不寫收據。"""
    if report.model_digest != matrices_digest(model.matrices):
        raise ValueError("驗證報告不是對這份替身做的（模型內容摘要不同）")
    if report.computation_digest != computation_digest(model):
        raise ValueError("驗證報告不是對這份替身的計算條件做的（頻率軸、空氣、喇叭／座位順序或出身不同）")
    seats = set(model.receiver_names)
    if set(report.frequency_db_by_seat) != seats or set(report.band_db_by_seat) != seats:
        raise ValueError("驗證報告的座位跟替身不同")
    if report.partial_band_db_by_seat and set(report.partial_band_db_by_seat) != seats:
        raise ValueError("驗證報告部分覆蓋帶的座位跟替身不同")
    shape = (len(model.frequencies_hz), len(model.source_names), len(model.receiver_names))
    if report.full_pressure.shape[1:] != shape or report.reduced_pressure.shape != report.full_pressure.shape:
        raise ValueError("驗證報告的頻點、喇叭或座位數跟替身不同")
    holdouts = report.full_pressure.shape[0]
    if holdouts < 1:
        raise ValueError(f"驗證報告沒有任何留出材料：{_UNFINISHED}")
    if report.full_solves != holdouts * len(model.frequencies_hz):
        raise ValueError(f"驗證報告的完整解次數不等於留出組數×頻點數，報告不是整份做完的：{_UNFINISHED}")
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"驗證耗時必須是大於零的有限數：{_UNFINISHED}")
    values = (*report.frequency_db_by_seat.values(), *report.band_db_by_seat.values())
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"驗證報告有座位的逐頻或帶數字不是有限數：{_UNFINISHED}")
    for points, names, label in ((report.source_points, model.source_names, "喇叭"),
                                 (report.receiver_points, model.receiver_names, "座位")):
        if set(points) != set(names) or not all(_is_point_hex(list(value)) for value in points.values()):
            raise ValueError(f"驗證報告的{label}座標跟替身的{label}不同或不是三個有限的 float.hex：{_UNFINISHED}")


def save_validation_receipt(npz_path: Path, report: ValidationReport, *, rulers: HoldoutRulers,
                            seconds: float) -> Path:
    """寫 ``<名>.validation.json``，回收據路徑；``seconds`` 是驗證（不含建模）的耗時。

    先 :func:`load_reduced` 核過那份替身，再核報告是對它做的、而且整份做完（見 :func:`_checked_report`）；
    收據已存在就報錯，不覆寫。
    """
    model = load_reduced(npz_path)
    _checked_report(report, model, seconds)
    within = within_reduced_contract(report, frequency_db=rulers.frequency.value, band_db=rulers.band.value)
    ledger = model.ledger
    seats = {
        name: {"frequency_db": _number(report.frequency_db_by_seat[name]),
               "band_db": _number(report.band_db_by_seat[name]),
               "partial_band_db": (_number(report.partial_band_db_by_seat[name])
                                   if report.partial_band_db_by_seat else None)}
        for name in model.receiver_names}
    document: dict[str, object] = {
        "format_version": RECEIPT_FORMAT_VERSION,
        "reduced_model": {
            **_model_identity(npz_path, model),
            "source_points": {name: list(report.source_points[name]) for name in model.source_names},
            "receiver_points": {name: list(report.receiver_points[name]) for name in model.receiver_names}},
        "build": {"full_solves": ledger.full_solves, "stop_reason": ledger.stop_reason, "rank": ledger.rank,
                  "seconds": dict(ledger.seconds), "max_full_solves": ledger.settings.max_full_solves,
                  "stop_residual": ledger.settings.stop_residual},
        "validation": {"holdout_sets": int(report.full_pressure.shape[0]),
                       "validation_full_solves": report.full_solves, "validation_seconds": seconds},
        "budget_note": "build.full_solves 與 max_full_solves 只算建模；留出驗證的完整解次數與耗時另列在 validation，不相加",
        "seats": seats,
        "rulers": {key: {"name": ruler.name, "value": _number(ruler.value)}
                   for key, ruler in (("frequency", rulers.frequency), ("band", rulers.band))},
        "within_contract": within,
        "status": _PASSED if within else RECEIPT_STATUSES[1],
    }
    text = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + "\n"
    target = receipt_path(npz_path)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return target


def _section(document: Mapping[str, object], key: str) -> dict[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ReducedModelNotValidated(f"驗證收據的 {key} 格式不符")
    return value


def _read_receipt(npz_path: Path) -> dict[str, object]:
    path = receipt_path(npz_path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReducedModelNotValidated(f"替身沒有驗證收據 {path}：{_UNFINISHED}，拒用") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReducedModelNotValidated(f"驗證收據讀不開：{path}") from exc
    if not isinstance(document, dict):
        raise ReducedModelNotValidated(f"驗證收據讀不開：{path}")
    version = document.get("format_version")
    if type(version) is not int or version not in _READABLE_VERSIONS:
        raise ReducedModelNotValidated("驗證收據格式版本不符")
    return document


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_point_hex(value: object) -> bool:
    """三個字串、每個都是正規化的有限 ``float.hex()``。"""
    if not isinstance(value, list) or len(value) != len(("x", "y", "z")):
        return False
    for item in value:
        if not isinstance(item, str):
            return False
        try:
            number = float.fromhex(item)
        except (ValueError, OverflowError):
            return False
        if not math.isfinite(number) or number.hex() != item:
            return False
    return True


def _check_points(document: Mapping[str, object], model: ReducedModel,
                  expected: Mapping[str, Mapping[str, PointHex] | None]) -> None:
    """版本 2 收據的座標要齊、格式要對；呼叫端給了預期座標就逐名逐座標比（版本 1 收據沒有座標，給了預期就拒）。"""
    recorded = _section(document, "reduced_model")
    if document.get("format_version") == 1:
        if any(value is not None for value in expected.values()):
            raise ReducedModelNotValidated(f"驗證收據是版本 1，收據沒記座標，{_UNFINISHED}（這次要核喇叭與座位座標），拒用")
        return
    for key, label in _POINT_KINDS:
        points = recorded.get(key)
        names = model.source_names if key == "source_points" else model.receiver_names
        if (not isinstance(points, dict) or set(points) != set(names)
                or not all(_is_point_hex(value) for value in points.values())):
            raise ReducedModelNotValidated(f"驗證收據的{label}座標缺、名字跟替身不同或不是 float.hex，拒用")
        wanted = expected[key]
        if wanted is None:
            continue
        if set(wanted) != set(points):
            raise ReducedModelNotValidated(
                f"這次方案的{label} {sorted(wanted)} 跟驗證收據記的 {sorted(points)} 不同，拒用")
        for name in names:
            if points[name] != list(wanted[name]):
                raise ReducedModelNotValidated(
                    f"{label} {name} 的座標跟驗證收據記的不同（收據 {points[name]}、這次方案 {list(wanted[name])}）："
                    "替身是別的位置建的，拒用")


def _check_validation(document: Mapping[str, object], model: ReducedModel) -> None:
    """驗證區塊要在、而且是整份做完的：組數 ≥ 1、完整解次數＝組數×這份替身的頻點數、耗時 > 0。"""
    validation = document.get("validation")
    if not isinstance(validation, dict):
        raise ReducedModelNotValidated(f"驗證收據沒有驗證區塊：{_UNFINISHED}，拒用")
    sets, solves, seconds = (validation.get(key) for key in (
        "holdout_sets", "validation_full_solves", "validation_seconds"))
    if not _is_int(sets) or sets < 1:
        raise ReducedModelNotValidated(f"驗證收據的留出組數不是正整數：{_UNFINISHED}，拒用")
    if not _is_int(solves) or solves != sets * len(model.frequencies_hz):
        raise ReducedModelNotValidated(f"驗證收據的完整解次數不等於留出組數×頻點數：{_UNFINISHED}，拒用")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0.0 < seconds < math.inf:
        raise ReducedModelNotValidated(f"驗證收據的驗證耗時不是大於零的有限數：{_UNFINISHED}，拒用")


def _hex_value(name: str, field: str, number: object) -> float:
    """hex 是權威；十進位那份必須等於 hex 還原的值（非有限數十進位記 null），不一致就拒。"""
    if not isinstance(number, dict) or not isinstance(number.get("hex"), str):
        raise ReducedModelNotValidated(f"驗證收據座位 {name} 的 {field} 格式不符")
    try:
        value = float.fromhex(str(number["hex"]))
    except (ValueError, OverflowError) as exc:  # 超大指數是 OverflowError，一樣是不合法的 hex
        raise ReducedModelNotValidated(f"驗證收據座位 {name} 的 {field} hex 不合法") from exc
    decimal = number.get("decimal")
    expected = value if math.isfinite(value) else None
    if type(decimal) is not type(expected) or decimal != expected:
        raise ReducedModelNotValidated(f"驗證收據座位 {name} 的 {field} 收據數字兩份不一致（hex 與十進位）")
    return value


def _seat_values(document: Mapping[str, object], model: ReducedModel, field: str) -> dict[str, float]:
    """收據抄的每座位數字，取 hex 那一份還原成 float；座位要跟替身一樣。"""
    seats = _section(document, "seats")
    if set(seats) != set(model.receiver_names):
        raise ReducedModelNotValidated("驗證收據的座位跟替身不同")
    values: dict[str, float] = {}
    for name, entry in seats.items():
        number = entry.get(field) if isinstance(entry, dict) else None
        if field == "partial_band_db" and number is None:
            continue
        values[name] = _hex_value(name, field, number)
    return values


def _check_verdict(document: Mapping[str, object], model: ReducedModel, rulers: HoldoutRulers) -> None:
    status = document.get("status")
    if status not in RECEIPT_STATUSES:
        raise ReducedModelNotValidated(f"驗證收據狀態「{status}」不在封閉集合 {RECEIPT_STATUSES}，拒用")
    if status != _PASSED:
        raise ReducedModelNotValidated(f"驗證收據狀態是「{status}」，拒用")
    if document.get("within_contract") is not True:
        raise ReducedModelNotValidated("驗證收據的 within_contract 不是 True，拒用")
    recorded = _section(document, "rulers")
    for key, ruler in (("frequency", rulers.frequency), ("band", rulers.band)):
        entry = recorded.get(key)
        if not isinstance(entry, dict) or entry.get("name") != ruler.name:
            raise ReducedModelNotValidated(f"驗證收據記的尺名跟這次的 {ruler.name} 不同，拒用")
    _seat_values(document, model, "partial_band_db")  # 只報不判，但兩份數字一樣要一致
    if not seats_within_contract(
            _seat_values(document, model, "frequency_db"), _seat_values(document, model, "band_db"),
            frequency_db=rulers.frequency.value, band_db=rulers.band.value):
        raise ReducedModelNotValidated("收據抄的每座位數字用這次讀進來的尺重判不過（尺改緊了，舊收據不算），拒用")


def load_validated_reduced(npz_path: Path, *, rulers: HoldoutRulers, expected_mesh_hash: str | None = None,
                           expected_physics_identity: str | None = None,
                           expected_source_points: Mapping[str, PointHex] | None = None,
                           expected_receiver_points: Mapping[str, PointHex] | None = None) -> ReducedModel:
    """准用閘門：:func:`load_reduced` 核完替身，再核驗證收據；任何一條不過都拋 :class:`ReducedModelNotValidated`
    （替身本身讀不進來也包成它，原錯誤留在 ``__cause__``）。給了預期的喇叭或座位座標（名字 → 三個 ``float.hex()``）
    就跟收據記的逐名逐座標比；版本 1 收據沒有座標，給了預期一律拒。"""
    try:
        model = load_reduced(npz_path, expected_mesh_hash=expected_mesh_hash,
                             expected_physics_identity=expected_physics_identity)
        identity = _model_identity(npz_path, model)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReducedModelNotValidated(f"替身讀不進來或對不上，拒用：{exc}") from exc
    document = _read_receipt(npz_path)
    recorded = _section(document, "reduced_model")
    for field, value in identity.items():
        if recorded.get(field) != value:
            raise ReducedModelNotValidated(f"驗證收據對不到這份替身：{field} 不同（替身換過或收據是別份的），拒用")
    _check_points(document, model, {"source_points": expected_source_points,
                                    "receiver_points": expected_receiver_points})
    _check_validation(document, model)
    _check_verdict(document, model, rulers)
    return model
