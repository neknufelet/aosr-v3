"""替身驗證收據與准用閘門（#561，老闆 2026-10-28 拍 A′）：建完必做留出驗證、兩把尺都過才准用。

沿用 ``test_fem_reduced`` 的小房間替身與留出材料（``HOLDOUT``）；尺一律從精度契約登記簿讀，收據只抄錄。
閘門只看收據不看建模停止原因：預算停止、殘差目標未達但收據通過照樣准用；殘差達標但收據不通過照樣拒用。
修補一（獨立審查三條）：別份替身的報告寫不成收據、沒做完的驗證寫不成也過不了閘門、到預算之前先量殘差。
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from aosr.physics.fem_reduced import FullOperators, ReducedModel, even_indices
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics.fem_reduced_check import ValidationReport
from aosr.physics.fem_reduced_receipt import (
    HoldoutRulers, ReducedModelNotValidated, Ruler, load_validated_reduced, receipt_path, save_validation_receipt,
)
from aosr.physics.fem_reduced_store import load_reduced, manifest_path, matrices_digest, save_reduced
from tests.engine._precision_contracts import contract_value
# holdout_report、model、operators 是借來的 module 範圍 fixture（本檔各建一次，不跟原檔共用）。
from tests.engine.test_fem_reduced import (
    AXIS, HOLDOUT, SEATS, TRAINING, _settings, holdout_report, model, operators,
)

FREQUENCY_NAME = "fem_reduced_holdout_frequency_db"
BAND_NAME = "fem_reduced_holdout_band_db"
RULERS = HoldoutRulers(frequency=Ruler(FREQUENCY_NAME, contract_value(FREQUENCY_NAME)),
                       band=Ruler(BAND_NAME, contract_value(BAND_NAME)))
BUDGET_STOP = "預算停止、殘差目標未達"
RESIDUAL_STOP = "抽到的候選殘差都在停止值內"
UNFINISHED = "驗證未完成不能當成通過"
VALIDATION_SECONDS = 7.25
IDENTITY = ("manifest_sha256", "npz_sha256", "input_hash", "mesh_hash", "physics_identity")
ARRAYS = ("stiffness", "mass", "boundary", "sources", "receivers")
FIRST_SEAT = next(iter(SEATS))


def _saved(model: ReducedModel, report: ValidationReport, path: Path, rulers: HoldoutRulers = RULERS) -> Path:
    save_reduced(model, path)
    return save_validation_receipt(path, report, rulers=rulers, seconds=VALIDATION_SECONDS)


def _edit(path: Path, change: Callable[[dict[str, object]], object]) -> None:
    receipt = receipt_path(path)
    document = json.loads(receipt.read_text(encoding="utf-8"))
    change(document)
    receipt.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def _same_matrices(left: ReducedModel, right: ReducedModel) -> bool:
    return all(getattr(left.matrices, name).dtype == getattr(right.matrices, name).dtype
               and getattr(left.matrices, name).tobytes() == getattr(right.matrices, name).tobytes()
               for name in ARRAYS)


def test_receipt_roundtrip_bitwise_and_solve_counts_apart(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path) -> None:
    """抓收據掉位、身分從呼叫端抄、建模與驗證次數混成一格：每座位 hex 逐位等於報告，六格身分等於 manifest 與模型。"""
    path = tmp_path / "rom.npz"
    receipt = _saved(model, holdout_report, path)
    assert receipt == receipt_path(path)
    document = json.loads(receipt.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    assert holdout_report.model_digest == matrices_digest(model.matrices)
    assert document["reduced_model"] == {
        **{field: manifest[field] for field in IDENTITY}, "matrices_digest": holdout_report.model_digest}
    for name in SEATS:
        seat = document["seats"][name]
        for field, values in (("frequency_db", holdout_report.frequency_db_by_seat),
                              ("band_db", holdout_report.band_db_by_seat),
                              ("partial_band_db", holdout_report.partial_band_db_by_seat)):
            assert seat[field]["hex"] == values[name].hex()
            assert seat[field]["decimal"] == values[name]
    build, validation = document["build"], document["validation"]
    assert build["full_solves"] == model.ledger.full_solves
    assert build["stop_reason"] == model.ledger.stop_reason == RESIDUAL_STOP
    assert build["max_full_solves"] == model.ledger.settings.max_full_solves
    assert validation["validation_full_solves"] == holdout_report.full_solves == len(HOLDOUT) * len(AXIS)
    assert validation["holdout_sets"] == len(HOLDOUT)
    assert validation["validation_seconds"] == VALIDATION_SECONDS
    assert holdout_report.full_solves > 0  # 兩格若被加在一起，建模那一格就會多出這麼多
    assert "validation_full_solves" not in build and "full_solves" not in validation
    assert document["rulers"]["frequency"] == {
        "name": FREQUENCY_NAME, "value": {"decimal": RULERS.frequency.value, "hex": RULERS.frequency.value.hex()}}
    assert document["rulers"]["band"]["name"] == BAND_NAME
    assert document["within_contract"] is True and document["status"] == "通過"
    with pytest.raises(FileExistsError):
        save_validation_receipt(path, holdout_report, rulers=RULERS, seconds=VALIDATION_SECONDS)


def test_report_from_other_model_cannot_write_receipt(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path) -> None:
    """審查 B1 原樣重現：座位取值列乘 100（聲壓多 40 dB）另存一份，拿原替身的合格報告替它寫收據要被拒。"""
    other = dataclasses.replace(model, matrices=dataclasses.replace(
        model.matrices, receivers=model.matrices.receivers * 100))
    path = tmp_path / "other.npz"
    save_reduced(other, path)
    with pytest.raises(ValueError, match="驗證報告不是對這份替身做的"):
        save_validation_receipt(path, holdout_report, rulers=RULERS, seconds=VALIDATION_SECONDS)
    assert not receipt_path(path).exists()


def _empty(report: ValidationReport) -> tuple[ValidationReport, float]:
    """審查 B2 原樣重現：零組留出、零次完整解、零秒。"""
    return dataclasses.replace(report, full_pressure=report.full_pressure[:0],
                               reduced_pressure=report.reduced_pressure[:0], full_solves=0), 0.0


def _short(report: ValidationReport) -> tuple[ValidationReport, float]:
    return dataclasses.replace(report, full_solves=report.full_solves - 1), VALIDATION_SECONDS


def _instant(report: ValidationReport) -> tuple[ValidationReport, float]:
    return report, 0.0


def _not_finite(report: ValidationReport) -> tuple[ValidationReport, float]:
    return dataclasses.replace(report, frequency_db_by_seat={
        **report.frequency_db_by_seat, FIRST_SEAT: math.nan}), VALIDATION_SECONDS


@pytest.mark.parametrize("damage", (_empty, _short, _instant, _not_finite),
                         ids=("review_empty", "solves_short", "zero_seconds", "nan_seat"))
def test_unfinished_validation_cannot_write_receipt(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path,
        damage: Callable[[ValidationReport], tuple[ValidationReport, float]]) -> None:
    """抓沒做完的驗證寫成通過：零組零次零秒、次數少一次、耗時零、座位數字非有限數，一律不寫收據。"""
    path = tmp_path / "rom.npz"
    save_reduced(model, path)
    report, seconds = damage(holdout_report)
    with pytest.raises(ValueError, match=UNFINISHED):
        save_validation_receipt(path, report, rulers=RULERS, seconds=seconds)
    assert not receipt_path(path).exists()


def _drop(path: Path) -> HoldoutRulers:
    receipt_path(path).unlink()
    return RULERS


def _change(change: Callable[[dict[str, object]], object]) -> Callable[[Path], HoldoutRulers]:
    def mutate(path: Path) -> HoldoutRulers:
        _edit(path, change)
        return RULERS
    return mutate


def _section(document: dict[str, object], *keys: str) -> dict[str, object]:
    node: object = document
    for key in keys:
        assert isinstance(node, dict)
        node = node[key]
    assert isinstance(node, dict)
    return node


def _flip_last(document: dict[str, object], field: str) -> None:
    identity = _section(document, "reduced_model")
    value = str(identity[field])
    identity[field] = value[:-1] + ("0" if value[-1] != "0" else "1")


def _tighten(path: Path) -> HoldoutRulers:
    """逐頻尺傳成收據最大值的一半：收據本身一字不改，換尺之後就該不算。"""
    seats = json.loads(receipt_path(path).read_text(encoding="utf-8"))["seats"]
    largest = max(float.fromhex(seat["frequency_db"]["hex"]) for seat in seats.values())
    return dataclasses.replace(RULERS, frequency=Ruler(FREQUENCY_NAME, largest / 2.0))


_GATE_CASES: tuple[tuple[str, Callable[[Path], HoldoutRulers], str], ...] = (
    ("no_receipt", _drop, UNFINISHED),
    ("npz_sha256", _change(lambda d: _flip_last(d, "npz_sha256")), "npz_sha256 不同"),
    ("physics_identity", _change(lambda d: _flip_last(d, "physics_identity")), "physics_identity 不同"),
    ("matrices_digest", _change(lambda d: _flip_last(d, "matrices_digest")), "matrices_digest 不同"),
    ("status", _change(lambda d: d.__setitem__("status", "不通過")), "狀態是「不通過」"),
    ("within_contract", _change(lambda d: d.__setitem__("within_contract", False)), "within_contract 不是 True"),
    ("tightened_ruler", _tighten, "這次讀進來的尺重判不過"),
    ("drop_validation", _change(lambda d: d.pop("validation")), "沒有驗證區塊"),
    ("zero_solves", _change(lambda d: _section(d, "validation").__setitem__("validation_full_solves", 0)),
     "完整解次數不等於"),
    ("zero_sets", _change(lambda d: _section(d, "validation").__setitem__("holdout_sets", 0)), "留出組數"),
    ("decimal_mismatch", _change(
        lambda d: _section(d, "seats", FIRST_SEAT, "frequency_db").__setitem__("decimal", 999.0)),
     "收據數字兩份不一致"),
    ("bad_hex", _change(lambda d: _section(d, "seats", FIRST_SEAT, "band_db").__setitem__("hex", "zz")),
     "hex 不合法"),
    ("overflow_hex", _change(
        lambda d: _section(d, "seats", FIRST_SEAT, "band_db").__setitem__("hex", "0x1p+999999999")),
     "hex 不合法"),
)


@pytest.mark.parametrize(("mutate", "message"), [case[1:] for case in _GATE_CASES],
                         ids=[case[0] for case in _GATE_CASES])
def test_gate_rejects_with_named_reason(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path,
        mutate: Callable[[Path], HoldoutRulers], message: str) -> None:
    """抓閘門漏一條：沒收據、對到別份替身、狀態不通過、within_contract 假、尺改緊、驗證區塊缺或次數組數不對、
    數字兩份不一致、hex 不合法，各自拋 ReducedModelNotValidated 並說出是哪一條。"""
    path = tmp_path / "rom.npz"
    _saved(model, holdout_report, path)
    rulers = mutate(path)
    with pytest.raises(ReducedModelNotValidated, match=message):
        load_validated_reduced(path, rulers=rulers)


def test_gate_wraps_model_load_failure(model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path) -> None:
    """抓閘門漏包替身本身的錯：npz 壞一個位元也是拋 ReducedModelNotValidated，原錯誤留在 __cause__。"""
    path = tmp_path / "rom.npz"
    _saved(model, holdout_report, path)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x04
    path.write_bytes(bytes(data))
    with pytest.raises(ReducedModelNotValidated, match="替身讀不進來") as caught:
        load_validated_reduced(path, rulers=RULERS)
    assert isinstance(caught.value.__cause__, ValueError)


def test_residual_reached_but_holdout_fails_is_refused(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path) -> None:
    """邊界 2：殘差達標的替身，收據在較緊的尺下寫成「不通過」，之後拿正式尺也一律拒用。"""
    path = tmp_path / "rom.npz"
    strict = dataclasses.replace(RULERS, frequency=Ruler(
        FREQUENCY_NAME, max(holdout_report.frequency_db_by_seat.values()) / 2.0))
    receipt = _saved(model, holdout_report, path, rulers=strict)
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert model.ledger.stop_reason == RESIDUAL_STOP
    assert document["status"] == "不通過" and document["within_contract"] is False
    with pytest.raises(ReducedModelNotValidated, match="不通過"):
        load_validated_reduced(path, rulers=RULERS)


def test_budget_stop_with_passing_receipt_is_allowed(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path) -> None:
    """邊界 1：建模帳記「預算停止、殘差目標未達」但留出驗證通過，閘門放行，停止原因照實帶出。"""
    budget = dataclasses.replace(model, ledger=dataclasses.replace(model.ledger, stop_reason=BUDGET_STOP))
    path = tmp_path / "budget.npz"
    receipt = _saved(budget, holdout_report, path)
    assert json.loads(receipt.read_text(encoding="utf-8"))["build"]["stop_reason"] == BUDGET_STOP
    gated = load_validated_reduced(path, rulers=RULERS, expected_mesh_hash=model.mesh_hash,
                                   expected_physics_identity=model.physics_identity)
    assert gated.ledger.stop_reason == BUDGET_STOP


def test_budget_equal_to_actual_solves_still_measures_residual(
        model: ReducedModel, operators: FullOperators) -> None:
    """審查 B3 原樣重現：上限設成不設上限那次實際的建模次數；最後一批解完先量殘差，已達標就記殘差達標、
    不記「殘差目標未達」，而且矩陣跟不設上限那次逐位相同。"""
    capped = build_reduced_model(operators, AXIS, TRAINING, settings=_settings().model_copy(
        update={"max_full_solves": model.ledger.full_solves}))
    assert capped.ledger.stop_reason == RESIDUAL_STOP
    assert capped.ledger.full_solves == model.ledger.full_solves
    assert capped.ledger.rounds[-1].max_residual <= capped.ledger.settings.stop_residual
    assert _same_matrices(capped, model)


def test_budget_exhausted_records_budget_stop(operators: FullOperators) -> None:
    """抓預算停止記成舊字、沒量殘差就記未達：上限＝起始快照數，起始那批解完先量一輪（真的沒達），
    再記「預算停止、殘差目標未達」，那一輪準備了下一批但不解。"""
    settings = _settings()
    initial = (len(even_indices(settings.initial_impedance_count, len(TRAINING)))
               * len(even_indices(settings.initial_frequency_count, len(AXIS))))
    built = build_reduced_model(operators, AXIS, TRAINING,
                                settings=settings.model_copy(update={"max_full_solves": initial}))
    ledger = built.ledger
    assert ledger.stop_reason == BUDGET_STOP
    assert ledger.full_solves == initial
    assert ledger.rounds and ledger.rounds[-1].max_residual > settings.stop_residual
    assert ledger.rounds[-1].solved == 0 and ledger.rounds[-1].proposed > 0


def test_gate_returns_same_model_as_load_reduced(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path) -> None:
    """抓閘門回的不是核過的那一份：放行時回的模型跟 load_reduced 逐陣列相同、出身與建模帳相同。"""
    path = tmp_path / "rom.npz"
    _saved(model, holdout_report, path)
    gated = load_validated_reduced(path, rulers=RULERS)
    loaded = load_reduced(path)
    assert _same_matrices(gated, loaded)
    for name in ARRAYS:
        assert getattr(loaded.matrices, name).shape == getattr(gated.matrices, name).shape
    assert dataclasses.replace(gated, matrices=loaded.matrices) == loaded
