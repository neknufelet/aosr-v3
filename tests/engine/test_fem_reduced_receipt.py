"""替身驗證收據與准用閘門（#561，老闆 2026-10-28 拍 A′）：建完必做留出驗證、兩把尺都過才准用。

沿用 ``test_fem_reduced`` 的小房間替身與留出材料（``HOLDOUT``）；尺一律從精度契約登記簿讀，收據只抄錄。
閘門只看收據不看建模停止原因：預算停止、殘差目標未達但收據通過照樣准用；殘差達標但收據不通過照樣拒用。
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from aosr.physics.fem_reduced import FullOperators, ReducedModel, even_indices
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics.fem_reduced_check import ValidationReport
from aosr.physics.fem_reduced_receipt import (
    HoldoutRulers, ReducedModelNotValidated, Ruler, load_validated_reduced, receipt_path, save_validation_receipt,
)
from aosr.physics.fem_reduced_store import load_reduced, manifest_path, save_reduced
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
VALIDATION_SECONDS = 7.25
IDENTITY = ("manifest_sha256", "npz_sha256", "input_hash", "mesh_hash", "physics_identity")


def _saved(model: ReducedModel, report: ValidationReport, path: Path, rulers: HoldoutRulers = RULERS) -> Path:
    save_reduced(model, path)
    return save_validation_receipt(path, report, rulers=rulers, seconds=VALIDATION_SECONDS)


def _edit(path: Path, change: Callable[[dict[str, object]], None]) -> None:
    receipt = receipt_path(path)
    document = json.loads(receipt.read_text(encoding="utf-8"))
    change(document)
    receipt.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def test_receipt_roundtrip_bitwise_and_solve_counts_apart(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path) -> None:
    """抓收據掉位、身分從呼叫端抄、建模與驗證次數混成一格：每座位 hex 逐位等於報告，五格身分等於 manifest。"""
    path = tmp_path / "rom.npz"
    receipt = _saved(model, holdout_report, path)
    assert receipt == receipt_path(path)
    document = json.loads(receipt.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    assert document["reduced_model"] == {field: manifest[field] for field in IDENTITY}
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


def _drop(path: Path) -> HoldoutRulers:
    receipt_path(path).unlink()
    return RULERS


def _swap_npz_sha(path: Path) -> HoldoutRulers:
    def change(document: dict[str, object]) -> None:
        identity = document["reduced_model"]
        assert isinstance(identity, dict)
        value = str(identity["npz_sha256"])
        identity["npz_sha256"] = value[:-1] + ("0" if value[-1] != "0" else "1")
    _edit(path, change)
    return RULERS


def _set(field: str, value: object) -> Callable[[Path], HoldoutRulers]:
    def mutate(path: Path) -> HoldoutRulers:
        _edit(path, lambda document: document.__setitem__(field, value))
        return RULERS
    return mutate


def _tighten(path: Path) -> HoldoutRulers:
    """逐頻尺傳成收據最大值的一半：收據本身一字不改，換尺之後就該不算。"""
    seats = json.loads(receipt_path(path).read_text(encoding="utf-8"))["seats"]
    largest = max(float.fromhex(seat["frequency_db"]["hex"]) for seat in seats.values())
    return dataclasses.replace(RULERS, frequency=Ruler(FREQUENCY_NAME, largest / 2.0))


@pytest.mark.parametrize(("mutate", "message"), (
    (_drop, "驗證未完成不能當成通過"),
    (_swap_npz_sha, "npz_sha256 不同"),
    (_set("status", "不通過"), "狀態是「不通過」"),
    (_set("within_contract", False), "within_contract 不是 True"),
    (_tighten, "這次讀進來的尺重判不過"),
), ids=("no_receipt", "npz_sha256", "status", "within_contract", "tightened_ruler"))
def test_gate_rejects_with_named_reason(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path,
        mutate: Callable[[Path], HoldoutRulers], message: str) -> None:
    """抓閘門漏一條：沒收據、收據對到別份替身、狀態不通過、within_contract 假、尺改緊，各自拒用並說出是哪一條。"""
    path = tmp_path / "rom.npz"
    _saved(model, holdout_report, path)
    rulers = mutate(path)
    with pytest.raises(ReducedModelNotValidated, match=message):
        load_validated_reduced(path, rulers=rulers)


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


def test_budget_exhausted_records_budget_stop(operators: FullOperators) -> None:
    """抓預算停止記成舊字或別的原因：完整解上限比起始快照還少，第一批就解到上限、不進貪婪輪。"""
    settings = _settings()
    training = TRAINING[:1]
    initial = (len(even_indices(settings.initial_impedance_count, len(training)))
               * len(even_indices(settings.initial_frequency_count, len(AXIS))))
    built = build_reduced_model(operators, AXIS, training,
                                settings=settings.model_copy(update={"max_full_solves": initial - 1}))
    assert built.ledger.stop_reason == BUDGET_STOP
    assert built.ledger.full_solves == initial - 1
    assert not built.ledger.rounds


def test_gate_returns_same_model_as_load_reduced(
        model: ReducedModel, holdout_report: ValidationReport, tmp_path: Path) -> None:
    """抓閘門回的不是核過的那一份：放行時回的模型跟 load_reduced 逐陣列相同、出身與建模帳相同。"""
    path = tmp_path / "rom.npz"
    _saved(model, holdout_report, path)
    gated = load_validated_reduced(path, rulers=RULERS)
    loaded = load_reduced(path)
    for name in ("stiffness", "mass", "boundary", "sources", "receivers"):
        before, after = getattr(loaded.matrices, name), getattr(gated.matrices, name)
        assert before.dtype == after.dtype and before.shape == after.shape
        assert before.tobytes() == after.tobytes()
    assert dataclasses.replace(gated, matrices=loaded.matrices) == loaded
