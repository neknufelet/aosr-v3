"""完整模型逐位指紋與跨物理身分沿用；獨立完整解守算式漂移。"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from aosr.physics import fem_reduced as reduced
from aosr.physics import fem_reduced_check as check
from aosr.physics import fem_reduced_store as store
from aosr.physics.fem_reduced import FullOperators, ReducedModel, even_indices
from aosr.physics.fem_reduced_check import ValidationReport, points_hex
from aosr.physics.fem_reduced_boundary import BoundaryImpedances, PatchKeys, boundary_patches
from aosr.physics.fem_reduced_receipt import ReducedModelNotValidated, load_validated_reduced, receipt_path
from tests.engine.test_fem_reduced import AXIS, HOLDOUT, holdout_report, model, operators
from tests.engine.test_fem_reduced_receipt import RULERS, _saved

OTHER_IDENTITY = "phys-v1:" + "1" * 64


def test_digest_canonical_sparse_and_identity(operators: FullOperators) -> None:
    """抓顯式零、索引寬度、映射順序或身分造成假差異；不能改動原矩陣。"""
    original = reduced.operators_digest(operators)
    matrix = operators.stiffness.copy()
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    absent = next((i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i, j] == 0)
    extra = csr_matrix((np.append(matrix.data, 0.0),
                        (np.append(rows, absent[0]), np.append(matrix.indices, absent[1]))), shape=matrix.shape)
    extra.indices = extra.indices.astype(np.int64)
    extra.indptr = extra.indptr.astype(np.int64)
    changed = dataclasses.replace(operators, p2=dataclasses.replace(operators.p2, stiffness=extra),
                                  physics_identity=OTHER_IDENTITY, sources=dict(reversed(tuple(operators.sources.items()))))
    assert reduced.operators_digest(changed) == original == reduced.operators_digest(operators)
    assert extra.nnz == matrix.nnz + 1


@pytest.mark.parametrize("field", ("stiffness", "patch_keys", "sources", "receivers", "density", "speed", "mesh"))
def test_digest_changes_with_full_model(operators: FullOperators, field: str) -> None:
    """抓漏雜湊任何一個完整模型輸出。"""
    changed = operators
    if field == "stiffness":
        matrix = operators.stiffness.copy()
        matrix.data[0] = np.nextafter(matrix.data[0], np.inf)
        changed = dataclasses.replace(operators, p2=dataclasses.replace(operators.p2, stiffness=matrix))
    elif field == "patch_keys":
        changed = dataclasses.replace(operators, patch_keys=tuple(reversed(operators.patch_keys)))
    elif field == "sources":
        loads = {name: row.copy() for name, row in operators.sources.items()}
        loads[next(iter(loads))][0] += 1
        changed = dataclasses.replace(operators, sources=loads)
    elif field == "receivers":
        probes = {name: row.copy() for name, row in operators.receivers.items()}
        probes[next(iter(probes))][0] += 1
        changed = dataclasses.replace(operators, receivers=probes)
    elif field == "density":
        changed = dataclasses.replace(operators, density_kg_m3=np.nextafter(operators.density_kg_m3, np.inf))
    elif field == "speed":
        changed = dataclasses.replace(operators, sound_speed_m_s=np.nextafter(operators.sound_speed_m_s, np.inf))
    else:
        changed = dataclasses.replace(operators, mesh_hash="different")
    assert reduced.operators_digest(changed) != reduced.operators_digest(operators)


def _stamped(path: Path, model: ReducedModel, report: ValidationReport, operators: FullOperators) -> Path:
    _saved(model, report, path)
    return store.save_operators_stamp(path, operators, rulers=RULERS)


def test_stamp_roundtrip_exclusive(model: ReducedModel, holdout_report: ValidationReport,
                                  operators: FullOperators, tmp_path: Path) -> None:
    path = tmp_path / "rom.npz"
    stamp = _stamped(path, model, holdout_report, operators)
    assert stamp == path.with_suffix(".operators.json")
    document = store.load_operators_stamp(path)
    manifest = json.loads(store.manifest_path(path).read_text())
    assert document["manifest_sha256"] == manifest["manifest_sha256"]
    assert document["npz_sha256"] == manifest["npz_sha256"]
    assert document["physics_identity"] == document["stamped_physics_identity"] == model.physics_identity
    assert document["operators_digest"] == reduced.operators_digest(operators)
    before = stamp.read_bytes()
    assert store.save_operators_stamp(path, operators, rulers=RULERS) == stamp
    assert stamp.read_bytes() == before
    document["operators_digest"] = "f" * 64
    stamp.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="不同|覆寫"):
        store.save_operators_stamp(path, operators, rulers=RULERS)


def test_identity_change_equal_digest_passes_other_checks(model: ReducedModel, holdout_report: ValidationReport,
                                                         operators: FullOperators, tmp_path: Path) -> None:
    path = tmp_path / "rom.npz"
    _stamped(path, model, holdout_report, operators)
    loaded = load_validated_reduced(path, rulers=RULERS, expected_physics_identity=OTHER_IDENTITY,
                                    current_operators_digest=reduced.operators_digest(operators))
    assert loaded.physics_identity == model.physics_identity
    with pytest.raises(ReducedModelNotValidated, match="網格"):
        load_validated_reduced(path, rulers=RULERS, expected_mesh_hash="wrong", expected_physics_identity=OTHER_IDENTITY,
                               current_operators_digest=reduced.operators_digest(operators))
    moved = points_hex(operators.source_points)
    moved["left"] = ("0x0.0p+0", *moved["left"][1:])
    with pytest.raises(ReducedModelNotValidated, match="座標"):
        load_validated_reduced(path, rulers=RULERS, expected_physics_identity=OTHER_IDENTITY,
                               current_operators_digest=reduced.operators_digest(operators), expected_source_points=moved)
    receipt = receipt_path(path)
    document = json.loads(receipt.read_text())
    document["within_contract"] = False
    receipt.write_text(json.dumps(document))
    with pytest.raises(ReducedModelNotValidated, match="within_contract"):
        load_validated_reduced(path, rulers=RULERS, expected_physics_identity=OTHER_IDENTITY,
                               current_operators_digest=reduced.operators_digest(operators))


@pytest.mark.parametrize("damage", ("missing", "digest", "npz_sha256", "manifest_sha256", "physics_identity", "version", "field"))
def test_carryover_rejects_bad_stamp(model: ReducedModel, holdout_report: ValidationReport,
                                   operators: FullOperators, tmp_path: Path, damage: str) -> None:
    path = tmp_path / "rom.npz"
    stamp = _stamped(path, model, holdout_report, operators)
    document = json.loads(stamp.read_text())
    reason = "指紋"
    if damage == "missing":
        stamp.unlink()
        reason = "沒有指紋小檔.*重建"
    else:
        if damage == "digest":
            document["operators_digest"] = "f" * 64
            reason = "完整模型指紋不同.*完整模型真的變了.*重建"
        elif damage == "version":
            document["format_version"] = 999
        elif damage == "field":
            document.pop("stamped_physics_identity")
        else:
            document[damage] = "wrong"
        stamp.write_text(json.dumps(document))
    with pytest.raises(ReducedModelNotValidated, match=reason):
        load_validated_reduced(path, rulers=RULERS, expected_physics_identity=OTHER_IDENTITY,
                               current_operators_digest=reduced.operators_digest(operators))


def test_same_identity_never_reads_stamp(model: ReducedModel, holdout_report: ValidationReport,
                                         operators: FullOperators, tmp_path: Path) -> None:
    path = tmp_path / "rom.npz"
    stamp = _stamped(path, model, holdout_report, operators)
    stamp.write_text("invalid json")
    assert load_validated_reduced(path, rulers=RULERS, expected_physics_identity=model.physics_identity,
                                  current_operators_digest="wrong").physics_identity == model.physics_identity


@pytest.mark.parametrize("damage", ("identity", "receipt", "patches", "points"))
def test_stamp_requires_original_complete_gate(model: ReducedModel, holdout_report: ValidationReport,
                                               operators: FullOperators, tmp_path: Path, damage: str) -> None:
    path = tmp_path / "rom.npz"
    _saved(model, holdout_report, path)
    if damage == "identity":
        operators = dataclasses.replace(operators, physics_identity=OTHER_IDENTITY)
    elif damage == "receipt":
        receipt_path(path).unlink()
    elif damage == "patches":
        operators = dataclasses.replace(operators, patch_keys=tuple(reversed(operators.patch_keys)))
    else:
        operators = dataclasses.replace(operators, source_points={**operators.source_points, "left": operators.source_points["right"]})
    with pytest.raises(ValueError):
        store.save_operators_stamp(path, operators, rulers=RULERS)
    assert not path.with_suffix(".operators.json").exists()


@pytest.mark.parametrize("flip", (False, True), ids=("product", "reversed_system_coefficients"))
def test_spot_check_detects_equation_drift(model: ReducedModel, operators: FullOperators,
                                          monkeypatch: pytest.MonkeyPatch, flip: bool) -> None:
    indices = even_indices(8, len(AXIS))
    axis = tuple(AXIS[i] for i in indices)
    walls = {key: tuple(values[i] for i in indices)
             for key, values in reduced.impedance_rows([HOLDOUT[0]], AXIS)[0].items()}
    if flip:
        original = reduced.system_coefficients

        def reversed_coefficients(frequency_hz: float, wall_impedances: BoundaryImpedances, *,
                                   density_kg_m3: float, sound_speed_m_s: float,
                                   patch_keys: PatchKeys = boundary_patches()
                                   ) -> tuple[float, np.typing.NDArray[np.complex128]]:
            k, coefficients = original(frequency_hz, wall_impedances, density_kg_m3=density_kg_m3,
                                       sound_speed_m_s=sound_speed_m_s, patch_keys=patch_keys)
            return -k, -coefficients

        monkeypatch.setattr(check, "system_coefficients", reversed_coefficients)
    result = check.spot_check_reduced(model, operators, walls, axis, ruler_name=RULERS.frequency.name,
                                      frequency_db=RULERS.frequency.value)
    assert result.frequencies_hz == axis
    assert result.ruler_name == RULERS.frequency.name and result.ruler_value == RULERS.frequency.value
    assert result.seconds > 0
    assert result.passed is (not flip), dataclasses.asdict(result)
    if flip:
        sys.stdout.write(f"植錯抽查：最大差 {result.max_db} dB > 尺 {result.ruler_value} dB；passed={result.passed}\n")


@pytest.mark.parametrize("replacement", ("npz", "identity"))
def test_stamp_rejects_replacement_valid_artifacts(model: ReducedModel, holdout_report: ValidationReport,
                                                  operators: FullOperators, tmp_path: Path, replacement: str) -> None:
    """保留小檔，整組換成各自合法的替身、manifest、收據，不能只核小檔自己抄的綁定。"""
    path = tmp_path / "rom.npz"
    _stamped(path, model, holdout_report, operators)
    if replacement == "npz":
        other = dataclasses.replace(model, matrices=dataclasses.replace(model.matrices, stiffness=model.matrices.stiffness * 2))
    else:
        other = dataclasses.replace(model, physics_identity=OTHER_IDENTITY)
    report = dataclasses.replace(holdout_report, model_digest=store.matrices_digest(other.matrices),
                                 computation_digest=store.computation_digest(other))
    alternate = tmp_path / "alternate.npz"
    _saved(other, report, alternate)
    for old, new in ((path, alternate), (store.manifest_path(path), store.manifest_path(alternate)),
                     (receipt_path(path), receipt_path(alternate))):
        old.write_bytes(new.read_bytes())
    # 三件組本身合法；原小檔綁定到先前那一組，所以跨身分沿用仍須拒收。
    assert store.load_reduced(path).physics_identity == other.physics_identity
    with pytest.raises(ReducedModelNotValidated, match="指紋小檔綁定不同"):
        load_validated_reduced(path, rulers=RULERS, expected_physics_identity="another-current",
                               current_operators_digest=reduced.operators_digest(operators))


@pytest.mark.parametrize("workers", (1, 2))
def test_spot_check_aligns_names_independent_of_mapping_order(model: ReducedModel, operators: FullOperators, workers: int) -> None:
    """名字排序的同指紋必須可沿用，完整解的插入序不能把喇叭或座位配錯。"""
    reordered = dataclasses.replace(operators, sources=dict(reversed(tuple(operators.sources.items()))),
                                    receivers=dict(reversed(tuple(operators.receivers.items()))))
    assert reduced.operators_digest(reordered) == reduced.operators_digest(operators)
    indices = even_indices(8, len(AXIS))
    axis = tuple(AXIS[i] for i in indices)
    walls = {key: tuple(values[i] for i in indices)
             for key, values in reduced.impedance_rows([HOLDOUT[0]], AXIS)[0].items()}
    reference = check.spot_check_reduced(model, operators, walls, axis, ruler_name=RULERS.frequency.name,
                                         frequency_db=RULERS.frequency.value)
    actual = check.spot_check_reduced(model, reordered, walls, axis, ruler_name=RULERS.frequency.name,
                                      frequency_db=RULERS.frequency.value, workers=workers)
    assert actual.passed and reference.passed
    assert actual.max_db.hex() == reference.max_db.hex()


@pytest.mark.parametrize("kind", ("sources", "receivers"))
def test_spot_check_rejects_missing_name(model: ReducedModel, operators: FullOperators, kind: str) -> None:
    if kind == "sources":
        operators = dataclasses.replace(operators, sources={})
    else:
        operators = dataclasses.replace(operators, receivers={})
    with pytest.raises(ValueError, match="喇叭或座位"):
        check.spot_check_reduced(model, operators, HOLDOUT[0], AXIS, ruler_name=RULERS.frequency.name,
                                 frequency_db=RULERS.frequency.value)
