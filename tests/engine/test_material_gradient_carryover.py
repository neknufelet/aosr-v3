"""替身沿用的開始、接續與終點接線；開始截在建搜尋之前。"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.gradient import cli, run, endpoint_run
from aosr.physics import fem_reduced_store as store
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_check import ValidationReport
from aosr.physics.fem_reduced_receipt import save_validation_receipt
from aosr.reporting import fem_reduced_scheme
from tests.engine._six_search_cli_case import operators as operators, model as model
from tests.engine.test_material_gradient_cli import (
    PRESET_TEXT, _chain_operators, _good_report, _schemes, _env,
)
from tests.engine.test_material_gradient_mount import _project
from tests.engine.test_material_gradient import PURPOSE, _chain_model

CURRENT = "phys-v1:" + "1" * 64


@pytest.fixture(scope="module")
def chain_data(request: pytest.FixtureRequest, model: ReducedModel, operators: FullOperators
               ) -> tuple[ReducedModel, FullOperators, ValidationReport]:
    chain = _chain_model(model)
    full = _chain_operators(operators, chain)
    return chain, full, _good_report(request, chain)


@pytest.mark.parametrize("carryover", (False, True))
def test_start_records_carryover_only_when_identity_changes(
        chain_data: tuple[ReducedModel, FullOperators, ValidationReport], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], carryover: bool) -> None:
    chain, full, report = chain_data
    scheme = _schemes()["scheme.json"]
    scheme_path, surrogate = tmp_path / "scheme.json", tmp_path / "rom.npz"
    scheme_path.write_text(scheme.model_dump_json())
    store.save_reduced(chain, surrogate)
    save_validation_receipt(surrogate, report, rulers=run.holdout_rulers(), seconds=1.0)
    if carryover:
        store.save_operators_stamp(surrogate, full, rulers=run.holdout_rulers())
    done = subprocess.run([sys.executable, "-m", "tests.engine._material_gradient_carryover_probe",
                           "carryover" if carryover else "same", str(tmp_path)],
                          env=_env(tmp_path), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    project = run.read_project(tmp_path)
    written = (tmp_path / run.PROJECT_FILE).read_bytes()
    document = json.loads(written)
    stderr = done.stderr
    if carryover:
        assert "替身身分換過、完整模型逐位相同、抽查 8 點最大差" in stderr and "沿用" in stderr
        assert document["surrogate_physics_identity"] == chain.physics_identity
        assert document["operators_digest"] == store.load_operators_stamp(surrogate)["operators_digest"]
        assert document["surrogate_spot_check"]["passed"] is True
        assert len(document["surrogate_spot_check"]["frequencies_hz"]) == 8
    else:
        assert "沿用" not in stderr
        assert not {"surrogate_physics_identity", "operators_digest", "surrogate_spot_check"} & document.keys()
        assert written == _legacy_project_bytes(project)


def _legacy_project_bytes(project: run.GradientProject) -> bytes:
    """起點版本的固定欄位與順序；新增欄位若漏 exclude_if，逐位比較就紅。"""
    keys = ("format_version", "gradient_id", "scheme_path", "scheme_sha256", "surrogate_path",
            "surrogate_manifest_sha256", "receipt_path", "receipt_sha256", "presets", "purpose", "settings_path",
            "settings_fingerprint", "capabilities_path", "physics_identity", "mesh_hash", "source_points",
            "receiver_points", "engine_commit", "started_at", "objective_settings_fingerprint")
    document = {key: getattr(project, key) for key in keys}
    return (json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n").encode()


def test_endpoint_rejects_changed_full_operators(operators: FullOperators) -> None:
    project = _project("rigid").model_copy(update={"operators_digest": "f" * 64})
    with pytest.raises(ValueError, match="完整模型指紋.*不同"):
        endpoint_run.check_operators_digest(project, operators)


def test_stamp_command_writes_bound_file(chain_data: tuple[ReducedModel, FullOperators, ValidationReport],
                                        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain, full, report = chain_data
    scheme_path, surrogate = tmp_path / "scheme.json", tmp_path / "rom.npz"
    scheme_path.write_text(_schemes()["scheme.json"].model_dump_json())
    store.save_reduced(chain, surrogate)
    save_validation_receipt(surrogate, report, rulers=run.holdout_rulers(), seconds=1.0)
    monkeypatch.setattr(fem_reduced_scheme, "export_operators", lambda *args, **kwargs: full)
    assert cli.main(["stamp-surrogate", "--scheme", str(scheme_path), "--surrogate", str(surrogate)]) == 0
    assert store.load_operators_stamp(surrogate)["physics_identity"] == chain.physics_identity
