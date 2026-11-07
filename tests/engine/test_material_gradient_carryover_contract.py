"""沿用記錄必須完整；抽查使用下降真正的投影起點，不套終點匯出規則。"""
from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from aosr.config.material_gradient import load_material_gradient_settings
from aosr.materials.construction_models import ConstantImpedance
from aosr.geometry.shoebox import Wall
from aosr.config.paths import config_path
from aosr.gradient import cli, run
from aosr.physics import fem_reduced_check as check
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_check import SurrogateSpotCheck
from tests.engine.test_material_gradient_cli import _schemes, _env
from tests.engine.test_material_gradient_mount import _project

RULER = run.holdout_rulers().frequency
SPOT = SurrogateSpotCheck((20.0, 40.0), 0.0, RULER.name, RULER.value, True, 1.0)
CARRYOVER = {"surrogate_physics_identity": "old", "operators_digest": "a" * 64,
             "surrogate_spot_check": asdict(SPOT)}


@pytest.mark.parametrize("missing", tuple(CARRYOVER))
def test_project_rejects_partial_carryover_record(missing: str) -> None:
    document = {**_project().model_dump(mode="json"), **CARRYOVER}
    document.pop(missing)
    with pytest.raises(ValueError, match="沿用"):
        run.GradientProject.model_validate(document)


@pytest.mark.parametrize("damage", ("failed", "zero_seconds", "negative_error", "same_identity", "wrong_ruler", "empty_axis"))
def test_project_rejects_invalid_carryover_evidence(damage: str) -> None:
    spot = asdict(SPOT)
    document = {**_project().model_dump(mode="json"), **CARRYOVER, "surrogate_spot_check": spot}
    if damage == "failed":
        spot["passed"] = False
    elif damage == "zero_seconds":
        spot["seconds"] = 0.0
    elif damage == "negative_error":
        spot["max_db"] = -1.0
    elif damage == "same_identity":
        document["surrogate_physics_identity"] = document["physics_identity"]
    elif damage == "wrong_ruler":
        spot["ruler_name"] = "fem_reduced_holdout_band_db"
    else:
        spot["frequencies_hz"] = []
    with pytest.raises(ValueError, match="沿用"):
        run.GradientProject.model_validate(document)


def test_spot_check_uses_projected_start_without_export_rules(tmp_path: Path) -> None:
    done = subprocess.run([sys.executable, "-m", "tests.engine._material_gradient_carryover_probe", "projected", str(tmp_path)],
                          env=_env(tmp_path), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    document = json.loads((tmp_path / "projected.json").read_text())
    actual = np.asarray(document["actual_real"]) + 1j * np.asarray(document["actual_imag"])
    expected = np.asarray(document["reference_real"]) + 1j * np.asarray(document["reference_imag"])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
