"""精度契約登記簿的載入、雙格一致與變異考卷接線。"""
from __future__ import annotations

import os
import subprocess
import sys
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.precision_contracts import load_precision_contracts
from tests.conftest import GitSandbox


_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO / "blueprint" / "precision_contracts.toml"


def _write_registry(path: Path, *, value: str, display: str) -> None:
    path.write_text(
        "[[contract]]\n"
        'name = "example"\n'
        f"value = {value}\n"
        f'display = "{display}"\n'
        'unit = "relative"\n'
        'decision_paper = "example.md"\n'
        'truth = "hand-derived example"\n'
        'mutant_test = "tests/engine/test_late_energy_contract.py::test_mutant_beyond_tolerance_is_red"\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("value", "display"),
    (("9.5367431640625e-07", "2^-20"), ("2.0e-4", "2·1e-4")),
)
def test_display_and_value_are_the_same_number(
    tmp_path: Path,
    value: str,
    display: str,
) -> None:
    path = tmp_path / "contracts.toml"
    _write_registry(path, value=value, display=display)

    assert load_precision_contracts(path)["example"].value == float(value)


def test_display_drift_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "contracts.toml"
    _write_registry(path, value="9.5367431640625e-07", display="2^-19")

    with pytest.raises(ValidationError, match="display.*value"):
        load_precision_contracts(path)


def test_duplicate_contract_name_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "contracts.toml"
    _write_registry(path, value="2.0e-4", display="2·1e-4")
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="同名.*example"):
        load_precision_contracts(path)


def test_contract_extra_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "contracts.toml"
    _write_registry(path, value="2.0e-4", display="2·1e-4")
    path.write_text(path.read_text(encoding="utf-8") + "extra = true\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="extra"):
        load_precision_contracts(path)


def test_registry_path_has_no_default() -> None:
    parameter = inspect.signature(load_precision_contracts).parameters["path"]

    assert parameter.default is inspect.Parameter.empty


def test_registry_has_the_named_contracts() -> None:
    contracts = load_precision_contracts(_REGISTRY)

    assert set(contracts) == {
        "late_energy_vs_legacy",
        "late_decay_t20_vs_legacy",
        "late_decay_t30_property",
        "fem_rigid_modal_vs_analytic",
        "fem_vs_fenics_frozen",
        "direct_energy_vs_legacy",
        "reflected_energy_floor",
        "reflection_product_ulp",
        "catalog_absorption_property",
    }


def test_every_registered_mutant_node_collects(git_sandbox: GitSandbox) -> None:
    contracts = load_precision_contracts(_REGISTRY)
    nodes = tuple(contracts[name].mutant_test for name in contracts)
    absolute_nodes = []
    for node in nodes:
        test_path, separator, test_name = node.partition("::")
        assert separator and test_name
        absolute_nodes.append(f"{_REPO / test_path}::{test_name}")
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": os.pathsep.join((str(_REPO / "src"), str(_REPO))),
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            *absolute_nodes,
        ],
        cwd=git_sandbox.root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
