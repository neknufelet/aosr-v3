"""已驗材料的唯一出生地與真正中央處理器、雙精度下的下游接線。"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aosr.runtime import child_process_env


def _birthplaces(path: Path, root: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constructors = {"ConstantImpedance"} | {alias.asname or alias.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) for alias in node.names if alias.name == "ConstantImpedance"}
    owners = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    places = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if name not in constructors:
            continue
        owner: ast.AST = node
        scope = []
        while owner in owners:
            owner = owners[owner]
            if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope.append(owner.name)
        places.add((path.relative_to(root.parent.parent).as_posix(), ".".join(reversed(scope)) or "<module>"))
    return places


def test_constant_material_has_one_birthplace() -> None:
    """下游再建 ConstantImpedance 即紅；比較位置集合，不比較呼叫數量。"""
    root = Path(__file__).resolve().parents[2] / "src" / "aosr"
    actual = set().union(*(_birthplaces(path, root) for path in root.rglob("*.py")
                          if path.relative_to(root).parts[0] != "materials"))
    assert actual == {("src/aosr/physics/report_io.py", "solver_inputs")}


@pytest.mark.parametrize("mode", ("fem", "late", "geometry", "slices", "screen", "table", "window", "capability"))
def test_construction_reaches_downstream(tmp_path: Path, mode: str) -> None:
    """各下游忽略傳入材料、取錯頻率或冒稱固定能力列都必須紅。"""
    _child(tmp_path, mode)


@pytest.mark.parametrize("mode", ("stage", "pair-window", "cli"))
def test_formal_callers_keep_materials(tmp_path: Path, mode: str) -> None:
    """正式方案與評估呼叫者漏傳、換回固定材料必須紅。"""
    _child(tmp_path, mode)


def test_fixed_cli_matches_main_answer(tmp_path: Path) -> None:
    """整份命令列固定材料報表逐位對主線實跑的摘要；用既有替身避開機器解法器差異。"""
    _child(tmp_path, "constant")


def _child(tmp_path: Path, mode: str) -> None:
    env = child_process_env(threads=1)
    pins = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")
    env.update({name: os.environ[name] for name in pins if name in os.environ})
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(root)))
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(tmp_path), "1"
    probe = Path(__file__).with_name("_material_flow_probe.py")
    result = subprocess.run([sys.executable, str(probe), mode, str(tmp_path)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
