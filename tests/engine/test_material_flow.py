"""已驗材料的唯一出生地與真正中央處理器、雙精度下的下游接線。"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aosr.materials.construction_models import ConstantImpedance
from aosr.runtime import child_process_env

CONSTANT_FIELDS = frozenset(ConstantImpedance.model_fields) - {"kind"}


def _skipped_nodes(tree: ast.AST) -> set[int]:
    """型別註記與 isinstance 第二個引數只是在讀型別，不建材料；其餘提到固定材料型別的地方都算出生。"""
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance" \
                and len(node.args) == 2:
            roots.append(node.args[1])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            roots += [arg.annotation for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                                                 node.args.vararg, node.args.kwarg) if arg and arg.annotation]
            roots += [node.returns] if node.returns else []
        elif isinstance(node, ast.AnnAssign):
            roots.append(node.annotation)
    return {id(child) for root in roots for child in ast.walk(root)}


def _rebuilds_constant(node: ast.AST, names: set[str]) -> bool:
    """直接呼叫、別名、模組屬性、partial、model_validate／model_construct、帶阻抗欄位的 model_copy。"""
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return node.attr in names
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "model_copy"
            and any(keyword.arg == "update" and isinstance(keyword.value, ast.Dict)
                    and any(isinstance(key, ast.Constant) and key.value in CONSTANT_FIELDS
                            for key in keyword.value.keys) for keyword in node.keywords))


def _birthplaces(source: str, label: str) -> set[tuple[str, str]]:
    tree = ast.parse(source)
    names = {"ConstantImpedance"} | {alias.asname or alias.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) for alias in node.names if alias.name == "ConstantImpedance"}
    owners = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    skipped = _skipped_nodes(tree)
    places = set()
    for node in ast.walk(tree):
        if id(node) in skipped or not _rebuilds_constant(node, names):
            continue
        owner: ast.AST = node
        scope = []
        while owner in owners:
            owner = owners[owner]
            if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope.append(owner.name)
        places.add((label, ".".join(reversed(scope)) or "<module>"))
    return places


def test_constant_material_has_one_birthplace() -> None:
    """下游再建 ConstantImpedance 即紅；比較位置集合，不比較呼叫數量。"""
    root = Path(__file__).resolve().parents[2] / "src" / "aosr"
    actual = set().union(*(_birthplaces(path.read_text(encoding="utf-8"), path.relative_to(root.parent.parent).as_posix())
                          for path in root.rglob("*.py") if path.relative_to(root).parts[0] != "materials"))
    assert actual == {("src/aosr/physics/report_io.py", "solver_inputs")}


_IMPORT = "from aosr.materials.construction_models import ConstantImpedance\n"


@pytest.mark.parametrize("source", (
    _IMPORT + "def f():\n    return ConstantImpedance(impedance_pa_s_m=1.)\n",
    "from aosr.materials.construction_models import ConstantImpedance as C\ndef f():\n    return C(impedance_pa_s_m=1.)\n",
    "from aosr.materials import construction_models as m\ndef f():\n    return m.ConstantImpedance(impedance_pa_s_m=1.)\n",
    _IMPORT + "from functools import partial\ndef f():\n    return partial(ConstantImpedance, impedance_pa_s_m=1.)()\n",
    _IMPORT + "def f():\n    return ConstantImpedance.model_validate({'impedance_pa_s_m': 1.})\n",
    _IMPORT + "def f():\n    return ConstantImpedance.model_construct(impedance_pa_s_m=1.)\n",
    _IMPORT + "def f():\n    make = ConstantImpedance\n    return make(impedance_pa_s_m=1.)\n",
    "def f(material):\n    return material.model_copy(update={'impedance_pa_s_m': 2.})\n",
), ids=("call", "import-alias", "module-attribute", "partial", "model-validate", "model-construct", "local-alias",
        "copy-with-impedance"))
def test_birthplace_scan_sees_every_rebuild(source: str) -> None:
    """常見的重建寫法都要落在所在函式；漏掉一種，下游就能繞過唯一出生地。"""
    assert _birthplaces(source, "x.py") == {("x.py", "f")}


@pytest.mark.parametrize("source", (
    _IMPORT + "def f(m):\n    return isinstance(m, ConstantImpedance)\n",
    _IMPORT + "def f(m):\n    return isinstance(m, (int, ConstantImpedance))\n",
    _IMPORT + "def f(m: ConstantImpedance) -> ConstantImpedance:\n    found: ConstantImpedance = m\n    return found\n",
    "def f(view):\n    return view.model_copy(update={'status': 'done'})\n",
), ids=("isinstance", "isinstance-tuple", "annotations", "copy-other-field"))
def test_birthplace_scan_ignores_type_reads(source: str) -> None:
    """只讀型別或複製別的模型不算出生；這一組守住上面那一題不是靠全部都算。"""
    assert _birthplaces(source, "x.py") == set()


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
