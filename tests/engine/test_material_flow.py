"""已驗材料的唯一出生地與真正中央處理器、雙精度下的下游接線。

固定命令列控制組在 HEAD 5339182ca3fd1831a684a19ab5c03462eb5b7e1f 錄製：
工作區 785-tolerance-step5/evidence（repo 外）的錄製腳本；舊雜湊同跑核過，共用 _material_flow_answers.py。
子行程把原始標準輸出寫 constant.stdout.json，父行程取原文骨架與逐格比，量測外掛可見。
機器相關用①級：top.f_s_hz／eyring_t60_by_band_s 經 catalog_absorption.complex_random_incidence_absorption
的 Paris 積分 leggauss 特徵值、crossover.eyring_t60_by_band 的 math.log；Schroeder 承接它。
points／bands 的 direct／reflected／interference／geometric_energy 及 path_table.relative_direct_energy
承接 amplitude.path_pressure 的 cmath.exp；方向角經 report_path_table._direction 的 atan2／hypot。
points.fem_energy 與 bands.fem_energy 經 fake_fem_energy 的 sin；late_energy 經 fake_late_energy 的 cos／sin，
bands.late_energy 另經 frequency_axis.octave_cells_in_range 的 log2 加權。
points.scattering／path_table.scattering_coefficient 經 geometric_lane._scattering_at 的複數取模／次方。
points／bands 的 w_fem／w_geo 經 crossover._soft_geometric_weight 的 log2；bands 另經八度 log2 加權；
points／bands 的 total_energy、bands.f_s_hz／fem_contribution／geometric_contribution 承接以上計算。
上述絕對 0，唯 interference_energy 會相消，絕對尺取同列 direct_energy＋reflected_energy 的答案 × ①級。
late_energy 是 _late_share_energy 的非負散射份＋正尾段，沒有尾段相減；不用相消尺。沒有真的晚期線性方程組，非②級。
留骨架：scene.source_m／receiver_m、frequencies_hz／frequency_hz／center_frequency_hz、capability.frequency_hz，
交接上下限、room_volume_m3；path_table.delay_s／distance_m／direction_vector 經四則／sqrt。
bands.t20_s／t30_s 的 fast_late_decay 只用四則與 sqrt 選軸，再由 _band_decay_values 四則平均，也留骨架。
頻率軸雖源自非整數次方，依決策紙的鍵例外仍逐位；不能據此把使用頻率作 sin／log2 的能量也留骨架。
CLI 骨架無完整位數的數字字串；eyring 字典鍵 500.0、1000.0 是頻率軸，保留逐字。
字串、整數、換行、縮排與鍵順序逐字；只證明機器相關浮點在①級內，不再證明整份跨處理器逐位。
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aosr.materials.construction_models import ConstantImpedance
from aosr.runtime import child_process_env
from tests.engine._json_control import assert_json_control
from tests.engine._material_flow_answers import ANSWERS

CONSTANT_FIELDS = frozenset(ConstantImpedance.model_fields) - {"kind"}


def _skipped_nodes(tree: ast.AST) -> set[int]:
    """型別註記、isinstance／issubclass 第二個引數、cast 第一個引數、型別比較與 type 陳述式只是在讀型別，不建材料。"""
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and len(node.args) == 2 and _called(node) in {"isinstance", "issubclass"}:
            roots.append(node.args[1])
        elif isinstance(node, ast.Call) and node.args and _called(node) == "cast":
            roots.append(node.args[0])
        elif isinstance(node, ast.Compare) and all(isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq))
                                                   for op in node.ops):
            roots += [side for side in (node.left, *node.comparators) if isinstance(side, (ast.Name, ast.Attribute))]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            roots += [arg.annotation for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                                                 node.args.vararg, node.args.kwarg) if arg and arg.annotation]
            roots += [node.returns] if node.returns else []
        elif isinstance(node, ast.AnnAssign):
            roots.append(node.annotation)
        elif isinstance(node, ast.TypeAlias):
            roots.append(node.value)
    return set().union(*(_reads_without_calls(root) for root in roots))


def _reads_without_calls(root: ast.AST) -> set[int]:
    """只排除到呼叫為止：`ConstantImpedance(...).kind == m.kind`、`issubclass(t, type(ConstantImpedance(...)))` 裡的建構照掃。"""
    found: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if not isinstance(node, ast.Call):
            found.add(id(node))
            stack.extend(ast.iter_child_nodes(node))
    return found


def _called(node: ast.Call) -> str:
    return node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""


def _rebuilds_constant(node: ast.AST, names: set[str]) -> bool:
    """提到固定材料型別（直接呼叫、別名、模組屬性、partial、model_validate／model_construct），或寫出它的物理欄位名
    （關鍵字引數、字典鍵、model_copy 的 update，不論字典是當場寫、先存變數還是用 dict(...) 建）。"""
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return node.attr in names
    if isinstance(node, ast.keyword):
        return node.arg in CONSTANT_FIELDS
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in CONSTANT_FIELDS


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
    "def f(material):\n    update = {'impedance_pa_s_m': 2.}\n    return material.model_copy(update=update)\n",
    "def f(material):\n    return material.model_copy(update=dict(impedance_pa_s_m=2.))\n",
    _IMPORT + "def f(m):\n    return m == ConstantImpedance.model_validate({'kind': 'constant'})\n",
    _IMPORT + "def f(m):\n    return ConstantImpedance(impedance_pa_s_m=2.).kind == m.kind\n",
    _IMPORT + "def f(m):\n    return issubclass(type(m), type(ConstantImpedance(impedance_pa_s_m=2.)))\n",
), ids=("call", "import-alias", "module-attribute", "partial", "model-validate", "model-construct", "local-alias",
        "copy-with-impedance", "copy-with-stored-dict", "copy-with-dict-call", "compare-with-new",
        "compare-attribute-of-new", "issubclass-of-new"))
def test_birthplace_scan_sees_every_rebuild(source: str) -> None:
    """常見的重建寫法都要落在所在函式；漏掉一種，下游就能繞過唯一出生地。"""
    assert _birthplaces(source, "x.py") == {("x.py", "f")}


@pytest.mark.parametrize("source", (
    _IMPORT + "def f(m):\n    return isinstance(m, ConstantImpedance)\n",
    _IMPORT + "def f(m):\n    return isinstance(m, (int, ConstantImpedance))\n",
    _IMPORT + "def f(m: ConstantImpedance) -> ConstantImpedance:\n    found: ConstantImpedance = m\n    return found\n",
    "def f(view):\n    return view.model_copy(update={'status': 'done'})\n",
    _IMPORT + "def f(m):\n    return type(m) is ConstantImpedance\n",
    _IMPORT + "def f(cls):\n    return issubclass(cls, ConstantImpedance)\n",
    _IMPORT + "from typing import cast\ndef f(m):\n    return cast(ConstantImpedance, m)\n",
    _IMPORT + "type Fixed = ConstantImpedance\ndef f(m):\n    return m.impedance_pa_s_m\n",
), ids=("isinstance", "isinstance-tuple", "annotations", "copy-other-field", "type-identity", "issubclass", "cast",
        "type-statement"))
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
    """原文骨架逐字、機器相關欄①級；用既有替身，裁判在父行程。"""
    assert_json_control(_child(tmp_path, "constant"), "cli", ANSWERS["cli"])


def _child(tmp_path: Path, mode: str) -> str:
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
    if mode in {"constant", "constant-zones"}:
        assert (tmp_path / "aosr-source.txt").read_text(encoding="utf-8") == str(root / "src" / "aosr" / "__init__.py")
        return (tmp_path / "constant.stdout.json").read_text(encoding="utf-8")
    return result.stdout
