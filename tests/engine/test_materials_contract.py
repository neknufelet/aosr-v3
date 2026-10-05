"""Materials 建好之後不准原地改兩張牆面表（#608 邊界核對）：constant_impedances 是快取，原地改了反射乘積會沿用舊阻抗。

正式流程不這樣用（三個建立點都當場新建字典、只在同一支函式裡用）；這裡掃全部產品程式，任何地方原地改
``walls``／``wall_grids`` 就紅，並考「要換材料就建新的」那條路拿到的是新的快取。真的改成唯讀映射會換物理身分，
跟下一次換物理身分的改動同批做。
"""
import ast
import dataclasses
from pathlib import Path

from aosr.physics.amplitude import Materials

SRC = Path(__file__).resolve().parents[2] / "src" / "aosr"
GUARDED = {"walls", "wall_grids"}
MUTATORS = {"update", "pop", "popitem", "clear", "setdefault", "__setitem__", "__delitem__"}


def _guarded(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr in GUARDED


def _mutations(tree: ast.Module) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        lines.extend(target.lineno for target in targets
                     if isinstance(target, ast.Subscript) and _guarded(target.value))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS
                and _guarded(node.func.value)):
            lines.append(node.lineno)
    return lines


def test_no_product_code_mutates_material_tables_in_place() -> None:
    files = sorted(SRC.rglob("*.py"))
    assert files
    found = {str(path.relative_to(SRC)): lines for path in files
             if (lines := _mutations(ast.parse(path.read_text(encoding="utf-8"))))}
    assert found == {}


def test_the_guard_sees_in_place_mutations() -> None:
    """掃描本身要咬得到：直接寫幾種原地改法，每一種都要被認出來。"""
    sample = ("m.walls['ceiling'] = (1j,)\nm.wall_grids['floor'] = g\ndel m.walls['x']\nm.walls.update(other)\n"
              "m.wall_grids.pop('x')\nm.walls['x'] += (1,)\n")
    assert sorted(_mutations(ast.parse(sample))) == [1, 2, 3, 4, 5, 6]


def test_a_new_materials_gets_a_fresh_cache() -> None:
    """要換材料就建新的：dataclasses.replace 的新物件重算 constant_impedances，不沿用舊快取。"""
    first = Materials(rho_c=400.0, frequencies_hz=(100.0, 200.0), walls={"ceiling": (1000 + 0j, 1000 + 0j)})
    assert first.constant_impedances == {"ceiling": 1000 + 0j}
    second = dataclasses.replace(first, walls={"ceiling": (2000 + 0j, 2000 + 0j)}, wall_grids=None)
    assert second.constant_impedances == {"ceiling": 2000 + 0j}
    assert first.constant_impedances == {"ceiling": 1000 + 0j}
