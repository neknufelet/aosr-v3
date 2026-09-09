#!/usr/bin/env python3
"""斷言不准鎖死數量。

掃描面是 ``<scan_root>/tests/`` 底下所有進得了版控的 ``.py``（含子目錄、含測試用的
輔助模組——同一種病躲進 helper 一樣要咬）。真的用 :mod:`ast` 解析，不用正則：正則
分不出 ``assert len(x) == 3`` 跟文件字串裡舉的例子，也走不進 ``and`` 底下。

**什麼會紅**

斷言（``ast.Assert``）的條件式裡，只要有一組比較是「數量表達式」對「寫死的整數」：

* 數量表達式＝任何呼叫。``len(<任何東西>)`` 是最典型的一種，``count_rows()``、
  ``x.count("a")`` 也算——不是只有 ``len`` 會鎖死數量。
* 寫死的整數＝整數字面值（``3``、``-1``、``0``），或**同一支檔裡**被指定成整數字面值
  的名字（``n = 3`` 之後拿 ``n`` 比）。後者是找碴席點名的零成本繞法：數字換個地方寫，
  行為一模一樣。分界線在「數字是不是寫死在同一支檔裡」——``import`` 進來、對照別處
  登記的值算合規。
* 比較運算子是 ``==`` 或 ``!=``。兩邊哪邊放數字都算，``3 == len(x)`` 一樣咬。
* 位置不限最外層：``and``／``or``／``not``／三元／推導式的 if 底下一樣走進去
  （用 ``ast.walk`` 掃整個條件式），這是主線 ``tests/`` 當初真的長成的形狀。

``== 0`` 刻意不開例外。它不會懲罰改善，但跟其他數字共用同一個毛病：紅的時候只印得出
``1 != 0``，印不出那一筆是什麼。

**什麼不會紅（合規寫法，也是控制樣本裡放的四條）**

* ``assert names == {"a", "b"}``——逐項具名比對。右邊不是整數，不咬。
* ``assert collect() == REGISTERED``——對照別處登記、import 進來的值，不咬。
* ``assert len(names) == len(set(names))``——兩個 len 互比，右邊不是寫死的數字，不咬。
* ``assert len(x) >= 1``——不等式一律放行。下界不是「鎖死」，這是已知的洞，寫在卡面。

**為什麼**

v2 事故 ``exact-count-assertion-punishes-improvement``：舊 DIRGRAD CLI 要求 trace-only
drift「恰好兩筆」，有一筆被改善成 exact signature 之後閘就紅了。「只允許哪些」與
「必須仍有幾個」寫成同一條，改善會把閘弄紅。那筆事故的病灶在產品／守門用的 checker，
不在 ``tests/``，所以這張卡對它不算血債——理由寫在卡的 ``related_lessons_why``。

**沒掃到東西就回 2**

``tests/`` 底下一支 ``.py`` 都沒有的時候 raise :class:`ToolBroken`，讓外殼回 2。
零命中回 0 的前提是真的讀過檔；一支都沒讀到，「沒問題」這句話不算數。
解不開的 ``.py`` 也回 2——我沒看懂就不出結論。
"""
from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

from governance.exit_codes import ToolBroken, run

TESTS_DIR = "tests"

# 只咬等值／不等值。不等式（>= <= > <）一律放行，理由寫在卡面。
PINNING_OPS = (ast.Eq, ast.NotEq)

SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _callee(func: ast.expr) -> str:
    """把 ``x.count`` 這種呼叫對象攤成 ``"x.count"``。認不出來就回空字串。"""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _callee(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _int_literal(expr: ast.expr) -> int | None:
    """整數字面值就回那個值（``-1`` 這種一元負號也認）。布林不算——``True`` 不是數量。"""
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.USub, ast.UAdd)):
        inner = _int_literal(expr.operand)
        if inner is None:
            return None
        return -inner if isinstance(expr.op, ast.USub) else inner
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    return None


def _own_statements(scope: ast.AST) -> Iterator[ast.stmt]:
    """這一層自己的每一個 statement（走進 if／for／try，但不走進巢狀的 def／class）。"""
    stack: list[ast.stmt] = list(getattr(scope, "body", []))
    while stack:
        stmt = stack.pop()
        yield stmt
        if isinstance(stmt, SCOPE_NODES):
            continue
        for field in ("body", "orelse", "finalbody"):
            stack.extend(getattr(stmt, field, None) or [])
        for handler in getattr(stmt, "handlers", None) or []:
            stack.extend(handler.body)


def _pinned_names(scope: ast.AST) -> dict[str, int]:
    """這一層被指定成整數字面值的名字。

    只認「整支檔裡只被指定一次、而且指定的是整數字面值」的名字。被指定超過一次，
    或指定的是呼叫結果，都不算——寧可漏，不要拿猜的東西判人違規。
    """
    seen: dict[str, list[int | None]] = {}
    for stmt in _own_statements(scope):
        if isinstance(stmt, ast.Assign):
            targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
            value: ast.expr | None = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            targets = [stmt.target]
            value = stmt.value
        else:
            continue
        literal = _int_literal(value) if value is not None else None
        for target in targets:
            seen.setdefault(target.id, []).append(literal)
    return {name: values[0] for name, values in seen.items() if len(values) == 1 and values[0] is not None}


def _pinned_value(expr: ast.expr, pinned: dict[str, int]) -> tuple[int, str] | None:
    """這一邊是不是「寫死的整數」？回 (值, 怎麼寫死的)。"""
    literal = _int_literal(expr)
    if literal is not None:
        return literal, "整數字面值"
    if isinstance(expr, ast.Name) and expr.id in pinned:
        return pinned[expr.id], f"同一支檔裡 {expr.id} = {pinned[expr.id]}"
    return None


def _count_kind(expr: ast.expr) -> str | None:
    """這一邊是不是「數量表達式」？任何呼叫都算，``len(...)`` 另外標出來。"""
    if not isinstance(expr, ast.Call):
        return None
    return "len(...)" if _callee(expr.func) == "len" else f"呼叫 {_callee(expr.func) or '<運算出來的>'}()"


def _compare_problems(cmp: ast.Compare, rel: str, pinned: dict[str, int]) -> list[str]:
    bad: list[str] = []
    operands = [cmp.left, *cmp.comparators]
    for op, left, right in zip(cmp.ops, operands, operands[1:]):
        if not isinstance(op, PINNING_OPS):
            continue
        for side, other in ((left, right), (right, left)):
            kind = _count_kind(side)
            hit = _pinned_value(other, pinned)
            if kind is None or hit is None:
                continue
            value, how = hit
            bad.append(
                f"{rel}:{cmp.lineno} 斷言把數量鎖死：{ast.unparse(cmp)}"
                f"（{kind} 對死 {value}，{how}）"
                "——「必須仍有 N 筆」不是單調安全的性質，多抓到一筆真的東西就把閘弄紅，"
                "斷言變成在懲罰改善。改成逐項具名比對（assert names == {\"a\", \"b\"}），"
                "或對照別處登記、import 進來的值"
            )
            break
    return bad


def _scope_problems(scope: ast.AST, rel: str, inherited: dict[str, int]) -> list[str]:
    """一個作用域裡的斷言。名字的解析由外往內疊，內層蓋掉外層。"""
    pinned = {**inherited, **_pinned_names(scope)}
    bad: list[str] = []
    for stmt in _own_statements(scope):
        if isinstance(stmt, ast.Assert):
            for node in ast.walk(stmt.test):
                if isinstance(node, ast.Compare):
                    bad += _compare_problems(node, rel, pinned)
        elif isinstance(stmt, SCOPE_NODES):
            bad += _scope_problems(stmt, rel, pinned)
    return bad


def _parse(path: Path, rel: str) -> ast.Module:
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc
    try:
        return ast.parse(src, filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行解不開（{exc.msg}）——我沒看懂就不出結論") from exc


def check(scan_root: Path, files: list[Path]) -> list[str]:
    base = scan_root / TESTS_DIR
    targets = sorted(f for f in files if f.suffix == ".py" and f.is_relative_to(base))
    if not targets:
        raise ToolBroken(
            f"{scan_root}/{TESTS_DIR} 底下一支版控裡的 .py 都沒有"
            "——這一跑沒讀到任何測試檔，「沒問題」這句話不算數"
        )

    bad: list[str] = []
    for path in targets:
        rel = str(path.relative_to(scan_root))
        bad += _scope_problems(_parse(path, rel), rel, {})
    return sorted(bad)


if __name__ == "__main__":
    sys.exit(run(check, description="測試檔裡的斷言不准把數量鎖死"))
