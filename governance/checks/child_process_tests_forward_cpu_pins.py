#!/usr/bin/env python3
"""開子行程的考卷要帶三格處理器釘選：``child_process_env(...)`` 要直接包在共用小工具裡。

票 #1077。計算子行程的環境由 ``aosr.runtime.child_process_env`` 建，它只繼承白名單，雲端 job 層設好的
三格處理器釘選（numpy 的向量指令、OpenBLAS 的核心、XLA 的指令集）不會跟進子行程；抽到有 AVX-512 的
雲端機器時，子行程算出來的數字跟答案差最後一位。這一類至少九次、三次紅到主線（盤點見票上）。

規則（判準全從卡的 ``[settings]`` 讀，程式裡沒有預設值）：

- ``tests_dir`` 底下每一支 .py（含探針與輔助模組）裡，呼叫 ``child_env_name`` 的每一處（直接叫、
  ``模組.函式`` 叫、``from … import … as 別名`` 再叫，都算），都必須**直接**是共用小工具 ``helper_name``
  的第一個位置引數：``forward_cpu_pins(child_process_env(threads=1))``。包在別的東西裡（``dict(...)``、
  先指給名字再傳、字典展開）都不算——審的人要一眼看得出那一處帶了釘選。
- 立卡前就在的檔整批列在 ``[[settings.allow]]``：一筆一支檔（``path``）、凍結當時沒包的處數（``calls``，
  正整數）、理由與到期日（形狀與到期由 exemptions-need-expiry 那張卡判）。那支檔沒包的處數必須**等於**
  ``calls``：多了就是新寫的沒帶釘選，少了（改好了、或檔不在了）就把那一筆改小或拿掉——放行清單不准
  比實況寬。
- 共用小工具本身要在：``helper_file`` 在版控裡、而且最外層定義了 ``helper_name`` 這支函式。

**按檔凍結只比處數**：凍結的檔修掉一處、同時另寫一處沒包的，總數不變就看不出來；凍結檔要改動時順手包好。

**刻意沒管**：把函式本身當值傳出去（``factory=child_process_env``）、``getattr`` 取、在別的模組包一層再叫，
都看不到；另外定義一支同名的函式當成小工具也看不到；那個 diff 在合併請求上看得見。產品程式自己開的子行程（搜尋工作行程等）不在這張卡：正式工作
行程刻意不繼承釘選（``tests/engine/test_six_category_endpoint.py`` 那一題，#1051），要改是產品那一側的事。
小工具帶的是父行程「有設的」那幾格——本機三格是空的，包了也是原樣，這張卡守的是雲端那一跑。

**回 2**：讀不到卡的 settings、放行那一格形狀壞掉（缺 ``path``、``calls`` 不是正整數）、共用小工具不在或
沒有那支函式、``tests_dir`` 底下有 .py 剖不開。
"""
from __future__ import annotations

import ast
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from governance.exit_codes import ToolBroken, note, run
from governance.loader import setting_tables, setting_text

CARD_ID = "child-process-tests-forward-cpu-pins"
RULES_DIR = "governance/rules"
PYTHON_SUFFIX = ".py"
TEXT_KEYS = ("tests_dir", "child_env_name", "helper_file", "helper_name")
ALLOW_KEY = "allow"


class Frozen(NamedTuple):
    """一筆放行：哪支檔、凍結當時沒包的處數。"""

    path: str
    calls: int


class Settings(NamedTuple):
    tests_dir: str
    child_env_name: str
    helper_file: str
    helper_name: str
    frozen: tuple[Frozen, ...]


def _card_path(scan_root: Path, files: list[Path]) -> Path:
    path = scan_root / RULES_DIR / f"{CARD_ID}.toml"
    if path not in files:
        raise ToolBroken(f"樹裡沒有這張卡（{RULES_DIR}/{CARD_ID}.toml）——判準只住卡上，沒有卡就不出結論")
    return path


def _frozen(entries: list[dict[str, object]]) -> tuple[Frozen, ...]:
    out: list[Frozen] = []
    for index, entry in enumerate(entries, start=1):
        path = entry.get("path")
        calls = entry.get("calls")
        if not isinstance(path, str) or not path.strip():
            raise ToolBroken(f"[[settings.allow]] 第 {index} 筆沒有 path（哪一支檔）——放行指不到東西就不出結論")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
            raise ToolBroken(
                f"[[settings.allow]] 第 {index} 筆（{path}）的 calls 必須是正整數（凍結當時沒包的處數），實際是 {calls!r}"
            )
        out.append(Frozen(path, calls))
    return tuple(out)


def read_settings(scan_root: Path, files: list[Path]) -> Settings:
    path = _card_path(scan_root, files)
    try:
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"{path.relative_to(scan_root)} 剖不開：{exc}") from exc
    fields = data.get("settings")
    if not isinstance(fields, dict):
        raise ToolBroken(f"{CARD_ID} 沒有 [settings]——判準只住卡上，讀不到就不出結論")
    texts = {key: setting_text(fields, key) for key in TEXT_KEYS}
    return Settings(**texts, frozen=_frozen(setting_tables(fields, ALLOW_KEY)))


def _rel(scan_root: Path, path: Path) -> str:
    return path.relative_to(scan_root).as_posix()


def _test_python(scan_root: Path, files: list[Path], settings: Settings) -> list[Path]:
    prefix = settings.tests_dir.rstrip("/") + "/"
    return [f for f in files if f.suffix == PYTHON_SUFFIX and _rel(scan_root, f).startswith(prefix)]


def _parse(scan_root: Path, path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_bytes().decode("utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as exc:
        raise ToolBroken(f"{_rel(scan_root, path)} 剖不開：{exc}") from exc


def _aliases(tree: ast.Module, name: str) -> set[str]:
    """這支檔裡哪些名字指的是 ``name``：原名加上 ``from … import name as 別名`` 的別名。"""
    names = {name}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname for alias in node.names if alias.name == name and alias.asname)
    return names


def _called(node: ast.Call, names: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in names
    return isinstance(func, ast.Attribute) and func.attr in names


def unwrapped_calls(tree: ast.Module, settings: Settings) -> list[int]:
    """沒有直接包在共用小工具裡的那幾處呼叫的行號。"""
    child_names = _aliases(tree, settings.child_env_name)
    helper_names = _aliases(tree, settings.helper_name)
    wrapped: set[int] = set()
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called(node, child_names):
            calls.append(node)
        if _called(node, helper_names) and node.args and isinstance(node.args[0], ast.Call):
            wrapped.add(id(node.args[0]))
    return sorted(call.lineno for call in calls if id(call) not in wrapped)


def _helper_problems(scan_root: Path, files: list[Path], settings: Settings) -> None:
    path = scan_root / settings.helper_file
    if path not in files:
        raise ToolBroken(f"共用小工具 {settings.helper_file} 不在這棵樹的版控裡——規矩要大家包的東西不在，就不出結論")
    tree = _parse(scan_root, path)
    if not any(isinstance(n, ast.FunctionDef) and n.name == settings.helper_name for n in tree.body):
        raise ToolBroken(f"{settings.helper_file} 的最外層沒有定義 {settings.helper_name}——規矩要大家包的那支函式不在，就不出結論")


def _counts(scan_root: Path, files: list[Path], settings: Settings) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for path in _test_python(scan_root, files, settings):
        lines = unwrapped_calls(_parse(scan_root, path), settings)
        if lines:
            out[_rel(scan_root, path)] = lines
    return out


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = read_settings(scan_root, files)
    _helper_problems(scan_root, files, settings)
    found = _counts(scan_root, files, settings)
    frozen = {entry.path: entry.calls for entry in settings.frozen}
    bad: list[str] = []
    for path, times in sorted(Counter(entry.path for entry in settings.frozen).items()):
        if times > 1:
            bad.append(f"[[settings.allow]] 裡 {path} 出現 {times} 筆——一支檔只准一筆，處數寫在 calls")
    wrap = f"{settings.helper_name}({settings.child_env_name}(...))"
    for path, lines in sorted(found.items()):
        where = "、".join(f"第 {n} 行" for n in lines)
        if path not in frozen:
            bad.append(
                f"{path} {where} 呼叫 {settings.child_env_name}(...) 沒有直接包在 {settings.helper_name} 裡"
                f"（{settings.helper_file}）：雲端 job 層的三格處理器釘選不會跟進子行程，抽到 AVX-512 的機器就差最後一位。"
                f"寫成 {wrap}；子行程刻意要跟不帶釘選的正式工作行程同一環境的，才在卡上開一筆放行寫理由"
            )
        elif len(lines) != frozen[path]:
            bad.append(
                f"{path} 沒包的呼叫現在 {len(lines)} 處（{where}），放行凍結的是 {frozen[path]} 處："
                f"多了就是新寫的沒帶釘選（寫成 {wrap}），少了就把那一筆的 calls 改成實況"
            )
    for path in sorted(set(frozen) - set(found)):
        bad.append(f"[[settings.allow]] 的 {path} 已經沒有沒包的呼叫（或檔不在了）——拿掉這一筆，放行清單不准比實況寬")
    note(
        f"考卷 .py {len(_test_python(scan_root, files, settings))} 支，有沒包呼叫的 {len(found)} 支"
        f"（{sum(len(v) for v in found.values())} 處），放行 {len(settings.frozen)} 筆"
    )
    return bad


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    settings = read_settings(scan_root, files)
    picked = set(_test_python(scan_root, files, settings))
    picked.update(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")
    return sorted(picked)


if __name__ == "__main__":
    sys.exit(run(check, description="開子行程的考卷要帶三格處理器釘選：child_process_env(...) 要直接包在共用小工具裡", targets=targets))
