"""共用靜態 import 閉包；只讀原始碼，不載入產品設定或評分。"""
from __future__ import annotations

import ast
import hashlib
import sys
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

@dataclass(frozen=True)
class ImportReference:
    """一條本地 import 的來源、目標與是否位於函式內，供載入邊界考卷核對。"""

    source: str
    module: str
    in_function: bool


@dataclass(frozen=True)
class PhysicsImportClosure:
    """依模組名排序的本地閉包、import 位置、資料名與第三方頂層名。"""

    modules: tuple[str, ...]
    imports: tuple[ImportReference, ...]
    data_files: tuple[str, ...]
    third_party: tuple[str, ...]


def _source_path(root: Path, module: str) -> Path:
    """只解析本地單檔與有原始碼的套件；找不到就丟例外。"""
    parts = module.split(".")
    if parts[0] != "aosr":
        raise ValueError(f"不是本地模組：{module}")
    base = root.joinpath(*parts[1:])
    package = (base / "__init__").with_suffix(".py")
    if package.is_file():
        return package
    source = base.with_suffix(".py")
    if module != "aosr" and source.is_file():
        return source
    raise FileNotFoundError(f"找不到模組原始碼：{module}")


def _is_module(root: Path, name: str) -> bool:
    try:
        _source_path(root, name)
    except FileNotFoundError:
        return False
    return True


def _import_targets(root: Path, source: str, node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
    """解析絕對／相對 import；from 的名字若也是模組，連那支一起追。"""
    if isinstance(node, ast.Import):
        return tuple(item.name for item in node.names)
    package = source if _source_path(root, source).stem == "__init__" else source.rpartition(".")[0]
    base = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
    if base is None:
        raise ValueError(f"無法解析 import：{source}")
    targets = [base]
    if base == "aosr" or base.startswith("aosr."):
        targets.extend(f"{base}.{item.name}" for item in node.names
                       if item.name != "*" and _is_module(root, f"{base}.{item.name}"))
    return tuple(targets)


def _in_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
    return False


def _config_literals(tree: ast.Module, source: str, package: str) -> set[str]:
    """收 config_path 的字面值；動態參數、自己拿 CONFIG_DIR 組路徑都無法誠實量資料，直接拒收。"""
    names = {"config_path"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = (resolve_name("." * node.level + (node.module or ""), package)
                      if node.level else node.module)
            if module == "aosr.config.paths":
                names.update(item.asname or item.name for item in node.names
                             if item.name == "config_path")
        if source != "aosr.config.paths" and (
                isinstance(node, ast.Name) and node.id == "CONFIG_DIR"
                or isinstance(node, ast.Attribute) and node.attr == "CONFIG_DIR"
                or isinstance(node, ast.alias) and node.name == "CONFIG_DIR"):
            raise ValueError(f"{source} 自己拿 CONFIG_DIR 組路徑：資料檔要經 config_path 字面值才量得到")
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = (isinstance(node.func, ast.Name) and node.func.id in names
                  or isinstance(node.func, ast.Attribute) and node.func.attr == "config_path")
        if not called:
            continue
        arguments = node.args + [item.value for item in node.keywords if item.arg == "name"]
        if len(arguments) != 1 or len(node.args) + len(node.keywords) != 1:
            raise ValueError("config_path 必須只有一個字面字串參數")
        argument = arguments[0]
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            raise ValueError("config_path 的參數不是字面字串")
        found.add(argument.value)
    return found


def scan_import_closure(root: Path, entry_module: str) -> tuple[PhysicsImportClosure, dict[str, ast.Module]]:
    pending = {entry_module}
    trees: dict[str, ast.Module] = {}
    references: set[ImportReference] = set()
    third_party: set[str] = set()
    data_files: set[str] = set()
    while pending:
        source = pending.pop()
        if source in trees:
            continue
        path = _source_path(root, source)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        trees[source] = tree
        package = source if path.stem == "__init__" else source.rpartition(".")[0]
        data_files.update(_config_literals(tree, source, package))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        # 入口與每一支目標的上層套件都必須讀；父套件本身也可能有 import。
        targets: list[tuple[str, bool]] = [(source, False)]
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                targets.extend((name, _in_function(node, parents))
                               for name in _import_targets(root, source, node))
        for target, in_function in targets:
            top = target.split(".", 1)[0]
            if top != "aosr":
                if top not in sys.stdlib_module_names:
                    third_party.add(top)
                continue
            parts = target.split(".")
            for end in range(1, len(parts) + 1):
                name = ".".join(parts[:end])
                if target != source:
                    references.add(ImportReference(source, name, in_function))
                if name not in trees:
                    pending.add(name)
    closure = PhysicsImportClosure(
        tuple(sorted(trees)), tuple(sorted(references, key=lambda item:
                                          (item.source, item.module, item.in_function))),
        tuple(sorted(data_files)), tuple(sorted(third_party)),
    )
    return closure, trees



def import_code_digest(root: Path, entry_module: str) -> str:
    """照物理身分的 AST 與具名長度框架算程式摘要；資料值由各輸入鑰匙負責。"""
    closure, trees = scan_import_closure(root, entry_module)
    digest = hashlib.sha256()
    for name in closure.modules:
        tree = trees[name]
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    value = node.body[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        node.body.pop(0)
        content = ast.dump(tree, include_attributes=False).encode("utf-8")
        digest.update(name.encode() + b"\0" + len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()
