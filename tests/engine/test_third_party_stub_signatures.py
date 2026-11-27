"""本地型別存根逐項對真正第三方套件的執行期簽章。"""
from __future__ import annotations

import ast
import importlib
import inspect
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from aosr import runtime

STUB_ROOT = Path(__file__).parents[2] / "src" / "typings"

# 目前挑進存根的 API 都拿得到簽章：inspect.signature 拿得到的直接比；拿不到的 pybind11
# 擴充物件（rhino3dm，C++ 包成 Python 的套件）把簽章寫在 ``__doc__`` 第一行或「Overloaded
# function.」底下逐條列，改從那裡讀，存根要等於其中一條。這份明列集合刻意為空；若日後
# 加入兩邊都拿不到簽章的物件，必須在這裡逐名說明，且考卷會確認它仍然真的拿不到。
UNSIGNABLE_DECLARATIONS: frozenset[str] = frozenset()

Parameters = tuple[tuple[str, inspect._ParameterKind, bool], ...]


@dataclass(frozen=True)
class StubDeclaration:
    qualified_name: str
    runtime_path: tuple[str, ...]
    parameters: tuple[tuple[str, inspect._ParameterKind, bool], ...]
    is_property: bool = False


def _module_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    if path.name == "__init__.pyi":
        return ".".join(parts[:-1])
    return ".".join((*parts[:-1], path.stem))


def _parameters(arguments: ast.arguments) -> tuple[tuple[str, inspect._ParameterKind, bool], ...]:
    positional = (*arguments.posonlyargs, *arguments.args)
    default_start = len(positional) - len(arguments.defaults)
    found: list[tuple[str, inspect._ParameterKind, bool]] = []
    for index, argument in enumerate(positional):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(arguments.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        found.append((argument.arg, kind, index >= default_start))
    if arguments.vararg is not None:
        found.append((arguments.vararg.arg, inspect.Parameter.VAR_POSITIONAL, False))
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        found.append((argument.arg, inspect.Parameter.KEYWORD_ONLY, default is not None))
    if arguments.kwarg is not None:
        found.append((arguments.kwarg.arg, inspect.Parameter.VAR_KEYWORD, False))
    return tuple(found)


def _class_parameters(node: ast.ClassDef) -> tuple[tuple[str, inspect._ParameterKind, bool], ...]:
    initializer = next(
        (item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
        None,
    )
    if initializer is None:
        return ()
    parameters = _parameters(initializer.args)
    return parameters[1:]


def _walk_declarations(
    nodes: list[ast.stmt],
    *,
    module_name: str,
    parent_path: tuple[str, ...] = (),
) -> list[StubDeclaration]:
    declarations: list[StubDeclaration] = []
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            path = (*parent_path, node.name)
            declarations.append(
                StubDeclaration(
                    qualified_name=".".join((module_name, *path)),
                    runtime_path=path,
                    parameters=_class_parameters(node),
                )
            )
            declarations.extend(
                _walk_declarations(node.body, module_name=module_name, parent_path=path)
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            path = (*parent_path, node.name)
            declarations.append(
                StubDeclaration(
                    qualified_name=".".join((module_name, *path)),
                    runtime_path=path,
                    parameters=_parameters(node.args),
                    is_property=any(
                        isinstance(item, ast.Name) and item.id == "property" for item in node.decorator_list
                    ),
                )
            )
    return declarations


def _runtime_value(module: ModuleType, path: tuple[str, ...]) -> object:
    """取真物件；類別裡的屬性從 ``__dict__`` 拿，才拿得到 property 本身而不是它算出的值。"""
    current: object = module
    for name in path:
        owner = current
        current = getattr(owner, name)
        if isinstance(owner, type) and isinstance(owner.__dict__.get(name), property):
            current = owner.__dict__[name]
    return current


def _split_top_level(text: str) -> list[str]:
    """``a: tuple[int, int], b: str`` → 兩段；括號裡的逗號不切。"""
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        depth += char in "([{"
        depth -= char in ")]}"
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return parts


def _doc_overloads(value: object) -> list[Parameters]:
    """pybind11 寫在 ``__doc__`` 的簽章；類別讀它的 ``__init__``，並去掉 self。"""
    is_class = isinstance(value, type)
    target: object = operator.attrgetter("__init__")(value) if is_class else value
    name = getattr(target, "__name__", "")
    doc = getattr(target, "__doc__", None) or ""
    lines = doc.splitlines()
    if "Overloaded function." in lines:
        lines = lines[lines.index("Overloaded function.") + 1:]
    pattern = re.compile(rf"^(?:\d+\. )?{re.escape(name)}\((.*)\) -> ")
    overloads: list[Parameters] = []
    for line in lines:
        match = pattern.match(line)
        if match is None:
            continue
        found = tuple(
            (part.split(":")[0].split("=")[0].strip(), inspect.Parameter.POSITIONAL_OR_KEYWORD, "=" in part)
            for part in _split_top_level(match.group(1))
        )
        overloads.append(found[1:] if is_class else found)
    if not overloads and is_class and doc == object.__init__.__doc__:
        # 沒有建構子的 pybind11 類別（只能由套件交出來）：存根也不准宣告 __init__。
        overloads.append(())
    return overloads


def _runtime_parameters(value: object) -> list[Parameters]:
    if not callable(value):
        raise TypeError(f"declared callable resolved to non-callable {value!r}")
    try:
        signature = inspect.signature(value)
    except ValueError:
        overloads = _doc_overloads(value)
        if not overloads:
            raise
        return overloads
    return [tuple(
        (parameter.name, parameter.kind, parameter.default is not inspect.Parameter.empty)
        for parameter in signature.parameters.values()
    )]


def _compare_stub_tree(root: Path) -> list[str]:
    runtime.preload_mkl()
    problems: list[str] = []
    unseen_unsignable = set(UNSIGNABLE_DECLARATIONS)
    paths = sorted(root.rglob("*.pyi"))
    if not paths:
        return [f"{root}: no .pyi declarations found"]
    for path in paths:
        module_name = _module_name(path, root)
        module = importlib.import_module(module_name)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for declaration in _walk_declarations(tree.body, module_name=module_name):
            unseen_unsignable.discard(declaration.qualified_name)
            try:
                runtime_value = _runtime_value(module, declaration.runtime_path)
            except AttributeError:
                problems.append(f"{declaration.qualified_name}: real object is missing")
                continue
            if declaration.is_property:
                if not isinstance(runtime_value, property):
                    problems.append(f"{declaration.qualified_name}: stub property is not a real property")
                continue
            try:
                actual = _runtime_parameters(runtime_value)
            except (TypeError, ValueError):
                if declaration.qualified_name not in UNSIGNABLE_DECLARATIONS:
                    problems.append(f"{declaration.qualified_name}: inspect.signature unavailable")
                continue
            if declaration.qualified_name in UNSIGNABLE_DECLARATIONS:
                problems.append(f"{declaration.qualified_name}: signature is now inspectable")
            elif declaration.parameters not in actual:
                problems.append(
                    f"{declaration.qualified_name}: stub {declaration.parameters!r} != real {actual!r}"
                )
    for qualified_name in sorted(unseen_unsignable):
        problems.append(f"{qualified_name}: unsignable list entry has no stub declaration")
    return problems


def test_every_local_stub_declaration_matches_the_installed_package() -> None:
    """改名、改參數種類、增減預設值或宣告不存在的物件都必須紅。"""
    assert _compare_stub_tree(STUB_ROOT) == []


def test_signature_exam_rejects_a_wrong_parameter_name(tmp_path: Path) -> None:
    """控制組證明考卷不是只會回綠。"""
    wrong_stub = tmp_path / "gmsh.pyi"
    wrong_stub.write_text(
        "def initialize(wrong_name: list[str] = ...) -> None: ...\n",
        encoding="utf-8",
    )

    problems = _compare_stub_tree(tmp_path)

    assert problems
    assert problems[0].startswith("gmsh.initialize: stub")


def test_signature_exam_reads_pybind_docstrings(tmp_path: Path) -> None:
    """拿不到 inspect.signature 的 pybind11 物件改讀 ``__doc__``，參數名寫錯一樣紅。"""
    wrong_stub = tmp_path / "rhino3dm.pyi"
    wrong_stub.write_text(
        "class MeshFaceList:\n"
        "    def AddFace(self, vertex1: int, vertex2: int, vertex3: int) -> int: ...\n"
        "    def __len__(self, extra: int) -> int: ...\n",
        encoding="utf-8",
    )

    problems = _compare_stub_tree(tmp_path)

    assert len(problems) == 1
    assert problems[0].startswith("rhino3dm.MeshFaceList.__len__: stub")
