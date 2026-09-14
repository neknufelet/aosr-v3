"""本地型別存根逐項對真正第三方套件的執行期簽章。"""
from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from aosr import runtime

STUB_ROOT = Path(__file__).parents[2] / "src" / "typings"

# 目前挑進存根的 API 都能由 inspect.signature 取得簽章。這份明列集合刻意為空；若日後
# 加入拿不到簽章的 C 擴充物件，必須在這裡逐名說明，且考卷會確認它仍然真的拿不到。
UNSIGNABLE_DECLARATIONS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StubDeclaration:
    qualified_name: str
    runtime_path: tuple[str, ...]
    parameters: tuple[tuple[str, inspect._ParameterKind, bool], ...]


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
                )
            )
    return declarations


def _runtime_object(module: ModuleType, path: tuple[str, ...]) -> object:
    current: object = module
    for name in path:
        current = getattr(current, name)
    return current


def _runtime_parameters(value: object) -> tuple[tuple[str, inspect._ParameterKind, bool], ...]:
    if not callable(value):
        raise TypeError(f"declared callable resolved to non-callable {value!r}")
    signature = inspect.signature(value)
    return tuple(
        (parameter.name, parameter.kind, parameter.default is not inspect.Parameter.empty)
        for parameter in signature.parameters.values()
    )


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
                runtime_value = _runtime_object(module, declaration.runtime_path)
            except AttributeError:
                problems.append(f"{declaration.qualified_name}: real object is missing")
                continue
            try:
                actual = _runtime_parameters(runtime_value)
            except (TypeError, ValueError):
                if declaration.qualified_name not in UNSIGNABLE_DECLARATIONS:
                    problems.append(f"{declaration.qualified_name}: inspect.signature unavailable")
                continue
            if declaration.qualified_name in UNSIGNABLE_DECLARATIONS:
                problems.append(f"{declaration.qualified_name}: signature is now inspectable")
            elif actual != declaration.parameters:
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
