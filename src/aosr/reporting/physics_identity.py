"""靜態量物理入口的原始碼、資料與求解環境；不載入物理或評分模組。

只追 import 語句（含函式內與 TYPE_CHECKING）；動態 import、任意檔案讀取不在
這個分析器的範圍（資料只認 ``config_path`` 的字面值；閉包裡自己拿 ``CONFIG_DIR`` 組路徑一律拒收）。
能力表的說明與給人看的清單不參與身分。

程式那一半雜湊的是 ``ast.dump`` 的文字，它的格式跟著 Python 小版本走：換直譯器小版本時程式摘要也會變
（環境那一半本來就收 Python 版本，所以身分照樣換，不會多逼一次重算；釘答案的考卷會先比直譯器小版本）。
數值函式庫執行緒數與 CPU 造成的最後一位差異不在身分內：同一個身分只保證「不用重跑物理」，
不保證換一台機器重跑會逐位相同（產品的計算子行程一律單緒，見 ``gui/jobs.py`` 與 ``runtime.child_process_env``）。
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from importlib.util import resolve_name
from pathlib import Path
from typing import Protocol

from packaging.requirements import Requirement

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.paths import config_path


PHYSICS_ENTRY_MODULE = "aosr.reporting.physics_stage"
PHYSICS_CAPABILITY_ENTRIES = ("three_lane_report", "source_directivity")
# 數值、網格、直接求解與輸入模型的執行期依賴，加上 pydiso 的 Cython／MKL 建置輸入。
PHYSICS_DEPENDENCY_ROOTS = (
    "numpy", "scipy", "scikit-fem", "gmsh", "pydiso", "mkl", "pydantic", "cython", "mkl-devel",
)
_CAPABILITY_FIELDS = ("room", "materials", "frequency_hz", "outputs", "status", "evidence")
_EXTRA_MARKER = re.compile(r"\bextra\b")


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


@dataclass(frozen=True)
class PhysicsIdentityParts:
    """物理身分的可核對分件；程式與資料摘要不含套件版本與平台（但 ``ast.dump`` 格式隨 Python 小版本）。"""

    closure: tuple[str, ...]
    data_files: tuple[str, ...]
    third_party: tuple[str, ...]
    imports: tuple[ImportReference, ...]
    code_digest: str
    environment_digest: str
    identity: str


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


def _feed(digest: _Digest, label: bytes, content: bytes) -> None:
    """名稱、零位元組與八位元組長度隔開每段，避免拼接碰撞。"""
    digest.update(label + b"\0" + len(content).to_bytes(8, "big") + content)


def _package_root(package_root: Path | None) -> Path:
    return package_root if package_root is not None else config_path("capabilities.toml").parents[2]


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


def _scan(root: Path) -> tuple[PhysicsImportClosure, dict[str, ast.Module]]:
    pending = {PHYSICS_ENTRY_MODULE}
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


def physics_import_closure(*, package_root: Path | None = None) -> PhysicsImportClosure:
    """回傳靜態閉包與每條 import 的位置，不執行套件原始碼。"""
    return _scan(_package_root(package_root))[0]


def _without_docstrings(tree: ast.Module) -> ast.Module:
    """只移除每個模組、類別與函式本體第一句的字串常數；其他字串保留。"""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body.pop(0)
    return tree


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _capability_bytes(table: CapabilityTable) -> bytes:
    entries: list[dict[str, object]] = []
    for name in PHYSICS_CAPABILITY_ENTRIES:
        entry = table.for_entry(name)
        rows = [item.model_dump(mode="json", include=set(_CAPABILITY_FIELDS))
                for item in entry.capability]
        entries.append({"name": entry.name, "module": entry.module, "capability": rows})
    return _json_bytes(entries)


def normalized_distribution_name(name: str) -> str:
    """發行名小寫，底線、點與連字號按套件名規則正規化。"""
    return re.sub(r"[-_.]+", "-", name).lower()


def physics_dependency_versions() -> tuple[tuple[str, str], ...]:
    """由已安裝 metadata 展開必要依賴，略過 extra 並求值環境標記；缺套件就炸。"""
    pending = set(PHYSICS_DEPENDENCY_ROOTS)
    versions: dict[str, str] = {}
    while pending:
        name = normalized_distribution_name(pending.pop())
        if name in versions:
            continue
        distribution = metadata.distribution(name)
        versions[name] = metadata.version(name)
        for raw in distribution.requires or ():
            requirement = Requirement(raw)
            marker = requirement.marker
            if marker is not None and (_EXTRA_MARKER.search(str(marker)) or not marker.evaluate()):
                continue
            dependency = normalized_distribution_name(requirement.name)
            if dependency not in versions:
                pending.add(dependency)
    return tuple(sorted(versions.items()))


def _environment_digest() -> str:
    digest = hashlib.sha256()
    _feed(digest, b"python", _json_bytes(sys.version_info[:3]))
    # 平台：只有真的換機器或換系統 C 函式庫才會變；數值函式庫依 CPU 挑程式碼、數學函式走系統函式庫。
    _feed(digest, b"machine", platform.machine().encode())
    _feed(digest, b"libc", _json_bytes(platform.libc_ver()))
    for name, version in physics_dependency_versions():
        _feed(digest, b"dependency:" + name.encode(), version.encode())
    return digest.hexdigest()


def physics_identity_parts(
    *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
    package_root: Path | None = None,
) -> PhysicsIdentityParts:
    """回傳 phys-v1 的閉包、程式＋資料摘要、環境摘要與完整身分。"""
    root = _package_root(package_root)
    closure, trees = _scan(root)
    digest = hashlib.sha256()
    for name in closure.modules:
        normalized = ast.dump(_without_docstrings(trees[name]), include_attributes=False)
        _feed(digest, name.encode(), normalized.encode("utf-8"))
    for name in closure.data_files:
        _feed(digest, b"data:" + name.encode(), (root / "config" / "data" / name).read_bytes())
    _feed(digest, b"capabilities", _capability_bytes(capabilities))
    _feed(digest, b"directivity", _json_bytes(directivity.model_dump(mode="json")))
    code_digest = digest.hexdigest()
    environment_digest = _environment_digest()
    identity_digest = hashlib.sha256()
    _feed(identity_digest, b"code", code_digest.encode())
    _feed(identity_digest, b"environment", environment_digest.encode())
    return PhysicsIdentityParts(
        closure.modules, closure.data_files, closure.third_party, closure.imports,
        code_digest, environment_digest, "phys-v1:" + identity_digest.hexdigest(),
    )


def physics_identity(
    *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
    package_root: Path | None = None,
) -> str:
    """只回物理身分；呼叫端供給同一次計算的能力表與指向性預設。"""
    return physics_identity_parts(capabilities=capabilities, directivity=directivity,
                                  package_root=package_root).identity
