"""靜態量物理入口的原始碼、資料與求解環境；不載入物理或評分模組。

只追 import 語句（含函式內與 TYPE_CHECKING）；動態 import、任意檔案讀取不在
這個分析器的範圍（資料只認 ``config_path`` 的字面值；閉包裡自己拿 ``CONFIG_DIR`` 組路徑一律拒收）。
能力表的說明與給人看的清單不參與身分。

程式那一半雜湊的是 ``ast.dump`` 的文字，它的格式跟著 Python 小版本走：換直譯器小版本時程式摘要也會變
（環境那一半本來就收 Python 版本，所以身分照樣換，不會多逼一次重算；釘答案的考卷會先比直譯器小版本）。
數值函式庫執行緒數與 CPU 造成的最後一位差異不在身分內：同一個身分只保證「不用重跑物理」，
不保證換一台機器重跑會逐位相同（產品的計算子行程一律單緒，見 ``aosr.gui.jobs`` 與 ``aosr.runtime.child_process_env``）。
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.paths import config_path
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.reporting.import_closure import PHYSICS_DEPENDENCY_ROOTS as PHYSICS_DEPENDENCY_ROOTS
from aosr.reporting.import_closure import ImportReference, PhysicsImportClosure, scan_import_closure
from aosr.reporting.import_closure import normalized_distribution_name as normalized_distribution_name
from aosr.reporting.import_closure import physics_dependency_versions as physics_dependency_versions
from aosr.reporting.import_closure import environment_digest as _environment_digest
from aosr.reporting.import_closure import feed_segment as _feed
from aosr.reporting.import_closure import json_bytes as _json_bytes


PHYSICS_ENTRY_MODULE = "aosr.reporting.physics_stage"
PHYSICS_CAPABILITY_ENTRIES = ("three_lane_report", "source_directivity")
_CAPABILITY_FIELDS = ("room", "materials", "frequency_hz", "outputs", "status", "evidence")


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


def _package_root(package_root: Path | None) -> Path:
    return package_root if package_root is not None else config_path("capabilities.toml").parents[2]


def _scan(root: Path, entry_module: str = PHYSICS_ENTRY_MODULE) -> tuple[PhysicsImportClosure, dict[str, ast.Module]]:
    return scan_import_closure(root, entry_module)


def physics_import_closure(*, package_root: Path | None = None,
                           entry_module: str = PHYSICS_ENTRY_MODULE) -> PhysicsImportClosure:
    """回傳靜態閉包與每條 import 的位置，不執行套件原始碼。"""
    return _scan(_package_root(package_root), entry_module)[0]


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


def _capability_bytes(table: CapabilityTable) -> bytes:
    entries: list[dict[str, object]] = []
    for name in PHYSICS_CAPABILITY_ENTRIES:
        entry = table.for_entry(name)
        rows = [item.model_dump(mode="json", include=set(_CAPABILITY_FIELDS))
                for item in entry.capability]
        entries.append({"name": entry.name, "module": entry.module, "capability": rows})
    return _json_bytes(entries)


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
    _feed(digest, b"furniture_geometry_contact",
          furniture_contact_rel(default_precision_contracts_path()).hex().encode())
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
