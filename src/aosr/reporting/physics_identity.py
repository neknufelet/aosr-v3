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
import json
import platform
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol

from packaging.requirements import Requirement

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.paths import config_path
from aosr.reporting.import_closure import ImportReference, PhysicsImportClosure, scan_import_closure


PHYSICS_ENTRY_MODULE = "aosr.reporting.physics_stage"
PHYSICS_CAPABILITY_ENTRIES = ("three_lane_report", "source_directivity")
# 數值、網格、直接求解與輸入模型的執行期依賴，加上 pydiso 的 Cython／MKL 建置輸入。
PHYSICS_DEPENDENCY_ROOTS = (
    "numpy", "scipy", "scikit-fem", "gmsh", "pydiso", "mkl", "pydantic", "cython", "mkl-devel",
)
_CAPABILITY_FIELDS = ("room", "materials", "frequency_hz", "outputs", "status", "evidence")
_EXTRA_MARKER = re.compile(r"\bextra\b")


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
