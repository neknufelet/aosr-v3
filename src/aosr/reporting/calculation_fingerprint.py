"""計算指紋量套件位元組、能力表、Python 與正式依賴的執行期版本。

只扣 gui：分層卡 layers-import-downward-only 守住任何計算層不得載入 gui。
已知保守處是 reporting 內純顯示或只給命令列用的程式也會入指紋；
它們改動只可能誤判「不同」，不會把不同的計算誤判成「相同」。
"""
from __future__ import annotations

import hashlib
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Protocol


DIST_NAME = "aosr-v3-governance"
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


def _feed(digest: _Digest, label: bytes, content: bytes) -> None:
    """每段以名稱與位元組長度隔開，避免不同欄位拼接碰撞。"""
    digest.update(label + b"\0" + len(content).to_bytes(8, "big") + content)


def calculation_fingerprint(*, capabilities_path: Path,
                            package_root: Path | None = None) -> str:
    """回傳計算當下的 calc-v1 SHA-256；能力表內容與所在路徑無關。"""
    root = package_root if package_root is not None else Path(__file__).resolve().parents[1]
    if not root.is_dir():
        raise ValueError("計算套件目錄不存在")
    digest = hashlib.sha256()
    files = sorted((path for path in root.rglob("*") if path.is_file()
                    and path.relative_to(root).parts[0] != "gui"
                    and "__pycache__" not in path.relative_to(root).parts),
                   key=lambda path: path.relative_to(root).as_posix())
    for path in files:
        _feed(digest, path.relative_to(root).as_posix().encode("utf-8"), path.read_bytes())
    _feed(digest, b"capabilities", capabilities_path.read_bytes())
    _feed(digest, b"python", ".".join(str(part) for part in sys.version_info[:3]).encode())
    requires = metadata.distribution(DIST_NAME).requires or ()
    names = {match.group(1).lower() for requirement in requires
             if (match := _REQUIREMENT_NAME.match(requirement)) is not None}
    for name in sorted(names):
        _feed(digest, b"dependency:" + name.encode(), metadata.version(name).encode())
    return "calc-v1:" + digest.hexdigest()


def short_fingerprint(value: str) -> str:
    """給人看的前 12 碼：跳過「calc-v1:」前綴，不然只剩 4 碼指紋。"""
    return value.split(":", 1)[-1][:12]
