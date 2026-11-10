"""#1066 有限元素分片自己的身分 fem-v1；不放進 fem_slices，免得物理身分的閉包多收身分工具。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from aosr.reporting.import_closure import environment_digest, feed_segment, import_code_digest, scan_import_closure

FEM_ENTRY_MODULE = "aosr.reporting.fem_slices"


def fem_identity(*, package_root: Path | None = None) -> str:
    """fem-v1＝分片模組靜態閉包的程式、閉包經 ``config_path`` 讀的資料檔與執行環境；收法照物理身分。

    閉包含求解、網格、阻抗與鑰匙的程式，不含晚期一族；資料檔照收（物理常數兩族都讀，改它照樣要重解）。
    """
    root = package_root if package_root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    feed_segment(digest, b"code", import_code_digest(root, FEM_ENTRY_MODULE).encode())
    for name in scan_import_closure(root, FEM_ENTRY_MODULE)[0].data_files:
        feed_segment(digest, b"data:" + name.encode(), (root / "config" / "data" / name).read_bytes())
    feed_segment(digest, b"environment", environment_digest().encode())
    return "fem-v1:" + digest.hexdigest()
