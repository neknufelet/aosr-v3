"""細算與選取共用的逐列判讀與同步寫入；只捨棄未寫完的末列。"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import BinaryIO, TypeVar

from pydantic import BaseModel

RowModel = TypeVar("RowModel", bound=BaseModel)


def read_rows(lines: Sequence[bytes], row_type: type[RowModel], label: str, *,
              first_line: int = 1) -> tuple[tuple[RowModel, ...], bool, int]:
    """完整 JSON（資料格式）的欄位違規必須報錯，不能冒充寫入中斷。"""
    rows: list[RowModel] = []
    valid_bytes = 0
    dropped = False
    for index, line in enumerate(lines):
        is_last = index == len(lines) - 1
        if is_last and not line.endswith(b"\n"):
            dropped = True
            break
        try:
            document: object = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if not is_last:
                raise ValueError(f"corrupt {label} line {first_line + index}") from error
            dropped = True
            break
        rows.append(row_type.model_validate(document))
        valid_bytes += len(line)
    return tuple(rows), dropped, valid_bytes


def write_line(handle: BinaryIO, value: BaseModel) -> None:
    """每列沖出緩衝並同步到磁碟；兩本帳共用相同序列化方式。"""
    handle.write((json.dumps(value.model_dump(mode="json"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())
