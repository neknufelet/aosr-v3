"""鏡像的共用鎖：寫的人握互斥鎖、讀的人握共用鎖，擋住「抄到一半」的半份鏡像。

寫 `mirror_receipts.mirror` 先抄收據、最後才寫 `provenance.json`；中間那一瞬間，讀
`cloud_receipts.read_mirror` 的檢查會拿到「收據在、來源檔不在」的半份鏡像，判成工具自壞。
修法（票 #235）：writer 整段握互斥鎖、reader 整段握共用鎖，同一條鎖。

**鎖誰。** 不新造鎖檔（鏡像目錄整套會被 `shutil.rmtree` 清掉，鎖檔跟不上）。改成對**已經在的
`out.parent` 目錄**開唯讀 fd、用 `fcntl.flock` 上鎖。兩邊永遠解到同一個穩定的父目錄 inode，
鏡像從不刪自己的父目錄。代價是同一個父目錄底下的兄弟鏡像互相排隊——可以接受。

**平台。** `fcntl` 與 `O_DIRECTORY` 是 Unix 專用；目前本機與 CI 都是 Linux/Ubuntu。不宣稱新的
可移植性。程序退出時鎖自行釋放；這支函式額外保證關 fd。
"""
from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator
from pathlib import Path

from governance.exit_codes import ToolBroken


@contextlib.contextmanager
def mirror_lock(out: Path, *, exclusive: bool) -> Iterator[None]:
    """鎖住 `out.parent` 這個目錄；`exclusive` 拿寫鎖、否則拿讀鎖。拿不到就 ToolBroken（回 2）。"""
    parent = out.parent
    try:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise ToolBroken(f"開不出鏡像鎖目錄 {parent}：{exc}") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except OSError as exc:
            raise ToolBroken(f"拿不到鏡像鎖（{parent}）：{exc}") from exc
        yield
    finally:
        try:
            os.close(fd)
        except OSError as exc:
            raise ToolBroken(f"關不掉鏡像鎖 fd（{parent}）：{exc}") from exc
