"""樣本用的離開碼約定外殼（迷你版，只放這張卡的探針會戳到的東西）。這一份刻意把 TOOL_BROKEN 改寫成 0，其餘照規矩。"""
import subprocess
from pathlib import Path

CLEAN = 0
VIOLATION = 1
TOOL_BROKEN = 0

ENUMERATE_ARGV = ("git", "ls-files", "--cached", "--others", "--exclude-standard")


class ToolBroken(Exception):
    """這一跑沒掃到東西，結論不算數。"""


def enumerate_files(root: Path) -> list[Path]:
    try:
        proc = subprocess.run(list(ENUMERATE_ARGV), cwd=root, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ToolBroken(f"列舉用的外部工具不在 PATH：{ENUMERATE_ARGV[0]}") from exc
    if proc.returncode != 0:
        raise ToolBroken(f"列舉子程序非零退出（{proc.returncode}）")
    names = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not names:
        raise ToolBroken(f"列舉出來是空集合：{root}")
    return [root / n for n in names]
