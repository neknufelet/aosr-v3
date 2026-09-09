"""退出碼約定 ＋ 每支檢查共用的外殼。

約定（v2 的頭號死因是「只會回綠的檢查」，所以三種結局要分得開）：

* ``0`` 乾淨——真的掃過東西，而且沒有違規。
* ``1`` 有違規。
* ``2`` 工具自壞——**沒有掃到東西**，所以「沒找到違規」這句話不算數。
  以下一律 2：掃描根解析不到、列舉檔案的子程序非零退出、列舉出來是空集合、
  掃描根落在 repo 外面。

每支檢查在回傳前一定印一行 ``scan_root=<路徑> files=<整數> hits=<整數>``，
連 2 的路徑也要印——沒有這一行就看不出「回綠是因為乾淨，還是因為什麼都沒掃到」。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

CLEAN = 0
VIOLATION = 1
TOOL_BROKEN = 2

# 列舉版控裡的檔案用的子程序。只認本機 git，不連網。
ENUMERATE_ARGV = ("git", "ls-files", "--cached", "--others", "--exclude-standard")


class ToolBroken(Exception):
    """工具自壞：這一跑沒有真的掃到東西，結論不算數（退出碼 2）。"""


def repo_root() -> Path:
    """這個 repo 的根（往上找 .git）。檢查程式不准讀這個範圍外的路徑。"""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    raise ToolBroken(f"從 {here} 往上找不到 .git，無法確定 repo 根")


def resolve_scan_root(raw: str | Path) -> Path:
    """把 --scan-root 解析成一個真的存在、而且在 repo 裡面的目錄。"""
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    if not root.is_dir():
        raise ToolBroken(f"掃描根不存在或不是目錄：{root}")
    inside = repo_root()
    if root != inside and inside not in root.parents:
        raise ToolBroken(f"掃描根 {root} 落在 repo（{inside}）外面，檢查程式不准讀")
    return root


def enumerate_files(root: Path) -> list[Path]:
    """列舉掃描根底下「進得了版控」的檔案。

    刻意走 ``git ls-files`` 而不是自己走檔案系統：規矩只管版控裡的東西，
    而且這樣「外部工具被抽掉」是一個真的會發生的失敗，不是假設。
    """
    try:
        proc = subprocess.run(
            [*ENUMERATE_ARGV],
            cwd=root,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ToolBroken(f"列舉用的外部工具不在 PATH：{ENUMERATE_ARGV[0]}（{exc}）") from exc
    if proc.returncode != 0:
        raise ToolBroken(
            f"列舉子程序非零退出（{proc.returncode}）：{' '.join(ENUMERATE_ARGV)}"
            f"／{proc.stderr.strip()[:200]}"
        )
    names = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not names:
        raise ToolBroken(f"列舉出來是空集合：{root} 底下一個版控檔案都沒有")
    return [root / n for n in names]


def report(scan_root: str | Path, files: int, hits: int) -> None:
    """印那一行報告。每支檢查回傳前都要叫一次。"""
    print(f"scan_root={scan_root} files={files} hits={hits}")


CheckFn = Callable[[Path, list[Path]], list[str]]


def run(check: CheckFn, argv: Sequence[str] | None = None, *, description: str = "") -> int:
    """每支檢查的外殼：解析參數、列舉檔案、印報告行、決定 0／1／2。

    ``check(scan_root, files)`` 回傳違規訊息的 list，空 list 就是乾淨。
    它可以 raise ToolBroken 表示「這一跑不算數」。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scan-root", required=True, help="要掃的樹根（必須在 repo 裡）")
    args = parser.parse_args(argv)

    files: list[Path] = []
    try:
        root = resolve_scan_root(args.scan_root)
        files = enumerate_files(root)
        hits = check(root, files)
    except ToolBroken as exc:
        report(args.scan_root, len(files), 0)
        print(f"FAIL(2) 工具自壞：{exc}", file=sys.stderr)
        return TOOL_BROKEN

    for hit in hits:
        print(f"HIT: {hit}", file=sys.stderr)
    report(root, len(files), len(hits))
    if hits:
        print(f"FAIL(1) 有 {len(hits)} 筆違規", file=sys.stderr)
        return VIOLATION
    return CLEAN
