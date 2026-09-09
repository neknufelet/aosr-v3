#!/usr/bin/env python3
"""樣本用的假檢查：只掃 docs 第一層的 .md，刻意不遞迴進子目錄。

只給 scan-scope-has-no-holes 的樣本當道具用，不是真的檢查。刻意只用標準庫、不 import
治理層，這樣這棵迷你掃描根可以完全獨立餵下去。

被用到的只有列舉模式（``--list-files``）：那是這張卡量「實際掃描面」的方式。
正常模式印報告行、回 0——這一份沒有要判的東西，它的角色是「它說自己掃哪些檔」。
掃描根不存在／列舉工具缺席／列舉子程序非零退出／列舉出來是空集合，一律回 2。
"""
import argparse
import subprocess
import sys
from pathlib import Path

ENUMERATE_ARGV = ("git", "ls-files", "--cached", "--others", "--exclude-standard")
LIST_BEGIN = "list_files_begin"
LIST_END = "list_files_end"


class ToolBroken(Exception):
    """這一跑沒掃到東西，結論不算數。"""


def enumerate_files(root: Path) -> list[str]:
    try:
        proc = subprocess.run(list(ENUMERATE_ARGV), cwd=root, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ToolBroken(f"列舉用的外部工具不在 PATH：{ENUMERATE_ARGV[0]}") from exc
    if proc.returncode != 0:
        raise ToolBroken(f"列舉子程序非零退出（{proc.returncode}）")
    names = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not names:
        raise ToolBroken(f"列舉出來是空集合：{root}")
    return names


def report(scan_root: str, files: int, hits: int) -> None:
    print(f"scan_root={scan_root} files={files} hits={hits}")


def picked(names: list[str]) -> list[str]:
    """docs 第一層的 .md（子目錄裡的不看——這就是「不遞迴」長出來的形狀）。"""
    out = []
    for n in names:
        rest = n[len("docs/") :] if n.startswith("docs/") else ""
        if rest and "/" not in rest and n.endswith(".md"):
            out.append(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", required=True)
    ap.add_argument("--list-files", action="store_true")
    args = ap.parse_args()
    root = Path(args.scan_root)
    if not root.is_dir():
        if not args.list_files:
            report(args.scan_root, 0, 0)
        print("FAIL(2) 掃描根不存在", file=sys.stderr)
        return 2
    try:
        names = enumerate_files(root)
    except ToolBroken as exc:
        if not args.list_files:
            report(args.scan_root, 0, 0)
        print(f"FAIL(2) 工具自壞：{exc}", file=sys.stderr)
        return 2
    mine = sorted(picked(names))
    if args.list_files:
        if not mine:
            print("FAIL(2) 列舉出來是空集合", file=sys.stderr)
            return 2
        print(f"{LIST_BEGIN}={len(mine)}")
        for name in mine:
            print(name)
        print(LIST_END)
        return 0
    report(args.scan_root, len(names), 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
