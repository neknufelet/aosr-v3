#!/usr/bin/env python3
"""樣本用的假檢查：只印結論，不印 scan_root= files= hits= 那一行。

壞在：它印「乾淨」「有違規」這種結論，看不出它到底掃了幾個檔。沒有那一行，就分不出「回綠是因為乾淨」還是「因為什麼都沒掃到」。

只給 check-exit-code-honest 的樣本當道具用，不是真的檢查。刻意只用標準庫、不 import
治理層，這樣這棵迷你掃描根可以完全獨立餵給載入器與探針。
誠實版的行為（除了上面那一條，其餘照這個）：掃描根不存在／列舉工具缺席／列舉子程序
非零退出／列舉出來是空集合，一律回 2；掃到叫 BITE 的檔回 1；其餘回 0；每條路徑回傳前
都印一行 scan_root= files= hits=。
"""
import argparse
import subprocess
import sys
from pathlib import Path

ENUMERATE_ARGV = ("git", "ls-files", "--cached", "--others", "--exclude-standard")


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
    print("結論：乾淨" if hits == 0 else f"結論：有 {hits} 筆違規")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", required=True)
    args = ap.parse_args()
    root = Path(args.scan_root)
    if not root.is_dir():
        report(args.scan_root, 0, 0)
        print("FAIL(2) 掃描根不存在", file=sys.stderr)
        return 2
    try:
        names = enumerate_files(root)
    except ToolBroken as exc:
        report(args.scan_root, 0, 0)
        print(f"FAIL(2) 工具自壞：{exc}", file=sys.stderr)
        return 2
    hits = [n for n in names if Path(n).name == "BITE"]
    report(args.scan_root, len(names), len(hits))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
