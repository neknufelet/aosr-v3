#!/usr/bin/env python3
"""樣本用的假檢查：掃描根裡有 BITE 這個檔就回 1，沒有就回 0。

只給必紅樣本當道具用，不是真的檢查。刻意只用標準庫、不 import 治理層，
這樣樣本迷你樹可以完全獨立餵給載入器。
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", required=True)
    args = ap.parse_args()
    root = Path(args.scan_root)
    if not root.is_dir():
        print(f"scan_root={root} files=0 hits=0")
        print("FAIL(2): 掃描根不存在", file=sys.stderr)
        return 2
    files = sum(1 for _ in root.rglob("*") if _.is_file())
    hits = 1 if (root / "BITE").exists() else 0
    print(f"scan_root={root} files={files} hits={hits}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
