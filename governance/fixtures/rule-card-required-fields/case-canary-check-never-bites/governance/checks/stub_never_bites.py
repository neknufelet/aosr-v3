#!/usr/bin/env python3
"""金絲雀樣本用的假檢查：不管餵什麼都回 0。

存在的目的只有一個——證明後設測試的第二關（附的樣本必須真的讓檢查回 1）有牙。
如果哪天這支假檢查被判「乾淨」，就是第二關失效了。
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", required=True)
    args = ap.parse_args()
    root = Path(args.scan_root)
    print(f"scan_root={root} files=0 hits=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
