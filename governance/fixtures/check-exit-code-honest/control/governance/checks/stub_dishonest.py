#!/usr/bin/env python3
"""控制樣本用的假檢查：離開碼回 9，而且沒有任何一張卡宣告它。

已知會咬的最小輸入。它的用途不是測某個特定寫法，而是「檢查還活著」的活體證明——
這一份餵下去回 0，就代表 check-exit-code-honest 自己死了。
"""
import sys

print("結論：乾淨")
sys.exit(9)
