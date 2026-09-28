"""網頁幾個頁面共用的中文字樣：只是顯示用的對照表，不含任何判斷。"""
from __future__ import annotations

# 周圍點相對主位的方向：（平面圖短標記, 全名）。
DIRECTIONS = {"front": ("前", "主位前方"), "back": ("後", "主位後方"),
              "left": ("左", "主位左方"), "right": ("右", "主位右方"),
              "up": ("上", "主位上方"), "down": ("下", "主位下方")}
# 聲源模型：跟輸入頁下拉選單的字一樣。
SOURCE_MODELS = {"product_default": "產品預設指向", "omnidirectional": "全向"}
# 低頻取樣軸。
LOW_FREQUENCY_AXES = {"search_octave_24": "搜尋軸（每八度 24 點）",
                      "verification_linear_1hz": "驗證軸（每 1 Hz）"}
