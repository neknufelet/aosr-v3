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
# 喇叭：用聲道代號叫它。喇叭的 left 跟座位的 left（主位左方）是兩回事，名字要分得開。
SPEAKERS = {"left": "左聲道喇叭", "right": "右聲道喇叭"}
# 六面牆：跟方案輸入頁表單上阻抗、散射兩排的牆名一樣（比較頁「改了哪裡」也用這一張）。
# x、y 起點終點沒有前後左右的定義，不自己翻成前牆後牆。
WALLS = {"floor": "地板", "ceiling": "天花", "x0": "x 起點牆", "xL": "x 終點牆",
         "y0": "y 起點牆", "yL": "y 終點牆"}
# 房間長寬高：跟方案輸入頁表單上的欄名一樣。
ROOM_LENGTHS = {"Lx": "長 Lx（公尺）", "Ly": "寬 Ly（公尺）", "Lz": "高 Lz（公尺）"}
# 座位：主位加上周圍點方向的全名。代號沿用範例方案的座位代號（周圍點代號就是它的方向），
# 網頁只改座標、不改代號。
LISTENING_POINTS = {"main": "主位", **{key: full for key, (_, full) in DIRECTIONS.items()}}


def speaker_label(channel_id: str) -> str:
    """喇叭顯示名；表上沒有的代號照原樣回，不猜。"""
    return SPEAKERS.get(channel_id, channel_id)


def listening_point_label(point_id: str) -> str:
    """座位顯示名；表上沒有的代號照原樣回，不猜。"""
    return LISTENING_POINTS.get(point_id, point_id)


def label_tables() -> dict[str, dict[str, str]]:
    """給網頁的顯示名稱表（/api/labels）：頁面只查表，查不到就顯示原代號。"""
    return {"speakers": dict(SPEAKERS), "listening_points": dict(LISTENING_POINTS)}
