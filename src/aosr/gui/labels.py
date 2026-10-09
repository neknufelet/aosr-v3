"""網頁幾個頁面共用的中文字樣：只是顯示用的對照表，不含任何判斷。"""
from __future__ import annotations

from aosr.reporting.display import (
    LABELS as LABELS, DIRECTIONS as DIRECTIONS, LISTENING_POINTS as LISTENING_POINTS, SPEAKERS as SPEAKERS,
    listening_point_label as listening_point_label, speaker_label as speaker_label,
)
from aosr.search.labels import SPEAKER_SETUP as SPEAKER_SETUP
# 聲源模型：跟輸入頁下拉選單的字一樣。
SOURCE_MODELS = {"product_default": "產品預設指向", "omnidirectional": "全向"}
# 低頻取樣軸。
LOW_FREQUENCY_AXES = {"search_octave_24": "搜尋軸（每八度 24 點）",
                      "verification_linear_1hz": "驗證軸（每 1 Hz）"}
# 喇叭：用聲道代號叫它。喇叭的 left 跟座位的 left（主位左方）是兩回事，名字要分得開。
# 六面牆：跟方案輸入頁表單上阻抗、散射兩排的牆名一樣（比較頁「改了哪裡」也用這一張）。
# x、y 起點終點沒有前後左右的定義，不自己翻成前牆後牆。
WALLS = {"floor": "地板", "ceiling": "天花", "x0": "x 起點牆", "xL": "x 終點牆",
         "y0": "y 起點牆", "yL": "y 終點牆"}
# 房間長寬高：跟方案輸入頁表單上的欄名一樣。
ROOM_LENGTHS = {"Lx": "長 Lx（公尺）", "Ly": "寬 Ly（公尺）", "Lz": "高 Lz（公尺）"}
# 座位：主位加上周圍點方向的全名。代號沿用範例方案的座位代號（周圍點代號就是它的方向），
# 網頁只改座標、不改代號。


# 結果產物的計算狀態：清單字句、比較摘要裡的短名、頁首診斷警語只住這張表。
RESULT_RUN_LABELS = {
    "done": ("完成", "完成", ""),
    "none": ("完成（沒有計算紀錄）", "完成（沒有計算紀錄）", ""),
    "failed": ("失敗：內容可能不完整", "失敗",
               "這一筆計算回報失敗{exit_text}，結果檔雖然寫出來了，內容可能不完整；留著供診斷，不是正常完成的結果"),
    "stopped": ("已停止：內容可能不完整", "已停止",
                "這一筆計算被停止，結果檔雖然寫出來了，內容可能不完整；留著供診斷，不是正常完成的結果"),
    "running": ("計算中：結果檔還可能再變", "計算中",
                "這一筆還在計算中，結果檔還可能再變；不是正常完成的結果"),
}
RUN_EXIT_TEXT = "（離開碼 {code}）"


def label_tables() -> dict[str, dict[str, str]]:
    """給網頁的顯示名稱表（/api/labels）：頁面只查表，查不到就顯示原代號。"""
    return {"speakers": dict(SPEAKERS), "listening_points": dict(LISTENING_POINTS),
            "speaker_setup": dict(SPEAKER_SETUP)}



# 各類結果主表的旗標白話；每一句照設旗標的那段程式寫，不多說。表上沒有的退回 LABELS 的短名。
FLAG_TEXTS = {
    "unvalidated": "有一部分計算尚未驗證",
    "baseline_settings": "用的線是暫定的，尚未正式校準",
    "analytic_directivity_unvalidated": "喇叭指向性用解析近似，尚未獨立驗證",
    "feature_narrower_than_axis": "有峰谷比頻率取樣點的間距還窄",
    "feature_boundary_incomplete": "有峰谷找不到完整邊緣，寬度量不到",
    "feature_too_narrow": "有峰谷窄於設定的最小寬度",
    "data_coverage_short": "資料的頻率範圍不夠寬或中間有缺口",
    "window_only_delay_screen": "反射只看直達音後的時間窗",
    "geometry_material_conservative_screen": "反射用幾何與材料做保守篩選",
    "no_directivity": "沒有喇叭指向資料",
}
