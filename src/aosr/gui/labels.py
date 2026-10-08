"""網頁幾個頁面共用的中文字樣：只是顯示用的對照表，不含任何判斷。"""
from __future__ import annotations

from aosr.reporting.display import (
    DIRECTIONS as DIRECTIONS, LISTENING_POINTS as LISTENING_POINTS, SPEAKERS as SPEAKERS,
    listening_point_label as listening_point_label, speaker_label as speaker_label,
)
# 聲源模型：跟輸入頁下拉選單的字一樣。
SOURCE_MODELS = {"product_default": "產品預設指向", "omnidirectional": "全向"}
# 低頻取樣軸。
LOW_FREQUENCY_AXES = {"search_octave_24": "搜尋軸（每八度 24 點）",
                      "verification_linear_1hz": "驗證軸（每 1 Hz）"}
# 喇叭設定的欄名與選項只有這一份；輸入頁透過 /api/labels、比較與問題訊息直接共用。
SPEAKER_SETUP = {
    "speaker_setup": "喇叭類型與擺法", "kind": "喇叭類型", "mount": "喇叭擺法",
    "cabinet": "喇叭箱體", "representative": "代表模型",
    "bookshelf": "書架喇叭", "floorstanding": "落地喇叭",
    "stand": "腳架", "desk": "桌面", "floor": "地面",
    "representative_model": "代表模型，非實際型號", "actual_model": "實際型號",
    "width_m": "箱寬", "depth_m": "箱深", "height_m": "箱高",
    "acoustic_center_behind_front_m": "聲學中心離前面板",
    "acoustic_center_above_bottom_m": "聲學中心離箱底",
}
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


LABELS = {
    "frequency_axis": "頻率（Hz）", "level_axis": "聲級（dB）",
    "left": "左聲道", "right": "右聲道", "primary": "主位",
    "surrounding": "周圍點", "other_seat": "其他座位",
    "primary_to_surrounding": "主位對周圍點",
    "surrounding_to_surrounding": "周圍點彼此",
    "tilt": "傾斜差", "ripple_rms": "起伏差（均方根）", "overall_level": "音量差",
    "timbre_balance": "音色平衡", "listening_area_stability": "聆聽區穩定性",
    "low_frequency_decay": "低頻拖尾", "reflections_and_echo": "反射與回聲",
    "reverberation": "殘響", "channel_matching": "聲道匹配",
    "spatial_impression": "空間感", "peak": "峰值警戒", "dip": "谷值警戒",
    "flutter": "牆間顫動警戒", "listening_area_worst_deviation": "聆聽區最差差距",
    "window_only_delay_screen": "只看時間窗",
    "geometry_material_conservative_screen": "幾何與材料的保守篩選",
    "measured": "已量", "costed": "已算代價", "unavailable": "不可估",
    "not_computable": "無法計算", "validated": "已驗證", "unvalidated": "尚未驗證",
    "experimental": "試驗中", "unsupported": "不支援", "unchecked": "未檢查",
    "complete": "完整", "not_provable": "無法證明完整", "missing": "缺資料",
    "approximate": "近似", "furniture": "家具",
    "front": "前方", "lateral": "側向", "rear": "後方", "vertical": "上下方",
    "below": "交界以下", "above": "交界以上", "crossing": "跨過交界",
    "baseline_settings": "使用暫定基線", "partial_frequency_overlap": "頻率範圍部分重疊",
    "listening_area_peer_group_missing": "周圍點彼此組缺資料",
    "no_directivity": "沒有指向資料",
    "rankable": "可排名", "eliminated": "淘汰", "not_evaluated": "未評估",
    "not_comparable": "不可同表比較", "illegal": "方案不合法",
    "insufficient_coverage": "覆蓋範圍不足", "timbre_scoring_range_gap": "音色計分頻段有缺口",
    "missing_points": "缺逐點資料", "non_positive_energy": "能量不是正值",
    "solver_unavailable": "求解不可用", "evaluator_not_implemented": "評估器尚未實作",
    "reflections_evaluation_missing": "缺反射評估", "candidate_id_mismatch": "候選代號不符",
    "speaker_id_mismatch": "喇叭代號不符", "receiver_set_fingerprint_mismatch": "座位配置指紋不符",
    "evaluator_version_mismatch": "評估器版本不符", "scene_fingerprint_mismatch": "房間指紋不符",
    "placement_mismatch": "擺位不符", "settings_fingerprint_mismatch": "設定指紋不符",
    "timbre_settings_fingerprint_mismatch": "音色設定指紋不符",
    "listening_area_settings_fingerprint_mismatch": "聆聽區設定指紋不符",
    "channel_group_fingerprint_mismatch": "聲道組指紋不符",
    "channel_result_unavailable": "聲道結果不可估",
    "required_channel_point_unavailable": "必要聲道位置不可估",
    "channel_role_mismatch": "聲道角色不符", "frequency_axis_mismatch": "頻率軸不符",
    "invalid_direct_distance": "直達距離無效", "receiver_id_mismatch": "座位代號不符",
    "timbre_not_measured": "音色尚未量到", "zero_total_importance": "周圍點重要性總和為零",
    "no_surrounding_pairs": "沒有周圍點配對", "insufficient_decay_range": "衰減範圍不足",
    "band_row_missing": "缺頻帶資料", "non_positive_value": "數值不是正值",
    "other_error": "其他錯誤", "path_table_missing": "缺路徑表",
    "reflection_screen_or_window_missing": "反射篩選或時間窗缺資料",
    "reflection_screen_or_window_mismatch": "反射篩選或時間窗不符",
    "reflection_window_incomplete": "反射時間窗不完整",
    "listening_axis_undefined": "聆聽方向無法定義",
    "no_reflection_in_zone_point": "這個方向沒有反射路徑",
    "zero_reflection_energy": "反射能量為零", "zero_retention": "反射保留率為零",
    "approximate_no_reflection_in_zone_point": "近似：已算路徑中沒有（家具參與的多次反射未納入）",
    "approximate_zero_reflection_energy": "近似：反射能量為零（家具參與的多次反射未納入）",
    "full_reflection": "全反射", "t20_band_unavailable": "本房 T20 頻帶不可估",
    "subband_sampling_incomplete": "子帶取樣不完整", "source_model_mismatch": "聲源模型不符",
    "mandatory_category_missing": "缺必要類別", "mandatory_category_unavailable": "必要類別不可估",
    "cost_not_computed": "代價尚未算出",
    "reverberation_too_many_unavailable_bands": "不可估殘響頻帶太多",
    "reverberation_critical_band_unavailable": "重要殘響頻帶不可估",
    "reverberation_insufficient_valid_bands": "可用殘響頻帶不足",
    "external_floor_failed": "外部底線未過",
    "timbre_peak_beyond_limit": "音色峰值超線",
    "timbre_dip_beyond_limit": "音色谷值超線",
    "listening_area_tilt_primary_to_surrounding_worst_beyond_limit": "主位對周圍點傾斜差超線",
    "listening_area_tilt_surrounding_to_surrounding_worst_beyond_limit": "周圍點彼此傾斜差超線",
    "listening_area_ripple_primary_to_surrounding_worst_beyond_limit": "主位對周圍點起伏差超線",
    "listening_area_ripple_surrounding_to_surrounding_worst_beyond_limit": "周圍點彼此起伏差超線",
    "listening_area_level_primary_to_surrounding_worst_beyond_limit": "主位對周圍點音量差超線",
    "listening_area_level_surrounding_to_surrounding_worst_beyond_limit": "周圍點彼此音量差超線",
    "channel_matching_tilt_worst_beyond_limit": "聲道傾斜差超線",
    "channel_matching_ripple_worst_beyond_limit": "聲道起伏差超線",
    "channel_matching_level_worst_beyond_limit": "聲道音量差超線",
    "channel_matching_direct_time_worst_beyond_limit": "聲道直達時間差超線",
    "data_coverage_short": "資料覆蓋不足", "crossover_band": "跨越頻帶交界",
    "feature_too_narrow": "特徵過窄", "feature_boundary_incomplete": "特徵邊界不完整",
    "feature_narrower_than_axis": "特徵窄於頻率軸",
    "reflection_front_above_threshold": "前方反射超線",
    "reflection_lateral_above_threshold": "側向反射超線",
    "reflection_rear_above_threshold": "後方反射超線",
    "reflection_vertical_above_threshold": "上下反射超線",
    "analytic_directivity_unvalidated": "解析指向性尚未驗證",
    "furniture_model_approximate": "近似",
    "furniture_parallel_flutter_not_assessed": "家具平行面顫動未評估",
    "floor": "地板", "ceiling": "天花", "x0": "x 起點牆", "xL": "x 終點牆",
    "y0": "y 起點牆", "yL": "y 終點牆",
}


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
