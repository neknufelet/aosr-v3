"""報表呈現共用的字句與純單位換算。"""
from __future__ import annotations

import math
import json

from aosr.physics.fem_modal_check import FemModalCheck
from aosr.reporting.modal_diagnosis_model import CheckSummary, ModalDiagnosis
from aosr.materials.furniture_materials import FurnitureImpedanceOnAxis


# 家具決策紙第 3、16、17 條；面方向依施工單按房間軸命名。
FURNITURE_KINDS = {"sofa": "沙發", "chair": "座椅", "coffee_table": "茶几",
                   "desk": "書桌", "ceiling_cloud": "天雲"}
FURNITURE_MATERIALS = {"fabric": "布面", "leather": "皮面", "wood": "木質",
                       "glass": "玻璃", "absorptive_cloud": "吸音天雲"}
FURNITURE_FACES = {"top": "頂面", "bottom": "底面", "+x": "朝 +x 的面",
                   "-x": "朝 -x 的面", "+y": "朝 +y 的面", "-y": "朝 -y 的面"}
# 第 13 條（第 75、77、78 行）、第 16、23、25、27、22 條，逐字取用。
FURNITURE_REFLECTION_NOTE = "家具僅一次反射、混合反射未納入"
FURNITURE_MODEL_NOTE = "家具模型：近似"
FURNITURE_REASON = ("已含家具一次反射、遮擋與有限尺寸鏡面修正；未含家具與牆之間的多次反射、"
                    "完整繞射，以及家具吸音對整房殘響的影響")
FURNITURE_ESTIMATE_NOTE = "估計，非本件實測"
# 第 18 條：這一版只新增決策紙明寫要註明的材質附註。
FURNITURE_MATERIAL_NOTES = {"leather": "參考的是合成皮"}
# 第 17 條拿掉中間的出處說明；第 10 條原文；第 11 條第 67 行接主詞。
FURNITURE_WOOD_CLOUD_NOTE = "數值借用木質桌面那組估計值，懸空的板直接借用是近似"
FURNITURE_SCATTERING_NOTE = "家具表面的粗糙散射第一版不算"
FURNITURE_FINITE_SIZE_NOTE = ("有限尺寸鏡面修正的限制：原理論的前提「家具尺寸遠小於距離」對桌面與沙發不成立；"
    "截止以上一律截在 1（精確式會略超過 1）；反射點靠板邊時會高估，最多約 6 dB；"
    "只乘實數、沒有相位與邊緣繞射路徑；靠牆或接靠背的非自由邊會低估；只收矩形面。")
FURNITURE_REVERBERATION_NOTE = "未包含家具吸音"
FURNITURE_REVERBERATION_REPORT_NOTE = f"整房殘響{FURNITURE_REVERBERATION_NOTE}"
FURNITURE_REVERBERATION_COMPARISON_NOTE = "不能用來判斷增加家具後的殘響改善量"
FURNITURE_MODEL_TITLE = "家具模型說明"
FURNITURE_TRANSMISSION_NOTE = "透射未算"
FURNITURE_DIRECTIVITY_NOTE = "喇叭指向性往下的方向尚未獨立驗證，桌面反射強度靠這個假設"
FURNITURE_FLUTTER_NOTE = "顫動警戒第一版只看三對牆，家具形成的平行面未評估"
FURNITURE_BOUNDARY_NOTE = "遮擋邊界上的反射會突然出現或消失"
FURNITURE_COVERAGE_NOTE = "原本牆面覆蓋條件成立"
# 第 11 條第 68 行，施工單批准的畫面改寫。
FURNITURE_VALIDATION_NOTE = "家具反射只驗證公式實作一致，實際家具精度未驗證"
# 第 13 條第 79 行、已拍設計 B1。
FURNITURE_COMPARISON_REASON = "兩者計算涵蓋範圍不同"
OTHER_SETTINGS_LABEL = "其他設定（未逐項列出）"  # 施工單指定的保險列。
NO_SCHEME_CHANGES_TEXT = "兩份方案設定相同"
# 第 6、7 條：相對擺法與天雲的房間座標。
FURNITURE_FIELDS = {"width_m": ("寬", "公尺"), "depth_m": ("深", "公尺"),
                    "height_m": ("高", "公尺"), "material": ("材質類型", ""),
                    "kind": ("種類", ""), "forward_m": ("前方", "公尺"),
                    "left_m": ("左方", "公尺"), "bottom_height_m": ("底面離地", "公尺"),
                    "yaw_deg": ("相對角", "度"), "bottom_center_m": ("底面中心的房間座標", "公尺")}
FURNITURE_ROOM_ANGLE = "房間角度"
APPROXIMATE_TEXT = "近似"
CONDITIONS_DIFFER_TEXT = "評分條件不同"


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
    "direction_zones": "方向分區", "ranking": "排名規則",
    "spatial_impression": "空間感", "peak": "峰值警戒", "dip": "谷值警戒",
    "flutter": "牆間顫動警戒", "listening_area_worst_deviation": "聆聽區最差差距",
    "window_only_delay_screen": "只看時間窗",
    "geometry_material_conservative_screen": "幾何與材料的保守篩選",
    "measured": "已量", "costed": "已算代價", "unavailable": "不可估",
    "not_computable": "無法計算", "validated": "已驗證", "unvalidated": "尚未驗證",
    "experimental": "試驗中", "unsupported": "不支援", "unchecked": "未檢查",
    "complete": "完整", "not_provable": "無法證明完整", "missing": "缺資料",
    "approximate": APPROXIMATE_TEXT, "furniture": "家具",
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
    "furniture_model_approximate": APPROXIMATE_TEXT,
    "furniture_parallel_flutter_not_assessed": "家具平行面顫動未評估",
    "floor": "地板", "ceiling": "天花", "x0": "x 起點牆", "xL": "x 終點牆",
    "y0": "y 起點牆", "yL": "y 終點牆",
}


def reflection_models_differ(a_support: str, b_support: str) -> bool:
    """B1 只在比較支撐的 reflection_model 不同時補原因。"""
    if not a_support or not b_support:
        return False
    a: dict[str, object] = json.loads(a_support)
    b: dict[str, object] = json.loads(b_support)
    return a.get("reflection_model") != b.get("reflection_model")


def furniture_name(kind: str, furniture_id: str) -> str:
    return f"{FURNITURE_KINDS[kind]}（{furniture_id}）"


def furniture_unknown_text(bands_hz: tuple[float, ...]) -> str:
    """第 20 條：讀物理層既有 unknown_label，不再抄一份。"""
    frequencies = "、".join(f"{frequency:g}" for frequency in bands_hz)
    return f"{FurnitureImpedanceOnAxis.unknown_label}：{frequencies} Hz" if bands_hz else ""


def furniture_ranking_note(categories: tuple[str, ...]) -> str:
    """第 13 條第 72、77 行接類別主詞；B3 只提示，不改排名。"""
    return (f"{FURNITURE_MODEL_NOTE}；{'、'.join(categories)}以近似模型參與第二階段第一版的擺位排名"
            if categories else "")


LOW_FREQUENCY_DECAY_NOTE = "低頻拖尾不計分，另有模態診斷報告"
SPATIAL_IMPRESSION_NOTE = "空間感：尚未評估"
REVERBERATION_ROOM_NOTE = "殘響是整間房的統計量，換座位不變"
BASELINE_NOTE = "尚未正式校準"
MODAL_STATE_TEXT = {"diagnosed_not_scored": "已診斷不計分", "not_computed": "未計算",
                    "failed": "失敗", "out_of_scope": "範圍外"}
MODAL_PLACEMENT_NOTE = "相對 dB：相對同一喇叭到同一座位最強的共振。前幾名只是版面長度，不是品質門檻；完整排序與整群大小剖面可展開。"
MODAL_GROUP_NOTE = ("重疊分組：相鄰共振的峰寬互相重疊就連成一組；成員數超過 1 的組不是只有一個共振，"
                    "而是一串互相重疊的共振；峰寬跟誰都不重疊的單獨共振自成一組（成員數 1）。"
                    "組內每個共振仍在模態表逐列列出")
DIRECTIONS = {"front": ("前", "主位前方"), "back": ("後", "主位後方"),
              "left": ("左", "主位左方"), "right": ("右", "主位右方"),
              "up": ("上", "主位上方"), "down": ("下", "主位下方")}
SPEAKERS = {"left": "左聲道喇叭", "right": "右聲道喇叭"}
LISTENING_POINTS = {"main": "主位", **{key: full for key, (_, full) in DIRECTIONS.items()}}


def modal_reason(diagnosis: ModalDiagnosis) -> str:
    return diagnosis.reason_text or {"unsupported_room": "房型不是長方形",
        "unsupported_impedance": "六面阻抗必須齊全、為有限正實數"}.get(diagnosis.reason_code or "", diagnosis.reason_code or "")


def modal_reference_text(excess: float | None) -> str:
    return "參考線範圍外" if excess is None else "超過" if excess > 0 else "未超過"


def speaker_label(channel_id: str) -> str:
    return SPEAKERS.get(channel_id, channel_id)


def listening_point_label(point_id: str) -> str:
    return LISTENING_POINTS.get(point_id, point_id)


def modal_guarantee_text(summary: CheckSummary) -> str:
    if summary.guaranteed_decay_rate_rad_s == 0:
        return "沒有保證（保證找齊的衰減高度為零，沒有 T60 門檻）"
    return (f"保證找齊的衰減高度：{summary.guaranteed_decay_rate_rad_s:.6g} rad/s（弧度／秒），"
            f"也就是 T60 不短於 {summary.guaranteed_min_t60_s:.6g} 秒的共振都保證找齊"
            "（這是門檻，不是找到的共振裡最短的 T60）")


def modal_check_text(summary: CheckSummary) -> str:
    weyl = ("Weyl 估計（幾何上該有幾個模態）只算體積項與表面積項，不含稜邊項"
            if summary.weyl_terms == FemModalCheck.__dataclass_fields__["weyl_terms"].default
            else f"求解器自報的 Weyl 估計公式：{summary.weyl_terms}")
    return (f"{weyl}；靜態 {summary.static_count}、零根延續 {summary.zero_mode_continuation_count}、"
            f"過阻尼 {summary.overdamped_count}、未確認衰減 {summary.unconfirmed_decay_count}、上限外返回 {summary.returned_above_limit_count}。"
            "保證以求解收斂且最近根選取完整為前提，只在開圓內成立；強阻尼完備性仍有未驗限制。")


def level_db(energy: float) -> float | None:
    """相對能量轉 dB；非正值沒有可顯示的對數。"""
    return 10.0 * math.log10(energy) if energy > 0 else None


def impedance_multiple(impedance: float, density: float, sound_speed: float) -> float | None:
    """牆阻抗相對於本方案 ρc 的顯示倍數。"""
    if impedance <= 0 or density <= 0 or sound_speed <= 0:
        return None
    return impedance / (density * sound_speed)
