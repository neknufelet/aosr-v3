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
# 第 17 條拿掉中間的出處說明；第 10 條原文；第 11 條第 67 行接主詞。
FURNITURE_WOOD_CLOUD_NOTE = "數值借用木質桌面那組估計值，懸空的板直接借用是近似"
FURNITURE_SCATTERING_NOTE = "家具表面的粗糙散射第一版不算"
FURNITURE_FINITE_SIZE_NOTE = ("有限尺寸鏡面修正的限制：原理論的前提「家具尺寸遠小於距離」對桌面與沙發不成立；"
    "截止以上一律截在 1（精確式會略超過 1）；反射點靠板邊時會高估，最多約 6 dB；"
    "只乘實數、沒有相位與邊緣繞射路徑；靠牆或接靠背的非自由邊會低估；只收矩形面。")
FURNITURE_REVERBERATION_NOTE = "未包含家具吸音"
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
