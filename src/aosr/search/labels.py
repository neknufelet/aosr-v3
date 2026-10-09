"""搜尋報告與網頁共用的中文對照；報告既有文字不變。"""

from aosr.search.sampler import RankingZone
from aosr.reporting.scheme import SpeakerSetup
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.config.three_lane_crossover import CROSSOVER_LOWER_FLOOR_HZ
from aosr.reporting.display import (
    FURNITURE_FACES as FURNITURE_FACES, LISTENING_POINTS as LISTENING_POINTS, SPEAKERS as SPEAKERS,
)

# 喇叭設定的欄名與選項只有這一份；輸入頁、比較頁、搜尋頁與報告共用。
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


def speaker_setup_text(setup: SpeakerSetup) -> str:
    """主對話定的搜尋喇叭一行；尺寸只讀方案自己的五格。"""
    cabinet = setup.cabinet
    model = SPEAKER_SETUP["representative_model" if setup.representative else "actual_model"]
    return (f"喇叭：{SPEAKER_SETUP[setup.kind]}、放{SPEAKER_SETUP[setup.mount]}（{model}）；"
            f"箱體 寬 {cabinet.width_m} × 深 {cabinet.depth_m} × 高 {cabinet.height_m} m，"
            f"聲學中心離箱底 {cabinet.acoustic_center_above_bottom_m} m")

# 家具決策紙第 12 條原文；不算原方案，但 B4 搜尋照跑。
BASELINE_BLOCKED_TEXT = "原方案不符合擺位要求"
BASELINE_BLOCKED = "direct_path_blocked"

STABILITY_SHIFTS = {
    "speakers_forward": "喇叭向前", "speakers_backward": "喇叭向後",
    "speakers_outward": "喇叭各向外", "speakers_inward": "喇叭各向內",
    "seat_forward": "座位向前", "seat_backward": "座位向後",
    "seat_left": "座位向左", "seat_right": "座位向右",
    "ear_up": "耳高向上", "ear_down": "耳高向下",
    "acoustic_center_up": "聲學中心向上", "acoustic_center_down": "聲學中心向下",
}
STABILITY_OUTCOMES = {
    "unplaceable": "擺不出來", "placement_requirement_failed": "移位後不符合擺位要求",
    "excluded": "淘汰", "not_evaluated": "未評估", "not_comparable": "不能同表", "scored": "有分數",
    "ready": "可以算",
}
STABILITY_FLAGS = {"model_discontinuity": "模型不連續", "out_of_spec": "超出規格",
                   "outside_search": "超出搜尋範圍", "search_range_not_checked": "面向不同，未判搜尋範圍"}
STABILITY_EVENTS = {"appeared": "家具反射出現", "disappeared": "家具反射消失"}
STABILITY_CROSSOVERS = {"hard": f"{FEM_GEOMETRIC_CROSSOVER_CAP_HZ:g} Hz 硬切", "legacy": "上一代接法",
                       "wide": f"{CROSSOVER_LOWER_FLOOR_HZ:g}～{FEM_GEOMETRIC_CROSSOVER_CAP_HZ:g} Hz 平滑"}
IDENTICAL_CROSSOVER_REASON = "與正式接法逐點權重相同，不算另一種比較"
STABILITY_CROSSOVER_INCOMPLETE = "交接敏感度摘要未完成，各接法第一名不能用"
STABILITY_MINIMAX_INCOMPLETE = "最差情況不完整"
STABILITY_STATES = {"running": "進行中", "done": "完成", "skipped": "跳過", "failed": "失敗", "stopped": "停止"}
STABILITY_IDENTITIES = {"physics_identity": "物理身分", "program_fingerprint": "整支程式指紋",
                        "purpose_settings": "評分設定"}
PARAM_LABELS = {"front_distance": "喇叭離前牆", "spacing": "兩支喇叭間距", "listening_distance": "聆聽距離"}
STABILITY_LIMITATION = "只移單一個量（單軸），不證明整個 ±2 公分範圍都穩；±2 公分是穩定性測試範圍，不是施工公差；喇叭跟著移位後的主位重新對準"
STABILITY_MODEL_NOTE = "多半是幾何聲學近似造成的突變，不直接當成真實聲音的變化"


def trial_label(number: int | None) -> str:
    return "原方案" if number is None else f"試算 {number}"

SEARCH_STATES = {
    "running": "進行中", "converged": "達到停止條件", "budget_exhausted": "因預算停止",
    "user_stopped": "使用者停止", "failed": "失敗", "interrupted": "中斷",
}
REFINE_STATES = {
    "not_started": "未開始", "running": "進行中", "stopped": "已停",
    "failed": "失敗", "interrupted": "中斷",
}
REFINE_STOP_REASONS = {
    "stable": "細算第一名連續一段沒被換掉", "refine_budget": "用完細算上限",
    "candidates_exhausted": "沒有候選可以再細算", "user_stopped": "使用者停止",
}
COUNT_REASONS = {
    "cabinet_outside_room": "箱體越界", "wall_gap": "離牆間隙不足",
    "cabinets_overlap": "箱體重疊", "cabinet_in_keep_out": "箱體進入禁區",
    "seat_in_keep_out": "座位進入禁區", "seat_outside_room": "座位越界",
    "outside_speaker_area": "喇叭超出可用區", "listening_distance_out_of_range": "型號適用聆聽距離超出範圍",
    # 「不符合擺位要求」是家具決策紙第 12 條原文；冒號後半句與家具擺放錯整句是主對話定（#559 第七支第一步），老闆可改。
    "direct_path_blocked": "不符合擺位要求：直達路徑被家具擋住",
    "furniture_placement_invalid": "家具擺放不合法：超出房間、間隙不足，或喇叭、座位在家具裡",
    # 主對話定，老闆可改；腳架柱體是第 8 條「箱體下方保留空間」的主對話判讀。
    "cabinet_in_furniture": "箱體穿入家具", "cabinet_off_table": "箱體超出桌面",
    "stand_space_occupied": "腳架下方有家具",
    # 主對話定，老闆可改；共用搜尋紙二.1 延伸到跟著主位走的家具，固定天雲不判。
    "furniture_in_keep_out": "家具進入禁區",
    # 三個區名的鍵拿排名區的列舉值，不另打字串（#615 就是鍵對不上把區數全算成未辨識）。
    "base_angle_out_of_range": "水平夾角超出範圍", RankingZone.ELIMINATED.value: "淘汰",
    RankingZone.UNASSESSED.value: "未評估", RankingZone.INCOMPARABLE.value: "不能同表",
}


def counts_text(counts: dict[str, int]) -> str:
    """原因或區的數量：不認得的保留代碼，不能冒充已辨識。"""
    return "、".join(f"{COUNT_REASONS.get(key, f'未辨識原因（{key}）')}：{value}"
                    for key, value in counts.items()) or "沒有"
