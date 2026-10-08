"""搜尋報告與網頁共用的中文對照；報告既有文字不變。"""

from aosr.search.sampler import RankingZone

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
    "outside_speaker_area": "喇叭超出可用區", "listening_distance_out_of_range": "聆聽距離超出範圍",
    # 「不符合擺位要求」是家具決策紙第 12 條原文；冒號後半句與家具擺放錯整句是主對話定（#559 第七支第一步），老闆可改。
    "direct_path_blocked": "不符合擺位要求：直達路徑被家具擋住",
    "furniture_placement_invalid": "家具擺放不合法：超出房間、間隙不足，或喇叭、座位在家具裡",
    # 三個區名的鍵拿排名區的列舉值，不另打字串（#615 就是鍵對不上把區數全算成未辨識）。
    "base_angle_out_of_range": "水平夾角超出範圍", RankingZone.ELIMINATED.value: "淘汰",
    RankingZone.UNASSESSED.value: "未評估", RankingZone.INCOMPARABLE.value: "不能同表",
}


def counts_text(counts: dict[str, int]) -> str:
    """原因或區的數量：不認得的保留代碼，不能冒充已辨識。"""
    return "、".join(f"{COUNT_REASONS.get(key, f'未辨識原因（{key}）')}：{value}"
                    for key, value in counts.items()) or "沒有"
