"""報告品質段的尺校準進度：只數這次搜尋快照裡登記簿每一條尺的狀態，不判任何候選過不過。

#585 第 3 題（老闆轉貼外部覆核、主對話判斷）：整體合格規則等 #358 把門檻回原始文獻查證、升 calibrated 之後再定；
在那之前報告把知道的與不知道的分開列。「某一類的尺全部校準完」要連它依賴的別類尺一起算（照排名層
category_registry 登記的 registry_sources），方向分區、排名規則是判定共用的尺，不當成會判過不過的品質類（複查）。
"""

from pydantic import BaseModel

from aosr.config.quality_targets import QualityPurpose
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.search.store import FROZEN

# 跟網頁結果頁（src/aosr/gui/result_view.py 的 LABELS）同一套叫法；考卷核兩邊一致。
CATEGORY_LABELS = {
    "timbre_balance": "音色平衡", "channel_matching": "聲道匹配", "reverberation": "殘響",
    "reflections_and_echo": "反射與回聲", "listening_area_stability": "聆聽區穩定性",
    "direction_zones": "方向分區", "ranking": "排名規則",
}


class CategoryProgress(BaseModel):
    model_config = FROZEN
    category: str
    calibrated: int
    total: int
    judged: bool
    uses_all_calibrated: bool = False


class CalibrationProgress(BaseModel):
    model_config = FROZEN
    calibrated: int
    total: int
    categories: tuple[CategoryProgress, ...]
    dependency_error: str | None = None


def _statuses(purpose: QualityPurpose) -> list[tuple[str, str]]:
    """每一條帶狀態的尺（設定、目標、權重表裡的每一項、資格規則）；權重項用它那張表的鍵。"""
    entries = (*purpose.setting, *purpose.target, *purpose.qualification)
    keyed: list[tuple[str, str]] = [(entry.key, entry.status) for entry in entries]
    keyed += [(table.key, item.status) for table in purpose.weight for item in table.item]
    return keyed


def calibration_progress(purpose: QualityPurpose) -> CalibrationProgress:
    """各類取鍵名第一段數自己的尺；品質類另看「自己的尺＋排名層登記的依賴」是不是全部校準。"""
    keyed = _statuses(purpose)
    counts: dict[str, tuple[int, int]] = {}
    for key, status in keyed:
        category = key.split(".", 1)[0]
        calibrated, total = counts.get(category, (0, 0))
        counts[category] = (calibrated + (status == "calibrated"), total + 1)
    judged = {category.value: registration for category, registration in CATEGORY_REGISTRY.items()}
    known = [name for name in CATEGORY_LABELS if name in counts]
    order = known + sorted(name for name in counts if name not in CATEGORY_LABELS)
    categories = []
    dependency_error: str | None = None
    for name in order:
        uses_all = False
        if name in judged and dependency_error is None:
            own = [status for key, status in keyed if key.split(".", 1)[0] == name]
            try:
                depends = [status for _, status in judged[name].registry_sources(purpose)]
            except (KeyError, TypeError, ValueError) as error:
                # 快照是舊版登記簿、現在的排名層要讀它沒有的尺：照實說判不出，不讓整份報告失敗（複查）。
                dependency_error = str(error)
                depends = []
            uses_all = dependency_error is None and all(status == "calibrated" for status in (*own, *depends))
        categories.append(CategoryProgress(category=name, calibrated=counts[name][0], total=counts[name][1],
                                           judged=name in judged, uses_all_calibrated=uses_all))
    if dependency_error is not None:
        categories = [item.model_copy(update={"uses_all_calibrated": False}) for item in categories]
    return CalibrationProgress(calibrated=sum(item.calibrated for item in categories),
                               total=sum(item.total for item in categories), categories=tuple(categories),
                               dependency_error=dependency_error)


def calibration_lines(progress: CalibrationProgress) -> tuple[str, ...]:
    def label(item: CategoryProgress) -> str:
        return f"{CATEGORY_LABELS.get(item.category, item.category)} {item.calibrated}／{item.total}"

    judged = [item for item in progress.categories if item.judged]
    shared = [item for item in progress.categories if not item.judged]
    breakdown = "各品質類已校準／共：" + "、".join(label(item) for item in judged)
    if shared:
        breakdown += "；判定共用的尺：" + "、".join(label(item) for item in shared)
    full = [CATEGORY_LABELS.get(item.category, item.category) for item in judged if item.uses_all_calibrated]
    closing = (f"這次搜尋快照裡的登記簿跟現在的排名層對不上（{progress.dependency_error}），判不出有沒有一類用到的尺全部校準完"
               if progress.dependency_error is not None else
               "沒有任何一類用到的尺（含它依賴的別類尺，照排名層登記）全部校準完，所以還沒有「已校準項目通過幾項」可以報（等 #358）"
               if not full else
               "用到的尺（含依賴，照排名層登記）全部校準完的類別：" + "、".join(full) + "；這幾類過不過還沒接進報告（等 #358）")
    return (f"尺的校準進度（這次搜尋快照裡的登記簿）：共 {progress.total} 條，已校準 {progress.calibrated} 條、"
            f"未校準 {progress.total - progress.calibrated} 條", breakdown, closing)
