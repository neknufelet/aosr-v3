"""報告品質段的尺校準進度：只數登記簿每一條尺的狀態，不判任何候選過不過。

#585 第 3 題（老闆轉貼外部覆核、主對話判斷）：整體合格規則等 #358 把門檻回原始文獻查證、升 calibrated 之後再定；
在那之前報告把知道的與不知道的分開列——共幾條、校準幾條、各類校準幾條，沒有任何一類全部校準完就照實說報不出「已校準項目通過幾項」。
"""

from pydantic import BaseModel

from aosr.config.quality_targets import QualityPurpose
from aosr.search.store import FROZEN

CATEGORY_LABELS = {
    "timbre_balance": "音色平衡", "channel_matching": "聲道一致", "reverberation": "殘響",
    "reflections_and_echo": "反射與回音", "direction_zones": "方向分區",
    "listening_area_stability": "聆聽區穩定", "ranking": "排名規則",
}


class CategoryProgress(BaseModel):
    model_config = FROZEN
    category: str
    calibrated: int
    total: int


class CalibrationProgress(BaseModel):
    model_config = FROZEN
    calibrated: int
    total: int
    categories: tuple[CategoryProgress, ...]


def calibration_progress(purpose: QualityPurpose) -> CalibrationProgress:
    """每一條帶狀態的尺都算（設定、目標、權重表裡的每一項、資格規則）；類別取鍵名第一段。"""
    keyed = [(entry.key, entry.status) for entry in (*purpose.setting, *purpose.target, *purpose.qualification)]
    keyed += [(table.key, item.status) for table in purpose.weight for item in table.item]
    counts: dict[str, tuple[int, int]] = {}
    for key, status in keyed:
        category = key.split(".", 1)[0]
        calibrated, total = counts.get(category, (0, 0))
        counts[category] = (calibrated + (status == "calibrated"), total + 1)
    known = [name for name in CATEGORY_LABELS if name in counts]
    order = known + sorted(name for name in counts if name not in CATEGORY_LABELS)
    categories = tuple(CategoryProgress(category=name, calibrated=counts[name][0], total=counts[name][1])
                       for name in order)
    return CalibrationProgress(calibrated=sum(item.calibrated for item in categories),
                               total=sum(item.total for item in categories), categories=categories)


def calibration_lines(progress: CalibrationProgress) -> tuple[str, ...]:
    labels = [(CATEGORY_LABELS.get(item.category, item.category), item) for item in progress.categories]
    full = [label for label, item in labels if item.calibrated == item.total]
    closing = ("沒有任何一類的尺全部校準完，所以還沒有「已校準項目通過幾項」可以報（等 #358）" if not full
               else "全部校準完的類別：" + "、".join(full) + "；這幾類過不過還沒接進報告（等 #358）")
    return (f"尺的校準進度：共 {progress.total} 條，已校準 {progress.calibrated} 條、未校準 {progress.total - progress.calibrated} 條",
            "各類已校準／共：" + "、".join(f"{label} {item.calibrated}／{item.total}" for label, item in labels),
            closing)
