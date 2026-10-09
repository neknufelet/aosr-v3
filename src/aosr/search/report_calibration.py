"""報告品質段的尺校準進度：只數這次搜尋快照裡登記簿每一條尺的狀態，不判任何候選過不過。

#585 第 3 題（老闆轉貼外部覆核、主對話判斷）：整體合格規則等 #358 把門檻回原始文獻查證、升 calibrated 之後再定；
在那之前報告把知道的與不知道的分開列。「某一類的尺全部校準完」要連它依賴的別類尺一起算：排名層讀的
（category_registry 登記的 registry_sources）與評估器那一層讀的（evaluator_keys，含吃進來的上游評估——
例如聆聽區與聲道匹配吃逐座位音色——與編排層代讀交進來的，#633）。算不算「用到」：值改了、這一類的評估輸出
（數字、狀態、旗標、診斷欄位）就可能不同就算；整份登記簿的指紋不算。這是保守方向，多算只會晚一點說「全部校準」。
方向分區、排名規則是判定共用的尺，不當成會判過不過的品質類（複查）。低頻拖尾這類登記簿已有尺、還沒接進評分的品質類
另列一組，不跟共用的尺混在一起（#669）。
"""

from pydantic import BaseModel

from aosr.config.quality_targets import EntryStatus, QualityPurpose, WeightTable
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import QualityCategory
from aosr.search.store import FROZEN
from aosr.reporting.display import LABELS

# 跟網頁結果頁（src/aosr/gui/result_view.py 的 LABELS）同一套叫法；考卷核兩邊一致。
CATEGORY_LABELS = {key: LABELS[key] for key in (
    "timbre_balance", "channel_matching", "reverberation", "reflections_and_echo",
    "listening_area_stability", "low_frequency_decay", "direction_zones", "ranking",
)}


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


def _key_statuses(purpose: QualityPurpose, key: str) -> list[EntryStatus]:
    """一條尺的狀態；宣告指到一整張權重表時照排名層的寫法逐項算（每一項各有狀態，複查）。"""
    entry = purpose.entry(key)
    return [item.status for item in entry.item] if isinstance(entry, WeightTable) else [entry.status]


def calibration_progress(purpose: QualityPurpose) -> CalibrationProgress:
    """各類取鍵名第一段數自己的尺；品質類另看「自己的尺＋排名層與評估器讀的依賴」是不是全部校準。"""
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
                depends += [status for key in judged[name].evaluator_keys for status in _key_statuses(purpose, key)]
            except (KeyError, TypeError, ValueError) as error:
                # 快照是舊版登記簿、現在的評估器或排名層要讀它沒有的尺：照實說判不出，不讓整份報告失敗（複查）。
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

    quality = {category.value for category in QualityCategory}
    judged = [item for item in progress.categories if item.judged]
    pending = [item for item in progress.categories if not item.judged and item.category in quality]
    shared = [item for item in progress.categories if not item.judged and item.category not in quality]
    breakdown = "各品質類已校準／共：" + "、".join(label(item) for item in judged)
    if pending:
        breakdown += "；還沒接進評分的品質類：" + "、".join(label(item) for item in pending)
    if shared:
        breakdown += "；判定共用的尺：" + "、".join(label(item) for item in shared)
    full = [CATEGORY_LABELS.get(item.category, item.category) for item in judged if item.uses_all_calibrated]
    closing = (f"這次搜尋快照裡的登記簿跟現在的評分程式（評估器與排名層）對不上（{progress.dependency_error}），"
               "判不出有沒有一類用到的尺全部校準完"
               if progress.dependency_error is not None else
               "沒有任何一類用到的尺（含評估器與排名層讀到的別類尺）全部校準完，所以還沒有「已校準項目通過幾項」可以報（等 #358）"
               if not full else
               "用到的尺（含評估器與排名層讀到的別類尺）全部校準完的類別：" + "、".join(full) + "；這幾類過不過還沒接進報告（等 #358）")
    return (f"尺的校準進度（這次搜尋快照裡的登記簿）：共 {progress.total} 條，已校準 {progress.calibrated} 條、"
            f"未校準 {progress.total - progress.calibrated} 條", breakdown, closing)
