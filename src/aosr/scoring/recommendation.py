"""排名第幾、外部驗收、複核狀態、能不能當最終推薦——四件事分開記（票 #449）。

名次是代價排出來的；外部驗收（可施工、預算、噪音、失真）只由外部底線決定，複核警戒不改它的意思；
複核狀態說還有沒有事情沒確認完；推薦狀態說這一列能不能當最終推薦、不能的話為什麼。
兩個狀態都是受控字串不是布林值：以後加「已解除」之類的狀態不必重做契約。
今天沒有任何機制能把一列升成最終推薦（誰可以解除警戒、要什麼證據另題），所以推薦狀態只有「非最終」一種。
"""

from __future__ import annotations

from enum import StrEnum

from aosr.config.quality_targets import EntryStatus
from aosr.scoring.review_alert import ReviewAlert


class ReviewStatus(StrEnum):
    """有沒有產生複核警戒：有未解除的警戒是 pending；沒有是 clear。

    clear 只表示「目前沒有產生複核警戒」，不表示各類都查完了：例如反射顫動比較不可估的帶記在那一類的
    未評估帶，同一列照樣可能是 clear。對人要說「目前沒有產生複核警戒；另有若干頻帶未評估」，不能說全部通過（#480）。
    """

    CLEAR = "clear"
    PENDING = "pending"


class RecommendationStatus(StrEnum):
    """能不能當最終推薦；今天只有「非最終」，升級的機制另題。"""

    NOT_FINAL = "not_final"


class NotFinalReason(StrEnum):
    """一列不是最終推薦的受控原因；一列可以同時有好幾個。"""

    REVIEW_PENDING = "review_pending"
    EXTERNAL_NOT_CHECKED = "external_not_checked"
    CALIBRATION_BASELINE = "calibration_baseline"
    NO_FINALIZING_PROCESS = "no_finalizing_process"


def review_status(alerts: tuple[ReviewAlert, ...]) -> ReviewStatus:
    """有任何一條警戒就是待複核；今天沒有解除機制，所以警戒一律算未解除。"""
    return ReviewStatus.PENDING if alerts else ReviewStatus.CLEAR


def not_final_reasons(
    alerts: tuple[ReviewAlert, ...], external_checked: bool, calibration: EntryStatus
) -> tuple[NotFinalReason, ...]:
    """依固定順序列出這一列不能當最終推薦的全部原因；最後一條永遠在，因為升級的機制還沒有。"""
    reasons: list[NotFinalReason] = []
    if review_status(alerts) is ReviewStatus.PENDING:
        reasons.append(NotFinalReason.REVIEW_PENDING)
    if not external_checked:
        reasons.append(NotFinalReason.EXTERNAL_NOT_CHECKED)
    if calibration == "baseline":
        reasons.append(NotFinalReason.CALIBRATION_BASELINE)
    reasons.append(NotFinalReason.NO_FINALIZING_PROCESS)
    return tuple(reasons)
