"""從各類共同接點收齊排名列的複核警戒（review alert）。"""

from __future__ import annotations

from aosr.config.quality_targets import QualityPurpose
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import CategoryEvaluation
from aosr.scoring.review_alert import FlutterReviewAlert, ListeningAreaReviewAlert, ReviewAlert


def _alert_key(item: ReviewAlert) -> tuple[str, str, str, str, str, float]:
    """峰谷用喇叭與接收點；顫動用牆對；聆聽區用角色、喇叭、量、比較組排序。"""
    if isinstance(item, FlutterReviewAlert):
        return (item.category.value, item.walls[0], item.walls[1], "", "", item.center_frequency_hz)
    if isinstance(item, ListeningAreaReviewAlert):
        return (item.category.value, item.role, item.speaker_id, item.metric,
                item.group, item.deviation)
    return (item.category.value, item.speaker_id, item.receiver_id, "", "",
            item.center_frequency_hz)


def collect_review_alerts(
    evaluations: tuple[CategoryEvaluation, ...], purpose: QualityPurpose
) -> tuple[ReviewAlert, ...]:
    """收齊警戒並穩定排序：峰谷依喇叭與接收點，顫動依牆對，聆聽區依角色、量與比較組。"""
    alerts = (
        alert
        for evaluation in evaluations
        if (registration := CATEGORY_REGISTRY.get(evaluation.category)) is not None
        for alert in registration.review_alerts(evaluation, purpose)
    )
    return tuple(
        sorted(
            alerts,
            key=_alert_key,
        )
    )
