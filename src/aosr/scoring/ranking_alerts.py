"""從各類共同接點收齊排名列的複核警戒（review alert）。"""

from __future__ import annotations

from aosr.config.quality_targets import QualityPurpose
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import CategoryEvaluation
from aosr.scoring.review_alert import ReviewAlert


def collect_review_alerts(
    evaluations: tuple[CategoryEvaluation, ...], purpose: QualityPurpose
) -> tuple[ReviewAlert, ...]:
    """收齊各類警戒，依類別、喇叭、接收點與中心頻率給穩定順序。"""
    alerts = (
        alert
        for evaluation in evaluations
        if (registration := CATEGORY_REGISTRY.get(evaluation.category)) is not None
        for alert in registration.review_alerts(evaluation, purpose)
    )
    return tuple(
        sorted(
            alerts,
            key=lambda item: (
                item.category.value,
                item.speaker_id,
                item.receiver_id,
                item.center_frequency_hz,
            ),
        )
    )
