"""聲道匹配評估契約的列唯一性考卷（票 #410）：重複列、漏列、歸類不符一律拒收；列的順序不是身分。

樣本與輔助函式沿用 ``test_channel_matching`` 那一支；從那裡搬出來是因為那一支檔逼近行數上限。
"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest
from pydantic import ValidationError

from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.channel_matching import ChannelComparison, ChannelGroup
from aosr.scoring.channel_matching_cost import (
    channel_matching_floor_reasons,
    cost_channel_matching_evaluation,
)
from aosr.scoring.contract import CategoryEvaluation, ChannelMatchingPayload
from tests.engine.test_channel_matching import (
    _PURPOSE,
    _TARGETS,
    _channel_points,
    _evaluate,
    _group,
    _payload,
    _receivers,
)


def _two_pair_evaluation() -> CategoryEvaluation:
    receivers = _receivers()
    base = _group(include_center=True)
    group = ChannelGroup(
        channels=base.channels,
        comparisons=(
            *base.comparisons,
            ChannelComparison(left_role="right", right_role="center"),
        ),
        feature_match_tolerance_hz=base.feature_match_tolerance_hz,
    )
    return _evaluate(receivers, group, _channel_points(receivers, group))


def _costed(evaluation: CategoryEvaluation) -> CategoryEvaluation:
    registry = load_quality_targets(_TARGETS)
    return cost_channel_matching_evaluation(
        evaluation,
        registry.purpose(_PURPOSE),
        registry.fingerprint,
    )


def _broadband_mean(aggregate: dict[str, object]) -> float:
    summary = cast(dict[str, object], aggregate["broadband_level_difference"])
    return cast(float, summary["weighted_mean_absolute_difference"])


def test_contract_rejects_duplicate_aggregate_for_the_better_comparison() -> None:
    """重讀時重複較好比較對，不得讓逐列平均把它的權重偷偷加倍。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    aggregates = cast(tuple[dict[str, object], ...], document["aggregates"])
    better = min(aggregates, key=_broadband_mean)
    left = cast(str, better["left_role"])
    right = cast(str, better["right_role"])
    document["aggregates"] = (*aggregates, deepcopy(better))

    with pytest.raises(ValidationError, match=rf"{left}/{right}.*彙總列"):
        ChannelMatchingPayload.model_validate(document)


def test_contract_rejects_duplicate_receiver_and_comparison_result() -> None:
    """同一接收點與比較對只能有一列，否則重讀後的彙總身分不唯一。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    results = cast(tuple[dict[str, object], ...], document["point_results"])
    repeated = results[0]
    receiver = cast(str, repeated["receiver_id"])
    left = cast(str, repeated["left_role"])
    right = cast(str, repeated["right_role"])
    document["point_results"] = (*results, deepcopy(repeated))

    with pytest.raises(
        ValidationError,
        match=rf"{receiver}.*{left}/{right}.*逐點結果",
    ):
        ChannelMatchingPayload.model_validate(document)


@pytest.mark.parametrize(
    "receiver_list",
    ("assessed_receiver_ids", "unavailable_receiver_ids"),
)
def test_contract_rejects_duplicate_receiver_inside_each_aggregate_list(
    receiver_list: str,
) -> None:
    """已評與不可估清單各自都不能用重複接收點偽造列數。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    aggregates = list(cast(tuple[dict[str, object], ...], document["aggregates"]))
    changed = deepcopy(aggregates[0])
    assessed = cast(tuple[str, ...], changed["assessed_receiver_ids"])
    unavailable = cast(tuple[str, ...], changed["unavailable_receiver_ids"])
    receiver = assessed[0]
    if receiver_list == "assessed_receiver_ids":
        changed[receiver_list] = (*assessed, receiver)
    else:
        changed["assessed_receiver_ids"] = tuple(
            item for item in assessed if item != receiver
        )
        changed[receiver_list] = (*unavailable, receiver, receiver)
    aggregates[0] = changed
    document["aggregates"] = tuple(aggregates)
    left = cast(str, changed["left_role"])
    right = cast(str, changed["right_role"])

    with pytest.raises(
        ValidationError,
        match=rf"{left}/{right}.*{receiver_list}.*{receiver}",
    ):
        ChannelMatchingPayload.model_validate(document)


def test_contract_rejects_missing_comparison_result_for_a_receiver() -> None:
    """接收點仍存在於另一比較對時，不能漏掉其中一對的逐點列。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    results = cast(tuple[dict[str, object], ...], document["point_results"])
    missing = results[0]
    receiver = cast(str, missing["receiver_id"])
    left = cast(str, missing["left_role"])
    right = cast(str, missing["right_role"])
    document["point_results"] = results[1:]
    aggregates = list(cast(tuple[dict[str, object], ...], document["aggregates"]))
    for aggregate in aggregates:
        if (
            aggregate["left_role"] == left
            and aggregate["right_role"] == right
        ):
            assessed = cast(tuple[str, ...], aggregate["assessed_receiver_ids"])
            aggregate["assessed_receiver_ids"] = tuple(
                item for item in assessed if item != receiver
            )
    document["aggregates"] = tuple(aggregates)

    with pytest.raises(
        ValidationError,
        match=rf"{receiver}.*{left}/{right}.*缺逐點結果",
    ):
        ChannelMatchingPayload.model_validate(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("overlap", "同時列為已評與不可估"),
        ("missing", "漏掉逐點結果"),
        ("phantom", "沒有逐點結果"),
        ("misfiled", "歸類跟逐點結果不符"),
    ),
)
def test_contract_rejects_aggregate_receiver_membership_errors(
    mutation: str, message: str
) -> None:
    """彙總的接收點分類若重疊或漏列，不能再代表那一對的逐點結果。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    aggregates = list(cast(tuple[dict[str, object], ...], document["aggregates"]))
    changed = deepcopy(aggregates[0])
    assessed = cast(tuple[str, ...], changed["assessed_receiver_ids"])
    receiver = assessed[0]
    left = cast(str, changed["left_role"])
    right = cast(str, changed["right_role"])
    unavailable = cast(tuple[str, ...], changed["unavailable_receiver_ids"])
    if mutation == "overlap":
        changed["unavailable_receiver_ids"] = (*unavailable, receiver)
    elif mutation == "phantom":
        receiver = "receiver-without-results"
        changed["unavailable_receiver_ids"] = (*unavailable, receiver)
    elif mutation == "misfiled":
        changed["assessed_receiver_ids"] = tuple(
            item for item in assessed if item != receiver
        )
        changed["unavailable_receiver_ids"] = (*unavailable, receiver)
    else:
        changed["assessed_receiver_ids"] = tuple(
            item for item in assessed if item != receiver
        )
    aggregates[0] = changed
    document["aggregates"] = tuple(aggregates)

    with pytest.raises(
        ValidationError,
        match=rf"{left}/{right}.*{receiver}.*{message}",
    ):
        ChannelMatchingPayload.model_validate(document)


def test_contract_and_cost_are_independent_of_result_row_order() -> None:
    """彙總與逐點列換序後，重讀、類代價及淘汰原因都必須相同。"""
    evaluation = _two_pair_evaluation()
    document = evaluation.model_dump(mode="python")
    payload = cast(dict[str, object], document["payload"])
    aggregates = cast(tuple[dict[str, object], ...], payload["aggregates"])
    results = cast(tuple[dict[str, object], ...], payload["point_results"])
    payload["aggregates"] = tuple(reversed(aggregates))
    payload["point_results"] = tuple(reversed(results))
    reordered = CategoryEvaluation.model_validate(document)

    registry = load_quality_targets(_TARGETS)
    purpose = registry.purpose(_PURPOSE)
    original_costed = _costed(evaluation)
    reordered_costed = _costed(reordered)

    assert original_costed.category_cost is not None
    assert reordered_costed.category_cost is not None
    assert reordered_costed.category_cost.value == pytest.approx(
        original_costed.category_cost.value
    )
    assert channel_matching_floor_reasons(
        reordered_costed, purpose
    ) == channel_matching_floor_reasons(original_costed, purpose)
