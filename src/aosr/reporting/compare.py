"""把相容方案交給既有排名器並保留不相容分表。"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from aosr.config.quality_targets import QualityTargets
from aosr.scoring.ranking import rank_candidates
from aosr.scoring.ranking_models import RankingContext, RankingResult
from aosr.reporting.result import SchemeResult


def compare_results(results: Sequence[SchemeResult], *, quality_targets: QualityTargets,
                    run_date: date) -> RankingResult:
    """先核同表所需的固定身分，再讓排名器處理評估支撐與分表。"""
    if not results:
        raise ValueError("比較至少需要一份方案結果")
    first = results[0]
    seen: set[str] = set()
    for index, result in enumerate(results, start=1):
        name = f"第 {index} 份 {result.scheme.scheme_id}"
        if result.scheme.scheme_id in seen:
            raise ValueError(f"{name} 的候選代號重複")
        seen.add(result.scheme.scheme_id)
        if result.scheme.purpose != first.scheme.purpose:
            raise ValueError(f"{name} 的 purpose 不同")
        if result.scheme.channel_group.fingerprint != first.scheme.channel_group.fingerprint:
            raise ValueError(f"{name} 的聲道組指紋不同")
        if (result.listening_area_channel_role != first.listening_area_channel_role
                or result.listening_area_speaker_id != first.listening_area_speaker_id):
            raise ValueError(f"{name} 的聆聽區角色或喇叭代號不同")
        if result.engine_commit != first.engine_commit:
            raise ValueError(f"{name} 的 engine_commit 不同")
    context = RankingContext(
        purpose=first.scheme.purpose,
        receiver_set_fingerprint=first.scheme.receiver_set.fingerprint,
        channel_group_fingerprint=first.scheme.channel_group.fingerprint,
        run_date=run_date, engine_version=first.engine_commit,
    )
    return rank_candidates([result.candidate for result in results], quality_targets, context)
