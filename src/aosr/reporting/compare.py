"""把相容方案交給既有排名器並保留不相容分表。"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from aosr.config.quality_targets import QualityTargets
from aosr.scoring.ranking import rank_candidates
from aosr.scoring.ranking_models import ComparisonIdentity, RankingContext, RankingResult
from aosr.reporting.result import SchemeResult


def comparison_problems(results: Sequence[SchemeResult]) -> tuple[str, ...]:
    """列出所有固定身分不符，讓每個入口給同一份拒收理由。"""
    if not results:
        return ("比較至少需要一份方案結果",)
    first = results[0]
    seen: dict[str, int] = {}
    problems: list[str] = []
    for index, result in enumerate(results, start=1):
        name = f"第 {index} 份 {result.scheme.scheme_id}"
        if result.scheme.scheme_id in seen:
            problems.append(f"{name} 的候選代號重複："
                            f"第 {seen[result.scheme.scheme_id]} 份 {result.scheme.scheme_id}／{name}")
        seen[result.scheme.scheme_id] = index
        if result.scheme.purpose != first.scheme.purpose:
            problems.append(f"{name} 的 purpose 不同：{first.scheme.purpose}／{result.scheme.purpose}")
        if result.scheme.channel_group.fingerprint != first.scheme.channel_group.fingerprint:
            problems.append(f"{name} 的聲道組指紋不同："
                            f"{first.scheme.channel_group.fingerprint[:7]}／"
                            f"{result.scheme.channel_group.fingerprint[:7]}")
        if result.engine_commit != first.engine_commit:
            problems.append(f"{name} 的 engine_commit 不同："
                            f"{first.engine_commit[:7]}／{result.engine_commit[:7]}")
    return tuple(problems)


def identity_difference(ranking: RankingResult, identity: Sequence[ComparisonIdentity]) -> str:
    """不能同表的原因：主表有、這一份沒有；這一份有、主表沒有；同一類但身分不同。"""
    main = {part.category: part for part in ranking.header.main_table_identity}
    mine = {part.category: part for part in identity}
    groups = (
        ("少了", [category for category in main if category not in mine]),
        ("多了", [category for category in mine if category not in main]),
        ("同一類但身分不同", [category for category in mine
                             if category in main and mine[category] != main[category]]),
    )
    detail = "；".join(f"{label} {','.join(category.value for category in categories)}"
                      for label, categories in groups if categories)
    return "與主表的比較身分不同" + (f"：{detail}" if detail else "")


def compare_results(results: Sequence[SchemeResult], *, quality_targets: QualityTargets,
                    run_date: date) -> RankingResult:
    """先核固定身分；主表仍由排名層依比較身分決定。"""
    problems = comparison_problems(results)
    if problems:
        raise ValueError("；".join(problems))
    first = results[0]
    context = RankingContext(
        purpose=first.scheme.purpose,
        receiver_set_fingerprint=first.scheme.receiver_set.fingerprint,
        channel_group_fingerprint=first.scheme.channel_group.fingerprint,
        run_date=run_date, engine_version=first.engine_commit,
    )
    return rank_candidates([result.candidate for result in results], quality_targets, context)
