"""主對話判斷：篩選分數只由既有排名層產生，原方案釘住比較身分。

一個候選的評估包，用 rank_candidates（排名入口）單獨排一次：被淘汰回 eliminated（淘汰），
未評估（缺類）回 unassessed（未評估）；可排名但比較身分跟釘住的不同回 incomparable（不能同表），
相同才回 Scored（有分數）的總代價。不填假數字、不偏愛缺資料的候選，不改排名層。
"""

from datetime import date

from aosr.config.quality_targets import QualityTargets
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity, RankableRow, RankingContext, rank_candidates
from aosr.search.sampler import Excluded, Outcome, RankingZone, Scored


def comparison_identity(row: RankableRow) -> tuple[ComparisonIdentity, ...]:
    """跟排名層相同：逐類 identity（比較身分）照類名排序。"""
    return tuple(sorted((line.identity for line in row.categories), key=lambda item: item.category.value))


def screening_outcome(candidate: CandidateEvaluation, scheme: Scheme, *, registry: QualityTargets,
                      run_date: date, engine_version: str,
                      pinned: tuple[ComparisonIdentity, ...] | None
                      ) -> tuple[Outcome, tuple[ComparisonIdentity, ...] | None]:
    """主對話判斷：淘汰、未評估、不能同表都沒有假數字；可排名才回身分。"""
    context = RankingContext(purpose=scheme.purpose,
                             receiver_set_fingerprint=scheme.receiver_set.fingerprint,
                             channel_group_fingerprint=scheme.channel_group.fingerprint,
                             run_date=run_date, engine_version=engine_version)
    ranking = rank_candidates([candidate], registry, context)
    if ranking.eliminated:
        return Excluded(RankingZone.ELIMINATED), None
    if ranking.not_evaluated:
        return Excluded(RankingZone.UNASSESSED), None
    if not ranking.rankable:
        return Excluded(RankingZone.INCOMPARABLE), None
    row, = ranking.rankable
    identity = comparison_identity(row)
    if pinned is not None and identity != pinned:
        return Excluded(RankingZone.INCOMPARABLE), None
    return Scored(row.total_cost), identity
