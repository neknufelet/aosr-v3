"""搜尋釘主表用的公開比較身分入口，獨立於排名狀態。"""

from pathlib import Path

import pytest

from aosr.scoring import ranking
from tests.engine import _ranking_fixtures as fixtures
from tests.engine._search_review_cases import with_matching


def test_rankable_identity_equals_main_table_header(tmp_path: Path) -> None:
    """防身分另用一套排序或欄位；既有搜尋考卷只看篩選結果，沒有考這支公開入口。"""
    assert tmp_path.is_dir()
    candidate = fixtures._three()[0]
    registry = fixtures._registry()
    result = ranking.rank_candidates([candidate], registry, fixtures._CONTEXT)
    assert ranking.comparison_identity_of(candidate, registry, fixtures._CONTEXT) == result.header.main_table_identity


def test_eliminated_candidate_still_has_comparison_identity(tmp_path: Path) -> None:
    """防淘汰把身分一起丟掉；既有排名只要求淘汰區，不要求搜尋仍能釘住原方案。"""
    assert tmp_path.is_dir()
    candidate = fixtures._three()[0]
    safe, breached = (with_matching(candidate, breached=value) for value in (False, True))
    registry = fixtures._registry()
    assert ranking.rank_candidates([breached], registry, fixtures._CONTEXT).eliminated
    expected = ranking.rank_candidates([safe], registry, fixtures._CONTEXT).header.main_table_identity
    assert expected
    assert ranking.comparison_identity_of(breached, registry, fixtures._CONTEXT) == expected


@pytest.mark.parametrize("eliminated", (False, True))
def test_missing_category_has_no_comparison_identity(tmp_path: Path, eliminated: bool) -> None:
    """防缺類仍拼出不完整主表身分；既有缺類考卷只查排名區，沒查公開入口的 None。"""
    assert tmp_path.is_dir()
    candidate = fixtures._candidate(fixtures._reverberation("missing"))
    if eliminated:
        candidate = with_matching(candidate, breached=True)
    assert ranking.comparison_identity_of(candidate, fixtures._registry(), fixtures._CONTEXT) is None
