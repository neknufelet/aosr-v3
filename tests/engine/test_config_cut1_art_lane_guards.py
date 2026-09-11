"""``aosr.config.art_lane`` 兩支護欄函式的裁判。

v2 **沒有任何考卷呼叫過** ``guard_art_patch_count``（只有生產碼用到它），
``guard_polygon_art_patch_count`` 只在 ``test_art_polygon`` 裡被驗到一條訊息。
所以這一支是**補的新考卷**：把邊界（負值／正常／剛好在上限／跨過上限）與 ``context``
那一格一起餵進去，再拿答案檔比對——答案檔那一邊是 v2 真的跑出來的。

每一筆要跑哪一組輸入由 case 表（``blueprint/config_cut1_cases.py``）宣告，這一支照 id
去答案檔拿那一筆。判定與殘餘風險見 ``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。
"""
from __future__ import annotations

from typing import TypedDict, cast

import pytest

import aosr.config.art_lane as art_lane

from tests.engine._config_answers import (
    case_args,
    constant,
    expected_block,
    is_approx,
    probe_by_id,
    probe_raised,
    probe_value,
    probe_ids,
)


class GuardArgs(TypedDict, total=False):
    """兩支護欄的參數（``context`` 有預設，所以是 ``total=False``）。"""

    n_per_wall: int
    n_tris: int
    context: str


def _ids(*, polygon: bool) -> list[str]:
    """答案檔裡屬於這一支函式的 case id（case id 的第三段就是函式名）。"""
    fn = "guard_polygon_art_patch_count" if polygon else "guard_art_patch_count"
    return sorted(case_id for case_id in probe_ids("art_lane") if f".{fn}." in case_id)


def _run(fn: object, case_id: str) -> None:
    """一筆 case：donor 說炸就炸（那條守門訊息也要一樣），說沒事就要真的沒事。"""
    case = probe_by_id(case_id)
    expected = expected_block(case)
    args = cast(GuardArgs, case_args(case))
    raised = probe_raised(expected)
    if raised is not None:
        with pytest.raises(ValueError) as caught:
            fn(**args)  # type: ignore[operator]  # expires=2026-12-08 reason=fn 是執行期才知道的函式物件；這一支要測的就是它
        assert str(caught.value) == raised["message"], "訊息跟 donor 不一樣——守門的說法也是行為"
    else:
        assert probe_value(expected) is None
        fn(**args)  # type: ignore[operator]  # expires=2026-12-08 reason=同上


def test_patch_count_guard_matches_donor_on_every_declared_case() -> None:
    """``guard_art_patch_count``：case 表宣告的每一筆都跟 donor 一樣。"""
    ids = _ids(polygon=False)
    assert ids, "答案檔裡沒有 guard_art_patch_count 的 case——這一支就沒有對象"
    for case_id in ids:
        _run(art_lane.guard_art_patch_count, case_id)


def test_polygon_patch_count_guard_matches_donor_on_every_declared_case() -> None:
    """``guard_polygon_art_patch_count``：同上，三角形數那一支。"""
    ids = _ids(polygon=True)
    assert ids, "答案檔裡沒有 guard_polygon_art_patch_count 的 case——這一支就沒有對象"
    for case_id in ids:
        _run(art_lane.guard_polygon_art_patch_count, case_id)


def test_patch_count_boundary_is_the_donor_one() -> None:
    """上限那一格是「剛好塞得下」與「跨過去」的交界。

    ``ART_P_CAP`` 對 ``6*n*n``：36 剛好塞得下、37 就跨過去。這一條拿模組自己的常數算，
    所以常數或邊界換了都會紅。
    """
    assert is_approx(art_lane.ART_P_CAP, constant("art_lane", "ART_P_CAP"))
    assert 6 * 36 * 36 <= art_lane.ART_P_CAP
    assert 6 * 37 * 37 > art_lane.ART_P_CAP
