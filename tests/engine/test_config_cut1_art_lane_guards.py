"""``aosr.config.art_lane`` 兩支護欄函式的裁判。

v2 **沒有任何考卷呼叫過** ``guard_art_patch_count``（只有生產碼用到它），
``guard_polygon_art_patch_count`` 只在 ``test_art_polygon`` 裡被驗到一條訊息。
所以這一支是**補的新考卷**：把邊界（負值／正常／剛好在上限／跨過上限）與 ``context``
那一格一起餵進去，再拿答案檔比對——答案檔那一邊是 v2 真的跑出來的。

判定與殘餘風險見 ``tests/engine/test_config_cut1_table``。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict, cast

import pytest

import aosr.config.art_lane as art_lane

from tests.engine._config_answers import (
    constant,
    is_approx,
    module_answers,
    probe_args,
    probe_raised,
    probe_value,
)


class GuardArgs(TypedDict, total=False):
    """兩支護欄的參數（``context`` 有預設，所以是 ``total=False``）。"""

    n_per_wall: int
    n_tris: int
    context: str


def _field(record: object, key: str) -> object:
    if not isinstance(record, dict) or key not in record:
        raise AssertionError(f"答案檔這一筆沒有 {key}：{record!r}")
    return record[key]


def _table(node: object, where: str) -> dict[str, object]:
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是表：{node!r}")
    return {str(name): item for name, item in node.items()}


def _records(*, polygon: bool) -> list[object]:
    key = "n_tris" if polygon else "n_per_wall"
    probes = module_answers("art_lane")["probes"]
    if not isinstance(probes, list):
        raise AssertionError("答案檔的 art_lane 探針不是一串東西")
    picked: list[object] = []
    for record in probes:
        args = _field(record, "args")
        if isinstance(args, dict) and key in args:
            picked.append(record)
    return picked


def _run(fn: Callable[..., object], record: object) -> None:
    """一筆探針：donor 說炸就炸（那條守門訊息也要一樣），說沒事就要真的沒事。"""
    expected = _table(_field(record, "expected"), "答案檔這一筆的 expected")
    args = cast(GuardArgs, probe_args(_table(record, "答案檔這一筆的探針")))
    raised = probe_raised(expected)
    if raised is not None:
        with pytest.raises(ValueError) as caught:
            fn(**args)
        assert str(caught.value) == raised["message"], "訊息跟 donor 不一樣——守門的說法也是行為"
    else:
        assert probe_value(expected) is None
        fn(**args)


def test_patch_count_guard_matches_donor_on_every_probed_input() -> None:
    """``guard_art_patch_count``：每一組輸入的結果（回 ``None`` 或炸）都跟 donor 一樣。"""
    records = _records(polygon=False)
    assert records, "答案檔裡沒有 guard_art_patch_count 的探針——這一支就沒有對象"
    for record in records:
        _run(art_lane.guard_art_patch_count, record)


def test_polygon_patch_count_guard_matches_donor_on_every_probed_input() -> None:
    """``guard_polygon_art_patch_count``：同上，三角形數那一支。"""
    records = _records(polygon=True)
    assert records, "答案檔裡沒有 guard_polygon_art_patch_count 的探針——這一支就沒有對象"
    for record in records:
        _run(art_lane.guard_polygon_art_patch_count, record)


def test_patch_count_boundary_is_the_donor_one() -> None:
    """上限那一格是「剛好塞得下」與「跨過去」的交界。

    ``ART_P_CAP`` 對 ``6*n*n``：36 剛好塞得下、37 就跨過去。這一條拿模組自己的常數算，
    所以常數或邊界換了都會紅。
    """
    assert is_approx(art_lane.ART_P_CAP, constant("art_lane", "ART_P_CAP"))
    assert 6 * 36 * 36 <= art_lane.ART_P_CAP
    assert 6 * 37 * 37 > art_lane.ART_P_CAP
