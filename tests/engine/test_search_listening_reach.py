"""型號適用聆聽距離跟搜尋範圍碰不到時，建資料夾前拒收（#765）；答案手算。

沒鎖定：參考房前牆 x0、中軸 y＝2.0，喇叭與耳朵同高 1.25。聆聽距離 0.375～0.75、間距 1.0～2.0 時，
最近那端 √(0.375²＋0.5²)＝0.625、最遠那端 √(0.75²＋1.0²)＝1.25（二進位除得盡，訊息沒有尾差）。
座位鎖定：主位 (3.2, 1.9, 1.2)、喇叭高 1.25；離前牆 0.25～2.0 推出聆聽距離 2.95～1.2，間距 0.5～2.0，
最遠 √(2.95²＋1.0²＋0.05²)≈3.115、最近 √(1.2²＋0.25²＋0.05²)≈1.227。
"""
from pathlib import Path
from typing import cast

import pytest

from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._seat_locked_cases import locked_settings
from tests.engine._search_store_cases import purpose_settings, reference_project, settings_document

NEAR_FAR = "範圍兩端擺出來的三維距離只有 0.625～1.25 m"


def create(root: Path, project: Scheme, settings: SearchSettings) -> SearchStore:
    settings = settings.model_copy(update={"purpose": project.purpose})
    return SearchStore.create(root, project=project, settings=settings,
        identity=SearchIdentity("phys-test", "calc-test", purpose_settings(project.purpose)),
        versions={"python": "test", "optuna": "test", "numpy": "test"})


def unlocked(limits: tuple[float, float] | None) -> SearchSettings:
    document = settings_document()
    layout = cast(dict[str, object], document["layout"]) | {
        "listening_distance_m": {"low": 0.375, "high": 0.75}, "spacing_m": {"low": 1.0, "high": 2.0},
        "listening_range_m": None if limits is None else {"low": limits[0], "high": limits[1]}}
    return SearchSettings.model_validate(document | {"layout": layout})


@pytest.mark.parametrize("limits,message", [
    ((1.5, 4.0), "型號適用聆聽距離 1.5～4.0 m（喇叭聲學中心到主位的三維距離）跟搜尋範圍碰不到：" + NEAR_FAR),
    ((0.25, 0.5), "型號適用聆聽距離 0.25～0.5 m（喇叭聲學中心到主位的三維距離）跟搜尋範圍碰不到：" + NEAR_FAR),
])
def test_unreachable_listening_range_is_rejected_before_mkdir(
    tmp_path: Path, limits: tuple[float, float], message: str,
) -> None:
    root = tmp_path / "searches"
    with pytest.raises(SchemeValidationError) as caught:
        create(root, reference_project(tmp_path), unlocked(limits))
    assert [(problem.path, problem.message) for problem in caught.value.problems] == [
        ("settings.layout.listening_range_m", message)]
    assert not root.exists()


@pytest.mark.parametrize("limits", [(1.25, 4.0), (0.25, 0.625), None])
def test_listening_range_touching_the_reach_or_unset_is_accepted(
    tmp_path: Path, limits: tuple[float, float] | None,
) -> None:
    # 剛好等於兩端也放行：只擋一定全滅的範圍；沒寫型號距離就不判。
    assert create(tmp_path / "searches", reference_project(tmp_path), unlocked(limits)).path.is_dir()


@pytest.mark.parametrize("limits,accepted", [((3.5, 5.0), False), ((0.5, 1.0), False), ((2.0, 4.0), True)])
def test_seat_locked_reach_follows_the_derived_listening_distance(
    tmp_path: Path, limits: tuple[float, float], accepted: bool,
) -> None:
    # 鎖定時聆聽距離隨離前牆變小：最遠那端用離前牆下限、最近那端用上限，反過來 (2.0, 4.0) 就會被誤擋。
    project = reference_project(tmp_path)
    settings = locked_settings(listening_range_m={"low": limits[0], "high": limits[1]})
    root = tmp_path / "searches"
    if accepted:
        assert create(root, project, settings).path.is_dir()
        return
    with pytest.raises(SchemeValidationError) as caught:
        create(root, project, settings)
    assert [problem.path for problem in caught.value.problems] == ["settings.layout.listening_range_m"]
    assert not root.exists()
