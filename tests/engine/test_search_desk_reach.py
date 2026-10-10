"""喇叭放桌面、座位沒鎖：建資料夾前拒收一定放不上桌面的聆聽距離與間距範圍（#734）。

答案手算；尺寸挑二進位除得盡的值，訊息裡的數字沒有尾差。
方案面向 x0：主位 (2, 2)，書桌跟著主位走，中心在主位前方 0.75 m、深 0.75、寬 2.0，
所以桌面在主位前方 0.375～1.125 m、左右各 1.0 m。箱體寬 0.25、深 0.25、聲學中心在前面板：
往牆那側一定伸出 min(0.25, 0.125)＝0.125、往聽者那側 min(0, 0.125)＝0、左右外側 0.125，
聲學中心只能在主位前方 0.375～1.0 m，間距至多 2×(1.0−0.125)＝1.75 m。
"""
from pathlib import Path
from typing import cast

import pytest

from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._furniture_cases import reference_document, relative_item
from tests.engine._search_store_cases import purpose_settings, settings_document
from tests.engine._speaker_setup_cases import setup

CABINET = {"width_m": 0.25, "depth_m": 0.25, "height_m": 0.375,
           "acoustic_center_behind_front_m": 0.0, "acoustic_center_above_bottom_m": 0.25}
TAIL = "（箱體朝主位時一定伸出聲學中心的部分已扣掉）"
LISTENING_FAR = ("喇叭放桌面時聆聽距離範圍 1.5～2.5 m 內，箱體都放不上桌面頂：桌子跟著主位走，"
                 "聲學中心要在主位前方 0.375～1.0 m" + TAIL)
SPACING_WIDE = "喇叭放桌面時間距下限 1.875 m 超過桌面放得下的 1.75 m" + TAIL


def desk_project(*, facing: str = "x0", mount: str = "desk", left_m: float = 0.0) -> Scheme:
    """facing 是原方案面向的牆：x0 照上面的數字；xL 整組鏡過去；y0 整組轉 90°（主位 (3, 2)）。"""
    document = reference_document()
    document["speaker_setup"] = setup("bookshelf", mount) | {"cabinet": CABINET}
    height = 1.0 if mount == "desk" else 1.25  # 桌面頂 0.75＋聲學中心離箱底 0.25；腳架由案子給。
    # 面向軸上的座標 a（離前牆那一軸）與橫軸座標 c；喇叭在主位前方 0.625、左右各 0.375，箱體四角都在桌面內。
    ahead = 0.625 if mount == "desk" else 1.5  # 腳架那份喇叭放在桌子前面，桌子不在腳架下。
    if facing == "x0":
        def point(a: float, c: float) -> tuple[float, float]:
            return 2.0 - a, 2.0 - c
    elif facing == "xL":
        def point(a: float, c: float) -> tuple[float, float]:
            return 4.0 + a, 2.0 + c
    else:
        def point(a: float, c: float) -> tuple[float, float]:
            return 3.0 + c, 2.0 - a
    (lx, ly), (rx, ry) = point(ahead, 0.375), point(ahead, -0.375)
    document["speakers"] = {"left": {"x": lx, "y": ly, "z": height}, "right": {"x": rx, "y": ry, "z": height}}
    px, py = point(0.0, 0.0)
    sx, sy = point(-0.125, 0.0)
    document["receiver_set"] = {"points": [
        {"receiver_id": "main", "role": "primary", "position_m": [px, py, 1.25], "importance": 1.0},
        {"receiver_id": "back", "role": "surrounding", "position_m": [sx, sy, 1.25], "importance": 1.0,
         "direction_relative_to_primary": "back"}]}
    placement = {"forward_m": 0.75, "left_m": left_m, "bottom_height_m": 0.6875, "yaw_deg": 0}
    document["furniture"] = [relative_item(furniture_id="table", kind="desk", material="wood",
        width_m=2.0, depth_m=0.75, height_m=0.0625, placement=placement)]
    return Scheme.model_validate(document)


def create(root: Path, project: Scheme, *, front_wall: str = "x0", **ranges: tuple[float, float]) -> SearchStore:
    assert project.speaker_setup is not None
    document = settings_document()
    layout = cast(dict[str, object], document["layout"]) | {
        "front_wall": front_wall, "axis_offset_m": 0.0, "ear_height_m": 1.25,
        "speaker_height_m": project.speakers["left"].z, "cabinet": project.speaker_setup.cabinet.model_dump(),
    } | {name: {"low": low, "high": high} for name, (low, high) in ranges.items()}
    settings = SearchSettings.model_validate(document | {"purpose": project.purpose, "layout": layout})
    return SearchStore.create(root, project=project, settings=settings,
        identity=SearchIdentity("phys-test", "calc-test", purpose_settings(project.purpose)),
        versions={"python": "test", "optuna": "test", "numpy": "test"})


def rejected(tmp_path: Path, project: Scheme, *, listening: tuple[float, float], spacing: tuple[float, float],
             front_wall: str = "x0") -> list[tuple[str, str]]:
    root = tmp_path / "searches"
    with pytest.raises(SchemeValidationError) as caught:
        create(root, project, front_wall=front_wall, listening_distance_m=listening, spacing_m=spacing)
    assert not root.exists()
    return [(problem.path, problem.message) for problem in caught.value.problems]


@pytest.mark.parametrize("listening,spacing,expected", [
    ((1.5, 2.5), (0.5, 1.5), [("settings.layout.listening_distance_m", LISTENING_FAR)]),
    # 往聽者那側：範圍上限 0.3125 碰不到桌緣 0.375。
    ((0.25, 0.3125), (0.5, 1.5), [("settings.layout.listening_distance_m",
        "喇叭放桌面時聆聽距離範圍 0.25～0.3125 m 內，箱體都放不上桌面頂：桌子跟著主位走，"
        "聲學中心要在主位前方 0.375～1.0 m" + TAIL)]),
    ((0.5, 1.0), (1.875, 2.5), [("settings.layout.spacing_m", SPACING_WIDE)]),
    ((1.5, 2.5), (1.875, 2.5), [("settings.layout.listening_distance_m", LISTENING_FAR),
                                ("settings.layout.spacing_m", SPACING_WIDE)]),
])
def test_ranges_that_cannot_reach_the_desk_are_rejected_before_mkdir(
    tmp_path: Path, listening: tuple[float, float], spacing: tuple[float, float],
    expected: list[tuple[str, str]],
) -> None:
    assert rejected(tmp_path, desk_project(), listening=listening, spacing=spacing) == expected


@pytest.mark.parametrize("listening,spacing", [
    # 剛好碰到界線也要放行：只擋一定全滅的範圍。
    ((1.0, 2.5), (1.75, 2.0)),
    ((0.25, 0.375), (0.5, 1.75)),
])
def test_ranges_touching_the_desk_reach_are_accepted(
    tmp_path: Path, listening: tuple[float, float], spacing: tuple[float, float],
) -> None:
    store = create(tmp_path / "searches", desk_project(), listening_distance_m=listening, spacing_m=spacing)
    assert store.path.is_dir()


@pytest.mark.parametrize("facing,front_wall", [("xL", "xL"), ("y0", "y0"), ("x0", "y0"), ("x0", "yL")])
def test_reach_is_measured_from_the_primary_whatever_the_walls(tmp_path: Path, facing: str, front_wall: str) -> None:
    # 桌子跟著主位走：原方案換一面牆，或搜尋的前牆跟原方案面向不同，扣出來的數字都一樣。
    project = desk_project(facing=facing)
    assert rejected(tmp_path / "far", project, listening=(1.5, 2.5), spacing=(1.875, 2.5),
                    front_wall=front_wall) == [("settings.layout.listening_distance_m", LISTENING_FAR),
                                   ("settings.layout.spacing_m", SPACING_WIDE)]
    store = create(tmp_path / "near" / "searches", project, front_wall=front_wall,
                   listening_distance_m=(1.0, 2.5), spacing_m=(1.75, 2.0))
    assert store.path.is_dir()


def test_desk_shifted_sideways_narrows_the_spacing(tmp_path: Path) -> None:
    # 桌子往主位左邊挪 0.25：左右各 1.25 與 0.75，窄的那邊扣 0.125 後間距至多 2×0.625＝1.25。
    project = desk_project(left_m=0.25)
    assert rejected(tmp_path, project, listening=(0.5, 1.0), spacing=(1.375, 2.0)) == [
        ("settings.layout.spacing_m", "喇叭放桌面時間距下限 1.375 m 超過桌面放得下的 1.25 m" + TAIL)]
    assert create(tmp_path / "ok", project, listening_distance_m=(0.5, 1.0), spacing_m=(1.25, 2.0)).path.is_dir()


def test_stand_mount_is_not_judged_against_the_desk(tmp_path: Path) -> None:
    # 不放桌面的不判桌面：同一張書桌、同一組會被擋的範圍，放腳架照樣建得起來。
    store = create(tmp_path / "searches", desk_project(mount="stand"),
                   listening_distance_m=(1.5, 2.5), spacing_m=(1.875, 2.5))
    assert store.path.is_dir()
