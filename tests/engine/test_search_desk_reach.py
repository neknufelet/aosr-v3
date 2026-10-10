"""喇叭放桌面、座位沒鎖：建資料夾前拒收一定放不上桌面的聆聽距離與間距範圍（#734）。

答案手算；尺寸挑二進位除得盡的值，訊息裡的數字沒有尾差。
方案面向 x0：主位 (2, 2)，書桌跟著主位走，中心在主位前方 0.75 m、深 0.75、寬 2.0，
所以桌面在主位前方 0.375～1.125 m、左右各 1.0 m。箱體寬 0.25、深 0.25、聲學中心在前面板：
往牆那側一定伸出 min(0.25, 0.125)＝0.125、往聽者那側 min(0, 0.125)＝0、左右外側 0.125，
聲學中心至少要在主位前方 0.375～1.0 m，間距至多 2×(1.0−0.125)＝1.75 m。
聲學中心離前面板 0.1875 時：箱背 0.0625，往牆 min(0.0625, 0.125)＝0.0625、往聽者 min(0.1875, 0.125)＝0.125，
所以是 0.5～1.0625 m、間距至多 2×(1.0−0.0625)＝1.875 m。
接觸界線＝房間最長邊 6 m × 2^-40 ≈ 5.46e-12 m：每支箱角伸出桌緣在它以內不算超出（間距兩支各一把）。
"""
from pathlib import Path
from typing import cast

import pytest

from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError
from aosr.search import seat_lock
from aosr.search.layout_settings import LayoutSettings
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._furniture_cases import reference_document, relative_item
from tests.engine._search_store_cases import purpose_settings, settings_document
from tests.engine._speaker_setup_cases import setup

NOTE = "（只扣了箱體朝主位時一定伸出聲學中心的部分；範圍內也不一定都放得上）"
LISTENING_FAR = ("喇叭放桌面時聆聽距離範圍 1.5～2.5 m 內，箱體都放不上桌面頂：桌子跟著主位走，"
                 "聲學中心至少要落在主位前方 0.375～1.0 m 之內" + NOTE)
SPACING_WIDE = "喇叭放桌面時間距下限 1.875 m 超過桌面最多放得下的 1.75 m" + NOTE
LOCKED_IMPOSSIBLE = ("座位鎖定時離前牆與間距範圍內，沒有任何一組能讓兩支喇叭的箱體放上桌面頂"
                     "（箱體朝主位時一定伸出聲學中心的部分已扣掉）")


def desk_project(*, facing: str = "x0", mount: str = "desk", left_m: float = 0.0,
                 behind_front: float = 0.0, rotated: bool = False) -> Scheme:
    """facing 是原方案面向的牆：x0 照上面的數字；xL 整組鏡過去；y0、yL 整組轉 90°（主位 (3, 2)）。

    rotated 時書桌寬 0.75、深 2.0、相對轉 90°，桌面在房間裡的範圍跟沒轉的那張一樣。
    """
    document = reference_document()
    cabinet = {"width_m": 0.25, "depth_m": 0.25, "height_m": 0.375,
               "acoustic_center_behind_front_m": behind_front, "acoustic_center_above_bottom_m": 0.25}
    document["speaker_setup"] = setup("bookshelf", mount) | {"cabinet": cabinet}
    height = 1.0 if mount == "desk" else 1.25  # 桌面頂 0.75＋聲學中心離箱底 0.25；腳架由案子給。
    # 面向軸上的座標 a（主位前方）與橫軸座標 c（主位左方）；喇叭在主位前方 0.625、左右各 0.375，箱體四角都在桌面內。
    ahead = 0.625 if mount == "desk" else 1.5  # 腳架那份喇叭放在桌子前面，桌子不在腳架下。
    if facing == "x0":
        def point(a: float, c: float) -> tuple[float, float]:
            return 2.0 - a, 2.0 - c
    elif facing == "xL":
        def point(a: float, c: float) -> tuple[float, float]:
            return 4.0 + a, 2.0 + c
    elif facing == "y0":
        def point(a: float, c: float) -> tuple[float, float]:
            return 3.0 + c, 2.0 - a
    else:
        def point(a: float, c: float) -> tuple[float, float]:
            return 3.0 - c, 2.0 + a
    (lx, ly), (rx, ry) = point(ahead, 0.375), point(ahead, -0.375)
    document["speakers"] = {"left": {"x": lx, "y": ly, "z": height}, "right": {"x": rx, "y": ry, "z": height}}
    px, py = point(0.0, 0.0)
    sx, sy = point(-0.125, 0.0)
    document["receiver_set"] = {"points": [
        {"receiver_id": "main", "role": "primary", "position_m": [px, py, 1.25], "importance": 1.0},
        {"receiver_id": "back", "role": "surrounding", "position_m": [sx, sy, 1.25], "importance": 1.0,
         "direction_relative_to_primary": "back"}]}
    width, depth = (0.75, 2.0) if rotated else (2.0, 0.75)
    document["furniture"] = [relative_item(furniture_id="table", kind="desk", material="wood",
        width_m=width, depth_m=depth, height_m=0.0625,
        placement={"forward_m": 0.75, "left_m": left_m, "bottom_height_m": 0.6875, "yaw_deg": 90 if rotated else 0})]
    return Scheme.model_validate(document)


def layout_settings(project: Scheme, *, front_wall: str = "x0", seat_locked: bool = False,
                    **ranges: tuple[float, float]) -> LayoutSettings:
    assert project.speaker_setup is not None
    layout = cast(dict[str, object], settings_document()["layout"]) | {
        "front_wall": front_wall, "axis_offset_m": 0.0, "ear_height_m": 1.25,
        "speaker_height_m": project.speakers["left"].z, "cabinet": project.speaker_setup.cabinet.model_dump(),
    } | ({"seat_locked": True, "listening_distance_m": None} if seat_locked else {}) | {
        name: {"low": low, "high": high} for name, (low, high) in ranges.items()}
    return LayoutSettings.model_validate(layout)


def create(root: Path, project: Scheme, *, front_wall: str = "x0", seat_locked: bool = False,
           **ranges: tuple[float, float]) -> SearchStore:
    layout = layout_settings(project, front_wall=front_wall, seat_locked=seat_locked, **ranges)
    settings = SearchSettings.model_validate(settings_document() | {
        "purpose": project.purpose, "layout": layout.model_dump(mode="json")})
    return SearchStore.create(root, project=project, settings=settings,
        identity=SearchIdentity("phys-test", "calc-test", purpose_settings(project.purpose)),
        versions={"python": "test", "optuna": "test", "numpy": "test"})


def rejected(tmp_path: Path, project: Scheme, *, front_wall: str = "x0", seat_locked: bool = False,
             **ranges: tuple[float, float]) -> list[tuple[str, str]]:
    root = tmp_path / "searches"
    with pytest.raises(SchemeValidationError) as caught:
        create(root, project, front_wall=front_wall, seat_locked=seat_locked, **ranges)
    assert not root.exists()
    return [(problem.path, problem.message) for problem in caught.value.problems]


@pytest.mark.parametrize("listening,spacing,expected", [
    ((1.5, 2.5), (0.5, 1.5), [("settings.layout.listening_distance_m", LISTENING_FAR)]),
    # 往聽者那側：範圍上限 0.3125 碰不到桌緣 0.375。
    ((0.25, 0.3125), (0.5, 1.5), [("settings.layout.listening_distance_m",
        "喇叭放桌面時聆聽距離範圍 0.25～0.3125 m 內，箱體都放不上桌面頂：桌子跟著主位走，"
        "聲學中心至少要落在主位前方 0.375～1.0 m 之內" + NOTE)]),
    ((0.5, 1.0), (1.875, 2.5), [("settings.layout.spacing_m", SPACING_WIDE)]),
    ((1.5, 2.5), (1.875, 2.5), [("settings.layout.listening_distance_m", LISTENING_FAR),
                                ("settings.layout.spacing_m", SPACING_WIDE)]),
])
def test_ranges_that_cannot_reach_the_desk_are_rejected_before_mkdir(
    tmp_path: Path, listening: tuple[float, float], spacing: tuple[float, float],
    expected: list[tuple[str, str]],
) -> None:
    assert rejected(tmp_path, desk_project(), listening_distance_m=listening, spacing_m=spacing) == expected


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


@pytest.mark.parametrize("listening,spacing,expected", [
    ((0.25, 0.4375), (0.5, 1.5), [("settings.layout.listening_distance_m",
        "喇叭放桌面時聆聽距離範圍 0.25～0.4375 m 內，箱體都放不上桌面頂：桌子跟著主位走，"
        "聲學中心至少要落在主位前方 0.5～1.0625 m 之內" + NOTE)]),
    ((1.125, 2.0), (0.5, 1.5), [("settings.layout.listening_distance_m",
        "喇叭放桌面時聆聽距離範圍 1.125～2.0 m 內，箱體都放不上桌面頂：桌子跟著主位走，"
        "聲學中心至少要落在主位前方 0.5～1.0625 m 之內" + NOTE)]),
    ((0.5, 1.0), (2.0, 2.5), [("settings.layout.spacing_m",
        "喇叭放桌面時間距下限 2.0 m 超過桌面最多放得下的 1.875 m" + NOTE)]),
    ((0.25, 0.5), (0.5, 1.875), []),
    ((1.0625, 2.0), (1.875, 2.0), []),
])
def test_acoustic_center_behind_the_front_panel_moves_both_sides(
    tmp_path: Path, listening: tuple[float, float], spacing: tuple[float, float],
    expected: list[tuple[str, str]],
) -> None:
    # 聲學中心離前面板 0.1875：往聽者那側改扣 0.125、往牆與左右外側改扣箱背的 0.0625。
    project = desk_project(behind_front=0.1875)
    if expected:
        assert rejected(tmp_path, project, listening_distance_m=listening, spacing_m=spacing) == expected
    else:
        assert create(tmp_path / "searches", project, listening_distance_m=listening, spacing_m=spacing).path.is_dir()


@pytest.mark.parametrize("listening,spacing,path", [
    ((1.0 + 4e-12, 2.5), (1.0, 1.5), None),
    ((1.0 + 1e-11, 2.5), (1.0, 1.5), "settings.layout.listening_distance_m"),
    ((0.25, 0.375 - 4e-12), (1.0, 1.5), None),
    ((0.25, 0.375 - 1e-11), (1.0, 1.5), "settings.layout.listening_distance_m"),
    # 間距兩支各伸出一把界線：8e-12 超過一把、不到兩把，要放行。
    ((0.5, 1.0), (1.75 + 8e-12, 2.0), None),
    ((0.5, 1.0), (1.75 + 2e-11, 2.0), "settings.layout.spacing_m"),
])
def test_contact_boundary_absorbs_tail_error_like_the_candidate_check(
    tmp_path: Path, listening: tuple[float, float], spacing: tuple[float, float], path: str | None,
) -> None:
    project = desk_project()
    if path is None:
        assert create(tmp_path / "searches", project, listening_distance_m=listening, spacing_m=spacing).path.is_dir()
    else:
        assert [found for found, _ in rejected(tmp_path, project, listening_distance_m=listening,
                                               spacing_m=spacing)] == [path]


@pytest.mark.parametrize("front,spacing,accepted", [
    # 座位鎖定版同一把界線：聲學中心 x＝離前牆，桌緣 0.875＋往牆 0.125＝1.0 是下限。
    ((0.5, 1.0 - 4e-12), (0.5, 1.5), True),
    ((0.5, 1.0 - 1e-11), (0.5, 1.5), False),
    ((0.5, 1.5), (1.75 + 8e-12, 2.0), True),
    ((0.5, 1.5), (1.75 + 2e-11, 2.0), False),
])
def test_seat_locked_desk_check_uses_the_same_contact_boundary(
    tmp_path: Path, front: tuple[float, float], spacing: tuple[float, float], accepted: bool,
) -> None:
    project = desk_project()
    if accepted:
        store = create(tmp_path / "searches", project, seat_locked=True, front_distance_m=front, spacing_m=spacing)
        assert store.path.is_dir()
    else:
        assert rejected(tmp_path, project, seat_locked=True, front_distance_m=front,
                        spacing_m=spacing) == [("settings.layout", LOCKED_IMPOSSIBLE)]


@pytest.mark.parametrize("facing,front_wall,rotated", [
    ("xL", "xL", False), ("y0", "y0", False), ("yL", "yL", False),
    ("x0", "y0", False), ("x0", "yL", False), ("x0", "x0", True), ("yL", "yL", True),
])
def test_reach_is_measured_from_the_primary_whatever_the_walls(
    tmp_path: Path, facing: str, front_wall: str, rotated: bool,
) -> None:
    # 桌子跟著主位走：原方案換一面牆、搜尋的前牆跟原方案面向不同、或桌子轉 90° 換寬深，扣出來的數字都一樣。
    project = desk_project(facing=facing, rotated=rotated)
    assert rejected(tmp_path / "far", project, front_wall=front_wall, listening_distance_m=(1.5, 2.5),
                    spacing_m=(1.875, 2.5)) == [("settings.layout.listening_distance_m", LISTENING_FAR),
                                                ("settings.layout.spacing_m", SPACING_WIDE)]
    store = create(tmp_path / "near" / "searches", project, front_wall=front_wall,
                   listening_distance_m=(1.0, 2.5), spacing_m=(1.75, 2.0))
    assert store.path.is_dir()


def test_desk_shifted_sideways_narrows_the_spacing(tmp_path: Path) -> None:
    # 桌子往主位左邊挪 0.25：左右各 1.25 與 0.75，窄的那邊扣 0.125 後間距至多 2×0.625＝1.25。
    project = desk_project(left_m=0.25)
    assert rejected(tmp_path, project, listening_distance_m=(0.5, 1.0), spacing_m=(1.375, 2.0)) == [
        ("settings.layout.spacing_m", "喇叭放桌面時間距下限 1.375 m 超過桌面最多放得下的 1.25 m" + NOTE)]
    assert create(tmp_path / "ok", project, listening_distance_m=(0.5, 1.0), spacing_m=(1.25, 2.0)).path.is_dir()


def test_stand_mount_is_not_judged_and_does_not_read_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 不放桌面的不判桌面，也不讀登記簿：同一張書桌、同一組會被擋的範圍，放腳架照樣建得起來。
    def registry_read(_: Path) -> float:
        raise AssertionError("不放桌面時不該讀登記簿")

    monkeypatch.setattr(seat_lock, "furniture_contact_rel", registry_read)
    store = create(tmp_path / "searches", desk_project(mount="stand"),
                   listening_distance_m=(1.5, 2.5), spacing_m=(1.875, 2.5))
    assert store.path.is_dir()
    with pytest.raises(AssertionError, match="不放桌面時不該讀登記簿"):
        create(tmp_path / "desk", desk_project(), listening_distance_m=(0.5, 1.0), spacing_m=(0.5, 1.5))


def test_layout_errors_inside_the_check_become_validation_problems() -> None:
    # 建搜尋時前面的方案驗證會先擋；這支自己被直接呼叫時也只交出中文的驗證錯，不漏出 ValueError。
    document = desk_project().model_dump(mode="json")
    document["furniture"][0]["placement"]["forward_m"] = 1.9  # 桌子中心 x＝0.1，前緣伸出前牆。
    project = Scheme.model_validate(document)
    with pytest.raises(SchemeValidationError) as caught:
        seat_lock.check_desk_reach(project, layout_settings(project, listening_distance_m=(0.5, 1.0)))
    assert [(problem.path, problem.message) for problem in caught.value.problems] == [
        ("settings.layout", "家具 table 超出房間接觸界線")]
