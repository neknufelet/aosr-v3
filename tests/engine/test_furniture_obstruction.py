"""手算折線遮擋；中段方向無關，另設外段控制組咬反彈點反序。"""
from __future__ import annotations

from aosr.geometry.furniture import FurnitureBox, Vec3
from aosr.geometry.shoebox import Point, Room
from aosr.physics.furniture_paths import Furniture, direct_path_blockers, filter_room_paths, single_bounce_furniture_paths
from aosr.physics.room_paths import RoomPath, image_source_paths
from tests.engine.test_furniture_paths import TABLE_EXAMPLE_C, table, table_path


def second_order_wall_path() -> RoomPath:
    # 手算：S(1,1,1) → x0(0,2,1) → yL(2,4,1) → E(3,3,1)。
    # 鏡像 (-1,7,1)，身分 x0 一次、yL 一次；bounces 從 E 往 S 排。
    paths = image_source_paths(Room(4.0, 4.0, 3.0), Point(1.0, 1.0, 1.0), Point(3.0, 3.0, 1.0),
                               TABLE_EXAMPLE_C, max_order=2)
    return next(path for path in paths if path.identity == (0, -1, 1, -1, 0, 1))


def test_direct_pairs_identify_all_blocking_furniture() -> None:
    sources: dict[str, Vec3] = {"left": (-2.0, 0.0, 1.0), "right": (-2.0, 2.0, 1.0)}
    receivers: dict[str, Vec3] = {"main": (2.0, 0.0, 1.0), "surround": (2.0, 2.0, 1.0)}
    furniture = (table("b", center=(1.0, 0.0, 0.8), width=0.2, depth=0.2, height=0.4),
                 table("a", center=(-1.0, 0.0, 0.8), width=0.2, depth=0.2, height=0.4))
    got = direct_path_blockers(sources, receivers, furniture, margin_m=0.0)
    assert got == {("left", "main"): ("a", "b"), ("left", "surround"): (),
                   ("right", "main"): (), ("right", "surround"): ()}
    assert got == direct_path_blockers(sources, receivers, tuple(reversed(furniture)), margin_m=0.0)


def test_only_reflected_leg_is_blocked() -> None:
    source, receiver = (-0.6, 0.0, 1.05), (0.6, 0.0, 1.05)
    # 第二段在 x=.3 時 z=.825；第一段 x≤0，直達 z=1.05，都不穿這顆盒子。
    blocker = table("blocker", center=(0.3, 0.0, 0.775), width=0.1, depth=0.1, height=0.1)
    assert table_path().hit
    got = single_bounce_furniture_paths(source, receiver, (table(), blocker), c=TABLE_EXAMPLE_C, margin_m=0.0)
    assert all(path.furniture_id != "table" for path in got)
    assert direct_path_blockers({"speaker": source}, {"main": receiver}, (blocker,), margin_m=0.0) == {
        ("speaker", "main"): ()}


def test_only_incident_leg_is_blocked() -> None:
    blocker = table("blocker", center=(-0.3, 0.0, 0.775), width=0.1, depth=0.1, height=0.1)
    got = single_bounce_furniture_paths((-0.6, 0.0, 1.05), (0.6, 0.0, 1.05),
                                        (table(), blocker), c=TABLE_EXAMPLE_C, margin_m=0.0)
    assert all(path.furniture_id != "table" for path in got)


def test_second_order_middle_segment_is_checked() -> None:
    path = second_order_wall_path()
    assert tuple(bounce.point for bounce in path.bounces) == ((2.0, 4.0, 1.0), (0.0, 2.0, 1.0))
    blocker = table("middle", center=(1.0, 3.0, 0.8), width=0.2, depth=0.2, height=0.4)
    kept, blocked = filter_room_paths((path,), (1.0, 1.0, 1.0), (3.0, 3.0, 1.0), (blocker,), margin_m=0.0)
    assert kept == () and blocked == (path.identity,)


def test_receiver_order_is_reversed_before_attaching_outer_segments() -> None:
    path = second_order_wall_path()
    # 誤用 bounces 原序時 S→(2,4) 與 (0,2)→E 都穿 (1.5,2.5)；實際三段都不穿。
    blocker = table("wrong-polyline", center=(1.5, 2.5, 0.8), width=0.2, depth=0.2, height=0.4)
    kept, blocked = filter_room_paths((path,), (1.0, 1.0, 1.0), (3.0, 3.0, 1.0), (blocker,), margin_m=0.0)
    assert kept == (path,) and kept[0] is path and blocked == ()


def test_direct_room_path_is_also_filtered() -> None:
    source, receiver = Point(1.0, 2.0, 1.0), Point(3.0, 2.0, 1.0)
    direct = next(path for path in image_source_paths(Room(4.0, 4.0, 3.0), source, receiver,
                                                      TABLE_EXAMPLE_C) if not path.bounces)
    blocker = table("solid", center=(2.0, 2.0, 0.8), width=0.2, depth=0.2, height=0.4)
    assert filter_room_paths((direct,), source.as_tuple(), receiver.as_tuple(), (blocker,), margin_m=0.0) == (
        (), (direct.identity,))


def test_cloud_flush_ceiling_contact_is_clear_but_penetration_blocks() -> None:
    cloud = Furniture("cloud", FurnitureBox(kind="ceiling_cloud", bottom_center_m=(2.0, 2.0, 2.8),
                                            width_m=0.4, depth_m=0.4, height_m=0.2, margin_m=0.0))
    # 面上接觸不擋；頂面端點往下穿實體則擋，不能把整件反射家具跳過。
    assert direct_path_blockers({"speaker": (1.0, 2.0, 3.0)}, {"hit": (2.0, 2.0, 3.0)},
                                (cloud,), margin_m=0.0) == {("speaker", "hit"): ()}
    assert direct_path_blockers({"hit": (2.0, 2.0, 3.0)}, {"main": (3.0, 2.0, 2.0)},
                                (cloud,), margin_m=0.0) == {("hit", "main"): ("cloud",)}
    source, receiver = Point(1.0, 2.0, 2.0), Point(3.0, 2.0, 2.0)
    ceiling = next(path for path in image_source_paths(Room(4.0, 4.0, 3.0), source, receiver,
                                                       TABLE_EXAMPLE_C) if path.identity == (0, 1, 0, 1, 1, -1))
    assert ceiling.bounces[0].point == (2.0, 2.0, 3.0)
    assert filter_room_paths((ceiling,), source.as_tuple(), receiver.as_tuple(), (cloud,), margin_m=0.0) == (
        (), (ceiling.identity,))


def test_filter_keeps_original_paths_and_input_order() -> None:
    source, receiver = Point(1.0, 2.0, 1.0), Point(3.0, 2.0, 1.0)
    paths = tuple(image_source_paths(Room(4.0, 4.0, 3.0), source, receiver, TABLE_EXAMPLE_C))
    blocker = table("solid", center=(2.0, 2.0, 0.8), width=0.2, depth=0.2, height=0.4)
    kept, blocked = filter_room_paths(paths, source.as_tuple(), receiver.as_tuple(), (blocker,), margin_m=0.0)
    assert kept and blocked
    assert tuple(path for path in paths if path.identity not in blocked) == kept
    assert all(any(path is original for original in paths) for path in kept)
    assert tuple(path.identity for path in paths if path not in kept) == blocked


def test_no_furniture_returns_same_wall_tuple_and_clear_pairs() -> None:
    source, receiver = Point(1.0, 1.0, 1.0), Point(3.0, 3.0, 1.0)
    paths = tuple(image_source_paths(Room(4.0, 4.0, 3.0), source, receiver, TABLE_EXAMPLE_C, max_order=2))
    kept, blocked = filter_room_paths(paths, source.as_tuple(), receiver.as_tuple(), (), margin_m=0.0)
    assert kept is paths and blocked == ()
    assert direct_path_blockers({"speaker": source.as_tuple()}, {"main": receiver.as_tuple()}, (), margin_m=0.0) == {
        ("speaker", "main"): ()}
