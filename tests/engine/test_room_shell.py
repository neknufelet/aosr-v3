"""房間殼與匯入檢查（票 #1299）：手造的盒子逐項弄壞，加上老闆的第一份測試房當固定考題。

盒子的答案是手算的（六面、三對平行面、體積 −長×寬×高），不從被測程式反推。
測試房的面積與面數是 2026-11-26 用另一支獨立腳本（不是這支程式）對同一份檔量到的。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.room_shell_import import RoomShellImportSettings, load_room_shell_import_settings
from aosr.geometry.rhino_room import RhinoRoomError, read_rhino_room
from aosr.geometry.room_shell import (
    LayerTriangle, Placement, RoomShell, Solid, Vec3, check_room_shell, signed_volume,
)


TEST_ROOM = Path(__file__).parent / "data" / "room_shell_test_room.3dm"


@pytest.fixture(scope="module")
def settings() -> RoomShellImportSettings:
    return load_room_shell_import_settings(config_path("room_shell_import.toml"))


def _quad(layer: str, p: Vec3, q: Vec3, r: Vec3, s: Vec3) -> tuple[LayerTriangle, LayerTriangle]:
    return LayerTriangle(layer, p, q, r), LayerTriangle(layer, p, r, s)


def box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, layer: str,
        inward: bool) -> tuple[LayerTriangle, ...]:
    """長方體六面；``inward`` 為真時法線朝裡（外殼），否則朝外（房內實體）。"""
    c = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    # 每面四個角照「從盒子外面看是逆時針」排，法線朝外。
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    triangles: list[LayerTriangle] = []
    for a, b, d, e in faces:
        order = (a, e, d, b) if inward else (a, b, d, e)
        triangles.extend(_quad(layer, *(c[i] for i in order)))
    return tuple(triangles)


def good_room() -> RoomShell:
    shell = box(0, 0, 0, 5, 4, 3, "Wall", inward=True)
    cloud = Solid("cloud", box(1, 1, 2.5, 3, 3, 2.54, "Cloud", inward=False))
    speakers = (Placement("SPK_BOOKSHELF", (1.0, 1.0, 1.1), (1.0, 0.0, 0.0)),
                Placement("SPK_BOOKSHELF", (1.0, 3.0, 1.1), (1.0, 0.0, 0.0)))
    seats = (Placement("SEAT_MAIN", (3.5, 2.0, 1.2), (-1.0, 0.0, 0.0)),)
    return RoomShell("m", shell, (cloud,), speakers, seats)


def codes(room: RoomShell, settings: RoomShellImportSettings) -> set[str]:
    return {problem.code for problem in check_room_shell(room, settings).problems}


def test_good_box_room_passes_with_hand_counted_facts(settings: RoomShellImportSettings) -> None:
    report = check_room_shell(good_room(), settings)
    assert report.ok, report.problems
    assert report.problems == ()
    # 外殼六面加天雲六面；天雲頂面與天花、底面與地板同向但位置不同，所以不合併。
    assert report.plane_count == 12
    assert dict(report.area_by_layer) == pytest.approx({"Wall": 2 * (20 + 15 + 12), "Cloud": 2 * 4 + 4 * 2 * 0.04})
    assert signed_volume(good_room().shell) == pytest.approx(-60.0)
    assert signed_volume(good_room().solids[0].triangles) == pytest.approx(0.16)


def test_parallel_pairs_face_each_other(settings: RoomShellImportSettings) -> None:
    pairs = check_room_shell(good_room(), settings).parallel_pairs
    gaps = sorted(round(pair.gap_m, 6) for pair in pairs)
    # 外殼三對（3、4、5 公尺），加上天雲底面對地板（2.5）、天雲頂面對天花（0.46）；
    # 天雲側面每面 0.08 平方公尺，小於 0.1 的下限，不列。
    assert gaps == [0.46, 2.5, 3.0, 4.0, 5.0]
    assert all(pair.angle_deg == pytest.approx(0.0, abs=1e-9) for pair in pairs)


def test_hole_in_shell_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    broken = replace(room, shell=room.shell[1:])
    assert "not_closed" in codes(broken, settings)
    assert not check_room_shell(broken, settings).ok


def test_one_flipped_face_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    first = room.shell[0]
    flipped = replace(room, shell=(LayerTriangle(first.layer, first.a, first.c, first.b), *room.shell[1:]))
    assert "inconsistent_faces" in codes(flipped, settings)


def test_whole_shell_flipped_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    outward = replace(room, shell=box(0, 0, 0, 5, 4, 3, "Wall", inward=False))
    assert "faces_flipped" in codes(outward, settings)


def test_solid_facing_inward_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    inward_cloud = replace(room, solids=(Solid("cloud", box(1, 1, 2.5, 3, 3, 2.54, "Cloud", inward=True)),))
    assert "faces_flipped" in codes(inward_cloud, settings)


def test_solid_poking_out_of_room_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    poking = replace(room, solids=(Solid("cloud", box(1, 1, 2.9, 3, 3, 3.2, "Cloud", inward=False)),))
    assert "solid_outside" in codes(poking, settings)


def test_panel_touching_wall_counts_as_inside(settings: RoomShellImportSettings) -> None:
    room = good_room()
    panel = Solid("panel", box(0, 1, 0.5, 0.1, 3, 2.0, "Panel", inward=False))
    assert codes(replace(room, solids=(*room.solids, panel)), settings) == set()


def test_speaker_outside_or_inside_solid_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    outside = replace(room, speakers=(Placement("SPK_FLOOR", (6.0, 2.0, 1.0), (1.0, 0.0, 0.0)),))
    assert "placement_outside" in codes(outside, settings)
    buried = replace(room, speakers=(Placement("SPK_FLOOR", (2.0, 2.0, 2.52), (1.0, 0.0, 0.0)),))
    assert "placement_in_solid" in codes(buried, settings)


def test_seats_and_speakers_must_exist(settings: RoomShellImportSettings) -> None:
    room = good_room()
    assert {"no_speaker", "main_seat_count"} <= codes(replace(room, speakers=(), seats=()), settings)
    no_facing = replace(room, seats=(Placement("SEAT_MAIN", (3.5, 2.0, 1.2), (0.0, 0.0, 0.0)),))
    assert "no_facing" in codes(no_facing, settings)


def test_units_layers_and_unreadable_objects(settings: RoomShellImportSettings) -> None:
    room = good_room()
    assert "units" in codes(replace(room, units="Millimeters"), settings)
    unnamed = replace(room, shell=box(0, 0, 0, 5, 4, 3, "", inward=True))
    assert "no_material_layer" in codes(unnamed, settings)
    assert "unreadable" in codes(replace(room, unreadable=("牆（Brep）",)), settings)
    skipped = check_room_shell(replace(room, skipped=("家具",)), settings)
    assert skipped.ok and {p.code for p in skipped.problems} == {"skipped"}


def test_plane_limit_and_tiny_faces(settings: RoomShellImportSettings) -> None:
    room = good_room()
    assert "too_many_planes" in codes(room, settings.model_copy(update={"max_planes": 11}))
    speck = Solid("speck", box(2, 2, 1, 2.005, 2.005, 1.005, "Speck", inward=False))
    report = check_room_shell(replace(room, solids=(*room.solids, speck)), settings)
    assert {p.code for p in report.problems} == {"tiny_faces"}
    assert report.ok


def test_settings_reject_missing_or_bad_values(settings: RoomShellImportSettings) -> None:
    data = settings.model_dump()
    with pytest.raises(ValueError, match="十進位"):
        RoomShellImportSettings.model_validate({**data, "weld_m": 0.0001})
    with pytest.raises(ValueError, match="大於零"):
        RoomShellImportSettings.model_validate({**{k: str(v) for k, v in data.items()}, "weld_m": "0"})


def test_owner_test_room_reads_and_passes(settings: RoomShellImportSettings) -> None:
    room = read_rhino_room(TEST_ROOM, settings.max_file_bytes)
    assert room.units == "m"
    assert len(room.shell) == 124
    assert [len(solid.triangles) for solid in room.solids] == [12]
    assert [s.kind for s in room.speakers] == ["SPK_BOOKSHELF", "SPK_BOOKSHELF"]
    assert [s.kind for s in room.seats] == ["SEAT_MAIN"]
    report = check_room_shell(room, settings)
    assert report.ok, report.problems
    assert report.plane_count == 32
    areas = dict(report.area_by_layer)
    expected = {"Ceiling_石膏板": 25.50, "Door_金屬門": 1.94, "Floor_木地板": 25.50, "Panel_吸音棉": 5.73,
                "Wall_乳膠漆": 42.05, "Wall_吸音區A": 0.77, "Wall_吸音區_B": 3.68, "Wall_吸音區_C": 4.70,
                "Window_玻璃窗": 2.54, "Cloud_木板": 11.51}
    assert areas == pytest.approx(expected, abs=0.006)
    assert signed_volume(room.shell) == pytest.approx(-70.003, abs=0.001)


def test_owner_test_room_flags_floor_against_cloud(settings: RoomShellImportSettings) -> None:
    report = check_room_shell(read_rhino_room(TEST_ROOM, settings.max_file_bytes), settings)
    layer_pairs = {frozenset((pair.first.layers, pair.second.layers)) for pair in report.parallel_pairs}
    assert frozenset((("Floor_木地板",), ("Cloud_木板",))) in layer_pairs
    assert frozenset((("Ceiling_石膏板",), ("Floor_木地板",))) in layer_pairs


def test_reader_refuses_big_or_foreign_files(tmp_path: Path) -> None:
    with pytest.raises(RhinoRoomError, match="超過上限"):
        read_rhino_room(TEST_ROOM, 1000)
    foreign = tmp_path / "not-a-model.3dm"
    foreign.write_bytes(b"solid nope\nendsolid nope\n")
    with pytest.raises(RhinoRoomError, match="讀不出"):
        read_rhino_room(foreign, 10_000)
