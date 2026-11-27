"""房間殼與匯入檢查（票 #1299）：手造的盒子逐項弄壞，加上老闆的第一份測試房當固定考題。

盒子的答案是手算的（六面、三對平行面、體積 −長×寬×高），不從被測程式反推。
測試房的面積與面數是 2026-11-26 用另一支獨立腳本（不是這支程式）對同一份檔量到的。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import rhino3dm

from aosr.config.paths import config_path
from aosr.config.room_shell_import import RoomShellImportSettings, load_room_shell_import_settings
from aosr.geometry import room_shell
from aosr.geometry.rhino_room import RhinoRoomError, read_rhino_room
from aosr.geometry.room_shell import (
    ImportReport, LayerTriangle, Placement, RoomShell, Solid, Vec3, check_room_shell, plane_groups, signed_volume,
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


def assert_blocked(report: ImportReport, code: str) -> None:
    # 警告也能帶同一個代號，必須同時核對這一項和整份報告的判決。
    assert any(p.code == code and p.blocking is True for p in report.problems), report.problems
    assert report.ok is False


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
    assert_blocked(check_room_shell(broken, settings), "not_closed")


def test_one_flipped_face_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    first = room.shell[0]
    flipped = replace(room, shell=(LayerTriangle(first.layer, first.a, first.c, first.b), *room.shell[1:]))
    assert_blocked(check_room_shell(flipped, settings), "inconsistent_faces")


def test_whole_shell_flipped_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    outward = replace(room, shell=box(0, 0, 0, 5, 4, 3, "Wall", inward=False))
    assert_blocked(check_room_shell(outward, settings), "faces_flipped")


def test_solid_facing_inward_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    inward_cloud = replace(room, solids=(Solid("cloud", box(1, 1, 2.5, 3, 3, 2.54, "Cloud", inward=True)),))
    assert_blocked(check_room_shell(inward_cloud, settings), "faces_flipped")


def test_solid_poking_out_of_room_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    poking = replace(room, solids=(Solid("cloud", box(1, 1, 2.9, 3, 3, 3.2, "Cloud", inward=False)),))
    assert_blocked(check_room_shell(poking, settings), "solid_outside")


def test_panel_touching_wall_counts_as_inside(settings: RoomShellImportSettings) -> None:
    room = good_room()
    panel = Solid("panel", box(0, 1, 0.5, 0.1, 3, 2.0, "Panel", inward=False))
    assert check_room_shell(replace(room, solids=(*room.solids, panel)), settings).problems == ()


def test_speaker_outside_or_inside_solid_is_blocked(settings: RoomShellImportSettings) -> None:
    room = good_room()
    outside = replace(room, speakers=(Placement("SPK_FLOOR", (6.0, 2.0, 1.0), (1.0, 0.0, 0.0)),))
    assert_blocked(check_room_shell(outside, settings), "placement_outside")
    buried = replace(room, speakers=(Placement("SPK_FLOOR", (2.0, 2.0, 2.52), (1.0, 0.0, 0.0)),))
    assert_blocked(check_room_shell(buried, settings), "placement_in_solid")


def test_seats_and_speakers_must_exist(settings: RoomShellImportSettings) -> None:
    room = good_room()
    missing = check_room_shell(replace(room, speakers=(), seats=()), settings)
    assert_blocked(missing, "no_speaker")
    assert_blocked(missing, "main_seat_count")
    no_facing = replace(room, seats=(Placement("SEAT_MAIN", (3.5, 2.0, 1.2), (0.0, 0.0, 0.0)),))
    assert_blocked(check_room_shell(no_facing, settings), "no_facing")


def test_units_layers_and_unreadable_objects(settings: RoomShellImportSettings) -> None:
    room = good_room()
    assert_blocked(check_room_shell(replace(room, units="Millimeters"), settings), "units")
    unnamed = replace(room, shell=box(0, 0, 0, 5, 4, 3, "", inward=True))
    assert_blocked(check_room_shell(unnamed, settings), "no_material_layer")
    assert_blocked(check_room_shell(replace(room, unreadable=("牆（Brep）",)), settings), "unreadable")
    skipped = check_room_shell(replace(room, skipped=("家具",)), settings)
    assert skipped.ok and {p.code for p in skipped.problems} == {"skipped"}


def test_units_and_flipped_faces_each_block_import(settings: RoomShellImportSettings) -> None:
    # 兩個錯誤同時存在時，只看 report.ok 會漏掉其中一個被降成提醒。
    shell = box(0, 0, 0, 5, 4, 3, "Wall", inward=False)
    report = check_room_shell(replace(good_room(), shell=shell, units="Millimeters"), settings)
    assert_blocked(report, "units")
    assert_blocked(report, "faces_flipped")


def test_plane_limit_and_tiny_faces(settings: RoomShellImportSettings) -> None:
    room = good_room()
    assert_blocked(check_room_shell(room, settings.model_copy(update={"max_planes": 11})), "too_many_planes")
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
    # 由檔內各元件的變換矩陣讀出平移與 +Y 欄，釘住三個元件各自的位置和朝向。
    expected_poses = (
        ((1.160823941230774, 2.9000000953674316, 1.1000001430511475), (0.911921501159668, -0.41036468744277954, 0.0)),
        ((1.160823941230774, 1.1000001430511475, 1.1000001430511475), (0.911921501159668, 0.41036468744277954, 0.0)),
        ((3.1608235836029053, 2.0, 1.1999999284744263), (-1.0, 4.832871947314723e-16, 0.0)),
    )
    for item, (position, facing) in zip((*room.speakers, *room.seats), expected_poses, strict=True):
        assert item.position == pytest.approx(position, rel=0, abs=1e-9)
        assert item.facing == pytest.approx(facing, rel=0, abs=1e-9)
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


# ── 獨立審查（11-26）找到的十個洞，各一題 ──────────────────────────────────────


def _layer_index(model: rhino3dm.File3dm, full_path: str) -> int:
    return next(i for i in range(len(model.Layers)) if model.Layers[i].FullPath == full_path)


def _tetra_mesh() -> rhino3dm.Mesh:
    mesh = rhino3dm.Mesh()
    for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        mesh.Vertices.Add(*point)
    for a, b, c in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)):
        mesh.Faces.AddFace(a, b, c)
    return mesh


def test_block_members_and_loose_speaker_objects(settings: RoomShellImportSettings, tmp_path: Path) -> None:
    """元件定義裡的網格不是場景裡的造型；喇叭圖層上不是元件的東西要擋，不准靜靜消失。"""
    model = rhino3dm.File3dm.Read(str(TEST_ROOM))
    assert model is not None
    member = rhino3dm.ObjectAttributes()
    member.LayerIndex = _layer_index(model, "TREAT::Cloud_木板")
    model.InstanceDefinitions.Add("PAIR", "", "", "", rhino3dm.Point3d(0.0, 0.0, 0.0), (_tetra_mesh(),), (member,))
    loose = rhino3dm.ObjectAttributes()
    loose.LayerIndex = _layer_index(model, "SPEAKER")
    model.Objects.AddPoint(rhino3dm.Point3d(2.0, 2.0, 1.0), loose)
    edited = tmp_path / "edited.3dm"
    assert model.Write(str(edited), 8)
    room = read_rhino_room(edited, settings.max_file_bytes)
    assert [len(solid.triangles) for solid in room.solids] == [12]
    assert len(room.unreadable) == 1 and "SPEAKER" in room.unreadable[0]
    assert_blocked(check_room_shell(room, settings), "unreadable")


def test_unknown_speaker_block_blocks(settings: RoomShellImportSettings, tmp_path: Path) -> None:
    """喇叭圖層上認不出名字的元件不准只留一句提醒就算過。"""
    model = rhino3dm.File3dm.Read(str(TEST_ROOM))
    assert model is not None
    index = model.InstanceDefinitions.Add("Genelec_8030", "", "", "", rhino3dm.Point3d(0.0, 0.0, 0.0),
                                          (_tetra_mesh(),), (rhino3dm.ObjectAttributes(),))
    definition = model.InstanceDefinitions.FindIndex(index)
    assert definition is not None
    placed = rhino3dm.ObjectAttributes()
    placed.LayerIndex = _layer_index(model, "SPEAKER")
    model.Objects.AddInstanceObject(
        rhino3dm.InstanceReference(definition.Id, rhino3dm.Transform.Translation(2.0, 2.0, 1.0)), placed)
    edited = tmp_path / "unknown-block.3dm"
    assert model.Write(str(edited), 8)
    room = read_rhino_room(edited, settings.max_file_bytes)
    assert len(room.speakers) == 2
    assert len(room.unreadable) == 1 and "SPK_BOOKSHELF" in room.unreadable[0]
    assert_blocked(check_room_shell(room, settings), "unreadable")


def test_every_disconnected_piece_is_judged(settings: RoomShellImportSettings) -> None:
    room = good_room()
    stray = box(10, 10, 0, 11, 11, 1, "Wall", inward=True)
    assert_blocked(check_room_shell(replace(room, shell=(*room.shell, *stray)), settings), "shell_pieces")
    mixed = Solid("cloud", (*room.solids[0].triangles, *box(2, 2, 1, 2.5, 2.5, 1.5, "Cloud", inward=True)))
    assert_blocked(check_room_shell(replace(room, solids=(mixed,)), settings), "faces_flipped")


def test_in_wall_speaker_may_sit_on_the_wall(settings: RoomShellImportSettings) -> None:
    room = good_room()
    flush = Placement("SPK_INWALL", (0.0, 2.0, 1.2), (1.0, 0.0, 0.0))
    assert check_room_shell(replace(room, speakers=(flush,)), settings).problems == ()
    floating = Placement("SPK_BOOKSHELF", (0.0, 2.0, 1.2), (1.0, 0.0, 0.0))
    assert_blocked(check_room_shell(replace(room, speakers=(floating,)), settings), "placement_outside")


def test_points_whose_ray_grazes_a_shared_edge_stay_inside(settings: RoomShellImportSettings) -> None:
    # 審查實測：這一點用舊的單一方向會擦過天花對角線、被判成在房外。
    room = good_room()
    seat = Placement("SEAT_MAIN", (2.09589, 1.57097, 2.62221), (-1.0, 0.0, 0.0))
    assert check_room_shell(replace(room, seats=(seat,)), settings).ok


def test_weld_tolerates_noise_across_grid_lines(settings: RoomShellImportSettings) -> None:
    shifted = box(1.00015, 0, 0, 6.00015, 4, 3, "Wall", inward=True)
    noisy = tuple(
        LayerTriangle(t.layer, *((x + 2e-9, y, z) for x, y, z in (t.a, t.b, t.c))) if i % 2 else t
        for i, t in enumerate(shifted))
    room = replace(good_room(), shell=noisy, solids=(), speakers=(Placement("SPK_FLOOR", (2.0, 1.0, 1.0), (1.0, 0.0, 0.0)),),
                   seats=(Placement("SEAT_MAIN", (4.0, 2.0, 1.2), (-1.0, 0.0, 0.0)),))
    assert check_room_shell(room, settings).problems == ()


def test_plane_grouping_does_not_depend_on_where_the_model_sits(settings: RoomShellImportSettings) -> None:
    near = box(0, 0, 0, 5, 4, 3, "Wall", inward=True)
    far = box(100, 100, 0, 105, 104, 3, "Wall", inward=True)
    count = [len(plane_groups(shell, settings.coplanar_angle_deg, settings.coplanar_offset_m)) for shell in (near, far)]
    assert count == [6, 6]
    # 遠處一個只歪 0.3 度的三角形仍然跟同一面牆算同一個平面。
    tilt = 0.3 * 3.141592653589793 / 180.0
    wall = LayerTriangle("Wall", (100.0, 100.0, 0.0), (100.0 + 1.0 * tilt, 101.0, 0.0), (100.0, 100.0, 3.0))
    assert len(plane_groups((*far, wall), settings.coplanar_angle_deg, settings.coplanar_offset_m)) == 6


def test_flat_closed_solid_is_zero_volume(settings: RoomShellImportSettings) -> None:
    room = good_room()
    # 厚 0.15 毫米：比焊點距離大（面不會被焊在一起），體積卻小到等於零。
    flat = Solid("flat", box(1, 1, 1, 2, 2, 1.00015, "Flat", inward=False))
    assert_blocked(check_room_shell(replace(room, solids=(*room.solids, flat)), settings), "zero_volume")


@pytest.mark.parametrize(("width", "thickness", "flipped", "volume", "blocker"), [
    (.1, .04, False, .0004, None), (.1, .04, True, -.0004, "faces_flipped"),
    (1., .00015, False, .00015, "zero_volume"),
], ids=["outward", "inward", "flat"])
def test_survey_coordinates_preserve_panel_volume_and_orientation(
    settings: RoomShellImportSettings, width: float, thickness: float, flipped: bool, volume: float, blocker: str | None,
) -> None:
    # 方板移到測量座標後，手算的面積乘厚度、翻面和近零體積的判決都應相同。
    x, y, z = 500000, 2750000, 100
    panel = box(x + 1, y + 1, z + 1, x + 1 + width, y + 1 + width, z + 1 + thickness, "Panel", inward=flipped)
    room = good_room()
    room = replace(room, shell=box(x, y, z, x + 5, y + 4, z + 3, "Wall", inward=True), solids=(Solid("panel", panel),),
                   speakers=(replace(room.speakers[0], position=(x + 1, y + 3, z + 1.1)),),
                   seats=(replace(room.seats[0], position=(x + 3.5, y + 2, z + 1.2)),))
    report = check_room_shell(room, settings)
    if blocker is None:
        assert report.ok, report.problems
    else:
        assert_blocked(report, blocker)
    assert signed_volume(panel) == pytest.approx(volume, rel=1e-8)


@pytest.mark.parametrize(("bounds", "speaker", "inside"), [
    ((-.105275, .0625, -.098675, .144325, 1., .134925), (.02, .3, .03), False),
    ((-.105275, -1., -.098675, .28865, .30645, .6), (.02, 0., .03), True),
], ids=["outside", "inside"])
def test_two_grazing_rays_do_not_decide_the_seat(settings: RoomShellImportSettings,
                                                 bounds: tuple[float, float, float, float, float, float],
                                                 speaker: Vec3, inside: bool) -> None:
    # 前兩條射線都擦共用邊；由盒子三軸的上下界可直接判原點在不在裡面。
    room = good_room()
    room = replace(room, shell=box(*bounds, "Wall", inward=True), solids=(),
                   speakers=(replace(room.speakers[0], position=speaker),),
                   seats=(replace(room.seats[0], position=(0., 0., 0.)),))
    report = check_room_shell(room, settings)
    if inside:
        assert report.ok, report.problems
    else:
        assert_blocked(report, "placement_outside")
        assert all("座位" in p.text for p in report.problems if p.code == "placement_outside")


def test_each_placement_uses_its_own_clean_rays(settings: RoomShellImportSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    # 前六個方向從喇叭射出會擦天花的 x=y 對角線，第七個從座位射出會擦側牆的 y=z 對角線。
    directions = tuple((d, d, 1.) for d in (-.25, -.125, -.0625, .0625, .125, .25)) + ((1., .125, .125),)
    monkeypatch.setattr(room_shell, "_RAY_DIRECTIONS", directions)
    room = good_room()
    room = replace(room, shell=box(0, 0, 0, 1, 1, 1, "Wall", inward=True), solids=(),
                   speakers=(replace(room.speakers[0], position=(.25, .25, .5)),),
                   seats=(replace(room.seats[0], position=(.25, .625, .625)),))
    report = check_room_shell(room, settings)
    assert report.ok, report.problems


def test_seat_in_overlapping_pieces_of_one_solid_is_blocked(settings: RoomShellImportSettings) -> None:
    a = box(1, 1, .5, 3, 3, 2.5, "foam", inward=False)
    b = box(2, 1.5, .75, 4, 3.5, 2.75, "foam", inward=False)
    room = good_room()
    room = replace(room, solids=(Solid("foam", a + b),), speakers=(replace(room.speakers[0], position=(.3, .4, 1)),),
                   seats=(replace(room.seats[0], position=(2.5, 2, 1.5)),))
    report = check_room_shell(room, settings)
    assert_blocked(report, "placement_in_solid")
    buried, = (p for p in report.problems if p.code == "placement_in_solid")
    assert "SEAT_MAIN" in buried.text and "foam" in buried.text


def test_chained_weld_reports_affected_layers(settings: RoomShellImportSettings) -> None:
    room = good_room()
    steps = iter(i * .00009 for i in range(6))
    shell = tuple(LayerTriangle("Floor" if i < 2 else t.layer,
                                *((next(steps), 0., 0.) if p == (0, 0, 0) else p for p in (t.a, t.b, t.c)))
                  for i, t in enumerate(room.shell))
    report = check_room_shell(replace(room, shell=shell), settings)
    assert_blocked(report, "weld_spread")
    text = next(p.text for p in report.problems if p.code == "weld_spread")
    assert "Floor" in text and "Wall" in text


def test_weld_cluster_within_tolerance_pairwise_is_not_blocked(settings: RoomShellImportSettings) -> None:
    # 盒角的幾次出現輪流錯開到三個軸上：兩兩相距 0.9 個焊點距離（沒有接力），
    # 但外接盒對角線超過焊點距離；量外接盒會誤擋，這一題釘住不擋。
    room = good_room()
    d = 0.9 * settings.weld_m / 2 ** 0.5
    offsets = iter([(d, 0., 0.), (0., d, 0.), (0., 0., d)] * 4)
    shell = tuple(LayerTriangle(t.layer, *(next(offsets) if p == (0, 0, 0) else p for p in (t.a, t.b, t.c)))
                  for t in room.shell)
    report = check_room_shell(replace(room, shell=shell), settings)
    assert "weld_spread" not in {p.code for p in report.problems}
    assert report.ok, report.problems


@pytest.mark.parametrize("layer", ["ROOM::Wall_乳膠漆", "TREAT::Cloud_木板"], ids=["room", "treat"])
def test_faceless_mesh_is_unreadable(settings: RoomShellImportSettings, tmp_path: Path, layer: str) -> None:
    model = rhino3dm.File3dm.Read(str(TEST_ROOM))
    assert model is not None
    attrs = rhino3dm.ObjectAttributes()
    attrs.LayerIndex, attrs.Name = _layer_index(model, layer), "faceless"
    mesh = rhino3dm.Mesh()
    mesh.Vertices.Add(1., 1., 1.)
    objects = model.Objects
    assert hasattr(objects, "AddMesh")  # 寫考題用的 API 超出匯入器唯讀存根的範圍。
    objects.AddMesh(mesh, attrs)
    edited = tmp_path / "faceless.3dm"
    assert model.Write(str(edited), 8)
    room = read_rhino_room(edited, settings.max_file_bytes)
    assert any("faceless" in note and "沒有面" in note for note in room.unreadable)
    assert_blocked(check_room_shell(room, settings), "unreadable")


def test_empty_solid_from_another_importer_is_blocked(settings: RoomShellImportSettings) -> None:
    room = replace(good_room(), solids=(Solid("empty", ()),))
    assert_blocked(check_room_shell(room, settings), "empty_solid")


def test_rhino_speaker_with_nan_facing_is_blocked(settings: RoomShellImportSettings, tmp_path: Path) -> None:
    model = rhino3dm.File3dm.Read(str(TEST_ROOM))
    assert model is not None
    attrs = rhino3dm.ObjectAttributes()
    attrs.LayerIndex = _layer_index(model, "SPEAKER")
    reference = next(obj.Geometry for obj in model.Objects if obj.Attributes.LayerIndex == attrs.LayerIndex
                     and isinstance(obj.Geometry, rhino3dm.InstanceReference))
    transform = rhino3dm.Transform.Translation(2., 1., 1.)
    transform.M01 = float("nan")
    model.Objects.AddInstanceObject(rhino3dm.InstanceReference(reference.ParentIdefId, transform), attrs)
    edited = tmp_path / "nan-facing.3dm"
    assert model.Write(str(edited), 8)
    room = read_rhino_room(edited, settings.max_file_bytes)
    assert_blocked(check_room_shell(room, settings), "bad_placement")


@pytest.mark.parametrize("speaker", [True, False], ids=["speaker", "seat"])
@pytest.mark.parametrize("position", [True, False], ids=["position", "facing"])
def test_nonfinite_placement_is_blocked(settings: RoomShellImportSettings, speaker: bool, position: bool) -> None:
    room = good_room()
    item = room.speakers[0] if speaker else room.seats[0]
    bad = replace(item, position=(1., float("inf"), 1.)) if position else replace(item, facing=(0., 1., float("nan")))
    room = replace(room, speakers=(bad,)) if speaker else replace(room, seats=(bad,))
    assert_blocked(check_room_shell(room, settings), "bad_placement")


@pytest.mark.parametrize("shell", [True, False], ids=["shell-nan", "solid-inf"])
def test_nonfinite_vertex_is_blocked_before_geometry(settings: RoomShellImportSettings, shell: bool) -> None:
    room = good_room()
    if shell:
        room = replace(room, shell=(replace(room.shell[0], b=(1., float("nan"), 1.)), *room.shell[1:]))
    else:
        solid, = room.solids
        faces = (*solid.triangles[:-1], replace(solid.triangles[-1], c=(1., 1., float("inf"))))
        room = replace(room, solids=(replace(solid, triangles=faces),))
    report = check_room_shell(room, settings)
    assert_blocked(report, "bad_coordinates")
    assert (report.plane_count, report.area_by_layer, report.parallel_pairs) == (0, (), ())


def test_nonmanifold_shell_edge_is_blocked(settings: RoomShellImportSettings) -> None:
    # 重複一面使它的三條邊各接三面；只驗破洞的考題漏得過這種未封閉。
    room = good_room()
    assert_blocked(check_room_shell(replace(room, shell=(*room.shell, room.shell[0])), settings), "not_closed")


def test_millimeter_file_is_blocked_on_readback(settings: RoomShellImportSettings, tmp_path: Path) -> None:
    model = rhino3dm.File3dm.Read(str(TEST_ROOM))
    assert model is not None
    units = rhino3dm.UnitSystem
    assert hasattr(units, "Millimeters")  # 存根只列匯入器用到的 Meters，考題用真套件寫毫米檔。
    model.Settings.ModelUnitSystem = units.Millimeters
    edited = tmp_path / "millimeters.3dm"
    assert model.Write(str(edited), 8)
    room = read_rhino_room(edited, settings.max_file_bytes)
    assert room.units == "Millimeters"
    assert_blocked(check_room_shell(room, settings), "units")
