"""自訂房間（房間殼）高音那一路的考卷：逐面鏡像法、遮擋、晚期能量交換、接成報表四欄。

第一關是第五版計畫第 11 條的「長方體畫成碎面走新路線，要重現舊答案」：同一個長方體分別交給
長方體那一路（:mod:`aosr.physics.room_paths`、:mod:`aosr.physics.geometric_lane`）與房間殼這一路，
鏡像路徑逐階要一條不差、壓力要對到捨入量級；晚期混響兩路分格方式不同，只要求在同一個收斂值附近。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.config.room_shell_import import RoomShellImportSettings, load_room_shell_import_settings
from aosr.config.shell_lane import SHELL_GEOMETRY_TOL_M
from aosr.geometry.rhino_room import read_rhino_room
from aosr.geometry.room_shell import LayerTriangle, RoomShell, Solid
from aosr.geometry.shell_faces import ShellFaces, shell_faces
from aosr.geometry.shoebox import Point, Room
from aosr.physics.amplitude import Materials
from aosr.physics.geometric_lane import solve_geometric_lane
from aosr.physics import late_energy
from aosr.physics.late_energy import LateEnergyInputs, solve_late_energy_by_order
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.room_paths import image_source_paths
from aosr.physics.shell_geometric_lane import prepare_shell_acoustics, solve_shell_geometric_lane
from aosr.physics.shell_late_energy import (
    normalized_form_factors, shell_late_geometry, solve_shell_late_energy_by_order,
)
from aosr.physics.shell_paths import load_path_table, path_pressures, save_path_table, shell_path_table


TEST_ROOM = Path(__file__).parent / "data" / "room_shell_test_room.3dm"
L, W, H = 6.0, 4.0, 3.0
SOURCE = (1.3, 1.7, 1.2)
RECEIVER = (4.1, 2.3, 1.1)
C = 343.0
RHO_C = 413.0
FREQUENCIES = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
IMPEDANCE = {"floor": 3000 + 100j, "ceiling": 800 - 300j, "x0": 5000.0 + 0j,
             "xL": 2000 + 50j, "y0": 1500.0 + 0j, "yL": 9000.0 + 0j}
OMNI = SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL)
TOL = SHELL_GEOMETRY_TOL_M


@pytest.fixture(scope="module")
def settings() -> RoomShellImportSettings:
    return load_room_shell_import_settings(config_path("room_shell_import.toml"))


Corner = tuple[float, float, float]


def _quad(layer: str, origin: Corner, du: Corner, dv: Corner, n: int) -> list[LayerTriangle]:
    """一個矩形切成 n×n 格、每格兩個三角形；法線是 du×dv 的方向。"""
    p0, u, v = np.array(origin), np.array(du), np.array(dv)
    triangles = []
    for i in range(n):
        for j in range(n):
            a = p0 + u * i / n + v * j / n
            b, c, d = a + u / n, a + u / n + v / n, a + v / n
            triangles.append(LayerTriangle(layer, tuple(a), tuple(b), tuple(c)))
            triangles.append(LayerTriangle(layer, tuple(a), tuple(c), tuple(d)))
    return triangles


def _box(low: Corner, high: Corner, n: int, inward: bool, names: dict[str, str] | None = None) -> list[LayerTriangle]:
    """長方體六面：``inward`` 為真法線朝內（房間外殼），否則朝外（房內造型）。"""
    (x0, y0, z0), (x1, y1, z1) = low, high
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    faces = {  # 每一面：原點、兩個邊，du×dv 朝房內。
        "floor": ((x0, y0, z0), (dx, 0, 0), (0, dy, 0)), "ceiling": ((x0, y0, z1), (0, dy, 0), (dx, 0, 0)),
        "x0": ((x0, y0, z0), (0, dy, 0), (0, 0, dz)), "xL": ((x1, y0, z0), (0, 0, dz), (0, dy, 0)),
        "y0": ((x0, y0, z0), (0, 0, dz), (dx, 0, 0)), "yL": ((x0, y1, z0), (dx, 0, 0), (0, 0, dz)),
    }
    triangles = []
    for wall, (origin, du, dv) in faces.items():
        layer = (names or {}).get(wall, wall)
        triangles.extend(_quad(layer, origin, du, dv, n) if inward else _quad(layer, origin, dv, du, n))
    return triangles


def _shoebox_shell(n: int = 2, solids: tuple[Solid, ...] = ()) -> RoomShell:
    return RoomShell("m", tuple(_box((0, 0, 0), (L, W, H), n, inward=True)), solids, (), ())


def _faces(room: RoomShell, settings: RoomShellImportSettings) -> ShellFaces:
    return shell_faces(room, settings.coplanar_angle_deg, settings.coplanar_offset_m, TOL)


def _impedance_table(faces: ShellFaces) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    return np.array([[IMPEDANCE[layer]] * len(FREQUENCIES) for layer in faces.layers], dtype=np.complex128)


def test_faceted_shoebox_reproduces_old_image_paths(settings: RoomShellImportSettings) -> None:
    """長方體每面切成 2×2 格（每面八個三角形）走新路線：逐階路徑數與壓力跟舊程式一樣。"""
    faces = _faces(_shoebox_shell(), settings)
    assert faces.plane_count == 6 and faces.shell_convex and not faces.occluders.groups
    table = shell_path_table(faces, SOURCE, RECEIVER, max_order=3, tol_m=TOL)
    old = image_source_paths(Room(L, W, H), Point(*SOURCE), Point(*RECEIVER), C, max_order=3,
                             materials=Materials(RHO_C, FREQUENCIES, {k: (v,) * len(FREQUENCIES)
                                                                      for k, v in IMPEDANCE.items()}))
    pressure = path_pressures(table, _impedance_table(faces), RHO_C, C, np.array(FREQUENCIES))
    assert table.path_count == len(old) == 63 and table.blocked_count == 0
    for order in range(4):
        expected = np.sum([path.path_pressure for path in old if path.order == order], axis=0)
        assert int((table.order == order).sum()) == sum(1 for path in old if path.order == order)
        assert np.allclose(pressure[table.order == order].sum(axis=0), expected, rtol=1e-11, atol=0)
    old_lengths = sorted(path.dist_m for path in old)
    assert np.allclose(np.sort(table.dist_m), old_lengths, rtol=1e-13, atol=0)


def test_faceted_shoebox_lane_matches_old_lane(settings: RoomShellImportSettings) -> None:
    """整條高音路（散射全部 0.1）：直達逐位相同、早期兩欄對到捨入量級、晚期與總量只差分格。"""
    prepared = prepare_shell_acoustics(_shoebox_shell(), SOURCE, RECEIVER, settings)
    new = solve_shell_geometric_lane(prepared=prepared, source_model=OMNI, sound_speed_m_s=C,
                                     rho_c_pa_s_per_m=RHO_C, frequencies_hz=FREQUENCIES,
                                     impedance_by_layer=IMPEDANCE)
    old = solve_geometric_lane(source_model=OMNI, room=Room(L, W, H), source=Point(*SOURCE),
                               receiver=Point(*RECEIVER), sound_speed_m_s=C, rho_c_pa_s_per_m=RHO_C,
                               frequencies_hz=FREQUENCIES, impedance_by_wall=IMPEDANCE,
                               scattering_by_wall={wall: 0.1 for wall in IMPEDANCE})
    assert new.direct_energy == old.direct_energy
    assert new.scattering == old.scattering
    for name in ("reflected_energy", "interference_energy"):
        assert np.allclose(getattr(new, name), getattr(old, name), rtol=1e-10, atol=0)
    # 兩路分格不同：長方體每面 6×6 中心點、這一路約 1 公尺三角格多點積分，量到差 0.5% 左右。
    assert np.allclose(new.late_energy, old.late_energy, rtol=0.01, atol=0)
    assert np.allclose(new.geometric_energy, old.geometric_energy, rtol=0.005, atol=0)


def test_late_energy_converges_to_the_same_value_as_the_shoebox_lane(settings: RoomShellImportSettings) -> None:
    """吸音很不均勻的長方體：舊程式加密（每面 6→18 格）一路往下走，新路線比 6 格更接近 18 格。

    2026-11-27 量到：舊 6 格 1.13218、舊 18 格 1.12755、新路線 1.12705；新路線再加密（0.35 公尺）是 1.12456，
    兩路往同一個值收斂。
    """
    alpha = {"floor": (0.05,), "ceiling": (0.8,), "x0": (0.1,), "xL": (0.3,), "y0": (0.02,), "yL": (0.05,)}
    old = {n: solve_late_energy_by_order(LateEnergyInputs(Room(L, W, H), RHO_C, (250.0,), dict(alpha), n_per_wall=n,
                                                          domain_alpha_bar_max=math.inf), max_order=3)
           .bands[0].late_reverberant_energy for n in (6, 18)}
    faces = _faces(_shoebox_shell(1), settings)
    geometry = shell_late_geometry(faces)
    absorption = np.array([alpha[layer] for layer in faces.layers])
    new = solve_shell_late_energy_by_order(geometry, absorption, (250.0,), max_order=3).bands[0]
    assert new.late_reverberant_energy < old[6]
    assert math.isclose(new.late_reverberant_energy, old[18], rel_tol=1e-3)
    assert abs(new.late_reverberant_energy - old[18]) < abs(new.late_reverberant_energy - old[6])
    assert new.tail_energy >= 0.0
    assert math.isclose(sum(new.energy_by_order) + new.tail_energy, new.late_reverberant_energy, rel_tol=1e-12)


def test_form_factors_keep_reciprocity_and_closure(settings: RoomShellImportSettings) -> None:
    """交換後的形狀因子：每列加起來是 1、面積乘形狀因子對稱（互易）。"""
    geometry = shell_late_geometry(_faces(_shoebox_shell(1, (_cloud(),)), settings))
    exchange = geometry.areas[:, None] * geometry.form_factors
    assert np.allclose(geometry.form_factors.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    assert np.allclose(exchange, exchange.T, rtol=1e-9, atol=1e-15)


def _cloud(low: Corner = (2.0, 1.0, 2.4), high: Corner = (4.5, 3.0, 2.5), name: str = "雲") -> Solid:
    return Solid(name, tuple(_box(low, high, 1, inward=False, names={w: "Cloud" for w in IMPEDANCE})))


def test_cloud_blocks_ceiling_reflection_and_adds_its_own(settings: RoomShellImportSettings) -> None:
    """天雲擋住天花板那一條一階反射（反射點在雲正上方），換成雲底面自己那一條。"""
    faces = _faces(_shoebox_shell(2, (_cloud(),)), settings)
    table = shell_path_table(faces, SOURCE, RECEIVER, max_order=1, tol_m=TOL)
    first = [faces.layers[int(layer)] for layer in table.bounce_layer[table.order == 1, 0]]
    assert sorted(first) == ["Cloud", "floor", "x0", "xL", "y0", "yL"]
    assert table.blocked_count >= 1
    cloud_path = int(np.flatnonzero(table.order == 1)[first.index("Cloud")])
    image = np.array([SOURCE[0], SOURCE[1], 2 * 2.4 - SOURCE[2]])
    assert table.dist_m[cloud_path] == pytest.approx(float(np.linalg.norm(image - np.array(RECEIVER))), rel=1e-12)
    assert table.bounce_points[cloud_path, 0, 2] == pytest.approx(2.4, abs=1e-12)


def test_direct_path_blocked_by_a_solid_is_refused(settings: RoomShellImportSettings) -> None:
    column = _cloud((2.5, 1.5, 0.0), (3.0, 2.5, 3.0), "柱子")
    faces = _faces(_shoebox_shell(2, (column,)), settings)
    with pytest.raises(ValueError, match="擋住"):
        shell_path_table(faces, SOURCE, RECEIVER, max_order=1, tol_m=TOL)


def test_panel_back_against_the_wall_is_hidden(settings: RoomShellImportSettings) -> None:
    """貼牆板子背面整片看不到任何一格，從交換裡拿掉，面積記下來。"""
    panel = _cloud((0.0, 1.0, 1.0), (0.1, 2.0, 2.0), "板子")
    geometry = shell_late_geometry(_faces(_shoebox_shell(1, (panel,)), settings))
    assert geometry.hidden_area_m2 >= 1.0 - 1e-9
    assert np.allclose(geometry.form_factors.sum(axis=1), 1.0, rtol=0, atol=1e-9)


def test_flush_panel_also_hides_the_wall_it_covers(settings: RoomShellImportSettings) -> None:
    """1×1 公尺貼牆板：背面 1 m² 加上被蓋住的牆 1 m² 都不算（只要有一個取樣點看得到就算整格的舊寫法只扣 1.375）。"""
    panel = _cloud((0.0, 1.0, 1.0), (0.1, 2.0, 2.0), "板子")
    geometry = shell_late_geometry(_faces(_shoebox_shell(1, (panel,)), settings))
    assert geometry.hidden_area_m2 == pytest.approx(2.0, abs=0.1)


def _panel_with_back(back: str) -> Solid:
    names = {wall: "Panel" for wall in IMPEDANCE} | {"x0": back}
    return Solid("板子", tuple(_box((0.0, 1.0, 1.0), (0.1, 2.0, 2.0), 1, inward=False, names=names)))


def test_scattering_of_a_layer_nobody_sees_does_not_move_the_room_average(settings: RoomShellImportSettings) -> None:
    """板子背面（整片貼牆、晚期已剔掉）換散射係數，全房平均散射與晚期能量都不動。"""
    prepared = prepare_shell_acoustics(_shoebox_shell(2, (_panel_with_back("Back"),)), SOURCE, RECEIVER, settings,
                                       reflection_order_k=1)
    impedance = IMPEDANCE | {"Panel": 3000.0 + 0j, "Back": 3000.0 + 0j}
    lanes = [solve_shell_geometric_lane(prepared=prepared, source_model=OMNI, sound_speed_m_s=C,
                                        rho_c_pa_s_per_m=RHO_C, frequencies_hz=FREQUENCIES,
                                        impedance_by_layer=impedance,
                                        scattering_by_layer={layer: 0.0 for layer in impedance} | {"Back": value})
             for value in (0.0, 1.0)]
    assert lanes[0].scattering == lanes[1].scattering == (0.0,) * len(FREQUENCIES)
    assert lanes[0].late_energy == lanes[1].late_energy


def test_fully_absorbing_room_scattering_is_zero_like_the_box_lane(settings: RoomShellImportSettings) -> None:
    """每一面阻抗都等於 ρc（完全不反射）：權重和為 0，跟長方體那一路一樣回 0，不是那個共同的散射值。"""
    prepared = prepare_shell_acoustics(_shoebox_shell(), SOURCE, RECEIVER, settings, reflection_order_k=1)
    lane = solve_shell_geometric_lane(prepared=prepared, source_model=OMNI, sound_speed_m_s=C, rho_c_pa_s_per_m=RHO_C,
                                      frequencies_hz=FREQUENCIES, impedance_by_layer={k: RHO_C for k in IMPEDANCE},
                                      scattering_by_layer={k: 0.7 for k in IMPEDANCE})
    old = solve_geometric_lane(source_model=OMNI, room=Room(L, W, H), source=Point(*SOURCE),
                               receiver=Point(*RECEIVER), sound_speed_m_s=C, rho_c_pa_s_per_m=RHO_C,
                               frequencies_hz=FREQUENCIES, impedance_by_wall={k: RHO_C for k in IMPEDANCE},
                               scattering_by_wall={k: 0.7 for k in IMPEDANCE}, reflection_order_k=1)
    assert lane.scattering == old.scattering == (0.0,) * len(FREQUENCIES)


def test_nan_scattering_is_refused(settings: RoomShellImportSettings) -> None:
    prepared = prepare_shell_acoustics(_shoebox_shell(), SOURCE, RECEIVER, settings, reflection_order_k=1)
    with pytest.raises(ValueError, match="散射係數"):
        solve_shell_geometric_lane(prepared=prepared, source_model=OMNI, sound_speed_m_s=C, rho_c_pa_s_per_m=RHO_C,
                                   frequencies_hz=FREQUENCIES, impedance_by_layer=IMPEDANCE,
                                   scattering_by_layer={k: 0.1 for k in IMPEDANCE} | {"floor": math.nan})


def test_scattering_follows_each_bounce_material(settings: RoomShellImportSettings) -> None:
    """只有天花板散射 0.5、其他 0：鏡面那一路每條路徑留下 √0.5 的天花板次數次方。"""
    prepared = prepare_shell_acoustics(_shoebox_shell(), SOURCE, RECEIVER, settings, reflection_order_k=2)
    scattering = {layer: 0.0 for layer in IMPEDANCE} | {"ceiling": 0.5}
    lane = solve_shell_geometric_lane(prepared=prepared, source_model=OMNI, sound_speed_m_s=C,
                                      rho_c_pa_s_per_m=RHO_C, frequencies_hz=FREQUENCIES,
                                      impedance_by_layer=IMPEDANCE, scattering_by_layer=scattering)
    paths = prepared.paths
    pressure = path_pressures(paths, np.array([[IMPEDANCE[layer]] * len(FREQUENCIES) for layer in paths.layers]),
                              RHO_C, C, np.array(FREQUENCIES))
    ceiling = paths.layers.index("ceiling")
    hits = (paths.bounce_layer == ceiling).sum(axis=1)
    expected = (pressure[1:] * math.sqrt(0.5) ** hits[1:, None]).sum(axis=0)
    assert np.allclose(lane.reflected_energy, np.abs(expected) ** 2, rtol=1e-12, atol=0)
    assert 0.0 < lane.scattering[0] < 0.5


def test_missing_impedance_names_the_layers(settings: RoomShellImportSettings) -> None:
    prepared = prepare_shell_acoustics(_shoebox_shell(), SOURCE, RECEIVER, settings, reflection_order_k=1)
    with pytest.raises(ValueError, match="ceiling"):
        solve_shell_geometric_lane(prepared=prepared, source_model=OMNI, sound_speed_m_s=C, rho_c_pa_s_per_m=RHO_C,
                                   frequencies_hz=FREQUENCIES,
                                   impedance_by_layer={k: v for k, v in IMPEDANCE.items() if k != "ceiling"})


def test_path_table_round_trips_through_a_file(settings: RoomShellImportSettings, tmp_path: Path) -> None:
    table = shell_path_table(_faces(_shoebox_shell(2, (_cloud(),)), settings), SOURCE, RECEIVER, max_order=2,
                             tol_m=TOL)
    target = tmp_path / "paths.npz"
    save_path_table(table, target)
    loaded = load_path_table(target)
    assert loaded.layers == table.layers and loaded.max_order == table.max_order
    assert loaded.source == table.source and loaded.receiver == table.receiver
    for name in ("order", "dist_m", "bounce_layer", "bounce_plane", "bounce_cos", "bounce_points", "departure"):
        assert np.array_equal(getattr(loaded, name), getattr(table, name))


def test_owner_test_room_path_table(settings: RoomShellImportSettings) -> None:
    """KKT 的測試房（32 個平面、天雲、凸出的吸音板）：兩支喇叭到主聽位的有效路徑逐階數目釘住。"""
    room = read_rhino_room(TEST_ROOM, settings.max_file_bytes)
    faces = _faces(room, settings)
    assert faces.plane_count == 32 and not faces.shell_convex
    seat = room.seats[0].position
    counts = []
    for speaker in room.speakers:
        table = shell_path_table(faces, speaker.position, seat, max_order=3, tol_m=TOL)
        counts.append((np.bincount(table.order).tolist(), table.blocked_count))
        assert np.all(table.dist_m[1:] > table.dist_m[0])
        assert np.all((table.bounce_cos[table.bounce_layer >= 0] > 0) & (table.bounce_cos[table.bounce_layer >= 0] <= 1))
    assert counts == [([1, 6, 18, 34], 86), ([1, 6, 17, 32], 119)]


def _l_shaped_room() -> RoomShell:
    """6×4×3 公尺扣掉 x≥3、y≥2 那一角的 L 形外殼（不是凸的），法線朝內。"""
    outline = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (3.0, 2.0), (3.0, 4.0), (0.0, 4.0)]
    fan = [(outline[0], outline[i], outline[i + 1]) for i in range(1, len(outline) - 1)]
    triangles = []
    for a, b, c in fan:
        triangles.append(LayerTriangle("floor", (a[0], a[1], 0.0), (b[0], b[1], 0.0), (c[0], c[1], 0.0)))
        triangles.append(LayerTriangle("ceiling", (a[0], a[1], H), (c[0], c[1], H), (b[0], b[1], H)))
    for index, (p, q) in enumerate(zip(outline, outline[1:] + outline[:1], strict=True)):
        low_p, low_q, high_q, high_p = (p[0], p[1], 0.0), (q[0], q[1], 0.0), (q[0], q[1], H), (p[0], p[1], H)
        triangles.extend((LayerTriangle(f"w{index}", low_p, high_q, low_q),
                          LayerTriangle(f"w{index}", low_p, high_p, high_q)))
    return RoomShell("m", tuple(triangles), (), (), ())


def test_points_outside_a_non_convex_room_are_refused(settings: RoomShellImportSettings) -> None:
    """審查抓到的：L 形房的聲源放在牆外、貼在牆上、或放進缺掉的那一角，都要擋下，不能默默算。"""
    faces = _faces(_l_shaped_room(), settings)
    assert not faces.shell_convex
    table = shell_path_table(faces, (1.0, 1.0, 1.2), (4.5, 1.0, 1.1), max_order=2, tol_m=TOL)
    assert table.path_count > 1
    for outside in ((-0.2, 1.0, 1.2), (0.0, 1.0, 1.2), (4.5, 3.0, 1.2)):
        with pytest.raises(ValueError, match="不在房間裡面"):
            shell_path_table(faces, outside, (1.0, 1.0, 1.1), max_order=1, tol_m=TOL)


def test_thin_panel_on_a_wall_keeps_its_own_reflection(settings: RoomShellImportSettings) -> None:
    """審查抓到的：4 毫米薄板貼在 y=0 牆上，在共平面界線（5 毫米）以內也不准跟牆併成一面，板子正面的反射要在。"""
    panel = Solid("薄板", tuple(_box((1.0, 0.0, 0.5), (5.0, 0.004, 2.5), 1, inward=False,
                                     names={wall: "Panel" for wall in IMPEDANCE})))
    faces = _faces(_shoebox_shell(2, (panel,)), settings)
    table = shell_path_table(faces, SOURCE, RECEIVER, max_order=1, tol_m=TOL)
    first = [faces.layers[int(layer)] for layer in table.bounce_layer[table.order == 1, 0]]
    assert "Panel" in first


def test_floor_to_ceiling_pillar_hides_the_floor_and_ceiling_under_it(settings: RoomShellImportSettings) -> None:
    """審查抓到的：落地頂天的柱子，柱底下的地板與柱頂上的天花板彼此看得到是漏能量；它們要整片看不到任何一格。"""
    pillar = _cloud((2.0, 1.0, 0.0), (4.0, 3.0, H), "柱子")
    faces = _faces(_shoebox_shell(1, (pillar,)), settings)
    geometry = shell_late_geometry(faces)
    under = ((geometry.centroids[:, 0] > 2.0) & (geometry.centroids[:, 0] < 4.0) & (geometry.centroids[:, 1] > 1.0)
             & (geometry.centroids[:, 1] < 3.0) & ((geometry.centroids[:, 2] < 1e-9) | (geometry.centroids[:, 2] > H - 1e-9)))
    leaked = geometry.form_factors[np.ix_(under, under)].sum() if under.any() else 0.0
    assert leaked == pytest.approx(0.0, abs=1e-12)
    assert geometry.hidden_area_m2 >= 8.0 - 1e-9


def test_normalization_copy_matches_the_shoebox_lane_bit_for_bit() -> None:
    """正規化那一段是從長方體那一路抄來的：同一份中心點原始值餵兩邊，結果要逐位相同。"""
    patches = late_energy._patch_geometry(Room(L, W, H), 6)
    displacement = patches.centroids[None, :, :] - patches.centroids[:, None, :]
    distance_squared = np.sum(displacement * displacement, axis=-1)
    valid = distance_squared > 1e-12
    safe = np.where(valid, distance_squared, 1.0)
    direction = displacement * np.where(valid, 1.0 / np.sqrt(safe), 0.0)[..., None]
    cos_i = np.einsum("ik,ijk->ij", patches.normals, direction)
    cos_j = np.einsum("jk,ijk->ij", patches.normals, -direction)
    raw = cos_i * cos_j * patches.areas[None, :] / (np.pi * safe)
    raw = np.where(valid & (raw > 0.0), raw, 0.0)
    assert np.array_equal(normalized_form_factors(raw, patches.areas), late_energy._form_factors(patches))
