"""任意形狀房間的低頻有限元素（票 #1310）。

三道關：①同一張長方體網格，用「每圖層一份邊界」那一路算出來要跟舊的「每面牆一份」那一路
逐點相同（只差浮點加法次序）；②長方體畫成碎面走新的切網格那一路，硬牆模態要對上解析解；
③老闆的測試房切得出來、空氣體積等於外殼扣掉天雲、每個材料圖層都在邊界上。
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.room_shell_import import load_room_shell_import_settings
from aosr.geometry.rhino_room import read_rhino_room
from aosr.geometry.room_shell import LayerTriangle, RoomShell, Solid, signed_volume
from aosr.geometry.room_shell_mesh import ShellMesh, generate_room_shell_mesh
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.physics.fem_helmholtz import solve_fem_helmholtz
from aosr.physics.fem_shell import _Pattern, assemble_shell_operators, rigid_mode_frequencies, solve_shell_responses

TEST_ROOM = Path(__file__).parent / "data" / "room_shell_test_room.3dm"

LENGTHS = (5.0, 4.0, 3.0)
WELD_M = load_room_shell_import_settings(config_path("room_shell_import.toml")).weld_m


def box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, layer: str,
        inward: bool) -> tuple[LayerTriangle, ...]:
    """長方體十二個三角形；``inward`` 為真時法線朝裡（外殼），否則朝外（房內造型）。"""
    c = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    triangles: list[LayerTriangle] = []
    for a, b, d, e in faces:
        p, q, r, s = (c[i] for i in ((a, e, d, b) if inward else (a, b, d, e)))
        triangles.extend((LayerTriangle(layer, p, q, r), LayerTriangle(layer, p, r, s)))
    return tuple(triangles)


@pytest.fixture(scope="module")
def air() -> PhysicsConstants:
    return load_physics_constants(config_path("physics_constants.toml"))


def _faceted_box_mesh(air: PhysicsConstants, max_frequency_hz: float) -> ShellMesh:
    shell = box(0, 0, 0, *LENGTHS, "Wall", inward=True)
    return generate_room_shell_mesh(RoomShell("m", shell, (), (), ()), weld_m=WELD_M,
                                    max_frequency_hz=max_frequency_hz, elements_per_wavelength=6,
                                    sound_speed_m_s=air.sound_speed_m_s, random_seed=1)


def test_layer_path_reproduces_the_wall_path_on_the_same_mesh(air: PhysicsConstants) -> None:
    """每面牆當成一個圖層：新的一路跟舊的一路在同一張網格上要算出同一個聲壓（含一面逐頻複數阻抗）。"""
    room = Room(*LENGTHS)
    old = generate_shoebox_mesh(room, max_frequency_hz=120.0, elements_per_wavelength=6,
                                sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    mesh = ShellMesh(old.nodes, old.tetrahedra, old.boundary_triangles, old.boundary_wall_indices,
                     Wall.wall_names())
    impedances: dict[Wall, float | None | list[complex]] = {
        Wall.FLOOR: 2000.0, Wall.CEILING: None, Wall.X0: [900 - 300j, 700 + 50j, 1200 - 800j, 650 + 0j],
        Wall.XL: None, Wall.Y0: 4000.0, Wall.YL: 1500.0}
    source, receiver = (1.1, 0.9, 1.2), (3.7, 2.6, 1.4)
    frequencies = [31.5, 47.0, 63.0, 88.0]
    expected = solve_fem_helmholtz(old, wall_impedances=impedances, source=Point(*source),
                                   receiver=Point(*receiver), frequencies_hz=frequencies,
                                   density_kg_m3=air.air_density_kg_m3, sound_speed_m_s=air.sound_speed_m_s)
    actual = solve_shell_responses(
        assemble_shell_operators(mesh), layer_impedances={w.wall_name(): z for w, z in impedances.items()},
        source=source, receivers=[receiver], frequencies_hz=frequencies,
        density_kg_m3=air.air_density_kg_m3, sound_speed_m_s=air.sound_speed_m_s)[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_faceted_box_rigid_modes_match_the_closed_form(air: PhysicsConstants) -> None:
    """長方體只畫成十二個大三角形，切網格、組裝、求模態全走新路，前十個模態對上 c/2·√Σ(n/L)²。"""
    mesh = _faceted_box_mesh(air, 120.0)
    assert mesh.volume_m3() == pytest.approx(math.prod(LENGTHS), rel=1e-12)
    assert mesh.layer_area_m2("Wall") == pytest.approx(94.0, rel=1e-12)
    found = rigid_mode_frequencies(assemble_shell_operators(mesh), count=10, sound_speed_m_s=air.sound_speed_m_s)
    closed = sorted(air.sound_speed_m_s / 2 * math.sqrt(sum((n / length) ** 2 for n, length in zip(index, LENGTHS)))
                    for index in itertools.product(range(4), repeat=3))[:10]
    assert found[0] == pytest.approx(0.0, abs=1e-3)
    np.testing.assert_allclose(found[1:], closed[1:], rtol=1e-3)


@pytest.mark.parametrize("impedance", [None, 2000.0 - 600j])
def test_faceted_box_response_matches_the_old_mesh(air: PhysicsConstants, impedance: complex | None) -> None:
    """兩張不同的網格（gmsh 盒子 vs 碎面殼）算同一間房（硬牆、六面同一個吸音），離共振遠的頻點差不到千分之五。

    比的尺是「千分之五 × 這幾點最大的聲壓」：有吸音時 61 Hz 那點聲壓只有最大值的 3%，
    兩張網格的離散誤差在這種接近抵銷的點上換成相對誤差會被放大到 1.5%，不是新路錯。
    """
    source, receiver = (1.1, 0.9, 1.2), (3.7, 2.6, 1.4)
    frequencies = [25.0, 47.0, 61.0]
    old = generate_shoebox_mesh(Room(*LENGTHS), max_frequency_hz=120.0, elements_per_wavelength=6,
                                sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    expected = solve_fem_helmholtz(old, wall_impedances={w: impedance for w in Wall.all()}, source=Point(*source),
                                   receiver=Point(*receiver), frequencies_hz=frequencies,
                                   density_kg_m3=air.air_density_kg_m3, sound_speed_m_s=air.sound_speed_m_s)
    actual = solve_shell_responses(
        assemble_shell_operators(_faceted_box_mesh(air, 120.0)), layer_impedances={"Wall": impedance},
        source=source, receivers=[receiver], frequencies_hz=frequencies,
        density_kg_m3=air.air_density_kg_m3, sound_speed_m_s=air.sound_speed_m_s)[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-3 * float(np.abs(expected).max()))
    # 離抵銷遠的兩點另外用相對誤差咬，免得低聲壓點的放寬把整題放鬆。
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=5e-3)


def test_faceted_box_layers_follow_the_split_faces(air: PhysicsConstants) -> None:
    """碎面長方體的地板放自己的圖層：gmsh 把大三角形切碎後，吸音要跟著地板走，不能貼到別面。

    對照是舊網格「只有地板吸音」；音源與接收點的高度刻意不上下對稱（0.4 與 2.2 m，房高 3 m），
    吸音若錯貼到等面積的天花板，結果就會不同。
    """
    source, receiver = (1.1, 0.9, 0.4), (3.7, 2.6, 2.2)
    frequencies = [25.0, 47.0]
    shell = tuple(LayerTriangle("Floor", t.a, t.b, t.c) if max(t.a[2], t.b[2], t.c[2]) == 0.0 else t
                  for t in box(0, 0, 0, *LENGTHS, "Wall", inward=True))
    mesh = generate_room_shell_mesh(RoomShell("m", shell, (), (), ()), weld_m=WELD_M, max_frequency_hz=120.0,
                                    elements_per_wavelength=6, sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    assert mesh.layer_area_m2("Floor") == pytest.approx(LENGTHS[0] * LENGTHS[1], rel=1e-12)
    floor = 300.0 - 200j
    old = generate_shoebox_mesh(Room(*LENGTHS), max_frequency_hz=120.0, elements_per_wavelength=6,
                                sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    expected = solve_fem_helmholtz(old, wall_impedances={w: (floor if w is Wall.FLOOR else None) for w in Wall.all()},
                                   source=Point(*source), receiver=Point(*receiver), frequencies_hz=frequencies,
                                   density_kg_m3=air.air_density_kg_m3, sound_speed_m_s=air.sound_speed_m_s)
    def solve(impedances: dict[str, complex | None]) -> NDArray[np.complex128]:
        return solve_shell_responses(assemble_shell_operators(mesh), layer_impedances=impedances, source=source,
                                     receivers=[receiver], frequencies_hz=frequencies,
                                     density_kg_m3=air.air_density_kg_m3, sound_speed_m_s=air.sound_speed_m_s)[:, 0]

    np.testing.assert_allclose(solve({"Floor": floor, "Wall": None}), expected, rtol=5e-3)
    swapped = solve({"Floor": None, "Wall": floor})
    assert np.all(np.abs(swapped - expected) > 0.05 * np.abs(expected))
    ceiling = tuple(LayerTriangle("Floor", t.a, t.b, t.c) if min(t.a[2], t.b[2], t.c[2]) == LENGTHS[2]
                    else LayerTriangle("Wall", t.a, t.b, t.c) for t in shell)
    mirrored = generate_room_shell_mesh(RoomShell("m", ceiling, (), (), ()), weld_m=WELD_M, max_frequency_hz=120.0,
                                        elements_per_wavelength=6, sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    on_ceiling = solve_shell_responses(assemble_shell_operators(mirrored), layer_impedances={"Floor": floor, "Wall": None},
                                       source=source, receivers=[receiver], frequencies_hz=frequencies,
                                       density_kg_m3=air.air_density_kg_m3, sound_speed_m_s=air.sound_speed_m_s)[:, 0]
    assert np.any(np.abs(on_ceiling - expected) > 0.01 * np.abs(expected))


def test_a_cell_that_cancels_to_exactly_zero_keeps_its_slot() -> None:
    """某頻 K 與 k²M 在一格剛好抵銷成零：非零位置仍要跟其他頻一樣，PARDISO 才能沿用同一個分解結構。"""
    stiffness = csr_matrix(np.array([[1.0, 0.5], [0.5, 2.0]]))
    mass = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    empty = csr_matrix((2, 2), dtype=np.float64)
    pattern = _Pattern(stiffness, mass, {"Hard": empty})
    # c = 2πf 讓 k² 恰為 1，(0, 0) 那格 1 − 1·1 精確為零。
    cancel = pattern.system({"Hard": None}, 1.0, 1.2, 2.0 * math.pi)
    other = pattern.system({"Hard": None}, 0.5, 1.2, 2.0 * math.pi)
    assert cancel.toarray()[0, 0] == 0.0
    assert np.array_equal(cancel.indptr, other.indptr) and np.array_equal(cancel.indices, other.indices)
    assert cancel.nnz == other.nnz == 4


def test_test_room_meshes_with_the_cloud_cut_out(air: PhysicsConstants) -> None:
    room = read_rhino_room(TEST_ROOM, 10**8)
    mesh = generate_room_shell_mesh(room, weld_m=WELD_M, max_frequency_hz=120.0, elements_per_wavelength=6,
                                    sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    cloud = sum(signed_volume(solid.triangles) for solid in room.solids)
    assert mesh.volume_m3() == pytest.approx(-signed_volume(room.shell) - cloud, rel=1e-9)
    shell_layers = {t.layer for t in room.shell} | {t.layer for s in room.solids for t in s.triangles}
    assert set(mesh.layer_names) == shell_layers
    for name in mesh.layer_names:
        source = [t for t in (*room.shell, *(t for s in room.solids for t in s.triangles)) if t.layer == name]
        area = sum(np.linalg.norm(np.cross(np.subtract(t.b, t.a), np.subtract(t.c, t.a))) / 2 for t in source)
        assert mesh.layer_area_m2(name) == pytest.approx(area, rel=1e-9)
    assert np.all(np.linalg.det(mesh.nodes[mesh.tetrahedra][:, 1:] - mesh.nodes[mesh.tetrahedra][:, :1]) > 0)


def test_missing_layer_impedance_and_source_inside_the_cloud_are_refused(air: PhysicsConstants) -> None:
    shell = box(0, 0, 0, *LENGTHS, "Wall", inward=True)
    cloud = Solid("cloud", box(1, 1, 2.0, 3, 3, 2.3, "Cloud", inward=False))
    mesh = generate_room_shell_mesh(RoomShell("m", shell, (cloud,), (), ()), weld_m=WELD_M,
                                    max_frequency_hz=80.0, elements_per_wavelength=6,
                                    sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    operators = assemble_shell_operators(mesh)
    def solve(impedances: dict[str, float | None], source: tuple[float, float, float]) -> None:
        solve_shell_responses(operators, layer_impedances=impedances, source=source, receivers=[(4.0, 3.0, 1.0)],
                              frequencies_hz=[40.0], density_kg_m3=air.air_density_kg_m3,
                              sound_speed_m_s=air.sound_speed_m_s)

    with pytest.raises(ValueError, match="Cloud"):
        solve({"Wall": None}, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        solve({"Wall": None, "Cloud": None}, (2.0, 2.0, 2.15))

def _subdivide(triangles: tuple[LayerTriangle, ...], n: int) -> tuple[LayerTriangle, ...]:
    """每個三角形用重心座標格切成 n² 個小三角形；共用的邊切法一樣，所以還是封閉的。"""
    result = []
    for t in triangles:
        a, b, c = (np.array(p) for p in (t.a, t.b, t.c))
        def at(i: int, j: int, a: NDArray[np.float64] = a, b: NDArray[np.float64] = b,
               c: NDArray[np.float64] = c) -> tuple[float, float, float]:
            p = a + (b - a) * i / n + (c - a) * j / n
            return float(p[0]), float(p[1]), float(p[2])
        for i in range(n):
            for j in range(n - i):
                result.append(LayerTriangle(t.layer, at(i, j), at(i + 1, j), at(i, j + 1)))
                if i + j < n - 1:
                    result.append(LayerTriangle(t.layer, at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)))
    return tuple(result)


def _shrink(triangles: tuple[LayerTriangle, ...], amount: float) -> tuple[LayerTriangle, ...]:
    """每個三角形的角往自己的中心挪 ``amount``：相鄰三角形之間留一條比焊點距離窄的縫。"""
    result = []
    for t in triangles:
        center = np.mean([t.a, t.b, t.c], axis=0)
        a, b, c = (np.add(p, amount * (center - p) / np.linalg.norm(center - p)) for p in (t.a, t.b, t.c))
        result.append(LayerTriangle(t.layer, (float(a[0]), float(a[1]), float(a[2])),
                                    (float(b[0]), float(b[1]), float(b[2])), (float(c[0]), float(c[1]), float(c[2]))))
    return tuple(result)


def _move(triangles: tuple[LayerTriangle, ...], degrees: float,
          offset: tuple[float, float, float]) -> tuple[LayerTriangle, ...]:
    """繞 z 軸轉 ``degrees`` 再平移 ``offset``（模擬測量座標裡沒對齊座標軸的房）。"""
    angle = math.radians(degrees)
    def place(p: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = p
        return (offset[0] + x * math.cos(angle) - y * math.sin(angle),
                offset[1] + x * math.sin(angle) + y * math.cos(angle), offset[2] + z)
    return tuple(LayerTriangle(t.layer, place(t.a), place(t.b), place(t.c)) for t in triangles)


@pytest.mark.parametrize("degrees, offset", [(0.0, (0.0, 0.0, 0.0)), (37.0, (1.0e6, -2.0e6, 50.0))])
def test_seams_narrower_than_the_weld_still_mesh(air: PhysicsConstants, degrees: float,
                                                 offset: tuple[float, float, float]) -> None:
    """面切成 144 塊、每個小三角形往中心縮 0.4 個焊點距離：焊完是封閉的合法房，體積核對不能把它擋掉；
    第二組再轉 37° 放到一百萬公尺外，遠座標也不能被浮點誤差誤擋。

    沒焊點的原始三角形之間有縫，拿它算應有體積會差出「縫的面積 × 房間尺寸」，比焊點距離 × 邊界面積還大。
    """
    def prepare(triangles: tuple[LayerTriangle, ...]) -> tuple[LayerTriangle, ...]:
        return _move(_shrink(_subdivide(triangles, 12), 0.4 * WELD_M), degrees, offset)

    shell = prepare(box(0, 0, 0, *LENGTHS, "Wall", inward=True))
    cloud = Solid("cloud", prepare(box(1, 1, 2.0, 3, 3, 2.3, "Cloud", inward=False)))
    area = 2 * (5 * 4 + 5 * 3 + 4 * 3) + 2 * (2 * 2 + 2 * 0.3 * 2)
    exact = math.prod(LENGTHS) - 2.0 * 2.0 * 0.3
    # 這組縫大到用沒焊點的三角形核對體積就會誤擋（考卷要能抓到那個錯）。
    unwelded = -signed_volume(shell) - signed_volume(cloud.triangles)
    assert abs(unwelded - exact) > WELD_M * area
    mesh = generate_room_shell_mesh(RoomShell("m", shell, (cloud,), (), ()), weld_m=WELD_M, max_frequency_hz=80.0,
                                    elements_per_wavelength=6, sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    local = mesh.nodes - mesh.nodes.min(axis=0)
    volume = float(np.linalg.det(local[mesh.tetrahedra][:, 1:] - local[mesh.tetrahedra][:, :1]).sum() / 6.0)
    # 焊點只把頂點挪到代表點，體積最多差「焊點距離 × 邊界面積」。
    assert volume == pytest.approx(exact, abs=WELD_M * area)
    corners = np.array([c for t in shell for c in (t.a, t.b, t.c)])
    np.testing.assert_allclose(mesh.nodes.min(axis=0), corners.min(axis=0), rtol=0, atol=WELD_M)
    np.testing.assert_allclose(mesh.nodes.max(axis=0), corners.max(axis=0), rtol=0, atol=WELD_M)


def test_a_large_room_far_from_the_origin_and_turned_still_meshes(air: PhysicsConstants) -> None:
    """50×40×30 m 的房轉 37° 放到一百萬公尺外：座標本身只準到 1e-10 m，不先搬回原點附近切，
    五萬個四面體的體積捨入就會超過「四面體數 × 焊點距離³」而誤擋一間合法的房。"""
    shell = _move(box(0, 0, 0, 50.0, 40.0, 30.0, "Wall", inward=True), 37.0, (1.0e6, 1.0e6, 0.0))
    mesh = generate_room_shell_mesh(RoomShell("m", shell, (), (), ()), weld_m=WELD_M, max_frequency_hz=20.0,
                                    elements_per_wavelength=6, sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
    assert mesh.volume_m3() == pytest.approx(50.0 * 40.0 * 30.0, rel=1e-9)
    corners = np.array([c for t in shell for c in (t.a, t.b, t.c)])
    np.testing.assert_allclose(mesh.nodes.min(axis=0), corners.min(axis=0), rtol=0, atol=WELD_M)
    np.testing.assert_allclose(mesh.nodes.max(axis=0), corners.max(axis=0), rtol=0, atol=WELD_M)


def test_nested_solids_are_refused_instead_of_meshing_nothing(air: PhysicsConstants) -> None:
    """造型套在造型裡：gmsh 可能安靜切出零個四面體，切完核對空氣體積要擋下來。"""
    shell = box(0, 0, 0, 5.0, 5.0, 5.0, "Wall", inward=True)
    outer = Solid("outer", box(1, 1, 1, 4, 4, 4, "Outer", inward=False))
    inner = Solid("inner", box(2, 2, 2, 3, 3, 3, "Inner", inward=False))
    with pytest.raises(ValueError, match="空氣體積"):
        generate_room_shell_mesh(RoomShell("m", shell, (outer, inner), (), ()), weld_m=WELD_M,
                                 max_frequency_hz=80.0, elements_per_wavelength=6,
                                 sound_speed_m_s=air.sound_speed_m_s, random_seed=1)
