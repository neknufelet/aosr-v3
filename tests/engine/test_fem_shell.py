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

from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.room_shell_import import load_room_shell_import_settings
from aosr.geometry.rhino_room import read_rhino_room
from aosr.geometry.room_shell import LayerTriangle, RoomShell, Solid, signed_volume
from aosr.geometry.room_shell_mesh import ShellMesh, generate_room_shell_mesh
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.physics.fem_helmholtz import solve_fem_helmholtz
from aosr.physics.fem_shell import assemble_shell_operators, rigid_mode_frequencies, solve_shell_responses

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
