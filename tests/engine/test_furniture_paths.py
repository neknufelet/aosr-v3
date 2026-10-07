"""家具一次反射與聲壓：手算鏡射、有限邊界、逐位慣例、身分閉包。"""
from __future__ import annotations

import cmath
import math

import pytest

from aosr.geometry.furniture import FaceDirection, FurnitureBox, Vec3
from aosr.physics.amplitude import CANONICAL_WALLS, Materials, path_pressure
from aosr.physics.finite_reflector import finite_size_energy
from aosr.physics.furniture_paths import (
    Furniture,
    FurniturePath,
    furniture_path_amplitude,
    furniture_path_pressure,
    furniture_reflection_coefficients,
    single_bounce_furniture_paths,
)
from tests.engine._precision_contracts import contract_value

# 判定檔第 3.1 節自選桌面算例的聲速與聲壓對照的介質條件。
TABLE_EXAMPLE_C = 343.0
PRESSURE_REFERENCE_RHO_C = 415.03


def table(identifier: str = "table", *, center: Vec3 = (0.0, 0.0, 0.5),
          width: float = 0.8, depth: float = 1.6, height: float = 0.1) -> Furniture:
    return Furniture(identifier, FurnitureBox(kind="desk", width_m=width, depth_m=depth,
                     height_m=height, bottom_center_m=center, margin_m=0.0))


def table_path() -> FurniturePath:
    paths = single_bounce_furniture_paths((-0.6, 0.0, 1.05), (0.6, 0.0, 1.05),
                                          (table(),), c=TABLE_EXAMPLE_C, margin_m=0.0)
    return next(path for path in paths if path.face_direction == FaceDirection.TOP)


def test_single_bounce_carries_true_image_lengths_and_departure() -> None:
    path = table_path()
    assert path.furniture_id == "table"
    assert path.hit == pytest.approx((0.0, 0.0, 0.6))
    assert path.image == pytest.approx((-0.6, 0.0, 0.15))
    assert (path.d_inc, path.d_refl, path.dist_m) == pytest.approx((0.75, 0.75, 1.5))
    assert path.delay_s == path.dist_m / TABLE_EXAMPLE_C
    assert path.cos_theta == pytest.approx(0.6)
    assert path.departure_direction == pytest.approx((0.8, 0.0, -0.6))
    assert math.dist(path.image, (0.6, 0.0, 1.05)) == pytest.approx(path.d_inc + path.d_refl)


@pytest.mark.parametrize("direction", list(FaceDirection))
def test_all_exposed_face_directions_mirror_actual_source(direction: FaceDirection) -> None:
    item = table(center=(2.0, 3.0, 1.0), width=2.0, depth=2.0, height=1.0)
    # 中心到各面一公尺，沿外法向再走兩公尺；真鏡像在面內側兩公尺。
    center: Vec3 = (2.0, 3.0, 1.5)
    offset = 0.5 if direction.axis == 2 else 1.0
    source = list(center)
    image = list(center)
    hit = list(center)
    source[direction.axis] += direction.sign * (offset + 2.0)
    image[direction.axis] += direction.sign * (offset - 2.0)
    hit[direction.axis] += direction.sign * offset
    point: Vec3 = (source[0], source[1], source[2])
    paths = single_bounce_furniture_paths(point, point, (item,), c=TABLE_EXAMPLE_C, margin_m=0.0)
    got = next(path for path in paths if path.face_direction == direction)
    assert got.hit == tuple(hit) and got.image == tuple(image)
    assert got.dist_m == 4.0 and got.cos_theta == 1.0
    assert math.dist(got.image, point) == got.d_inc + got.d_refl


def test_grounded_bottom_face_never_produces_a_path() -> None:
    item = Furniture("sofa", FurnitureBox(kind="sofa", width_m=2.0, depth_m=2.0,
                     height_m=1.0, bottom_center_m=(2.0, 2.0, 0.0), margin_m=0.0))
    assert single_bounce_furniture_paths((2.0, 2.0, -1.0), (2.0, 2.0, -2.0),
                                         (item,), c=TABLE_EXAMPLE_C, margin_m=0.0) == ()


@pytest.mark.parametrize(("source_z", "receiver_z"), [(0.4, 1.05), (1.05, 0.4), (0.4, 0.4)])
def test_back_side_has_no_top_reflection(source_z: float, receiver_z: float) -> None:
    paths = single_bounce_furniture_paths((-0.6, 0.0, source_z), (0.6, 0.0, receiver_z),
                                          (table(),), c=TABLE_EXAMPLE_C, margin_m=0.0)
    assert all(path.face_direction != FaceDirection.TOP for path in paths)


@pytest.mark.parametrize(("factor", "accepted"), [(0.5, True), (1.5, False)])
def test_hit_boundary_band_preserves_real_unclipped_point(factor: float, accepted: bool) -> None:
    margin = contract_value("furniture_geometry_contact") * 8.0
    x = 0.4 + factor * margin
    paths = single_bounce_furniture_paths((x, -0.6, 1.05), (x, 0.6, 1.05),
                                          (table(),), c=TABLE_EXAMPLE_C, margin_m=margin)
    top = tuple(path for path in paths if path.face_direction == FaceDirection.TOP)
    assert bool(top) is accepted
    if accepted:
        assert top[0].hit[0] == x and top[0].hit[0] > 0.4


def test_path_order_and_outputs_are_bitwise_deterministic() -> None:
    # 兩件分開的高盒子，聲源與接收點在中間；+y 與 −y 面各有有效反射。
    furniture = (table("z", height=2.0), table("a", center=(0.0, 3.0, 0.5), depth=1.0, height=2.0))
    args = ((-0.6, 1.5, 1.05), (0.6, 1.5, 1.05))
    first = single_bounce_furniture_paths(*args, furniture, c=TABLE_EXAMPLE_C, margin_m=0.0)
    second = single_bounce_furniture_paths(*args, tuple(reversed(furniture)), c=TABLE_EXAMPLE_C, margin_m=0.0)
    assert first and first == second
    keys = tuple((path.furniture_id, list(FaceDirection).index(path.face_direction)) for path in first)
    assert keys == tuple(sorted(keys))
    assert repr(first) == repr(single_bounce_furniture_paths(*args, furniture, c=TABLE_EXAMPLE_C, margin_m=0.0))


def test_each_leg_checks_every_box_including_reflecting_box(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.geometry.furniture import segment_blocked_by_box
    from aosr.physics import furniture_paths as core

    source, receiver = (-0.6, 0.0, 1.05), (0.6, 0.0, 1.05)
    reflecting, remote = table(), table("remote", center=(4.0, 4.0, 0.5))
    checked: list[tuple[Vec3, Vec3, FurnitureBox]] = []
    def observe(start: Vec3, end: Vec3, box: FurnitureBox, *, margin_m: float) -> bool:
        checked.append((start, end, box))
        return segment_blocked_by_box(start, end, box, margin_m=margin_m)
    monkeypatch.setattr(core, "segment_blocked_by_box", observe)
    paths = core.single_bounce_furniture_paths(source, receiver, (reflecting, remote),
                                              c=TABLE_EXAMPLE_C, margin_m=0.0)
    path = next(path for path in paths if path.furniture_id == "table")
    assert path.hit == pytest.approx((0.0, 0.0, 0.6))
    for item in (reflecting, remote):
        assert (source, path.hit, item.box) in checked
        assert (path.hit, receiver, item.box) in checked


def test_no_furniture_has_no_single_bounce_paths() -> None:
    assert single_bounce_furniture_paths((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (),
                                         c=TABLE_EXAMPLE_C, margin_m=0.0) == ()


def test_real_impedance_is_resolved_per_frequency() -> None:
    z = (2.0 * PRESSURE_REFERENCE_RHO_C, 4.0 * PRESSURE_REFERENCE_RHO_C, PRESSURE_REFERENCE_RHO_C / 0.6)
    got = furniture_reflection_coefficients(z, cos_theta=0.6, rho_c=PRESSURE_REFERENCE_RHO_C)
    assert got == pytest.approx(tuple(complex((value * 0.6 - PRESSURE_REFERENCE_RHO_C)
                                              / (value * 0.6 + PRESSURE_REFERENCE_RHO_C)) for value in z))
    assert got[0] != got[1]


def test_unit_correction_matches_wall_path_pressure_bitwise() -> None:
    frequencies = (125.0, 250.0, 1000.0)
    impedances = (2.0 * PRESSURE_REFERENCE_RHO_C, 4.0 * PRESSURE_REFERENCE_RHO_C, PRESSURE_REFERENCE_RHO_C / 0.6)
    coefficients = furniture_reflection_coefficients(impedances, cos_theta=0.6, rho_c=PRESSURE_REFERENCE_RHO_C)
    materials = Materials(PRESSURE_REFERENCE_RHO_C, frequencies,
                          {wall: tuple(complex(z) for z in impedances) for wall in CANONICAL_WALLS})
    expected = path_pressure(materials, 1.5, TABLE_EXAMPLE_C, coefficients)
    got = furniture_path_pressure(frequencies, dist_m=1.5, c=TABLE_EXAMPLE_C,
                                  refl_per_freq=coefficients, energy_factors=(1.0, 1.0, 1.0))
    assert tuple((z.real.hex(), z.imag.hex()) for z in got) == tuple((z.real.hex(), z.imag.hex()) for z in expected)


def test_table_125hz_pressure_uses_square_root_energy() -> None:
    path = table_path()
    coefficients, pressure = furniture_path_amplitude(path, (125.0,), (4.0 * PRESSURE_REFERENCE_RHO_C,),
                                                       rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    # 判定檔第 4.3 節：0.4732 是聲壓比，0.2239 是能量比。
    amplitude_ratio = abs(pressure[0]) * path.dist_m / abs(coefficients[0])
    assert round(amplitude_ratio, 4) == 0.4732
    assert round(amplitude_ratio ** 2, 4) == 0.2239


def test_path_amplitude_uses_the_paths_own_incidence_angle() -> None:
    # 桌面算例：比桌面高 0.45 m、水平半距 0.6 m，入射段 0.75 m，cosθ＝0.45/0.75＝0.6（手算）。
    path = table_path()
    z = 4.0 * PRESSURE_REFERENCE_RHO_C
    coefficients, _ = furniture_path_amplitude(path, (125.0,), (z,),
                                               rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    hand_cos = 0.6
    expected = (z * hand_cos - PRESSURE_REFERENCE_RHO_C) / (z * hand_cos + PRESSURE_REFERENCE_RHO_C)
    assert coefficients[0] == pytest.approx(complex(expected))
    normal = (z - PRESSURE_REFERENCE_RHO_C) / (z + PRESSURE_REFERENCE_RHO_C)
    assert abs(coefficients[0] - normal) > abs(expected - normal) / 2.0


def _original_form_energy(frequency: float, in_plane_edge: float, other_edge: float,
                          cos_theta: float, d_inc: float, d_refl: float, c: float) -> float:
    """入射平面平行一邊時的原文式（判定檔第 1.3 節）；考卷自己寫，不走產品的投影與歸一。"""
    a_star = 2.0 * d_inc * d_refl / (d_inc + d_refl)
    f_in = c * a_star / (2.0 * (in_plane_edge * cos_theta) ** 2)
    f_other = c * a_star / (2.0 * other_edge ** 2)
    return min(1.0, frequency / f_in) * min(1.0, frequency / f_other)


def test_assembled_amplitude_matches_hand_values_with_unequal_legs() -> None:
    # S(−0.4,0,1.2)、E(0.6,0,0.9)，桌頂 z=0.6：高差 0.6 與 0.3，反射點 x＝−0.4＋1.0·0.6/0.9。
    # 入射平面是 x–z 平面，平行 0.8 m 那條邊；兩段不等長，d_refl 傳錯就會差。
    source, receiver = (-0.4, 0.0, 1.2), (0.6, 0.0, 0.9)
    path = next(item for item in single_bounce_furniture_paths(source, receiver, (table(),), c=TABLE_EXAMPLE_C,
                                                               margin_m=0.0)
                if item.face_direction == FaceDirection.TOP)
    hit_x = -0.4 + 1.0 * 0.6 / 0.9
    d_inc = math.hypot(hit_x + 0.4, 0.6)
    d_refl = math.hypot(0.6 - hit_x, 0.3)
    cos_theta = 0.6 / d_inc
    z = 1.2 * PRESSURE_REFERENCE_RHO_C  # 靠近 ρc/cosθ：cosθ 用錯時 R 的正負會翻。
    coefficients, pressure = furniture_path_amplitude(path, (125.0, 500.0), (z, z),
                                                      rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    r = (z * cos_theta - PRESSURE_REFERENCE_RHO_C) / (z * cos_theta + PRESSURE_REFERENCE_RHO_C)
    for index, frequency in enumerate((125.0, 500.0)):
        energy = _original_form_energy(frequency, 0.8, 1.6, cos_theta, d_inc, d_refl, TABLE_EXAMPLE_C)
        assert coefficients[index] == pytest.approx(complex(r))
        assert abs(pressure[index]) == pytest.approx(abs(r) * math.sqrt(energy) / (d_inc + d_refl))


@pytest.mark.parametrize(("face", "source", "receiver", "in_plane_edge"), [
    # 沙發 0.75×1.8×0.65、底面中心 (2,2,0)：+x 面在 x=2.375（邊 y 1.8、z 0.65）；−y 面在 y=1.1（邊 x 0.75、z 0.65）。
    (FaceDirection.X_PLUS, (4.0, 1.0, 0.35), (4.0, 3.2, 0.35), 1.8),
    (FaceDirection.Y_MINUS, (1.5, 0.0, 0.35), (2.5, 0.0, 0.35), 0.75),
])
def test_vertical_faces_use_their_own_normal_and_edges(face: FaceDirection, source: Vec3, receiver: Vec3,
                                                       in_plane_edge: float) -> None:
    sofa = Furniture("sofa", FurnitureBox(kind="sofa", width_m=0.75, depth_m=1.8, height_m=0.65,
                                          bottom_center_m=(2.0, 2.0, 0.0), margin_m=0.0))
    path = next(item for item in single_bounce_furniture_paths(source, receiver, (sofa,), c=TABLE_EXAMPLE_C,
                                                               margin_m=0.0) if item.face_direction == face)
    # 聲源與接收點同高、對稱：反射點在兩點中間的投影上，兩段等長（手算）。
    plane_distance = 4.0 - 2.375 if face == FaceDirection.X_PLUS else 1.1 - 0.0
    half_span = (3.2 - 1.0) / 2.0 if face == FaceDirection.X_PLUS else (2.5 - 1.5) / 2.0
    d = math.hypot(plane_distance, half_span)
    frequencies = (125.0, 250.0, 500.0, 1000.0)
    got = finite_size_energy(frequencies, path.face, path.departure_direction,
                             d_inc=path.d_inc, d_refl=path.d_refl, c=TABLE_EXAMPLE_C)
    expected = tuple(_original_form_energy(f, in_plane_edge, 0.65, plane_distance / d, d, d, TABLE_EXAMPLE_C)
                     for f in frequencies)
    assert got == pytest.approx(expected)


def test_phase_uses_both_incident_and_reflected_distance() -> None:
    paths = single_bounce_furniture_paths((-0.4, 0.0, 1.2), (0.6, 0.0, 0.9),
                                          (table(),), c=TABLE_EXAMPLE_C, margin_m=0.0)
    path = next(path for path in paths if path.face_direction == FaceDirection.TOP)
    assert path.d_inc != path.d_refl
    _coefficients, got = furniture_path_amplitude(path, (125.0,), (4.0 * PRESSURE_REFERENCE_RHO_C,),
                                                 rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    # 手算真鏡像 (-.4,0,0) 到 E(.6,0,.9)；不能拿單段距離或兩倍入射距離。
    expected_phase = cmath.exp(-2j * math.pi * 125.0 * math.hypot(1.0, 0.9) / TABLE_EXAMPLE_C)
    assert got[0] / abs(got[0]) == pytest.approx(expected_phase)


def test_pressure_and_coefficients_repeat_bitwise() -> None:
    args = (table_path(), (125.0, 250.0, 1000.0), (2000.0, 3000.0, 4000.0))
    first = furniture_path_amplitude(*args, rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    second = furniture_path_amplitude(*args, rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    assert tuple((z.real.hex(), z.imag.hex()) for row in first for z in row) == (
        tuple((z.real.hex(), z.imag.hex()) for row in second for z in row))


def test_frequency_rows_must_align() -> None:
    with pytest.raises(ValueError):
        furniture_path_amplitude(table_path(), (125.0, 250.0), (2000.0,),
                                 rho_c=PRESSURE_REFERENCE_RHO_C, c=TABLE_EXAMPLE_C)
    with pytest.raises(ValueError):
        furniture_path_pressure((125.0,), dist_m=1.5, c=TABLE_EXAMPLE_C,
                                refl_per_freq=(0.5 + 0j,), energy_factors=())


def test_new_modules_are_outside_both_identity_closures() -> None:
    from aosr.reporting.modal_diagnosis import modal_identity, modal_import_closure
    from aosr.reporting.physics_identity import physics_import_closure

    for modules in (physics_import_closure().modules, modal_import_closure().modules):
        assert "aosr.physics.furniture_paths" not in modules
        assert "aosr.physics.finite_reflector" not in modules
    assert modal_identity() == "modal-v1:5dcb7e682c04552785515d72567623f0bd8c42ae82a742ac4b7f5be015955e9e"
