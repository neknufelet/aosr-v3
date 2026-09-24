"""票 #351：平行牆對與下一階幾何延遲的獨立考卷。"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Wall
from aosr.materials.response import MATERIAL_SCATTERING_DEFAULT_S
from aosr.physics.report_io import ReportInput, load_input_document, scene_fingerprint
from aosr.physics.reflection_screen import ReflectionScreen, build_reflection_screen


_FREQUENCIES = (125.0, 250.0)


def _inputs(**overrides: object) -> ReportInput:
    document: dict[str, object] = {
        "room_m": {"Lx": 5.0, "Ly": 7.0, "Lz": 9.0},
        "source_m": {"x": 1.0, "y": 2.0, "z": 3.0},
        "receiver_m": {"x": 3.0, "y": 4.0, "z": 5.0},
        "sound_speed_m_s": 320.0,
        "density_kg_m3": 1.25,
        "impedance_pa_s_per_m_by_wall": {
            wall.wall_name(): 1600.0 for wall in Wall.all()
        },
        "reflection_order_k": 1,
    }
    document.update(overrides)
    return load_input_document(
        document, load_capabilities(config_path("capabilities.toml"))
    )


def test_round_trip_delay_uses_each_room_length_and_input_sound_speed() -> None:
    screen = build_reflection_screen(_inputs(), _FREQUENCIES)
    pairs = {row.pair: row for row in screen.pairs}

    assert pairs["x"].distance_m == pytest.approx(5.0)
    assert pairs["y"].distance_m == pytest.approx(7.0)
    assert pairs["z"].distance_m == pytest.approx(9.0)
    assert pairs["x"].round_trip_delay_s == pytest.approx(2.0 * 5.0 / 320.0)
    assert pairs["y"].round_trip_delay_s == pytest.approx(2.0 * 7.0 / 320.0)
    assert pairs["z"].round_trip_delay_s == pytest.approx(2.0 * 9.0 / 320.0)


def test_face_energy_uses_hand_calculated_normal_incidence_and_pair_product() -> None:
    scattering = {wall.wall_name(): 0.2 for wall in Wall.all()}
    inputs = _inputs(scattering_by_wall=scattering)
    row = next(row for row in build_reflection_screen(inputs, _FREQUENCIES).pairs if row.pair == "x")
    rho_c = inputs.density_kg_m3 * inputs.sound_speed_m_s
    face_energy = ((1600.0 - rho_c) / (1600.0 + rho_c)) ** 2 * (1.0 - 0.2)

    assert row.faces == ("x0", "xL")
    assert row.face_retained_energy[0] == pytest.approx(
        tuple(face_energy for _frequency in _FREQUENCIES)
    )
    assert row.face_retained_energy[1] == pytest.approx(
        tuple(face_energy for _frequency in _FREQUENCIES)
    )
    assert row.round_trip_retained_energy == pytest.approx(
        tuple(face_energy * face_energy for _frequency in _FREQUENCIES)
    )


def test_each_face_uses_its_own_impedance() -> None:
    """一對牆的兩面阻抗不同時，各面各用自己的；兩面都拿同一面的會算錯。"""
    impedance = {wall.wall_name(): 1600.0 for wall in Wall.all()}
    impedance.update(x0=800.0, xL=4000.0)
    inputs = _inputs(impedance_pa_s_per_m_by_wall=impedance)
    row = next(row for row in build_reflection_screen(inputs, _FREQUENCIES).pairs if row.pair == "x")
    rho_c = inputs.density_kg_m3 * inputs.sound_speed_m_s
    unscattered = 1.0 - MATERIAL_SCATTERING_DEFAULT_S
    near = ((800.0 - rho_c) / (800.0 + rho_c)) ** 2 * unscattered
    far = ((4000.0 - rho_c) / (4000.0 + rho_c)) ** 2 * unscattered

    assert row.face_retained_energy[0] == pytest.approx(tuple(near for _f in _FREQUENCIES))
    assert row.face_retained_energy[1] == pytest.approx(tuple(far for _f in _FREQUENCIES))
    assert row.round_trip_retained_energy == pytest.approx(
        tuple(near * far for _f in _FREQUENCIES)
    )


def test_missing_scattering_uses_material_default_symbol() -> None:
    row = next(row for row in build_reflection_screen(_inputs(), _FREQUENCIES).pairs if row.pair == "x")
    reflection_energy = ((1600.0 - 400.0) / (1600.0 + 400.0)) ** 2
    expected = reflection_energy * (1.0 - MATERIAL_SCATTERING_DEFAULT_S)

    assert row.face_retained_energy[0] == pytest.approx(
        tuple(expected for _frequency in _FREQUENCIES)
    )


def test_missing_one_wall_scattering_uses_material_default_symbol() -> None:
    inputs = _inputs()
    partial = inputs.model_copy(update={"scattering_by_wall": {"x0": 0.0}})
    row = next(row for row in build_reflection_screen(partial, _FREQUENCIES).pairs if row.pair == "x")
    reflection_energy = ((1600.0 - 400.0) / (1600.0 + 400.0)) ** 2

    assert row.face_retained_energy[0][0] == pytest.approx(reflection_energy)
    assert row.face_retained_energy[1][0] == pytest.approx(
        reflection_energy * (1.0 - MATERIAL_SCATTERING_DEFAULT_S)
    )


def test_each_wall_uses_its_own_scattering_instead_of_room_average() -> None:
    scattering = {
        "x0": 0.0, "xL": 0.5, "y0": 0.2,
        "yL": 0.4, "floor": 0.6, "ceiling": 0.8,
    }
    screen = build_reflection_screen(_inputs(scattering_by_wall=scattering), _FREQUENCIES)
    reflection_energy = ((1600.0 - 400.0) / (1600.0 + 400.0)) ** 2

    for row in screen.pairs:
        first, second = row.faces
        assert row.face_retained_energy[0][0] == pytest.approx(
            reflection_energy * (1.0 - scattering[first])
        )
        assert row.face_retained_energy[1][0] == pytest.approx(
            reflection_energy * (1.0 - scattering[second])
        )
        assert row.round_trip_retained_energy[0] == pytest.approx(
            row.face_retained_energy[0][0] * row.face_retained_energy[1][0]
        )


def test_next_order_earliest_delay_matches_hand_mirrored_second_order() -> None:
    screen = build_reflection_screen(_inputs(), _FREQUENCIES)
    # x0 與 y0 各反射一次：最近鏡像源 (-1,-2,3)，接收點 (3,4,5)。
    expected_distance = math.sqrt((-1.0 - 3.0) ** 2 + (-2.0 - 4.0) ** 2 + (3.0 - 5.0) ** 2)

    assert screen.next_order_earliest_delay_s == pytest.approx(expected_distance / 320.0)
    assert screen.reflection_order_k == 1


def test_next_order_is_none_at_supported_maximum() -> None:
    screen = build_reflection_screen(_inputs(reflection_order_k=8), _FREQUENCIES)

    assert screen.next_order_earliest_delay_s is None


@pytest.mark.parametrize(
    ("coordinate", "changed"),
    (
        ("source_m", Point(1.5, 2.0, 3.0)),
        ("receiver_m", Point(2.5, 4.0, 5.0)),
    ),
)
def test_wall_pairs_ignore_coordinates_but_next_order_changes(
    coordinate: str, changed: Point
) -> None:
    inputs = _inputs()
    moved = inputs.model_copy(update={coordinate: changed})
    original = build_reflection_screen(inputs, _FREQUENCIES)
    shifted = build_reflection_screen(moved, _FREQUENCIES)

    assert shifted.pairs == original.pairs
    assert shifted.next_order_earliest_delay_s != original.next_order_earliest_delay_s


def test_scene_fingerprint_tracks_wall_impedance() -> None:
    inputs = _inputs()
    changed = inputs.model_copy(update={
        "impedance_pa_s_per_m_by_wall": {
            **inputs.impedance_pa_s_per_m_by_wall, "x0": 1800.0
        }
    })
    original = build_reflection_screen(inputs, _FREQUENCIES)
    modified = build_reflection_screen(changed, _FREQUENCIES)

    assert original.scene_fingerprint == scene_fingerprint(inputs)
    assert modified.scene_fingerprint == scene_fingerprint(changed)
    assert modified.scene_fingerprint != original.scene_fingerprint


def test_screen_rejects_mismatched_band_length_nonascending_axis_and_duplicate_pair() -> None:
    screen = build_reflection_screen(_inputs(), _FREQUENCIES)
    document = screen.model_dump()
    document["pairs"][0]["round_trip_retained_energy"] = (0.5,)
    with pytest.raises(ValidationError, match="round_trip_retained_energy"):
        ReflectionScreen.model_validate(document)

    document = screen.model_dump()
    document["frequencies_hz"] = (250.0, 125.0)
    with pytest.raises(ValidationError, match="frequencies_hz"):
        ReflectionScreen.model_validate(document)

    document = screen.model_dump()
    document["pairs"] = (document["pairs"][0], document["pairs"][0], document["pairs"][2])
    with pytest.raises(ValidationError, match="pairs"):
        ReflectionScreen.model_validate(document)


def test_screen_rejects_nonfinite_values_and_extra_fields() -> None:
    screen = build_reflection_screen(_inputs(), _FREQUENCIES)
    document = screen.model_dump()
    document["pairs"][0]["face_retained_energy"] = ((float("nan"), 0.5), (0.5, 0.5))
    with pytest.raises(ValidationError, match="face_retained_energy"):
        ReflectionScreen.model_validate(document)

    document = screen.model_dump()
    document["unknown"] = "extra"
    with pytest.raises(ValidationError, match="unknown"):
        ReflectionScreen.model_validate(document)

    document = screen.model_dump()
    document["pairs"][0]["unknown"] = "extra"
    with pytest.raises(ValidationError, match="unknown"):
        ReflectionScreen.model_validate(document)
