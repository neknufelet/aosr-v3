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
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER, image_source_paths


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


def test_screen_keeps_the_input_source_and_receiver_as_given() -> None:
    inputs = _inputs()
    screen = build_reflection_screen(inputs, _FREQUENCIES)

    assert screen.source_m == inputs.source_m
    assert screen.receiver_m == inputs.receiver_m


def test_next_order_is_computed_just_below_the_supported_maximum() -> None:
    """K＝上限減一時第 K+1 階還算得出來；上限判斷差一就會變成 None。"""
    screen = build_reflection_screen(
        _inputs(reflection_order_k=SUPPORTED_MAX_ORDER - 1), _FREQUENCIES
    )

    assert screen.next_order_earliest_delay_s is not None
    assert screen.next_order_earliest_delay_s > 0.0


def test_corner_grazing_second_order_counts_by_mirror_identity() -> None:
    """鏡像 (−1,−1) 到接收點的直線剛好擦過 x0 與 y0 的交線（票 #305：一個反彈點碰兩面牆算兩階）。

    手算：對 x0、y0 各鏡射一次得 (−1,−1,1.5)，到 (2,2,1.5) 的距離 3√2；其他二階鏡像都更遠
    （地板＋天花板 √38、x0＋xL √50）。按反彈點個數數階數的話，這一條會被當成一階而漏掉。
    """
    inputs = _inputs(
        room_m={"Lx": 4.0, "Ly": 4.0, "Lz": 3.0},
        source_m={"x": 1.0, "y": 1.0, "z": 1.5},
        receiver_m={"x": 2.0, "y": 2.0, "z": 1.5},
    )
    screen = build_reflection_screen(inputs, _FREQUENCIES)

    assert screen.next_order_earliest_delay_s == pytest.approx(math.hypot(3.0, 3.0) / 320.0)


@pytest.mark.parametrize("order_k", [1, 2, 3])
@pytest.mark.parametrize(
    "room",
    [{"Lx": 5.0, "Ly": 7.0, "Lz": 9.0}, {"Lx": 3.5, "Ly": 20.0, "Lz": 20.0}],
)
def test_next_order_earliest_is_earliest_of_every_order_not_computed(
    order_k: int, room: dict[str, float]
) -> None:
    """評估器把它當「K 階以內沒算到的最早那一條」：更高階不准比第 K+1 階更早到。"""
    inputs = _inputs(room_m=room, reflection_order_k=order_k)
    screen = build_reflection_screen(inputs, _FREQUENCIES)
    deeper = image_source_paths(
        inputs.room_m, inputs.source_m, inputs.receiver_m, inputs.sound_speed_m_s,
        max_order=order_k + 3, materials=None,
    )

    assert screen.next_order_earliest_delay_s == pytest.approx(
        min(path.delay_s for path in deeper if path.order > order_k)
    )


@pytest.mark.parametrize("frequencies", [(125.0, 125.0), (-125.0, 250.0), (-125.0,)])
def test_screen_rejects_repeated_or_nonpositive_frequencies(
    frequencies: tuple[float, ...]
) -> None:
    """只有一個頻點時沒有相鄰的一對可比，負頻率要靠第一格那一道擋。"""
    screen = build_reflection_screen(_inputs(), tuple(abs(f) + index for index, f in enumerate(frequencies)))
    document = screen.model_dump()
    document["frequencies_hz"] = frequencies
    with pytest.raises(ValidationError, match="frequencies_hz"):
        ReflectionScreen.model_validate(document)


def test_screen_rejects_negative_next_order_delay() -> None:
    document = build_reflection_screen(_inputs(), _FREQUENCIES).model_dump()
    document["next_order_earliest_delay_s"] = -0.001
    with pytest.raises(ValidationError, match="next_order_earliest_delay_s"):
        ReflectionScreen.model_validate(document)


@pytest.mark.parametrize("faces", [("xL", "x0"), ("x0", "window")])
def test_screen_rejects_wall_pair_faces_out_of_order_or_unknown(faces: tuple[str, str]) -> None:
    document = build_reflection_screen(_inputs(), _FREQUENCIES).model_dump()
    document["pairs"][0]["faces"] = faces
    with pytest.raises(ValidationError, match="faces"):
        ReflectionScreen.model_validate(document)


def test_screen_rejects_a_fourth_pair_and_a_nonfinite_coordinate() -> None:
    screen = build_reflection_screen(_inputs(), _FREQUENCIES)
    document = screen.model_dump()
    document["pairs"] = (*document["pairs"], document["pairs"][2])
    with pytest.raises(ValidationError, match="pairs"):
        ReflectionScreen.model_validate(document)

    with pytest.raises(ValidationError, match="座標"):
        ReflectionScreen.model_validate(
            {**screen.model_dump(), "receiver_m": Point(float("nan"), 4.0, 5.0)}
        )


def _earliest_image_distance(
    room: dict[str, float], source: tuple[float, float, float],
    receiver: tuple[float, float, float], order: int,
) -> float:
    """另一條路：鞋盒鏡像公式（Allen–Berkley），不經過 room_paths。

    每一軸的鏡像座標是 ``(1−2p)·s + 2nL``，那一軸撞牆次數是 ``|2n − p|``；三軸加起來等於
    ``order`` 的鏡像裡取離接收點最近的那一個。
    """
    lengths = (room["Lx"], room["Ly"], room["Lz"])
    axis_images: list[list[tuple[int, float]]] = []
    for length, s, r in zip(lengths, source, receiver):
        axis_images.append([
            (abs(2 * n - p), ((1 - 2 * p) * s + 2 * n * length) - r)
            for n in range(-order - 1, order + 2)
            for p in (0, 1)
            if abs(2 * n - p) <= order
        ])
    return min(
        math.sqrt(dx * dx + dy * dy + dz * dz)
        for ox, dx in axis_images[0]
        for oy, dy in axis_images[1]
        for oz, dz in axis_images[2]
        if ox + oy + oz == order
    )


@pytest.mark.parametrize("order_k", [1, 2, 3])
@pytest.mark.parametrize(
    ("room", "source", "receiver"),
    [
        ({"Lx": 5.0, "Ly": 7.0, "Lz": 9.0}, (1.0, 2.0, 3.0), (3.0, 4.0, 5.0)),
        ({"Lx": 4.5, "Ly": 3.5, "Lz": 2.6}, (1.0, 2.2, 1.2), (3.2, 1.9, 1.2)),
        ({"Lx": 4.0, "Ly": 4.0, "Lz": 3.0}, (1.0, 1.0, 1.5), (2.0, 2.0, 1.5)),
    ],
)
def test_next_order_earliest_matches_the_image_formula_for_k_one_to_three(
    order_k: int,
    room: dict[str, float],
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
) -> None:
    """K＝1、2、3 的下一階最早到達，對另一條路算的鏡像距離（第三間房的二階鏡像剛好擦過牆邊交線）。"""
    inputs = _inputs(
        room_m=room,
        source_m=dict(zip("xyz", source)),
        receiver_m=dict(zip("xyz", receiver)),
        reflection_order_k=order_k,
    )
    screen = build_reflection_screen(inputs, _FREQUENCIES)

    assert screen.next_order_earliest_delay_s == pytest.approx(
        _earliest_image_distance(room, source, receiver, order_k + 1) / 320.0
    )


def test_screen_delays_follow_a_legitimate_sound_speed_change() -> None:
    """同一份場景合法換聲速（報表與篩查同一份輸入）：來回延遲與下一階到達都照距離÷聲速跟著變。"""
    slow = build_reflection_screen(_inputs(sound_speed_m_s=320.0), _FREQUENCIES)
    fast = build_reflection_screen(_inputs(sound_speed_m_s=343.0), _FREQUENCIES)

    for slow_row, fast_row in zip(slow.pairs, fast.pairs, strict=True):
        assert fast_row.round_trip_delay_s == pytest.approx(slow_row.round_trip_delay_s * 320.0 / 343.0)
    assert slow.next_order_earliest_delay_s is not None
    assert fast.next_order_earliest_delay_s == pytest.approx(
        slow.next_order_earliest_delay_s * 320.0 / 343.0
    )
