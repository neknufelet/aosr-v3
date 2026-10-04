"""#608 第二項：房間散射在每面牆阻抗與散射都不隨頻率變時只算一個頻帶；以主線舊寫法驗證逐位不變。"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import cast

import pytest

from aosr.geometry.shoebox import Room, Wall
from aosr.physics import geometric_lane as gl


# 主線原函式逐字保留，只改名稱與 _wall_areas 的取用方式；不跟隨新寫法。
def _legacy_room_scattering(
    room: Room,
    rho_c_pa_s_per_m: float,
    impedance_by_wall: dict[str, tuple[complex, ...]],
    scattering_by_wall: dict[str, tuple[float, ...]],
    frequencies_hz: tuple[float, ...],
) -> tuple[float, ...]:
    """依 ``lib.physics.m4_pipeline`` 的反射能量權重合成房間 s。"""
    areas = gl._wall_areas(room)
    combined = []
    for frequency_index, _frequency in enumerate(frequencies_hz):
        scattering_values = tuple(
            scattering_by_wall[wall][frequency_index] for wall in Wall.wall_names()
        )
        reflected_weights = tuple(
            areas[wall]
            * abs(
                (
                    impedance_by_wall[wall][frequency_index]
                    - rho_c_pa_s_per_m
                )
                / (
                    impedance_by_wall[wall][frequency_index]
                    + rho_c_pa_s_per_m
                )
            )
            ** 2
            for wall in Wall.wall_names()
        )
        denominator = sum(reflected_weights)
        if denominator == 0.0:
            combined.append(0.0)
            continue
        if all(value == scattering_values[0] for value in scattering_values):
            combined.append(scattering_values[0])
            continue
        numerator = sum(
            weight * scattering
            for weight, scattering in zip(
                reflected_weights, scattering_values, strict=True
            )
        )
        combined.append(numerator / denominator)
    return tuple(combined)


def _bits(values: tuple[float, ...]) -> tuple[str, ...]:
    return tuple(float.hex(value) for value in values)


def _inputs(rng: random.Random, bands: int, varying_impedance: int, varying_scattering: int,
            ) -> tuple[dict[str, tuple[complex, ...]], dict[str, tuple[float, ...]]]:
    impedance: dict[str, tuple[complex, ...]] = {}
    scattering: dict[str, tuple[float, ...]] = {}
    for index, wall in enumerate(Wall.wall_names()):
        z = tuple(complex(10.0 ** rng.uniform(1.0, 6.0), rng.uniform(-1e4, 1e4)) for _ in range(bands))
        s = tuple(rng.uniform(0.0, 1.0) for _ in range(bands))
        impedance[wall] = z if index < varying_impedance else (z[0],) * bands
        scattering[wall] = s if index < varying_scattering else (s[0],) * bands
    return impedance, scattering


@pytest.mark.parametrize("varying_impedance,varying_scattering", [(0, 0), (1, 0), (0, 1), (6, 6)])
def test_random_inputs_match_legacy_bits(tmp_path: Path, varying_impedance: int, varying_scattering: int) -> None:
    rng = random.Random(6082)
    for _ in range(200):
        room = Room(rng.uniform(2.0, 12.0), rng.uniform(2.0, 9.0), rng.uniform(2.0, 5.0))
        bands = rng.randrange(1, 12)
        impedance, scattering = _inputs(rng, bands, varying_impedance, varying_scattering)
        rho_c = rng.uniform(300.0, 500.0)
        frequencies = tuple(100.0 * (i + 1) for i in range(bands))
        expected = _legacy_room_scattering(room, rho_c, impedance, scattering, frequencies)
        actual = gl._room_scattering(room, rho_c, impedance, scattering, frequencies)
        assert _bits(actual) == _bits(expected)


def _count_bands(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    original = gl._scattering_at

    def counting(index: int, areas: dict[str, float], rho_c: float,
                 impedance: dict[str, tuple[complex, ...]], scattering: dict[str, tuple[float, ...]]) -> float:
        seen.append(index)
        return original(index, areas, rho_c, impedance, scattering)

    monkeypatch.setattr(gl, "_scattering_at", counting)
    return seen


def test_constant_inputs_compute_one_band(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    impedance, scattering = _inputs(random.Random(1), 9, 0, 0)
    room, frequencies = Room(6.0, 4.0, 3.0), tuple(100.0 * (i + 1) for i in range(9))
    expected = _legacy_room_scattering(room, 411.6, impedance, scattering, frequencies)
    seen = _count_bands(monkeypatch)
    actual = gl._room_scattering(room, 411.6, impedance, scattering, frequencies)
    assert _bits(actual) == _bits(expected)
    assert seen == [0]


@pytest.mark.parametrize("case", ["impedance", "scattering", "signed-zero-scattering", "signed-zero-impedance", "nan"])
def test_anything_not_bitwise_constant_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """一面牆隨頻率變、散射或阻抗有正負零混在一起、NaN 不是同一個物件：都照逐頻帶算，數字跟舊寫法逐位相同。"""
    bands = 4
    impedance, scattering = _inputs(random.Random(2), bands, 0, 0)
    wall = Wall.wall_names()[0]
    if case == "impedance":
        impedance[wall] = (complex(500.0, 1.0),) + (complex(600.0, 1.0),) * (bands - 1)
    elif case == "scattering":
        scattering[wall] = (0.2,) + (0.3,) * (bands - 1)
    elif case == "signed-zero-scattering":
        scattering = {name: (0.0, -0.0, 0.0, -0.0) for name in Wall.wall_names()}
    elif case == "signed-zero-impedance":
        impedance[wall] = (complex(700.0, 0.0), complex(700.0, -0.0)) * 2
    else:
        scattering[wall] = tuple(float("nan") for _ in range(bands))
    room, frequencies = Room(6.0, 4.0, 3.0), tuple(100.0 * (i + 1) for i in range(bands))
    expected = _legacy_room_scattering(room, 411.6, impedance, scattering, frequencies)
    seen = _count_bands(monkeypatch)
    actual = gl._room_scattering(room, 411.6, impedance, scattering, frequencies)
    assert _bits(actual) == _bits(expected)
    assert seen == list(range(bands))


def test_zero_denominator_and_empty_axis_match_legacy(tmp_path: Path) -> None:
    room = Room(6.0, 4.0, 3.0)
    rho_c = 411.6
    matched: dict[str, tuple[complex, ...]] = {wall: (complex(rho_c, 0.0),) * 3 for wall in Wall.wall_names()}
    scattering: dict[str, tuple[float, ...]] = {wall: (0.5,) * 3 for wall in Wall.wall_names()}
    frequencies = (100.0, 200.0, 300.0)
    assert _bits(gl._room_scattering(room, rho_c, matched, scattering, frequencies)) == _bits(
        _legacy_room_scattering(room, rho_c, matched, scattering, frequencies))
    empty_impedance: dict[str, tuple[complex, ...]] = {wall: () for wall in Wall.wall_names()}
    empty_scattering: dict[str, tuple[float, ...]] = {wall: () for wall in Wall.wall_names()}
    assert gl._room_scattering(room, rho_c, empty_impedance, empty_scattering, ()) == ()
    assert not any(math.isnan(value) for value in gl._room_scattering(room, rho_c, matched, scattering, frequencies))


@pytest.mark.parametrize("which,row", [
    ("scattering", (0.2, 0.2, 0.9)),
    ("scattering", (0.2, 0.9, 0.2)),
    ("impedance", (complex(500.0, 10.0), complex(500.0, 10.0), complex(3000.0, -200.0))),
    ("impedance", (complex(500.0, 10.0), complex(3000.0, -200.0), complex(500.0, 10.0))),
    ("impedance", (0.0, 0j, 0j)),
], ids=["scattering-first-two", "scattering-first-last", "impedance-first-two", "impedance-first-last", "mixed-types"])
def test_only_fully_identical_rows_take_the_shortcut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str, row: tuple[complex, ...] | tuple[float, ...],
) -> None:
    """頭兩格相同、頭尾相同、型別混用都照逐頻帶算，不准當掉，數字跟舊寫法逐位相同（複查）。"""
    impedance: dict[str, tuple[complex, ...]] = {wall: (complex(500.0, 10.0),) * 3 for wall in Wall.wall_names()}
    scattering: dict[str, tuple[float, ...]] = {wall: (0.3,) * 3 for wall in Wall.wall_names()}
    wall = Wall.wall_names()[0]
    if which == "scattering":
        scattering[wall] = tuple(float(value.real) for value in row)
    else:
        impedance[wall] = cast(tuple[complex, ...], row)  # 原樣保留型別（含混用的 0.0 與 0j）
    room, frequencies = Room(6.0, 4.0, 3.0), (100.0, 200.0, 300.0)
    expected = _legacy_room_scattering(room, 411.6, impedance, scattering, frequencies)
    seen = _count_bands(monkeypatch)
    actual = gl._room_scattering(room, 411.6, impedance, scattering, frequencies)
    assert _bits(actual) == _bits(expected)
    assert seen == [0, 1, 2]


def test_short_rows_still_raise_like_before(tmp_path: Path) -> None:
    """阻抗或散射列比頻率軸短：舊寫法丟索引錯，捷徑不准悄悄排滿（複查）。"""
    impedance: dict[str, tuple[complex, ...]] = {wall: (complex(500.0, 10.0),) for wall in Wall.wall_names()}
    scattering: dict[str, tuple[float, ...]] = {wall: (0.3,) for wall in Wall.wall_names()}
    room, frequencies = Room(6.0, 4.0, 3.0), (100.0, 200.0, 300.0)
    with pytest.raises(IndexError):
        _legacy_room_scattering(room, 411.6, impedance, scattering, frequencies)
    with pytest.raises(IndexError):
        gl._room_scattering(room, 411.6, impedance, scattering, frequencies)
