"""鞋盒房間的細頻率軸幾何能量路。

鏡像路徑、帶相位的反射壓力與同調總量分別委派給
:mod:`aosr.physics.room_paths`、:mod:`aosr.physics.amplitude` 與
:mod:`aosr.physics.totals`；晚期能量委派給 :mod:`aosr.physics.late_energy`。
本模組只接既有結果，不抄第二份算法。

幾何能量含干涉項（票 #302），不是上一代定義；此項是直達與反射同調和之間的交叉項。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.config.frequency_axis import GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.response import MATERIAL_SCATTERING_DEFAULT_S
from aosr.physics.amplitude import Materials
from aosr.physics.late_energy import LateEnergyInputs, solve_late_energy
from aosr.physics.room_paths import image_source_paths
from aosr.physics.totals import totals_and_pressure_sums_from_paths

WallImpedance = complex | float | Sequence[complex]
WallScattering = float | Sequence[float]


@dataclass(frozen=True)
class GeometricLaneResult:
    """細軸上逐頻的直達、同調反射、干涉與晚期能量。"""

    frequencies_hz: tuple[float, ...]
    direct_energy: tuple[float, ...]
    reflected_energy: tuple[float, ...]
    interference_energy: tuple[float, ...]
    late_energy: tuple[float, ...]
    scattering: tuple[float, ...]
    geometric_energy: tuple[float, ...]


@dataclass(frozen=True)
class GeometricEarlyResult:
    """任意頻率軸上的鏡像法早期能量與房間散射係數。"""

    frequencies_hz: tuple[float, ...]
    direct_energy: tuple[float, ...]
    reflected_energy: tuple[float, ...]
    interference_energy: tuple[float, ...]
    scattering: tuple[float, ...]


@dataclass(frozen=True)
class GeometricBandResult:
    """八度報表帶內依早期密軸、晚期細軸組成的算術平均。"""

    band_centers_hz: tuple[float, ...]
    direct_energy: tuple[float, ...]
    reflected_energy: tuple[float, ...]
    interference_energy: tuple[float, ...]
    late_energy: tuple[float, ...]
    scattering: tuple[float, ...]
    geometric_energy: tuple[float, ...]


def _impedance_rows(
    impedance_by_wall: Mapping[str, WallImpedance],
    frequencies_hz: tuple[float, ...],
) -> dict[str, tuple[complex, ...]]:
    rows: dict[str, tuple[complex, ...]] = {}
    for wall in Wall.wall_names():
        value = impedance_by_wall[wall]
        row = (
            tuple(complex(value) for _frequency in frequencies_hz)
            if isinstance(value, int | float | complex)
            else tuple(complex(item) for item in value)
        )
        if len(row) != len(frequencies_hz):
            if len(row) == len(GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ):
                raise ValueError(
                    "只有頻帶值的材料在細軸上怎麼取值待拍（實測資料要內插）"
                )
            raise ValueError(f"{wall} 的阻抗頻點數與 frequencies_hz 不同")
        rows[wall] = row
    return rows


def _scattering_rows(
    scattering_by_wall: Mapping[str, WallScattering],
    frequencies_hz: tuple[float, ...],
) -> dict[str, tuple[float, ...]]:
    rows: dict[str, tuple[float, ...]] = {}
    for wall in Wall.wall_names():
        value = scattering_by_wall.get(wall, MATERIAL_SCATTERING_DEFAULT_S)
        row = (
            tuple(float(value) for _frequency in frequencies_hz)
            if isinstance(value, int | float)
            else tuple(float(item) for item in value)
        )
        if len(row) != len(frequencies_hz):
            raise ValueError(f"{wall} 的散射頻點數與 frequencies_hz 不同")
        rows[wall] = row
    return rows


def _wall_areas(room: Room) -> dict[str, float]:
    return {
        Wall.FLOOR.wall_name(): room.Lx * room.Ly,
        Wall.CEILING.wall_name(): room.Lx * room.Ly,
        Wall.X0.wall_name(): room.Ly * room.Lz,
        Wall.XL.wall_name(): room.Ly * room.Lz,
        Wall.Y0.wall_name(): room.Lx * room.Lz,
        Wall.YL.wall_name(): room.Lx * room.Lz,
    }


def _room_scattering(
    room: Room,
    rho_c_pa_s_per_m: float,
    impedance_by_wall: dict[str, tuple[complex, ...]],
    scattering_by_wall: dict[str, tuple[float, ...]],
    frequencies_hz: tuple[float, ...],
) -> tuple[float, ...]:
    """依 ``lib.physics.m4_pipeline`` 的反射能量權重合成房間 s。"""
    areas = _wall_areas(room)
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


def _geometric_energy(
    direct: tuple[float, ...],
    reflected: tuple[float, ...],
    interference: tuple[float, ...],
    late: tuple[float, ...],
    scattering: tuple[float, ...],
) -> tuple[float, ...]:
    energies = []
    for direct_value, reflected_value, interference_value, late_value, s_value in zip(
        direct, reflected, interference, late, scattering, strict=True
    ):
        energies.append(
            direct_value
            + (1.0 - s_value) * (reflected_value + interference_value)
            + s_value * late_value
        )
    return tuple(energies)


def _interference_energy(
    direct_pressure: tuple[complex, ...],
    reflected_pressure: tuple[complex, ...],
) -> tuple[float, ...]:
    """逐頻算 ``2*Re(pd*conj(pr))``；不用三個能量相減。"""
    return tuple(
        2.0 * (direct * reflected.conjugate()).real
        for direct, reflected in zip(
            direct_pressure, reflected_pressure, strict=True
        )
    )


def _selected_mean(values: tuple[float, ...], indices: tuple[int, ...]) -> float:
    return sum(values[index] for index in indices) / len(indices)


def average_geometric_lane_to_bands(
    result: GeometricLaneResult,
    *,
    band_centers_hz: tuple[float, ...] = GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
) -> GeometricBandResult:
    """依 ``[fc/√2, fc·√2)`` 算各項細頻點能量的算術平均。"""
    direct = []
    reflected = []
    interference = []
    late = []
    scattering = []
    geometric = []
    root_two = math.sqrt(2.0)
    for center in band_centers_hz:
        lower = center / root_two
        upper = center * root_two
        indices = tuple(
            index
            for index, frequency in enumerate(result.frequencies_hz)
            if lower <= frequency < upper
        )
        if not indices:
            raise ValueError(f"{center} Hz 頻帶內沒有頻點")
        direct.append(_selected_mean(result.direct_energy, indices))
        reflected.append(_selected_mean(result.reflected_energy, indices))
        interference.append(_selected_mean(result.interference_energy, indices))
        late.append(_selected_mean(result.late_energy, indices))
        scattering.append(_selected_mean(result.scattering, indices))
        geometric.append(_selected_mean(result.geometric_energy, indices))
    return GeometricBandResult(
        band_centers_hz=band_centers_hz,
        direct_energy=tuple(direct),
        reflected_energy=tuple(reflected),
        interference_energy=tuple(interference),
        late_energy=tuple(late),
        scattering=tuple(scattering),
        geometric_energy=tuple(geometric),
    )


def average_geometric_lane_to_bands_with_dense_early(
    fine_result: GeometricLaneResult,
    dense_early_result: GeometricEarlyResult,
    *,
    band_centers_hz: tuple[float, ...] = GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
) -> GeometricBandResult:
    """密軸平均鏡像法早期項，細軸平均散射後晚期項。"""
    direct = []
    reflected = []
    interference = []
    late = []
    scattering = []
    geometric = []
    root_two = math.sqrt(2.0)
    for center in band_centers_hz:
        lower = center / root_two
        upper = center * root_two
        dense_indices = tuple(
            index
            for index, frequency in enumerate(dense_early_result.frequencies_hz)
            if lower <= frequency < upper
        )
        fine_indices = tuple(
            index
            for index, frequency in enumerate(fine_result.frequencies_hz)
            if lower <= frequency < upper
        )
        if not dense_indices or not fine_indices:
            raise ValueError(f"{center} Hz 頻帶內沒有頻點")
        direct.append(_selected_mean(dense_early_result.direct_energy, dense_indices))
        reflected.append(
            _selected_mean(dense_early_result.reflected_energy, dense_indices)
        )
        interference.append(
            _selected_mean(dense_early_result.interference_energy, dense_indices)
        )
        late.append(_selected_mean(fine_result.late_energy, fine_indices))
        scattering.append(
            _selected_mean(dense_early_result.scattering, dense_indices)
        )
        dense_early_energy = tuple(
            dense_early_result.direct_energy[index]
            + (1.0 - dense_early_result.scattering[index])
            * (
                dense_early_result.reflected_energy[index]
                + dense_early_result.interference_energy[index]
            )
            for index in dense_indices
        )
        fine_scattered_late = tuple(
            fine_result.scattering[index] * fine_result.late_energy[index]
            for index in fine_indices
        )
        geometric.append(
            _selected_mean(dense_early_energy, tuple(range(len(dense_early_energy))))
            + _selected_mean(
                fine_scattered_late, tuple(range(len(fine_scattered_late)))
            )
        )
    return GeometricBandResult(
        band_centers_hz=band_centers_hz,
        direct_energy=tuple(direct),
        reflected_energy=tuple(reflected),
        interference_energy=tuple(interference),
        late_energy=tuple(late),
        scattering=tuple(scattering),
        geometric_energy=tuple(geometric),
    )


def solve_geometric_early_lane(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    sound_speed_m_s: float,
    rho_c_pa_s_per_m: float,
    frequencies_hz: tuple[float, ...],
    impedance_by_wall: Mapping[str, WallImpedance],
    scattering_by_wall: Mapping[str, WallScattering] | None = None,
) -> GeometricEarlyResult:
    """以既有三階鏡像法算任意頻率軸上的早期幾何項，不求晚期。"""
    impedance_rows = _impedance_rows(impedance_by_wall, frequencies_hz)
    scattering_rows = _scattering_rows(scattering_by_wall or {}, frequencies_hz)
    materials = Materials(
        rho_c=rho_c_pa_s_per_m,
        frequencies_hz=frequencies_hz,
        walls=impedance_rows,
    )
    paths = image_source_paths(
        room,
        source,
        receiver,
        sound_speed_m_s,
        max_order=3,
        materials=materials,
    )
    path_totals, pressure_sums = totals_and_pressure_sums_from_paths(paths)
    return GeometricEarlyResult(
        frequencies_hz=frequencies_hz,
        direct_energy=path_totals.direct_energy,
        reflected_energy=path_totals.reflected_energy,
        interference_energy=_interference_energy(
            pressure_sums.direct_pressure,
            pressure_sums.reflected_pressure,
        ),
        scattering=_room_scattering(
            room,
            rho_c_pa_s_per_m,
            impedance_rows,
            scattering_rows,
            frequencies_hz,
        ),
    )


def solve_geometric_lane(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    sound_speed_m_s: float,
    rho_c_pa_s_per_m: float,
    frequencies_hz: tuple[float, ...],
    impedance_by_wall: Mapping[str, WallImpedance],
    scattering_by_wall: Mapping[str, WallScattering] | None = None,
) -> GeometricLaneResult:
    """以既有三階鏡像法與晚期精確解計算細軸上的各項能量。"""
    impedance_rows = _impedance_rows(impedance_by_wall, frequencies_hz)
    early = solve_geometric_early_lane(
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=sound_speed_m_s,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        frequencies_hz=frequencies_hz,
        impedance_by_wall=impedance_rows,
        scattering_by_wall=scattering_by_wall,
    )
    late_result = solve_late_energy(
        LateEnergyInputs(
            room=room,
            rho_c_pa_s_per_m=rho_c_pa_s_per_m,
            frequencies_hz=frequencies_hz,
            impedance_by_wall=impedance_rows,
            n_per_wall=ART_N_PER_WALL_DEFAULT,
            domain_alpha_bar_max=math.inf,
        )
    )
    late_energy = tuple(
        band.late_reverberant_energy for band in late_result.bands
    )
    return GeometricLaneResult(
        frequencies_hz=frequencies_hz,
        direct_energy=early.direct_energy,
        reflected_energy=early.reflected_energy,
        interference_energy=early.interference_energy,
        late_energy=late_energy,
        scattering=early.scattering,
        geometric_energy=_geometric_energy(
            early.direct_energy,
            early.reflected_energy,
            early.interference_energy,
            late_energy,
            early.scattering,
        ),
    )
