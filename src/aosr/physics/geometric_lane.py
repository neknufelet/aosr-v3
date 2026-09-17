"""鞋盒房間的細頻率軸幾何能量路。

鏡像路徑、帶相位的反射壓力與同調總量分別委派給
:mod:`aosr.physics.room_paths`、:mod:`aosr.physics.amplitude` 與
:mod:`aosr.physics.totals`；晚期能量委派給 :mod:`aosr.physics.late_energy`。
本模組只接既有結果，不抄第二份算法。

幾何能量含干涉項（票 #302），不是上一代定義；此項是直達與反射同調和之間的交叉項。

**幾何路與晚期混響按反射階數分工**（票 #337，決策紙
``docs/decisions/stage-nine-reflection-order-is-a-setting.md`` 第 5 條）：交接階數 K 以內
的鏡面部分留在鏡像法，被散射掉的那一份與 K 階以上全部交給晚期混響。逐頻是

``E_geo = |p_direct + Σ_{k≤K} (1−s)^{k/2}·p_k|² + Σ_{k≤K} [1−(1−s)^k]·A_k
+ (E_late − Σ_{k≤K} A_k)``

``p_k`` 是第 k 階全部反射路徑壓力的複數和（同階內的干涉本來就在裡面），``(1−s)^{k/2}``
是「每次反射壓力幅值乘 √(1−s)」連乘 k 次；``A_k`` 是晚期混響精確解按反射階數展開的
第 k 階分量。報表四欄照這條定義拆：直達 ``|p_direct|²``、反射
``|Σ_{k≤K}(1−s)^{k/2}·p_k|²``、干涉 ``2·Re(p_direct·conj(Σ …))``、晚期
``Σ_{k≤K}[1−(1−s)^k]·A_k + (E_late − Σ_{k≤K}A_k)``；四欄相加等於 ``E_geo``。

取代的是原定義 ``直達 + (1−s)·(反射 + 干涉) + s·晚期``——原式只乘一次 (1−s)、不是逐次
分流，而且 s=0 時 K 階以上鏡面沒有任何一路接手。

**K 是呼叫端可調的設定**（票 #341）：這一層每一支入口都吃一個 ``reflection_order_k``，
預設值是產品設定 ``config.three_lane_crossover.REFLECTION_ORDER_K``，所以沒給的呼叫端跟
先前逐位相同。合法範圍由 ``physics.room_paths`` 那一格
（``SUPPORTED_MIN_ORDER``～``SUPPORTED_MAX_ORDER``）守，這一層不再抄第二份界線；晚期那
一路的逐階展開跟鏡像法走**同一個** K，兩邊不准各拿各的。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.config.frequency_axis import GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.response import MATERIAL_SCATTERING_DEFAULT_S
from aosr.physics.amplitude import Materials
from aosr.physics.late_energy import (
    LateEnergyInputs,
    LateEnergyOrderBand,
    solve_late_energy_by_order,
)
from aosr.physics.room_paths import image_source_paths
from aosr.physics.totals import totals_and_pressure_sums_from_paths

WallImpedance = complex | float | Sequence[complex]
WallScattering = float | Sequence[float]


@dataclass(frozen=True)
class GeometricLaneResult:
    """細軸上逐頻的報表四欄、房間散射係數與幾何能量。

    ``reflected_energy`` 是**逐階縮放後**的鏡面同調和模平方
    ``|Σ_{k≤K}(1−s)^{k/2}·p_k|²``、``interference_energy`` 是它與直達的交叉項，兩欄都
    已經含散射留存；``late_energy`` 是晚期混響交給幾何路的**那一份**
    ``Σ_{k≤K}[1−(1−s)^k]·A_k + (E_late − Σ_{k≤K}A_k)``，不是晚期混響總量 ``E_late``。
    四欄相加等於 ``geometric_energy``。``reflection_order_k`` 是這一跑用的交接階數。
    """

    frequencies_hz: tuple[float, ...]
    direct_energy: tuple[float, ...]
    reflected_energy: tuple[float, ...]
    interference_energy: tuple[float, ...]
    late_energy: tuple[float, ...]
    scattering: tuple[float, ...]
    geometric_energy: tuple[float, ...]
    reflection_order_k: int


@dataclass(frozen=True)
class GeometricEarlyResult:
    """任意頻率軸上的鏡像法早期能量與房間散射係數。

    三欄早期能量與 :class:`GeometricLaneResult` 同名那三欄同一個定義（含散射留存），
    三欄相加等於 ``|p_direct + Σ_{k≤K}(1−s)^{k/2}·p_k|²``。
    """

    frequencies_hz: tuple[float, ...]
    direct_energy: tuple[float, ...]
    reflected_energy: tuple[float, ...]
    interference_energy: tuple[float, ...]
    scattering: tuple[float, ...]
    reflection_order_k: int


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
    late_share: tuple[float, ...],
) -> tuple[float, ...]:
    """報表四欄相加就是 ``E_geo``（決策紙第 5 條最後一行）。"""
    return tuple(
        direct_value + reflected_value + interference_value + late_value
        for direct_value, reflected_value, interference_value, late_value in zip(
            direct, reflected, interference, late_share, strict=True
        )
    )


def _order_scaled_reflected_pressure(
    reflected_pressure_by_order: tuple[tuple[complex, ...], ...],
    scattering: tuple[float, ...],
) -> tuple[complex, ...]:
    """逐頻算 ``Σ_{k=1..K} (1−s)^{k/2}·p_k``。

    能量乘 (1−s)、壓力幅值就乘 √(1−s)，連乘 k 次得 ``(1−s)^{k/2}``；散射掉的那一份不留
    在鏡像法這一路，改由晚期混響接手（``_late_share_energy``）。
    """
    combined = []
    for frequency_index, s_value in enumerate(scattering):
        retained = 1.0 - s_value
        total = 0j
        for order, column in enumerate(reflected_pressure_by_order, start=1):
            total += retained ** (order / 2.0) * column[frequency_index]
        combined.append(total)
    return tuple(combined)


def _late_share_energy(
    late_bands: tuple[LateEnergyOrderBand, ...],
    scattering: tuple[float, ...],
) -> tuple[float, ...]:
    """逐頻算晚期那一欄：``Σ_{k≤K}[1−(1−s)^k]·A_k + (E_late − Σ_{k≤K}A_k)``。

    前一段是 K 階以內被散射掉的那一份，後一段是 K 階以上那一整段尾巴；兩段都非負，
    所以這一欄不會把總能量拉成負的。
    """
    shares = []
    for band, s_value in zip(late_bands, scattering, strict=True):
        retained = 1.0 - s_value
        scattered = 0.0
        for order, energy in enumerate(band.energy_by_order, start=1):
            scattered += (1.0 - retained**order) * energy
        shares.append(scattered + band.tail_energy)
    return tuple(shares)


def _reflected_energy(reflected_pressure: tuple[complex, ...]) -> tuple[float, ...]:
    """反射那一欄：逐階縮放後同調和的模平方。"""
    return tuple(abs(pressure) ** 2 for pressure in reflected_pressure)


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


def _same_order_k(
    fine_result: GeometricLaneResult,
    dense_early_result: GeometricEarlyResult,
    reflection_order_k: int,
) -> None:
    """兩軸與報表宣告的交接階數必須是同一個 K，不是就當場 ``ValueError``。

    K 不同的兩份平均起來會得到一份**誰的 K 都不是**的報表，而報表只印得出一個 K
    ——那種不一致沒有人看得出來，所以在平均之前就擋。
    """
    for name, result_k in (
        ("細軸", fine_result.reflection_order_k),
        ("密軸", dense_early_result.reflection_order_k),
    ):
        if result_k != reflection_order_k:
            raise ValueError(
                f"{name}那一份用的交接階數是 {result_k}，不是這一份報表宣告的 "
                f"{reflection_order_k}：兩軸與報表必須是同一個 K"
            )


def average_geometric_lane_to_bands_with_dense_early(
    fine_result: GeometricLaneResult,
    dense_early_result: GeometricEarlyResult,
    *,
    band_centers_hz: tuple[float, ...] = GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
    reflection_order_k: int = REFLECTION_ORDER_K,
) -> GeometricBandResult:
    """密軸平均鏡像法早期三欄，細軸平均晚期那一欄。

    兩欄早期與晚期都已經含散射留存（逐階分工，見模組說明），所以這裡只是各自取平均
    再相加，不再乘任何一次 ``1−s``。

    ``reflection_order_k`` 是這一份報表宣告用的交接階數，由 :func:`_same_order_k` 比對
    （理由見那一支）。"""
    _same_order_k(fine_result, dense_early_result, reflection_order_k)
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
            + dense_early_result.reflected_energy[index]
            + dense_early_result.interference_energy[index]
            for index in dense_indices
        )
        fine_late_share = tuple(
            fine_result.late_energy[index] for index in fine_indices
        )
        geometric.append(
            _selected_mean(dense_early_energy, tuple(range(len(dense_early_energy))))
            + _selected_mean(fine_late_share, tuple(range(len(fine_late_share))))
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
    reflection_order_k: int = REFLECTION_ORDER_K,
) -> GeometricEarlyResult:
    """以既有鏡像法算任意頻率軸上的早期幾何項（含逐階散射留存），不求晚期。

    鏡像法只列舉到交接階數 ``reflection_order_k``；該階以上不在這一路，由晚期混響接手。
    不給就是產品設定 ``REFLECTION_ORDER_K``；超出合法範圍由 :func:`image_source_paths`
    丟 ``ValueError``（界線住那一支，這裡不抄第二份）。
    """
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
        max_order=reflection_order_k,
        materials=materials,
    )
    path_totals, pressure_sums = totals_and_pressure_sums_from_paths(paths)
    scattering = _room_scattering(
        room,
        rho_c_pa_s_per_m,
        impedance_rows,
        scattering_rows,
        frequencies_hz,
    )
    reflected_pressure = _order_scaled_reflected_pressure(
        pressure_sums.reflected_pressure_by_order,
        scattering,
    )
    return GeometricEarlyResult(
        frequencies_hz=frequencies_hz,
        direct_energy=path_totals.direct_energy,
        reflected_energy=_reflected_energy(reflected_pressure),
        interference_energy=_interference_energy(
            pressure_sums.direct_pressure,
            reflected_pressure,
        ),
        scattering=scattering,
        reflection_order_k=reflection_order_k,
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
    reflection_order_k: int = REFLECTION_ORDER_K,
) -> GeometricLaneResult:
    """以既有鏡像法與晚期精確解計算細軸上的報表四欄與幾何能量。

    晚期那一路按**同一個** ``reflection_order_k`` 展開（:func:`~aosr.physics.late_energy
    .solve_late_energy_by_order`），K 階以內被散射掉的那一份與 K 階以上的尾巴合成
    ``late_energy`` 那一欄；鏡像法那三欄只留 K 階以內沒被散射掉的部分。不給就是產品設定
    ``REFLECTION_ORDER_K``。
    """
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
        reflection_order_k=reflection_order_k,
    )
    late_result = solve_late_energy_by_order(
        LateEnergyInputs(
            room=room,
            rho_c_pa_s_per_m=rho_c_pa_s_per_m,
            frequencies_hz=frequencies_hz,
            impedance_by_wall=impedance_rows,
            n_per_wall=ART_N_PER_WALL_DEFAULT,
            domain_alpha_bar_max=math.inf,
        ),
        max_order=reflection_order_k,
    )
    late_energy = _late_share_energy(late_result.bands, early.scattering)
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
        ),
        reflection_order_k=reflection_order_k,
    )
