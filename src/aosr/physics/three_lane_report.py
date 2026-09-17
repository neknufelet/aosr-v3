"""有限元素、幾何與晚期衰減的結構化物理量報表。

本模組只接既有純計算入口：阻抗先轉 Paris 無規入射吸音率，交給 crossover 算交接；
有限元素使用 300 Hz 正式網格；幾何逐點使用完整細軸，頻帶的鏡像法部分另用
0.5 Hz 密軸。頻帶的直達、反射、干涉與 s 欄是密軸平均；晚期、T20、T30
與權重欄是細軸平均，其中權重只供閱讀。幾何能量由密軸早期平均加細軸
晚期平均；幾何貢獻則在兩軸逐點乘各自權重後才平均，所以權重欄乘幾何
能量欄不等於幾何貢獻欄，總和要用 FEM 與幾何兩個貢獻欄驗算。
它不讀檔、不印字，也不提供命令列入口。

幾何能量含干涉項（票 #302），不是上一代定義。幾何路與晚期混響按反射階數分工
（票 #337）：報表四欄的反射與干涉已經含散射留存、晚期那一欄是晚期混響交給幾何路
的那一份，四欄相加等於幾何能量；定義與算式住 :mod:`aosr.physics.geometric_lane`。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aosr.config.art_lane import (
    ART_NEUMANN_K_MAX,
    ART_N_PER_WALL_DEFAULT,
    ART_WLS_T20_LO_DB,
    ART_WLS_T30_LO_DB,
)
from aosr.config.capabilities import Capability
from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
from aosr.config.frequency_axis import (
    FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
    FEM_LANE_FREQUENCIES_HZ,
    GEOMETRIC_BAND_FREQUENCIES_HZ,
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
)
from aosr.config.three_lane_crossover import (
    CROSSOVER_LOWER_FLOOR_HZ,
    SCHROEDER_T60_BANDS_HZ,
)
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.physics.crossover import (
    CrossoverWeights,
    crossover_weights,
    eyring_t60_by_band,
    schroeder_frequency_hz,
)
from aosr.physics.fem_helmholtz import solve_fem_helmholtz
from aosr.physics.geometric_lane import (
    GeometricEarlyResult,
    GeometricLaneResult,
    average_geometric_lane_to_bands_with_dense_early,
    solve_geometric_early_lane,
    solve_geometric_lane,
)
from aosr.physics.late_decay import (
    DecayRangeError,
    LateDecayBand,
    LateDecayResult,
    solve_late_decay,
    solve_late_decay_t20,
)
from aosr.physics.late_energy import LateEnergyInputs


@dataclass(frozen=True)
class ThreeLanePoint:
    """一個有完整接合支撐的細軸頻點。

    ``reflected_energy``、``interference_energy`` 已經含散射留存，``late_energy`` 是
    晚期混響交給幾何路的那一份（不是晚期混響總量）；四欄相加等於 ``geometric_energy``。
    """

    frequency_hz: float
    fem_energy: float | None
    direct_energy: float
    reflected_energy: float
    interference_energy: float
    late_energy: float
    scattering: float
    geometric_energy: float
    w_fem: float
    w_geo: float
    total_energy: float


@dataclass(frozen=True)
class ThreeLaneBandReport:
    """一個八度帶的線性能量、平均權重與晚期衰減時間。

    ``fem_energy`` 只平均帶內實際有有限元素值的點；沒有就為 ``None``，
    ``fem_point_count`` 明列這個子集的點數。``direct_energy``、
    ``reflected_energy``、``interference_energy`` 與 ``scattering`` 是帶內密頻率
    點平均；``late_energy``、``t20_s``、``t30_s``、``w_fem`` 與 ``w_geo`` 是帶內
    細軸點平均，兩個權重欄只供閱讀。四欄能量都已經含散射留存（逐階分工），
    ``late_energy`` 是晚期混響交給幾何路的那一份，不是晚期混響總量。

    ``geometric_energy`` 是密頻率點的早期能量平均，加上細軸點的晚期那一欄
    平均。``geometric_contribution`` 在密頻率早期與細軸晚期各自逐點乘
    ``w_geo`` 後才平均。因此 ``w_geo * geometric_energy`` 不等於
    ``geometric_contribution``；``total_energy`` 要用 ``fem_contribution`` 與
    ``geometric_contribution`` 兩欄相加驗算。
    """

    center_frequency_hz: float
    fem_energy: float | None
    fem_point_count: int
    direct_energy: float
    reflected_energy: float
    interference_energy: float
    late_energy: float
    scattering: float
    geometric_energy: float
    fem_contribution: float
    geometric_contribution: float
    total_energy: float
    w_fem: float
    w_geo: float
    f_s_hz: float
    capped_by_upper_limit: bool
    t20_s: float | None
    t20_unavailable_reason: str | None
    t30_s: float | None
    t30_unavailable_reason: str | None


@dataclass(frozen=True)
class _BandDecayUnavailable:
    """報表帶內只有衰減範圍不足可以轉成欄位狀態。"""

    t20_reason: str | None = None
    t30_reason: str | None = None


@dataclass(frozen=True)
class _ReportLateDecay:
    """報表專用的可用細軸結果與逐帶不可算原因。"""

    result: LateDecayResult
    unavailable_by_center_hz: dict[float, _BandDecayUnavailable]


@dataclass(frozen=True)
class _BandSelection:
    """一個報表帶在細軸、密軸、FEM 與衰減結果中的成員。"""

    geometric_indices: tuple[int, ...]
    dense_indices: tuple[int, ...]
    report_points: tuple[ThreeLanePoint, ...]
    fem_values: tuple[float, ...]
    decay_points: tuple[LateDecayBand, ...]


@dataclass(frozen=True)
class ReportCapability:
    """這份報表落在能力表哪一條組合，以及那一條本人（或沒查表）。

    報表帶著自己的驗證範圍走：讀報表的人不必另外翻能力表，就知道這條路
    今天是 validated 還是 experimental、憑什麼這麼說，以及那一條宣告的頻率
    範圍與輸出欄——只印狀態的話，validated 會看起來蓋到整條軸與所有欄位。

    ``record`` 只收表上那一條 :class:`~aosr.config.capabilities.Capability`
    本人或 ``None``（``None``＝呼叫端沒給能力表、這一跑沒有查證）。不再逐格收
    ``status``／``evidence``／``frequency_hz``／``outputs``：那條路讓「validated
    卻沒有 evidence」與「範圍沒給、輸出欄給了卻被丟掉」都造得出來；收成一格
    之後，規則只剩載入器一處，上面兩條在建構那一刻就 :class:`~pydantic.ValidationError`。
    """

    entry: str
    room: str
    materials: str
    record: Capability | None


@dataclass(frozen=True)
class ThreeLaneReport:
    """一個房間、源與收點的三路頻率域物理量報表。

    ``reflection_order_k`` 是這一跑幾何路與晚期混響的交接階數（決策紙要求報表把當次用
    的 K 印出來）；它就是 ``geometric_lane.reflection_order_k`` 那一格。
    """

    capability: ReportCapability
    f_s_hz: float
    reflection_order_k: int
    crossover_lower_hz: float
    crossover_upper_hz: float
    capped_by_upper_limit: bool
    eyring_t60_by_band_s: dict[float, float]
    fem_frequencies_hz: tuple[float, ...]
    fem_energy: tuple[float, ...]
    full_axis_weights: CrossoverWeights
    geometric_lane: GeometricLaneResult
    late_decay: LateDecayResult
    points: tuple[ThreeLanePoint, ...]
    bands: tuple[ThreeLaneBandReport, ...]
    late_decay_frequency_policy: str


def _wall_impedances(
    impedance_by_wall: Mapping[Wall, object],
) -> dict[Wall, float]:
    """只收 FEM 目前支援的六面頻率無關正有限實數阻抗。"""
    if set(impedance_by_wall) != set(Wall.all()):
        raise ValueError("impedance_by_wall 必須恰好包含 Wall.all() 的六面牆")
    result: dict[Wall, float] = {}
    for wall in Wall.all():
        value = impedance_by_wall[wall]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{wall.wall_name()} 的阻抗必須是頻率無關的正有限實數")
        impedance = float(value)
        if not math.isfinite(impedance) or impedance <= 0.0:
            raise ValueError(f"{wall.wall_name()} 的阻抗必須是頻率無關的正有限實數")
        result[wall] = impedance
    return result


def _scattering_by_name(
    scattering_by_wall: Mapping[Wall, float] | None,
) -> dict[str, float] | None:
    if scattering_by_wall is None:
        return None
    unknown = set(scattering_by_wall) - set(Wall.all())
    if unknown:
        raise ValueError("scattering_by_wall 只能使用 Wall.all() 的牆面")
    result: dict[str, float] = {}
    for wall, value in scattering_by_wall.items():
        scattering = float(value)
        if not math.isfinite(scattering) or not 0.0 <= scattering <= 1.0:
            raise ValueError(f"{wall.wall_name()} 的散射係數必須有限且落在 [0,1]")
        result[wall.wall_name()] = scattering
    return result


def _random_absorption_by_wall(
    wall_impedances: Mapping[Wall, float],
    rho_c_pa_s_per_m: float,
) -> dict[Wall, dict[float, float]]:
    if not math.isfinite(rho_c_pa_s_per_m) or rho_c_pa_s_per_m <= 0.0:
        raise ValueError("rho_c_pa_s_per_m 必須是有限正數")
    result: dict[Wall, dict[float, float]] = {}
    for wall in Wall.all():
        impedance = wall_impedances[wall]
        absorption = complex_random_incidence_absorption(
            complex(impedance / rho_c_pa_s_per_m)
        )
        result[wall] = {
            frequency_hz: absorption for frequency_hz in SCHROEDER_T60_BANDS_HZ
        }
    return result


def _named_impedance_rows(
    wall_impedances: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
) -> dict[str, tuple[complex, ...]]:
    return {
        wall.wall_name(): tuple(
            complex(wall_impedances[wall]) for _frequency in frequencies_hz
        )
        for wall in Wall.all()
    }


def stitch_energy_points(
    *,
    frequencies_hz: tuple[float, ...],
    fem_energy_by_frequency: Mapping[float, float],
    geometric_lane: GeometricLaneResult,
    geometric_indices: tuple[int, ...],
    weights: CrossoverWeights,
) -> tuple[ThreeLanePoint, ...]:
    """逐點接合能量；正 FEM 權重沒有 FEM 值時直接報錯。"""
    size = len(frequencies_hz)
    if not (
        len(geometric_indices) == size
        and len(weights.w_fem) == size
        and len(weights.w_geo) == size
    ):
        raise ValueError("接合頻率、幾何索引與權重長度不同")
    points = []
    for index, frequency in enumerate(frequencies_hz):
        geometric_index = geometric_indices[index]
        if geometric_lane.frequencies_hz[geometric_index] != frequency:
            raise ValueError("接合頻率與幾何細軸沒有對齊")
        w_fem = weights.w_fem[index]
        fem_energy = fem_energy_by_frequency.get(frequency)
        if w_fem > 0.0 and fem_energy is None:
            raise ValueError(f"{frequency:g} Hz 的 w_fem > 0 卻沒有有限元素能量")
        fem_value = None if fem_energy is None else float(fem_energy)
        geometric_value = geometric_lane.geometric_energy[geometric_index]
        fem_contribution = 0.0 if fem_value is None else w_fem * fem_value
        total = fem_contribution + weights.w_geo[index] * geometric_value
        points.append(
            ThreeLanePoint(
                frequency_hz=frequency,
                fem_energy=fem_value,
                direct_energy=geometric_lane.direct_energy[geometric_index],
                reflected_energy=geometric_lane.reflected_energy[geometric_index],
                interference_energy=(
                    geometric_lane.interference_energy[geometric_index]
                ),
                late_energy=geometric_lane.late_energy[geometric_index],
                scattering=geometric_lane.scattering[geometric_index],
                geometric_energy=geometric_value,
                w_fem=w_fem,
                w_geo=weights.w_geo[index],
                total_energy=total,
            )
        )
    return tuple(points)


def _mean(values: Sequence[float]) -> float:
    """平均接合總量、FEM 子集或權重；幾何分項用幾何路既有聚合入口。"""
    if not values:
        raise ValueError("頻帶內沒有可平均的頻點")
    return sum(values) / len(values)


def _band_contributions(
    points: tuple[ThreeLanePoint, ...],
    *,
    dense_early: GeometricEarlyResult,
    dense_indices: tuple[int, ...],
    dense_weights: CrossoverWeights,
) -> tuple[float, float]:
    """逐點乘權重後平均：FEM 與晚期用細軸，幾何早期用密軸。

    早期三欄與晚期那一欄都已經含散射留存（逐階分工，見
    :mod:`aosr.physics.geometric_lane`），這裡不再乘任何一次 ``1−s`` 或 ``s``。
    """
    fem = _mean(
        tuple(
            0.0
            if point.fem_energy is None
            else point.w_fem * point.fem_energy
            for point in points
        )
    )
    dense_geometric = _mean(
        tuple(
            dense_weights.w_geo[index]
            * (
                dense_early.direct_energy[index]
                + dense_early.reflected_energy[index]
                + dense_early.interference_energy[index]
            )
            for index in dense_indices
        )
    )
    fine_late = _mean(
        tuple(point.w_geo * point.late_energy for point in points)
    )
    return fem, dense_geometric + fine_late


def _band_decay_values(
    *,
    center_hz: float,
    points: tuple[LateDecayBand, ...],
    unavailable: _BandDecayUnavailable,
) -> tuple[float | None, float | None]:
    """把可用細軸擬合平均成頻帶值；只有已記原因者可為 ``None``。"""
    if not points and unavailable.t20_reason is None:
        raise ValueError(f"{center_hz:g} Hz 頻帶內沒有晚期衰減細軸點")
    t30_values = tuple(point.t30_s for point in points if point.t30_s is not None)
    if unavailable.t30_reason is None and len(t30_values) != len(points):
        raise ValueError(f"{center_hz:g} Hz 頻帶內有晚期衰減沒有 T30")
    t20_s = (
        None
        if unavailable.t20_reason is not None
        else _mean(tuple(point.t20_s for point in points))
    )
    t30_s = (
        None if unavailable.t30_reason is not None else _mean(t30_values)
    )
    return t20_s, t30_s


def _select_band(
    *,
    center_hz: float,
    report_points: tuple[ThreeLanePoint, ...],
    fem_energy_by_frequency: Mapping[float, float],
    geometric_lane: GeometricLaneResult,
    dense_early_lane: GeometricEarlyResult,
    late_decay: LateDecayResult,
) -> _BandSelection:
    root_two = math.sqrt(2.0)
    lower, upper = center_hz / root_two, center_hz * root_two
    return _BandSelection(
        geometric_indices=tuple(
            index
            for index, frequency in enumerate(geometric_lane.frequencies_hz)
            if lower <= frequency < upper
        ),
        dense_indices=tuple(
            index
            for index, frequency in enumerate(dense_early_lane.frequencies_hz)
            if lower <= frequency < upper
        ),
        report_points=tuple(
            point for point in report_points if lower <= point.frequency_hz < upper
        ),
        fem_values=tuple(
            energy
            for frequency, energy in fem_energy_by_frequency.items()
            if lower <= frequency < upper
        ),
        decay_points=tuple(
            decay
            for decay in late_decay.bands
            if lower <= decay.frequency_hz < upper
        ),
    )


def _selected_weight_means(
    weights: CrossoverWeights,
    indices: tuple[int, ...],
) -> tuple[float, float]:
    return (
        _mean(tuple(weights.w_fem[index] for index in indices)),
        _mean(tuple(weights.w_geo[index] for index in indices)),
    )


def _band_reports(
    *,
    report_points: tuple[ThreeLanePoint, ...],
    fem_energy_by_frequency: Mapping[float, float],
    geometric_lane: GeometricLaneResult,
    dense_early_lane: GeometricEarlyResult,
    full_axis_weights: CrossoverWeights,
    dense_axis_weights: CrossoverWeights,
    late_decay: LateDecayResult,
    decay_unavailable_by_center_hz: Mapping[float, _BandDecayUnavailable],
    f_s_hz: float,
) -> tuple[ThreeLaneBandReport, ...]:
    """依欄位契約分別從密軸與細軸組成六個頻帶報表。

    密軸供直達、反射、干涉、s 與加權後早期貢獻；細軸供晚期、T20/T30、
    閱讀用權重與加權後晚期貢獻。幾何能量與貢獻都跨兩組取樣，不能拿頻帶
    權重平均乘頻帶幾何能量平均代替；總和只由兩個貢獻欄相加。
    """
    reports = []
    geometric_bands = average_geometric_lane_to_bands_with_dense_early(
        geometric_lane,
        dense_early_lane,
    )
    for band_index, center in enumerate(GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ):
        selected = _select_band(
            center_hz=center,
            report_points=report_points,
            fem_energy_by_frequency=fem_energy_by_frequency,
            geometric_lane=geometric_lane,
            dense_early_lane=dense_early_lane,
            late_decay=late_decay,
        )
        unavailable = decay_unavailable_by_center_hz.get(
            center, _BandDecayUnavailable()
        )
        fem_contribution, geometric_contribution = _band_contributions(
            selected.report_points,
            dense_early=dense_early_lane,
            dense_indices=selected.dense_indices,
            dense_weights=dense_axis_weights,
        )
        t20_s, t30_s = _band_decay_values(
            center_hz=center,
            points=selected.decay_points,
            unavailable=unavailable,
        )
        w_fem, w_geo = _selected_weight_means(full_axis_weights, selected.geometric_indices)
        reports.append(
            ThreeLaneBandReport(
                center_frequency_hz=center,
                fem_energy=(
                    _mean(selected.fem_values) if selected.fem_values else None
                ),
                fem_point_count=len(selected.fem_values),
                direct_energy=geometric_bands.direct_energy[band_index],
                reflected_energy=geometric_bands.reflected_energy[band_index],
                interference_energy=geometric_bands.interference_energy[band_index],
                late_energy=geometric_bands.late_energy[band_index],
                scattering=geometric_bands.scattering[band_index],
                geometric_energy=geometric_bands.geometric_energy[band_index],
                fem_contribution=fem_contribution,
                geometric_contribution=geometric_contribution,
                total_energy=fem_contribution + geometric_contribution,
                w_fem=w_fem,
                w_geo=w_geo,
                f_s_hz=f_s_hz,
                capped_by_upper_limit=full_axis_weights.capped_by_upper_limit,
                t20_s=t20_s,
                t20_unavailable_reason=unavailable.t20_reason,
                t30_s=t30_s,
                t30_unavailable_reason=unavailable.t30_reason,
            )
        )
    return tuple(reports)


def _solve_fem_energy(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> tuple[float, ...]:
    mesh = generate_shoebox_mesh(
        room,
        max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
        sound_speed_m_s=sound_speed_m_s,
        random_seed=FEM_MESH_RANDOM_SEED,
    )
    pressure = solve_fem_helmholtz(
        mesh,
        wall_impedances=wall_impedances,
        source=source,
        receiver=receiver,
        frequencies_hz=frequencies_hz,
        density_kg_m3=density_kg_m3,
        sound_speed_m_s=sound_speed_m_s,
    )
    return tuple(float(abs(value) ** 2) for value in pressure)


def _solve_geometric_report_lane(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    scattering_by_wall: Mapping[Wall, float] | None,
    rho_c_pa_s_per_m: float,
    sound_speed_m_s: float,
) -> GeometricLaneResult:
    named_impedances = {
        wall.wall_name(): complex(value) for wall, value in wall_impedances.items()
    }
    return solve_geometric_lane(
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=sound_speed_m_s,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        frequencies_hz=GEOMETRIC_LANE_FREQUENCIES_HZ,
        impedance_by_wall=named_impedances,
        scattering_by_wall=_scattering_by_name(scattering_by_wall),
    )


def _solve_dense_geometric_report_lane(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    scattering_by_wall: Mapping[Wall, float] | None,
    rho_c_pa_s_per_m: float,
    sound_speed_m_s: float,
) -> GeometricEarlyResult:
    named_impedances = {
        wall.wall_name(): complex(value) for wall, value in wall_impedances.items()
    }
    return solve_geometric_early_lane(
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=sound_speed_m_s,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        frequencies_hz=GEOMETRIC_BAND_FREQUENCIES_HZ,
        impedance_by_wall=named_impedances,
        scattering_by_wall=_scattering_by_name(scattering_by_wall),
    )


def _solve_and_stitch_report_fem(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    density_kg_m3: float,
    sound_speed_m_s: float,
    geometric: GeometricLaneResult,
    full_weights: CrossoverWeights,
) -> tuple[tuple[float, ...], dict[float, float], tuple[ThreeLanePoint, ...]]:
    fem_energy = _solve_fem_energy(
        room=room,
        source=source,
        receiver=receiver,
        wall_impedances=wall_impedances,
        frequencies_hz=FEM_LANE_FREQUENCIES_HZ,
        density_kg_m3=density_kg_m3,
        sound_speed_m_s=sound_speed_m_s,
    )
    if len(fem_energy) != len(FEM_LANE_FREQUENCIES_HZ):
        raise ValueError("有限元素能量數量必須等於正式有限元素頻率軸")
    fem_by_frequency = dict(
        zip(FEM_LANE_FREQUENCIES_HZ, fem_energy, strict=True)
    )
    points = stitch_energy_points(
        frequencies_hz=GEOMETRIC_LANE_FREQUENCIES_HZ,
        fem_energy_by_frequency=fem_by_frequency,
        geometric_lane=geometric,
        geometric_indices=tuple(range(len(GEOMETRIC_LANE_FREQUENCIES_HZ))),
        weights=full_weights,
    )
    return fem_energy, fem_by_frequency, points


def _solve_report_late_decay(
    *,
    room: Room,
    wall_impedances: Mapping[Wall, float],
    rho_c_pa_s_per_m: float,
    sound_speed_m_s: float,
) -> _ReportLateDecay:
    """逐報表帶求解，只捕捉 ``DecayRangeError`` 轉成不可算欄位。"""
    root_two = math.sqrt(2.0)
    solved_bands: list[LateDecayBand] = []
    unavailable_by_center_hz: dict[float, _BandDecayUnavailable] = {}
    for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ:
        frequencies_hz = tuple(
            frequency
            for frequency in GEOMETRIC_LANE_FREQUENCIES_HZ
            if center / root_two <= frequency < center * root_two
        )
        inputs = LateEnergyInputs(
            room=room,
            rho_c_pa_s_per_m=rho_c_pa_s_per_m,
            frequencies_hz=frequencies_hz,
            impedance_by_wall=_named_impedance_rows(
                wall_impedances, frequencies_hz
            ),
            n_per_wall=ART_N_PER_WALL_DEFAULT,
            domain_alpha_bar_max=math.inf,
        )
        try:
            result = solve_late_decay(inputs, sound_speed_m_s=sound_speed_m_s)
        except DecayRangeError as exc:
            if exc.lower_db == ART_WLS_T20_LO_DB:
                unavailable_by_center_hz[center] = _BandDecayUnavailable(
                    t20_reason=str(exc),
                    t30_reason=exc.reason_for(ART_WLS_T30_LO_DB),
                )
                continue
            if exc.lower_db != ART_WLS_T30_LO_DB:
                raise
            try:
                result = solve_late_decay_t20(
                    inputs, sound_speed_m_s=sound_speed_m_s
                )
            except DecayRangeError as t20_exc:
                if t20_exc.lower_db != ART_WLS_T20_LO_DB:
                    raise
                unavailable_by_center_hz[center] = _BandDecayUnavailable(
                    t20_reason=str(t20_exc),
                    t30_reason=exc.reason_for(ART_WLS_T30_LO_DB),
                )
                continue
            unavailable_by_center_hz[center] = _BandDecayUnavailable(
                t30_reason=str(exc)
            )
        solved_bands.extend(result.bands)
    return _ReportLateDecay(
        result=LateDecayResult(
            orders_used=ART_NEUMANN_K_MAX,
            bands=tuple(solved_bands),
        ),
        unavailable_by_center_hz=unavailable_by_center_hz,
    )


def _report_result(
    *,
    capability: ReportCapability,
    f_s_hz: float,
    t60: dict[float, float],
    fem_frequencies: tuple[float, ...],
    fem_energy: tuple[float, ...],
    full_weights: CrossoverWeights,
    geometric: GeometricLaneResult,
    dense_early: GeometricEarlyResult,
    dense_weights: CrossoverWeights,
    late_decay: LateDecayResult,
    decay_unavailable_by_center_hz: Mapping[float, _BandDecayUnavailable],
    points: tuple[ThreeLanePoint, ...],
    fem_by_frequency: Mapping[float, float],
) -> ThreeLaneReport:
    lower_hz = (
        FEM_GEOMETRIC_CROSSOVER_CAP_HZ
        if full_weights.capped_by_upper_limit
        else max(f_s_hz, CROSSOVER_LOWER_FLOOR_HZ)
    )
    bands = _band_reports(
        report_points=points,
        fem_energy_by_frequency=fem_by_frequency,
        geometric_lane=geometric,
        dense_early_lane=dense_early,
        full_axis_weights=full_weights,
        dense_axis_weights=dense_weights,
        late_decay=late_decay,
        decay_unavailable_by_center_hz=decay_unavailable_by_center_hz,
        f_s_hz=f_s_hz,
    )
    return ThreeLaneReport(
        capability=capability,
        f_s_hz=f_s_hz,
        reflection_order_k=geometric.reflection_order_k,
        crossover_lower_hz=lower_hz,
        crossover_upper_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
        capped_by_upper_limit=full_weights.capped_by_upper_limit,
        eyring_t60_by_band_s=t60,
        fem_frequencies_hz=fem_frequencies,
        fem_energy=fem_energy,
        full_axis_weights=full_weights,
        geometric_lane=geometric,
        late_decay=late_decay,
        points=points,
        bands=bands,
        late_decay_frequency_policy=(
            "T20/T30 在每個頻帶的每個細軸頻點計算，再取帶內算術平均。"
        ),
    )


def _unchecked_capability() -> ReportCapability:
    """沒有能力表可查時的回報：record 留 None，不假裝 validated、也不自創第四個狀態。"""
    return ReportCapability(
        entry="three_lane_report",
        room="shoebox",
        materials="real_frequency_independent_impedance",
        record=None,
    )


def _solve_both_geometric_report_lanes(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    scattering_by_wall: Mapping[Wall, float] | None,
    rho_c_pa_s_per_m: float,
    sound_speed_m_s: float,
) -> tuple[GeometricLaneResult, GeometricEarlyResult]:
    """細軸幾何路與 0.5 Hz 密軸早期路各解一次，回傳兩者。"""
    geometric = _solve_geometric_report_lane(
        room=room,
        source=source,
        receiver=receiver,
        wall_impedances=wall_impedances,
        scattering_by_wall=scattering_by_wall,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        sound_speed_m_s=sound_speed_m_s,
    )
    dense_early = _solve_dense_geometric_report_lane(
        room=room,
        source=source,
        receiver=receiver,
        wall_impedances=wall_impedances,
        scattering_by_wall=scattering_by_wall,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        sound_speed_m_s=sound_speed_m_s,
    )
    return geometric, dense_early


def solve_three_lane_report(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    sound_speed_m_s: float,
    density_kg_m3: float,
    impedance_by_wall: Mapping[Wall, object],
    scattering_by_wall: Mapping[Wall, float] | None = None,
    capability: ReportCapability | None = None,
) -> ThreeLaneReport:
    """計算一個接收點的三路細軸結果與六個八度帶報表。

    ``capability`` 由命令列層從能力表查好傳進來；不給時代表呼叫端直接把這一支
    當純計算入口用（例如考卷），回報一筆沒有查證的紀錄——``record`` 是 ``None``，
    不假裝它 validated，也不自創第四個狀態。
    """
    wall_impedances = _wall_impedances(impedance_by_wall)
    if capability is None:
        capability = _unchecked_capability()
    rho_c_pa_s_per_m = density_kg_m3 * sound_speed_m_s
    t60 = eyring_t60_by_band(
        room,
        _random_absorption_by_wall(wall_impedances, rho_c_pa_s_per_m),
    )
    f_s_hz = schroeder_frequency_hz(room, t60)
    full_weights = crossover_weights(GEOMETRIC_LANE_FREQUENCIES_HZ, f_s_hz)
    dense_weights = crossover_weights(GEOMETRIC_BAND_FREQUENCIES_HZ, f_s_hz)
    geometric, dense_early = _solve_both_geometric_report_lanes(
        room=room,
        source=source,
        receiver=receiver,
        wall_impedances=wall_impedances,
        scattering_by_wall=scattering_by_wall,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        sound_speed_m_s=sound_speed_m_s,
    )
    fem_energy, fem_by_frequency, points = _solve_and_stitch_report_fem(
        room=room,
        source=source,
        receiver=receiver,
        wall_impedances=wall_impedances,
        density_kg_m3=density_kg_m3,
        sound_speed_m_s=sound_speed_m_s,
        geometric=geometric,
        full_weights=full_weights,
    )
    report_decay = _solve_report_late_decay(
        room=room,
        wall_impedances=wall_impedances,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        sound_speed_m_s=sound_speed_m_s,
    )
    return _report_result(
        capability=capability,
        f_s_hz=f_s_hz,
        t60=t60,
        fem_frequencies=FEM_LANE_FREQUENCIES_HZ,
        fem_energy=fem_energy,
        full_weights=full_weights,
        geometric=geometric,
        dense_early=dense_early,
        dense_weights=dense_weights,
        late_decay=report_decay.result,
        decay_unavailable_by_center_hz=report_decay.unavailable_by_center_hz,
        points=points,
        fem_by_frequency=fem_by_frequency,
    )
