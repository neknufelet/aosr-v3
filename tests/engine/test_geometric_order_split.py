"""幾何路與晚期混響按反射階數分工的性質考卷（票 #337）。

定義照決策紙 ``docs/decisions/stage-nine-reflection-order-is-a-setting.md`` 第 5 條：
交接階數 K 以內的鏡面留在鏡像法，被散射掉的那一份與 K 階以上全部交給晚期混響。
這一支只驗**性質**，不對任何凍結答案，也不新增門檻數字：需要容差的地方一律拿既有
精度契約登記簿的相對界線去縮放本題的能量量級（``_energy_roundoff_bound``），
跟 ``test_geometric_lane.py`` 同一條路。
"""

from __future__ import annotations

import math

import pytest

from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.amplitude import Materials
from aosr.physics.geometric_lane import GeometricLaneResult, solve_geometric_lane
from aosr.physics.late_energy import (
    LateEnergyInputs,
    solve_late_energy,
    solve_late_energy_by_order,
)
from aosr.physics.room_paths import RoomPath, image_source_paths
from aosr.physics.totals import totals_and_pressure_sums_from_paths
from tests.engine._precision_contracts import contract_value


_ROOM = Room(Lx=6.0, Ly=4.0, Lz=3.0)
_SOURCE = Point(x=1.2, y=1.3, z=1.1)
_RECEIVER = Point(x=4.7, y=2.8, z=1.4)
_SOUND_SPEED_M_S = 343.0
_RHO_C_PA_S_PER_M = 411.6
_FREQUENCIES_HZ = (63.0, 125.0, 500.0, 2000.0)
_TOLERANCE_REL = contract_value("direct_energy_vs_legacy")


def _energy_roundoff_bound(*values: float) -> float:
    """只用既有直達能量契約界線縮放本題各能量量級，不新增門檻數字。"""
    return _TOLERANCE_REL * sum(abs(value) for value in values)


def _walls(impedance_multiple: float) -> dict[str, complex]:
    return {
        wall: complex(impedance_multiple * _RHO_C_PA_S_PER_M, 0.0)
        for wall in Wall.wall_names()
    }


def _wall_rows(
    impedance_multiple: float,
    frequencies_hz: tuple[float, ...],
) -> dict[str, tuple[complex, ...]]:
    return {
        wall: tuple(value for _frequency in frequencies_hz)
        for wall, value in _walls(impedance_multiple).items()
    }


def _paths(
    impedance_multiple: float,
    frequencies_hz: tuple[float, ...],
) -> list[RoomPath]:
    return image_source_paths(
        _ROOM,
        _SOURCE,
        _RECEIVER,
        _SOUND_SPEED_M_S,
        max_order=REFLECTION_ORDER_K,
        materials=Materials(
            rho_c=_RHO_C_PA_S_PER_M,
            frequencies_hz=frequencies_hz,
            walls=_wall_rows(impedance_multiple, frequencies_hz),
        ),
    )


def _late_inputs(
    impedance_multiple: float,
    frequencies_hz: tuple[float, ...],
) -> LateEnergyInputs:
    return LateEnergyInputs(
        room=_ROOM,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=frequencies_hz,
        impedance_by_wall=_wall_rows(impedance_multiple, frequencies_hz),
        n_per_wall=ART_N_PER_WALL_DEFAULT,
        domain_alpha_bar_max=math.inf,
    )


def _lane(impedance_multiple: float, scattering: float) -> GeometricLaneResult:
    return solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=_FREQUENCIES_HZ,
        impedance_by_wall=_walls(impedance_multiple),
        scattering_by_wall={
            wall: scattering for wall in Wall.wall_names()
        },
    )


# ── ① 逐階壓力和 ────────────────────────────────────────────────────────────────
def test_order_sums_are_the_paths_of_that_order() -> None:
    """逐階和若混進別階的路徑、或改掉同階內的相加順序，必須紅。"""
    paths = _paths(4.0, _FREQUENCIES_HZ)
    _totals, sums = totals_and_pressure_sums_from_paths(paths)

    assert len(sums.reflected_pressure_by_order) == REFLECTION_ORDER_K
    for slot, column in enumerate(sums.reflected_pressure_by_order):
        order = slot + 1
        same_order = [path for path in paths if path.order == order]
        assert same_order, order
        for index in range(len(_FREQUENCIES_HZ)):
            expected = sum(
                (path.path_pressure[index] for path in same_order),
                complex(0.0, 0.0),
            )
            assert column[index] == expected


def test_order_sums_add_up_to_the_existing_reflected_sum() -> None:
    """逐階和相加不等於既有的反射同調和時必須紅（只差浮點結合律）。"""
    paths = _paths(4.0, _FREQUENCIES_HZ)
    _totals, sums = totals_and_pressure_sums_from_paths(paths)

    for index, reflected in enumerate(sums.reflected_pressure):
        regrouped = sum(
            (column[index] for column in sums.reflected_pressure_by_order),
            complex(0.0, 0.0),
        )
        assert abs(regrouped - reflected) <= _energy_roundoff_bound(abs(reflected))


def test_direct_path_is_not_counted_as_a_reflection_order() -> None:
    """直達被混進第一階時必須紅：逐階和只收 order ≥ 1 的路徑。"""
    paths = _paths(4.0, _FREQUENCIES_HZ)
    direct = next(path for path in paths if path.order == 0)
    _totals, sums = totals_and_pressure_sums_from_paths(paths)

    for index, first_order in enumerate(sums.reflected_pressure_by_order[0]):
        assert first_order != direct.path_pressure[index]
        assert sums.direct_pressure[index] == direct.path_pressure[index]


# ── ② 晚期混響的逐階展開 ────────────────────────────────────────────────────────
def test_late_orders_and_tail_add_up_to_the_existing_total() -> None:
    """逐階晚期解若換了正規化或 Eyring 比值，總量就對不上既有那一支，必須紅。"""
    inputs = _late_inputs(4.0, _FREQUENCIES_HZ)
    total = solve_late_energy(inputs)
    split = solve_late_energy_by_order(inputs, max_order=REFLECTION_ORDER_K)

    assert split.max_order == REFLECTION_ORDER_K
    for band, expected in zip(split.bands, total.bands, strict=True):
        assert band.frequency_hz == expected.frequency_hz
        # 總量逐位相同：兩支走同一條 raw × Eyring 比值。
        assert band.late_reverberant_energy == expected.late_reverberant_energy
        assert len(band.energy_by_order) == REFLECTION_ORDER_K
        assert all(value > 0.0 for value in band.energy_by_order)
        assert band.tail_energy > 0.0
        assembled = sum(band.energy_by_order) + band.tail_energy
        assert abs(assembled - band.late_reverberant_energy) <= _energy_roundoff_bound(
            band.late_reverberant_energy
        )


def test_late_tail_shrinks_as_the_crossover_order_rises() -> None:
    """尾巴不隨交接階數變小，就表示逐階展開不是同一個級數，必須紅。"""
    inputs = _late_inputs(4.0, _FREQUENCIES_HZ)
    low = solve_late_energy_by_order(inputs, max_order=REFLECTION_ORDER_K)
    high = solve_late_energy_by_order(inputs, max_order=REFLECTION_ORDER_K + 2)

    for low_band, high_band in zip(low.bands, high.bands, strict=True):
        assert high_band.tail_energy < low_band.tail_energy
        assert high_band.tail_energy > 0.0
        # 前面幾階逐位相同：多算兩階不會回頭改前面那幾階。
        assert (
            high_band.energy_by_order[: len(low_band.energy_by_order)]
            == low_band.energy_by_order
        )


def test_late_order_expansion_rejects_a_crossover_below_one() -> None:
    """交接階數小於 1 時不准靜靜回一份空展開。"""
    with pytest.raises(ValueError, match="max_order"):
        solve_late_energy_by_order(_late_inputs(4.0, _FREQUENCIES_HZ), max_order=0)


# ── ③ 兩端極限 ─────────────────────────────────────────────────────────────────
def test_zero_scattering_is_mirror_orders_plus_only_the_late_tail() -> None:
    """s=0 若不是「K 階以內鏡面（含干涉）＋晚期混響 K 階以上的尾巴」，必須紅。

    兩邊的參考都是這一支自己算的：鏡面那一項用路徑逐階複數和，尾巴用既有晚期總量
    減掉逐階展開的前 K 階。
    """
    paths = _paths(4.0, _FREQUENCIES_HZ)
    _totals, sums = totals_and_pressure_sums_from_paths(paths)
    inputs = _late_inputs(4.0, _FREQUENCIES_HZ)
    total = solve_late_energy(inputs)
    split = solve_late_energy_by_order(inputs, max_order=REFLECTION_ORDER_K)

    actual = _lane(4.0, 0.0)

    for index in range(len(_FREQUENCIES_HZ)):
        coherent = sums.direct_pressure[index] + sum(
            (column[index] for column in sums.reflected_pressure_by_order),
            complex(0.0, 0.0),
        )
        tail = total.bands[index].late_reverberant_energy - sum(
            split.bands[index].energy_by_order
        )
        expected = abs(coherent) ** 2 + tail
        assert abs(actual.geometric_energy[index] - expected) <= (
            _energy_roundoff_bound(abs(coherent) ** 2, tail)
        )
        assert actual.late_energy[index] > 0.0


def test_full_scattering_hands_the_whole_reflected_part_to_late_energy() -> None:
    """s=1 若還留著鏡面反射或干涉、或晚期那一欄不是晚期混響總量，必須紅。"""
    paths = _paths(4.0, _FREQUENCIES_HZ)
    _totals, sums = totals_and_pressure_sums_from_paths(paths)
    total = solve_late_energy(_late_inputs(4.0, _FREQUENCIES_HZ))

    actual = _lane(4.0, 1.0)

    for index, band in enumerate(total.bands):
        direct = abs(sums.direct_pressure[index]) ** 2
        assert actual.reflected_energy[index] == 0.0
        assert actual.interference_energy[index] == 0.0
        assert abs(
            actual.late_energy[index] - band.late_reverberant_energy
        ) <= _energy_roundoff_bound(band.late_reverberant_energy)
        assert abs(actual.geometric_energy[index] - direct - band.late_reverberant_energy) <= (
            _energy_roundoff_bound(direct, band.late_reverberant_energy)
        )


# ── ④ 報表四欄與非負 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("impedance_multiple", "scattering"),
    (
        (1.5, 0.0),
        (4.0, 0.1),
        (10.0, 0.3),
        (400.0, 0.5),
        (400.0, 1.0),
    ),
    ids=("absorptive-s0", "flat-s01", "lowabs-s03", "hard-s05", "hard-s1"),
)
def test_four_columns_add_up_and_stay_nonnegative(
    impedance_multiple: float,
    scattering: float,
) -> None:
    """四欄相加不等於幾何能量、或任一頻點的總能量變負時必須紅。

    第一項是模平方、其餘兩項非負，所以總能量恆不為負——這一題把材料從強吸音掃到近硬牆、
    散射從 0 掃到 1，兩端都在裡面。
    """
    actual = _lane(impedance_multiple, scattering)

    for index in range(len(_FREQUENCIES_HZ)):
        assert actual.geometric_energy[index] == (
            actual.direct_energy[index]
            + actual.reflected_energy[index]
            + actual.interference_energy[index]
            + actual.late_energy[index]
        )
        assert actual.geometric_energy[index] >= 0.0
        assert actual.late_energy[index] >= 0.0
        assert actual.reflected_energy[index] >= 0.0
        assert actual.direct_energy[index] > 0.0
    assert actual.reflection_order_k == REFLECTION_ORDER_K


@pytest.mark.parametrize("max_order", (40, 200), ids=("k40", "k200"))
def test_deep_order_expansion_never_returns_a_negative_tail(max_order: int) -> None:
    """階數拉深到尾巴只剩捨入量級時，那一格必須是 0 而不是負數。

    ``Σ_{k≥1} A_k`` 收斂到總量，所以 K 一大，「總量減掉前 K 階」就會落進浮點捨入
    裡、算出 ``-0.0`` 這種負值。幾何路拿這一格當晚期欄，負的會讓總能量少算而且不會
    有人叫；#341 把 K 露出成呼叫端可調的設定之後，深階數是真的走得到的路。
    """
    result = solve_late_energy_by_order(
        _late_inputs(4.0, _FREQUENCIES_HZ), max_order=max_order
    )

    for band in result.bands:
        assert band.tail_energy >= 0.0
        assert sum(band.energy_by_order) <= band.late_reverberant_energy + (
            _energy_roundoff_bound(band.late_reverberant_energy)
        )
