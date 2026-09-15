"""晚期衰減 T20 的雙精度相容契約考卷。

答案 JSON 由本考卷自行剖析，只把建好的物理輸入交給產品碼，不走產品答案載入器。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from aosr.config import art_lane
from aosr.geometry.shoebox import Room, Wall
from aosr.physics import late_decay, late_energy
from tests.engine._precision_contracts import contract_value


_ROOT = Path(__file__).resolve().parents[2]
_CASES = ("flat", "varied", "lowabs")
_TOLERANCE_REL = contract_value("late_decay_t20_vs_legacy")


@dataclass(frozen=True)
class ContractRun:
    """一組獨立讀入的題目、v3 結果、上一代答案與正式裁判。"""

    case_name: str
    inputs: late_energy.LateEnergyInputs
    sound_speed_m_s: float
    result: late_decay.LateDecayResult
    expected_t20_s: tuple[float, ...]
    report: late_decay.LateDecayContractReport


def _path(case: str) -> Path:
    return _ROOT / "blueprint" / f"reference_art_decay_{case}.json"


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{where} 不是一層表")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError(f"{where} 不是一串值")
    return value


def _real(value: object, where: str) -> float:
    cell = _mapping(value, where)
    hexadecimal = cell.get("hex")
    if not isinstance(hexadecimal, str):
        raise AssertionError(f"{where}.hex 不是字串")
    result = float.fromhex(hexadecimal)
    if not math.isfinite(result):
        raise AssertionError(f"{where} 不是有限數")
    return result


def _integer(value: object, where: str) -> int:
    cell = _mapping(value, where)
    hexadecimal = cell.get("hex")
    if not isinstance(hexadecimal, str):
        raise AssertionError(f"{where}.hex 不是字串")
    return int(hexadecimal, 16)


def _impedance_row(value: object, where: str) -> tuple[complex, ...]:
    result = []
    for index, raw in enumerate(_sequence(value, where)):
        cell = _mapping(raw, f"{where}[]")
        real = _real(cell.get("real"), f"{where}[{index}].real")
        imag = _real(cell.get("imag"), f"{where}[{index}].imag")
        result.append(complex(real, imag))
    return tuple(result)


def _independent_case(
    case: str,
) -> tuple[late_energy.LateEnergyInputs, float, tuple[float, ...]]:
    with _path(case).open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    root = _mapping(loaded, case)
    parameters = _mapping(root.get("parameters"), "parameters")
    room = _mapping(parameters.get("room_m"), "room_m")
    material = _mapping(parameters.get("material"), "material")
    by_wall = _mapping(material.get("impedance_by_wall"), "impedance_by_wall")
    frequencies = tuple(
        _real(cell, "frequencies_hz[]")
        for cell in _sequence(parameters.get("frequencies_hz"), "frequencies_hz")
    )
    impedances = {
        wall: _impedance_row(by_wall.get(wall), f"impedance_by_wall.{wall}")
        for wall in Wall.wall_names()
    }
    inputs = late_energy.LateEnergyInputs(
        room=Room(
            Lx=_real(room.get("Lx"), "room.Lx"),
            Ly=_real(room.get("Ly"), "room.Ly"),
            Lz=_real(room.get("Lz"), "room.Lz"),
        ),
        rho_c_pa_s_per_m=_real(parameters.get("rho_c_pa_s_per_m"), "rho_c"),
        frequencies_hz=frequencies,
        impedance_by_wall=impedances,
        n_per_wall=_integer(parameters.get("n_per_wall"), "n_per_wall"),
        domain_alpha_bar_max=_real(
            parameters.get("ART_RADIOSITY_ALPHA_BAR_MAX"), "domain limit"
        ),
    )
    expected = tuple(
        _real(_mapping(row, "bands[]").get("t20_s"), "t20_s")
        for row in _sequence(root.get("bands"), "bands")
    )
    sound_speed = _real(parameters.get("sound_speed_m_s"), "sound_speed_m_s")
    assert len(expected) == len(frequencies)
    return inputs, sound_speed, expected


@pytest.fixture(scope="module", params=_CASES)
def contract_run(request: pytest.FixtureRequest) -> ContractRun:
    case_name = str(request.param)
    inputs, sound_speed, expected = _independent_case(case_name)
    result = late_decay.solve_late_decay_t20(inputs, sound_speed_m_s=sound_speed)
    report = late_decay.judge_late_decay_t20(result, expected, _TOLERANCE_REL)
    return ContractRun(case_name, inputs, sound_speed, result, expected, report)


def test_each_reference_band_meets_t20_contract(contract_run: ContractRun) -> None:
    """抓三組材料任一頻帶相對差超過決策紙的 2^-20。"""
    report = contract_run.report
    rows = "; ".join(
        f"{point.frequency_hz:g}Hz={point.relative_difference:.9g}"
        for point in report.points
    )
    message = (
        f"{contract_run.case_name} 最大相對差 {report.max_relative_difference:.9g}，"
        f"用掉 {report.max_contract_fraction * 100.0:.6f}%：{rows}"
    )
    assert report.within_contract, message
    assert all(point.within_contract for point in report.points), message


def test_decay_result_exposes_the_fitted_physics(contract_run: ContractRun) -> None:
    """抓階數、碰撞頻率、擬合斜率、權重或禁止備援的語意接錯。"""
    result = contract_run.result
    assert result.orders_used == art_lane.ART_NEUMANN_K_MAX
    assert len(result.bands) == len(contract_run.expected_t20_s)
    for band in result.bands:
        assert band.collision_frequency_hz > 0.0
        assert band.soft_weight_sum > late_decay.ART_WLS_MIN_WEIGHT
        assert band.slope_db_per_s < 0.0
        assert band.t20_s == -60.0 / band.slope_db_per_s
        assert band.fell_back_to_perron is False
        assert band.perron_t60_s > 0.0


def test_mutant_beyond_tolerance_is_red(contract_run: ContractRun) -> None:
    """真值在界線內推一點判綠、界線外推一點由同一裁判判紅。"""
    inside = late_decay.LateDecayResult(
        orders_used=contract_run.result.orders_used,
        bands=tuple(
            replace(band, t20_s=value * (1.0 + _TOLERANCE_REL / 2.0))
            for band, value in zip(
                contract_run.result.bands,
                contract_run.expected_t20_s,
                strict=True,
            )
        ),
    )
    outside = late_decay.LateDecayResult(
        orders_used=contract_run.result.orders_used,
        bands=tuple(
            replace(band, t20_s=value * (1.0 + 2.0 * _TOLERANCE_REL))
            for band, value in zip(
                contract_run.result.bands,
                contract_run.expected_t20_s,
                strict=True,
            )
        ),
    )

    assert late_decay.judge_late_decay_t20(
        inside, contract_run.expected_t20_s, _TOLERANCE_REL
    ).within_contract
    assert not late_decay.judge_late_decay_t20(
        outside, contract_run.expected_t20_s, _TOLERANCE_REL
    ).within_contract


def test_fully_absorbing_room_raises_instead_of_falling_back() -> None:
    """控制組：全吸音造成零斜率時必須報錯，不准回 Perron 或地板值。"""
    inputs, sound_speed, _expected = _independent_case("flat")
    matched = {
        wall: tuple(complex(inputs.rho_c_pa_s_per_m) for _frequency in inputs.frequencies_hz)
        for wall in Wall.wall_names()
    }
    absorbing = replace(inputs, impedance_by_wall=matched)
    with pytest.raises(ValueError, match="擬合無效"):
        late_decay.solve_late_decay_t20(absorbing, sound_speed_m_s=sound_speed)
