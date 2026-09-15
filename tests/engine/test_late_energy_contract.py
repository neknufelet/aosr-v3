"""晚期混響能量的雙精度精確解契約考卷。"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from aosr.geometry.shoebox import Wall
from aosr.physics import late_energy
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value


_ROOT = Path(__file__).resolve().parents[2]
_CASES = ("flat", "varied", "lowabs")
_TOLERANCE_REL = contract_value("late_energy_vs_legacy")
_INSIDE = (1.0 - MUTANT_MARGIN) * _TOLERANCE_REL
_OUTSIDE = (1.0 + MUTANT_MARGIN) * _TOLERANCE_REL


@dataclass(frozen=True)
class ContractRun:
    """一組正式輸入、精確解、上一代答案、裁判與牆鐘時間。"""

    case_name: str
    inputs: late_energy.LateEnergyInputs
    result: late_energy.LateEnergyResult
    expected_bands: tuple[dict[str, object], ...]
    report: late_energy.LateEnergyContractReport
    elapsed_seconds: float


def _answer_path(case_name: str) -> Path:
    return _ROOT / "blueprint" / f"reference_art_{case_name}.json"


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
    return float.fromhex(hexadecimal)


def _expected_bands(path: Path) -> tuple[dict[str, object], ...]:
    with path.open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    answer = _mapping(loaded, "答案檔")
    return tuple(_mapping(item, "bands[]") for item in _sequence(answer.get("bands"), "bands"))


def _float64_absorption(impedance: complex, rho_c: float) -> float:
    """用 v3 指定的雙精度法向入射定義獨立算吸收率。"""
    z = np.complex128(impedance)
    medium = np.float64(rho_c)
    reflection = (z - medium) / (z + medium)
    return float(np.float64(1.0) - np.abs(reflection) ** np.float64(2.0))


@pytest.mark.parametrize("case_name", _CASES)
def test_load_legacy_late_energies_returns_finite_positive_bands(
    case_name: str,
) -> None:
    """抓答案載入器漏頻帶，或放行非有限、非正的上一代能量。"""
    path = _answer_path(case_name)
    expected = tuple(
        _real(band.get("late_rev_E"), "late_rev_E")
        for band in _expected_bands(path)
    )
    energies = late_energy.load_legacy_late_energies(path)

    assert energies == expected
    assert all(math.isfinite(energy) and energy > 0.0 for energy in energies)


def test_load_legacy_late_energies_rejects_non_sequence_bands(tmp_path: Path) -> None:
    """抓 ``bands`` 不是 JSON 陣列時被誤當成可用答案。"""
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"bands": {}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="bands 不是一串值"):
        late_energy.load_legacy_late_energies(malformed)


@pytest.fixture(scope="module", params=_CASES)
def contract_run(request: pytest.FixtureRequest) -> ContractRun:
    """每組材料只跑一次 216×216 的六頻帶直接解。"""
    case_name = str(request.param)
    path = _answer_path(case_name)
    inputs = late_energy.load_late_energy_inputs(path)
    expected_bands = _expected_bands(path)
    expected = tuple(
        _real(band.get("late_rev_E"), "late_rev_E")
        for band in expected_bands
    )
    started = time.perf_counter()
    result = late_energy.solve_late_energy(inputs)
    report = late_energy.judge_late_energy(result, expected, _TOLERANCE_REL)
    elapsed = time.perf_counter() - started
    return ContractRun(case_name, inputs, result, expected_bands, report, elapsed)


def test_exact_late_energy_meets_each_band_contract(contract_run: ContractRun) -> None:
    """任一材料任一頻帶超過正式相對界線都必須判紅。"""
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


def test_result_keeps_physical_properties_and_reference_metadata(
    contract_run: ContractRun,
) -> None:
    """抓負能量、非有限值、法向吸收率算法或上一代適用域判斷接錯。

    不比答案吸收率：上一代單精度逐運算累積誤差差幾格屬預期，且治理籃考卷已核對它。
    """
    result_by_frequency = {band.frequency_hz: band for band in contract_run.result.bands}
    for frequency_index, expected in enumerate(contract_run.expected_bands):
        frequency = _real(expected.get("frequency_hz"), "frequency_hz")
        actual = result_by_frequency[frequency]
        assert math.isfinite(actual.raw_reverberant_energy)
        assert actual.raw_reverberant_energy >= 0.0
        assert all(
            actual.alpha_by_wall[wall]
            == _float64_absorption(
                contract_run.inputs.impedance_by_wall[wall][frequency_index],
                contract_run.inputs.rho_c_pa_s_per_m,
            )
            for wall in Wall.wall_names()
        )
        assert actual.in_domain is expected.get("in_domain")


def test_in_domain_includes_the_exact_alpha_bar_boundary() -> None:
    """抓 ``<=`` 被反寫成 ``<``、``>=`` 或 ``>`` 的上一代適用域邊界錯誤。"""
    inputs = late_energy.load_late_energy_inputs(_answer_path("flat"))
    alpha_bar = late_energy.solve_late_energy(inputs).bands[0].alpha_bar

    at_boundary = late_energy.solve_late_energy(
        replace(inputs, domain_alpha_bar_max=alpha_bar)
    )
    below_boundary = late_energy.solve_late_energy(
        replace(inputs, domain_alpha_bar_max=math.nextafter(alpha_bar, 0.0))
    )

    assert at_boundary.bands[0].in_domain is True
    assert below_boundary.bands[0].in_domain is False


def test_mutant_beyond_tolerance_is_red() -> None:
    """真值在界線內推 δ 判綠、界線外推 δ 判紅（兩側各 δ）。

    產品判 ``|R3−R2| ≤ T·R2``，所以答案真值放 ``R2·(1+(1∓δ)·T)`` 就讓「差÷界線」
    剛好是 ``1∓δ``。
    """
    path = _answer_path("flat")
    inputs = late_energy.load_late_energy_inputs(path)
    expected = tuple(
        _real(band.get("late_rev_E"), "late_rev_E")
        for band in _expected_bands(path)
    )
    genuine = late_energy.solve_late_energy(inputs)
    inside = late_energy.LateEnergyResult(
        bands=tuple(
            replace(band, late_reverberant_energy=value * (1.0 + _INSIDE))
            for band, value in zip(genuine.bands, expected, strict=True)
        )
    )
    outside = late_energy.LateEnergyResult(
        bands=tuple(
            replace(band, late_reverberant_energy=value * (1.0 + _OUTSIDE))
            for band, value in zip(genuine.bands, expected, strict=True)
        )
    )

    assert late_energy.judge_late_energy(inside, expected, _TOLERANCE_REL).within_contract
    assert not late_energy.judge_late_energy(outside, expected, _TOLERANCE_REL).within_contract


def test_fully_absorbing_walls_have_no_reflected_energy() -> None:
    """α=1 使 R=0；本模組回 reflected-only，故均勻直接注入不列入輸出、raw 恰為零。"""
    base = late_energy.load_late_energy_inputs(_answer_path("flat"))
    matched = {
        wall: tuple(complex(base.rho_c_pa_s_per_m) for _frequency in base.frequencies_hz)
        for wall in Wall.wall_names()
    }
    inputs = replace(base, impedance_by_wall=matched)

    result = late_energy.solve_late_energy(inputs)
    guarded_alpha = 1.0 - float(np.finfo(np.float32).eps)
    expected_ratio = guarded_alpha / -math.log1p(-guarded_alpha)

    assert all(band.raw_reverberant_energy == 0.0 for band in result.bands)
    assert all(band.eyring_ratio == pytest.approx(expected_ratio) for band in result.bands)
    assert all(band.late_reverberant_energy == 0.0 for band in result.bands)
