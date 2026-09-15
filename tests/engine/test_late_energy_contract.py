"""晚期混響能量的第二類相容紀錄考卷。"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from aosr.geometry.shoebox import Wall
from aosr.materials import catalog_absorption
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
    """用材料層已獨立驗過的 Paris 入口核對晚期接線。"""
    return catalog_absorption.complex_random_incidence_absorption(impedance / rho_c)


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


def test_exact_late_energy_records_each_legacy_band_difference(
    contract_run: ContractRun,
) -> None:
    """第二類相容紀錄仍須逐帶量出有限差距，但不拿舊界線擋合併。"""
    report = contract_run.report
    assert tuple(point.frequency_hz for point in report.points) == tuple(
        band.frequency_hz for band in contract_run.result.bands
    )
    assert all(
        math.isfinite(point.actual_energy)
        and math.isfinite(point.expected_energy)
        and math.isfinite(point.relative_difference)
        for point in report.points
    )


def test_result_keeps_physical_properties_and_reference_metadata(
    contract_run: ContractRun,
) -> None:
    """抓負能量、非有限值、無規入射吸收率或上一代適用域判斷接錯。

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
    剛好是 ``1∓δ``。這一條已降為第二類相容紀錄（照量、照留、不擋合併），所以
    這支考卷驗的是「紀錄的判決欄誠實」：超界那一組要記成不通過、回傳的是一份
    讀得到的紀錄而不是例外，而且那個不通過只是紀錄、不是讓整支考卷炸掉。
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

    inside_report = late_energy.judge_late_energy(inside, expected, _TOLERANCE_REL)
    outside_report = late_energy.judge_late_energy(outside, expected, _TOLERANCE_REL)

    # 判決欄：界線內全過、界線外那幾格各自被記成不通過。
    assert inside_report.within_contract is True
    assert all(point.within_contract for point in inside_report.points)
    assert outside_report.within_contract is False
    assert all(not point.within_contract for point in outside_report.points)
    # 紀錄本身不炸：兩組都拿回逐帶點、每點一份判決，不是例外。
    assert len(inside_report.points) == len(expected)
    assert len(outside_report.points) == len(expected)
    assert outside_report.max_contract_fraction > 1.0
    assert _TOLERANCE_REL == outside_report.tolerance_rel


def test_zero_transfer_has_no_reflected_energy() -> None:
    """R=0 時 reflected-only 的 raw 能量恰為零，不把均勻源項算進輸出。"""
    base = late_energy.load_late_energy_inputs(_answer_path("flat"))
    patches = late_energy._patch_geometry(base.room, base.n_per_wall)
    transfer = np.zeros(
        (patches.areas.size, patches.areas.size, len(base.frequencies_hz)),
        dtype=np.float64,
    )
    result = late_energy._exact_raw_energy(transfer, patches.areas)
    assert np.array_equal(result, np.zeros_like(result))
