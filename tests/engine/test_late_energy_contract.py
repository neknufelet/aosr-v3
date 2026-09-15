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
from aosr.materials import catalog_absorption
from aosr.physics import late_energy
from blueprint import reference_art_check as art_reference


_ROOT = Path(__file__).resolve().parents[2]
_CASES = ("flat", "varied", "lowabs")


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
    report = late_energy.judge_late_energy(result, expected)
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


def test_contract_constant_matches_reference_check() -> None:
    """產品裁判的界線必須等於錨在決策紙、由治理籃守住的獨立常數。"""
    assert late_energy.LATE_ENERGY_CONTRACT_REL == art_reference.ART_CONTRACT_REL


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


def test_scaled_solver_energy_is_visible_in_legacy_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """放大 solver 的修正後能量仍在第二類逐帶比較顯示，不會被漏量。"""
    path = _answer_path("flat")
    inputs = late_energy.load_late_energy_inputs(path)
    expected = tuple(
        _real(band.get("late_rev_E"), "late_rev_E")
        for band in _expected_bands(path)
    )
    genuine = late_energy.solve_late_energy(inputs)

    def scaled_solver(case: late_energy.LateEnergyInputs) -> late_energy.LateEnergyResult:
        assert case is inputs
        factor = 1.0 + 2.0 * late_energy.LATE_ENERGY_CONTRACT_REL
        bands = tuple(
            replace(band, late_reverberant_energy=band.late_reverberant_energy * factor)
            for band in genuine.bands
        )
        return late_energy.LateEnergyResult(bands=bands)

    monkeypatch.setattr(late_energy, "solve_late_energy", scaled_solver)
    report = late_energy.solve_late_energy_contract(inputs, expected)

    assert not report.within_contract
    assert report.max_contract_fraction > 1.0


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
