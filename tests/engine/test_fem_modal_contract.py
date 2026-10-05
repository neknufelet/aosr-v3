"""V1 指定案例的逐分量契約；變異只動已配對共振，不改數量與種類。"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.physics.fem_modal import FemModalSpectrum
from aosr.physics.fem_modal_contract import ModalReference, ModalContractReport, judge_modal_spectrum
from aosr.physics.modal_convention import ModalKind
from aosr.physics.modal_rectangle_truth import RectangleModalProblem, trace_modes
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine.test_fem_modal import C, RHO, LENGTHS, CAP, _rigid_frequencies, _solve
from tests.engine.test_fem_modal_trapezoid import (
    ROOT, CaseAnswer, ProblemData, problem as problem, run as trapezoid_run,
)


@pytest.fixture(scope="module")
def rectangles() -> dict[str, tuple[FemModalSpectrum, tuple[ModalReference, ...]]]:
    results = {}
    for name, beta in (("rigid", (0.0, 0.0, 0.0)), ("uniform", (0.25, 0.25, 0.25)),
                       ("axes", (0.10, 0.17, 0.23))):
        spectrum = _solve(beta, 7)
        if name == "rigid":
            expected = tuple(ModalReference(complex(2 * math.pi * f), complex(2 * math.pi * f),
                                           ModalKind.STATIC if f == 0 else ModalKind.RESONANCE)
                             for f in _rigid_frequencies(LENGTHS, CAP))
        else:
            truth = trace_modes(RectangleModalProblem(LENGTHS, C, RHO, beta),
                                frequency_max_hz=CAP, seed_frequency_max_hz=CAP + 60)
            assert all(mode.reached_target for mode in truth.modes)
            expected = tuple(ModalReference(
                complex(0 if mode.quantities.kind is not ModalKind.RESONANCE else mode.omega.real,
                        0 if mode.quantities.kind is ModalKind.STATIC else mode.omega.imag),
                mode.omega, mode.quantities.kind)
                for mode in truth.modes if mode.quantities.frequency_hz <= CAP)
            expected = (ModalReference(0j, C * truth.static_wave_number, ModalKind.STATIC), *expected)
        results[name] = spectrum, expected
    return results


def _rectangle_report(spectrum: FemModalSpectrum, expected: tuple[ModalReference, ...]) -> ModalContractReport:
    return judge_modal_spectrum(spectrum, expected,
                               frequency_rel=contract_value("modal_rectangle_frequency"),
                               decay_rel=contract_value("modal_rectangle_decay"))


def _external_references(answer: CaseAnswer) -> tuple[ModalReference, ...]:
    return tuple(ModalReference(complex(*row["omega_rad_s"]), complex(*row["raw_omega_rad_s"]),
                               ModalKind(row["kind"])) for row in answer["solutions"])


def _external_report(spectrum: FemModalSpectrum, expected: tuple[ModalReference, ...]) -> ModalContractReport:
    limit = contract_value("modal_trapezoid_vs_fenics")
    return judge_modal_spectrum(spectrum, expected, frequency_rel=limit, decay_rel=limit)


def _assert_report(report: ModalContractReport) -> None:
    assert report.within_contract, report
    assert not report.structural_errors
    assert report.points and all(point.within_contract for point in report.points)


@pytest.mark.parametrize("case", ["rigid", "uniform", "axes"])
def test_rectangle_all_resonances_meet_component_contracts(
    rectangles: dict[str, tuple[FemModalSpectrum, tuple[ModalReference, ...]]], case: str,
) -> None:
    _assert_report(_rectangle_report(*rectangles[case]))


def test_trapezoid_all_roots_meet_component_contracts(
    trapezoid_run: tuple[FemModalSpectrum, CaseAnswer],
) -> None:
    spectrum, answer = trapezoid_run
    _assert_report(_external_report(spectrum, _external_references(answer)))
    for row in spectrum.solutions:
        if row.kind is ModalKind.STATIC:
            assert row.t60_s is None and row.q is None
        elif row.omega.imag == 0:
            assert row.t60_s == math.inf and row.q == math.inf
        else:
            assert row.t60_s == pytest.approx(3 * math.log(10) / row.omega.imag)
            assert row.q == pytest.approx(row.omega.real / (2 * row.omega.imag))


def _mutant(spectrum: FemModalSpectrum, expected: tuple[ModalReference, ...],
            baseline: ModalContractReport, component: str, limit: float,
            fraction: float) -> FemModalSpectrum:
    point = next(point for point in baseline.points if point.kind is ModalKind.RESONANCE
                 and (component == "real" or point.expected_omega.imag > 0))
    rows = list(spectrum.solutions)
    original = rows[point.actual_index]
    reference = expected[point.expected_index].omega
    value = (reference.real if component == "real" else reference.imag) * (1 + fraction * limit)
    omega = complex(value, original.omega.imag) if component == "real" else complex(original.omega.real, value)
    raw = complex(value, original.raw_omega.imag) if component == "real" else complex(original.raw_omega.real, value)
    rows[point.actual_index] = replace(original, omega=omega, raw_omega=raw)
    return replace(spectrum, solutions=tuple(rows))


def _assert_same_pairs(baseline: ModalContractReport, changed: ModalContractReport) -> None:
    assert not changed.structural_errors
    assert [(p.actual_index, p.expected_index, p.kind) for p in changed.points] == [
        (p.actual_index, p.expected_index, p.kind) for p in baseline.points]


def _rectangle_mutation(rectangles: dict[str, tuple[FemModalSpectrum, tuple[ModalReference, ...]]],
                        component: str, contract: str) -> None:
    spectrum, expected = rectangles["uniform"]
    baseline = _rectangle_report(spectrum, expected)
    _assert_report(baseline)
    for fraction, green in ((1 - MUTANT_MARGIN, True), (1 + MUTANT_MARGIN, False)):
        changed = _rectangle_report(_mutant(spectrum, expected, baseline, component,
                                           contract_value(contract), fraction), expected)
        _assert_same_pairs(baseline, changed)
        assert changed.within_contract is green
        measured = changed.max_frequency_fraction if component == "real" else changed.max_decay_fraction
        assert measured == pytest.approx(fraction)


def test_rectangle_frequency_mutant_beyond_tolerance_is_red(
    rectangles: dict[str, tuple[FemModalSpectrum, tuple[ModalReference, ...]]],
) -> None:
    _rectangle_mutation(rectangles, "real", "modal_rectangle_frequency")


def test_rectangle_decay_mutant_beyond_tolerance_is_red(
    rectangles: dict[str, tuple[FemModalSpectrum, tuple[ModalReference, ...]]],
) -> None:
    _rectangle_mutation(rectangles, "imag", "modal_rectangle_decay")


def test_trapezoid_mutant_beyond_tolerance_is_red(trapezoid_run: tuple[FemModalSpectrum, CaseAnswer]) -> None:
    spectrum, answer = trapezoid_run
    expected = _external_references(answer)
    baseline = _external_report(spectrum, expected)
    _assert_report(baseline)
    components = ("real",) if all(row.omega.imag == 0 for row in spectrum.solutions) else ("real", "imag")
    for component in components:
        for fraction, green in ((1 - MUTANT_MARGIN, True), (1 + MUTANT_MARGIN, False)):
            changed = _external_report(_mutant(spectrum, expected, baseline, component,
                                              contract_value("modal_trapezoid_vs_fenics"), fraction), expected)
            _assert_same_pairs(baseline, changed)
            assert changed.within_contract is green
            measured = changed.max_frequency_fraction if component == "real" else changed.max_decay_fraction
            assert measured == pytest.approx(fraction)


@pytest.mark.parametrize("side", ["actual", "expected"])
@pytest.mark.parametrize("kind", list(ModalKind))
def test_raw_zero_noise_cannot_be_erased(
    trapezoid_run: tuple[FemModalSpectrum, CaseAnswer], side: str, kind: ModalKind,
) -> None:
    spectrum, answer = trapezoid_run
    expected = _external_references(answer)
    baseline = _external_report(spectrum, expected)
    # 剛性沒有純衰減支：該案例在這題改考共振零虛部。
    selected = kind if any(p.kind is kind for p in baseline.points) else ModalKind.RESONANCE
    candidates = [p for p in baseline.points if p.kind is selected
                  and (p.expected_omega.imag == 0 or p.expected_omega.real == 0)]
    if not candidates:
        # 含阻尼共振兩個分量皆非零：改考同一份頻譜的靜態零根。
        candidates = [p for p in baseline.points if p.kind is ModalKind.STATIC]
    point = candidates[0]
    component = "imag" if point.expected_omega.imag == 0 else "real"
    bound = spectrum.zero_rad_s if point.kind is ModalKind.STATIC else spectrum.component_zero_rad_s
    for fraction, green in ((1 - MUTANT_MARGIN, True), (1 + MUTANT_MARGIN, False)):
        actual_rows, reference_rows = list(spectrum.solutions), list(expected)
        raw = (complex(point.expected_omega.real, fraction * bound) if component == "imag"
               else complex(fraction * bound, point.expected_omega.imag))
        if side == "actual":
            actual_rows[point.actual_index] = replace(actual_rows[point.actual_index], raw_omega=raw)
        else:
            reference_rows[point.expected_index] = replace(reference_rows[point.expected_index], raw_omega=raw)
        changed = _external_report(replace(spectrum, solutions=tuple(actual_rows)), tuple(reference_rows))
        _assert_same_pairs(baseline, changed)
        assert changed.within_contract is green


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_kind", "empty", "relabel"])
def test_judge_rejects_structural_changes(
    rectangles: dict[str, tuple[FemModalSpectrum, tuple[ModalReference, ...]]], mutation: str,
) -> None:
    spectrum, expected = rectangles["uniform"]
    rows = list(expected)
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(rows[-1])
    elif mutation == "duplicate":
        rows[-1] = rows[-2]
    elif mutation == "wrong_kind":
        rows[-1] = replace(rows[-1], kind=ModalKind.STATIC)
    elif mutation == "relabel":
        # 只改種類標籤、數值不動：每一對的數值都在門檻內，只靠結構判定（種類不同）判紅。
        rows[-1] = replace(rows[-1], kind=ModalKind.NONOSCILLATING_DECAY)
    else:
        rows.clear()
    report = _rectangle_report(spectrum, tuple(rows))
    assert report.structural_errors and not report.within_contract
    if mutation == "relabel":
        assert report.points and all(point.within_contract for point in report.points)


def test_static_root_must_fit_modulus_bound(
    rectangles: dict[str, tuple[FemModalSpectrum, tuple[ModalReference, ...]]],
) -> None:
    spectrum, expected = rectangles["rigid"]
    rows = list(spectrum.solutions)
    i = next(i for i, row in enumerate(rows) if row.kind is ModalKind.STATIC)
    # 兩個分量各自界內，模長仍可能界外；不能把靜態 Jordan 雜訊當線性分量界。
    rows[i] = replace(rows[i], raw_omega=complex(spectrum.zero_rad_s, spectrum.zero_rad_s))
    report = _rectangle_report(replace(spectrum, solutions=tuple(rows)), expected)
    assert not report.within_contract and not report.structural_errors


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("parameter", ["frequency_rel", "decay_rel"])
def test_judge_rejects_invalid_relative_limits(value: float, parameter: str) -> None:
    spectrum = FemModalSpectrum((), (), 0.0, 0.0, 0)
    limits = {"frequency_rel": contract_value("modal_rectangle_frequency"),
              "decay_rel": contract_value("modal_rectangle_decay"), parameter: value}
    with pytest.raises(ValueError, match="相對門檻"):
        judge_modal_spectrum(spectrum, (), **limits)


def test_capability_ranges_cover_the_tested_resonances(
    rectangles: dict[str, tuple[FemModalSpectrum, tuple[ModalReference, ...]]],
    trapezoid_run: tuple[FemModalSpectrum, CaseAnswer], problem: ProblemData,
) -> None:
    table = load_capabilities(ROOT / "src" / "aosr" / "config" / "data" / "capabilities.toml")
    entry = table.for_entry("low_frequency_modal")
    for capability in entry.capability:
        if capability.status != "validated":
            continue
        rectangular = capability.room.startswith("長方形")
        upper = CAP if rectangular else float.fromhex(problem["frequency_max_hz"])
        assert capability.frequency_hz[1] == upper
        spectra = [run[0] for run in rectangles.values()] if rectangular else [trapezoid_run[0]]
        assert all(capability.frequency_hz[0] <= row.frequency_hz <= capability.frequency_hz[1]
                   for spectrum in spectra for row in spectrum.modal_table)


def test_capability_note_keeps_the_four_limitations() -> None:
    """升 validated 不准把限制弄丟：能力表這一節的說明要留著四條限制（老闆 10-06 拍板）。"""
    from aosr.config.capabilities import load_capabilities
    from aosr.config.paths import config_path

    entry = next(item for item in load_capabilities(config_path("capabilities.toml")).entry
                 if item.name == "low_frequency_modal")
    for phrase in ("#661 併根標記尚未完成", "最近根選取完整", "嚴格小於", "輪廓積分", "逾時",
                   "照 v3 同一套規則重寫", "聽感門檻", "評分", "逐頻材料", "沒有獨立門檻"):
        assert phrase in entry.note, phrase
