"""外部有限元素能量只准完整、有限、非負，且必須一路傳到報表。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import cast

import pytest

from aosr.config import frequency_axis
from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.capabilities import load_capabilities
from aosr.geometry.shoebox import Point
from aosr.physics import three_lane_report as report, three_lane_report_batch as batch
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from tests.engine import test_three_lane_reports_batch as control
from tests.engine import _scoring_source_model_control as scoring
from tests.engine import _source_model_control as source_control
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_scheme_pipeline import _many_fem
from tests.engine import _fem_shared_cases as cases


@pytest.fixture(autouse=True)
def small_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    report_axis = cases.FREQUENCIES + tuple(
        value for value in frequency_axis.GEOMETRIC_LANE_FREQUENCIES_HZ
        if value > frequency_axis.FEM_GEOMETRIC_CROSSOVER_CAP_HZ)
    monkeypatch.setattr(frequency_axis, "FEM_LANE_FREQUENCIES_HZ", cases.FREQUENCIES)
    monkeypatch.setattr(frequency_axis, "GEOMETRIC_LANE_FREQUENCIES_HZ", report_axis)


def _energy() -> dict[tuple[str, str], Sequence[float]]:
    frequencies, _ = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)
    energies: dict[tuple[str, str], Sequence[float]] = {pair: list(values) for pair, values in control._fake_many_fem(
        room=control.FEM_ROOM, sources=control.FEM_SOURCES, receivers=control.FEM_RECEIVERS,
        wall_impedances=control.WALLS, frequencies_hz=frequencies,
        density_kg_m3=control.DENSITY, sound_speed_m_s=control.SOUND_SPEED).items()}
    pair = next(iter(energies))
    energies[pair] = [int(value) for value in energies[pair]]
    return energies


def _solve(energies: Mapping[tuple[str, str], Sequence[float]] | None,
           *, batch_fem: bool = True) -> dict[tuple[str, str], report.ThreeLaneReport]:
    return batch.solve_reports(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=control.FEM_ROOM, sources=control.FEM_SOURCES, receivers=control.FEM_RECEIVERS,
        sound_speed_m_s=control.SOUND_SPEED, density_kg_m3=control.DENSITY,
        impedance_by_wall=control.WALLS, scattering_by_wall=None, capability=None,
        reflection_order_k=1, low_frequency_axis=LowFrequencyAxis.SEARCH,
        batch_fem=batch_fem, fem_energies=energies)


def _forbidden(**kwargs: object) -> object:
    raise AssertionError("注入時不得求解 FEM")


def test_injection_equals_solver_standin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "_solve_report_late_decay", control._fast_late_decay)
    monkeypatch.setattr(batch, "solve_geometric_late_energy", source_control.fake_late_energy)
    energies = _energy()
    monkeypatch.setattr(report, "_solve_fem_energies", lambda **kwargs: {pair: tuple(values) for pair, values in energies.items()})
    expected = _solve(None)
    monkeypatch.setattr(report, "_solve_fem_energies", _forbidden)
    actual = _solve(energies)
    assert actual == expected
    for raw in actual.values():
        assert isinstance(raw.fem_energy, tuple)
        assert all(isinstance(value, float) for value in raw.fem_energy)


@pytest.mark.parametrize("change", ("missing", "extra", "length", "nan", "inf", "negative", "single"))
def test_invalid_injection_rejected(monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    monkeypatch.setattr(report, "_solve_report_late_decay", control._fast_late_decay)
    monkeypatch.setattr(batch, "solve_geometric_late_energy", source_control.fake_late_energy)
    monkeypatch.setattr(report, "_solve_fem_energies", _forbidden)
    energies = _energy()
    pair = next(iter(energies))
    if change == "missing":
        del energies[pair]
    elif change == "extra":
        energies["missing", "extra"] = energies[pair]
    elif change == "length":
        energies[pair] = energies[pair][1:]
    elif change in {"nan", "inf", "negative"}:
        energies[pair] = [dict(nan=float("nan"), inf=float("inf"), negative=-1.0)[change]] + list(energies[pair][1:])
    with pytest.raises(ValueError):
        _solve(energies, batch_fem=change != "single")


@pytest.mark.parametrize("values", (["1e3", 1.0, 2.0], [True, 1.0, 2.0], "123", b"123",
                                   [-0.0, 1.0, 2.0], {0: 1.0, 1: 2.0, 2: 3.0}),
                         ids=("numeric_strings", "bool", "str_sequence", "bytes_sequence",
                              "negative_zero", "not_sequence"))
def test_injection_rejects_invalid_value_types(values: object) -> None:
    valid = (1.0, 2.0, 3.0)
    energies: dict[tuple[str, str], Sequence[float]] = {
        (source, receiver): valid for source in control.FEM_SOURCES for receiver in control.FEM_RECEIVERS}
    energies[next(iter(energies))] = cast(Sequence[float], values)
    with pytest.raises(ValueError):
        batch._injected_energies(energies, control.FEM_SOURCES, control.FEM_RECEIVERS, len(valid))


def test_pipeline_forwards_injection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from aosr.reporting import pipeline, physics_stage
    from aosr.reporting.result import SchemeResult
    from aosr.reporting.validation import checked_inputs
    from aosr.physics import report_io

    scheme = cases.small_scheme("wall-1").model_copy(update={"purpose": scoring.PURPOSE})
    table = load_capabilities(scoring.TARGETS.with_stem("capabilities"))
    scheme, documents = checked_inputs(scheme, capabilities=table, directivity=DIRECTIVITY)
    solved = report_io.solver_inputs(next(iter(documents.values()))[1])
    energies = _many_fem(
        room=solved.room, sources=scheme.speakers,
        receivers={point.receiver_id: Point(*point.position_m) for point in scheme.receiver_set.points},
        wall_impedances=solved.impedance_by_wall,
        frequencies_hz=low_frequency_axis_frequencies(solved.low_frequency_axis)[0],
        density_kg_m3=solved.density_kg_m3, sound_speed_m_s=solved.sound_speed_m_s)
    for module, name, fake in scoring.STAND_INS:
        monkeypatch.setattr(module, name, fake)
    monkeypatch.setattr(physics_stage, "report_capability", lambda table: report._unchecked_capability())
    def run(energy: Mapping[tuple[str, str], Sequence[float]] | None) -> 'SchemeResult':
        return pipeline.run_scheme(
            scheme, capabilities=table, directivity=DIRECTIVITY, quality_targets_path=scoring.TARGETS,
            engine_commit="control", program_fingerprint="calc-v1:" + "0" * 64,
            physics_identity="phys-v1:" + "0" * 64, run_date=date(2026, 9, 27), fem_energies=energy)

    monkeypatch.setattr(report, "_solve_fem_energies", lambda **kwargs: energies)
    expected = run(None)
    monkeypatch.setattr(report, "_solve_fem_energies", _forbidden)
    actual = run(energies)
    assert actual.pairs == expected.pairs
    assert actual.candidate == expected.candidate
    target = tmp_path / "result"
    target.write_text(actual.model_dump_json(), encoding="utf-8")
    assert target.read_text(encoding="utf-8") == actual.model_dump_json()
