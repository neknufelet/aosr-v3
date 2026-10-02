"""物理段搬家後的載入邊界、逐欄等值與子行程執行緒設定。"""
from __future__ import annotations

import copy
import inspect
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import ParamSpec, TypeVar

import pytest

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point
from aosr.physics import report_io, three_lane_report
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.report_output import output_from_report
from aosr.physics.third_octave_decay import build_third_octave_decay
from aosr.reporting import pipeline
from aosr.reporting.result import PairResult, SchemeResult
from aosr.reporting.scheme import Scheme, expected_pairs
from aosr.reporting.validation import SchemeValidationError, checked_inputs
from aosr.runtime import child_process_env
from tests.engine import _scoring_source_model_control as control
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_scheme_pipeline import _many_fem, _scheme

_P = ParamSpec("_P")
_T = TypeVar("_T")
_Reports = dict[tuple[str, str], three_lane_report.ThreeLaneReport]


def _capture(function: Callable[_P, _T], calls: list[dict[str, object]],
             results: list[_T]) -> Callable[_P, _T]:
    signature = inspect.signature(function)

    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        calls.append(copy.deepcopy(dict(arguments.arguments)))
        result = function(*args, **kwargs)
        results.append(result)
        # 報表內仍有可變字典：原回傳留給直接組裝，管線拿深拷貝，原地改值才不會兩邊一起變。
        return copy.deepcopy(result)

    return wrapped


@dataclass(frozen=True)
class _Parts:
    scheme: Scheme
    documents: dict[tuple[str, str], tuple[dict[str, object], report_io.ReportInput]]
    raw: dict[tuple[str, str], three_lane_report.ThreeLaneReport]
    result: SchemeResult


def _direct_reports(scheme: Scheme, table: CapabilityTable,
                    calls: list[dict[str, object]], reports: list[_Reports]) -> tuple[
    dict[tuple[str, str], tuple[dict[str, object], report_io.ReportInput]],
    dict[tuple[str, str], three_lane_report.ThreeLaneReport],
]:
    scheme, documents = checked_inputs(scheme, capabilities=table, directivity=DIRECTIVITY)
    solved = report_io.solver_inputs(next(iter(documents.values()))[1])
    expected = dict(
        source_model=solved.source_model, room=solved.room, sources=scheme.speakers,
        receivers={point.receiver_id: Point(*point.position_m)
                   for point in scheme.receiver_set.points},
        sound_speed_m_s=solved.sound_speed_m_s, density_kg_m3=solved.density_kg_m3,
        impedance_by_wall=solved.impedance_by_wall,
        scattering_by_wall=solved.scattering_by_wall,
        reflection_order_k=solved.reflection_order_k,
        low_frequency_axis=solved.low_frequency_axis,
        capability=three_lane_report._unchecked_capability(),
        fem_energies=None,
    )
    assert calls == [expected], "管線必須恰好真求解一次，完整引數等於直接路線"
    assert reports, "必須攔到真求解器的回傳"
    return documents, reports[0]


@pytest.fixture(scope="module")
def parts() -> Iterator[_Parts]:
    from aosr.reporting import physics_stage

    table = load_capabilities(config_path("capabilities.toml"))
    scheme = _scheme("wall-1")
    calls: list[str] = []
    solve_calls: list[dict[str, object]] = []
    reports: list[_Reports] = []

    def capability(table: CapabilityTable) -> three_lane_report.ReportCapability:
        calls.append("report_capability")
        return three_lane_report._unchecked_capability()

    with pytest.MonkeyPatch.context() as patch:
        for module, name, fake in control.STAND_INS:
            patch.setattr(module, name, fake)
        patch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
        patch.setattr(three_lane_report, "solve_three_lane_reports",
                      _capture(three_lane_report.solve_three_lane_reports, solve_calls, reports))
        patch.setattr(physics_stage, "report_capability", capability)
        result = pipeline.run_scheme(
            scheme, capabilities=table, directivity=DIRECTIVITY,
            quality_targets_path=control.TARGETS, engine_commit="control",
            program_fingerprint="calc-v1:" + "2" * 64, physics_identity="phys-v1:" + "0" * 64, run_date=date(2026, 9, 27),
        )
        assert "report_capability" in calls
        documents, raw = _direct_reports(scheme, table, solve_calls, reports)
    # 替身用完就拆（照 test_scheme_repair.py::wall_2 的寫法）；後面幾題只讀這份零件。
    yield _Parts(scheme, documents, raw, result)


def test_pipeline_records_caller_fingerprint(parts: _Parts) -> None:
    assert parts.result.program_fingerprint == "calc-v1:" + "2" * 64


def test_physics_stage_imports_no_scoring_registry_or_jax() -> None:
    # 求解時才在函式裡載入的兩支（批次求解、路徑表）也一起載入：只載入外層會看不到真的在跑的那段。
    probe = ("import sys; import aosr.reporting.physics_stage, aosr.physics.three_lane_report_batch, "
             "aosr.physics.report_path_table; print('\\n'.join(sorted(sys.modules)))")
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                               text=True, check=True)
    modules = set(completed.stdout.splitlines())
    assert {"aosr.reporting.physics_stage", "aosr.physics.three_lane_report_batch",
            "aosr.physics.report_path_table"} <= modules
    forbidden = {"aosr.config.quality_targets", "aosr.reporting.result",
                 "aosr.reporting.evaluation", "aosr.physics.reflection_window"}
    assert modules.isdisjoint(forbidden)
    assert {name for name in modules if name.startswith("aosr.scoring")} <= {
        "aosr.scoring", "aosr.scoring.channel_group", "aosr.scoring.receiver_set"}
    assert not any(name == root or name.startswith(f"{root}.")
                   for name in modules for root in ("jax", "flax"))


def _raw_pair(parts: _Parts, key: tuple[str, str]) -> PairResult:
    speaker_id, receiver_id = key
    document, inputs = parts.documents[key]
    raw = parts.raw[key]
    lane = raw.geometric_lane
    role = next(role for speaker, receiver, role in expected_pairs(parts.scheme)
                if (speaker, receiver) == key)
    return PairResult(
        role=role, speaker_id=speaker_id, receiver_id=receiver_id,
        report_id=f"report-{speaker_id}-{receiver_id}", input_document=document,
        report=output_from_report(raw, inputs=inputs, with_points=True,
                                  path_table_inputs=report_io.solver_inputs(inputs)),
        screen=build_reflection_screen(inputs, lane.frequencies_hz),
        third_octave_decay=build_third_octave_decay(raw, inputs),
    )


def test_run_scheme_pairs_equal_parts_built_directly(parts: _Parts) -> None:
    """管線零件等於直接組裝；傳錯求解引數與多叫、沒叫求解器由 parts 的完整引數比對守。

    同引數不同結果仍由 test_three_lane_reports_batch 的批次對逐份題守（本支不做 B8）。
    """
    expected = tuple(_raw_pair(parts, key) for key in parts.documents)
    assert parts.result.pairs == expected
    for stored, direct in zip(parts.result.pairs, expected, strict=True):
        assert stored.model_dump_json() == direct.model_dump_json()


def test_window_from_stored_parts_equals_window_from_raw_lane(parts: _Parts) -> None:
    from aosr.reporting.evaluation import build_pair_window
    from aosr.physics.reflection_window import build_reflection_window

    for pair in parts.result.pairs:
        key = pair.speaker_id, pair.receiver_id
        inputs = parts.documents[key][1]
        lane = parts.raw[key].geometric_lane
        for window_s in (control.WINDOW_S, 0.02):
            expected = build_reflection_window(
                inputs, frequencies_hz=lane.frequencies_hz,
                scattering_coefficient=lane.scattering, window_s=window_s)
            rebuilt = build_pair_window(inputs, pair.report, window_s)
            assert rebuilt == expected
            assert rebuilt.model_dump_json() == expected.model_dump_json()


def test_build_pair_window_refuses_missing_path_table(parts: _Parts) -> None:
    from aosr.reporting.evaluation import build_pair_window

    pair = parts.result.pairs[0]
    inputs = parts.documents[pair.speaker_id, pair.receiver_id][1]
    report = pair.report.model_copy(update={"path_table": None})
    with pytest.raises(ValueError, match="path_table"):
        build_pair_window(inputs, report, control.WINDOW_S)


def test_child_process_env_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    inherited = {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "VIRTUAL_ENV", "TMPDIR"}
    thread_vars = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
    for name in inherited:
        monkeypatch.setenv(name, f"fixture-{name}")
    for name in thread_vars:
        monkeypatch.setenv(name, "99")
    default = child_process_env()
    assert set(default) == inherited | {"PYTHONPATH"}
    assert all(default[name] == f"fixture-{name}" for name in inherited)
    single = child_process_env(threads=1)
    assert single == default | dict.fromkeys(thread_vars, "1")
    assert child_process_env(threads=3) == default | dict.fromkeys(thread_vars, "3")
    with pytest.raises(ValueError, match="正整數"):
        child_process_env(threads=0)
    with pytest.raises(ValueError, match="正整數"):
        child_process_env(threads=-1)
    with pytest.raises(ValueError, match="正整數"):
        child_process_env(threads=True)
    monkeypatch.delenv("LC_ALL")
    assert "LC_ALL" not in child_process_env()


def test_broken_pair_is_reported_before_broken_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """某一對輸入壞了、登記簿也讀不到：照拆分前的先後，先報那一對（驗完每一對才讀登記簿）。"""
    def broken_pair(document: object, capabilities: object, directivity: object) -> object:
        raise report_io.ReportInputError("probe", (("source_m", "probe: pair broken", True),))

    def forbidden_registry(*args: object) -> object:
        raise AssertionError("驗每一對之前就讀了登記簿")

    monkeypatch.setattr(report_io, "load_input_document", broken_pair)
    monkeypatch.setattr(pipeline, "read_registry_settings", forbidden_registry)
    with pytest.raises(SchemeValidationError, match="probe: pair broken"):
        pipeline.run_scheme(
            _scheme("wall-1"), capabilities=load_capabilities(config_path("capabilities.toml")),
            directivity=DIRECTIVITY, quality_targets_path=control.TARGETS, engine_commit="probe",
            program_fingerprint="calc-v1:" + "0" * 64, physics_identity="phys-v1:" + "0" * 64, run_date=date(2026, 10, 1),
        )
