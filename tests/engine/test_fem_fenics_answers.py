"""正式 P2 網格對凍結 FEniCS 外部答案的逐點精度契約。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from aosr.physics import fem_rigid


ROOT = Path(__file__).resolve().parents[2]
PROBLEM_PATH = ROOT / "blueprint" / "fem_fenics_problem.json"
ANSWER_PATH = ROOT / "blueprint" / "fem_fenics_answers.json"


@dataclass(frozen=True)
class ContractRun:
    """同一批正式求解、FEniCS 外部答案、裁判與牆鐘時間。"""

    problem: fem_rigid.FenicsProblem
    answers: fem_rigid.FenicsAnswers
    report: fem_rigid.FenicsContractReport
    elapsed_seconds: float


@pytest.fixture(scope="module")
def fenics_contract_run() -> ContractRun:
    """凍結網格只載入一次，兩案例各自組裝並逐頻求解。"""
    problem = fem_rigid.load_fenics_problem(PROBLEM_PATH)
    answers = fem_rigid.load_fenics_answers(ANSWER_PATH)
    started = time.perf_counter()
    report = fem_rigid.solve_fenics_contract(problem, answers.pressures)
    elapsed = time.perf_counter() - started
    return ContractRun(problem, answers, report, elapsed)


def test_frozen_mesh_pressures_meet_fenics_contract(
    fenics_contract_run: ContractRun,
) -> None:
    """網格、案例、吸音項或求解流程漂移時，外部答案逐點裁判必須紅。"""
    run = fenics_contract_run
    rows = "; ".join(
        f"{row.case_name}:{row.frequency_hz:g}Hz={row.relative_error:.9g}"
        for row in run.report.points
    )
    message = (
        f"最大相對差 {run.report.max_relative_error:.9g}，"
        f"用掉 {run.report.max_contract_fraction * 100.0:.6f}%，"
        f"耗時 {run.elapsed_seconds:.6f} 秒：{rows}"
    )

    assert run.answers.problem_file == "blueprint/fem_fenics_problem.json"
    assert {row.case_name for row in run.report.points} == {"flat", "lowabs"}
    for case_name in ("flat", "lowabs"):
        observed = tuple(
            row.frequency_hz
            for row in run.report.points
            if row.case_name == case_name
        )
        assert observed == run.problem.frequencies_hz
    assert run.report.within_contract, message
    assert all(row.within_contract for row in run.report.points), message


def test_scaled_fenics_answer_control_is_rejected() -> None:
    """外部答案整批放大一個超界量仍走同一裁判，證明考卷確實能紅。"""
    problem = fem_rigid.load_fenics_problem(PROBLEM_PATH)
    answers = fem_rigid.load_fenics_answers(ANSWER_PATH)
    scaled = {
        name: values * (1.0 + 2.0**-29)
        for name, values in answers.pressures.items()
    }

    report = fem_rigid.judge_fenics_pressures(
        problem,
        answers.pressures,
        scaled,
    )

    assert not report.within_contract
    assert report.max_contract_fraction > 1.0
    assert not all(row.within_contract for row in report.points)


def test_fenics_problem_uses_its_own_physical_conditions(tmp_path: Path) -> None:
    """FEniCS 題目若退回產品預設密度或聲速，載入結果必須紅。"""
    raw = json.loads(PROBLEM_PATH.read_text(encoding="utf-8"))
    raw["density_kg_m3"] = float(1.1).hex()
    raw["sound_speed_m_s"] = float(340.0).hex()
    changed_path = tmp_path / "changed-fenics-problem.json"
    changed_path.write_text(json.dumps(raw), encoding="utf-8")

    problem = fem_rigid.load_fenics_problem(changed_path)

    assert problem.density_kg_m3 == 1.1
    assert problem.sound_speed_m_s == 340.0


def test_fenics_compare_table_has_point_rows_and_final_verdict() -> None:
    """命令列表格若漏逐點 v3／答案／相對差或末行判決就紅。"""
    problem = fem_rigid.load_fenics_problem(PROBLEM_PATH)
    answers = fem_rigid.load_fenics_answers(ANSWER_PATH)
    report = fem_rigid.judge_fenics_pressures(
        problem,
        answers.pressures,
        answers.pressures,
    )

    lines = fem_rigid.fenics_compare_table(report).splitlines()

    assert lines[0] == "case frequency_hz v3_real v3_imag answer_real answer_imag relative_error verdict"
    assert lines[1].startswith("flat 10 ")
    assert lines[-1].startswith("FINAL PASS max_relative_error=")
