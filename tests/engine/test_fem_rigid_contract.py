"""正式 P2 網格的剛性九點解析精度契約。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.fem_lane import FEM_MESH_RANDOM_SEED
from aosr.physics import fem_rigid
from blueprint.reference_fem_check import (
    FEM_PHYSICS_CONTRACT_REL,
    FEM_PHYSICS_FMAX_HZ,
    ModalInputs,
    solve_modal_pressure,
)


ANSWER_PATH = Path(__file__).resolve().parents[2] / "blueprint" / "reference_fem_rigid.json"
EXPECTED_POINT_IDENTITIES = (
    ("A", 10.0),
    ("A", 11.220000267028809),
    ("A", 12.600000381469727),
    ("A", 14.140000343322754),
    ("A", 15.869999885559082),
    ("A", 17.81999969482422),
    ("A", 20.0),
    ("B", 10.0),
    ("B", 17.45758620689655),
)


@dataclass(frozen=True)
class ContractRun:
    """同一批正式求解及其獨立解析答案、裁判與牆鐘時間。"""

    case: fem_rigid.RigidReferenceCase
    expected: NDArray[np.complex128]
    report: fem_rigid.RigidContractReport
    elapsed_seconds: float


def _analytical_pressures(
    case: fem_rigid.RigidReferenceCase,
) -> NDArray[np.complex128]:
    inputs = ModalInputs(
        room=(case.room.Lx, case.room.Ly, case.room.Lz),
        source=case.source.as_tuple(),
        receiver=case.receiver.as_tuple(),
        sound_speed=case.sound_speed_m_s,
    )
    solutions = [
        solve_modal_pressure(float(frequency), inputs)
        for frequency in case.frequencies_hz
    ]
    assert all(solution.converged for solution in solutions), (
        "解析模態級數未在參考 oracle 的逐次加倍做法下收斂"
    )
    return np.asarray(
        [solution.pressure for solution in solutions],
        dtype=np.complex128,
    )


@pytest.fixture(scope="module")
def rigid_contract_run() -> ContractRun:
    """正式網格只組裝一次，九點在同一個 solver 流程逐頻 refactor。"""
    case = fem_rigid.load_rigid_reference_case(ANSWER_PATH)
    expected = _analytical_pressures(case)
    started = time.perf_counter()
    report = fem_rigid.solve_rigid_modal_contract(case, expected)
    elapsed = time.perf_counter() - started
    return ContractRun(case, expected, report, elapsed)


def test_reference_case_selects_the_contract_points_and_named_seed() -> None:
    """選點條件、參考題頭或具名固定種子接錯時，題目身分直接紅。"""
    case = fem_rigid.load_rigid_reference_case(ANSWER_PATH)
    identities = tuple(zip(case.set_names, case.frequencies_hz, strict=True))

    assert identities == EXPECTED_POINT_IDENTITIES
    assert isinstance(FEM_MESH_RANDOM_SEED, int)
    assert fem_rigid.RIGID_MODAL_CONTRACT_REL == FEM_PHYSICS_CONTRACT_REL
    assert fem_rigid.RIGID_MODAL_FMAX_HZ == FEM_PHYSICS_FMAX_HZ


def test_formal_rigid_pressures_meet_the_analytical_contract(
    rigid_contract_run: ContractRun,
) -> None:
    """P2 正式網格的剛性九點逐點相對差不可超過決策紙契約。"""
    report = rigid_contract_run.report
    rows = "; ".join(
        f"{row.set_name}:{row.frequency_hz:g}Hz={row.relative_error:.9g}"
        for row in report.points
    )
    message = (
        f"最大相對差 {report.max_relative_error:.9g}，"
        f"用掉 {report.max_contract_fraction * 100.0:.6f}%：{rows}"
    )

    assert report.within_contract, message
    assert all(row.within_contract for row in report.points), message


def test_scaled_pressure_control_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """solver 壓力被放大後仍走同一裁判，證明契約考卷確實能紅。"""
    case = fem_rigid.load_rigid_reference_case(ANSWER_PATH)
    expected = _analytical_pressures(case)

    def scaled_solver(*args: object, **kwargs: object) -> NDArray[np.complex128]:
        del args, kwargs
        return expected * (1.0 + 2.0**-9)

    monkeypatch.setattr(fem_rigid, "solve_fem_helmholtz", scaled_solver)
    report = fem_rigid.solve_rigid_modal_contract(case, expected)

    assert not report.within_contract
    assert report.max_contract_fraction > 1.0


def test_third_octave_bands_use_literal_edges_and_mean_pressure_squared() -> None:
    """抓最近頻帶、插值、平均振幅或對能量用 20log10 都會紅。"""
    frequencies = (14.14, 15.87, 17.82, 20.0)
    pressures = np.asarray((99.0, 1.0, 3.0, 1.0), dtype=np.complex128)
    bands = fem_rigid.third_octave_band_energies(frequencies, pressures)
    by_center = {band.center_hz: band for band in bands}

    assert by_center[12.5].frequencies_hz == ()
    assert by_center[12.5].energy_db is None
    assert by_center[16.0].frequencies_hz == (15.87, 17.82)
    assert by_center[20.0].frequencies_hz == (17.82, 20.0)
    assert by_center[16.0].energy_db == pytest.approx(10.0 * np.log10(5.0))
    assert by_center[20.0].energy_db == pytest.approx(10.0 * np.log10(5.0))
