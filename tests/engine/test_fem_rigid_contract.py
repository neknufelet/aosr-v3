"""正式 P2 網格的剛性九點解析精度契約。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.fem_lane import FEM_MESH_RANDOM_SEED
from aosr.config.frequency_axis import (
    FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
    FEM_LANE_FREQUENCIES_HZ,
)
from aosr.physics import fem_rigid
from blueprint.reference_fem_check import (
    FEM_PHYSICS_CONTRACT_REL,
    FEM_PHYSICS_FMAX_HZ,
    ModalInputs,
    rigid_eigenfrequencies,
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
    assert case.fem_lane_frequencies_hz == FEM_LANE_FREQUENCIES_HZ
    assert isinstance(FEM_MESH_RANDOM_SEED, int)
    assert fem_rigid.RIGID_MODAL_CONTRACT_REL == FEM_PHYSICS_CONTRACT_REL
    assert fem_rigid.RIGID_MODAL_FMAX_HZ == FEM_PHYSICS_FMAX_HZ


def test_reference_case_uses_its_own_sound_speed_and_rho_c(tmp_path: Path) -> None:
    """答案案例若不再用自己的物理條件，密度或解析模態比例必須紅。"""
    raw = json.loads(ANSWER_PATH.read_text(encoding="utf-8"))
    raw["parameters"]["c_m_s"] = 340.0
    raw["parameters"]["rho_c_pa_s_per_m"] = 420.0
    changed_path = tmp_path / "changed-physics.json"
    changed_path.write_text(json.dumps(raw), encoding="utf-8")

    baseline = fem_rigid.load_rigid_reference_case(ANSWER_PATH)
    changed = fem_rigid.load_rigid_reference_case(changed_path)
    baseline_modes = rigid_eigenfrequencies(
        (baseline.room.Lx, baseline.room.Ly, baseline.room.Lz),
        baseline.sound_speed_m_s,
    )
    changed_modes = rigid_eigenfrequencies(
        (changed.room.Lx, changed.room.Ly, changed.room.Lz),
        changed.sound_speed_m_s,
    )

    assert changed.sound_speed_m_s == 340.0
    assert changed.density_kg_m3 == pytest.approx(420.0 / 340.0)
    assert changed_modes[0] / baseline_modes[0] == pytest.approx(340.0 / 343.0)


def test_reference_case_rejects_a_different_mesh_identity(tmp_path: Path) -> None:
    """正式網格上限比對若被拿掉，改壞案例網格身分時必須紅。"""
    raw = json.loads(ANSWER_PATH.read_text(encoding="utf-8"))
    raw["parameters"]["mesh"]["f_max_cap_hz"] = 249.0
    changed_path = tmp_path / "changed-mesh.json"
    changed_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        fem_rigid.load_rigid_reference_case(changed_path)

    assert str(caught.value) == "答案檔的正式網格設定跟 config 不同"


def test_rigid_path_meshes_through_geometric_crossover_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式路徑若退回 250 Hz 產網格，攔截的交接上端必須讓考卷紅。"""
    case = fem_rigid.load_rigid_reference_case(ANSWER_PATH)
    received_max: dict[str, float] = {}

    def recording_mesh(*args: object, **kwargs: object) -> object:
        del args
        max_frequency_hz = kwargs.get("max_frequency_hz")
        assert isinstance(max_frequency_hz, float)
        received_max["value"] = max_frequency_hz
        return object()

    def inert_solver(*args: object, **kwargs: object) -> NDArray[np.complex128]:
        del args, kwargs
        return np.zeros(len(case.frequencies_hz), dtype=np.complex128)

    monkeypatch.setattr(fem_rigid, "generate_shoebox_mesh", recording_mesh)
    monkeypatch.setattr(fem_rigid, "solve_fem_helmholtz", inert_solver)

    fem_rigid.solve_rigid_reference_case(case)

    assert received_max["value"] == FEM_GEOMETRIC_CROSSOVER_CAP_HZ


def test_fem_lane_rigid_pressures_meet_the_analytical_contract(
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


def test_result_table_prints_only_complete_bands_and_names_unassigned_points() -> None:
    """部分頻帶被印出或軸尾頻點被無聲丟掉時，報表必須紅。"""
    case = fem_rigid.load_rigid_reference_case(ANSWER_PATH)
    axis = case.fem_lane_frequencies_hz
    pressures = np.ones(len(axis), dtype=np.complex128)
    complete_centers = tuple(
        center
        for center in fem_rigid.THIRD_OCTAVE_CENTERS_HZ
        if center * 2.0 ** (-1.0 / 6.0) >= axis[0]
        and center * 2.0 ** (1.0 / 6.0) <= axis[-1]
    )
    expected_unassigned = tuple(
        frequency
        for frequency in axis
        if not any(
            center * 2.0 ** (-1.0 / 6.0)
            <= frequency
            <= center * 2.0 ** (1.0 / 6.0)
            for center in complete_centers
        )
    )

    lines = fem_rigid.rigid_result_table(case, pressures).splitlines()
    band_header_index = lines.index(
        "third_octave_center_hz  frequencies_hz  mean_energy  energy_db_re_1"
    )
    band_lines = lines[band_header_index + 1 : -1]
    printed_centers = tuple(float(line.split(maxsplit=1)[0]) for line in band_lines)
    label, separator, raw_unassigned = lines[-1].partition("  ")

    assert label == "unassigned_frequencies_hz"
    assert separator == "  "
    printed_unassigned = tuple(float(value) for value in raw_unassigned.split(","))

    assert printed_centers == complete_centers
    assert printed_unassigned == pytest.approx(expected_unassigned)
