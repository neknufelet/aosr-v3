"""梯形房同網格的外部頻譜結構考卷；精度數字留在離線證據。"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from aosr.geometry.shoebox import Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh
from aosr.physics.fem_modal import FemModalSpectrum, solve_fem_modes
from aosr.physics.modal_convention import ModalKind


ROOT = Path(__file__).resolve().parents[2]
PROBLEM_PATH = ROOT / "blueprint" / "fem_fenics_problem_modal_trapezoid.json"
ANSWER_PATH = ROOT / "blueprint" / "fem_fenics_answers_modal_trapezoid.json"


class MeshData(TypedDict):
    nodes: list[list[str]]
    tetrahedra: list[list[int]]
    boundary_triangles: list[list[int]]
    boundary_wall_indices: list[int]


class CaseData(TypedDict):
    wall_impedances: list[str | None]


class SolverData(TypedDict):
    shifts_hz: list[float]
    modes_per_shift: list[int]
    count_band_edges_hz: list[float]


class ProblemData(TypedDict):
    mesh: MeshData
    cases: dict[str, CaseData]
    density_kg_m3: str
    sound_speed_m_s: str
    frequency_max_hz: str
    v3_solver: SolverData


class AnswerRow(TypedDict):
    omega_rad_s: list[float]
    kind: str


class CaseAnswer(TypedDict):
    solutions: list[AnswerRow]


def load_mesh(problem: ProblemData) -> ShoeboxMesh:
    raw = problem["mesh"]
    return ShoeboxMesh(
        np.asarray([[float.fromhex(x) for x in row] for row in raw["nodes"]]),
        np.asarray(raw["tetrahedra"], dtype=np.int64),
        np.asarray(raw["boundary_triangles"], dtype=np.int64),
        np.asarray(raw["boundary_wall_indices"], dtype=np.int64),
    )


@pytest.fixture(scope="module")
def problem() -> ProblemData:
    return cast(ProblemData, json.loads(PROBLEM_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module", params=["rigid", "uniform", "unequal"])
def run(request: pytest.FixtureRequest, problem: ProblemData) -> tuple[FemModalSpectrum, CaseAnswer]:
    name = str(request.param)
    answers = json.loads(ANSWER_PATH.read_text(encoding="utf-8"))
    walls = {wall: None if value is None else float.fromhex(value)
             for wall, value in zip(Wall.all(), problem["cases"][name]["wall_impedances"], strict=True)}
    spectrum = solve_fem_modes(
        load_mesh(problem), wall_impedances=walls,
        density_kg_m3=float.fromhex(problem["density_kg_m3"]),
        sound_speed_m_s=float.fromhex(problem["sound_speed_m_s"]),
        frequency_max_hz=float.fromhex(problem["frequency_max_hz"]),
        **problem["v3_solver"],
    )
    return spectrum, cast(CaseAnswer, answers["cases"][name])


def test_trapezoid_faces_volume_and_surface(problem: ProblemData) -> None:
    mesh = load_mesh(problem)
    vertices = mesh.nodes[mesh.boundary_triangles]
    scale = max(float(np.max(abs(mesh.nodes))), 1.0)
    rounding = 64 * np.finfo(float).eps * scale
    planes = (vertices[:, :, 2], vertices[:, :, 2] - 3,
              vertices[:, :, 0], vertices[:, :, 0] - 6,
              vertices[:, :, 1], vertices[:, :, 1] - 4 + vertices[:, :, 0] * (0.8 / 6))
    assert set(mesh.boundary_wall_indices) == set(range(len(Wall.all())))
    for index, plane in enumerate(planes):
        assert np.all(abs(plane[mesh.boundary_wall_indices == index]) <= rounding)
    cells = mesh.nodes[mesh.tetrahedra]
    volume = np.sum(abs(np.linalg.det(cells[:, 1:] - cells[:, :1]))) / 6
    surface = np.sum(np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0],
                                            vertices[:, 2] - vertices[:, 0]), axis=1)) / 2
    assert volume == pytest.approx(6 * (4 + 3.2) / 2 * 3)
    assert surface == pytest.approx(2 * 6 * (4 + 3.2) / 2 + 3 * (4 + 3.2 + 6 + math.hypot(6, 0.8)))


def assert_structural_pairs(spectrum: FemModalSpectrum, answer: CaseAnswer) -> None:
    expected = np.asarray([complex(*row["omega_rad_s"]) for row in answer["solutions"]])
    actual = np.asarray([row.omega for row in spectrum.solutions])
    scale = np.maximum(np.maximum(abs(actual[:, None]), abs(expected[None, :])), 1.0)
    distances = abs(actual[:, None] - expected[None, :]) / scale
    rows, columns = linear_sum_assignment(distances)
    assert set(rows) == set(range(len(actual)))
    assert set(columns) == set(range(len(expected)))
    # 不訂精度門檻：每邊最近鄰與整組最優配對須認到同一身分，種類亦相同。
    assert np.array_equal(columns, np.argmin(distances[rows], axis=1))
    assert np.array_equal(rows, np.argmin(distances[:, columns], axis=0))
    assert [spectrum.solutions[int(i)].kind.value for i in rows] == [
        answer["solutions"][int(j)]["kind"] for j in columns]


def test_fenics_all_roots_have_one_to_one_structural_pair(run: tuple[FemModalSpectrum, CaseAnswer]) -> None:
    assert_structural_pairs(*run)


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_kind", "duplicate"])
def test_structural_pairing_rejects_missing_extra_or_changed_identity(
    run: tuple[FemModalSpectrum, CaseAnswer], mutation: str,
) -> None:
    spectrum, original = run
    answer = deepcopy(original)
    if mutation == "missing":
        answer["solutions"].pop()
    elif mutation == "extra":
        answer["solutions"].append(deepcopy(answer["solutions"][-1]))
    elif mutation == "wrong_kind":
        answer["solutions"][-1]["kind"] = ModalKind.STATIC.value
    else:
        answer["solutions"][-1] = deepcopy(answer["solutions"][-2])
    with pytest.raises(AssertionError):
        assert_structural_pairs(spectrum, answer)


def test_guarantee_contains_external_and_v3_resonances(run: tuple[FemModalSpectrum, CaseAnswer]) -> None:
    spectrum, answer = run
    assert spectrum.check is not None
    height = spectrum.check.guaranteed_decay_rate_rad_s
    assert all(row.omega.imag < height for row in spectrum.modal_table)
    assert all(row["omega_rad_s"][1] < height for row in answer["solutions"]
               if row["kind"] == ModalKind.RESONANCE.value)
    assert not any(row.above_guaranteed_decay for row in spectrum.modal_table)


def test_count_bands_use_independent_trapezoid_weyl(
    run: tuple[FemModalSpectrum, CaseAnswer], problem: ProblemData,
) -> None:
    spectrum, answer = run
    assert spectrum.check is not None
    c = float.fromhex(problem["sound_speed_m_s"])
    volume = 6 * (4 + 3.2) / 2 * 3
    surface = 2 * 6 * (4 + 3.2) / 2 + 3 * (4 + 3.2 + 6 + math.hypot(6, 0.8))
    for band in spectrum.check.count_bands:
        external = [row for row in answer["solutions"] if row["kind"] == ModalKind.RESONANCE.value
                    and band.lower_hz < row["omega_rad_s"][0] / (2 * math.pi) <= band.upper_hz]
        assert band.found_resonances == len(external)
        cumulative = [volume * (2 * math.pi * f / c)**3 / (6 * math.pi**2)
                      + surface * (2 * math.pi * f / c)**2 / (16 * math.pi)
                      for f in (band.lower_hz, band.upper_hz)]
        assert band.weyl_estimate == pytest.approx(cumulative[1] - cumulative[0])
        assert band.found_minus_weyl == band.found_resonances - band.weyl_estimate
