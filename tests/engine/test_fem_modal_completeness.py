"""重現重阻尼漏根；離散化與延拓失敗不冒充幾何覆蓋保證。"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from aosr.geometry.shoebox import Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.physics.fem_modal import FemModalSpectrum, ModalSolverOptions, solve_fem_modes
from aosr.physics.fem_helmholtz import assemble_p2_operators
from aosr.physics.fem_modal_check import DecayOrigin, count_bands, mark_decay_origins, mark_zero_mode_continuation
from aosr.physics.modal_convention import ModalKind
from aosr.physics.modal_rectangle_truth import RectangleModalProblem, RectangleTruth, rigid_indices, trace_modes
from tests.engine.test_fem_modal import C, RHO, ROUNDING, _solve


LENGTHS = (6.0, 4.0, 3.0)
CAP = 174.9


@pytest.fixture(scope="module", params=[0.5, 0.7])
def case(request: pytest.FixtureRequest) -> tuple[float, RectangleTruth, FemModalSpectrum, FemModalSpectrum, FemModalSpectrum]:
    beta = float(request.param)
    problem = RectangleModalProblem(LENGTHS, C, RHO, (beta,) * 3)
    truth = trace_modes(problem, frequency_max_hz=CAP, seed_frequency_max_hz=CAP + 60)
    refs = [C / 2 * math.sqrt(sum((n / length)**2 for n, length in zip(index, LENGTHS)))
            for index in rigid_indices(problem, CAP + 60)]
    mesh = generate_shoebox_mesh(Room(*LENGTHS), max_frequency_hz=CAP,
                                 elements_per_wavelength=3, sound_speed_m_s=C, random_seed=1)

    def solve(reference: bool, enlarged: bool = False) -> FemModalSpectrum:
        return solve_fem_modes(mesh, wall_impedances={wall: RHO * C / beta for wall in Wall.all()},
                               density_kg_m3=RHO, sound_speed_m_s=C, frequency_max_hz=CAP,
                               reference_frequencies_hz=refs if reference else None,
                               rigid_reference_frequencies_hz=refs,
                               modes_per_shift=(80, 80, 80) if enlarged else None, retain_shapes=enlarged)

    return beta, truth, solve(False), solve(True), solve(True, True)


def _truth_pairs(truth: RectangleTruth, enriched: FemModalSpectrum) -> dict[tuple[int, int, int], int]:
    mesh = generate_shoebox_mesh(Room(*LENGTHS), max_frequency_hz=CAP,
                                 elements_per_wavelength=3, sound_speed_m_s=C, random_seed=1)
    ops = assemble_p2_operators(mesh)
    expected = []
    for mode in truth.modal_table:
        shape = np.ones(ops.basis.N, dtype=np.complex128)
        for axis, (q, n, length) in enumerate(zip(mode.axis_wave_numbers, mode.index, LENGTHS)):
            x = ops.basis.doflocs[axis] - length / 2
            shape *= np.cos(q * x) if n % 2 == 0 else np.sin(q * x) / q
        shape /= math.sqrt(float(np.vdot(shape, ops.mass @ shape).real))
        expected.append(shape)
    actual = []
    for fem_mode in enriched.modal_table:
        assert fem_mode.shape is not None
        actual.append(fem_mode.shape)
    overlaps = abs(np.asarray(expected).conj() @ (ops.mass @ np.asarray(actual).T))
    frequencies = abs(np.asarray([m.omega for m in truth.modal_table])[:, None]
                      - np.asarray([m.omega for m in enriched.modal_table])[None, :])
    # 強阻尼形狀不正交；兩個獨立整組一對一配對必須給相同物理身分。
    rows, columns = linear_sum_assignment(frequencies)
    shape_rows, shape_columns = linear_sum_assignment(-overlaps)
    assert set(rows) == set(range(len(expected)))
    assert np.array_equal(shape_rows, rows)
    assert np.array_equal(shape_columns, columns)
    # 還要核實整組最優配對唯一：禁止任一已選邊會使形狀配對總分嚴格下降。
    score = float(overlaps[rows, columns].sum())
    for i, j in zip(rows, columns, strict=True):
        excluded = -overlaps.copy()
        excluded[i, j] = math.inf
        alternatives, alternatives_columns = linear_sum_assignment(excluded)
        assert float(overlaps[alternatives, alternatives_columns].sum()) < score
    return {truth.modal_table[int(i)].index: int(j) for i, j in zip(rows, columns, strict=True)}


def _found_indices(sparse: FemModalSpectrum, enriched: FemModalSpectrum) -> set[int]:
    # 同一網格、同一矩陣，顯式增加名額後的根作身分錨，不靠近頻率硬去重。
    actual = np.asarray([mode.omega for mode in sparse.modal_table])
    reference = np.asarray([mode.omega for mode in enriched.modal_table])
    distances = abs(actual[:, None] - reference[None, :])
    rows, columns = linear_sum_assignment(distances)
    assert set(rows) == set(range(len(actual)))
    assert set(columns) == set(np.argmin(distances, axis=1))
    # 沿用既有同矩陣去重容差，這不是新訂 FEM 對連續真值的精度門檻。
    limit = ModalSolverOptions().dedup_relative_tolerance * np.maximum(
        np.maximum(abs(actual[rows]), abs(reference[columns])), 2 * math.pi)
    assert np.all(distances[rows, columns] <= limit)
    return {int(j) for j in columns}


@pytest.mark.parametrize("reference_plan", [False, True])
def test_missing_root_is_outside_guarantee_and_explicit_rerun_recovers_it(
    case: tuple[float, RectangleTruth, FemModalSpectrum, FemModalSpectrum, FemModalSpectrum], reference_plan: bool,
) -> None:
    beta, truth, weyl, reference, enriched = case
    sparse = reference if reference_plan else weyl
    assert sparse.check is not None and enriched.check is not None
    target = (3, 2, 2) if beta == 0.5 else (0, 2, 1)
    mode = next(mode for mode in truth.modal_table if mode.index == target)
    paired = _truth_pairs(truth, enriched)[target]
    found = enriched.modal_table[paired]
    assert paired not in _found_indices(sparse, enriched)
    assert mode.omega.imag > sparse.check.guaranteed_decay_rate_rad_s
    assert found.omega.imag > sparse.check.guaranteed_decay_rate_rad_s
    assert enriched.check.guaranteed_decay_rate_rad_s > sparse.check.guaranteed_decay_rate_rad_s
    assert mode.omega.imag < enriched.check.guaranteed_decay_rate_rad_s
    assert found.omega.imag < enriched.check.guaranteed_decay_rate_rad_s
    assert not found.above_guaranteed_decay


@pytest.mark.parametrize("reference_plan", [False, True])
def test_every_tracked_truth_resonance_inside_guarantee_has_unique_fem_match(
    case: tuple[float, RectangleTruth, FemModalSpectrum, FemModalSpectrum, FemModalSpectrum], reference_plan: bool,
) -> None:
    _, truth, weyl, reference, enriched = case
    sparse = reference if reference_plan else weyl
    assert sparse.check is not None
    pairs = _truth_pairs(truth, enriched)
    found = _found_indices(sparse, enriched)
    protected = {pairs[mode.index] for mode in truth.modal_table
                 if mode.omega.imag < sparse.check.guaranteed_decay_rate_rad_s}
    assert protected
    assert protected <= found
    for row in sparse.solutions:
        assert row.above_guaranteed_decay == (row.omega.imag > sparse.check.guaranteed_decay_rate_rad_s)


@pytest.mark.parametrize("beta", [0.0, 1e-6, 0.25])
def test_rigid_and_light_damping_found_modes_are_inside_guarantee(beta: float) -> None:
    spectrum = _solve((beta,) * 3, 3)
    assert spectrum.check is not None
    assert all(row.omega.imag < spectrum.check.guaranteed_decay_rate_rad_s for row in spectrum.modal_table)
    assert not any(row.above_guaranteed_decay for row in spectrum.modal_table)
    assert spectrum.check.guaranteed_min_t60_s == pytest.approx(
        3 * math.log(10) / spectrum.check.guaranteed_decay_rate_rad_s)


def test_counts_are_raw_numbers_and_zero_branch_needs_continuation_evidence() -> None:
    spectrum = _solve((0.25,) * 3, 3)
    truth = trace_modes(RectangleModalProblem((2.0, 1.7, 1.3), C, RHO, (0.25,) * 3), frequency_max_hz=165.0)
    zero_branch = next(mode for mode in truth.modes if mode.index == (0, 0, 0))
    assert zero_branch.reached_target and zero_branch.quantities.kind is ModalKind.NONOSCILLATING_DECAY
    decays = {i: zero_branch.index for i, row in enumerate(spectrum.solutions)
              if row.kind is ModalKind.NONOSCILLATING_DECAY}
    # 這組所有非零解已在原有收斂考卷一對一配到真值；本題再核實純衰減的唯一對應。
    truth_decays = [m for m in truth.modes if m.reached_target
                    and m.quantities.kind is ModalKind.NONOSCILLATING_DECAY]
    assert [m.index for m in truth_decays] == [zero_branch.index]
    assert list(decays.values()) == [zero_branch.index]
    assert decays
    assert all(spectrum.solutions[i].decay_origin is DecayOrigin.UNCONFIRMED for i in decays)
    marked = mark_decay_origins(spectrum, decays)
    assert marked.check is not None
    assert marked.check.zero_mode_continuation_count == len(decays)
    assert marked.check.unconfirmed_decay_count == 0
    assert [row.omega for row in marked.solutions] == [row.omega for row in spectrum.solutions]
    for band in marked.check.count_bands:
        expected = [r for r in marked.modal_table if band.lower_hz < r.frequency_hz <= band.upper_hz]
        assert band.found_resonances == len(expected)
        assert band.found_minus_weyl == band.found_resonances - band.weyl_estimate
        # Weyl 兩項估計，考卷自己用長方體體積與表面積算（不讀網格、不叫被測程式）。
        volume, surface = 2.0 * 1.7 * 1.3, 2 * (2.0 * 1.7 + 2.0 * 1.3 + 1.7 * 1.3)
        weyl = [volume * (2 * math.pi * f / C)**3 / (6 * math.pi**2) + surface * (2 * math.pi * f / C)**2 / (16 * math.pi)
                for f in (band.lower_hz, band.upper_hz)]
        assert band.weyl_estimate == pytest.approx(weyl[1] - weyl[0], rel=ROUNDING)
        assert band.rigid_reference_count is None and band.found_minus_rigid is None
    assert "no edge term" in marked.check.weyl_terms
    # 移位會回傳上限以外的根（這組上限 165 Hz、最高移位 125 Hz、名額照參考個數）：超界個數不能一律報零。
    assert marked.check.returned_above_limit_count > 0
    # 模擬呼叫端已由延拓配到不同剛性起點；純衰減種類相同，來源分列計數。
    i = next(iter(decays))
    other = replace(spectrum.solutions[i], omega=2 * spectrum.solutions[i].omega)
    two = replace(spectrum, solutions=(*spectrum.solutions, other))
    both = mark_decay_origins(two, {i: (0, 0, 0), len(two.solutions) - 1: (1, 0, 0)})
    assert both.solutions[i].decay_origin is DecayOrigin.ZERO_MODE_CONTINUATION
    assert both.solutions[-1].decay_origin is DecayOrigin.OVERDAMPED
    assert both.check is not None
    assert both.check.overdamped_count == sum(row.decay_origin is DecayOrigin.OVERDAMPED for row in both.solutions)


def test_count_band_partition_and_rigid_reference_keep_upper_edge_once() -> None:
    spectrum = _solve((0.0,) * 3, 3)
    edge = spectrum.modal_table[0].frequency_hz
    edges = (0.0, edge, 165.0)
    refs = [row.frequency_hz for row in spectrum.modal_table]
    bands = count_bands(spectrum.solutions, edges, (0.0, 1.0, 2.0), refs)
    assert sum(b.found_resonances for b in bands) == len(spectrum.modal_table)
    assert all(b.found_minus_rigid == 0 for b in bands)
    assert all(b.rigid_reference_count == b.found_resonances for b in bands)


@pytest.mark.parametrize("edges", [(0.0,), (1.0, 2.0), (0.0, 2.0, 1.0), (0.0, math.inf)])
def test_invalid_count_edges_rejected(edges: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        count_bands((), edges, tuple(0.0 for _ in edges))


def test_decay_origin_rejects_static_row_and_duplicate_labels() -> None:
    spectrum = _solve((0.25,) * 3, 3)
    static = next(i for i, row in enumerate(spectrum.solutions) if row.kind is ModalKind.STATIC)
    with pytest.raises(ValueError):
        mark_decay_origins(spectrum, {static: (0, 0, 0)})
    with pytest.raises(ValueError):
        mark_decay_origins(spectrum, {0: (0, 0, 0), 1: (0, 0, 0)})


def test_real_overdamped_roots_are_separate_from_traced_constant_branch(
    case: tuple[float, RectangleTruth, FemModalSpectrum, FemModalSpectrum, FemModalSpectrum],
) -> None:
    beta, truth, _, _, spectrum = case
    zero_branch = next(mode for mode in truth.modes if mode.index == (0, 0, 0))
    decays = [(i, row) for i, row in enumerate(spectrum.solutions)
              if row.kind is ModalKind.NONOSCILLATING_DECAY]
    assert decays
    if not zero_branch.reached_target:
        # β=0.7 的既有延拓不能穿越奇異點；不可猜零支來源。
        assert beta == 0.7
        assert all(row.decay_origin is DecayOrigin.UNCONFIRMED for _, row in decays)
        return
    assert zero_branch.quantities.kind is ModalKind.NONOSCILLATING_DECAY
    mesh = generate_shoebox_mesh(Room(*LENGTHS), max_frequency_hz=CAP,
                                 elements_per_wavelength=3, sound_speed_m_s=C, random_seed=1)
    ops = assemble_p2_operators(mesh)
    shape = np.prod([np.cos(q * (ops.basis.doflocs[axis] - length / 2))
                     for axis, (q, length) in enumerate(zip(zero_branch.axis_wave_numbers, LENGTHS))], axis=0)
    overlaps = []
    for _, row in decays:
        assert row.shape is not None
        overlaps.append(abs(np.vdot(shape, ops.mass @ row.shape)))
    best = int(np.argmax(overlaps))
    # 形狀辨識與複數頻率辨識必須同意，不能按衰減排序。
    nearest = min(range(len(decays)), key=lambda j: abs(decays[j][1].omega - zero_branch.omega))
    assert best == nearest
    assert np.all(overlaps[best] > np.delete(overlaps, best))
    distances = [abs(row.omega - zero_branch.omega) for _, row in decays]
    assert np.all(distances[nearest] < np.delete(distances, nearest))
    marked = mark_zero_mode_continuation(spectrum, decays[best][0])
    assert marked.solutions[decays[best][0]].decay_origin is DecayOrigin.ZERO_MODE_CONTINUATION
    others = [row for i, row in enumerate(marked.solutions)
              if row.kind is ModalKind.NONOSCILLATING_DECAY and i != decays[best][0]]
    assert others and all(row.decay_origin is DecayOrigin.OVERDAMPED for row in others)
    assert marked.check is not None
    assert marked.check.overdamped_count == len(others)
    assert marked.check.unconfirmed_decay_count == 0


def test_overdamped_mirror_branches_may_share_nonzero_rigid_index() -> None:
    spectrum = _solve((0.25,) * 3, 3)
    original = next(row for row in spectrum.solutions if row.kind is ModalKind.NONOSCILLATING_DECAY)
    # 僅驗來源接口：同一非零指標可有兩個鏡像延拓結局，不假稱此小案例真的過阻尼。
    rows = (replace(original, omega=2 * original.omega), replace(original, omega=3 * original.omega))
    marked = mark_decay_origins(replace(spectrum, solutions=rows), {0: (1, 0, 0), 1: (1, 0, 0)})
    assert all(row.decay_origin is DecayOrigin.OVERDAMPED for row in marked.solutions)
    assert marked.check is not None and marked.check.overdamped_count == len(rows)


def test_default_count_bands_have_native_json_numbers() -> None:
    # 正式參考房曾算完卻在 np.int64 寫 JSON 時失敗；此題守資料型別，沒有鎖個數。
    spectrum = _solve((0.0,) * 3, 3)
    assert spectrum.check is not None
    encoded = json.dumps(asdict(spectrum.check), allow_nan=False)
    payload = json.loads(encoded)
    assert payload["count_bands"] == [asdict(band) for band in spectrum.check.count_bands]
    assert all(type(band.found_resonances) is int for band in spectrum.check.count_bands)
