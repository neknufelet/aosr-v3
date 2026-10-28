"""有限元素替身方程式（投影式降階，#561）：小房間建模、留出材料對完整解、存讀、手算小例與變異考卷。

房間 1.6×1.2×1.0 m、正式 300 Hz 網格密度與種子（3287 個未知數）、兩支喇叭三個座位；頻率軸取正式有限元素軸
每隔一點（1/12 八度、20–293 Hz）。材料是固定種子抽的「阻力＋彈簧」型逐頻阻抗 Z/ρc = R − jX·(100 Hz / f)，
R 在 1.5–60、X 在 0–5，六面各自抽；訓練與留出用不同種子。這不是真實構造材料，只是一批被動、跨低到高吸音的阻抗。

建模設定取正式設定檔，只把選點停止值收緊到 1e-3。施工時實量（單緒）：這一題的設定照正式 0.01 建，
起點 64 次完整解抽到的殘差就在停止值內、不補點，留出材料逐頻最大 0.136 dB、帶能量最大 0.0077 dB，在尺內但餘裕只有
1.4–1.6 倍；換成訓練 24 組、整條 94 點軸、留出 3 組（同一間房、同一種材料抽法）時，正式 0.01 量到 0.296 dB 與
0.026 dB，兩把尺都超過。1e-3 補一輪（96 次完整解）後這一題量到 0.0123 dB 與 0.00063 dB。停止值只是選點用，
準不準由留出材料判——這一題守的是這件事，不是正式停止值夠不夠。
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.fem_reduced import ReducedModelSettings, load_reduced_model_settings
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ, FEM_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.fem_helmholtz import FrequencyWallImpedances, interpolate_pressure, point_source_load
from aosr.physics.fem_reduced import (
    FullOperators, ReducedMatrices, ReducedModel, even_indices, evaluate_reduced, export_room_operators,
    system_coefficients,
)
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics import fem_reduced_store
from aosr.physics.fem_reduced_check import (
    ValidationReport, third_octave_bands, validate_reduced, within_reduced_contract,
)
from aosr.physics.fem_reduced_store import load_reduced, manifest_path, save_reduced
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value

DENSITY = 1.2
SOUND_SPEED = 343.0
ROOM = Room(1.6, 1.2, 1.0)
SOURCES = {"left": Point(0.3, 0.3, 0.4), "right": Point(0.4, 0.8, 0.5)}
SEATS = {"front": Point(1.2, 0.7, 0.6), "back": Point(1.0, 0.4, 0.3), "side": Point(1.3, 0.9, 0.5)}
AXIS = FEM_LANE_FREQUENCIES_HZ[::2]
TEST_STOP_RESIDUAL = 1.0e-3
FREQUENCY_RULER = contract_value("fem_reduced_holdout_frequency_db")
BAND_RULER = contract_value("fem_reduced_holdout_band_db")


def _materials(seed: int, count: int) -> list[FrequencyWallImpedances]:
    rng = np.random.default_rng(seed)
    frequencies = np.asarray(AXIS)
    rho_c = DENSITY * SOUND_SPEED
    sets: list[FrequencyWallImpedances] = []
    for _ in range(count):
        walls: dict[Wall, tuple[complex, ...]] = {}
        for wall in Wall.all():
            resistance = math.exp(rng.uniform(math.log(1.5), math.log(60.0)))
            reactance = rng.uniform(0.0, 5.0)
            walls[wall] = tuple(complex(z) for z in rho_c * (resistance - 1j * reactance * 100.0 / frequencies))
        sets.append(walls)
    return sets


TRAINING = _materials(1, 12)
HOLDOUT = _materials(99, 2)


def _settings() -> ReducedModelSettings:
    return load_reduced_model_settings(config_path("fem_reduced.toml")).model_copy(
        update={"stop_residual": TEST_STOP_RESIDUAL})


@pytest.fixture(scope="module")
def operators() -> FullOperators:
    return export_room_operators(ROOM, SOURCES, SEATS, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED,
                                 physics_identity="phys-v1:" + "0" * 64,
                                 max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ)


@pytest.fixture(scope="module")
def model(operators: FullOperators) -> ReducedModel:
    return build_reduced_model(operators, AXIS, TRAINING, settings=_settings())


@pytest.fixture(scope="module")
def holdout_report(model: ReducedModel, operators: FullOperators) -> ValidationReport:
    return validate_reduced(model, operators, HOLDOUT)


def _truncated(model: ReducedModel, rank: int) -> ReducedModel:
    m = model.matrices
    return dataclasses.replace(model, matrices=ReducedMatrices(
        stiffness=m.stiffness[:rank, :rank], mass=m.mass[:rank, :rank], boundary=m.boundary[:, :rank, :rank],
        sources=m.sources[:, :rank], receivers=m.receivers[:, :rank]))


def test_settings_file_loads_and_guards(tmp_path: Path) -> None:
    """抓設定檔缺節、批量大於候選數、停止值不在 (0, 1) 時照樣收。"""
    settings = load_reduced_model_settings(config_path("fem_reduced.toml"))
    assert settings.batch <= settings.candidates_per_round
    missing = tmp_path / "missing.toml"
    missing.write_text("[other]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="build"):
        load_reduced_model_settings(missing)
    with pytest.raises(ValueError):
        ReducedModelSettings.model_validate({**settings.model_dump(), "batch": settings.candidates_per_round + 1})
    with pytest.raises(ValueError):
        ReducedModelSettings.model_validate({**settings.model_dump(), "stop_residual": 1.0})


def test_build_ledger_records_solves_rounds_and_rank(model: ReducedModel) -> None:
    """抓建模帳漏記或記錯：完整解次數＝起點＋每輪實際解的點數、停在殘差的那一輪實際解 0 點。"""
    ledger = model.ledger
    assert 0 < ledger.full_solves <= ledger.settings.max_full_solves
    assert ledger.rounds and ledger.rounds[-1].max_residual <= TEST_STOP_RESIDUAL
    assert ledger.stop_reason == "抽到的候選殘差都在停止值內"
    assert all(item.max_residual >= item.median_residual for item in ledger.rounds)
    assert ledger.rank == model.matrices.stiffness.shape[0] == model.matrices.receivers.shape[1]
    assert ledger.snapshots == ledger.full_solves * len(SOURCES)
    settings = ledger.settings
    initial = (len(even_indices(settings.initial_impedance_count, len(TRAINING)))
               * len(even_indices(settings.initial_frequency_count, len(AXIS))))
    assert ledger.full_solves == initial + sum(item.solved for item in ledger.rounds)
    assert all(item.solved <= item.proposed for item in ledger.rounds)
    assert not ledger.rounds[-1].solved and ledger.rounds[-1].proposed  # 停在殘差的那一輪準備了但沒有再解
    assert set(ledger.seconds) == {"snapshot", "svd", "projection", "greedy", "total"}
    assert model.source_names == tuple(SOURCES) and model.receiver_names == tuple(SEATS)


def test_training_materials_within_contract(model: ReducedModel, operators: FullOperators) -> None:
    """訓練材料（建模時看過的那一批的前兩組）逐頻完整解對替身，兩把尺都要過；係數慣例接反這裡就爆。"""
    report = validate_reduced(model, operators, TRAINING[:2])
    assert within_reduced_contract(report, frequency_db=FREQUENCY_RULER, band_db=BAND_RULER)


def test_holdout_materials_within_contract(holdout_report: ValidationReport) -> None:
    """留出材料（建模完全沒看過）逐頻完整解對替身：每個座位逐頻最大 dB 差與完整覆蓋帶能量差都在登記簿的尺內。"""
    assert set(holdout_report.frequency_db_by_seat) == set(SEATS)
    assert all(value <= FREQUENCY_RULER for value in holdout_report.frequency_db_by_seat.values())
    assert all(value <= BAND_RULER for value in holdout_report.band_db_by_seat.values())
    assert within_reduced_contract(holdout_report, frequency_db=FREQUENCY_RULER, band_db=BAND_RULER)
    assert any(band.partial for band in holdout_report.bands)
    assert any(not band.partial for band in holdout_report.bands)
    assert set(holdout_report.partial_band_db_by_seat) == set(SEATS)


def _report(frequency: Sequence[float], band: Sequence[float]) -> ValidationReport:
    empty = np.zeros((0,), dtype=np.complex128)
    return ValidationReport(
        frequency_db_by_seat=dict(zip(SEATS, frequency, strict=True)),
        band_db_by_seat=dict(zip(SEATS, band, strict=True)), partial_band_db_by_seat={}, bands=(),
        full_solves=0, full_pressure=empty, reduced_pressure=empty, model_digest="", source_points={},
        receiver_points={})


def _holds(report: ValidationReport) -> bool:
    return within_reduced_contract(report, frequency_db=FREQUENCY_RULER, band_db=BAND_RULER)


def test_frequency_ruler_mutant_beyond_tolerance_is_red(
        model: ReducedModel, operators: FullOperators) -> None:
    """逐頻那把尺：每個座位推到 (1−δ)·尺判綠，任一座位推到 (1+δ)·尺判紅；再把替身截到八分之一大小
    （故意弄差），留出材料逐頻差必須超過登記簿的尺。尺只從登記簿讀，判決換成別的值就紅。"""
    inside = [(1.0 - MUTANT_MARGIN) * FREQUENCY_RULER] * len(SEATS)
    band = [(1.0 - MUTANT_MARGIN) * BAND_RULER] * len(SEATS)
    assert _holds(_report(inside, band))
    for index in range(len(SEATS)):
        outside = [*inside[:index], (1.0 + MUTANT_MARGIN) * FREQUENCY_RULER, *inside[index + 1:]]
        assert not _holds(_report(outside, band))
    assert not _holds(_report([math.nan] * len(SEATS), band))
    degraded = validate_reduced(_truncated(model, model.ledger.rank // 8), operators, HOLDOUT)
    assert max(degraded.frequency_db_by_seat.values()) > FREQUENCY_RULER
    assert not _holds(degraded)


def test_band_ruler_mutant_beyond_tolerance_is_red() -> None:
    """帶能量那把尺：每個座位推到 (1−δ)·尺判綠，任一座位推到 (1+δ)·尺判紅。"""
    frequency = [(1.0 - MUTANT_MARGIN) * FREQUENCY_RULER] * len(SEATS)
    inside = [(1.0 - MUTANT_MARGIN) * BAND_RULER] * len(SEATS)
    assert _holds(_report(frequency, inside))
    for index in range(len(SEATS)):
        outside = [*inside[:index], (1.0 + MUTANT_MARGIN) * BAND_RULER, *inside[index + 1:]]
        assert not _holds(_report(frequency, outside))


def test_evaluate_reduced_matches_hand_example() -> None:
    """抓係數接錯牆、k² 乘錯矩陣、喇叭與座位軸對調：r=2 對角小例，逐項手算 c·f／(K − k²M + Σ c_w B_w)。"""
    boundary = np.zeros((6, 2, 2))
    boundary[2] = np.diag([0.5, 0.0])
    boundary[5] = np.diag([0.0, 2.0])
    matrices = ReducedMatrices(
        stiffness=np.diag([3.0, 7.0]), mass=np.diag([1.0, 2.0]), boundary=boundary,
        sources=np.array([[1.0 + 1.0j, 2.0], [0.0, 1.0j]]), receivers=np.array([[1.0, 0.0], [1.0, 3.0]]))
    walls = np.array([0, 0, 0.2j, 0, 0, 1.0 - 0.5j])
    k_squared = 0.25
    first = 3.0 - 0.25 * 1.0 + 0.2j * 0.5
    second = 7.0 - 0.25 * 2.0 + (1.0 - 0.5j) * 2.0
    expected = np.array([[(1 + 1j) / first, (1 + 1j) / first + 3 * 2.0 / second],
                         [0.0, 3 * 1j / second]])
    np.testing.assert_allclose(evaluate_reduced(matrices, k_squared, walls), expected, rtol=1e-14, atol=0)


def test_system_coefficients_follow_product_convention() -> None:
    """抓 k² 或 jωρ/Z 的符號、牆序、剛性牆沒歸零：逐面手算。"""
    walls: dict[Wall, complex | None] = {wall: 400.0 + 100.0j * index for index, wall in enumerate(Wall.all())}
    walls[Wall.Y0] = None
    k_squared, coefficients = system_coefficients(50.0, walls, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
    omega = 2.0 * math.pi * 50.0
    assert k_squared == (omega / SOUND_SPEED) ** 2
    for wall, value in zip(Wall.all(), coefficients, strict=True):
        impedance = walls[wall]
        assert value == (0j if impedance is None else 1j * omega * DENSITY / impedance)


def test_save_load_roundtrip_bitwise(model: ReducedModel, tmp_path: Path) -> None:
    """抓存讀掉位、換型別、漏出身欄位或建模帳：讀回逐位相同。"""
    path = tmp_path / "rom.npz"
    save_reduced(model, path)
    loaded = load_reduced(path, expected_mesh_hash=model.mesh_hash, expected_physics_identity=model.physics_identity)
    for name in ("stiffness", "mass", "boundary", "sources", "receivers"):
        before, after = getattr(model.matrices, name), getattr(loaded.matrices, name)
        assert before.dtype == after.dtype and before.shape == after.shape
        assert before.tobytes() == after.tobytes()
    assert dataclasses.replace(loaded, matrices=model.matrices) == model


@pytest.mark.parametrize("target", ("npz", "manifest"))
@pytest.mark.parametrize("position", (0.0, 0.31, 0.5, 0.77, 1.0))
def test_load_rejects_any_flipped_bit(model: ReducedModel, tmp_path: Path, target: str, position: float) -> None:
    """抓讀回不核雜湊：npz 或 manifest 在開頭、中段、結尾任一處翻一個位元都要拒收。"""
    path = tmp_path / "rom.npz"
    save_reduced(model, path)
    victim = path if target == "npz" else manifest_path(path)
    data = bytearray(victim.read_bytes())
    offset = min(int(position * len(data)), len(data) - 1)
    data[offset] ^= 0x04
    victim.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        load_reduced(path)


def test_load_rejects_other_mesh_or_identity_and_save_refuses_overwrite(model: ReducedModel, tmp_path: Path) -> None:
    """抓拿別間房的替身或舊物理身分的替身來用，以及存檔蓋掉舊檔。"""
    path = tmp_path / "rom.npz"
    save_reduced(model, path)
    with pytest.raises(ValueError, match="網格"):
        load_reduced(path, expected_mesh_hash="0" * 64)
    with pytest.raises(ValueError, match="物理身分"):
        load_reduced(path, expected_physics_identity="phys-v1:" + "1" * 64)
    with pytest.raises(FileExistsError):
        save_reduced(model, path)


def test_full_solve_residual_limit_is_enforced(operators: FullOperators) -> None:
    """抓完整解殘差沒核：上限設到比雙精度還小，第一次完整解就要報錯，不准默默收快照。"""
    settings = _settings().model_copy(update={"full_solve_max_relative_residual": 1.0e-300})
    with pytest.raises(RuntimeError, match="殘差"):
        build_reduced_model(operators, AXIS, TRAINING[:1], settings=settings)


def test_build_rejects_active_impedance(operators: FullOperators) -> None:
    """抓材料沒過產品阻抗閘門：實部為負（主動牆）要拒收。"""
    bad = {**TRAINING[0], Wall.X0: tuple(-1.0 + 0j for _ in AXIS)}
    with pytest.raises(ValueError, match="x0"):
        build_reduced_model(operators, AXIS, [bad], settings=_settings())


def test_export_from_scheme_matches_product_operators() -> None:
    """抓方案入口拿錯喇叭、座位、房間或空氣：右側逐位等於產品點源，取值列跟產品內插一致，身分照現行入口算。"""
    from aosr.config.capabilities import load_capabilities
    from aosr.reporting.fem_reduced_scheme import export_operators
    from aosr.reporting.physics_identity import physics_identity
    from tests.engine import _fem_shared_cases as cases
    from tests.engine import _scoring_source_model_control as control
    from tests.engine._directivity import DIRECTIVITY

    scheme = cases.small_scheme()
    table = load_capabilities(control.TARGETS.with_stem("capabilities"))
    exported = export_operators(scheme, capabilities=table, directivity=DIRECTIVITY)
    assert exported.physics_identity == physics_identity(capabilities=table, directivity=DIRECTIVITY)
    assert set(exported.sources) == set(scheme.speakers)
    assert set(exported.receivers) == {point.receiver_id for point in scheme.receiver_set.points}
    assert (exported.density_kg_m3, exported.sound_speed_m_s) == (
        scheme.scene.density_kg_m3, scheme.scene.sound_speed_m_s)
    for name, point in scheme.speakers.items():
        assert exported.sources[name].tobytes() == point_source_load(exported.p2, point).tobytes()
    field = np.random.default_rng(561).normal(size=exported.p2.basis.N).astype(np.complex128)
    for seat in scheme.receiver_set.points:
        expected = interpolate_pressure(exported.p2, Point(*seat.position_m), field)
        assert abs(exported.receivers[seat.receiver_id] @ field - expected) <= 1e-12 * abs(expected)
    assert len(exported.mesh_hash) == len(hashlib.sha256().hexdigest()) and int(exported.mesh_hash, 16) >= 0


def _bad_matrices(model: ReducedModel, kind: str) -> ReducedModel:
    m = model.matrices
    wrong: dict[str, NDArray[np.generic]] = {
        "float32": m.stiffness.astype(np.float32), "complex": m.stiffness + 1j, "not_square": m.stiffness[:, :1]}
    # 故意塞錯型別進去：cast 只是讓型別檢查器放行這個壞樣本，執行時陣列照原樣。
    return dataclasses.replace(model, matrices=dataclasses.replace(m, stiffness=cast(NDArray[np.float64], wrong[kind])))


@pytest.mark.parametrize("kind", ("float32", "complex", "not_square"))
def test_save_refuses_wrong_dtype_or_shape(model: ReducedModel, tmp_path: Path, kind: str) -> None:
    """抓存讀默默轉型：K 存成 float32、複數、或非方陣，存之前就拒收，一個檔都不留。"""
    path = tmp_path / "rom.npz"
    with pytest.raises(ValueError, match="stiffness"):
        save_reduced(_bad_matrices(model, kind), path)
    assert not path.exists() and not manifest_path(path).exists()


@pytest.mark.parametrize("kind", ("float32", "complex", "not_square"))
def test_load_refuses_wrong_dtype_or_shape(
        model: ReducedModel, tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓讀回只核雜湊不核型別：關掉存檔那一關硬存進去（三層雜湊全對），讀回仍要拒收，不准轉成 float64 或丟虛部。"""
    path = tmp_path / "rom.npz"
    with monkeypatch.context() as patch:
        patch.setattr(fem_reduced_store, "_check_arrays", lambda *args: None)
        save_reduced(_bad_matrices(model, kind), path)
    with pytest.raises(ValueError, match="stiffness"):
        load_reduced(path)


def test_save_with_existing_manifest_writes_nothing(model: ReducedModel, tmp_path: Path) -> None:
    """抓只有 manifest 已存在時先寫了 npz 才發現：拒絕之前一個檔都不新建。"""
    path = tmp_path / "rom.npz"
    manifest_path(path).write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_reduced(model, path)
    assert not path.exists()


def test_adjacent_bands_share_one_edge_and_edge_points_count_twice() -> None:
    """抓相鄰帶界各自算浮點、差幾個 ulp：帶 n 的上界就是帶 n+1 的下界，落在界上的頻點兩帶都算。"""
    bands = third_octave_bands(AXIS)
    assert all(left.upper_hz == right.lower_hz for left, right in zip(bands, bands[1:]))
    for edge in (left.upper_hz for left in bands[:-1]):
        axis = (AXIS[0], edge, AXIS[-1])
        containing = [band for band in third_octave_bands(axis) if 1 in band.indices]
        assert [band.upper_hz for band in containing][:1] == [edge]
        assert [band.lower_hz for band in containing][1:] == [edge]
