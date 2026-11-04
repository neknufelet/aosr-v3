"""#561 第五步：面片排序、非均勻快照、舊檔、存讀與可微求值。

控制答案：2026-11-04 在 c6f5138a286de8a1c73cebd21a536de18759af53 的產品程式單緒實量，只留輸入雜湊；矩陣經有限元素
直接解（PARDISO），跨處理器不逐位，不錄（verification-split-stand-in-wiring-and-cross-cpu-tolerance 第 2 條；雲端 Xeon 6975P 實紅過）。
舊檔由 ``_zone_rom_case.write_v1`` 照起點 ``save_reduced`` 的寫法現產（第 1 版格式）。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.geometry.shoebox import Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, WallPartitions, generate_shoebox_mesh
from aosr.physics import fem_reduced as reduced
from aosr.runtime import child_process_env, configure_jax
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics.fem_reduced_check import validate_reduced, within_reduced_contract
from aosr.physics.fem_reduced_store import computation_digest, load_reduced, manifest_path, matrices_digest, save_reduced
from tests.engine._precision_contracts import contract_value
from tests.engine._zone_rom_case import AXIS, PARTITIONS, ROOM, SETTINGS, materials, operators, write_v1

_CPU_PIN_VARS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")
EXPECTED_PATCHES = ((Wall.FLOOR, ""), (Wall.CEILING, ""), (Wall.X0, "right"), (Wall.X0, "left"),
                    (Wall.XL, ""), (Wall.Y0, ""), (Wall.YL, ""))
CONTROL_INPUT_HASH = "4ac3251bd369c68d0dfbad3dd11a4089f4bbe249f170b9b1df89636ba1a3ff76"


def _zoned_operators(reverse: bool = False) -> reduced.FullOperators:
    whole = operators()
    partitions = {wall: tuple(reversed(rectangles)) if reverse else rectangles for wall, rectangles in PARTITIONS.items()}
    return reduced.export_room_operators(ROOM, whole.source_points, whole.receiver_points,
        density_kg_m3=whole.density_kg_m3, sound_speed_m_s=whole.sound_speed_m_s,
        physics_identity=whole.physics_identity, max_frequency_hz=80.0, partitions=partitions)


def _zoned_materials(seed: int, count: int) -> list[reduced.FrequencyPatchImpedances]:
    # 每區獨立抽樣；同面兩區刻意不同，不能用整面均勻快照冒充。
    rng = np.random.default_rng(seed)
    return [{patch: tuple(complex(z) for z in 411.6 * (rng.uniform(2, 12) - 1j * rng.uniform(0.5, 2) * 100 / np.asarray(AXIS)))
             for patch in EXPECTED_PATCHES} for _ in range(count)]


@pytest.fixture(scope="module")
def zoned_operators() -> reduced.FullOperators:
    return _zoned_operators()


@pytest.fixture(scope="module")
def zoned_model(zoned_operators: reduced.FullOperators) -> reduced.ReducedModel:
    return build_reduced_model(zoned_operators, AXIS, _zoned_materials(1, 8), settings=SETTINGS)


def _jax(model: reduced.ReducedModel, z: np.ndarray, directory: Path) -> tuple[np.ndarray, np.ndarray]:
    save_reduced(model, directory / "model.npz")
    np.save(directory / "z.npy", z, allow_pickle=False)
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    probe = Path(__file__).with_name("_zone_rom_jax_probe.py")
    process = subprocess.run([sys.executable, str(probe), str(directory)], cwd=directory,
                             env=env, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    return np.load(directory / "pressure.npy", allow_pickle=False), np.load(directory / "jit.npy", allow_pickle=False)


def _numpy_pressure(model: reduced.ReducedModel, values: reduced.FrequencyPatchImpedances) -> np.ndarray:
    rows = reduced.impedance_rows([values], AXIS, patch_keys=model.patch_keys)[0]
    m = model.matrices
    expected = []
    for index, frequency in enumerate(AXIS):
        k2, coefficients = reduced.system_coefficients(frequency, reduced.walls_at(rows, index),
            density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s, patch_keys=model.patch_keys)
        expected.append(reduced.reduced_pressure(m.stiffness, m.mass, m.boundary, m.sources, m.receivers, k2, coefficients))
    return np.asarray(expected)


def _assert_jax_matches_numpy(pressures: tuple[np.ndarray, ...], reference: np.ndarray) -> None:
    for pressure in pressures:
        assert np.all(np.abs(pressure - reference) <= contract_value("fem_reduced_jax_vs_numpy") * np.abs(reference))


def test_whole_model_input_pinned_v1_round_trip_and_holdout(tmp_path: Path) -> None:
    whole = operators()
    model = build_reduced_model(whole, AXIS, materials(1, 8), settings=SETTINGS)
    assert model.input_hash == CONTROL_INPUT_HASH
    # 矩陣不對起點逐位（見檔頭）；物理由留出組對完整解守，尺跟分區那題同一組登記值。
    report = validate_reduced(model, whole, materials(99, 2))
    assert within_reduced_contract(report, frequency_db=contract_value("fem_reduced_holdout_frequency_db"),
                                  band_db=contract_value("fem_reduced_holdout_band_db")), report
    write_v1(model, tmp_path / "v1.npz")
    manifest = json.loads((tmp_path / "v1.manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1 and "patches" not in manifest
    old = load_reduced(tmp_path / "v1.npz")
    assert (matrices_digest(old.matrices), computation_digest(old), old.input_hash) == (
        matrices_digest(model.matrices), computation_digest(model), model.input_hash)
    for name in ("stiffness", "mass", "boundary", "sources", "receivers"):
        assert getattr(old.matrices, name).tobytes() == getattr(model.matrices, name).tobytes()
    # JAX 求值走 XLA，跨處理器不錄逐位答案（verification-split-stand-in-wiring-and-cross-cpu-tolerance 第 2、6 條）：
    # 改成同一次執行裡跟 numpy 比，尺是登記簿的 fem_reduced_jax_vs_numpy。
    values = materials(99, 1)[0]
    z = np.asarray([values[wall] for wall in Wall.all()])
    _assert_jax_matches_numpy(_jax(model, z, tmp_path), _numpy_pressure(model, {(wall, ""): row for wall, row in values.items()}))
    assert old.patch_keys == reduced.boundary_patches()


def test_patch_order_follows_scheme_not_triangle_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    zoned_operators = _zoned_operators()
    assert zoned_operators.patch_keys == EXPECTED_PATCHES
    assert _zoned_operators().patch_keys == EXPECTED_PATCHES
    swapped = _zoned_operators(reverse=True)
    assert swapped.patch_keys == (*EXPECTED_PATCHES[:2], EXPECTED_PATCHES[3], EXPECTED_PATCHES[2], *EXPECTED_PATCHES[4:])

    def permuted(room: Room, *, max_frequency_hz: float, elements_per_wavelength: int,
                 sound_speed_m_s: float, random_seed: int, partitions: WallPartitions | None = None) -> ShoeboxMesh:
        mesh = generate_shoebox_mesh(room, max_frequency_hz=max_frequency_hz, elements_per_wavelength=elements_per_wavelength,
                        sound_speed_m_s=sound_speed_m_s, random_seed=random_seed, partitions=partitions)
        assert mesh.boundary_zone_ids is not None
        return replace(mesh, boundary_triangles=mesh.boundary_triangles[::-1].copy(),
                       boundary_wall_indices=mesh.boundary_wall_indices[::-1].copy(),
                       boundary_zone_ids=mesh.boundary_zone_ids[::-1].copy())

    monkeypatch.setattr(reduced, "generate_shoebox_mesh", permuted)
    shuffled = _zoned_operators()
    assert shuffled.patch_keys == EXPECTED_PATCHES
    assert tuple(shuffled.p2.zone_boundary_mass[Wall.X0]) != tuple(zoned_operators.p2.zone_boundary_mass[Wall.X0])
    # 用反向首現的 P2 映射，邊界列仍須按方案指定的順序取矩陣。
    matrices = zoned_operators.p2.zone_boundary_mass[Wall.X0]
    p2 = replace(zoned_operators.p2, zone_boundary_mass={Wall.X0: dict(reversed(tuple(matrices.items())))})
    reordered = replace(zoned_operators, p2=p2)
    assert reordered.patch_keys == EXPECTED_PATCHES
    for actual, (wall, zone) in zip(reordered.boundary, EXPECTED_PATCHES, strict=True):
        expected = p2.zone_boundary_mass[wall][zone] if zone else p2.boundary_mass[wall]
        assert actual is expected


def test_nonuniform_holdout_within_registered_contract(zoned_model: reduced.ReducedModel,
                                                      zoned_operators: reduced.FullOperators) -> None:
    report = validate_reduced(zoned_model, zoned_operators, _zoned_materials(99, 2))
    assert within_reduced_contract(report, frequency_db=contract_value("fem_reduced_holdout_frequency_db"),
                                  band_db=contract_value("fem_reduced_holdout_band_db")), report
    assert zoned_model.matrices.boundary.shape[0] == 7


def test_partitioned_greedy_measures_remaining_candidates(zoned_operators: reduced.FullOperators) -> None:
    # 少量起始快照留下候選；漏傳殘差路徑的 patch_keys（面片序）就會拒收這七片阻抗。
    settings = SETTINGS.model_copy(update={"initial_impedance_count": 2, "initial_frequency_count": 2,
                                          "candidates_per_round": 8, "batch": 2, "max_full_solves": 12})
    model = build_reduced_model(zoned_operators, AXIS, _zoned_materials(1, 8), settings=settings)
    assert model.patch_keys == EXPECTED_PATCHES
    assert any(round.candidates > 0 and np.isfinite(round.max_residual) for round in model.ledger.rounds)
    assert any(round.solved > 0 for round in model.ledger.rounds)


def _resign(path: Path, document: dict[str, object]) -> None:
    document.pop("manifest_sha256", None)
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    manifest_path(path).write_text(json.dumps(document), encoding="utf-8")


def test_partitioned_roundtrip_and_layout_binding(zoned_model: reduced.ReducedModel,
                                                 zoned_operators: reduced.FullOperators, tmp_path: Path) -> None:
    path = tmp_path / "zoned.npz"
    save_reduced(zoned_model, path)
    loaded = load_reduced(path, expected_mesh_hash=zoned_operators.mesh_hash,
                          expected_patch_keys=zoned_operators.patch_keys)
    assert loaded.patch_keys == EXPECTED_PATCHES
    assert computation_digest(loaded) == computation_digest(zoned_model)
    for name in ("stiffness", "mass", "boundary", "sources", "receivers"):
        assert getattr(loaded.matrices, name).tobytes() == getattr(zoned_model.matrices, name).tobytes()
    with pytest.raises(ValueError, match="網格雜湊"):
        load_reduced(path, expected_mesh_hash="wrong")
    with pytest.raises(ValueError, match="面片"):
        load_reduced(path, expected_patch_keys=reduced.boundary_patches())
    document = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    assert document["format_version"] == 2
    document["patches"][2:4] = list(reversed(document["patches"][2:4]))
    _resign(path, document)
    with pytest.raises(ValueError, match="面片"):
        load_reduced(path, expected_mesh_hash=zoned_operators.mesh_hash,
                     expected_patch_keys=zoned_operators.patch_keys)


def test_partitioned_jax_matches_numpy(zoned_model: reduced.ReducedModel, tmp_path: Path) -> None:
    values = _zoned_materials(99, 1)[0]
    z = np.asarray([values[key] for key in EXPECTED_PATCHES])
    pressures = _jax(zoned_model, z, tmp_path)
    templates = np.load(tmp_path / "templates.npy", allow_pickle=False)
    reference_z = np.load(tmp_path / "template_reference.npy", allow_pickle=False)
    assert templates.shape == (7, len(AXIS))
    assert np.all(templates[0] == 0)
    assert np.all(np.abs(templates[1:] - reference_z) <= contract_value("fem_reduced_jax_vs_numpy") * np.abs(reference_z))
    checks = json.loads((tmp_path / "templates.json").read_text(encoding="utf-8"))
    assert checks["mask"] == [True, False, False, False, False, False, False]
    assert checks["same_mask"] and len(checks["errors"]) == 3
    _assert_jax_matches_numpy(pressures, _numpy_pressure(zoned_model, values))


@pytest.mark.parametrize("entry", ["wall_coefficients", "reduced_pressures"])
def test_partitioned_concrete_rigid_mask_rejected(entry: str, zoned_model: reduced.ReducedModel) -> None:
    import jax.numpy as jnp
    from aosr.physics import fem_reduced_jax

    constants = fem_reduced_jax.reduced_constants(zoned_model)
    assert isinstance(constants.partition_mask, np.ndarray) and constants.partition_mask.dtype == np.bool_
    assert constants.partition_mask.tolist() == [False, False, True, True, False, False, False]
    values = jnp.asarray([_zoned_materials(99, 1)[0][key] for key in EXPECTED_PATCHES], dtype=jnp.complex64)
    evaluate = getattr(fem_reduced_jax, entry)
    for index in (2, 3):
        rigid = np.zeros(len(EXPECTED_PATCHES), dtype=bool)
        rigid[index] = True
        for mask in (rigid, jnp.asarray(rigid)):
            with pytest.raises(ValueError, match="分區面片一定要有材料，不能是剛性"):
                evaluate(constants, values, rigid_mask=mask)


@pytest.mark.parametrize("entry", ["wall_coefficients", "reduced_pressures"])
def test_partitioned_jit_ignores_zone_rigid_mask(entry: str, zoned_model: reduced.ReducedModel) -> None:
    import jax
    import jax.numpy as jnp
    from aosr.physics import fem_reduced_jax

    constants = fem_reduced_jax.reduced_constants(zoned_model)
    values = jnp.asarray([_zoned_materials(99, 1)[0][key] for key in EXPECTED_PATCHES], dtype=jnp.complex64)
    compiled = jax.jit(getattr(fem_reduced_jax, entry))
    # 保留整面 FLOOR 的合法剛性，只把兩個分區位元從 False 翻成 True。
    rigid = jnp.asarray([True, False, False, False, False, False, False])
    expected = np.asarray(compiled(constants, values, rigid_mask=rigid))
    actual = np.asarray(compiled(constants, values, rigid_mask=rigid.at[2:4].set(True)))
    assert np.isfinite(expected).all()
    assert actual.tobytes() == expected.tobytes()


def test_partitioned_rigid_and_missing_materials_rejected(zoned_operators: reduced.FullOperators) -> None:
    values = dict(_zoned_materials(99, 1)[0])
    values[(Wall.X0, "right")] = None
    with pytest.raises(ValueError, match="分區.*材料"):
        build_reduced_model(zoned_operators, AXIS, [values], settings=SETTINGS)
    values.pop((Wall.X0, "right"))
    with pytest.raises(ValueError, match="面片"):
        build_reduced_model(zoned_operators, AXIS, [values], settings=SETTINGS)


def test_gradient_refuses_partitioned_model(zoned_model: reduced.ReducedModel) -> None:
    from aosr.gradient.material_gradient import chain_objective
    from tests.engine.test_material_gradient import _objective_constants

    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    # 這裡應先拒收分區替身，不能走到樣板與評分。
    with pytest.raises(ValueError, match="搜尋還不支援分區替身（#561 第六步）"):
        chain_objective(zoned_model, (), _objective_constants(zoned_model), air=air, ranges=ranges, frequencies=AXIS)


def test_scheme_exports_partitioned_operators_without_relaxing_solver_inputs() -> None:
    from aosr.physics import report_io, report_surfaces
    from aosr.reporting.fem_reduced_scheme import export_operators
    from aosr.reporting.scheme import Scheme
    from tests.engine._directivity import DIRECTIVITY
    from tests.engine._furniture_cases import CAPABILITIES, pair
    from tests.engine._surface_zone_cases import rectangle_document

    scheme = Scheme.model_validate(rectangle_document())
    first = export_operators(scheme, capabilities=CAPABILITIES, directivity=DIRECTIVITY, max_frequency_hz=40)
    assert len(first.patch_keys) == 21
    assert first.patch_keys[:5] == ((Wall.FLOOR, ""), (Wall.CEILING, "r0_c0"), (Wall.CEILING, "r0_c1"),
                                   (Wall.CEILING, "r1_c0"), (Wall.CEILING, "r1_c1"))
    zones = dict(scheme.scene.surface_zones_by_wall or {})
    zones["ceiling"] = tuple(reversed(zones["ceiling"]))
    swapped = scheme.model_copy(update={"scene": scheme.scene.model_copy(update={"surface_zones_by_wall": zones})})
    second = export_operators(swapped, capabilities=CAPABILITIES, directivity=DIRECTIVITY, max_frequency_hz=40)
    assert second.patch_keys[:5] == (first.patch_keys[0], *reversed(first.patch_keys[1:5]))
    inputs = report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY)
    with pytest.raises(report_surfaces.RectangleZonesUnsupported):
        report_io.solver_inputs(inputs)


@pytest.mark.parametrize("entry", ["scheme", "axis", "ledger", "full", "endpoint"])
def test_gradient_entry_points_refuse_partitioned_model(entry: str, zoned_model: reduced.ReducedModel,
                                                       zoned_operators: reduced.FullOperators) -> None:
    from aosr.config.frequency_axis import LowFrequencyAxis
    from aosr.gradient import endpoint, material_gradient_ledger, run
    from aosr.reporting.scheme import Scheme
    from tests.engine._surface_zone_cases import zone_document
    from tests.engine.test_material_gradient import _gradient_settings, _objective_constants, _presets, _receivers, _score, _settings

    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    scheme = Scheme.model_validate(zone_document())
    with pytest.raises(ValueError, match="搜尋還不支援分區替身（#561 第六步）"):
        if entry == "scheme":
            run.check_surrogate_matches_scheme(zoned_model, scheme, run.scheme_points_hex(scheme))
        elif entry == "axis":
            run.check_surrogate_axis_covers_scoring(zoned_model, scheme, _score())
        elif entry == "ledger":
            material_gradient_ledger.header_for(zoned_model, (), _settings(),
                                                objective_constants=_objective_constants(zoned_model))
        elif entry == "full":
            endpoint.full_model_pressures(zoned_operators, {}, AXIS, air=air, ranges=ranges,
                                           max_relative_residual=SETTINGS.full_solve_max_relative_residual)
        else:
            endpoint.endpoint_check(zoned_operators, (), np.asarray([]), model=zoned_model,
                objective_settings=_gradient_settings(), receiver_set=_receivers(),
                low_frequency_axis=LowFrequencyAxis.SEARCH_48, presets=_presets(), settings=_settings(), air=air, ranges=ranges)


def test_gradient_gate_refuses_validated_partitioned_model(zoned_model: reduced.ReducedModel,
                                                          zoned_operators: reduced.FullOperators, tmp_path: Path) -> None:
    from aosr.gradient import run
    from aosr.physics.fem_reduced_check import points_hex
    from aosr.physics.fem_reduced_receipt import save_validation_receipt

    path = tmp_path / "model.npz"
    save_reduced(zoned_model, path)
    report = validate_reduced(zoned_model, zoned_operators, _zoned_materials(99, 2))
    save_validation_receipt(path, report, rulers=run.holdout_rulers(), seconds=1.0)
    with pytest.raises(ValueError, match="搜尋還不支援分區替身（#561 第六步）"):
        run.gated_surrogate(path, mesh_hash=zoned_model.mesh_hash, physics_identity=zoned_model.physics_identity,
            points=run.SchemePoints(points_hex(zoned_operators.source_points), points_hex(zoned_operators.receiver_points)))


def test_manifest_missing_patch_list_is_rejected(zoned_model: reduced.ReducedModel, tmp_path: Path) -> None:
    path = tmp_path / "model.npz"
    save_reduced(zoned_model, path)
    document = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    document.pop("patches")
    _resign(path, document)
    with pytest.raises(ValueError, match="面片"):
        load_reduced(path)


def test_manifest_reordered_patches_rejected_without_expected_keys(zoned_model: reduced.ReducedModel,
                                                                  tmp_path: Path) -> None:
    path = tmp_path / "model.npz"
    save_reduced(zoned_model, path)
    document = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    document["patches"][2:4] = list(reversed(document["patches"][2:4]))
    _resign(path, document)
    with pytest.raises(ValueError, match="面片清單與網格綁定不符"):
        load_reduced(path)


def test_input_hash_binds_patch_order_with_identical_zone_impedances(zoned_operators: reduced.FullOperators) -> None:
    values = dict(_zoned_materials(1, 1)[0])
    values[(Wall.X0, "left")] = values[(Wall.X0, "right")]
    swapped = (*EXPECTED_PATCHES[:2], EXPECTED_PATCHES[3], EXPECTED_PATCHES[2], *EXPECTED_PATCHES[4:])
    training = reduced.impedance_rows([values], AXIS, patch_keys=EXPECTED_PATCHES)
    assert training[0][EXPECTED_PATCHES[2]] == training[0][EXPECTED_PATCHES[3]]
    original = reduced.input_hash(zoned_operators, AXIS, training, SETTINGS)
    reordered = reduced.input_hash(replace(zoned_operators, patch_keys=swapped), AXIS, training, SETTINGS)
    assert reordered != original


def test_patch_order_is_bound_to_input_and_computation(zoned_model: reduced.ReducedModel,
                                                      zoned_operators: reduced.FullOperators) -> None:
    swapped = (*EXPECTED_PATCHES[:2], EXPECTED_PATCHES[3], EXPECTED_PATCHES[2], *EXPECTED_PATCHES[4:])
    assert computation_digest(replace(zoned_model, patch_keys=swapped)) != computation_digest(zoned_model)
    training = reduced.impedance_rows(_zoned_materials(1, 8), AXIS, patch_keys=swapped)
    assert reduced.input_hash(replace(zoned_operators, patch_keys=swapped), AXIS, training, SETTINGS) != zoned_model.input_hash
    with pytest.raises(ValueError, match="面片"):
        validate_reduced(replace(zoned_model, patch_keys=swapped), zoned_operators, _zoned_materials(99, 1))


def test_full_system_is_independent_of_zone_matrix_discovery(zoned_operators: reduced.FullOperators) -> None:
    matrices = zoned_operators.p2.zone_boundary_mass[Wall.X0]
    reversed_p2 = replace(zoned_operators.p2, zone_boundary_mass={Wall.X0: dict(reversed(tuple(matrices.items())))})
    reordered = replace(zoned_operators, p2=reversed_p2)
    values = reduced.walls_at(reduced.impedance_rows(_zoned_materials(99, 1), AXIS,
                                                    patch_keys=EXPECTED_PATCHES)[0], 0)
    systems = [reduced.FullSolver(op, max_relative_residual=SETTINGS.full_solve_max_relative_residual).system(AXIS[0], values)
               for op in (zoned_operators, reordered)]
    assert systems[0].data.tobytes() == systems[1].data.tobytes()


def test_input_hash_ignores_material_mapping_insertion_order(zoned_operators: reduced.FullOperators) -> None:
    training = reduced.impedance_rows(_zoned_materials(1, 8), AXIS, patch_keys=EXPECTED_PATCHES)
    reordered = tuple(dict(reversed(tuple(rows.items()))) for rows in training)
    assert reduced.input_hash(zoned_operators, AXIS, training, SETTINGS) == reduced.input_hash(
        zoned_operators, AXIS, reordered, SETTINGS)
