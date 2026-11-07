"""#562 分區報表的有限元素接線；沿用 #1017 同網格獨立整面組裝的尺。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.physics import fem_batch, fem_helmholtz as fem, report_io
from aosr.physics.report_surfaces import mesh_partitions
from aosr.reporting import fem_slices, scheme_cli
from aosr.reporting.fem_slices import FemKey
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._precision_contracts import contract_value
from tests.engine._zoned_report_cases import small_scheme


def test_distinct_frequency_dependent_zones_match_direct_assembly(
    record_property: Callable[[str, object], None],
) -> None:
    """兩種材料各有不同的複數 Z(f)：抓第二區套首區，或逐頻取錯阻抗。"""
    scheme = small_scheme(zoned=True)
    solved = report_io.solver_inputs(report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY))
    mesh = generate_shoebox_mesh(solved.room, max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH, sound_speed_m_s=solved.sound_speed_m_s,
        random_seed=FEM_MESH_RANDOM_SEED, partitions=mesh_partitions(solved.room, scheme.scene.surface_zones_by_wall or {}))
    operators = fem.assemble_p2_operators(mesh)
    frequencies = (37.0, 73.0)
    # 直接給兩種材料的逐頻阻抗表；參考組不經 _frequency_system 的列與頻率索引。
    material_a = (600.0-320.0j, 1100.0+180.0j)
    material_b = (2100.0+460.0j, 800.0-240.0j)
    walls: dict[Wall, float] = {wall: 1646.4 for wall in Wall.all() if wall != Wall.X0}
    load = fem.point_source_load(operators, solved.source)
    reference = []
    for frequency, a, b in zip(frequencies, material_a, material_b, strict=True):
        system = fem.assemble_zoned_helmholtz_system(operators, frequency_hz=frequency,
            wall_impedances=walls, zone_impedances={Wall.X0: {"a": a, "b": b}},
            density_kg_m3=solved.density_kg_m3, sound_speed_m_s=solved.sound_speed_m_s)
        reference.append(fem.interpolate_pressure(operators, solved.receiver, spsolve(system, load)))
    actual = fem.solve_frequency_responses_many(operators, right_hand_sides={"source": load},
        receivers={"receiver": solved.receiver}, frequencies_hz=frequencies,
        wall_impedances=walls | {Wall.X0: None}, zone_impedances={Wall.X0: {"a": material_a, "b": material_b}},
        density_kg_m3=solved.density_kg_m3, sound_speed_m_s=solved.sound_speed_m_s)["source", "receiver"]
    difference = float(np.max(abs(actual-np.asarray(reference))/abs(np.asarray(reference))))
    assert difference <= contract_value("fem_vs_fenics_frozen")
    record_property("distinct_zones_pressure_max_relative_difference", difference)


def test_two_identical_zones_equal_independent_whole_fem(record_property: Callable[[str, object], None]) -> None:
    """真正多右側求解、逐區阻抗與分界網格；不靠粗細不同的網格另訂誤差。"""
    scheme = small_scheme(zoned=True)
    solved = report_io.solver_inputs(report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY))
    inputs = fem_slices.fem_inputs(scheme, solved, materials_by_wall=solved.materials_by_wall)
    mesh = generate_shoebox_mesh(solved.room, max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH, sound_speed_m_s=solved.sound_speed_m_s,
        random_seed=FEM_MESH_RANDOM_SEED, partitions=mesh_partitions(solved.room, scheme.scene.surface_zones_by_wall or {}))
    operators = fem.assemble_p2_operators(replace(mesh, boundary_zone_ids=None))
    source, receiver = inputs.sources["left"], inputs.receivers["main"]
    system = fem.assemble_helmholtz_system(operators, frequency_hz=37.0,
        wall_impedances=dict.fromkeys(Wall.all(), 1646.4), density_kg_m3=inputs.density_kg_m3,
        sound_speed_m_s=inputs.sound_speed_m_s)
    reference = fem.interpolate_pressure(operators, receiver, spsolve(system, fem.point_source_load(operators, source)))
    zone_rows = {wall: {name: (row[0],) if isinstance(row, tuple) else row for name, row in rows.items()}
                 for wall, rows in inputs.zone_impedances.items()}
    zoned_operators = fem.assemble_p2_operators(mesh)
    observed = fem.solve_frequency_responses_many(zoned_operators,
        right_hand_sides={"left": fem.point_source_load(zoned_operators, source)}, receivers={"main": receiver},
        wall_impedances=inputs.wall_impedances, zone_impedances=zone_rows, frequencies_hz=(37.0,),
        density_kg_m3=inputs.density_kg_m3, sound_speed_m_s=inputs.sound_speed_m_s)["left", "main"][0]
    difference = abs(observed-reference)/abs(reference)
    assert difference <= contract_value("fem_vs_fenics_frozen")
    record_property("fem_pressure_max_relative_difference", float(difference))
    actual = fem_batch.solve_fem_energies_many(room=inputs.room, candidates={"case": ({"left": source}, {"main": receiver})},
        wall_impedances=inputs.wall_impedances, zone_impedances=zone_rows,
        partitions=inputs.partitions, frequencies_hz=(37.0,), density_kg_m3=inputs.density_kg_m3,
        sound_speed_m_s=inputs.sound_speed_m_s)["case"]["left", "main"][0]
    assert actual == float(abs(observed)**2)


def test_fem_key_tracks_every_zone_material_and_boundary() -> None:
    """抓分片鑰匙漏第二區、幾何或代號，防止取錯矩陣的既存答案。"""
    scheme = small_scheme(zoned=True)
    solved = report_io.solver_inputs(report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY))
    inputs = fem_slices.fem_inputs(scheme, solved, materials_by_wall=solved.materials_by_wall)
    original = FemKey.from_inputs(inputs)
    with pytest.raises(ValueError, match="分界"):
        FemKey.from_inputs(replace(inputs, partitions={}))
    rows = {wall: dict(values) for wall, values in inputs.zone_impedances.items()}
    rows[Wall.X0]["b"] = 950.0
    assert original != FemKey.from_inputs(replace(inputs, zone_impedances=rows))


def test_zoned_fem_slice_cli_roundtrip_and_material_mismatch(tmp_path: Path) -> None:
    """命令列真算一片，完整回讀；改第二區的材料就不能沿用。"""
    scheme = small_scheme(zoned=True)
    source, out = tmp_path / "scheme.json", tmp_path / "part.json"
    source.write_text(scheme.model_dump_json())
    assert scheme_cli.main(["fem-slice", str(source), "--slice", "0", "--slices", "4", "--out", str(out),
        "--engine-commit", "control", "--capabilities", str(config_path("capabilities.toml"))]) == 0
    shard, = scheme_cli._read_fem_parts([out])
    assert all(np.isfinite(float.fromhex(v)) for c in shard.candidates.values() for p in c.energies for v in p.energy_hex)
    with pytest.raises(ValueError, match="鑰匙"):
        fem_slices.energies_from_shards([shard], scheme=small_scheme(zoned=True, changed=True), capabilities=CAPABILITIES,
            directivity=DIRECTIVITY, physics_identity=shard.physics_identity)
