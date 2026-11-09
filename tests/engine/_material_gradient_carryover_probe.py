"""沿用開始與起點阻抗的雙精度子行程；只隔離建搜尋與算子來源，不跑搜尋。"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, run
from aosr.gradient.endpoint import material_impedances
from aosr.materials.construction_models import AirLayer, Assembly, ConstantImpedance
from aosr.physics.fem_reduced import export_room_operators, impedance_rows
from aosr.physics.fem_reduced_store import load_reduced
from aosr.reporting import fem_reduced_scheme
from tests.engine.test_fem_reduced import SOURCES, SEATS, DENSITY, SOUND_SPEED
from tests.engine._six_search_cli_case import ROOM
from tests.engine.test_material_gradient import PURPOSE
from tests.engine.test_material_gradient_cli import PRESET_TEXT, _chain_operators, _schemes


def start(directory: Path, carryover: bool) -> int:
    surrogate = directory / "rom.npz"
    model = load_reduced(surrogate)
    full = export_room_operators(ROOM, SOURCES, SEATS, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED,
                                physics_identity=model.physics_identity, max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ)
    full = _chain_operators(full, model)
    current = dataclasses.replace(full, physics_identity="phys-v1:" + "1" * 64) if carryover else full

    def capture(root: Path, project: run.GradientProject) -> int:
        run.write_json_atomic(root / run.PROJECT_FILE, project)
        run.check_inputs(project)
        return 0

    with (patch.object(fem_reduced_scheme, "export_operators", return_value=current),
          patch.object(run, "now_text", return_value="2026-11-08T00:00:00+00:00"),
          patch.object(cli, "_create_and_run", side_effect=capture)):
        return cli.main(["start", "--scheme", str(directory / "scheme.json"), "--surrogate", str(surrogate),
                         "--presets", PRESET_TEXT, "--purpose", PURPOSE, "--root", str(directory),
                         "--engine-commit", "test"])


def projected_start(directory: Path) -> int:
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    scheme = _schemes()["scheme.json"]
    kept = {wall: ConstantImpedance(impedance_pa_s_m=12348.0) for wall in Wall.all()}
    designs = run.designs_for(("glass_wool_48k_50mm_gap50",) * 6, settings, kept=kept, max_depths={"x0": 0.08})
    axis = (20.0, 40.0)
    rows = cli._starting_impedances(designs, axis, scheme)
    material = next(design.material for design in designs if design.surface_id == "x0")
    assert material is not None
    # 手算投影：棉厚下界50mm，總深度80mm，故空腔必須30mm；不走下降或匯出建立參考。
    reference = Assembly(layers=(material.layers[0], AirLayer(thickness_m=0.03)), backing="rigid")
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    expected = material_impedances({wall: reference for wall in Wall.all()}, axis, air=air, ranges=ranges)
    actual_array = np.asarray(impedance_rows([rows], axis)[0][(Wall.X0, "")], dtype=np.complex128)
    reference_array = np.asarray(impedance_rows([expected], axis)[0][(Wall.X0, "")], dtype=np.complex128)
    document = {"actual_real": actual_array.real.tolist(), "actual_imag": actual_array.imag.tolist(),
                "reference_real": reference_array.real.tolist(), "reference_imag": reference_array.imag.tolist()}
    (directory / "projected.json").write_text(json.dumps(document))
    return 0


if __name__ == "__main__":
    mode, location = sys.argv[1:]
    raise SystemExit(projected_start(Path(location)) if mode == "projected" else start(Path(location), mode == "carryover"))
