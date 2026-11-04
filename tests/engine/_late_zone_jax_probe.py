"""矩形晚期雙精度子行程，精度設定只經 runtime（執行環境）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aosr.runtime import configure_jax

configure_jax(platforms="cpu", enable_x64=True)

import jax
import jax.numpy as jnp
import numpy as np

from aosr.physics.late_energy_jax import (
    late_energy_arrays, late_energy_geometry, zoned_late_energy_arrays, zoned_late_energy_geometry,
)
from tests.engine._late_zone_control import control_inputs
from tests.engine.test_late_energy_zones import ROOM, synthetic
from aosr.geometry.shoebox import Wall


def encode(values: object) -> object:
    """只轉輸出陣列，不把雙精度結果先轉成單精度。"""
    return np.asarray(values).tolist()


def run() -> dict[str, object]:
    result: dict[str, object] = {}
    for wall in Wall.all():
        problem = synthetic(wall, 2, 1.0 / 3.0, 0.5)
        cells = problem.absorption_by_zone
        assert cells is not None
        geometry = zoned_late_energy_geometry(ROOM, 2, cells)
        alpha = jnp.asarray(list(problem.absorption_by_wall.values()) +
                            [cell.absorption for name in Wall.wall_names() for cell in cells.get(name, ())])
        eager = zoned_late_energy_arrays(geometry, alpha)
        compiled = jax.jit(zoned_late_energy_arrays)(geometry, alpha)
        gradient = jax.grad(lambda a: jnp.sum(zoned_late_energy_arrays(geometry, a).late_reverberant_energy))(alpha)
        result[wall.wall_name()] = {"eager": [encode(v) for v in eager], "jit": [encode(v) for v in compiled],
                                    "gradient": encode(gradient), "dtype": str(eager.late_reverberant_energy.dtype)}
    problem = control_inputs()
    control = late_energy_arrays(late_energy_geometry(problem.room, 6),
                                jnp.asarray(list(problem.absorption_by_wall.values())))
    result["control"] = {key: [float(v).hex() for v in values] for key, values in control._asdict().items()
                         if key != "dissipation_resolved"}
    from aosr.materials.construction_models import (
        AirLayer, Assembly, PorousLayer, MeasuredImpedance, MeasuredImpedancePoint, MountedOnWall, SurfaceMaterial,
    )
    from aosr.physics.wall_surfaces import wall_surface_response, zoned_wall_surface_response
    from tests.engine.test_late_energy_zones import CONSTANTS, FREQUENCIES, MATERIAL, zones
    base = Assembly(layers=(PorousLayer(thickness_m=0.1, flow_resistivity_pa_s_m2=10000.0,
                                       material_class="fibrous"),), backing="rigid")
    measured = MeasuredImpedance(points=(
        MeasuredImpedancePoint(frequency_hz=500.0, real_pa_s_m=2000.0, imag_pa_s_m=-100.0),
        MeasuredImpedancePoint(frequency_hz=1000.0, real_pa_s_m=1500.0, imag_pa_s_m=-50.0)),
        base_assembly=base, source_and_conditions="合成考卷，垂直入射")
    materials: tuple[SurfaceMaterial, ...] = (MATERIAL, base, measured, MountedOnWall(layers=(AirLayer(thickness_m=0.01),), wall=measured))
    partition = zones(Wall.YL)
    partition["yL"] = tuple(zone.model_copy(update={"material": material})
                            for zone, material in zip(partition["yL"], materials, strict=True))
    kwargs = dict(density_kg_m3=CONSTANTS.air_density_kg_m3, sound_speed_m_s=CONSTANTS.sound_speed_m_s)
    response = zoned_wall_surface_response(ROOM, partition, FREQUENCIES, **kwargs)
    same = []
    for cell, material in zip(response.absorption_by_zone["yL"], materials, strict=True):
        reference = wall_surface_response(dict.fromkeys(Wall.all(), material), FREQUENCIES, **kwargs)
        same.append(cell.absorption == reference.absorption_by_wall[Wall.YL] and cell.notes == reference.notes_by_wall[Wall.YL])
    result["material_rows_equal"] = same
    selected = response.named_zone_rows(tuple(reversed(FREQUENCIES)))
    result["selected_rows_equal"] = all(cell.absorption == tuple(reversed(original.absorption))
        for cell, original in zip(selected["yL"], response.absorption_by_zone["yL"], strict=True))
    return result


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(run()))
