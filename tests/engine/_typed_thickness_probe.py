"""網頁真算題的對照小重算（雙精度子行程）：同一份求解輸入，只把地板第 1 層厚度換成指定值再算一次（#972 修補二）。

用法：python _typed_thickness_probe.py <seams|t60> <求解輸入 JSON 檔> <厚度>...
seams：實測表底構造在表兩端點的阻抗與無規吸音率（跟報表接縫同一支 assembly_endpoint_responses）。
t60：六面無規吸音（跟報表同一支 wall_surface_response、同一條聯集軸）接逐帶艾林 T60。
不解有限元素、不算幾何路，只算這兩個量；每個厚度輸出一筆，順序同引數。
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import cast

from aosr.runtime import configure_calculation_jax
configure_calculation_jax()

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.frequency_axis import LATE_DECAY_FREQUENCIES_HZ, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ
from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import MeasuredImpedance
from aosr.physics.crossover import eyring_t60_by_band
from aosr.physics.report_io import ReportInput, load_input_document, solver_inputs
from aosr.physics.wall_surfaces import assembly_endpoint_responses, wall_surface_response


def with_floor_thickness(document: dict[str, object], thickness: float) -> ReportInput:
    data = deepcopy(document)
    zones = cast(dict[str, list[dict[str, dict[str, object]]]], data["surface_zones_by_wall"])
    material = zones["floor"][0]["material"]
    assembly = cast(dict[str, object], material["base_assembly"]) if material["kind"] == "measured_impedance" else material
    cast(list[dict[str, object]], assembly["layers"])[0]["thickness_m"] = thickness
    # 跟正式計算同一道驗證（能力表與指向性預設都讀產品設定）。
    return load_input_document(data, load_capabilities(config_path("capabilities.toml")),
                               load_directivity_defaults(config_path("directivity_defaults.toml")))


def seams(inputs: ReportInput) -> list[list[float]]:
    solver = solver_inputs(inputs)
    material = solver.materials_by_wall[Wall.from_name("floor")]
    assert isinstance(material, MeasuredImpedance)
    ends = (material.points[0].frequency_hz, material.points[-1].frequency_hz)
    normal, diffuse, _ = assembly_endpoint_responses(material.base_assembly, ends,
        density_kg_m3=solver.density_kg_m3, sound_speed_m_s=solver.sound_speed_m_s)
    return [[z.real, z.imag, alpha] for z, alpha in zip(normal, diffuse, strict=True)]


def t60(inputs: ReportInput) -> dict[str, float]:
    solver = solver_inputs(inputs)
    # 家具會改晚期吸音；這支只給沒有家具的小房間用，有家具就停、不默默少算一項。
    assert not solver.furniture
    _, report_frequencies = low_frequency_axis_frequencies(solver.low_frequency_axis)
    axis = tuple(sorted(set(report_frequencies) | set(LATE_DECAY_FREQUENCIES_HZ) | set(SCHROEDER_T60_BANDS_HZ)))
    surfaces = wall_surface_response(solver.materials_by_wall, axis,
        density_kg_m3=solver.density_kg_m3, sound_speed_m_s=solver.sound_speed_m_s)
    return {str(f): value for f, value in eyring_t60_by_band(solver.room, surfaces.band_rows(SCHROEDER_T60_BANDS_HZ)).items()}


if __name__ == "__main__":
    mode, path, *thicknesses = sys.argv[1:]
    document = json.loads(Path(path).read_text())
    inputs = [with_floor_thickness(document, float(value)) for value in thicknesses]
    json.dump([seams(item) for item in inputs] if mode == "seams" else [t60(item) for item in inputs], sys.stdout)
