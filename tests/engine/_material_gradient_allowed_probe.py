"""每區准用清單＋固定材料進搜尋（#1082 段 2-3b）的雙精度子行程；只叫公開入口、寫帳本與原始數值，判決在父行程。

三段：
- ``segment``：合成目標（``_material_gradient_templates_probe.synthetic_factory``）跑整段換樣板，准用清單讓 x0 能整塊換成
  實測表、天花板能換內建地毯或不貼；偏移表讓第 0 輪接受 x0 換實測表、第 1 輪從那個狀態再評分（含 x0 換回家族）並接受天花板不貼。
  每一個列界各截一份接續；最後配置建回設計、照終點匯出材料。
- ``rows``：同一份實測表，搜尋鏈的常數列（``kept_impedance_rows``）與終點（``material_impedances``）在同一組頻點（含表外）逐位比。
- ``gate``：小房間真的建一份替身、真的完整解，准用清單裡的軟實測表與 ≥75ρc 的硬實測表各做一組留出驗證；
  另拿截斷到很小秩的替身再驗一次（一定過不了），看關卡拒收、訊息帶數字與尺。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import cast

import jax
import numpy as np
from pydantic import TypeAdapter

from aosr.config.construction_environment import construction_environment
from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import material_gradient as search
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.gradient import template_segment as segment
from aosr.gradient.endpoint import material_digest, material_impedances, materials_from_surfaces, surface_specs
from aosr.gradient.fixed_candidates import check_fixed_candidates
from aosr.gradient.run import presets_table
from aosr.materials import library
from aosr.materials import search_library as materials
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.physics.fem_reduced import ReducedMatrices, export_room_operators
from aosr.physics.fem_reduced_build import build_reduced_model
from tests.engine._material_gradient_templates_probe import (
    _cut_copy, kept_walls, settings_with, start_point, synthetic_factory, synthetic_header,
)

Record = dict[str, object]
_SURFACE: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)


def _hex(values: object) -> list[str]:
    return [float(value).hex() for value in np.asarray(values).ravel()]


def _table(problem: Record) -> tuple[materials.PresetMaterial, ...]:
    entries = tuple(library.FixedEntry.model_validate(item)
                    for item in cast(list[object], problem["user_fixed"]))
    targets = tuple(sorted({target for targets in cast(dict[str, list[str]], problem["allowed"]).values()
                            for target in targets if target.startswith("fixed:")}))
    return (*presets_table(), *materials.fixed_candidates(targets, presets_table(), entries))


def _segment(directory: Path, problem: Record) -> Record:
    base = load_material_gradient_settings(config_path("material_gradient.toml"))
    settings: MaterialGradientSettings = settings_with(base, {"templates": {"max_rounds": 3}})
    table = _table(problem)
    allowed = {face: tuple(targets) for face, targets in cast(dict[str, list[str]], problem["allowed"]).items()}
    kept = kept_walls(cast(list[float], problem["kept"]))
    counter = [0]
    context = rounds.TemplateContext(settings, table, kept,
                                     synthetic_factory(settings, cast(dict[str, float], problem["offsets"]), counter),
                                     allowed=allowed)
    surfaces = tuple((str(item[0]), item[1]) for item in cast(list[list[str | None]], problem["surfaces"]))
    start = start_point(context, surfaces)
    by_id = {preset.id: preset for preset in table}
    digests = {target: material_digest(_SURFACE.validate_python(by_id[target].material))
               for target in sorted({t for ts in allowed.values() for t in ts if t.startswith("fixed:")})}
    header = templates.TemplateHeader.model_validate({**synthetic_header(settings, surfaces, kept).model_dump(),
                                                      "allowed_targets": allowed, "fixed_candidate_digests": digests})
    path = directory / "segment.jsonl"
    result = segment.run_template_rounds(context, path, header, start)
    rows = templates.read_template_ledger(path, settings=settings, start=start)[1]
    resumed = []
    for keep in range(len(rows) + 1):
        cut = _cut_copy(path, directory / f"segment.cut{keep:04d}.jsonl", keep)
        segment.resume_template_rounds(context, cut, header, start)
        resumed.append(cut.name)
    designs = rounds.designs_for_surfaces(result.surfaces, context)
    exported = materials_from_surfaces(designs, result.theta, settings)
    specs = surface_specs(designs, result.theta)
    return {"ledger": path.name, "resumed": resumed, "result_surfaces": [list(item) for item in result.surfaces],
            "stop_reason": result.stop_reason, "rounds": result.rounds,
            "exported": {wall.wall_name(): None if material is None else material.model_dump(mode="json")
                         for wall, material in exported.items() if isinstance(wall, Wall)},
            "spec_presets": {spec.surface_id: spec.preset_id for spec in specs},
            "start": {"surfaces": [list(item) for item in start.surfaces], "theta_hex": list(start.theta_hex),
                      "value": start.value.hex(), "screening_total": start.screening_total.hex(),
                      "components": [value.hex() for value in start.components]}}


def _rows(problem: Record) -> Record:
    table = _table(problem)
    target = "fixed:" + str(problem["measured_id"])
    kept = kept_walls(cast(list[float], problem["kept"]))
    designs = search.designs_from_presets(
        tuple((wall.wall_name(), target if wall == Wall.X0 else None) for wall in Wall.all()),
        load_material_gradient_settings(config_path("material_gradient.toml")), table, kept=kept)
    axis = tuple(cast(list[float], problem["axis"]))
    air, ranges = construction_environment(density_kg_m3=1.2, sound_speed_m_s=343.0)
    search_row = search.kept_impedance_rows(designs, axis, air=air, ranges=ranges)[2]
    walls = {wall: design.kept_material for wall, design in zip(Wall.all(), designs, strict=True)}
    endpoint_row = cast(dict[object, object], material_impedances(walls, axis, air=air, ranges=ranges))[Wall.X0]
    assert search_row is not None and endpoint_row is not None
    return {"search": {"real": _hex(np.real(search_row)), "imag": _hex(np.imag(search_row))},
            "endpoint": {"real": _hex(np.real(np.asarray(endpoint_row))), "imag": _hex(np.imag(np.asarray(endpoint_row)))}}


def _gate(problem: Record) -> Record:
    from tests.engine.test_fem_reduced import AXIS, DENSITY, ROOM, SEATS, SOUND_SPEED, SOURCES, TRAINING, _settings
    from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ

    operators = export_room_operators(ROOM, SOURCES, SEATS, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED,
                                      physics_identity="phys-v1:" + "0" * 64, max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ)
    model = build_reduced_model(operators, AXIS, TRAINING, settings=_settings())
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    kept = {wall.wall_name(): ConstantImpedance(impedance_pa_s_m=10.0 * DENSITY * SOUND_SPEED) for wall in Wall.all()}
    designs = search.wall_designs_from_presets([None] * 6, settings, (), kept=kept)
    allowed = {face: tuple(targets) for face, targets in cast(dict[str, list[str]], problem["allowed"]).items()}
    table = _table(problem)
    found: Record = {}
    checks = check_fixed_candidates(model, operators, designs, allowed, table, density_kg_m3=DENSITY,
                                    sound_speed_m_s=SOUND_SPEED)
    found["checks"] = [check.model_dump(mode="json") for check in checks]
    shuffled = dataclasses.replace(operators, sources=dict(reversed(list(operators.sources.items()))),
                                   receivers=dict(reversed(list(operators.receivers.items()))))
    again = check_fixed_candidates(model, shuffled, designs, allowed, table, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
    found["shuffled"] = {"source_order_differs": tuple(shuffled.sources) != model.source_names,
                         "receiver_order_differs": tuple(shuffled.receivers) != model.receiver_names,
                         "same": [check.model_dump(mode="json", exclude={"seconds"}) for check in again]
                         == [check.model_dump(mode="json", exclude={"seconds"}) for check in checks]}
    rank = int(cast(int, problem["truncated_rank"]))
    m = model.matrices
    small = dataclasses.replace(model, matrices=ReducedMatrices(
        stiffness=m.stiffness[:rank, :rank], mass=m.mass[:rank, :rank], boundary=m.boundary[:, :rank, :rank],
        sources=m.sources[:, :rank], receivers=m.receivers[:, :rank]))
    try:
        check_fixed_candidates(small, operators, designs, allowed, table, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
        found["truncated_error"] = None
    except materials.MaterialsRefused as error:
        found["truncated_error"] = str(error)
    return found


def _builtins() -> Record:
    """每一筆內建構造預設當固定候選，走六類工廠的材料清單與無變數樣板（整條有限元素軸＋幾何軸），不求解；回收不了的。"""
    import jax.numpy as jnp

    from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ, GEOMETRIC_LANE_FREQUENCIES_HZ
    from aosr.gradient.six_category_chain import design_materials
    from aosr.materials.construction_diff import build_template

    presets = presets_table()
    targets = tuple("fixed:" + preset.id for preset in presets if not isinstance(preset, materials.LibraryPreset))
    table = {candidate.id: candidate for candidate in materials.fixed_candidates(targets, presets, ())}
    air, ranges = construction_environment(density_kg_m3=1.2, sound_speed_m_s=343.0)
    axis = jnp.asarray(np.asarray(sorted(set(FEM_LANE_FREQUENCIES_HZ) | set(GEOMETRIC_LANE_FREQUENCIES_HZ))))
    refused: dict[str, str] = {}
    for target in targets:
        try:
            (material,) = design_materials((search.fixed_candidate_design("x0", target, table, "rigid"),))
            if not isinstance(material, ConstantImpedance):
                build_template(material, axis, air=air, ranges=ranges, variables=())
        except ValueError as error:
            refused[target] = str(error)
    return {"targets": list(targets), "refused": refused}


def main() -> None:
    directory, part = Path(sys.argv[1]), sys.argv[2]
    problem = cast(Record, json.loads((directory / "problem.json").read_text(encoding="utf-8")))
    record: Record = {"x64": str(jax.numpy.asarray(0.0).dtype)}
    record.update({"segment": lambda: _segment(directory, problem), "rows": lambda: _rows(problem),
                   "gate": lambda: _gate(problem), "builtins": _builtins}[part]())
    (directory / f"output-{part}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
