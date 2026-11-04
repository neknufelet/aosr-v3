"""隔離雙精度子行程：分區鏈、短搜尋、樣板輪與完整模型終點。"""
from __future__ import annotations

from collections.abc import Callable
import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.material_gradient import MaterialGradientSettings
from aosr.materials.construction_models import SurfaceMaterial
from aosr.gradient.boundary import BoundaryKey
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ, LowFrequencyAxis
from aosr.config.construction_environment import construction_environment
from aosr.gradient import endpoint, material_gradient as gradient, material_gradient_ledger as ledger, run
from aosr.gradient.boundary import named_materials, partition_identity
from aosr.gradient.fixed_surfaces import expanded_fixed_surfaces
from aosr.gradient.template_rounds import TemplateContext, alternatives, surfaces_of
from aosr.gradient.template_ledger import read_template_ledger
from aosr.physics import fem_reduced as reduced
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.reporting.scheme import Scheme
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.scoring.gradient_objective_jax import GradientObjectiveConstants, gradient_objective_constants
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._material_gradient_zones_case import scheme
from tests.engine._zone_rom_case import SETTINGS
from tests.engine.test_material_gradient import DIFFERENCE_STEP, _gradient_settings, _settings, _numpy_score
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL


def _reference(model: reduced.ReducedModel, designs: tuple[gradient.SurfaceDesign, ...], theta: np.ndarray) -> np.ndarray:
    """獨立重建層與原牆，阻抗走非可微入口；縮減矩陣用 numpy 解。"""
    from aosr.materials.construction import surface_impedance
    from aosr.materials.construction_models import Assembly, MountedOnWall
    from aosr.gradient.boundary import patch_key

    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    rows = {}
    offset = 0
    for design in designs:
        material = design.kept_material
        if design.material is not None:
            layers = list(design.material.layers)
            for index, variable in enumerate(design.variables):
                layer = int(variable.split(".")[0].removeprefix("layer")) - 1
                layers[layer] = layers[layer].model_copy(update={"thickness_m": float(theta[offset + index])})
            offset += len(design.variables)
            material = (Assembly(layers=tuple(layers), backing="rigid") if design.backing_material is None else
                        MountedOnWall(layers=tuple(layers), wall=design.backing_material))
        assert material is not None
        z = surface_impedance(material, jnp.asarray(model.frequencies_hz), air=air, ranges=ranges)[0]
        rows[patch_key(design.surface_id)] = tuple(complex(v) for v in np.asarray(z))
    m = model.matrices
    result = []
    for index, frequency in enumerate(model.frequencies_hz):
        k2, coefficients = reduced.system_coefficients(frequency, {key: z[index] for key, z in rows.items()},
            density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s, patch_keys=model.patch_keys)
        result.append(reduced.reduced_pressure(m.stiffness, m.mass, m.boundary, m.sources, m.receivers, k2, coefficients))
    return np.asarray(result)


def _identities(model: reduced.ReducedModel, designs: tuple[gradient.SurfaceDesign, ...],
                settings: MaterialGradientSettings, constants: GradientObjectiveConstants,
                kept: dict[BoundaryKey, SurfaceMaterial],
                objective_for: Callable[[tuple[gradient.SurfaceDesign, ...]], gradient.ChainObjective]) -> None:
    header = ledger.header_for(model, designs, settings, objective_constants=constants)
    assert header.boundary_patches == partition_identity(model.patch_keys)
    assert header == ledger.header_for(model, designs, settings, objective_constants=constants)
    # 相同物理材料、只改代號或區序：搜尋身分與兩本標頭都不同。
    for keys in (tuple((w, "renamed" if z == "right" else z) for w, z in model.patch_keys),
                 (*model.patch_keys[:2], model.patch_keys[3], model.patch_keys[2], *model.patch_keys[4:])):
        other = dataclasses.replace(model, patch_keys=keys)
        from aosr.gradient.boundary import surface_ids
        changed = tuple(dataclasses.replace(d, surface_id=name) for d, name in zip(designs, surface_ids(keys), strict=True))
        changed_header = ledger.header_for(other, changed, settings, objective_constants=constants)
        assert header != changed_header
        assert run.template_header_for(header, settings, kept, base_ledger_sha256="e" * 64) != run.template_header_for(
            changed_header, settings, dict(zip(keys, kept.values(), strict=True)), base_ledger_sha256="e" * 64)
    for wrong in (designs[:-1], (*designs[:2], designs[3], designs[2], *designs[4:])):
        try:
            objective_for(tuple(wrong))
        except ValueError:
            pass
        else:
            raise AssertionError("沒有比對搜尋與替身的面片清單")


def _saved_result_report(directory: Path, base: Scheme, problem: run.GradientProblem) -> None:
    from aosr.config.paths import config_path
    from aosr.gradient.report import render_report
    from tests.engine.test_material_gradient_mount import _project

    scheme_path = directory / "scheme.json"
    scheme_path.write_text(base.model_dump_json(), encoding="utf-8")
    project = _project().model_copy(update={"scheme_path": str(scheme_path),
        "capabilities_path": str(config_path("capabilities.toml")),
        "presets": tuple(d.preset_id for d in problem.designs), "fixed_surfaces": problem.templates.fixed_surfaces})
    run.write_json_atomic(directory / run.PROJECT_FILE, project)
    result = run.write_result(directory, problem)
    assert len(result.surfaces) == 7 and run.parse_result((directory / run.RESULT_FILE).read_bytes()) == result
    text = render_report(directory)
    assert "x0@left：" in text and "x0@right：" in text
    for spec in result.surfaces:
        assert f"  {spec.surface_id}：" in text
        if spec.fixed_material_id is not None:
            assert f"  {spec.surface_id}：固定" in text
    (directory / "report.txt").write_text(text, encoding="utf-8")


def _search_and_endpoint(directory: Path, base: Scheme, operators: reduced.FullOperators, model: reduced.ReducedModel,
                         problem: run.GradientProblem) -> tuple[int, int, int]:
    designs, settings, header, kept = problem.designs, problem.settings, problem.header, problem.kept
    fixed = problem.templates.fixed_surfaces or {}
    fixed_patches = fixed
    axis = model.frequencies_hz
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    ledger.write_ledger(directory / run.LEDGER_FILE, header, ())
    run._fresh(problem, directory / run.LEDGER_FILE, None)
    run._template_segment(problem, directory, None)
    _saved_result_report(directory, base, problem)
    _, rows = ledger.read_ledger(directory / run.LEDGER_FILE)
    def reference(point: np.ndarray) -> float:
        return _numpy_score(model, _reference(model, designs, point))
    for row in rows:
        specs = run.surface_specs(header, row.theta())
        assert tuple(s.surface_id for s in specs) == tuple(d.surface_id for d in designs)
        expected = reference(np.asarray(row.theta()))
        np.testing.assert_allclose(row.value, expected, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    _, template_rows = read_template_ledger(directory / run.TEMPLATES_FILE)
    assert all(row.candidate is None or row.candidate.split("->")[0] not in fixed_patches for row in template_rows)
    final = template_rows[-1]
    final_designs = run.final_designs(final.surfaces, settings, kept, fixed_surfaces=fixed)
    theta = np.asarray(final.theta())
    result = endpoint.endpoint_check(operators, final_designs, theta, model=model,
        objective_settings=_gradient_settings(), receiver_set=base.receiver_set, low_frequency_axis=LowFrequencyAxis.SEARCH,
        presets=run.presets_table(), settings=settings, air=air, ranges=ranges)
    assert result.report.reduced_objective_total is not None
    np.testing.assert_allclose(result.report.reduced_objective_total, _numpy_score(model, result.reduced_pressure),
                               rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    materials = endpoint.materials_from_surfaces(final_designs, theta, settings)
    scene = endpoint.scene_with_materials(base.scene, materials)
    from aosr.reporting.scheme import Scheme
    exported = Scheme.model_validate({**base.model_dump(), "scene": scene.model_dump()})
    zones = exported.scene.surface_zones_by_wall
    original = base.scene.surface_zones_by_wall
    assert zones is not None and original is not None
    for before, after in zip(original["x0"], zones["x0"], strict=True):
        assert before.area == after.area
        assert before.model_dump(exclude={"material"}) == after.model_dump(exclude={"material"})
    assert tuple(z.material for z in zones["x0"]) == (materials[model.patch_keys[2]], materials[model.patch_keys[3]])
    full = endpoint.full_model_pressures(operators, materials, axis, air=air, ranges=ranges,
                                        max_relative_residual=SETTINGS.full_solve_max_relative_residual)
    np.testing.assert_allclose(full, result.full_pressure, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    return len(rows), len(template_rows), result.report.full_solves


def main(directory: Path) -> None:
    base = scheme()
    operators = export_operators(base, capabilities=CAPABILITIES, directivity=DIRECTIVITY, max_frequency_hz=80)
    axis = FEM_LANE_FREQUENCIES_HZ
    rng = np.random.default_rng(561)
    training = [{key: tuple(complex(z) for z in 411.6 * (rng.uniform(2, 12) - 1j * rng.uniform(0.5, 2) * 100 / np.asarray(axis)))
                 for key in operators.patch_keys} for _ in range(8)]
    model = build_reduced_model(operators, axis, training, settings=SETTINGS)
    settings = _settings()
    settings = settings.model_copy(update={
        "descent": settings.descent.model_copy(update={"max_steps": 1}),
        "retune": settings.retune.model_copy(update={"max_steps": 1}),
        "candidates": settings.candidates.model_copy(update={"max_rounds": 1, "max_per_round": 1}),
        "templates": settings.templates.model_copy(update={"max_rounds": 1, "max_per_round": 1}),
        "family": settings.family[:1]})
    kept = run.scene_wall_materials(base, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    presets = (None, None, "glass_wool_48k_50mm_gap50", None, None, None, None)
    fixed = {name: "carpet_on_concrete" for name in ("floor", "ceiling", "xL", "y0", "yL")}
    designs = run.designs_for(presets, settings, kept=kept, fixed_surfaces=fixed)
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    constants = gradient_objective_constants(axis, _gradient_settings(), base.receiver_set, 2)
    def objective_for(items: tuple[gradient.SurfaceDesign, ...]) -> gradient.ChainObjective:
        return gradient.chain_objective(model, items, constants, air=air, ranges=ranges, frequencies=axis)
    objective = objective_for(designs)
    theta = gradient.baseline_theta(designs)
    theta[:] = [0.071, 0.063]
    (value, _), automatic = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.asarray(theta))
    def reference(point: np.ndarray) -> float:
        pressure = _reference(model, designs, point)
        return _numpy_score(model, pressure)
    center = reference(theta)
    np.testing.assert_allclose(float(value), center, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    for index, step in enumerate(DIFFERENCE_STEP * theta):
        upper, lower = theta.copy(), theta.copy()
        upper[index] += step
        lower[index] -= step
        central = (reference(upper) - reference(lower)) / (2 * step) * theta[index]
        np.testing.assert_allclose(float(automatic[index]) * theta[index], central, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    header = ledger.header_for(model, designs, settings, objective_constants=constants)
    _identities(model, designs, settings, constants, kept, objective_for)
    fixed_patches = expanded_fixed_surfaces(fixed, tuple(d.surface_id for d in designs))
    choices = alternatives(surfaces_of(designs), settings, fixed_patches)
    assert choices and all(name in ("x0@right", "x0@left") for name, _ in choices)
    context = TemplateContext(settings, run.presets_table(), named_materials(kept), objective_for, "rigid", fixed_patches)
    problem = run.GradientProblem(designs, settings, objective, header, kept, context)
    row_count, template_count, full_solves = _search_and_endpoint(directory, base, operators, model, problem)
    (directory / "probe.json").write_text(json.dumps({"rows": row_count, "template_rows": template_count,
        "patches": len(model.patch_keys), "gradient": np.asarray(automatic).tolist(), "full_solves": full_solves}), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
