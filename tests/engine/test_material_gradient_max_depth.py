"""每面深度限制：只核盒界、規格、輸入與存檔，不建替身或求解。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, run
from aosr.gradient import material_gradient as search
from aosr.gradient import material_gradient_loop as loop
from aosr.gradient import fixed_surfaces as faces
from aosr.gradient import template_rounds as rounds
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import template_ledger as templates
from aosr.gradient.endpoint import design_digest, materials_from_surfaces
from aosr.gradient.endpoint_run import checked_result
from aosr.gradient.report import render_report
from aosr.materials.construction_models import Assembly, ConstantImpedance, PorousLayer, SurfaceMaterial
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_boundary import boundary_patches
from aosr.scoring.gradient_objective_jax import gradient_objective_constants
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from tests.engine._material_gradient_legacy import legacy_family_settings
from tests.engine.test_material_gradient import _gradient_settings, _receivers
from tests.engine.test_material_gradient_cli import _schemes
from tests.engine.test_material_gradient_mount import _project, _rigid_result


PRESET = "glass_wool_48k_50mm_gap50"


def _settings() -> MaterialGradientSettings:
    return load_material_gradient_settings(config_path("material_gradient.toml"))


def _kept() -> dict[Wall, SurfaceMaterial]:
    return {wall: ConstantImpedance(impedance_pa_s_m=12348.0) for wall in Wall.all()}


def _unequal_family_settings() -> MaterialGradientSettings:
    document = _settings().model_dump()
    document["family"][1]["max_total_depth_m"] = 0.15
    document["templates"].update(max_per_round=1, max_rounds=1)
    return MaterialGradientSettings.model_validate(document)


@pytest.mark.parametrize(("cap", "effective"), [(0.20, 0.15), (0.08, 0.08)])
def test_swapped_family_final_designs_and_export_use_minimum_depth(cap: float, effective: float) -> None:
    """抓終點重建誤用起跑拒收，或重建漏取家族與面上限較小值。"""
    settings, kept = _unequal_family_settings(), _kept()
    designs = run.designs_for((PRESET,) * 6, settings, kept=kept, max_depths={"x0": cap})
    context = rounds.TemplateContext(settings, run.presets_table(), {w.wall_name(): m for w, m in kept.items()},
                                     lambda designs: pytest.fail("不應求值"), max_depths={"x0": cap})
    swapped, theta = rounds.swap(designs, search.baseline_theta(designs), "x0", "glass_wool_24k", context)
    rebuilt = run.final_designs(rounds.surfaces_of(swapped), settings, kept, max_depths={"x0": cap})
    assert rebuilt == swapped
    assert rebuilt[2].max_depth_m == cap and rebuilt[2].max_total_depth_m == effective
    assert search.depth_limits(rebuilt)[2].max_total_m == effective
    material = materials_from_surfaces(rebuilt, theta, settings)[Wall.X0]
    assert isinstance(material, Assembly)
    assert isinstance(material.layers[0], PorousLayer) and material.layers[0].thickness_m == 0.05
    theta[4] = 0.20
    with pytest.raises(ValueError, match="深度|可行"):
        materials_from_surfaces(rebuilt, theta, settings)


def test_start_still_rejects_cap_above_initial_family_limit() -> None:
    settings = _unequal_family_settings()
    presets = list((PRESET,) * 6)
    presets[2] = settings.family_by_id("glass_wool_24k").template_preset
    with pytest.raises(ValueError, match="x0.*0.2.*glass_wool_24k.*0.15"):
        run.designs_for(presets, settings, kept=_kept(), max_depths={"x0": 0.20})


def _recorded_retune(theta: search.FloatVector, value: float) -> search.DescentResult:
    """手給帳本輸入，不求值、不跑搜尋；兩分項等權相加，代理為零。"""
    step = search.DescentStep(0, tuple(float(x) for x in theta), value, value,
                              (value / 2, value / 2, 0.0), 0.0, 0.0, 0.0)
    return search.DescentResult(theta, (step,), search.STOP_GRADIENT, 0.0, 1)


def _write_swapped_template_ledger(folder: Path, header: ledger.MaterialGradientHeader,
                                  designs: tuple[search.SurfaceDesign, ...], context: rounds.TemplateContext) -> None:
    """合成一輪接受 x0 換 24K 的合法帳本；讀寫與換家族都走產品函式。"""
    theta = search.baseline_theta(designs)
    start = rounds.surfaces_of(designs)
    th = run.template_header_for(header, context.settings, _kept(),
                                 base_ledger_sha256=run.file_sha256(folder / run.LEDGER_FILE))
    rows: list[templates.TemplateRow] = []
    for face, target in rounds.alternatives(start, context.settings):
        swapped, values = rounds.swap(designs, theta, face, target, context)
        value = 1.0 if (face, target) == ("x0", "glass_wool_24k") else 3.0
        score = rounds.QuickScore(face, target, rounds.surfaces_of(swapped), search.design_names(swapped), values,
                                   value, value, (value / 2, value / 2, 0.0), 0.0)
        rows.append(templates.quick_row(len(rows), 0, score))
    rows.extend(templates.segment_rows(_recorded_retune(theta, 2.0), "baseline_retune", len(rows), round=0,
                                       surfaces=start, variable_names=search.design_names(designs)))
    chosen = next(row for row in rows if row.candidate == "x0->glass_wool_24k")
    rows.extend(templates.segment_rows(_recorded_retune(np.asarray(chosen.theta()), 1.0), "candidate_retune",
                                       len(rows), round=0, surfaces=chosen.surfaces,
                                       variable_names=chosen.variable_names, candidate=chosen.candidate))
    decision = templates.TemplateDecision(
        baseline_value=2.0, quick_count=18,
        outcomes=(templates.TemplateOutcome(candidate="x0->glass_wool_24k", quick_value=1.0, value=1.0, improvement=1.0),),
        threshold=context.settings.candidates.improvement_threshold, accepted="x0->glass_wool_24k",
        round_evaluations=20, total_evaluations=20, next_round_bound=None)
    rows.append(templates.decision_row(len(rows), 0, surfaces=chosen.surfaces, variable_names=chosen.variable_names,
                                       theta_hex=chosen.theta_hex, value=1.0, screening_total=1.0,
                                       components=(0.5, 0.5, 0.0), seconds=0.0,
                                       decision=decision, stop_reason=rounds.TEMPLATE_STOP_MAX_ROUNDS))
    templates.create_template_ledger(folder / run.TEMPLATES_FILE, th)
    templates.append_template_rows(folder / run.TEMPLATES_FILE, th, rows, settings=context.settings,
                                    start=templates.TemplateStart(start, tuple(float(x).hex() for x in theta),
                                                                  2.0, 2.0, (1.0, 1.0, 0.0)))


def test_swapped_family_result_roundtrip_and_endpoint_check(tmp_path: Path) -> None:
    """抓帳本結果重建或終點核對仍用起跑的家族上限拒收。"""
    settings, kept = _unequal_family_settings(), _kept()
    caps = {"x0": 0.20}
    designs = run.designs_for((PRESET,) * 6, settings, kept=kept, max_depths=caps)
    context = rounds.TemplateContext(settings, run.presets_table(), {w.wall_name(): m for w, m in kept.items()},
                                     lambda designs: pytest.fail("不應求值"), max_depths=caps)
    header = ledger.MaterialGradientHeader(
        ledger_version=ledger.LEDGER_VERSION, input_hash="synthetic", mesh_hash="synthetic", physics_identity="synthetic",
        surfaces=rounds.surfaces_of(designs), variable_names=search.design_names(designs),
        settings_fingerprint=settings.fingerprint, objective_fingerprint="a" * 64, kept_digests={}, max_depths=caps,
        score_constants=ledger.ScoreConstants(component_weights=(1.0, 1.0), worse_references=(1.0, 1.0),
                                               proxy_weight=0.25, proxy_worse_reference=1.0, proxy_temperature_db=0.25))
    decision = ledger.RoundDecision(baseline_value=None, outcomes=(), threshold=settings.candidates.improvement_threshold,
                                     accepted=None, round_evaluations=1, total_evaluations=1, next_round_bound=None)
    row = ledger.decision_row(0, 0, tuple(float(x) for x in search.baseline_theta(designs)), 2.0, 2.0, (1.0, 1.0, 0.0),
                               0.0, decision, search.LOOP_STOP_NO_CANDIDATES)
    ledger.write_ledger(tmp_path / run.LEDGER_FILE, header, (row,))
    _write_swapped_template_ledger(tmp_path, header, designs, context)
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, _project().model_copy(update={"max_depths": caps}))
    result = run.result_from_ledger(tmp_path, settings, kept, expected_header=header)
    run.write_json_atomic(tmp_path / run.RESULT_FILE, result)
    restored = run.parse_result((tmp_path / run.RESULT_FILE).read_bytes())
    assert checked_result(tmp_path, restored, header, settings, kept) == result
    final = run.final_designs([(s.surface_id, s.preset_id) for s in result.surfaces], settings, kept, max_depths=caps)
    assert final[2].family_id == "glass_wool_24k" and final[2].max_total_depth_m == 0.15
    assert result.surfaces[2].material_digest == design_digest(final[2])
    material = materials_from_surfaces(final, np.asarray([float.fromhex(v) for v in result.theta_hex]), settings)[Wall.X0]
    assert isinstance(material, Assembly)
    assert isinstance(material.layers[0], PorousLayer) and material.layers[0].thickness_m == 0.05


@pytest.mark.parametrize("family", ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k"])
def test_new_family_continuous_box_and_legal_export(family: str) -> None:
    settings = _settings()
    preset = settings.family_by_id(family).template_preset
    designs = run.designs_for((preset,) * 6, settings, kept=_kept())
    design = designs[0]
    assert design.lower[0] == 0.05 and design.upper[0] == 0.20
    assert design.legal_thicknesses_m == (0.05, 0.10, 0.20)
    assert all(limit.max_total_m == 0.30 for limit in search.depth_limits(designs))
    theta = search.baseline_theta(designs)
    theta[0] = 0.18
    snapped = search.snap_to_specs(theta, designs)
    assert snapped.surfaces[0].thickness_m == 0.20
    material = materials_from_surfaces(designs, snapped.theta, settings)[Wall.FLOOR]
    assert isinstance(material, Assembly)
    assert isinstance(material.layers[0], PorousLayer) and material.layers[0].thickness_m == 0.20


def test_new_cavity_and_total_depth_edges() -> None:
    designs = run.designs_for((PRESET,) * 6, _settings(), kept=_kept())
    assert designs[0].lower == (0.05, 0.0) and designs[0].upper == (0.20, 0.25)
    theta = search.baseline_theta(designs)
    theta[:2] = (0.05, 0.25)
    assert search.is_feasible(theta, designs)
    theta[:2] = (0.10, 0.25)
    assert not search.is_feasible(theta, designs)
    theta[1] = np.nextafter(0.25, np.inf)
    with pytest.raises(ValueError, match="盒界"):
        search.checked_theta(theta, designs)
    theta[:2] = (np.nextafter(0.20, np.inf), 0.0)
    with pytest.raises(ValueError, match="盒界"):
        search.checked_theta(theta, designs)


@pytest.mark.parametrize(("values", "message"), [
    (["roof=0.1"], "roof"), (["x0=-0.1"], "正"), (["x0=0"], "正"),
    (["x0=nan"], "有限"), (["x0=inf"], "有限"), (["x0=no"], "公尺"),
    (["x0"], "面=公尺"), (["x0="], "面=公尺"),
    (["x0=0.1", "x0=0.2"], "兩次"),
    (["x0=0.1", "x0@u0v0=0.1"], "同一面"),
])
def test_parse_rejects_invalid_depths(values: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        faces.parse_max_depths(values)


def test_wall_depth_expands_and_patch_depth_matches_only_one() -> None:
    kept = {(Wall.FLOOR, ""): ConstantImpedance(impedance_pa_s_m=12348.0),
            (Wall.CEILING, ""): ConstantImpedance(impedance_pa_s_m=12348.0),
            (Wall.X0, "u0v0"): ConstantImpedance(impedance_pa_s_m=12348.0),
            (Wall.X0, "u1v0"): ConstantImpedance(impedance_pa_s_m=12348.0),
            (Wall.XL, ""): ConstantImpedance(impedance_pa_s_m=12348.0),
            (Wall.Y0, ""): ConstantImpedance(impedance_pa_s_m=12348.0),
            (Wall.YL, ""): ConstantImpedance(impedance_pa_s_m=12348.0)}
    for caps, expected in [({"x0": 0.08}, [0.30, 0.30, 0.08, 0.08, 0.30, 0.30, 0.30]),
                           ({"x0@u0v0": 0.08}, [0.30, 0.30, 0.08, 0.30, 0.30, 0.30, 0.30])]:
        designs = run.designs_for((PRESET,) * 7, _settings(), kept=kept, max_depths=caps)
        assert [limit.max_total_m for limit in search.depth_limits(designs)] == expected
    with pytest.raises(ValueError, match="清單"):
        run.designs_for((PRESET,) * 7, _settings(), kept=kept, max_depths={"x0@absent": 0.1})


@pytest.mark.parametrize(("caps", "fixed", "message"), [
    ({"x0": 0.31}, {}, "家族.*上限"), ({"x0": 0.049}, {}, "最薄"),
    ({"x0@u0v0": 0.1}, {}, "清單"),
    ({"x0": 0.1}, {"x0": "glass_wool_24k_50mm"}, "固定"),
])
def test_design_rejects_depth_outside_family_or_fixed(caps: dict[str, float], fixed: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run.designs_for((PRESET,) * 6, _settings(), kept=_kept(), max_depths=caps, fixed_surfaces=fixed)


def test_cap_controls_projection_snap_retune_export_and_template_swap() -> None:
    settings = _settings()
    designs = run.designs_for((PRESET,) * 6, settings, kept=_kept(), max_depths={"x0": 0.05})
    theta = search.baseline_theta(designs)
    projected = search.project_to_limits(theta, *search.design_bounds(designs), search.depth_limits(designs))
    assert tuple(projected[4:6]) == (0.05, 0.0)
    snapped = search.snap_to_specs(theta, designs)
    assert tuple(snapped.theta[4:6]) == (0.05, 0.0)
    assert search.retune_bounds(snapped.theta, designs)[1][5] == 0.0
    material = materials_from_surfaces(designs, snapped.theta, settings)[Wall.X0]
    assert isinstance(material, Assembly) and len(material.layers) == 1
    with pytest.raises(ValueError, match="深度|可行"):
        materials_from_surfaces(designs, theta, settings)
    context = rounds.TemplateContext(settings, run.presets_table(), {w.wall_name(): m for w, m in _kept().items()},
                                     lambda designs: pytest.fail("不應求值"), max_depths={"x0": 0.05})
    swapped, values = rounds.swap(designs, snapped.theta, "x0", "glass_wool_24k", context)
    assert swapped[2].max_total_depth_m == 0.05 and search.is_feasible(values, swapped)
    restored, values = rounds.swap(swapped, values, "x0", "glass_wool_48k", context)
    assert restored[2].max_total_depth_m == 0.05 and tuple(values[4:6]) == (0.05, 0.0)
    plain, values = rounds.swap(restored, values, "x0", "none", context)
    assert plain[2].max_depth_m == 0.05 and plain[2].max_total_depth_m is None
    restored, values = rounds.swap(plain, values, "x0", "glass_wool_48k", context)
    assert restored[2].max_total_depth_m == 0.05 and tuple(values[4:6]) == (0.05, 0.0)


def test_depth_identity_sorted_project_and_legacy_bytes(tmp_path: Path) -> None:
    project = _project()
    run.write_json_atomic(tmp_path / "old.json", project)
    assert hashlib.sha256((tmp_path / "old.json").read_bytes()).hexdigest() == "d51fbff4611f4915537fc6df82b5de4c1822dcb1b3c334c6245c9eb4f5979485"
    for caps in ({}, {"x0": 0.1, "floor": 0.2}, {"floor": 0.2, "x0": 0.1}):
        changed = run.GradientProject.model_validate({**project.model_dump(), "max_depths": caps})
        run.write_json_atomic(tmp_path / "project.json", changed)
        assert run.read_project(tmp_path) == changed
        if not caps:
            assert (tmp_path / "project.json").read_bytes() == (tmp_path / "old.json").read_bytes()
        else:
            assert list(json.loads((tmp_path / "project.json").read_text())["max_depths"]) == ["floor", "x0"]
    assert _identity() == _identity({}) == "198075c7e78d6f5eaa13d9a1f0ab5de4"
    assert _identity({"x0": 0.1, "floor": 0.2}) == _identity({"floor": 0.2, "x0": 0.1})
    assert len({_identity(), _identity({"x0": 0.1}), _identity({"x0": 0.2}), _identity({"xL": 0.1})}) == 4


def _identity(caps: dict[str, float] | None = None) -> str:
    return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=_project().presets,
                           purpose=_project().purpose, settings_fingerprint="c" * 64, physics_identity="physics-x",
                           objective_settings_fingerprint="d" * 64, max_depths=caps)


def test_cli_accepts_repeated_max_depth() -> None:
    args = cli._parser().parse_args(["start", "--scheme", "/absent", "--surrogate", "/absent", "--presets",
                                    ",".join([PRESET] * 6), "--purpose", "music", "--root", "/absent",
                                    "--max-depth", "x0=0.08", "--max-depth", "floor=0.2"])
    assert faces.parse_max_depths(args.max_depth) == {"floor": 0.2, "x0": 0.08}


def test_report_prints_effective_depth_for_every_face(tmp_path: Path) -> None:
    project = run.GradientProject.model_validate({**_project().model_dump(), "settings_path": str(config_path("material_gradient.toml")),
                                                 "settings_fingerprint": _settings().fingerprint,
                                                 "max_depths": {"x0": 0.08}, "fixed_surfaces": {"y0": "glass_wool_24k_50mm"}})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    text = render_report(tmp_path)
    assert "x0：總深度上限 0.08 m" in text
    assert "floor：總深度上限 0.3 m" in text
    assert "y0：固定面（不參與深度搜尋）" in text


def test_report_does_not_claim_limits_from_changed_settings(tmp_path: Path) -> None:
    project = run.GradientProject.model_validate({**_project().model_dump(), "settings_path": str(config_path("material_gradient.toml"))})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    text = render_report(tmp_path)
    assert "設定指紋不同" in text
    assert "總深度上限 0.3 m" not in text


def test_depth_caps_block_thicker_search_candidates() -> None:
    settings = _settings()
    caps = {wall.wall_name(): 0.05 for wall in Wall.all()}
    designs = run.designs_for((PRESET,) * 6, settings, kept=_kept(), max_depths=caps)
    theta = search.snap_to_specs(search.baseline_theta(designs), designs).theta
    assert loop.candidates_from_gradient(theta, -np.ones(theta.shape), designs, settings) == ()
    unlimited = run.designs_for((PRESET,) * 6, settings, kept=_kept())
    assert loop.candidates_from_gradient(theta, -np.ones(theta.shape), unlimited, settings)


@pytest.mark.parametrize(("fixed", "caps"), [
    ({"x0": "glass_wool_24k_50mm"}, {"x0@u0v0": 0.1}),
    ({"x0@u0v0": "glass_wool_24k_50mm"}, {"x0": 0.1}),
])
def test_project_rejects_overlapping_fixed_depths(fixed: dict[str, str], caps: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="固定"):
        run.GradientProject.model_validate({**_project().model_dump(), "fixed_surfaces": fixed, "max_depths": caps})


@pytest.mark.parametrize(("value", "fixed", "message"), [
    ("x0=0.31", None, "家族.*上限"), ("x0=0.049", None, "最薄"),
    ("x0@absent=0.1", None, "清單"), ("x0=0.1", "x0=glass_wool_24k_50mm", "固定"),
])
def test_cli_rejects_before_export_or_creating_folder(value: str, fixed: str | None, message: str,
                                                      tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    from aosr.reporting import fem_reduced_scheme

    scheme = tmp_path / "scheme.json"
    scheme.write_text(_schemes()["scheme.json"].model_dump_json())
    monkeypatch.setattr(fem_reduced_scheme, "export_operators", lambda *a, **kw: pytest.fail("拒收前不可匯出"))
    args = ["start", "--scheme", str(scheme), "--surrogate", "/absent", "--presets", ",".join([PRESET] * 6),
            "--purpose", _project().purpose, "--root", str(tmp_path / "out"), "--max-depth", value]
    if fixed:
        args += ["--fixed-surface", fixed]
    assert cli.main(args) == cli.EXIT_FAILED
    import re
    assert re.search(message, capsys.readouterr().err)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("frequency_batch", (None, 7))
def test_cli_saves_cap_and_binds_run_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            frequency_batch: int | None) -> None:
    """只隔離網格匯出、替身載入與起行程；驗參數、身分及 project 寫入走實際產品函式。"""
    from aosr.reporting import fem_reduced_scheme
    from aosr.physics.fem_reduced_store import manifest_path
    from aosr.physics.fem_reduced_receipt import receipt_path

    scheme = tmp_path / "scheme.json"
    scheme.write_text(_schemes()["scheme.json"].model_dump_json())
    surrogate = tmp_path / "absent.npz"
    manifest_path(surrogate).write_text(json.dumps({"manifest_sha256": "b" * 64}))
    receipt_path(surrogate).write_text("receipt")
    monkeypatch.setattr(fem_reduced_scheme, "export_operators", lambda *a, **kw: SimpleNamespace(
        physics_identity="physics-x", mesh_hash="mesh-x", source_points={}, receiver_points={}))
    monkeypatch.setattr(run, "gated_surrogate", lambda *a, **kw: object())
    monkeypatch.setattr(run, "check_surrogate_matches_scheme", lambda *a, **kw: None)
    monkeypatch.setattr(run, "check_surrogate_axis_covers_scoring", lambda *a, **kw: None)

    def save(root: Path, project: run.GradientProject) -> int:
        run.write_json_atomic(root / run.PROJECT_FILE, project)
        return 0

    monkeypatch.setattr(cli, "_create_and_run", save)
    root = tmp_path / "out"
    args = ["start", "--scheme", str(scheme), "--surrogate", str(surrogate), "--presets", ",".join([PRESET] * 6),
            "--purpose", _project().purpose, "--root", str(root), "--max-depth", "x0=0.08"]
    if frequency_batch is not None:
        args.extend(["--frequency-batch", str(frequency_batch)])
    assert cli.main(args) == 0
    project = run.read_project(root)
    assert project.max_depths == {"x0": 0.08}
    assert project.frequency_batch == frequency_batch
    expected = run.gradient_id(scheme_sha256=project.scheme_sha256, surrogate_manifest_sha256="b" * 64,
                               presets=project.presets, purpose=project.purpose, settings_fingerprint=_settings().fingerprint,
                               physics_identity="physics-x", objective_settings_fingerprint=cast(str, project.objective_settings_fingerprint),
                               max_depths={"x0": 0.08}, frequency_batch=frequency_batch)
    assert project.gradient_id == expected


def test_headers_bind_expanded_caps_and_preserve_legacy_bytes(tmp_path: Path) -> None:
    """標頭只讀模型中繼資料；這個測試物件沒有聲學矩陣或求值能力，不建立替身。"""
    model = cast(ReducedModel, SimpleNamespace(input_hash="input-x", mesh_hash="mesh-x", physics_identity="physics-x",
                                              patch_keys=boundary_patches(), matrices=SimpleNamespace(boundary=(None,) * 6)))
    constants = gradient_objective_constants(np.asarray(FEM_LANE_FREQUENCIES_HZ), _gradient_settings(), _receivers(), 2)
    settings = _settings()
    for caps in ({}, {"x0": 0.08}):
        designs = run.designs_for((PRESET,) * 6, settings, kept=_kept(), max_depths=caps)
        header = ledger.header_for(model, designs, settings, objective_constants=constants)
        assert header.max_depths == caps
        path = tmp_path / ("limited.jsonl" if caps else "plain.jsonl")
        ledger.write_ledger(path, header, ())
        assert ledger.read_ledger(path, expected_header=header)[0] == header
        template = run.template_header_for(header, settings, _kept(), base_ledger_sha256="e" * 64)
        assert template.max_depths == caps
        assert ("max_depths" in header.model_dump()) == bool(caps)
        assert ("max_depths" in template.model_dump()) == bool(caps)
    with pytest.raises(ValueError, match="標頭.*對不上"):
        ledger.read_ledger(tmp_path / "plain.jsonl", expected_header=header)
    # 歷史控制組只有序列化與設定指紋，沒有線性代數或 XLA 的固定答案。
    legacy = legacy_family_settings(settings)
    project = _project()
    kept = {wall: ConstantImpedance(impedance_pa_s_m=12348.0 + i) for i, wall in enumerate(Wall.all())}
    designs = run.designs_for(project.presets, legacy, kept=kept)
    header = ledger.header_for(model, designs, legacy, objective_constants=constants)
    ledger.write_ledger(tmp_path / "legacy.jsonl", header, ())
    assert hashlib.sha256((tmp_path / "legacy.jsonl").read_bytes()).hexdigest() == "a417256b0f40af5ab6703348c833bfb468d1e19593a3079061df943a849c2fe5"
    template = run.template_header_for(header, legacy, kept, base_ledger_sha256="e" * 64)
    templates.create_template_ledger(tmp_path / "legacy-templates.jsonl", template)
    assert hashlib.sha256((tmp_path / "legacy-templates.jsonl").read_bytes()).hexdigest() == "304b40fe88bb01f381ebce0962f3c0c3cf544151bb63f018769cf0bfe21ece3c"
    current_designs = run.designs_for(project.presets, settings, kept=kept)
    current = ledger.header_for(model, current_designs, settings, objective_constants=constants)
    assert current.model_dump() == {**header.model_dump(), "settings_fingerprint": settings.fingerprint}
    current_template = run.template_header_for(current, settings, kept, base_ledger_sha256="e" * 64)
    assert current_template.model_dump() == {**template.model_dump(), "settings_fingerprint": settings.full_fingerprint}
    ledger.write_ledger(tmp_path / "current.jsonl", current, ())
    templates.create_template_ledger(tmp_path / "current-templates.jsonl", current_template)
    assert hashlib.sha256((tmp_path / "current.jsonl").read_bytes()).hexdigest() == "3411307712a67d9977692db67fd607a56d825727c7048bc94b117db189891f21"
    assert hashlib.sha256((tmp_path / "current-templates.jsonl").read_bytes()).hexdigest() == "e97193100ab146a6f747045e022e21c6e89e9a3289d7c7dbf51f11dc8bbc3055"
    run.write_json_atomic(tmp_path / "result.json", _rigid_result(current, current_designs))
    assert hashlib.sha256((tmp_path / "result.json").read_bytes()).hexdigest() == "3ab492f75fd0110baca094025e12070f68264a8f6fa195b0a744473270121bb4"
