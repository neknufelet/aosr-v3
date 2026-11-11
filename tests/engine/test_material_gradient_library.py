"""第三階段搜尋吃材料庫（#1082 段 2-3a）考卷。

- 沒有使用者可最佳化材料：設定、構造清單、指紋、資料夾名、``project.json`` 一個位元都不動（資料夾名釘主線 b4358482 算的值）。
- 使用者可最佳化材料一筆一個家族接在設定檔家族後面：起跑、換樣板、結果重建、終點核身分都走同一份構造清單；
  不給那份清單（只拿產品構造預設）就重建不出來——考的就是每一個讀的人都接上了資料夾裡的快照。
- 快照：開跑寫進資料夾、進資料夾名；之後改網頁的材料庫不影響這一場；快照被改過就拒收。
- 開跑前拒收：固定材料當家族、掛構造預設（不存在的與存在的各講各的）、材料庫讀不進來、物理拒算的流阻（雙精度子行程）。
- 網頁：選使用者家族、固定材料拒收、讀不進來拒收；有使用者材料才帶 ``--materials``（工作資料夾裡一份複本）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from pydantic import TypeAdapter

from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.product_air import product_air_environment
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, run
from aosr.gradient import material_gradient as search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.gradient.endpoint import checked_design_identity
from aosr.gradient.endpoint_run import checked_result
from aosr.gui.gradient_jobs import GradientJobs
from aosr.gui.gradient_plan import GradientRequest, PlanRefused
from aosr.materials import library
from aosr.materials import search_library as materials
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.runtime import child_process_env, configure_jax
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._six_search_helpers import expand_components, weights
from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _data
from tests.engine.test_material_gradient_cli import _schemes
from tests.engine.test_material_gradient_mount import _project

USER = "rock_wool_60k"
FIXED = "panel_fixed"
PROBE = Path(__file__).with_name("_material_gradient_library_probe.py")
_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
# 主線 b4358482 的 gradient_id 用同一組純字串輸入算的值（main-src 解出來直接叫，工作區 1099-search-material-library/baseline/）。
MAIN_PLAIN_ID = "4b9652510d8cf350c2060cb58b174764"
MAIN_FULL_ID = "02f4ae17e5f1a0b2c79c262a4ac7a9a2"
_PURE_PRESETS = ("glass_wool_48k_50mm_gap50", None, "glass_wool_24k_50mm", None, None, "glass_wool_96k_50mm")


def _pure_id(*, full: bool = False, materials_sha256: str | None = None) -> str:
    """同一組純字串輸入的資料夾名；``full`` 加上安裝方式、固定面、深度上限、分批與裝置。"""
    if not full:
        return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=_PURE_PRESETS,
                               purpose="music", settings_fingerprint="c" * 64, physics_identity="phys",
                               objective_settings_fingerprint="d" * 64, materials_sha256=materials_sha256)
    return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=_PURE_PRESETS,
                           purpose="music", settings_fingerprint="c" * 64, physics_identity="phys",
                           objective_settings_fingerprint="d" * 64, mount="on_wall", fixed_surfaces={"floor": "scheme-material"},
                           max_depths={"ceiling": 0.2}, frequency_batch=16, device="gpu", materials_sha256=materials_sha256)


def _optimizable(**changes: object) -> library.OptimizableEntry:
    document: dict[str, object] = {
        "kind": "optimizable", "id": USER, "label": "60K 岩棉（自填）", "provenance": "model", "source": "使用者自填流阻",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 20000.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.075, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.04}, "max_total_depth_m": 0.15}
    document.update(changes)
    return cast(library.OptimizableEntry, _ENTRY.validate_python(document))


def _fixed() -> library.FixedEntry:
    return cast(library.FixedEntry, _ENTRY.validate_python({
        "kind": "fixed", "id": FIXED, "label": "固定吸音板（自填）", "provenance": "model", "source": "自填",
        "material": {"kind": "constant_impedance", "impedance_pa_s_m": 5000.0}}))


def _base() -> MaterialGradientSettings:
    return load_material_gradient_settings(config_path("material_gradient.toml"))


def _kept() -> dict[Wall, SurfaceMaterial]:
    return {wall: ConstantImpedance(impedance_pa_s_m=12348.0) for wall in Wall.all()}


# ---- 沒有使用者可最佳化材料：逐位不動 ----

def test_without_user_optimizable_materials_nothing_moves() -> None:
    """抓沒有使用者家族時也重建設定（指紋、家族順序可能被動到）或改了資料夾名的算法。"""
    base, table = _base(), run.presets_table()
    for entries in ((), (_fixed(),)):
        chosen = materials.with_user_materials(base, table, entries)
        assert chosen.settings is base and chosen.presets == table and chosen.entries == ()
    assert (_pure_id(), _pure_id(full=True)) == (MAIN_PLAIN_ID, MAIN_FULL_ID)
    assert _pure_id(materials_sha256="e" * 64) not in (MAIN_PLAIN_ID, _pure_id(materials_sha256="f" * 64))
    dumped = json.loads(_project().model_dump_json())
    assert "materials_path" not in dumped and "materials_sha256" not in dumped


def test_project_records_snapshot_path_and_hash_together() -> None:
    with pytest.raises(ValueError, match="材料快照的路徑與 SHA-256 要一起記"):
        run.GradientProject.model_validate({**_project().model_dump(), "materials_path": "materials.json"})


# ---- 使用者可最佳化材料 → 家族 ----

def test_user_material_becomes_one_appended_family_with_its_own_template() -> None:
    """抓家族換掉設定檔的、厚度空腔上限對錯欄、或樣板不是初值排法。"""
    base, entry = _base(), _optimizable()
    chosen = materials.with_user_materials(base, run.presets_table(), (_fixed(), entry))
    assert chosen.settings.family[:-1] == base.family and chosen.entries == (entry,)
    family = chosen.settings.family[-1]
    assert family.model_dump() == {"id": USER, "presets": (USER,), "thicknesses_m": (0.05, 0.075, 0.1),
                                   "cavity_range_m": (0.0, 0.1), "max_total_depth_m": 0.15, "template_preset": USER,
                                   "initial_thickness_m": 0.05, "initial_cavity_m": 0.04}
    assert chosen.presets[:-1] == run.presets_table() and chosen.presets[-1].id == USER
    assert chosen.presets[-1].material == library.template_assembly(entry).model_dump(mode="json")
    assert chosen.settings.fingerprint != base.fingerprint
    assert (chosen.settings.descent, chosen.settings.templates) == (base.descent, base.templates)


def test_user_family_defaults_cap_fixed_thickness_and_untuned_cavity() -> None:
    """總深沒寫取層排得出的最深；厚度不能調只有初值一個；空腔初值空白就沒有空腔變數、照實寫一行。"""
    entry = _optimizable(max_total_depth_m=None, cavity={"adjustable": True, "range_m": [0.0, 0.2], "initial_m": None},
                         porous_thickness={"adjustable": False, "range_m": [0.05, 0.05], "initial_m": 0.05})
    chosen = materials.with_user_materials(_base(), run.presets_table(), (entry,))
    family = chosen.settings.family[-1]
    assert family.max_total_depth_m == 0.25 and family.thicknesses_m == (0.05,) and family.initial_cavity_m is None
    design, = search.designs_from_presets((("x0", USER),), chosen.settings, chosen.presets)
    assert design.cavity_variable is None and design.lower == design.upper == (0.05,)
    note, = materials.cavity_notes(chosen)
    assert USER in note and "不調它的空腔" in note
    zero = materials.with_user_materials(_base(), run.presets_table(), (_optimizable(cavity={
        "adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.0}),))
    assert zero.settings.family[-1].initial_cavity_m is None and len(materials.cavity_notes(zero)) == 1


def test_user_family_starts_swaps_and_needs_the_folder_table() -> None:
    """抓換樣板沒把使用者家族列進候選、換過去用錯構造，或讀的人只拿產品構造預設。"""
    chosen = materials.with_user_materials(_base(), run.presets_table(), (_optimizable(),))
    kept = _kept()
    designs = run.designs_for((USER,) * 6, chosen.settings, kept=kept, table=chosen.presets)
    assert all(design.family_id == USER for design in designs)
    with pytest.raises(ValueError, match=f"preset id 不認識：{USER}"):
        run.designs_for((USER,) * 6, chosen.settings, kept=kept)
    start = run.designs_for(("glass_wool_48k_50mm_gap50",) * 6, chosen.settings, kept=kept, table=chosen.presets)
    assert (start[0].surface_id, USER) in rounds.alternatives(rounds.surfaces_of(start), chosen.settings)
    context = rounds.TemplateContext(chosen.settings, chosen.presets, {w.wall_name(): m for w, m in kept.items()},
                                     lambda designs: pytest.fail("不應求值"))
    swapped, theta = rounds.swap(start, search.baseline_theta(start), start[0].surface_id, USER, context)
    assert swapped[0].preset_id == USER and swapped[0].material == library.template_assembly(_optimizable())
    assert tuple(float(x) for x in theta[:2]) == (0.05, 0.04)


# ---- 開跑前拒收 ----

@pytest.mark.parametrize(("changes", "message"), [
    ({"preset_members": ["nope_50mm"]}, rf"{USER}.*preset_members.*不存在.*nope_50mm"),
    ({"preset_members": ["glass_wool_48k_50mm"], "template_preset": "glass_wool_48k_50mm"},
     rf"{USER}.*preset_members.*有填.*不能掛構造預設"),
])
def test_user_material_cannot_hang_on_construction_presets(changes: dict[str, object], message: str) -> None:
    with pytest.raises(materials.MaterialsRefused, match=message):
        materials.with_user_materials(_base(), run.presets_table(), (_optimizable(**changes),))


def test_fixed_material_is_never_a_family_and_unreadable_library_is_refused(tmp_path: Path) -> None:
    with pytest.raises(materials.MaterialsRefused, match=rf"{FIXED}.*固定材料.*整塊選或不選、不能微調"):
        materials.refuse_fixed_as_family((None, FIXED), (_fixed(),))
    materials.refuse_fixed_as_family((None, USER), (_fixed(), _optimizable()))
    broken = tmp_path / "materials.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(materials.MaterialsRefused, match="使用者材料庫 .* 讀不進來"):
        materials.read_user_entries(broken)
    assert materials.read_user_entries(tmp_path / "absent.json") == ()


def _probe(tmp_path: Path, entries: list[library.LibraryEntry]) -> dict[str, object]:
    env = forward_cpu_pins(child_process_env(threads=1))
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    env["TMPDIR"] = str(tmp_path)
    output = tmp_path / "probe.json"
    request = {"entries": [entry.model_dump(mode="json") for entry in entries], "frequencies_hz": [31.5, 63.0, 125.0]}
    done = subprocess.run([sys.executable, str(PROBE), str(output)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          input=json.dumps(request), timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))


def test_physics_refusal_names_the_material_before_the_search_starts(tmp_path: Path) -> None:
    """抓流阻出了小松模型手動試算範圍（3000、80000）存得進材料庫、卻在搜尋半路才壞；好的那筆照常過。"""
    porous = _optimizable().porous.model_dump(mode="json")
    assert _probe(tmp_path, [_optimizable()]) == {"ok": True}
    for flow in (3000.0, 80000.0):
        bad = _optimizable(porous={**porous, "flow_resistivity_pa_s_m2": flow})
        result = _probe(tmp_path, [_fixed(), bad])
        assert result.get("type") == "MaterialsRefused", result
        refused = str(result["refused"])
        assert f"（{USER}）的多孔層（porous）參數物理拒算" in refused and f"{flow:g}" in refused and "流阻" in refused


# ---- 命令列開跑：快照、身分、之後的讀者 ----

def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str, presets: str) -> tuple[int, list[object]]:
    """只隔離網格匯出、替身載入與起子行程；建資料夾（暫存名 → 正式名）走產品的 ``_create_and_run``。"""
    from aosr.physics.fem_reduced_receipt import receipt_path
    from aosr.physics.fem_reduced_store import manifest_path
    from aosr.reporting import fem_reduced_scheme

    scheme = tmp_path / "scheme.json"
    scheme.write_text(_schemes()["scheme.json"].model_dump_json())
    surrogate = tmp_path / "absent.npz"
    manifest_path(surrogate).write_text(json.dumps({"manifest_sha256": "b" * 64}))
    receipt_path(surrogate).write_text("receipt")
    checked: list[object] = []
    monkeypatch.setattr(fem_reduced_scheme, "export_operators", lambda *a, **kw: SimpleNamespace(
        physics_identity="physics-x", mesh_hash="mesh-x", source_points={}, receiver_points={}))
    monkeypatch.setattr(run, "gated_surrogate", lambda *a, **kw: SimpleNamespace(frequencies_hz=(31.5, 63.0)))
    monkeypatch.setattr(run, "check_surrogate_matches_scheme", lambda *a, **kw: None)
    monkeypatch.setattr(run, "check_surrogate_axis_covers_scoring", lambda *a, **kw: None)
    monkeypatch.setattr(run, "check_user_families", lambda chosen, frequencies, **kw: checked.append((chosen, frequencies)))
    monkeypatch.setattr(cli, "_spawn_and_wait", lambda *a, **kw: 0)
    args = ["start", "--scheme", str(scheme), "--surrogate", str(surrogate), "--presets", presets,
            "--purpose", _project().purpose, "--root", str(tmp_path / "out"), *extra]
    return cli.main(args), checked


def test_start_snapshots_user_families_into_the_folder_and_its_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓快照沒寫進資料夾、沒進資料夾名、接續讀網頁現在的材料庫，或快照被改了還照讀。"""
    user = tmp_path / "library.json"
    user.write_text(library.dump_user_library((_fixed(), _optimizable())), encoding="utf-8")
    code, checked = _start(tmp_path, monkeypatch, "--materials", str(user), presets=",".join([USER] * 6))
    assert code == 0
    folder, = (tmp_path / "out").iterdir()
    project = run.read_project(folder)
    snapshot = folder / materials.SNAPSHOT_FILE
    assert project.materials_path == str(snapshot) and snapshot.read_bytes() == materials.snapshot_bytes((_optimizable(),))
    assert project.materials_sha256 == run.file_sha256(snapshot)
    assert project.gradient_id == folder.name == run.gradient_id(
        scheme_sha256=project.scheme_sha256, surrogate_manifest_sha256="b" * 64, presets=project.presets,
        purpose=project.purpose, settings_fingerprint=project.settings_fingerprint, physics_identity="physics-x",
        objective_settings_fingerprint=cast(str, project.objective_settings_fingerprint), materials_sha256=project.materials_sha256)
    chosen, frequencies = cast(tuple[materials.SearchMaterials, tuple[float, ...]], checked[0])
    assert chosen.entries == (_optimizable(),) and frequencies == (31.5, 63.0)
    assert run.project_settings(project).fingerprint == project.settings_fingerprint
    assert run.project_presets(project)[-1].id == USER
    porous = _optimizable().porous.model_dump(mode="json")
    user.write_text(library.dump_user_library((_optimizable(porous={**porous, "flow_resistivity_pa_s_m2": 30000.0}),)),
                    encoding="utf-8")
    assert run.project_settings(project).fingerprint == project.settings_fingerprint
    assert run.project_presets(project)[-1].material == library.template_assembly(_optimizable()).model_dump(mode="json")
    snapshot.write_bytes(snapshot.read_bytes().replace(b"20000.0", b"20001.0"))
    with pytest.raises(ValueError, match="材料快照 .* 跟開跑時不同"):
        run.project_settings(project)


def test_start_without_user_families_writes_no_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = tmp_path / "library.json"
    user.write_text(library.dump_user_library((_fixed(),)), encoding="utf-8")
    preset = "glass_wool_48k_50mm_gap50"
    code, checked = _start(tmp_path, monkeypatch, "--materials", str(user), presets=",".join([preset] * 6))
    assert code == 0 and checked == []
    folder, = (tmp_path / "out").iterdir()
    assert sorted(path.name for path in folder.iterdir()) == [run.PROJECT_FILE]
    document = json.loads((folder / run.PROJECT_FILE).read_text(encoding="utf-8"))
    assert "materials_path" not in document and "materials_sha256" not in document
    assert document["settings_fingerprint"] == _base().fingerprint


def test_start_refuses_a_fixed_material_as_start_before_any_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    user = tmp_path / "library.json"
    user.write_text(library.dump_user_library((_fixed(), _optimizable())), encoding="utf-8")
    code, _ = _start(tmp_path, monkeypatch, "--materials", str(user), presets=",".join([FIXED] + [USER] * 5))
    assert code == cli.EXIT_FAILED
    assert "整塊選或不選" in capsys.readouterr().err and not (tmp_path / "out").exists()


def _ledgers(folder: Path, settings: MaterialGradientSettings, designs: tuple[search.SurfaceDesign, ...],
             context: rounds.TemplateContext) -> ledger.MaterialGradientHeader:
    """合成庚二收尾一列、換樣板一輪接受 x0 換 24K（起點每面都是使用者家族）；讀寫與換家族都走產品函式。"""
    header = ledger.MaterialGradientHeader(
        ledger_version=ledger.LEDGER_VERSION, input_hash="synthetic", mesh_hash="synthetic", physics_identity="synthetic",
        surfaces=rounds.surfaces_of(designs), variable_names=search.design_names(designs),
        settings_fingerprint=settings.fingerprint, objective_fingerprint="a" * 64, kept_digests={},
        score_constants=ledger.ScoreConstants(component_weights=(1.0, 1.0), worse_references=(1.0, 1.0), proxy_weight=0.25,
                                              proxy_worse_reference=1.0, proxy_temperature_db=0.25, category_weights=weights()))
    theta = search.baseline_theta(designs)
    decision = ledger.RoundDecision(baseline_value=None, outcomes=(), threshold=settings.candidates.improvement_threshold,
                                    accepted=None, round_evaluations=1, total_evaluations=1, next_round_bound=None)
    ledger.write_ledger(folder / run.LEDGER_FILE, header, (ledger.decision_row(
        0, 0, tuple(float(x) for x in theta), 2.0, 2.0, expand_components(2.0, 2.0, (1.0, 1.0, 0.0)), 0.0, decision,
        search.LOOP_STOP_NO_CANDIDATES),))
    start = rounds.surfaces_of(designs)
    th = run.template_header_for(header, settings, _kept(), base_ledger_sha256=run.file_sha256(folder / run.LEDGER_FILE))

    def retune(values: search.FloatVector, value: float) -> search.DescentResult:
        step = search.DescentStep(0, tuple(float(x) for x in values), value, value,
                                  expand_components(value, value, (value / 2, value / 2, 0.0)), 0.0, 0.0, 0.0)
        return search.DescentResult(values, (step,), search.STOP_GRADIENT, 0.0, 1)

    rows: list[templates.TemplateRow] = []
    for face, target in rounds.alternatives(start, settings):
        swapped, values = rounds.swap(designs, theta, face, target, context)
        value = 1.0 if (face, target) == ("x0", "glass_wool_24k") else 3.0
        rows.append(templates.quick_row(len(rows), 0, rounds.QuickScore(
            face, target, rounds.surfaces_of(swapped), search.design_names(swapped), values, value, value,
            expand_components(value, value, (value / 2, value / 2, 0.0)), 0.0)))
    quick = len(rows)
    rows.extend(templates.segment_rows(retune(theta, 2.0), "baseline_retune", len(rows), round=0, surfaces=start,
                                       variable_names=search.design_names(designs)))
    chosen = next(row for row in rows if row.candidate == "x0->glass_wool_24k")
    rows.extend(templates.segment_rows(retune(np.asarray(chosen.theta()), 1.0), "candidate_retune", len(rows), round=0,
                                       surfaces=chosen.surfaces, variable_names=chosen.variable_names,
                                       candidate=chosen.candidate))
    choice = templates.TemplateDecision(
        baseline_value=2.0, quick_count=quick,
        outcomes=(templates.TemplateOutcome(candidate="x0->glass_wool_24k", quick_value=1.0, value=1.0, improvement=1.0),),
        threshold=settings.candidates.improvement_threshold, accepted="x0->glass_wool_24k",
        round_evaluations=quick + 2, total_evaluations=quick + 2, next_round_bound=None)
    rows.append(templates.decision_row(len(rows), 0, surfaces=chosen.surfaces, variable_names=chosen.variable_names,
                                       theta_hex=chosen.theta_hex, value=1.0, screening_total=1.0,
                                       components=expand_components(1.0, 1.0, (0.5, 0.5, 0.0)), seconds=0.0,
                                       decision=choice, stop_reason=rounds.TEMPLATE_STOP_MAX_ROUNDS))
    templates.create_template_ledger(folder / run.TEMPLATES_FILE, th)
    templates.append_template_rows(folder / run.TEMPLATES_FILE, th, rows, settings=settings,
                                   start=templates.TemplateStart(start, tuple(float(x).hex() for x in theta), 2.0, 2.0,
                                                                 expand_components(2.0, 2.0, (1.0, 1.0, 0.0))))
    return header


def test_user_family_result_rebuild_and_endpoint_identity_read_the_snapshot(tmp_path: Path) -> None:
    """抓結果重建、終點核結果檔、終點核樣板身分其中任何一個還拿產品構造預設：使用者家族的面就重建不出來。"""
    entry = _optimizable()
    snapshot = tmp_path / materials.SNAPSHOT_FILE
    snapshot.write_bytes(materials.snapshot_bytes((entry,)))
    project = _project().model_copy(update={"presets": (USER,) * 6, "settings_path": str(config_path("material_gradient.toml")),
                                            "materials_path": str(snapshot), "materials_sha256": run.file_sha256(snapshot)})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    document = run.project_settings(project).model_dump()
    document["templates"].update(max_per_round=1, max_rounds=1)
    settings = MaterialGradientSettings.model_validate(document)
    table, kept = run.project_presets(project), _kept()
    designs = run.designs_for(project.presets, settings, kept=kept, table=table)
    context = rounds.TemplateContext(settings, table, {w.wall_name(): m for w, m in kept.items()},
                                     lambda designs: pytest.fail("不應求值"))
    header = _ledgers(tmp_path, settings, designs, context)
    with pytest.raises(ValueError, match=f"preset id 不認識：{USER}"):
        run.result_from_ledger(tmp_path, settings, kept, expected_header=header)
    result = run.result_from_ledger(tmp_path, settings, kept, expected_header=header, table=table)
    run.write_json_atomic(tmp_path / run.RESULT_FILE, result)
    restored = run.parse_result((tmp_path / run.RESULT_FILE).read_bytes())
    assert checked_result(tmp_path, restored, header, settings, kept, table=table) == restored
    assert [spec.preset_id for spec in restored.surfaces].count(USER) == 5
    final = run.final_designs([(s.surface_id, s.preset_id) for s in restored.surfaces], settings, kept, table=table)
    checked_design_identity(final, table)
    with pytest.raises(ValueError, match=f"preset {USER} 不在構造預設清單裡"):
        checked_design_identity(final, run.presets_table())


# ---- 網頁 ----

def _jobs(data_dir: Path, launcher: FakeLauncher) -> GradientJobs:
    return GradientJobs(data_dir, launcher=launcher, engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())


def test_web_offers_user_families_and_hands_a_library_copy_to_the_search(tmp_path: Path) -> None:
    """抓網頁的家族清單沒接使用者材料、開跑沒把材料庫交給搜尋，或交的是網頁之後還會改的那一份。"""
    data_dir = _data(tmp_path)
    (data_dir / "materials.json").write_text(library.dump_user_library((_fixed(), _optimizable())), encoding="utf-8")
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    preview = jobs.preview(NAME, GradientRequest(family=USER))
    plan = cast(dict[str, object], preview["plan"])
    assert plan["family"] == USER and USER in cast(list[str], plan["presets"])
    assert cast(dict[str, list[str]], preview["defaults"])["families"][-1] == USER
    view = jobs.start(NAME, GradientRequest(family=USER))
    (_, argv, _, _), = launcher.launched
    copy = data_dir / "gradient" / str(view["job_id"]) / "materials.json"
    assert argv[argv.index("--materials") + 1] == str(copy)
    assert copy.read_text(encoding="utf-8") == (data_dir / "materials.json").read_text(encoding="utf-8")


def test_web_refuses_fixed_family_and_unreadable_library_and_omits_flag_without_one(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    jobs.start(NAME, GradientRequest())
    (_, argv, _, _), = launcher.launched
    assert "--materials" not in argv
    (data_dir / "materials.json").write_text(library.dump_user_library((_fixed(),)), encoding="utf-8")
    with pytest.raises(PlanRefused, match="固定材料.*整塊選或不選"):
        jobs.preview(NAME, GradientRequest(family=FIXED))
    (data_dir / "materials.json").write_text("{", encoding="utf-8")
    with pytest.raises(PlanRefused, match="使用者材料庫（materials.json）讀不進來"):
        jobs.preview(NAME, GradientRequest())


def test_physics_check_never_blames_the_material_for_missing_double_precision() -> None:
    """pytest 行程刻意不開雙精度：物理檢查要講明是行程設定，不准回「材料參數物理拒算」。"""
    import jax.numpy as jnp

    assert jnp.asarray(0.0).dtype != jnp.float64
    chosen = materials.with_user_materials(_base(), run.presets_table(), (_optimizable(),))
    with pytest.raises(RuntimeError, match="沒有開雙精度.*不是材料參數的問題"):
        air, _ = product_air_environment()
        run.check_user_families(chosen, (63.0,), density_kg_m3=air.air_density_kg_m3, sound_speed_m_s=air.sound_speed_m_s)
