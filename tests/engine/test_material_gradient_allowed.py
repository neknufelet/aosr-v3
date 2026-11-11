"""每區准用清單＋固定材料進第三階段搜尋（#1082 段 2-3b）考卷（父行程、不求值的部分）。

- 准用清單：存法跟輸入順序無關、整面展到每一片、空的／重複的／認不得的／同牆混整面與面片都拒收。
- 換樣板候選：有清單的面只換清單裡的（扣掉現在的、照存法順序），沒清單的面照舊全部家族＋不貼；固定材料只從清單進來。
- 起跑前四條拒收：固定面給清單、家族不認識、起點不在清單裡、有深度上限又列固定材料；起跑配置直接寫固定材料也拒收。
- ``fixed:<代號>`` 的設計：整塊用那份材料、沒有變數；不在這一場的清單裡就拒收。材料從使用者固定材料或內建構造預設來，
  指到使用者的可最佳化材料、或不存在都拒收。
- 快照：沒有固定材料照舊寫 v1（逐位跟 2-3a 同一份）、有就寫 v2；2-3a 寫的 v1 讀得回來。
- 身分：沒有清單時 ``project.json``、資料夾名、換樣板帳本標頭逐位不動；有清單就進資料夾名與標頭（含固定材料摘要）。
- 網頁：設計設定的准用清單只給交給搜尋的區；轉成 ``--allowed``、起點照規則挑；沒有清單時參數逐位照舊。

雙精度的那幾題（整段換樣板接受固定材料再換走、接續、終點匯出、搜尋與終點同一份表、替身留出驗證）在
``test_material_gradient_allowed_runs``。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import material_gradient as search
from aosr.gradient import run
from aosr.gradient import template_rounds as rounds
from aosr.gradient.fixed_candidates import FixedCandidateCheck
from aosr.gradient.fixed_surfaces import (
    check_allowed_against_start, checked_allowed_targets, expanded_allowed_targets, fixed_targets_of, parse_allowed_targets,
)
from aosr.gui.gradient_plan import GradientRequest, PlanRefused, plan_search, start_arguments
from aosr.gui.zone_design import DesignSettings, ZoneDesign
from aosr.materials import library
from aosr.materials import search_library as materials
from aosr.materials.construction_models import ConstantImpedance
from aosr.reporting.scheme import Scheme
from tests.engine._material_gradient_templates_probe import kept_walls
from tests.engine.test_material_gradient_cli import _schemes
from tests.engine.test_material_gradient_mount import _project

if TYPE_CHECKING:
    from aosr.physics.fem_reduced import FullOperators, ReducedModel

GAP50 = "glass_wool_48k_50mm_gap50"
MEASURED = "panel_measured"
CARPET = "carpet_on_concrete"
_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
KEPT = [20000.0, 21000.0, 22000.0, 23000.0, 24000.0, 25000.0]
# 主線 4e2ad58e 的 gradient_id，跟 test_material_gradient_library 同一組純字串輸入（那份釘的是 b4358482 的同一個值）。
MAIN_PLAIN_ID = "4b9652510d8cf350c2060cb58b174764"
_PURE_PRESETS = ("glass_wool_48k_50mm_gap50", None, "glass_wool_24k_50mm", None, None, "glass_wool_96k_50mm")


def _settings() -> MaterialGradientSettings:
    return load_material_gradient_settings(config_path("material_gradient.toml"))


def _measured(*, imaginary: bool = True) -> library.FixedEntry:
    rows = [{"frequency_hz": frequency, "real_pa_s_m": 1200.0 + frequency,
             **({"imag_pa_s_m": -40000.0 / frequency} if imaginary else {})} for frequency in (50.0, 100.0, 200.0, 400.0)]
    return cast(library.FixedEntry, _ENTRY.validate_python({
        "kind": "fixed", "id": MEASURED, "label": "實測吸音板（自填）", "provenance": "measured", "source": "阻抗管實測",
        "material": {"kind": "measured_impedance", "points": rows, "source_and_conditions": "阻抗管，背 5 cm 空腔",
                     "base_assembly": {"kind": "assembly", "layers": [{"kind": "air", "thickness_m": 0.05}],
                                       "backing": "rigid"}}}))


def _optimizable() -> library.OptimizableEntry:
    return cast(library.OptimizableEntry, _ENTRY.validate_python({
        "kind": "optimizable", "id": "rock_wool_60k", "label": "60K 岩棉（自填）", "provenance": "model", "source": "自填",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 20000.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.04}, "max_total_depth_m": 0.15}))


def _table() -> tuple[materials.PresetMaterial, ...]:
    targets = ("fixed:" + CARPET, "fixed:" + MEASURED)
    return (*run.presets_table(), *materials.fixed_candidates(targets, run.presets_table(), (_measured(),)))


def _context(allowed: dict[str, tuple[str, ...]] | None) -> rounds.TemplateContext:
    def never(_: tuple[search.SurfaceDesign, ...]) -> search.ChainObjective:
        raise AssertionError("這一題不該建目標")
    return rounds.TemplateContext(_settings(), _table(), kept_walls(KEPT), never, allowed=allowed)


# ---- 准用清單本身 ----

def test_allowed_lists_store_in_a_fixed_order_and_expand_to_patches() -> None:
    """抓存法跟著輸入順序走（同一份清單換順序就換資料夾）、整面沒展到每一片。"""
    first = parse_allowed_targets(["x0=none,fixed:b,glass_wool_96k,fixed:a,glass_wool_24k", "ceiling=none"])
    second = checked_allowed_targets({"ceiling": ["none"], "x0": ["fixed:a", "glass_wool_24k", "none", "fixed:b",
                                                                   "glass_wool_96k"]})
    assert first == second == {"ceiling": ("none",),
                               "x0": ("glass_wool_24k", "glass_wool_96k", "fixed:a", "fixed:b", "none")}
    assert list(first) == ["ceiling", "x0"]
    assert expanded_allowed_targets({"x0": ("none",)}, ["x0@a", "x0@b", "y0"]) == {"x0@a": ("none",), "x0@b": ("none",)}
    assert fixed_targets_of(first) == ("fixed:a", "fixed:b")
    with pytest.raises(ValueError, match="不在搜尋面片清單"):
        expanded_allowed_targets({"x0@zz": ("none",)}, ["x0@a"])


@pytest.mark.parametrize(("given", "words"), [
    ({"x0": []}, "是空的"), ({"x0": ["none", "none"]}, "有重複"), ({"x0": ["Fixed:A"]}, "認不得"),
    ({"x0": ["fixed:"]}, "認不得"), ({"x0": ["none"], "x0@a": ["none"]}, "同時給牆名與面片代號")])
def test_allowed_lists_refuse_empty_duplicate_unknown_and_mixed(given: dict[str, list[str]], words: str) -> None:
    with pytest.raises(ValueError, match=words):
        checked_allowed_targets(given)


def test_allowed_flag_refuses_bad_shape_and_repeats() -> None:
    with pytest.raises(ValueError, match="要寫成"):
        parse_allowed_targets(["x0"])
    with pytest.raises(ValueError, match="給了兩次"):
        parse_allowed_targets(["x0=none", "x0=glass_wool_24k"])


# ---- 換樣板候選 ----

def test_listed_surfaces_swap_only_to_their_list_and_others_keep_every_family() -> None:
    """抓清單被忽略（固定材料進不來、或不准用的家族照樣換）、扣錯現在的目標、沒清單的面被改動。"""
    settings = _settings()
    surfaces = (("floor", None), ("ceiling", GAP50), ("x0", "fixed:" + MEASURED), ("xL", GAP50), ("y0", None), ("yL", None))
    allowed = {"x0": ("glass_wool_24k", "fixed:" + CARPET, "fixed:" + MEASURED, "none"),
               "xL": ("glass_wool_48k", "fixed:" + MEASURED)}
    found = rounds.alternatives(surfaces, settings, None, allowed)
    assert [item for item in found if item[0] in ("x0", "xL")] == [
        ("x0", "glass_wool_24k"), ("x0", "fixed:" + CARPET), ("x0", "none"), ("xL", "fixed:" + MEASURED)]
    assert [item for item in found if item[0] not in ("x0", "xL")] == [
        item for item in rounds.alternatives(surfaces, settings) if item[0] not in ("x0", "xL")]
    assert rounds.next_round_bound(surfaces, settings, None, allowed) < rounds.next_round_bound(surfaces, settings)
    assert rounds.target_of("fixed:" + MEASURED, settings) == "fixed:" + MEASURED
    assert rounds.preset_for_target("fixed:" + MEASURED, settings) == "fixed:" + MEASURED


def test_swap_to_a_fixed_material_has_no_variables_and_refuses_unlisted_targets() -> None:
    """抓換成固定材料時變數沒拿掉、別面的 θ 被動到、材料不是清單那一份、清單外的目標照樣換。"""
    context = _context({"x0": ("glass_wool_48k", "fixed:" + MEASURED)})
    designs = rounds.designs_for_surfaces(tuple(zip(search.wall_surface_ids(), (None, GAP50, GAP50, GAP50, None, None),
                                                    strict=True)), context)
    theta = search.snap_to_specs(search.baseline_theta(designs), designs).theta
    swapped, new_theta = rounds.swap(designs, theta, "x0", "fixed:" + MEASURED, context)
    x0 = swapped[2]
    assert (x0.preset_id, x0.variables, x0.material) == ("fixed:" + MEASURED, (), None)
    assert x0.kept_material == _measured().surface_material()
    assert list(new_theta) == [*theta[:2], *theta[4:]]
    assert rounds.surfaces_of(swapped)[2] == ("x0", "fixed:" + MEASURED)
    with pytest.raises(ValueError, match="准用清單"):
        rounds.swap(designs, theta, "x0", "glass_wool_24k", context)
    back, back_theta = rounds.swap(swapped, new_theta, "x0", "glass_wool_48k", context)
    assert back == designs and len(back_theta) == len(theta)


def test_fixed_target_designs_need_the_material_in_this_search_table() -> None:
    with pytest.raises(ValueError, match="不在這一場的固定材料清單"):
        search.designs_from_presets((("x0", "fixed:" + MEASURED),), _settings(), run.presets_table(),
                                    kept={"x0": ConstantImpedance(impedance_pa_s_m=9000.0)})
    with pytest.raises(ValueError, match="要帶那份材料"):
        search.SurfaceDesign("x0", "fixed:x", None, None, (), (), (), None, None, (), None)


# ---- 起跑前的拒收 ----

def _start(allowed: dict[str, tuple[str, ...]], *, fixed: dict[str, str] | None = None,
           caps: dict[str, float] | None = None, x0: str | None = GAP50) -> None:
    surfaces = (("floor", None), ("ceiling", GAP50), ("x0", x0), ("xL", GAP50), ("y0", None), ("yL", None))
    check_allowed_against_start(allowed, surfaces, _settings(), fixed_surfaces=fixed or {}, max_depths=caps or {})


@pytest.mark.parametrize(("allowed", "fixed", "caps", "x0", "words"), [
    ({"floor": ("none",)}, {"floor": "scheme-material"}, {}, GAP50, "是固定面"),
    ({"x0": ("glass_wool_48k", "rock_wool_60k")}, {}, {}, GAP50, "不認識的家族"),
    ({"x0": ("glass_wool_24k", "none")}, {}, {}, GAP50, "起跑是 glass_wool_48k"),
    ({"floor": ("glass_wool_24k",)}, {}, {}, GAP50, "起跑是 none"),
    ({"x0": ("glass_wool_48k", "fixed:" + CARPET)}, {}, {"x0": 0.1}, GAP50, "有深度上限"),
    ({"x0": ("glass_wool_48k",)}, {}, {}, "fixed:" + CARPET, "不能直接寫固定材料")])
def test_start_refuses_lists_that_break_the_start_configuration(
        allowed: dict[str, tuple[str, ...]], fixed: dict[str, str], caps: dict[str, float], x0: str, words: str) -> None:
    with pytest.raises(ValueError, match=words):
        _start(allowed, fixed=fixed, caps=caps, x0=x0)


def test_start_accepts_a_list_that_contains_the_start_target() -> None:
    _start({"x0": ("glass_wool_48k", "fixed:" + CARPET), "floor": ("none", "fixed:" + MEASURED), "xL": ("glass_wool_48k",)},
           caps={"ceiling": 0.15})


# ---- 固定材料從哪來、快照 ----

def test_fixed_candidates_resolve_user_then_builtin_and_refuse_the_rest() -> None:
    """抓使用者的與內建的取錯、可最佳化材料被當成固定材料、不存在的悄悄略過。"""
    table = materials.with_user_presets(run.presets_table(), (_optimizable(),))
    found = materials.fixed_candidates(("fixed:" + MEASURED, "fixed:" + CARPET), table, (_measured(),))
    assert [item.id for item in found] == ["fixed:" + CARPET, "fixed:" + MEASURED]
    carpet = next(preset for preset in run.presets_table() if preset.id == CARPET)
    assert found[0].material == carpet.material
    assert found[1].material == _measured().surface_material().model_dump(mode="json")
    for target in ("fixed:rock_wool_60k", "fixed:no_such_panel"):
        with pytest.raises(materials.MaterialsRefused, match="不在材料庫"):
            materials.fixed_candidates((target,), table, (_measured(),))


def test_snapshot_without_fixed_materials_stays_v1_and_v2_round_trips(tmp_path: Path) -> None:
    """抓沒有固定材料也升 v2（2-3a 的資料夾名就變了）、v2 讀不回固定材料、2-3a 寫的 v1 讀不回來。"""
    v1 = materials.snapshot_bytes((_optimizable(),))
    assert json.loads(v1)["format_version"] == "aosr.material_gradient_materials.v1" and "fixed" not in json.loads(v1)
    legacy = materials.MaterialsSnapshot(format_version="aosr.material_gradient_materials.v1", materials=(_optimizable(),))
    assert v1 == (json.dumps(legacy.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path = tmp_path / "materials.json"
    path.write_bytes(v1)
    assert materials.read_snapshot_entries(path, hashlib.sha256(v1).hexdigest()) == materials.Snapshot((_optimizable(),), ())
    v2 = materials.snapshot_bytes((), (_measured(),))
    path.write_bytes(v2)
    assert json.loads(v2)["format_version"] == "aosr.material_gradient_materials.v2"
    assert materials.read_snapshot_entries(path, hashlib.sha256(v2).hexdigest()) == materials.Snapshot((), (_measured(),))
    assert materials.user_fixed_entries([MEASURED, CARPET], (_optimizable(), _measured())) == (_measured(),)


# ---- 身分 ----

def test_without_lists_project_folder_name_and_template_header_do_not_move() -> None:
    """抓沒有清單也寫出新欄位、或資料夾名換了（舊資料夾就接不上）。"""
    dumped = json.loads(_project().model_dump_json())
    assert "allowed_targets" not in dumped and "fixed_candidate_checks" not in dumped
    identity = run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=_PURE_PRESETS,
                               purpose="music", settings_fingerprint="c" * 64, physics_identity="phys",
                               objective_settings_fingerprint="d" * 64, allowed_targets={})
    assert identity == MAIN_PLAIN_ID
    listed = run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=_PURE_PRESETS,
                             purpose="music", settings_fingerprint="c" * 64, physics_identity="phys",
                             objective_settings_fingerprint="d" * 64, allowed_targets={"x0": ("none",)})
    assert listed != MAIN_PLAIN_ID


def _check(target: str, **changes: object) -> dict[str, object]:
    return {"target": target, "surfaces": ["x0"], "frequency_db": 0.01, "band_db": 0.001, "frequency_ruler": 0.011,
            "band_ruler": 0.01, "full_solves": 47, "seconds": 1.0, **changes}


def test_project_needs_one_passing_holdout_check_per_fixed_target() -> None:
    """抓沒驗就開跑（清單有固定材料、紀錄少一筆）、驗沒過也記成可以開跑。"""
    base = _project().model_dump()
    allowed = {"x0": ["none", "fixed:" + CARPET, "fixed:" + MEASURED]}
    good = [_check("fixed:" + CARPET), _check("fixed:" + MEASURED)]
    project = run.GradientProject.model_validate({**base, "allowed_targets": allowed, "fixed_candidate_checks": good})
    assert [check.target for check in project.fixed_candidate_checks] == ["fixed:" + CARPET, "fixed:" + MEASURED]
    with pytest.raises(ValidationError, match="每一個固定材料都要有一筆"):
        run.GradientProject.model_validate({**base, "allowed_targets": allowed, "fixed_candidate_checks": good[:1]})
    with pytest.raises(ValidationError, match="沒過"):
        FixedCandidateCheck.model_validate(_check("fixed:" + CARPET, frequency_db=0.2))
    with pytest.raises(ValidationError, match="沒過"):
        FixedCandidateCheck.model_validate(_check("fixed:" + CARPET, band_db=0.02))


@dataclasses.dataclass(frozen=True)
class _FakeOperators:
    sources: dict[str, int]
    receivers: dict[str, int]


@pytest.mark.parametrize(("frequency_db", "band_db", "passes"), [(0.01, 0.5, False), (5.0, 0.0001, False),
                                                                 (0.01, 0.0001, True)])
def test_holdout_gate_judges_each_ruler_on_its_own(monkeypatch: pytest.MonkeyPatch, frequency_db: float, band_db: float,
                                                  passes: bool) -> None:
    """抓關卡只看一把尺：完整解那一步換成回固定數字的報告，只超帶能量、只超逐頻都要照實拒收（訊息帶兩個數字），
    兩把都在尺內才回紀錄、紀錄抄的就是報告的數字。"""
    from types import SimpleNamespace

    from aosr.gradient import fixed_candidates
    from aosr.physics import fem_reduced_check

    report = SimpleNamespace(frequency_db_by_seat={"front": frequency_db}, band_db_by_seat={"front": band_db}, full_solves=3)
    monkeypatch.setattr(fem_reduced_check, "validate_reduced", lambda *args, **kwargs: report)
    monkeypatch.setattr(fixed_candidates, "candidate_designs", lambda designs, *args: designs)
    monkeypatch.setattr(fixed_candidates, "design_impedances", lambda *args, **kwargs: {"x0": (1.0 + 0.0j,)})
    model = cast("ReducedModel", SimpleNamespace(frequencies_hz=(50.0,), source_names=("s",), receiver_names=("front",)))
    operators = cast("FullOperators", _FakeOperators(sources={"s": 0}, receivers={"front": 0}))

    def gate() -> tuple[FixedCandidateCheck, ...]:
        return fixed_candidates.check_fixed_candidates(model, operators, (), {"x0": ("none", "fixed:" + CARPET)}, (),
                                                       density_kg_m3=1.2, sound_speed_m_s=343.0)
    if passes:
        (check,) = gate()
        assert (check.frequency_db, check.band_db, check.full_solves) == (frequency_db, band_db, 3)
        return
    with pytest.raises(materials.MaterialsRefused, match="沒過留出驗證") as refused:
        gate()
    assert f"逐頻最大差 {frequency_db:g} dB" in str(refused.value) and f"帶能量最大差 {band_db:g} dB" in str(refused.value)


def test_template_header_binds_lists_and_fixed_material_digests() -> None:
    """抓標頭沒記清單或固定材料摘要（換了一份同名材料照樣接續）、沒有清單也寫出新欄位。"""
    from types import SimpleNamespace

    import numpy as np

    from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
    from aosr.gradient.endpoint import material_digest
    from aosr.physics.fem_reduced import ReducedModel
    from aosr.physics.fem_reduced_boundary import boundary_patches
    from aosr.scoring.gradient_objective_jax import gradient_objective_constants
    from tests.engine._six_search_helpers import six_header_for
    from tests.engine.test_material_gradient import _gradient_settings, _receivers
    from tests.engine.test_material_gradient_max_depth import _kept

    model = cast(ReducedModel, SimpleNamespace(input_hash="input-x", mesh_hash="mesh-x", physics_identity="physics-x",
                                              patch_keys=boundary_patches(), matrices=SimpleNamespace(boundary=(None,) * 6)))
    constants = gradient_objective_constants(np.asarray(FEM_LANE_FREQUENCIES_HZ), _gradient_settings(), _receivers(), 2)
    designs = run.designs_for((GAP50,) * 6, _settings(), kept=_kept())
    header = six_header_for(model, designs, _settings(), objective_constants=constants)
    plain = run.template_header_for(header, _settings(), _kept(), base_ledger_sha256="e" * 64)
    assert "allowed_targets" not in plain.model_dump_json() and "fixed_candidate_digests" not in plain.model_dump_json()
    allowed = {"x0": ("glass_wool_48k", "fixed:" + MEASURED)}
    listed = run.template_header_for(header, _settings(), _kept(), base_ledger_sha256="e" * 64, allowed=allowed,
                                     table=_table())
    assert listed.allowed_targets == allowed
    assert listed.fixed_candidate_digests == {"fixed:" + MEASURED: material_digest(_measured().surface_material())}
    with pytest.raises(ValueError, match="不在這一場的構造清單"):
        run.template_header_for(header, _settings(), _kept(), base_ledger_sha256="e" * 64, allowed=allowed,
                                table=run.presets_table())
    with pytest.raises(ValidationError, match="固定材料摘要"):
        listed.model_validate({**listed.model_dump(), "fixed_candidate_digests": {}})


# ---- 網頁 ----

def test_zone_design_lists_only_for_search_zones_and_old_files_keep_their_bytes() -> None:
    assert ZoneDesign(mode="search").model_dump_json() == '{"mode":"search"}'
    with pytest.raises(ValidationError, match="只給交給搜尋的區"):
        ZoneDesign(mode="fixed", allowed=("none",))
    with pytest.raises(ValidationError, match="有重複"):
        ZoneDesign(mode="search", allowed=("none", "none"))


def _scheme_and_design(allowed: dict[str, tuple[str, ...]]) -> tuple[Scheme, DesignSettings]:
    scheme = _schemes()["scheme.json"]
    zones = {wall.wall_name(): {"whole_wall": ZoneDesign(mode="search", allowed=allowed.get(wall.wall_name()))}
             for wall in Wall.all()}
    return scheme, DesignSettings(scheme_id=scheme.scheme_id, zones=zones)


def test_plan_maps_library_ids_to_targets_and_picks_each_start() -> None:
    """抓固定材料沒轉成 fixed:、起點沒照清單挑（選定家族不准用還照用）、沒有清單時參數變了。"""
    scheme, design = _scheme_and_design({"x0": ("glass_wool_24k", MEASURED), "floor": ("none", CARPET),
                                         "ceiling": ("glass_wool_48k", "none")})
    fixed_ids = frozenset((CARPET, MEASURED))
    plan = plan_search(scheme, design, GradientRequest(), _settings(), fixed_ids)
    starts = dict(zip(plan.surfaces, plan.presets, strict=True))
    assert starts["x0"] == "glass_wool_24k_50mm" and starts["floor"] is None and starts["ceiling"] == GAP50
    assert plan.allowed == {"ceiling": ("glass_wool_48k", "none"), "floor": ("none", "fixed:" + CARPET),
                            "x0": ("glass_wool_24k", "fixed:" + MEASURED)}
    arguments = start_arguments(plan)
    assert arguments[arguments.index("--allowed") + 1] == "ceiling=glass_wool_48k,none"
    _, bare = _scheme_and_design({})
    plain = plan_search(scheme, bare, GradientRequest(), _settings(), fixed_ids)
    assert "--allowed" not in start_arguments(plain) and "allowed" not in plain.model_dump_json()


@pytest.mark.parametrize(("allowed", "words"), [({"x0": (MEASURED,)}, "只有固定材料"), ({"x0": ("no_such",)}, "材料庫裡沒有")])
def test_plan_refuses_lists_without_a_start_or_with_unknown_ids(allowed: dict[str, tuple[str, ...]], words: str) -> None:
    scheme, design = _scheme_and_design(allowed)
    with pytest.raises(PlanRefused, match=words):
        plan_search(scheme, design, GradientRequest(), _settings(), frozenset((MEASURED,)))
