"""每一區自己的可調範圍走進第三階段搜尋（#1199 範圍第 1 支，第 2 單元）：計畫與開跑參數、命令列、起跑的界、
換樣板換進去與換回來、終點的設計、綁定、替身閘門、身分與接續。

數字全是考卷自己編的：材料是內建家族（設定檔 ``material_gradient.toml``：48K、24K、96K 產品厚度都是 0.05／0.1／0.2，
空腔 0～0.25、總深 0.3；48K 樣板起跑值 0.05／0.05，24K、96K 不調空腔）；房間是 ``_zone_symmetry_room`` 與
``test_gui_gradient_jobs`` 的假房間、命令列那幾題用替身沿用考卷的自編小房間（``_six_search_cli_case``）。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, cast

import numpy as np
import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, endpoint_run, run
from aosr.gradient import fixed_surfaces as faces
from aosr.gradient import material_gradient as search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import material_gradient_loop as loop
from aosr.gradient import template_rounds as rounds
from aosr.gradient.depth_fit import ZoneRange
from aosr.gradient.ties import TieRefused
from aosr.gui.gradient_jobs import plan_configuration
from aosr.gui.gradient_plan import PlanRefused, SearchPlan, configuration_arguments, plan_symmetry, start_arguments
from aosr.gui.zone_design import DesignSettings, write_design
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_boundary import boundary_patches
from aosr.scoring.gradient_objective_jax import gradient_objective_constants
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._six_search_cli_case import model as model, operators as operators
from tests.engine._six_search_helpers import six_header_for
from tests.engine._zone_range_control import SETTINGS, WOOL, control_design, control_plan, control_scheme
from tests.engine.test_material_gradient import _gradient_settings, _receivers
from tests.engine.test_material_gradient_mount import _project
from tests.engine.test_zone_range import _ranged

PRESET = "glass_wool_48k_50mm_gap50"
WOOL24 = "glass_wool_24k"
FRONT = ("x0@b", "x0@a")
SIDES = ("y0@s", "yL@s")
Record = dict[str, object]


def start_designs(plan: SearchPlan) -> tuple[tuple[search.SurfaceDesign, ...], dict[str, tuple[str, ...]]]:
    """命令列開跑那一組起跑配置（``cli.start_designs``），計畫的可調範圍一起交進去（控制組那一支只用主線就有的參數）。"""
    return cli.start_designs(control_scheme(), SETTINGS, run.presets_table(), plan.presets, mount=plan.request.mount,
                             fixed=dict(plan.fixed_surfaces), caps=dict(plan.max_depths), allowed=dict(plan.allowed),
                             capabilities=load_capabilities(config_path("capabilities.toml")),
                             directivity=load_directivity_defaults(config_path("directivity_defaults.toml")), ties=plan.ties,
                             ranges=plan.ranges)


def _kept() -> dict[Wall, SurfaceMaterial]:
    return {wall: ConstantImpedance(impedance_pa_s_m=12348.0) for wall in Wall.all()}


def _never(_: tuple[search.SurfaceDesign, ...]) -> search.ChainObjective:
    raise AssertionError("這一題不該建目標")


def _context(ranges: dict[str, dict[str, ZoneRange]], *, caps: dict[str, float] | None = None) -> rounds.TemplateContext:
    names = tuple(wall.wall_name() for wall in Wall.all())
    return rounds.TemplateContext(SETTINGS, run.presets_table(), {wall.wall_name(): material for wall, material in _kept().items()},
                                  _never, max_depths=caps, ranges=faces.expanded_ranges(ranges, names) or None)


def _box(design: search.SurfaceDesign) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    return design.lower, design.upper, design.legal_thicknesses_m


def _x0(theta: search.FloatVector, designs: tuple[search.SurfaceDesign, ...]) -> list[float]:
    offset = sum(len(design.variables) for design in designs[:2])
    return [float(value) for value in theta[offset:offset + len(designs[2].variables)]]


# ---- 起跑：界就是範圍、合法厚度跟著切 ----

def test_zone_range_search_start_bounds_and_legal_thicknesses_are_the_range() -> None:
    """抓起跑設計沒收窄（或只改界沒切合法厚度）：x0 那一面 48K 收窄到棉 0.05～0.1、空腔 0.02～0.08；其他面逐位照舊。"""
    ranges = {"x0": {WOOL: ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.02, 0.08))}}
    designs = run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept(), ranges=ranges)
    plain = run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept())
    assert _box(designs[2]) == ((0.05, 0.02), (0.1, 0.08), (0.05, 0.1))
    assert _box(plain[2]) == ((0.05, 0.0), (0.2, 0.25), (0.05, 0.1, 0.2))
    assert designs[:2] + designs[3:] == plain[:2] + plain[3:]
    assert _x0(search.baseline_theta(designs), designs) == [0.05, 0.05]
    capped = run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept(), ranges=ranges, max_depths={"x0": 0.12})
    assert _box(capped[2]) == _box(designs[2]) and capped[2].max_total_depth_m == 0.12
    bounds = search.design_bounds(designs)
    assert bounds[0][4:6].tolist() == [0.05, 0.02] and bounds[1][4:6].tolist() == [0.1, 0.08]


def test_zone_range_search_zoned_plan_carries_the_range_to_the_start_designs() -> None:
    """網頁那條路：設計設定的範圍進計畫（面片代號 → 家族 → 範圍）、開跑參數多一個 ``--range``，命令列開跑那一組
    起跑配置照它收窄；建替身核固定材料那一份參數跟沒寫範圍一樣。"""
    given = {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.08]}
    plan = control_plan(_ranged({"ceiling@c": {"ranges": {WOOL: given}}}))
    assert plan.ranges == {"ceiling@c": {WOOL: ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.0, 0.08))}}
    arguments = start_arguments(plan)
    assert arguments[arguments.index("--range") + 1] == "ceiling@c=glass_wool_48k:0.05,0.1:0.0,0.08"
    assert arguments.count("--range") == 1 and arguments.index("--range") < arguments.index("--tie")
    assert configuration_arguments(plan) == configuration_arguments(control_plan())
    designs, _ = start_designs(plan)
    ceiling = next(design for design in designs if design.surface_id == "ceiling@c")
    assert _box(ceiling) == ((0.05, 0.0), (0.1, 0.08), (0.05, 0.1))
    assert SearchPlan.model_validate_json(plan.model_dump_json()) == plan
    # 開跑計畫照分區頁標紅那一把擋（同一句、點名哪一區哪一種材料）。
    with pytest.raises(PlanRefused, match="立面圖上標紅.*天花「c」：48K 玻璃棉.*的可調範圍：棉厚 0.05～0.4 公尺超出材料頁的範圍.*要放寬請到材料頁改那份材料"):
        control_plan(_ranged({"ceiling@c": {"ranges": {WOOL: {"porous_m": [0.05, 0.4]}}}}))
    with pytest.raises(PlanRefused, match="立面圖上標紅.*左牆「s」：可調範圍的材料 demo_unknown_wool 材料庫裡沒有"):
        control_plan(_ranged({"y0@s": {"ranges": {"demo_unknown_wool": {"porous_m": [0.05, 0.1]}}}}))


def test_zone_range_search_start_value_outside_the_range_is_refused_like_the_material_page() -> None:
    """起跑值不在範圍：照材料頁「初值不在範圍內」、搜尋起點盒界外拒收那一條拒收（不投影、不換起點），點名那一面那一族；
    網頁開跑計畫也擋、句子講哪一區、哪一種材料從哪個值起跑。"""
    with pytest.raises(ValueError, match="面 x0 起跑的 glass_wool_48k：起跑值棉厚 0.05 公尺不在這一區的範圍 0.1～0.2 公尺內"):
        run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept(), ranges={"x0": {WOOL: ZoneRange(porous_m=(0.1, 0.2))}})
    with pytest.raises(ValueError, match="起跑值空腔 0.05 公尺不在這一區的範圍 0～0.03 公尺內.*初值不在範圍內"):
        run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept(),
                        ranges={"x0": {WOOL: ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.0, 0.03))}})
    with pytest.raises(PlanRefused, match="「c」從48K 玻璃棉.*起跑：起跑值棉厚 0.05 公尺不在這一區的範圍 0.1～0.2 公尺內"):
        control_plan(_ranged({"ceiling@c": {"ranges": {WOOL: {"porous_m": [0.1, 0.2]}}}}))
    # 接續與終點重建問題只走 designs_for（不經命令列的 check_ranges）：範圍本身不合、固定面給範圍也要在這裡擋。
    with pytest.raises(ValueError, match="面 x0 的 glass_wool_48k：.*要放寬請到材料頁改那份材料"):
        run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept(), ranges={"x0": {WOOL: ZoneRange(porous_m=(0.05, 0.4))}})
    with pytest.raises(ValueError, match="面 x0 是固定面（不換樣板），不能再給可調範圍"):
        run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept(), fixed_surfaces={"x0": "glass_wool_24k_50mm"},
                        ranges={"x0": {WOOL: ZoneRange(porous_m=(0.05, 0.1))}})


@pytest.mark.parametrize(("value", "extra", "message"), [
    ("x0=glass_wool_48k:0.1,0.2", (), "起跑值棉厚 0.05 公尺不在這一區的範圍"),
    ("x0=glass_wool_48k:0.05,0.4", (), "要放寬請到材料頁改那份材料"),
    ("x0=glass_wool_48k:0.05,0.15", (), "要是產品厚度之一"),
    ("x0=glass_wool_24k:0.05,0.1:0,0.03", (), "不調空腔"),
    ("x0=no_such_wool:0.05,0.1", (), "不認識的家族"),
    ("x0=glass_wool_24k:0.05,0.1", ("--allowed", "x0=glass_wool_48k,none"), "不在這一面的准用清單"),
    ("floor=glass_wool_48k:0.05,0.1", ("--fixed-surface", "floor=glass_wool_24k_50mm"), "固定面"),
    ("x0=glass_wool_24k:0.1,0.2", ("--max-depth", "x0=0.08", "--allowed", "x0=glass_wool_48k,glass_wool_24k,none"),
     "照這一面的可調範圍收窄後"),
    ("x0=glass_wool_24k:0.1,0.2", ("--max-depth", "x0=0.08"), "面 x0 照這一面的可調範圍收窄後"),
    ("x0=glass_wool_48k:0.05,0.1:0.05,0.1", ("--max-depth", "x0=0.08"), "最淺排法 0.1 m"),
    ("x0=glass_wool_48k:0.05", (), "要寫成 下,上"),
    ("x0=glass_wool_48k:0.05,abc", (), "不是數字"),
])
def test_zone_range_search_command_line_refuses_before_exporting_or_creating_a_folder(
        value: str, extra: tuple[str, ...], message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    from aosr.reporting import fem_reduced_scheme
    from tests.engine.test_material_gradient_cli import _schemes

    scheme = tmp_path / "scheme.json"
    scheme.write_text(_schemes()["scheme.json"].model_dump_json())
    monkeypatch.setattr(fem_reduced_scheme, "export_operators", lambda *a, **kw: pytest.fail("拒收前不可匯出"))
    args = ["start", "--scheme", str(scheme), "--surrogate", "/absent", "--presets", ",".join([PRESET] * 6),
            "--purpose", _project().purpose, "--root", str(tmp_path / "out"), "--range", value, *extra]
    assert cli.main(args) == cli.EXIT_FAILED
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# ---- 換樣板：換進去、換回來都照範圍 ----

def test_zone_range_search_swap_in_and_back_stay_inside_the_range() -> None:
    """抓換樣板沒套範圍：x0 起跑不貼，換到 48K（範圍棉 0.1～0.2、空腔 0～0.03）家族初值 0.05／0.05 投影進範圍成
    0.1／0.03；換到 24K（範圍棉 0.1～0.2）成 0.1；再換回 48K、換成不貼再換回 48K 都回到同一組界與值。沒寫範圍的 96K 照家族。"""
    ranges = {"x0": {WOOL: ZoneRange(porous_m=(0.1, 0.2), cavity_m=(0.0, 0.03)), WOOL24: ZoneRange(porous_m=(0.1, 0.2))}}
    context = _context(ranges)
    designs = run.designs_for((PRESET, PRESET, None, PRESET, PRESET, PRESET), SETTINGS, kept=_kept(), ranges=ranges)
    theta = search.baseline_theta(designs)
    wool, wool_theta = rounds.swap(designs, theta, "x0", WOOL, context)
    assert _box(wool[2]) == ((0.1, 0.0), (0.2, 0.03), (0.1, 0.2)) and _x0(wool_theta, wool) == [0.1, 0.03]
    light, light_theta = rounds.swap(wool, wool_theta, "x0", WOOL24, context)
    assert _box(light[2]) == ((0.1,), (0.2,), (0.1, 0.2)) and _x0(light_theta, light) == [0.1]
    back, back_theta = rounds.swap(light, light_theta, "x0", WOOL, context)
    assert back == wool and _x0(back_theta, back) == [0.1, 0.03]
    bare, bare_theta = rounds.swap(back, back_theta, "x0", "none", context)
    again, again_theta = rounds.swap(bare, bare_theta, "x0", WOOL, context)
    assert again == wool and np.array_equal(again_theta, wool_theta)
    heavy, heavy_theta = rounds.swap(wool, wool_theta, "x0", "glass_wool_96k", context)
    assert _box(heavy[2]) == ((0.05,), (0.2,), (0.05, 0.1, 0.2)) and _x0(heavy_theta, heavy) == [0.05]
    assert run.final_designs(rounds.surfaces_of(back), SETTINGS, _kept(), ranges=ranges) == back
    untouched = [index for index in range(len(theta)) if index < 4] + [index + 2 for index in range(4, len(theta))]
    assert [float(wool_theta[index]) for index in untouched] == [float(value) for value in theta]
    plain, plain_theta = rounds.swap(designs, theta, "x0", WOOL, _context({}))
    assert _box(plain[2]) == ((0.05, 0.0), (0.2, 0.25), (0.05, 0.1, 0.2)) and _x0(plain_theta, plain) == [0.05, 0.05]


def test_zone_range_search_swap_projects_into_the_range_and_the_cap_together() -> None:
    """範圍與個案上限一起：48K 範圍棉 0.1～0.2、空腔 0～0.03，上限 0.12：初值投影進「盒界 ∩ 總深」再取規格。"""
    ranges = {"x0": {WOOL: ZoneRange(porous_m=(0.1, 0.2), cavity_m=(0.0, 0.03))}}
    context = _context(ranges, caps={"x0": 0.12})
    designs = run.designs_for((PRESET, PRESET, None, PRESET, PRESET, PRESET), SETTINGS, kept=_kept(), ranges=ranges,
                              max_depths={"x0": 0.12})
    swapped, theta = rounds.swap(designs, search.baseline_theta(designs), "x0", WOOL, context)
    assert _box(swapped[2]) == ((0.1, 0.0), (0.2, 0.03), (0.1, 0.2)) and swapped[2].max_total_depth_m == 0.12
    assert _x0(theta, swapped) == [0.1, 0.12 - 0.1]  # 棉在範圍下界 0.1，空腔切到「上限減棉」
    assert search.is_feasible(theta, swapped)
    # 收窄後最淺 0.1 放不進 0.08（沒收窄的最淺 0.05 放得進）：換過去照收窄後的判拒收、句子是收窄後的最淺。
    tight = _context(ranges, caps={"x0": 0.08})
    start = run.designs_for((PRESET, PRESET, None, PRESET, PRESET, PRESET), SETTINGS, kept=_kept(), max_depths={"x0": 0.08})
    with pytest.raises(ValueError, match="面 x0 的深度上限 0.08 m 放不下家族 glass_wool_48k 最淺排法 0.1 m"):
        rounds.swap(start, search.baseline_theta(start), "x0", WOOL, tight)


def test_zone_range_search_candidates_only_step_to_the_cut_thicknesses() -> None:
    """取規格與跳點只到切過的合法厚度：x0 收窄到 48K 棉 0.1～0.2，在 0.1 時只剩往厚一格（0.2）的候選，
    沒有往薄跳到 0.05 那一個；沒寫範圍時兩個都有。取規格把連續的 0.13 吸到 0.1、0.16 吸到 0.2。"""
    ranges = {"x0": {WOOL: ZoneRange(porous_m=(0.1, 0.2), cavity_m=(0.0, 0.1))}}
    bare = run.designs_for((None,) * 6, SETTINGS, kept=_kept(), ranges=ranges)
    narrowed, start = rounds.swap(bare, search.baseline_theta(bare), "x0", WOOL, _context(ranges))
    full, _ = rounds.swap(bare, search.baseline_theta(bare), "x0", WOOL, _context({}))
    theta, slope = np.asarray([0.1, 0.03]), np.asarray([1.0, 1.0])
    assert start.tolist() == [0.1, 0.05]
    found = loop.candidates_from_gradient(theta, slope, narrowed, SETTINGS)
    assert [(item.direction, item.thickness_m) for item in found] == [("thicker", 0.2)]
    every = loop.candidates_from_gradient(theta, slope, full, SETTINGS)
    assert sorted((item.direction, item.thickness_m) for item in every) == [("thicker", 0.2), ("thinner", 0.05)]
    assert search.snap_to_specs(np.asarray([0.13, 0.03]), narrowed).theta.tolist() == [0.1, 0.03]
    assert search.snap_to_specs(np.asarray([0.16, 0.03]), narrowed).theta.tolist() == [0.2, 0.03]
    with pytest.raises(ValueError, match="超出盒界"):
        search.snap_to_specs(np.asarray([0.05, 0.03]), narrowed)


# ---- 終點：終點用的設計照範圍收窄 ----

class _Reached(Exception):
    def __init__(self, designs: tuple[search.SurfaceDesign, ...], theta: search.FloatVector) -> None:
        super().__init__("到了終點求解那一步")
        self.designs, self.theta = designs, theta


def _endpoint_designs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project: run.GradientProject,
                      final: tuple[tuple[str, str | None], ...], theta: list[float]) -> tuple[_Reached, list[Record]]:
    """真的 ``endpoint_run.run_endpoint``：讀檔、核身分、建完整模型、評分常數與核結果檔都換成樁（這一題只看設計），
    走到交給終點求解（``_endpoint_for``）那一步就停，回它拿到的設計與 θ、以及重建帳本標頭時給的範圍。"""
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    (tmp_path / run.RESULT_FILE).write_text("{}", encoding="utf-8")
    given = SimpleNamespace(format_version=run.RESULT_VERSION, theta_hex=tuple(value.hex() for value in theta),
                            surfaces=tuple(SimpleNamespace(surface_id=face, preset_id=preset) for face, preset in final))
    headers: list[Record] = []

    def header(model: object, designs: tuple[search.SurfaceDesign, ...], settings: object, **fields: object) -> str:
        headers.append({"ranges": fields.get("ranges"), "x0": _box(designs[2])})
        return "header"

    def reached(_project: object, _scheme: object, _model: object, _settings: object, designs: tuple[search.SurfaceDesign, ...],
                values: search.FloatVector, *args: object, **kwargs: object) -> NoReturn:
        raise _Reached(designs, values)

    operators = SimpleNamespace(mesh_hash=project.mesh_hash, physics_identity=project.physics_identity)
    for name, value in {"load_capabilities": lambda path: None, "load_directivity_defaults": lambda path: None,
                        "read_result": lambda folder: given, "check_inputs": lambda *a, **kw: ("scheme", "model", SETTINGS),
                        "_require_low_frequency_axis": lambda scheme: None, "scene_wall_materials": lambda *a, **kw: _kept(),
                        "export_operators": lambda *a, **kw: operators, "check_operators_digest": lambda *a: None,
                        "_ordered_operators": lambda ops, model: ops, "scoring_context": lambda *a, **kw: SimpleNamespace(constants="six"),
                        "header_for": header, "checked_result": lambda *a, **kw: given, "_endpoint_for": reached}.items():
        monkeypatch.setattr(endpoint_run, name, value)
    with pytest.raises(_Reached) as caught:
        endpoint_run.run_endpoint(tmp_path)
    return caught.value, headers


def test_zone_range_search_endpoint_designs_only_reach_the_cut_thicknesses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓終點不看範圍：最後 x0 換成 24K（範圍棉 0.1～0.2），終點用的設計合法厚度只剩 0.1、0.2；帳本標頭照展開過的範圍重建；
    θ 在範圍外（0.05）就不是可行解（終點入口照 ``is_feasible`` 拒收）。"""
    ranges = {"x0": {WOOL: ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.02, 0.08)), WOOL24: ZoneRange(porous_m=(0.1, 0.2))}}
    presets: tuple[str | None, ...] = (PRESET, None, PRESET, None, None, None)
    project = _project().model_copy(update={"presets": presets, "ranges": ranges})
    final = (("floor", PRESET), ("ceiling", None), ("x0", "glass_wool_24k_50mm"), ("xL", None), ("y0", None), ("yL", None))
    reached, headers = _endpoint_designs(tmp_path, monkeypatch, project, final, [0.05, 0.05, 0.1])
    assert _box(reached.designs[2]) == ((0.1,), (0.2,), (0.1, 0.2)) and reached.designs[2].family_id == WOOL24
    assert _box(reached.designs[0]) == ((0.05, 0.0), (0.2, 0.25), (0.05, 0.1, 0.2))
    assert headers == [{"ranges": ranges, "x0": ((0.05, 0.02), (0.1, 0.08), (0.05, 0.1))}]
    assert search.is_feasible(reached.theta, reached.designs)
    with pytest.raises(ValueError, match="超出盒界"):
        search.is_feasible(np.asarray([0.05, 0.05, 0.05]), reached.designs)


# ---- 左右對稱：成對兩區的範圍要完全相同 ----

def test_zone_range_search_tied_zones_with_different_ranges_are_refused() -> None:
    """抓綁定只比 θ 上下界：前牆兩區起跑那一族（48K）範圍相同、只有 24K 的範圍不同，θ 上下界一樣、照樣拒收（「可調範圍」）；
    48K 的空腔不同時 θ 上下界那一條先擋。網頁配對那一對算沒配成、原因是白話。"""
    same = {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.08]}
    document = _ranged({"x0@b": {"ranges": {WOOL: same, WOOL24: {"porous_m": [0.05, 0.1]}}},
                        "x0@a": {"ranges": {WOOL: same, WOOL24: {"porous_m": [0.05, 0.2]}}}})
    plan = control_plan(document)
    assert plan.ties == (SIDES,)
    with pytest.raises(TieRefused, match="可調範圍不同") as refused:
        start_designs(plan.model_copy(update={"ties": (FRONT, SIDES)}))
    assert refused.value.what == "可調範圍"
    found = plan_symmetry(control_scheme(), DesignSettings.model_validate(document), plan, SETTINGS)
    missed = [item for item in found.missed if (item.first, item.second) == FRONT]
    assert len(missed) == 1 and "兩區可調的範圍不一樣" in missed[0].reason
    bounds = _ranged({"x0@b": {"ranges": {WOOL: same}}, "x0@a": {"ranges": {WOOL: {**same, "cavity_m": [0.0, 0.06]}}}})
    with pytest.raises(TieRefused) as earlier:
        start_designs(control_plan(bounds).model_copy(update={"ties": (FRONT, SIDES)}))
    assert earlier.value.what == "θ 上下界"
    paired = _ranged({"x0@b": {"ranges": {WOOL: same}}, "x0@a": {"ranges": {WOOL: same}}})
    assert control_plan(paired).ties == (FRONT, SIDES)


# ---- 替身：範圍只收窄、家族代號不變，不用重建 ----

def test_zone_range_search_ranged_run_passes_the_launch_gate_without_a_rebuild(tmp_path: Path) -> None:
    """網頁開跑整段：寫了範圍的那一場，替身在格子裡（同一份沒寫範圍也開得了的那一顆）照樣開得了跑，不要求重建；
    建替身核固定材料那一份配置跟沒寫範圍逐位相同，交給命令列的參數多一個 ``--range``。預覽帶出實際用的範圍。"""
    from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _data, _jobs, _scheme
    from aosr.gui.gradient_plan import GradientRequest

    data_dir = _data(tmp_path)
    zones = {"x0": {"r0_c0": {"mode": "search", "ranges": {WOOL: {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.08]}}},
                    "r1_c1": {"mode": "search"}}, "ceiling": {"r0_c1": {"mode": "keep_original"}}}
    ranged = DesignSettings.model_validate({"scheme_id": NAME, "zones": zones})
    plain = ranged.model_copy(update={"zones": {**zones, "x0": {"r0_c0": {"mode": "search"}, "r1_c1": {"mode": "search"}}}})
    plain = DesignSettings.model_validate(plain.model_dump(mode="json"))
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    request = GradientRequest(max_steps=5)
    configurations = []
    for design in (plain, ranged):
        write_design(data_dir, design, _scheme())
        preview = jobs.preview(NAME, request)
        assert cast(Record, preview["surrogate"])["found"] is True
        _, scheme, saved = jobs.inputs(NAME)
        found, _, _ = jobs.plan(scheme, saved, request)
        configurations.append((configuration_arguments(found), plan_configuration(found, scheme)))
    assert configurations[0] == configurations[1]
    assert preview["ranges"] == {"x0@r0_c0": {WOOL: {"porous_m": [0.05, 0.1], "thicknesses_m": [0.05, 0.1], "cavity_m": [0.0, 0.08]}}}
    jobs.start(NAME, request)
    (_, argv, _, _), = launcher.launched
    assert argv[argv.index("--range") + 1] == "x0@r0_c0=glass_wool_48k:0.05,0.1:0.0,0.08"


# ---- 身分：沒寫範圍逐位不變；寫了範圍進身分 ----

def test_zone_range_search_ranges_enter_identity_in_a_fixed_order() -> None:
    """抓範圍沒進資料夾名、或存法跟著輸入順序走：命令列寫法讀回逐位相同；換順序同一個名；換一份範圍另一個名。"""
    first = faces.parse_ranges(["xL=glass_wool_24k:0.1,0.2", "x0=glass_wool_48k:0.05,0.1:0.0,0.08", "x0=glass_wool_96k:0.05,0.05"])
    second = faces.parse_ranges(["x0=glass_wool_96k:0.05,0.05", "x0=glass_wool_48k:0.05,0.1:0.0,0.08", "xL=glass_wool_24k:0.1,0.2"])
    assert first == second and list(first) == ["x0", "xL"] and list(first["x0"]) == [WOOL, "glass_wool_96k"]
    assert faces.parse_ranges([faces.range_argument("x0", WOOL, first["x0"][WOOL])]) == {"x0": {WOOL: first["x0"][WOOL]}}
    base: Record = {"scheme_sha256": "a" * 64, "surrogate_manifest_sha256": "b" * 64, "presets": (PRESET,) * 6,
                    "purpose": "music", "settings_fingerprint": "c" * 64, "physics_identity": "phys",
                    "objective_settings_fingerprint": "d" * 64}
    plain = run.gradient_id(**base)  # type: ignore[arg-type]
    assert run.gradient_id(**base, ranges={}) == plain  # type: ignore[arg-type]
    named = run.gradient_id(**base, ranges=first)  # type: ignore[arg-type]
    assert named != plain and run.gradient_id(**base, ranges=second) == named  # type: ignore[arg-type]
    raw = {"xL": {WOOL24: first["xL"][WOOL24]}, "x0": {"glass_wool_96k": first["x0"]["glass_wool_96k"], WOOL: first["x0"][WOOL]}}
    assert list(raw) == ["xL", "x0"] and run.gradient_id(**base, ranges=raw) == named  # type: ignore[arg-type]
    other = faces.parse_ranges(["xL=glass_wool_24k:0.1,0.2", "x0=glass_wool_48k:0.05,0.1:0.0,0.06", "x0=glass_wool_96k:0.05,0.05"])
    assert run.gradient_id(**base, ranges=other) != named  # type: ignore[arg-type]
    project = _project().model_copy(update={"ranges": first})
    assert run.GradientProject.model_validate_json(project.model_dump_json()) == project
    assert "ranges" not in _project().model_dump_json()
    with pytest.raises(ValueError, match="同一面"):
        faces.parse_ranges(["x0=glass_wool_48k:0.05,0.1", "x0=glass_wool_48k:0.05,0.2"])
    one = ZoneRange(porous_m=(0.05, 0.1))
    with pytest.raises(ValueError, match="同時給牆名與面片代號：x0"):
        faces.checked_ranges({"x0": {WOOL: one}, "x0@a": {WOOL: one}})
    with pytest.raises(ValueError, match="面 x0 的可調範圍是空的"):
        faces.checked_ranges({"x0": {}})
    with pytest.raises(ValueError, match="只給可最佳化材料（家族代號），收到 \\['none'\\]"):
        faces.checked_ranges({"x0": {"none": one}})
    with pytest.raises(ValueError, match="可調範圍的面 x0@q 不在搜尋面片清單"):
        faces.expanded_ranges({"x0@q": {WOOL: one}}, ["x0@a", "x0@b"])
    assert faces.expanded_ranges({"x0": {WOOL: one}}, ["floor", "x0@a", "x0@b"]) == {"x0@a": {WOOL: one}, "x0@b": {WOOL: one}}


def test_zone_range_search_ledger_headers_bind_the_ranges(tmp_path: Path) -> None:
    """抓帳本標頭沒記範圍（接續換一份範圍照收）：庚二標頭記展開到面片的範圍、換樣板標頭抄它；換一份範圍讀回拒收；
    沒有範圍的標頭不寫這一格。"""
    model = cast(ReducedModel, SimpleNamespace(input_hash="input-x", mesh_hash="mesh-x", physics_identity="physics-x",
                                              patch_keys=boundary_patches(), matrices=SimpleNamespace(boundary=(None,) * 6)))
    constants = gradient_objective_constants(np.asarray(FEM_LANE_FREQUENCIES_HZ), _gradient_settings(), _receivers(), 2)
    mine = {"x0": {WOOL: ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.0, 0.08))}}
    other = {"x0": {WOOL: ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.0, 0.06))}}
    headers = []
    for ranges in (mine, other, {}):
        designs = run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept(), ranges=ranges)
        header = six_header_for(model, designs, SETTINGS, objective_constants=constants).model_copy(
            update={"ranges": faces.expanded_ranges(ranges, [design.surface_id for design in designs])})
        headers.append(header)
    built = ledger.header_for(model, run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept(), ranges=mine), SETTINGS,
                              objective_constants=constants, ranges=mine)
    assert built.ranges == mine and "ranges" not in ledger.header_for(
        model, run.designs_for((PRESET,) * 6, SETTINGS, kept=_kept()), SETTINGS, objective_constants=constants).model_dump()
    path = tmp_path / run.LEDGER_FILE
    ledger.write_ledger(path, headers[0], ())
    assert ledger.read_ledger(path, expected_header=headers[0])[0].ranges == mine
    with pytest.raises(ValueError, match="標頭.*對不上"):
        ledger.read_ledger(path, expected_header=headers[1])
    template = run.template_header_for(headers[0], SETTINGS, _kept(), base_ledger_sha256="e" * 64)
    assert template.ranges == mine and "ranges" not in run.template_header_for(
        headers[2], SETTINGS, _kept(), base_ledger_sha256="e" * 64).model_dump()
    with pytest.raises(ValueError, match="可調範圍的面要在標頭的面裡"):
        ledger.MaterialGradientHeader.model_validate({**headers[0].model_dump(), "ranges": {"x0@a": mine["x0"]}})
    with pytest.raises(ValueError, match="可調範圍的面"):
        template.model_copy(update={"fixed_surfaces": {"x0": "glass_wool_24k_50mm"}}).model_validate(
            {**template.model_dump(), "fixed_surfaces": {"x0": "glass_wool_24k_50mm"}})


# ---- 真的命令列開跑與接續（雙精度子行程，替身是自編小房間那一顆） ----

def _start(folder: Path, model: ReducedModel, request: pytest.FixtureRequest, *ranges: str) -> subprocess.CompletedProcess[str]:
    """真的 ``cli.main(["start", …])``：替身與收據照替身沿用考卷的寫法放好，寫完 ``project.json`` 就停。
    起跑樣板 地板、x0 兩面是帶空腔的 48K（0.05／0.05）。"""
    from aosr.physics import fem_reduced_store as store
    from aosr.physics.fem_reduced_receipt import save_validation_receipt
    from tests.engine.test_material_gradient import _chain_model
    from tests.engine.test_material_gradient_cli import _env, _good_report, _schemes

    chain = _chain_model(model)
    folder.mkdir()
    (folder / "scheme.json").write_text(_schemes()["scheme.json"].model_dump_json(), encoding="utf-8")
    store.save_reduced(chain, folder / "rom.npz")
    save_validation_receipt(folder / "rom.npz", _good_report(request, chain), rulers=run.holdout_rulers(), seconds=1.0)
    arguments = [item for given in ranges for item in ("--range", given)]
    return subprocess.run([sys.executable, "-m", "tests.engine._material_gradient_carryover_probe", "same", str(folder),
                           *arguments], env=forward_cpu_pins(_env(folder)), capture_output=True, text=True, timeout=300)


def _probe(folder: Path, mode: str) -> Record:
    from tests.engine.test_material_gradient_cli import _env

    done = subprocess.run([sys.executable, "-m", "tests.engine._zone_range_probe", mode, str(folder)],
                          env=forward_cpu_pins(_env(folder)), capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return cast(Record, json.loads((folder / "probe.json").read_text(encoding="utf-8")))


def test_zone_range_search_command_line_start_and_resume_bind_the_ranges(
        model: ReducedModel, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """真的開跑一次（同一顆替身，沒寫範圍的那一場也開得了：不用重建替身）：範圍進 ``project.json``、資料夾名跟不寫的不同；
    照 ``project.json`` 重建問題時帳本標頭、換樣板那一側、x0 的起跑界都照範圍；寫下帳本標頭之後把 ``project.json`` 的範圍
    換一份，真的接續入口拒收（標頭對不上），換回原來那份照收；起跑值不在範圍的開跑非零離開、不留 ``project.json``。"""
    given = "x0=glass_wool_48k:0.05,0.1:0.0,0.08"
    plain, ranged = _start(tmp_path / "plain", model, request), _start(tmp_path / "ranged", model, request, given)
    assert plain.returncode == 0 and ranged.returncode == 0, plain.stderr + ranged.stderr
    before, after = run.read_project(tmp_path / "plain"), run.read_project(tmp_path / "ranged")
    assert "ranges" not in json.loads((tmp_path / "plain" / run.PROJECT_FILE).read_bytes())
    assert after.ranges == faces.parse_ranges([given]) and after.gradient_id != before.gradient_id
    saved = {"x0": {WOOL: {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.08]}}}
    assert json.loads((tmp_path / "ranged" / run.PROJECT_FILE).read_bytes())["ranges"] == saved
    assert _probe(tmp_path / "ranged", "problem") == {"header": saved, "templates": saved,
                                                       "x0": [[0.05, 0.0], [0.1, 0.08], [0.05, 0.1]]}
    assert _probe(tmp_path / "plain", "problem") == {"header": {}, "templates": {}, "x0": [[0.05, 0.0], [0.2, 0.25], [0.05, 0.1, 0.2]]}
    _probe(tmp_path / "ranged", "ledger")
    assert _probe(tmp_path / "ranged", "same") == {"error": None}
    document = json.loads((tmp_path / "ranged" / run.PROJECT_FILE).read_bytes())
    changed = {**document, "ranges": {"x0": {WOOL: {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.06]}}}}
    (tmp_path / "ranged" / run.PROJECT_FILE).write_text(json.dumps(changed), encoding="utf-8")
    refused = _probe(tmp_path / "ranged", "resume")
    assert refused == {"error": "帳本標頭跟這一次對不上：['ranges']"}
    (tmp_path / "ranged" / run.PROJECT_FILE).write_text(json.dumps(document), encoding="utf-8")
    assert _probe(tmp_path / "ranged", "same") == {"error": None}
    outside = _start(tmp_path / "outside", model, request, "x0=glass_wool_48k:0.1,0.2")
    assert outside.returncode != 0 and "起跑值棉厚 0.05 公尺不在這一區的範圍 0.1～0.2 公尺內" in outside.stderr
    assert not (tmp_path / "outside" / run.PROJECT_FILE).exists()


# ---- 控制組：沒寫範圍的網頁預覽回覆沒有這一格 ----

def test_zone_range_search_plan_without_ranges_has_no_ranges_anywhere() -> None:
    """沒寫範圍：計畫沒有這一格、開跑參數沒有 ``--range``、起跑設計跟不給範圍逐位相同（位元組控制組在 ``test_zone_range``）。"""
    plan = control_plan()
    assert plan.ranges == {} and "ranges" not in plan.model_dump_json() and "--range" not in start_arguments(plan)
    assert repr(start_designs(plan)) == repr(cli.start_designs(
        control_scheme(), SETTINGS, run.presets_table(), plan.presets, mount=plan.request.mount, fixed=dict(plan.fixed_surfaces),
        caps=dict(plan.max_depths), allowed=dict(plan.allowed), capabilities=load_capabilities(config_path("capabilities.toml")),
        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")), ties=plan.ties))


# ---- 深度：分區頁標紅與命令列開跑同一把 ----

@pytest.mark.parametrize("listed", [True, False], ids=["allowed-list", "no-list"])
def test_zone_range_search_zone_page_red_and_engine_refuse_the_same_depth_case(listed: bool) -> None:
    """抓兩邊判法分家：左牆 s 的 48K 收窄到棉 0.1～0.2、空腔 0.05 起，最淺 0.1＋0.05；上限 0.15 照搜尋那一把的浮點放不下、
    0.16 放得下。分區頁標紅（點名哪一種材料）與命令列開跑核可選目標（``check_depth_targets``，點名面與材料）在同一個邊上
    一起紅、一起不紅；有准用清單與沒有清單（可選的是全部家族）都一樣。"""
    from tests.engine.test_zone_range import _limits, _red

    given = ZoneRange(porous_m=(0.1, 0.2), cavity_m=(0.05, 0.2))
    zone: Record = {"ranges": {WOOL: given.model_dump(mode="json")}, **({"allowed": [WOOL, "none"]} if listed else {})}
    allowed = {"y0": (WOOL, "none")} if listed else {}
    for cap, red in ((0.15, True), (0.16, False)):
        page = _red(_limits(_ranged({"y0@s": {**zone, "max_depth_m": cap}})), "y0", "s")
        try:
            faces.check_depth_targets([("y0", None)], SETTINGS, fixed_surfaces={}, max_depths={"y0": cap},
                                      allowed=allowed, fixed_materials={}, ranges={"y0": {WOOL: given}})
        except ValueError as error:
            engine: str | None = str(error)
        else:
            engine = None
        assert bool(page) is red and (engine is not None) is red, (cap, page, engine)
        if red:
            ranged = [line for line in page if line["code"] == "range_depth"]
            assert len(ranged) == 1 and "48K 玻璃棉" in str(ranged[0]["text"]) and "收窄" in str(ranged[0]["text"])
            assert engine is not None and engine.startswith("面 y0 照這一面的可調範圍收窄後：") and WOOL in engine
