"""換樣板的候選確認迴圈（#561 庚三）：可換的目標、換一面、快速評分與取前幾名、一輪的接受規則、最壞求值數、
五種停止原因、換樣板帳本的寫讀與拒收、每一個列界的接續、重跑逐位相同。

合成目標（探針裡定義：每面加權二次式＋「哪一面換成哪個目標」的偏移表）求值便宜，給列界接續、停止原因、門檻邊界、
求值次數與停止查詢次數用；小房間替身照 ``test_material_gradient`` 衍生的那一份（六面 two_rigid、從預設樣板取規格起跑），
雙精度子行程裡對每個配置重建 ``chain_objective`` 跑整段。雙精度的計算與帳本在子行程（只叫公開入口）；比對與判決在父行程。
可換目標、換一面、最壞求值數、「沒有可換的樣板」與帳本竄改不必求值，父行程直接跑。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
import pytest

from aosr.config.construction_presets import ConstructionPreset
from aosr.config.material_gradient import MaterialGradientSettings
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_store import save_reduced
from aosr.runtime import child_process_env, configure_jax
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_loop as loop
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.gradient import template_segment as segment
from aosr.gradient.endpoint import material_digest
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine._scheme_cache import shared_json
from tests.engine._material_gradient_templates_probe import kept_walls, synthetic_header
from tests.engine.test_material_gradient import CASES, DESCENT_CASE, PURPOSE, _chain_model, _presets, _receivers, _settings
from tests.engine._gradient_objective_rows import assert_rows_split_scores
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS


PROBE = Path(__file__).with_name("_material_gradient_templates_probe.py")
Record = dict[str, object]
GAP50, MM50 = "glass_wool_48k_50mm_gap50", "glass_wool_48k_50mm"
# 六面照 Wall.all()（floor、ceiling、x0、xL、y0、yL）：地板不貼、其他五面 48K 棉板＋5 cm 空腔。
START: list[list[str | None]] = [["floor", None], ["ceiling", GAP50], ["x0", GAP50], ["xL", GAP50], ["y0", GAP50],
                                 ["yL", GAP50]]
KEPT = [20000.0, 21000.0, 22000.0, 23000.0, 24000.0, 25000.0]
# 偏移表：x0 換 24K 一定好 1、天花板不貼一定好 0.6；其餘的換法只差二次式那幾項（遠小於門檻）。
OFFSETS = {"x0->glass_wool_24k": -1.0, "ceiling->none": -0.6}
SMALL_ROOM_KEPT = [31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0]


def _trials(settings: MaterialGradientSettings) -> int:
    descent = settings.descent
    return sum(1 for power in range(200) if descent.initial_step * descent.backtrack_ratio ** power >= descent.min_step)


def _hand_bound(settings: MaterialGradientSettings, quick: int) -> int:
    """考卷自己數的一輪最壞求值次數：快速評分每個一次；基準與前 min(N, 快速評分數) 個候選各一段重調，
    每段起點 1 次＋K 步 × 每步最多試探數。"""
    if quick == 0:
        return 0
    return quick + (min(settings.templates.max_per_round, quick) + 1) * (1 + settings.retune.steps * _trials(settings))


def _problem() -> Record:
    bound = _hand_bound(_settings(), 18)
    cases: dict[str, Record] = {
        "main": {"override": {}, "again": True, "resume_every": True, "stops": True, "seconds": True,
                 "single_round": True},
        "max_rounds": {"override": {"templates": {"max_rounds": 1}}},
        "budget_fit": {"override": {"templates": {"max_evaluations": bound}}},
        "budget_closed": {"override": {"templates": {"max_evaluations": bound - 1}}},
    }
    return {"synthetic": {"surfaces": START, "kept": KEPT, "offsets": OFFSETS}, "cases": cases,
            "threshold_override": {},
            "surrogate": {"override": {"templates": {"max_rounds": 2}}, "kept": SMALL_ROOM_KEPT,
                          "presets": list(CASES[DESCENT_CASE])},
            "purpose": PURPOSE, "receivers": _receivers().model_dump(mode="json")}


def _compute(directory: Path, request: pytest.FixtureRequest) -> str:
    """建衍生替身、跑探針，回探針輸出（JSON 文字）。目錄先清空：帳本是獨佔建檔，上一次留下的半份不能被接著用。"""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    clock = time.perf_counter()
    save_reduced(_chain_model(cast(ReducedModel, request.getfixturevalue("model"))), directory / "model.npz")
    model_seconds = time.perf_counter() - clock
    (directory / "problem.json").write_text(json.dumps(_problem()), encoding="utf-8")
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    clock = time.perf_counter()
    process = subprocess.run([sys.executable, str(PROBE), str(directory)], cwd=directory, env=env,
                             capture_output=True, text=True, timeout=1500)
    assert process.returncode == 0, process.stdout + process.stderr
    output = cast(Record, json.loads((directory / "output.json").read_text(encoding="utf-8")))
    output["parent_seconds"] = {"model": model_seconds, "probe": time.perf_counter() - clock}
    text = json.dumps(output)
    (directory / "fixture.json").write_text(text, encoding="utf-8")
    return text


@pytest.fixture(scope="module")
def runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
         ) -> tuple[Path, Record]:
    """一場 pytest 只建一次替身、跑一次探針，平行的工人共用（``_scheme_cache.shared_json``）；讀的人只讀。"""
    directory = (tmp_path_factory.mktemp("material-gradient-templates") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "material-gradient-templates-shared")
    text = shared_json(tmp_path_factory, worker_id, "material-gradient-templates", lambda: _compute(directory, request))
    return directory, cast(Record, json.loads(text))


def _case(runs: tuple[Path, Record], name: str) -> Record:
    return cast(Record, cast(Record, runs[1]["synthetic"])[name])


def _read(runs: tuple[Path, Record], name: object) -> tuple[templates.TemplateHeader, tuple[templates.TemplateRow, ...]]:
    return templates.read_template_ledger(runs[0] / str(name))


def _rows(runs: tuple[Path, Record], name: object) -> tuple[templates.TemplateRow, ...]:
    return _read(runs, name)[1]


def _without_seconds(rows: Sequence[templates.TemplateRow]) -> list[Record]:
    return [{key: value for key, value in row.model_dump(mode="json").items() if key != "seconds"} for row in rows]


def _surfaces(value: object) -> rounds.Surfaces:
    return tuple((str(item[0]), item[1]) for item in cast(list[list[str | None]], value))


def _start(case: Record) -> templates.TemplateStart:
    start = cast(Record, case["start"])
    first, second, proxy = (float.fromhex(value) for value in cast(list[str], start["components"]))
    return templates.TemplateStart(_surfaces(start["surfaces"]), tuple(cast(list[str], start["theta_hex"])),
                                   float.fromhex(str(start["value"])), float.fromhex(str(start["screening_total"])),
                                   (first, second, proxy))


def _context(settings: MaterialGradientSettings | None = None) -> rounds.TemplateContext:
    """不求值的題用：目標建得出來（整段開頭會為起點配置建一份），一叫就失敗。"""
    def never(_: tuple[gradient_search.SurfaceDesign, ...]) -> gradient_search.ChainObjective:
        def objective(theta: object) -> NoReturn:
            raise AssertionError("這一題不該求值")
        return objective
    return rounds.TemplateContext(settings or _settings(), _presets(), kept_walls(KEPT), never)


def _blocks(rows: Sequence[templates.TemplateRow]) -> list[list[templates.TemplateRow]]:
    found: dict[int, list[templates.TemplateRow]] = {}
    for row in rows:
        found.setdefault(row.round, []).append(row)
    return [found[key] for key in sorted(found)]


# ---- 可換的目標、換一面、最壞求值數（不求值，父行程） ----

def test_alternatives_are_other_families_then_none_in_surface_order() -> None:
    """抓漏了「不貼」、沒扣掉自己的家族、不貼的面還換不貼、順序不照面序再目標序（家族照設定檔、不貼最後）。"""
    settings = _settings()
    families = [family.id for family in settings.family]
    assert families == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k"]
    surfaces = (("floor", None), ("ceiling", GAP50), ("x0", "glass_wool_24k_50mm"), ("xL", "glass_wool_96k_50mm"),
                ("y0", MM50), ("yL", None))
    expected = [("floor", "glass_wool_48k"), ("floor", "glass_wool_24k"), ("floor", "glass_wool_96k"),
                ("ceiling", "glass_wool_24k"), ("ceiling", "glass_wool_96k"), ("ceiling", "none"),
                ("x0", "glass_wool_48k"), ("x0", "glass_wool_96k"), ("x0", "none"),
                ("xL", "glass_wool_48k"), ("xL", "glass_wool_24k"), ("xL", "none"),
                ("y0", "glass_wool_24k"), ("y0", "glass_wool_96k"), ("y0", "none"),
                ("yL", "glass_wool_48k"), ("yL", "glass_wool_24k"), ("yL", "glass_wool_96k")]
    assert list(rounds.alternatives(surfaces, settings)) == expected


def test_family_id_none_is_reserved_for_keeping_the_wall() -> None:
    """抓家族叫 none 也收下（「不貼」的代號被家族佔走：可換目標漏掉那個家族也漏掉不貼、換過去變成不貼）：
    正式設定把第一個家族改名 none，重新驗證就拒收、訊息講明是保留代號。"""
    document = _settings().model_dump()
    document["family"][0]["id"] = "none"
    with pytest.raises(ValueError, match="保留"):
        MaterialGradientSettings.model_validate(document)


def test_target_order_has_no_duplicate_ids() -> None:
    """抓目標代號重複（家族代號撞「不貼」或彼此相撞）：正式設定的目標順序每個代號只出現一次、不貼在最後。"""
    order = rounds.target_order(_settings())
    assert len(set(order)) == len(order) and order[-1] == "none"


def _swap_start() -> tuple[tuple[gradient_search.SurfaceDesign, ...], np.ndarray]:
    context = _context()
    designs = rounds.designs_for_surfaces(_surfaces(START), context)
    theta = gradient_search.snap_to_specs(gradient_search.baseline_theta(designs), designs).theta
    return designs, np.asarray(theta, dtype=np.float64)


def _values(designs: Sequence[gradient_search.SurfaceDesign], theta: np.ndarray) -> dict[str, float]:
    return dict(zip(gradient_search.design_names(designs), (float(value) for value in theta), strict=True))


def test_swap_to_24k_has_no_cavity_and_copies_other_faces_bitwise() -> None:
    """抓換過去沒用家族的樣板與初值、24K 還帶空腔變數、其他面的值沒有逐位照搬、新點不可行。"""
    settings, context = _settings(), _context()
    designs, theta = _swap_start()
    family = settings.family_by_id("glass_wool_24k")
    new_designs, new_theta = rounds.swap(designs, theta, "x0", "glass_wool_24k", context)
    swapped = next(design for design in new_designs if design.surface_id == "x0")
    assert swapped.preset_id == family.template_preset and swapped.cavity_variable is None
    assert len(swapped.variables) == 1 and family.initial_cavity_m is None
    before, after = _values(designs, theta), _values(new_designs, new_theta)
    assert after.pop(f"x0.{swapped.variables[0]}") == family.initial_thickness_m
    assert after == {name: value for name, value in before.items() if not name.startswith("x0.")}
    assert gradient_search.is_feasible(new_theta, new_designs) and new_theta.dtype == theta.dtype


def test_swap_to_none_keeps_scheme_wall_and_none_to_family_uses_initial_values() -> None:
    """抓「不貼」被當成剛性（沒帶方案原本那面的牆）、不貼的面換回家族沒用家族的空腔初值、換成自己沒拒收。"""
    settings, context = _settings(), _context()
    designs, theta = _swap_start()
    to_none, none_theta = rounds.swap(designs, theta, "ceiling", "none", context)
    ceiling = next(design for design in to_none if design.surface_id == "ceiling")
    assert ceiling.preset_id is None and not ceiling.variables
    assert ceiling.kept_material is not None
    assert material_digest(ceiling.kept_material) == material_digest(kept_walls(KEPT)["ceiling"])
    assert len(none_theta) == len(theta) - 2
    family = settings.family_by_id("glass_wool_48k")
    to_48k, theta_48k = rounds.swap(designs, theta, "floor", "glass_wool_48k", context)
    floor = next(design for design in to_48k if design.surface_id == "floor")
    assert floor.preset_id == family.template_preset and floor.cavity_variable is not None
    assert [value for name, value in _values(to_48k, theta_48k).items() if name.startswith("floor.")] == [
        family.initial_thickness_m, family.initial_cavity_m]
    with pytest.raises(ValueError, match="換成自己不是替換"):
        rounds.swap(designs, theta, "x0", "glass_wool_48k", context)


def test_round_bound_matches_hand_count_and_shares_retune_bound() -> None:
    """抓最壞求值數少算基準那一段、進重調數沒跟快速評分數取小、跟庚二各算各的每段重調最壞值。"""
    settings = _settings()
    per_retune = 1 + settings.retune.steps * _trials(settings)
    assert loop.retune_evaluation_bound(settings) == per_retune
    assert loop.round_evaluation_bound(settings, 3) == 1 + 4 * per_retune
    for quick in (0, 1, 2, 3, 18):
        assert rounds.template_round_bound(settings, quick, rounds.retune_count(settings, quick)) == _hand_bound(
            settings, quick)
    assert rounds.next_round_bound(_surfaces(START), settings) == _hand_bound(settings, 18)
    for quick, count in ((3, 0), (3, 4), (0, 1)):
        with pytest.raises(ValueError):
            rounds.template_round_bound(settings, quick, count)


def test_no_alternatives_round_is_not_opened_and_ledger_says_so(tmp_path: Path) -> None:
    """抓沒有可換的還開輪（或求值）、停止原因不是「沒有可換的樣板」、記了下一輪最壞值。設定檔家族至少一個（載入器擋），
    所以這一題用沒有家族的設定（model_copy 跳過驗證）＋六面不貼：沒有任何可換的目標。"""
    settings = _settings().model_copy(update={"family": ()})
    context = _context(settings)
    surfaces = tuple((str(item[0]), None) for item in START)
    start = templates.TemplateStart(surfaces, (), 1.5, 1.25, (1.0, 2.0, 0.5))
    header = synthetic_header(settings, surfaces, kept_walls(KEPT))
    result = segment.run_template_rounds(context, tmp_path / "t.jsonl", header, start)
    assert (result.stop_reason, result.rounds, result.evaluations) == (rounds.TEMPLATE_STOP_NO_ALTERNATIVES, 1, 0)
    rows = templates.read_template_ledger(tmp_path / "t.jsonl", settings=settings, start=start)[1]
    assert len(rows) == 1 and rows[0].decision is not None
    assert (rows[0].decision.quick_count, rows[0].decision.next_round_bound, rows[0].value) == (0, None, 1.5)


# ---- 合成目標：一輪的內容 ----

def test_quick_scores_cover_every_alternative_and_top_n_are_retuned(runs: tuple[Path, Record]) -> None:
    """抓快速評分漏了哪個（面, 目標）、排序反了或同值沒照面序、進重調的不是前 N 名、重調超過 K 步、重調起點不是快速評分那一點。

    考卷從每一輪起點配置自己列可換目標（面數 × 每面可換數）、自己照（值, 位置）排序取前 min(N, 個數) 名。
    """
    settings = _settings()
    rows = _rows(runs, _case(runs, "main")["ledger"])
    surfaces = _surfaces(START)
    for block in _blocks(rows):
        quick = [row for row in block if row.phase == "quick_score"]
        wanted = rounds.alternatives(surfaces, settings)
        assert len(wanted) == 6 * 3
        assert [row.candidate for row in quick] == [rounds.candidate_id(*item) for item in wanted]
        order = sorted(range(len(quick)), key=lambda index: (quick[index].value, index))
        top = [quick[index].candidate for index in order[:min(settings.templates.max_per_round, len(quick))]]
        segments: dict[str, list[templates.TemplateRow]] = {}
        for row in block:
            if row.phase == "candidate_retune":
                segments.setdefault(str(row.candidate), []).append(row)
        assert list(segments) == top
        baseline = [row for row in block if row.phase == "baseline_retune"]
        for name, rows_of in (("baseline", baseline), *segments.items()):
            assert 1 <= len(rows_of) <= settings.retune.steps + 1, name
        for name, rows_of in segments.items():
            first = next(row for row in quick if row.candidate == name)
            assert (rows_of[0].surfaces, rows_of[0].theta_hex) == (first.surfaces, first.theta_hex)
        surfaces = block[-1].surfaces


def test_accepts_best_only_above_threshold_and_hands_over_retuned_point(runs: tuple[Path, Record]) -> None:
    """抓接受了沒超過門檻的、接受的不是改善最大的、交出去的是快速評分那一點（沒重調）、沒接受時交出沒重調的基準、
    改善算成反方向、下一輪不是從交出去的點開始。停止原因照字面。"""
    settings = _settings()
    rows = _rows(runs, _case(runs, "main")["ledger"])
    blocks = _blocks(rows)
    accepted = []
    for index, block in enumerate(blocks):
        closing = block[-1]
        decision = closing.decision
        assert decision is not None
        baseline = [row for row in block if row.phase == "baseline_retune"]
        assert decision.baseline_value == baseline[-1].value
        best = max(decision.outcomes, key=lambda item: item.improvement)
        for outcome in decision.outcomes:
            assert outcome.improvement == decision.baseline_value - outcome.value
        expected = best.candidate if best.improvement > settings.candidates.improvement_threshold else None
        assert decision.accepted == expected
        chosen = baseline[-1] if expected is None else [
            row for row in block if row.phase == "candidate_retune" and row.candidate == expected][-1]
        assert (closing.surfaces, closing.theta_hex, closing.value) == (chosen.surfaces, chosen.theta_hex, chosen.value)
        if index + 1 < len(blocks):
            nxt = [row for row in blocks[index + 1] if row.phase == "baseline_retune"][0]
            assert (nxt.surfaces, nxt.theta_hex) == (closing.surfaces, closing.theta_hex)
        accepted.append(decision.accepted)
    assert accepted == ["x0->glass_wool_24k", "ceiling->none", None]
    assert rows[-1].stop_reason == "在本輪已確認的單牆候選與計算預算內，未找到超過改善門檻的方案"


def test_swapped_face_without_cavity_is_marked_in_ledger(runs: tuple[Path, Record]) -> None:
    """抓帳本沒標「換過去那面無可重調變數」：快速評分列換到 24K、96K、不貼的是 false，換到 48K 的是 true。"""
    rows = _rows(runs, _case(runs, "main")["ledger"])
    for row in rows:
        if row.phase != "quick_score":
            assert row.swapped_retunable is None
            continue
        assert row.swapped_retunable is str(row.candidate).endswith("->glass_wool_48k"), row.candidate


def test_improvement_equal_to_threshold_is_not_accepted(runs: tuple[Path, Record]) -> None:
    """抓門檻用 ≥：同一輪最好的改善 X，門檻設 X 不接受（停在未找到超過門檻），設比 X 小一個浮點數就接受。"""
    edge = cast(Record, runs[1]["threshold_edge"])
    equal, below = cast(Record, edge["equal"]), cast(Record, edge["below"])
    assert equal["best"] == below["best"] == edge["best"] == equal["threshold"]
    assert (equal["accepted"], equal["stop_reason"]) == (None, rounds.TEMPLATE_STOP_NO_IMPROVEMENT)
    assert (below["accepted"], below["stop_reason"]) == ("x0->glass_wool_24k", None)


def test_evaluations_equal_objective_calls_and_stop_checks_within_bound(runs: tuple[Path, Record]) -> None:
    """抓帳本記的求值次數跟真的叫目標的次數不同、有求值之前沒查停止、一輪實際次數超過最壞值或最壞值算錯。"""
    main = _case(runs, "main")
    result = cast(Record, main["result"])
    rows = _rows(runs, main["ledger"])
    assert main["calls"] == main["checks"] == result["evaluations"]
    assert result["evaluations"] == sum(row.decision.round_evaluations for row in rows if row.decision is not None)
    single = cast(Record, main["single_round"])
    assert single["calls"] == single["evaluations"]
    assert cast(int, single["evaluations"]) <= cast(int, single["bound"]) == _hand_bound(_settings(), 18)
    assert (single["quick"], single["retuned"]) == (18, 3)


# ---- 五種停止原因（「沒有可換的樣板」在父行程那一題） ----

def test_stop_reason_max_rounds(runs: tuple[Path, Record]) -> None:
    """抓輪數上限沒停或多跑一輪：上限 1 輪，第 0 輪接受之後停，記「換樣板輪數到上限」、不記下一輪最壞值。"""
    rows = _rows(runs, _case(runs, "max_rounds")["ledger"])
    decisions = [row for row in rows if row.decision is not None]
    assert len(decisions) == 1 and rows[-1].stop_reason == rounds.TEMPLATE_STOP_MAX_ROUNDS
    assert decisions[0].decision is not None and decisions[0].decision.accepted == "x0->glass_wool_24k"
    assert decisions[0].decision.next_round_bound is None


def test_stop_reason_max_evaluations_after_a_round(runs: tuple[Path, Record]) -> None:
    """抓預算判法錯：上限恰等於一輪最壞值，第 0 輪開得了、跑完後累計加下一輪最壞值超過上限就停，記的數字對得上。"""
    rows = _rows(runs, _case(runs, "budget_fit")["ledger"])
    last = rows[-1]
    assert last.decision is not None and last.stop_reason == rounds.TEMPLATE_STOP_MAX_EVALUATIONS
    assert last.round == 0 and last.decision.accepted is not None
    limit = _hand_bound(_settings(), 18)
    assert last.decision.next_round_bound == limit
    assert last.decision.total_evaluations + last.decision.next_round_bound > limit >= last.decision.total_evaluations


def test_stop_reason_max_evaluations_before_opening(runs: tuple[Path, Record]) -> None:
    """抓預算放不下還開輪：上限比一輪最壞值少 1，第 0 輪不開、不求值、交出起點原樣，記「換樣板求值到上限」。"""
    case = _case(runs, "budget_closed")
    rows = _rows(runs, case["ledger"])
    start = _start(case)
    assert len(rows) == 1 and rows[0].decision is not None
    assert (rows[0].stop_reason, rows[0].decision.quick_count, rows[0].decision.total_evaluations) == (
        rounds.TEMPLATE_STOP_MAX_EVALUATIONS, 0, 0)
    assert (rows[0].surfaces, rows[0].theta_hex, rows[0].value) == (start.surfaces, start.theta_hex, start.value)
    assert case["calls"] == 0


def test_stop_reason_max_seconds_from_ledger_seconds(runs: tuple[Path, Record]) -> None:
    """抓總時間看牆鐘或只看最後一輪：截在第 1 輪半途、第 0 輪決定列秒數改成恰等於上限，接續重做第 1 輪後記
    「換樣板總時間到上限」（加總超過上限、恰等於不停）；改成超過上限：決定列沒停卻該停，接續拒收、帳本不動。"""
    seconds = cast(Record, _case(runs, "main")["seconds"])
    rows = _rows(runs, seconds["equal"])
    decisions = [row for row in rows if row.decision is not None]
    assert [row.round for row in decisions] == [0, 1]
    assert decisions[0].seconds == seconds["limit"] and decisions[0].stop_reason is None
    assert rows[-1].stop_reason == rounds.TEMPLATE_STOP_MAX_SECONDS
    assert templates.decisions_seconds(rows) > float(str(seconds["limit"]))
    assert "帳本與設定矛盾" in str(seconds["over_error"]) and seconds["over_unchanged"] is True


def test_stop_reason_no_improvement_hands_retuned_baseline(runs: tuple[Path, Record]) -> None:
    """抓沒有超過門檻時交出沒重調的起點、或停止原因用了別的字：最後一輪沒接受，交出基準重調末列，字面照老闆的話。"""
    rows = _rows(runs, _case(runs, "main")["ledger"])
    last = rows[-1]
    assert last.decision is not None and last.decision.accepted is None
    baseline = [row for row in _blocks(rows)[-1] if row.phase == "baseline_retune"][-1]
    assert (last.theta_hex, last.value) == (baseline.theta_hex, baseline.value)
    assert last.stop_reason == "在本輪已確認的單牆候選與計算預算內，未找到超過改善門檻的方案"
    assert "排除" not in str(last.stop_reason)


# ---- 帳本：寫讀、拒收、接續、重跑 ----

def test_ledger_reads_back_value_for_value_and_matches_result(runs: tuple[Path, Record]) -> None:
    """抓寫進去的跟讀回來的不同（θ 不是 float.hex、漏欄）、整段結果不是從帳本最後一列決定組的。"""
    main = _case(runs, "main")
    path = runs[0] / str(main["ledger"])
    header, rows = templates.read_template_ledger(path, settings=_settings(), start=_start(main))
    lines = path.read_bytes().splitlines()
    assert json.loads(lines[0]) == header.model_dump(mode="json")
    assert [json.loads(line) for line in lines[1:]] == [row.model_dump(mode="json") for row in rows]
    finished = segment.finished_result(rows)
    result = cast(Record, main["result"])
    assert finished is not None
    assert ([list(item) for item in finished.surfaces], [value.hex() for value in finished.theta]) == (
        result["surfaces"], result["theta"])
    assert (finished.stop_reason, finished.rounds, finished.evaluations) == (
        result["stop_reason"], result["rounds"], result["evaluations"])


def _tampered(runs: tuple[Path, Record], tmp_path: Path, edit: str) -> Path:
    path = runs[0] / str(_case(runs, "main")["ledger"])
    lines = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if edit == "surfaces":
        index = next(index for index, row in enumerate(rows[1:], start=1) if row["phase"] == "baseline_retune")
        rows[index]["surfaces"][0][1] = GAP50
    elif edit == "decision":
        index = next(index for index, row in enumerate(rows[1:], start=1) if row["phase"] == "template_decision")
        del rows[index]
        for step, row in enumerate(rows[1:]):
            row["step"] = step
    elif edit == "retunable":
        index = next(index for index, row in enumerate(rows[1:], start=1) if row["phase"] == "quick_score")
        rows[index]["swapped_retunable"] = not rows[index]["swapped_retunable"]
    target = tmp_path / f"{edit}.jsonl"
    target.write_bytes(b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows))
    return target


@pytest.mark.parametrize("edit", ["surfaces", "decision", "retunable"])
def test_tampered_ledger_is_rejected(runs: tuple[Path, Record], tmp_path: Path, edit: str) -> None:
    """抓讀回不驗：改一列的配置、刪掉第 0 輪決定列（後面重新連號）、改快速評分列的「有沒有可重調變數」，都要拒收。"""
    with pytest.raises(ValueError):
        templates.read_template_ledger(_tampered(runs, tmp_path, edit), settings=_settings(),
                                       start=_start(_case(runs, "main")))


@pytest.mark.parametrize("field", ["input_hash", "base_ledger_sha256"])
def test_ledger_with_other_header_is_rejected(runs: tuple[Path, Record], field: str) -> None:
    """抓接續不比標頭：這一次的標頭只差替身輸入雜湊、或只差庚二帳本摘要，讀回拒收並點名那一格。"""
    header = _read(runs, _case(runs, "main")["ledger"])[0]
    other = header.model_copy(update={field: hashlib.sha256(field.encode()).hexdigest()})
    with pytest.raises(ValueError, match=field):
        templates.read_template_ledger(runs[0] / str(_case(runs, "main")["ledger"]), expected_header=other)


def test_resume_at_every_row_boundary_matches_one_shot(runs: tuple[Path, Record]) -> None:
    """抓接續沒截掉半輪（列重複）、從錯的點接、累計次數或秒數沒接上：一口氣跑完的帳本在每一個列界（留 0～全部列）
    各截一份接續，扣秒數之後都逐值等於一口氣跑完的。"""
    main = _case(runs, "main")
    expected = _without_seconds(_rows(runs, main["ledger"]))
    resumed = cast(list[str], cast(Record, main["resume_every"])["resumed"])
    assert len(resumed) == len(expected) + 1
    for name in resumed:
        assert _without_seconds(_rows(runs, name)) == expected, name


def test_rerun_from_same_start_is_bitwise_identical(runs: tuple[Path, Record]) -> None:
    """抓有不確定性（集合序、牆鐘進決定）：同一起點再跑一次，扣秒數逐位相同。"""
    main = _case(runs, "main")
    assert _without_seconds(_rows(runs, main["again"])) == _without_seconds(_rows(runs, main["ledger"]))


def test_stop_leaves_whole_rounds_and_resume_matches_one_shot(runs: tuple[Path, Record]) -> None:
    """抓停下時寫了半輪、或沒在求值之前停：第 1 次、中間一次、最後一次查停止時叫停，都拋停止、帳本是參考的前綴
    且停在標頭或決定列；接續之後扣秒數逐值等於一口氣跑完的。"""
    main = _case(runs, "main")
    expected = _without_seconds(_rows(runs, main["ledger"]))
    stops = cast(list[Record], main["stops"])
    assert [stop["at"] for stop in stops] == sorted({1, cast(int, main["checks"]) // 2, cast(int, main["checks"])})
    for stop in stops:
        assert stop["stopped"] is True, stop["at"]
        stopped = _rows(runs, stop["stopped_ledger"])
        assert _without_seconds(stopped) == expected[:len(stopped)] and len(stopped) < len(expected), stop["at"]
        assert not stopped or stopped[-1].phase == "template_decision", stop["at"]
        assert _without_seconds(_rows(runs, stop["resumed"])) == expected, stop["at"]


# ---- 小房間替身（雙精度子行程） ----

def test_small_room_template_segment_runs_in_double_and_resumes(runs: tuple[Path, Record]) -> None:
    """抓替身路徑接不起來（每個配置重建 chain_objective、編譯快取撞）、不是雙精度、接續跟一口氣跑完不同。

    小房間 two_rigid 從預設樣板取規格那一點起跑、輪數上限 2：帳本照設定與起點讀得回、停止原因在封閉集合裡、
    截在第 0 輪基準重調第一列之後接續，扣秒數逐值相同。
    """
    case = cast(Record, runs[1]["surrogate"])
    assert case["x64"] == "float64"
    settings = _settings().model_copy(update={"templates": _settings().templates.model_copy(update={"max_rounds": 2})})
    origin, origin_rows = _start(case), _rows(runs, case["ledger"])
    assert origin.surfaces == tuple(zip(gradient_search.wall_surface_ids(), CASES[DESCENT_CASE], strict=True))
    templates.read_template_ledger(runs[0] / str(case["ledger"]), settings=settings, start=origin)
    assert origin_rows[-1].stop_reason in rounds.TEMPLATE_STOP_REASONS
    assert _without_seconds(_rows(runs, case["resumed"])) == _without_seconds(origin_rows)
    # #561 壬一：每一列的值是新合分＝舊合分 ＋ w·代理／ref，舊合分每列都在、可以單獨排序。
    assert_rows_split_scores(origin_rows)


def _preset_ids(presets: Sequence[ConstructionPreset]) -> set[str]:
    return {preset.id for preset in presets}


def test_family_templates_exist_in_presets() -> None:
    """抓家族的樣板 preset 不在構造預設清單（換過去建不出設計）。"""
    assert {family.template_preset for family in _settings().family} <= _preset_ids(_presets())
