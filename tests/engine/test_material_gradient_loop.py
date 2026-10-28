"""牆面材料梯度搜尋後半段（#561 庚二）：候選產生、一輪確認、整場停止原因、帳本 v2、接續、連續段的總深度投影、拒收。

替身照 ``test_material_gradient`` 從 ``test_fem_reduced`` 的小房間衍生（同一個 ``_chain_model``），六面 two_rigid、
從預設樣板取規格起跑、正式設定。另有合成目標（加權二次式，探針裡定義），求值便宜，給「每一個列界都接續一次」
與四種停止原因用。雙精度的迴圈、帳本與接續在子行程（只叫公開入口）；比對與判決在父行程。
候選產生、總深度投影、帳本竄改與拒收不必求值，父行程直接跑。單精度只量不登記。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import jax
import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.material_gradient import CandidateSettings, FamilySettings, MaterialGradientSettings
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_store import save_reduced
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.stage_one_objective_jax import StageOneObjectiveArrays
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import material_gradient_loop as loop
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine._scheme_cache import shared_json
from tests.engine.test_material_gradient import (
    CASES, DESCENT_CASE, PURPOSE, _chain_model, _designs, _presets, _receivers, _settings,
)
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS


PROBE = Path(__file__).with_name("_material_gradient_loop_probe.py")
Record = dict[str, object]
FOUR_SURFACES: list[list[str | None]] = [["a", "glass_wool_48k_50mm_gap50"], ["b", "glass_wool_48k_50mm_gap50"],
                                         ["c", "glass_wool_48k_50mm_gap50"], ["d", "glass_wool_24k_50mm"]]
SINGLE_THICKNESS: list[list[str | None]] = [["floor", "glass_wool_24k_50mm"], ["x0", "glass_wool_96k_50mm"],
                                            ["xL", "glass_wool_24k_50mm"], ["y0", "glass_wool_96k_50mm"],
                                            ["yL", None], ["ceiling", "glass_wool_24k_50mm"]]
SEGMENTS = ("baseline_retune", "candidate_retune")


def _hand_round_bound(settings: MaterialGradientSettings, candidates: int) -> int:
    """考卷自己數的一輪最壞求值次數：梯度 1 次；基準與每個候選各一段重調，起點 1 次＋K 步 × 每步最多試探數。"""
    descent = settings.descent
    trials = sum(1 for power in range(200) if descent.initial_step * descent.backtrack_ratio ** power >= descent.min_step)
    return 1 + (candidates + 1) * (1 + settings.retune.steps * trials)


def _problem() -> Record:
    bound = _hand_round_bound(_settings(), 3)
    synthetic: dict[str, Record] = {
        "multi": {"surfaces": FOUR_SURFACES, "override": {}, "again": True, "resume_every_row": True,
                  "single_round": True},
        "huge_threshold": {"surfaces": FOUR_SURFACES, "override": {"improvement_threshold": 1.0e6}},
        "no_candidates": {"surfaces": SINGLE_THICKNESS, "override": {}},
        "max_rounds": {"surfaces": FOUR_SURFACES, "override": {"max_rounds": 1}},
        "max_evaluations": {"surfaces": FOUR_SURFACES, "override": {"max_evaluations": bound}},
        "budget_too_small": {"surfaces": FOUR_SURFACES, "override": {"max_evaluations": bound - 1}},
    }
    return {"synthetic": synthetic, "surrogate": True, "presets": list(CASES[DESCENT_CASE]), "purpose": PURPOSE,
            "receivers": _receivers().model_dump(mode="json")}


def _compute(directory: Path, request: pytest.FixtureRequest) -> str:
    """建衍生替身、跑探針，回探針輸出加上父行程用同一份替身算的標頭與建替身秒數（JSON 文字）。

    只有算的那個工人會走到這裡、才去拿 ``model``（約 70 秒）；等的工人不建替身。目錄先清空：
    帳本是獨佔建檔，上一個拿鎖的工人若算到一半掛了，留下的半份不能被這一次接著用。
    """
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    clock = time.perf_counter()
    chain_model = _chain_model(cast(ReducedModel, request.getfixturevalue("model")))
    model_seconds = time.perf_counter() - clock
    save_reduced(chain_model, directory / "model.npz")
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
    output["chain_header"] = ledger.header_for(chain_model, _designs(DESCENT_CASE), _settings()).model_dump(mode="json")
    output["parent_seconds"] = {"model": model_seconds, "probe": time.perf_counter() - clock}
    text = json.dumps(output)
    (directory / "fixture.json").write_text(text, encoding="utf-8")
    return text


@pytest.fixture(scope="module")
def runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
         ) -> tuple[Path, Record]:
    """一場 pytest 只建一次替身、跑一次探針，平行的工人共用（#561 庚二修補一）。

    worksteal 會把這一模組的題分給每一個工人，模組級夾具在每個工人各算一次、彼此搶 CPU（本機全套一題記到 836 秒）。
    共用照既有的 ``_scheme_cache.shared_json``：檔案鎖、共用檔放在這一場的共同暫存根；``-n 0``（master）不共用，
    用自己的暫存目錄（basetemp 的上一層會跨場殘留，不能拿來共用）。帳本檔放在共用目錄，讀的人只讀不改。
    """
    directory = (tmp_path_factory.mktemp("material-gradient-loop") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "material-gradient-loop-shared")
    text = shared_json(tmp_path_factory, worker_id, "material-gradient-loop", lambda: _compute(directory, request))
    return directory, cast(Record, json.loads(text))


def _synthetic(runs: tuple[Path, Record], name: str) -> Record:
    return cast(Record, cast(Record, runs[1]["synthetic"])[name])


def _surrogate(runs: tuple[Path, Record]) -> Record:
    return cast(Record, runs[1]["surrogate"])


def _rows(runs: tuple[Path, Record], file_name: object) -> tuple[ledger.MaterialGradientRow, ...]:
    return ledger.read_ledger(runs[0] / str(file_name))[1]


def _without_seconds(rows: Sequence[ledger.MaterialGradientRow]) -> list[dict[str, object]]:
    return [{key: value for key, value in row.model_dump(mode="json").items() if key != "seconds"} for row in rows]


def _decisions(rows: Sequence[ledger.MaterialGradientRow]) -> list[ledger.MaterialGradientRow]:
    return [row for row in rows if row.phase == "round_decision"]


def _thickness_mask(names: Sequence[str]) -> NDArray[np.bool_]:
    return np.asarray([name.endswith("layer1.thickness_m") for name in names])


# ---- 候選的產生（不求值，父行程） ----

def _theta(designs: Sequence[gradient_search.SurfaceDesign], values: dict[str, float]) -> NDArray[np.float64]:
    names = gradient_search.design_names(designs)
    assert sorted(values) == sorted(names)
    return np.asarray([values[name] for name in names], dtype=np.float64)


def _largest_cavity(top: float, thickness: float) -> float:
    """考卷自己找的「厚度＋空腔 ≤ 上限」最大空腔：從 top − thickness 往下一個一個浮點數退到放得下。"""
    room = np.float64(top) - np.float64(thickness)
    while np.float64(thickness) + room > np.float64(top):
        room = np.nextafter(room, -np.inf)
    return float(room)


def test_candidates_match_hand_calculation() -> None:
    """抓候選位移只算厚度不算空腔、預測變化漏了空腔分量、厚到頂還給往厚、貼牆面不給候選、24K 給候選、排序錯。

    two_rigid：地板 0.05＋0.08、x0 0.1＋上限內最大空腔（厚到頂只有往薄）、xL 0.05 貼牆、yL 24K（只有一個合法厚度）。
    """
    designs = _designs(DESCENT_CASE)
    top = _settings().family[0].max_total_depth_m
    theta = _theta(designs, {"floor.layer1.thickness_m": 0.05, "floor.layer2.thickness_m": 0.08,
                             "x0.layer1.thickness_m": 0.1, "x0.layer2.thickness_m": _largest_cavity(top, 0.1),
                             "xL.layer1.thickness_m": 0.05, "yL.layer1.thickness_m": 0.05})
    gradient = np.asarray([-4.0, 2.0, 3.0, -1.0, -2.0, 7.0])
    floor_cavity = _largest_cavity(top, 0.1)
    expected = [  # (代號, 新厚度, 新空腔, Δθ)
        ("floor:thicker", 0.1, floor_cavity, (0.1 - 0.05, floor_cavity - 0.08, 0.0, 0.0, 0.0, 0.0)),
        ("x0:thinner", 0.05, theta[3], (0.0, 0.0, 0.05 - 0.1, 0.0, 0.0, 0.0)),
        ("xL:thicker", 0.1, None, (0.0, 0.0, 0.0, 0.0, 0.1 - 0.05, 0.0)),
    ]
    predicted = {name: float(gradient @ np.asarray(delta)) for name, _, _, delta in expected}
    assert predicted["floor:thicker"] < predicted["x0:thinner"] < predicted["xL:thicker"]
    found = loop.candidates_from_gradient(theta, gradient, designs, _settings())
    assert [item.candidate_id for item in found] == ["floor:thicker", "x0:thinner", "xL:thicker"]
    for item, (name, thickness, cavity, delta) in zip(found, expected, strict=True):
        assert (item.thickness_m, item.cavity_m, item.delta) == (thickness, cavity, delta), name
        assert item.predicted_change == pytest.approx(predicted[name], rel=1e-15, abs=1e-15)
    jumped = loop.apply_candidate(theta, found[0], designs)
    assert jumped[0] == 0.1 and jumped[1] == floor_cavity and gradient_search.is_feasible(jumped, designs)
    two = loop.candidates_from_gradient(theta, gradient, designs, _with_candidates(max_per_round=2))
    assert [item.candidate_id for item in two] == ["floor:thicker", "x0:thinner"]


def _with_candidates(**changes: object) -> MaterialGradientSettings:
    base = _settings()
    return base.model_copy(update={"candidates": CandidateSettings.model_validate(
        {**base.candidates.model_dump(), **changes})})


def _three_thickness_settings(**family_changes: object) -> MaterialGradientSettings:
    """考卷自己組的 48K 家族：三個合法厚度 0.05／0.075／0.1（產品設定檔只有兩個）。"""
    base = _settings()
    family = FamilySettings.model_validate({**base.family[0].model_dump(), "thicknesses_m": (0.05, 0.075, 0.1),
                                            **family_changes})
    return MaterialGradientSettings.model_validate({**base.model_dump(), "family": [
        family.model_dump(), *(item.model_dump() for item in base.family[1:])]})


def test_more_candidates_than_n_keeps_first_n_in_order() -> None:
    """抓沒有截到前 N 個、同分時排序不照「面序、再薄先厚後」：兩面都在中間厚度、各有往薄往厚，共四個候選；
    梯度為零時四個同分、取前三是 a 往薄、a 往厚、b 往薄；梯度不為零時照預測變化由小到大。"""
    settings = _three_thickness_settings()
    designs = gradient_search.designs_from_presets(
        (("a", "glass_wool_48k_50mm_gap50"), ("b", "glass_wool_48k_50mm_gap50"), ("c", None)), settings, _presets())
    theta = _theta(designs, {"a.layer1.thickness_m": 0.075, "a.layer2.thickness_m": 0.02,
                             "b.layer1.thickness_m": 0.075, "b.layer2.thickness_m": 0.03})
    tied = loop.candidates_from_gradient(theta, np.zeros(4), designs, settings)
    assert [item.candidate_id for item in tied] == ["a:thinner", "a:thicker", "b:thinner"]
    every = loop.candidates_from_gradient(theta, np.asarray([1.0, 0.0, -3.0, 0.0]), designs,
                                          _three_thickness_settings().model_copy(update={"candidates": CandidateSettings(
                                              max_per_round=10, improvement_threshold=0.05, max_rounds=1,
                                              max_evaluations=10)}))
    hand = sorted(((float(np.asarray([1.0, 0.0, -3.0, 0.0]) @ np.asarray(item.delta)), item.candidate_id)
                   for item in every))
    assert [name for _, name in hand] == [item.candidate_id for item in every] == [
        "b:thicker", "a:thinner", "a:thicker", "b:thinner"]
    top_three = loop.candidates_from_gradient(theta, np.asarray([1.0, 0.0, -3.0, 0.0]), designs, settings)
    assert [item.candidate_id for item in top_three] == ["b:thicker", "a:thinner", "a:thicker"]


def test_jump_that_cannot_fit_total_depth_is_not_a_candidate() -> None:
    """抓空腔切到下界以下仍當候選：空腔下界 0.06、總深度 0.15，厚度 0.05 → 0.1 時空腔最多 0.05，放不下，不是候選；
    往薄那一邊（0.05 是最薄）也沒有，所以這一面一個候選都沒有。"""
    settings = _settings()
    family = FamilySettings.model_validate({**settings.family[0].model_dump(), "cavity_range_m": (0.06, 0.15)})
    settings = MaterialGradientSettings.model_validate({**settings.model_dump(), "family": [
        family.model_dump(), *(item.model_dump() for item in settings.family[1:])]})
    designs = gradient_search.designs_from_presets((("a", "glass_wool_48k_50mm_gap50"),), settings, _presets())
    theta = _theta(designs, {"a.layer1.thickness_m": 0.05, "a.layer2.thickness_m": 0.07})
    assert gradient_search.is_feasible(theta, designs)
    assert loop.candidates_from_gradient(theta, np.asarray([-100.0, -100.0]), designs, settings) == ()


def test_single_thickness_families_have_no_candidates() -> None:
    """抓 24K／96K 單一厚度的面給候選：六面全是單厚度家族（加一面剛性），回空 tuple。"""
    designs = gradient_search.designs_from_presets(tuple((str(a), b) for a, b in SINGLE_THICKNESS), _settings(),
                                                   _presets())
    theta = gradient_search.baseline_theta(designs)
    assert loop.candidates_from_gradient(theta, np.ones(theta.shape), designs, _settings()) == ()


# ---- 一輪 ----

def _round_input(rows: Sequence[ledger.MaterialGradientRow], round_index: int) -> tuple[str, ...]:
    """這一輪的輸入 θ：上一輪決定列交出去的，第 0 輪是前段最後一列。"""
    if round_index == 0:
        return [row for row in rows if row.round is None][-1].theta_hex
    return next(row for row in rows if row.phase == "round_decision" and row.round == round_index - 1).theta_hex


def _segments(rows: Sequence[ledger.MaterialGradientRow]) -> list[list[ledger.MaterialGradientRow]]:
    segments: list[list[ledger.MaterialGradientRow]] = []
    for row in rows:
        if row.phase not in SEGMENTS:
            continue
        if segments and (segments[-1][-1].phase, segments[-1][-1].round, segments[-1][-1].candidate) == (
                row.phase, row.round, row.candidate) and segments[-1][-1].stop_reason is None:
            segments[-1].append(row)
        else:
            segments.append([row])
    return segments


@pytest.mark.parametrize("source", ("surrogate", "multi"))
def test_round_retunes_each_branch_at_most_k_steps_with_thickness_fixed(runs: tuple[Path, Record], source: str) -> None:
    """抓基準或候選重調超過 K 步、重調動到厚度、基準不是從這一輪的輸入 θ 起、候選不是從自己的跳點起、
    跳點動到別面、停止原因跟步數對不上。"""
    record = _surrogate(runs) if source == "surrogate" else _synthetic(runs, "multi")
    header, rows = ledger.read_ledger(runs[0] / str(record["ledger"]))
    settings = _settings()
    thickness = _thickness_mask(header.variable_names)
    segments = _segments(rows)
    assert len(segments) >= 8
    for segment in segments:
        taken = len(segment) - 1
        assert taken <= settings.retune.steps
        first = np.asarray(segment[0].theta())
        assert all(np.asarray(row.theta())[thickness].tobytes() == first[thickness].tobytes() for row in segment)
        reason = segment[-1].stop_reason
        assert reason in (gradient_search.STOP_MAX_STEPS, gradient_search.STOP_STEP)
        if reason == gradient_search.STOP_MAX_STEPS:
            assert taken == settings.retune.steps
        else:
            assert cast(float, segment[-1].final_trial_step) < settings.descent.min_step
        if segment[0].phase == "baseline_retune":
            assert segment[0].theta_hex == _round_input(rows, cast(int, segment[0].round))
    for row in rows:
        if row.phase != "candidate_jump" or row.jump is None:
            continue
        start = np.asarray([float.fromhex(value) for value in _round_input(rows, cast(int, row.round))])
        jumped, delta = np.asarray(row.theta()), np.asarray([float.fromhex(value) for value in row.jump.delta_hex])
        moved = [name.split(".")[0] for name, a, b in zip(header.variable_names, start, jumped, strict=True) if a != b]
        assert set(moved) <= {str(row.candidate).split(":")[0]} and moved
        assert np.array_equal(jumped - start, delta)
        retune = next(segment for segment in segments if segment[0].candidate == row.candidate
                      and segment[0].round == row.round)
        assert retune[0].theta_hex == row.theta_hex


@pytest.mark.parametrize("source", ("surrogate", "multi", "huge_threshold"))
def test_round_accepts_only_above_threshold_and_hands_over_retuned_theta(runs: tuple[Path, Record],
                                                                         source: str) -> None:
    """抓沒超過門檻也接受、接受的是跳點而不是重調後的 θ、都沒超過時交出沒重調的基準、值逐輪上升。"""
    record = _surrogate(runs) if source == "surrogate" else _synthetic(runs, source)
    rows = _rows(runs, record["ledger"])
    threshold = (_with_candidates(improvement_threshold=1.0e6) if source == "huge_threshold"
                 else _settings()).candidates.improvement_threshold
    segments = _segments(rows)
    previous = [row for row in rows if row.round is None][-1].value
    for row in _decisions(rows):
        decision = cast(ledger.RoundDecision, row.decision)
        baseline = next(item for item in segments if item[0].phase == "baseline_retune" and item[0].round == row.round)
        outcomes = {item[0].candidate: item for item in segments if item[0].phase == "candidate_retune"
                    and item[0].round == row.round}
        improvements = {name: baseline[-1].value - item[-1].value for name, item in outcomes.items()}
        best = max(improvements, key=lambda name: improvements[name])
        assert decision.threshold == threshold
        if improvements[best] > threshold:
            assert decision.accepted == best and row.theta_hex == outcomes[best][-1].theta_hex
        else:
            assert decision.accepted is None and row.theta_hex == baseline[-1].theta_hex
        assert row.value <= baseline[0].value and row.value <= previous
        previous = row.value
    if source == "huge_threshold":
        assert rows[-1].stop_reason == gradient_search.LOOP_STOP_NO_IMPROVEMENT and [row.round for row in _decisions(rows)] == [0]


def test_round_evaluation_count_equals_actual_calls(runs: tuple[Path, Record]) -> None:
    """抓求值次數少數線搜尋的試探或多數：合成目標包一層每執行一次就加一的回呼，整場與單獨一輪各對一次；
    每輪記的次數也要等於梯度 1 次加各段下降記的次數，最壞值等於考卷自己數的。"""
    multi = _synthetic(runs, "multi")
    result = cast(Record, multi["result"])
    assert multi["calls"] == result["evaluations"]
    single = cast(Record, multi["single_round"])
    assert single["calls"] == single["evaluations"] and cast(int, single["evaluations"]) > 1 + 4
    assert single["bound"] == _hand_round_bound(_settings(), cast(int, single["candidates"]))
    rows = _rows(runs, multi["ledger"])
    for row in _decisions(rows):
        spent = 1 + sum(item.evaluations or 0 for item in rows if item.round == row.round)
        assert cast(ledger.RoundDecision, row.decision).round_evaluations == spent


# ---- 整場：停止原因 ----

def test_stop_reason_no_improvement_and_rerun_is_bitwise(runs: tuple[Path, Record]) -> None:
    """抓正式設定在小房間上停不下來、記錯原因、重跑帶非決定性：替身與合成各跑兩次，帳本逐值相同（秒數除外），
    最後一輪沒接受任何候選、原因是「已確認候選內沒有超過門檻」，前面每一輪都接受了一個。"""
    for record in (_surrogate(runs), _synthetic(runs, "multi")):
        rows = _rows(runs, record["ledger"])
        assert _without_seconds(rows) == _without_seconds(_rows(runs, record["again"]))
        decisions = _decisions(rows)
        assert rows[-1].stop_reason == gradient_search.LOOP_STOP_NO_IMPROVEMENT
        assert len(decisions) >= 3 and cast(ledger.RoundDecision, decisions[-1].decision).accepted is None
        assert all(cast(ledger.RoundDecision, row.decision).accepted is not None for row in decisions[:-1])
        result = cast(Record, record["result"])
        assert result["stop_reason"] == rows[-1].stop_reason and result["rounds"] == len(decisions)
        assert result["theta"] == list(rows[-1].theta_hex)
        assert result["evaluations"] == cast(ledger.RoundDecision, rows[-1].decision).total_evaluations


def test_stop_reason_no_candidates(runs: tuple[Path, Record]) -> None:
    """抓六面全是單厚度家族時還去重調或記成別的原因：只有一列決定、求值 1 次（梯度那一次）、θ 原樣。"""
    rows = _rows(runs, _synthetic(runs, "no_candidates")["ledger"])
    decisions = _decisions(rows)
    assert [row.round for row in decisions] == [0] and rows[-1] is decisions[0]
    decision = cast(ledger.RoundDecision, decisions[0].decision)
    assert decisions[0].stop_reason == gradient_search.LOOP_STOP_NO_CANDIDATES
    assert decision.outcomes == () and decision.round_evaluations == decision.total_evaluations == 1
    assert decisions[0].theta_hex == rows[0].theta_hex


def test_stop_reason_max_rounds(runs: tuple[Path, Record]) -> None:
    """抓輪數上限沒守：上限 1 時只跑一輪、那一輪有接受、原因是「輪數到上限」。"""
    rows = _rows(runs, _synthetic(runs, "max_rounds")["ledger"])
    decisions = _decisions(rows)
    assert [row.round for row in decisions] == [0] and decisions[0].stop_reason == gradient_search.LOOP_STOP_MAX_ROUNDS
    assert cast(ledger.RoundDecision, decisions[0].decision).accepted is not None


@pytest.mark.parametrize("name", ("max_evaluations", "budget_too_small"))
def test_stop_reason_max_evaluations_matches_recorded_numbers(runs: tuple[Path, Record], name: str) -> None:
    """抓求值上限沒守或記的數字跟原因對不上：上限＝一輪最壞值時第一輪開得了、第二輪最壞值放不下；
    上限再少 1 時第一輪就不開（0 次求值、θ 原樣）。記「求值到上限」時累計加下一輪最壞值真的超過上限，整場沒超過上限。"""
    limit = _hand_round_bound(_settings(), 3) - (1 if name == "budget_too_small" else 0)
    rows = _rows(runs, _synthetic(runs, name)["ledger"])
    decisions = _decisions(rows)
    last = decisions[-1]
    decision = cast(ledger.RoundDecision, last.decision)
    assert last.stop_reason == gradient_search.LOOP_STOP_MAX_EVALUATIONS
    assert decision.next_round_bound == _hand_round_bound(_settings(), 3)
    assert decision.total_evaluations + decision.next_round_bound > limit >= decision.total_evaluations
    if name == "budget_too_small":
        assert [row.round for row in decisions] == [0] and decision.round_evaluations == 0 and decision.outcomes == ()
        assert last.theta_hex == rows[0].theta_hex
    else:
        assert [row.round for row in decisions] == [0] and decision.accepted is not None and decision.round_evaluations > 0


# ---- 帳本 v2 ----

def _rewrite(source: Path, target: Path, header: dict[str, object] | None, documents: list[dict[str, object]]) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    first = json.dumps(header, ensure_ascii=False) + "\n" if header is not None else lines[0]
    target.write_text(first + "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in documents),
                      encoding="utf-8")
    return target


def _documents(source: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()[1:]]


def _renumbered(documents: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{**item, "step": index} for index, item in enumerate(documents)]


def test_ledger_v2_round_trip(runs: tuple[Path, Record], tmp_path: Path) -> None:
    """抓帳本漏寫或改寫：讀回再寫一份逐位相同，讀回的停止原因、最後 θ、累計次數等於探針那一次的結果。"""
    source = runs[0] / str(_surrogate(runs)["ledger"])
    header, rows = ledger.read_ledger(source)
    assert header.ledger_version == ledger.LEDGER_VERSION == "aosr.material_gradient_ledger.v2"
    copy = tmp_path / "copy.jsonl"
    ledger.write_ledger(copy, header, rows)
    assert copy.read_bytes() == source.read_bytes()
    phases = {row.phase for row in rows}
    assert {"snap", "baseline_retune", "candidate_jump", "candidate_retune", "round_decision"} <= phases


def _tamper(source: Path, target: Path, edit: str) -> Path:
    documents = _documents(source)
    if edit == "round":
        index = next(i for i, item in enumerate(documents) if item["phase"] == "candidate_retune" and item["round"] == 1)
        documents[index]["round"] = 2
    elif edit == "drop_decision":
        index = next(i for i, item in enumerate(documents) if item["phase"] == "round_decision")
        documents = _renumbered(documents[:index] + documents[index + 1:])
    elif edit == "swap_candidates":
        blocks: list[list[int]] = []
        for i, item in enumerate(documents):
            if item["round"] == 0 and item["phase"] == "candidate_jump":
                blocks.append([i])
            elif item["round"] == 0 and item["phase"] == "candidate_retune":
                blocks[-1].append(i)
        first, second = blocks[0], blocks[1]
        reordered = (documents[:first[0]] + [documents[i] for i in second] + [documents[i] for i in first]
                     + documents[second[-1] + 1:])
        documents = _renumbered(reordered)
    return _rewrite(source, target, None, documents)


@pytest.mark.parametrize("edit", ("round", "drop_decision", "swap_candidates"))
def test_ledger_v2_rejects_tampered_rows(runs: tuple[Path, Record], tmp_path: Path, edit: str) -> None:
    """抓讀回不驗輪的結構：改一列的輪號、刪掉中間一列決定（列號重排）、交換兩個候選的跳點＋重調那幾列（列號重排），都拒收。"""
    source = runs[0] / str(_surrogate(runs)["ledger"])
    with pytest.raises(ValueError):
        ledger.read_ledger(_tamper(source, tmp_path / "tampered.jsonl", edit))


def test_ledger_v2_rejects_other_surrogate_and_v1(chain_header: ledger.MaterialGradientHeader,
                                                  runs: tuple[Path, Record], tmp_path: Path) -> None:
    """抓拿別的替身的帳本接著用、v1 帳本照收：預期標頭替身雜湊不同拒收；標頭版本是 v1 拒收、訊息講明版本。"""
    source = runs[0] / str(_surrogate(runs)["ledger"])
    ledger.read_ledger(source, expected_header=chain_header)
    other = chain_header.model_copy(update={"input_hash": "0" * 64})
    with pytest.raises(ValueError, match="input_hash"):
        ledger.read_ledger(source, expected_header=other)
    old = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    old["ledger_version"] = "aosr.material_gradient_ledger.v1"
    with pytest.raises(ValueError, match=r"版本.*v1"):
        ledger.read_ledger(_rewrite(source, tmp_path / "v1.jsonl", old, _documents(source)))


@pytest.fixture(scope="module")
def chain_header(runs: tuple[Path, Record]) -> ledger.MaterialGradientHeader:
    """父行程（算的那個工人）用同一份衍生替身算的標頭，跟探針子行程寫進帳本的那份各算各的；不另建替身。"""
    return ledger.MaterialGradientHeader.model_validate(runs[1]["chain_header"])


# ---- 接續 ----

def test_resume_at_every_row_boundary_matches_one_shot(runs: tuple[Path, Record]) -> None:
    """抓接續丟錯列、輪號或累計次數接錯、半輪沒丟乾淨：合成題在每一個列界截斷各接續一次、替身題在兩個列界，
    接續完的帳本都跟一口氣跑完的逐值相同（θ 的 float.hex、值、分項、停止原因、決定；秒數除外）。"""
    multi = _synthetic(runs, "multi")
    expected = _without_seconds(_rows(runs, multi["ledger"]))
    resumed = cast(list[str], multi["resumed"])
    assert len(resumed) == len(expected)
    for name in resumed:
        assert _without_seconds(_rows(runs, name)) == expected, name
    surrogate = _surrogate(runs)
    expected = _without_seconds(_rows(runs, surrogate["ledger"]))
    for name in cast(list[str], surrogate["resumed"]):
        assert _without_seconds(_rows(runs, name)) == expected, name


def _never_called(theta: jax.Array) -> tuple[jax.Array, StageOneObjectiveArrays]:
    raise AssertionError("這裡不該求值")


def test_resume_finished_ledger_returns_without_evaluating(runs: tuple[Path, Record],
                                                            chain_header: ledger.MaterialGradientHeader,
                                                            tmp_path: Path) -> None:
    """抓已經收尾的帳本又被重跑或改寫：直接交回帳本記的結果、檔案逐位不變；標頭不符拒收。"""
    source = runs[0] / str(_surrogate(runs)["ledger"])
    copy = tmp_path / "finished.jsonl"
    copy.write_bytes(source.read_bytes())
    result = loop.resume_rounds(_never_called, _designs(DESCENT_CASE), _settings(), copy, chain_header)
    assert copy.read_bytes() == source.read_bytes()
    assert [value.hex() for value in result.theta] == cast(Record, _surrogate(runs)["result"])["theta"]
    assert result.stop_reason == cast(Record, _surrogate(runs)["result"])["stop_reason"]
    with pytest.raises(ValueError, match="settings_fingerprint"):
        loop.resume_rounds(_never_called, _designs(DESCENT_CASE), _with_candidates(max_rounds=2), copy,
                           ledger.MaterialGradientHeader.model_validate(
                               {**chain_header.model_dump(), "settings_fingerprint": "f" * 64}))


# ---- 單精度：只量不登記 ----

def test_single_precision_round_stays_float32_and_finite(runs: tuple[Path, Record]) -> None:
    """抓單精度被悄悄提升或算出非有限：同一輪用 float32 θ 跑，交出的 θ 與基準重調都是 float32、值有限。"""
    single = cast(Record, _surrogate(runs)["single"])
    assert single["dtype"] == "float32" and single["baseline_dtype"] == "float32"
    assert np.isfinite(float.fromhex(cast(str, single["value"])))
    assert all(np.isfinite(float.fromhex(value)) for value in cast(list[str], single["candidate_values"]))


# ---- 連續段的總深度投影（不求值，父行程） ----

def _brute_nearest(point: NDArray[np.float64], low: NDArray[np.float64], high: NDArray[np.float64],
                   top: float) -> float:
    """細格點暴力找多邊形（盒 ∩ t＋c ≤ top）裡離 point 最近的距離：盒內 801×801 格點加斜邊上 20001 點。"""
    t, c = np.meshgrid(np.linspace(low[0], high[0], 801), np.linspace(low[1], high[1], 801))
    inside = t + c <= top
    distances = np.hypot(t[inside] - point[0], c[inside] - point[1])
    edge_t = np.linspace(max(low[0], top - high[1]), min(high[0], top - low[1]), 20001)
    edge = np.hypot(edge_t - point[0], top - edge_t - point[1])
    return float(min(distances.min(), edge.min()))


def test_total_depth_projection_is_exact_nearest_point() -> None:
    """抓投影只切空腔、只投斜邊不夾端點、或投到斜邊外：隨機點（盒內外、斜邊兩側、靠端點）投完要可行，
    距離不大於細格點暴力找的最近距離（相對差 1e-12 內），也不比它小超過格距（不是投到更近的不可行點）。"""
    low, high, top = np.asarray([0.05, 0.0]), np.asarray([0.1, 0.15]), 0.15
    limit = gradient_search.DepthLimit(0, 1, top)
    rng = np.random.default_rng(561)
    points = [*rng.uniform([-0.05, -0.1], [0.25, 0.3], size=(60, 2)), np.asarray([0.2, 0.0]),
              np.asarray([0.12, 0.2]), np.asarray([0.0, 0.2]), np.asarray([0.09, 0.08])]
    spacing = float(np.hypot(*(high - low))) / 800
    for point in points:
        projected = gradient_search.project_to_limits(np.asarray(point), low, high, (limit,))
        assert np.all(low <= projected) and np.all(projected <= high) and projected[0] + projected[1] <= top
        ours = float(np.hypot(*(projected - point)))
        brute = _brute_nearest(np.asarray(point), low, high, top)
        assert ours <= brute * (1.0 + 1e-12) + 1e-15, (point, projected)
        assert brute - ours <= spacing, (point, projected)


def test_projection_leaves_feasible_points_and_single_variable_faces() -> None:
    """抓投影動到已可行的點或沒有空腔層的面不守上限：可行點逐位不動；沒有空腔層的面厚度夾到上限。"""
    low, high = np.asarray([0.05, 0.0, 0.05]), np.asarray([0.1, 0.15, 0.2])
    limits = (gradient_search.DepthLimit(0, 1, 0.15), gradient_search.DepthLimit(2, None, 0.15))
    feasible = np.asarray([0.07, 0.06, 0.12])
    assert gradient_search.project_to_limits(feasible, low, high, limits).tobytes() == feasible.tobytes()
    assert gradient_search.project_to_limits(np.asarray([0.07, 0.06, 0.18]), low, high, limits)[2] == 0.15


# ---- 拒收 ----

def test_rejections(tmp_path: Path) -> None:
    """抓不可行 θ 照樣產候選或開一輪、剩下預算是負的照收、起點跟帳本對不上照跑、設定缺候選節。"""
    settings, designs = _settings(), _designs(DESCENT_CASE)
    infeasible = _theta(designs, {"floor.layer1.thickness_m": 0.071, "floor.layer2.thickness_m": 0.05,
                                  "x0.layer1.thickness_m": 0.05, "x0.layer2.thickness_m": 0.05,
                                  "xL.layer1.thickness_m": 0.05, "yL.layer1.thickness_m": 0.05})
    with pytest.raises(ValueError, match="可行"):
        loop.candidates_from_gradient(infeasible, np.zeros(6), designs, settings)
    with pytest.raises(ValueError, match="可行"):
        loop.confirmation_round(_never_called, infeasible, designs, settings, budget_left=100)
    feasible = gradient_search.snap_to_specs(infeasible, designs).theta
    with pytest.raises(ValueError, match="非負"):
        loop.confirmation_round(_never_called, feasible, designs, settings, budget_left=-1)
    with pytest.raises(ValueError, match="梯度"):
        loop.candidates_from_gradient(feasible, np.zeros(5), designs, settings)
    missing = tmp_path / "missing.toml"
    text = config_path("material_gradient.toml").read_text(encoding="utf-8")
    missing.write_text(text.replace("[candidates]", "[unused]"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_material_gradient_settings(missing)


def test_run_rounds_requires_start_in_ledger(runs: tuple[Path, Record], chain_header: ledger.MaterialGradientHeader,
                                             tmp_path: Path) -> None:
    """抓起點不在帳本裡就開跑（接續會找不到起點）、已有迴圈列還用 run_rounds 從頭跑：都在求值前拒收。"""
    source = runs[0] / str(_surrogate(runs)["ledger"])
    pre_loop = tmp_path / "pre.jsonl"
    pre_loop.write_bytes(b"".join(source.read_bytes().splitlines(keepends=True)[:2]))
    _, rows = ledger.read_ledger(pre_loop)
    start = np.asarray(rows[-1].theta())
    moved = start.copy()
    moved[1] = moved[1] / 2
    with pytest.raises(ValueError, match="逐位"):
        loop.run_rounds(_never_called, moved, _designs(DESCENT_CASE), _settings(), pre_loop, chain_header)
    with pytest.raises(ValueError, match="resume_rounds"):
        loop.run_rounds(_never_called, start, _designs(DESCENT_CASE), _settings(), source, chain_header)


@pytest.mark.parametrize("field, bad", (
    ("max_per_round", 0), ("max_per_round", 1.5), ("max_per_round", True), ("max_rounds", -1), ("max_rounds", "3"),
    ("max_evaluations", 0), ("improvement_threshold", 0.0), ("improvement_threshold", -0.1),
    ("improvement_threshold", 1),
))
def test_candidate_settings_reject_bad_values(field: str, bad: object) -> None:
    """抓候選數／輪數／求值上限收非正整數、門檻收非正或整數。"""
    with pytest.raises(ValueError):
        CandidateSettings.model_validate({**_settings().candidates.model_dump(), field: bad})


def test_settings_have_candidates_section() -> None:
    """抓 [candidates] 沒載進來或指紋沒跟著變：四個欄位都在、改一個值指紋就不同。"""
    settings = _settings()
    assert set(settings.candidates.model_dump()) == {"max_per_round", "improvement_threshold", "max_rounds",
                                                     "max_evaluations"}
    assert _with_candidates(max_rounds=settings.candidates.max_rounds + 1).fingerprint != settings.fingerprint
