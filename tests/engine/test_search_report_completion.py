"""收尾報告的手排名次與唯讀幾何檢查；只合成結果，不解物理。"""

from pathlib import Path
import re
from typing import Literal, get_args

import pytest

from aosr.config.paths import config_path
from aosr.config.placement_standards import load_placement_standards
from aosr.config.precision_contracts import load_precision_contracts
from aosr.geometry.shoebox import Point
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme, expected_pairs
from aosr.search.ledger import LedgerRow, create_for
from aosr.search.outer_status import OUTER_MESSAGES, OuterConclusion, OuterStatus, snapshot_of
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.refine_run import _progress, header_for
from aosr.search.report import build_report, render_text
from aosr.search.run import CandidateJob, SearchStatus
from aosr.search.sampler import RankingZone
from aosr.search.store import SearchStore, candidate_name, refine_result_name
from tests.engine._search_run_cases import RUN_DATE, make_store
from tests.engine.test_search_report import _ResultCompute, _pair, _sections, _snapshot
from tests.engine.test_search_standards_check import with_points



def _clause_block(section: str, clause_id: str) -> str:
    """檢查表一條一段：標題列（條號、說明）加底下縮排的兩邊與原文。"""
    lines = section.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(clause_id + " |"))
    end = next((index for index in range(start + 1, len(lines)) if not lines[index].startswith("  ")), len(lines))
    return "\n".join(lines[start:end])

def _store(tmp_path: Path) -> tuple[SearchStore, Path]:
    store, registry = make_store(tmp_path)
    ledger = create_for(store)
    compute = _ResultCompute(store)
    for number, score in ((None, 0.0), (0, 3.0), (1, 1.0), (2, 1.0), (3, 4.0), (4, 5.0)):
        scheme = store.project.model_copy(update={"scheme_id": f"scheme-{number}"})
        path = store.baseline_path if number is None else store.candidate_path(number)
        tuple(compute((CandidateJob(number, scheme, path),), 1))
        if number is not None:
            ledger.append(LedgerRow(
                batch_index=0, trial_number=number,
                unit_params_hex={key: 0.5.hex() for key in ("front_distance", "spacing", "listening_distance")},
                params_m={"front_distance": 1.0, "spacing": 2.0, "listening_distance": 3.0},
                outcome="scored", score=score, reason=None, violation_m=None,
                seconds=0.0, result_file=candidate_name(number),
            ))
    store.status_path.write_text(SearchStatus(state="converged", best_trial=1, best_score=1.0).model_dump_json())
    return store, registry


def _refined(store: SearchStore, costs: tuple[tuple[int | None, float], ...]) -> None:
    store.ensure_refine_dir()
    ledger = RefineLedger.create(store.refine_ledger_path, header_for(store))
    rows = []
    for index, (number, cost) in enumerate(costs):
        path = store.baseline_path if number is None else store.candidate_path(number)
        store.refine_result_path(number).write_bytes(path.read_bytes())
        row = RefineRow(round=1 if index < 2 else 2, trial_number=number,
                        result_file=refine_result_name(number), outcome="scored", total_cost=cost, seconds=0.0)
        ledger.append(row)
        rows.append(row)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    status = status.model_copy(update={"refine": _progress(rows, 2)})
    store.status_path.write_text(status.model_dump_json())


def _text(store: SearchStore, registry: Path, contracts: Path | None = None) -> str:
    if contracts is None:
        return render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    return render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE,
                                   precision_contracts_path=contracts))


def _with_scheme(result: SchemeResult, scheme: Scheme) -> SchemeResult:
    """換擺位時重建合成零件，讓結果檔的座標與題目相符；沒有物理解算。"""
    return SchemeResult.model_validate(result.model_dump() | {
        "scheme": scheme, "pairs": tuple(_pair(scheme, speaker, receiver, role)
                                         for speaker, receiver, role in expected_pairs(scheme)),
    })


@pytest.mark.parametrize("costs,expected,first", [
    (((None, 8.0), (2, 5.0), (1, 2.0), (0, 6.0)), [
        "1 號：搜尋排名第 1；在已細算的 4 個方案中，排名第 1。另有 2 個方案尚未細算。",
        "原方案：在已細算的 4 個方案中，排名第 4。",
    ], 1),
    (((None, 8.0), (2, 2.0), (1, 2.0), (0, 6.0)), [
        "1 號：搜尋排名第 1；在已細算的 4 個方案中，排名第 2。另有 2 個方案尚未細算。",
        "2 號：搜尋排名第 2；在已細算的 4 個方案中，排名第 1。另有 2 個方案尚未細算。",
        "原方案：在已細算的 4 個方案中，排名第 4。",
    ], 2),
    (((None, 1.0), (2, 5.0), (1, 2.0), (0, 6.0)), [
        "1 號：搜尋排名第 1；在已細算的 4 個方案中，排名第 2。另有 2 個方案尚未細算。",
        "原方案：在已細算的 4 個方案中，排名第 1。",
    ], "baseline"),
])
def test_hand_ranked_all_rounds_include_baseline_and_pending(
    tmp_path: Path, costs: tuple[tuple[int | None, float], ...], expected: list[str], first: int | str,
) -> None:
    store, registry = _store(tmp_path)
    _refined(store, costs)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.refine.best == first
    sections = _sections(_text(store, registry))
    assert sections["名次"].splitlines() == expected
    headings = list(sections)
    assert headings[headings.index("細算做完沒") + 1] == "名次"


def test_not_started_lists_search_rank_without_refine_rank(tmp_path: Path) -> None:
    store, registry = _store(tmp_path)
    section = _sections(_text(store, registry))["名次"]
    assert section.splitlines() == ["1 號：搜尋排名第 1；還沒有細算。", "原方案：還沒有細算。"]


@pytest.mark.parametrize("outcome,label", [("not_evaluated", "未評估"), ("not_comparable", "不能同表"),
                                           ("excluded", "淘汰")])
def test_nonrankable_refine_row_has_status_not_rank(tmp_path: Path, outcome: str, label: str) -> None:
    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (2, 1.0)))
    row = RefineRow.model_validate({"round": 2, "trial_number": 1, "result_file": refine_result_name(1),
                                   "outcome": outcome, "total_cost": None, "seconds": 0.0})
    RefineLedger.open(store.refine_ledger_path).append(row)
    lines = _sections(_text(store, registry))["名次"].splitlines()
    assert lines == [f"1 號：{label}。",
                     "2 號：搜尋排名第 2；在已細算的 2 個方案中，排名第 1。另有 3 個方案尚未細算。",
                     "原方案：在已細算的 2 個方案中，排名第 2。"]


def test_review_pending_comes_from_refined_result(tmp_path: Path) -> None:
    from aosr.scoring.contract import TimbreChannelsPayload

    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (2, 1.0), (1, 2.0)))
    path = store.refine_result_path(1)
    result = SchemeResult.model_validate_json(path.read_bytes())
    evaluations = []
    for evaluation in result.candidate.evaluations:
        payload = evaluation.payload
        if isinstance(payload, TimbreChannelsPayload):
            channels = []
            for channel in payload.channels:
                features = tuple(feature.model_copy(update={"depth_db": 100.0})
                                 if feature.kind == "peak" else feature for feature in channel.payload.features)
                channels.append(channel.model_copy(update={"payload": channel.payload.model_copy(update={"features": features})}))
            evaluation = evaluation.model_copy(update={"payload": payload.model_copy(update={"channels": tuple(channels)})})
        evaluations.append(evaluation)
    result = result.model_copy(update={"candidate": result.candidate.model_copy(update={"evaluations": tuple(evaluations)})})
    path.write_text(result.model_dump_json())
    line = _sections(_text(store, registry))["名次"].splitlines()[0]
    assert "排名第 2" in line and "尚待確認" in line
    assert "尚待確認" not in _sections(_text(store, registry))["品質合不合格"]


def test_unreadable_refined_result_does_not_guess_review(tmp_path: Path) -> None:
    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (1, 1.0)))
    store.refine_result_path(1).write_text("broken")
    line = _sections(_text(store, registry))["名次"].splitlines()[0]
    assert "結果檔讀不回" in line and "尚待確認" not in line


@pytest.mark.parametrize("changed_trial", [None, 2])
def test_checklist_parallel_quotes_once_and_refined_schemes(tmp_path: Path, changed_trial: int | None) -> None:
    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (2, 1.0), (1, 2.0)))
    result = SchemeResult.model_validate_json(store.refine_result_path(changed_trial).read_bytes())
    changed = with_points(result.scheme, Point(4.0, 5.0, 2.0), Point(7.0, 5.0, 2.0), Point(5.5, 8.0, 2.0))
    store.refine_result_path(changed_trial).write_text(_with_scheme(result, changed).model_dump_json())
    before = _snapshot(store)
    section = _sections(_text(store, registry))["擺位標準檢查表"]
    source = config_path("placement_standards.toml")
    for entry in load_placement_standards(source).entries:
        block = _clause_block(section, entry.id)
        assert re.findall(re.escape(entry.quote), block) == [entry.quote]
        assert re.findall(re.escape(entry.description), block) == [entry.description]
        assert "\n  細算第一名（2 號）：" in block and "\n  原方案：" in block
    width = _clause_block(section, "itu_8_5_3_1_base_width")
    side = "原方案" if changed_trial is None else "細算第一名（2 號）"
    changed_side = next(line for line in width.splitlines() if line.startswith("  " + side + "："))
    assert "聲學中心基寬=3.0" in changed_side
    assert _snapshot(store) == before


@pytest.mark.parametrize("which", [None, 1])
def test_checklist_unreadable_side_keeps_other_side(tmp_path: Path, which: int | None) -> None:
    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (1, 1.0)))
    store.refine_result_path(which).write_text("broken")
    sections = _sections(_text(store, registry))
    side = "原方案" if which is None else "細算第一名（1 號）"
    assert f"{side}：結果檔讀不回，沒檢查" in sections["擺位標準檢查表"]
    assert "聲學中心基寬=" in sections["擺位標準檢查表"]
    assert "判定：未判定合格" in sections["品質合不合格"]


@pytest.mark.parametrize("transport", ["build", "cli"])
def test_report_reads_boundary_from_supplied_registry(
    tmp_path: Path, transport: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, registry = _store(tmp_path)
    source = Path(__file__).resolve().parents[2] / "blueprint" / "precision_contracts.toml"
    contracts = tmp_path / "contracts"
    contracts.write_bytes(source.read_bytes())
    boundary = load_precision_contracts(contracts)["placement_standard_boundary"]
    standards_path = config_path("placement_standards.toml")
    entry = next(item for item in load_placement_standards(standards_path).entries
                 if item.id == "ebu_a1_2_base_width")
    assert entry.thresholds.lower is not None
    base = entry.thresholds.lower * (1.0 + boundary.value * 16.0)
    result = SchemeResult.model_validate_json(store.candidate_path(1).read_bytes())
    scheme = with_points(result.scheme, Point(4.0, 5.0, 2.0), Point(4.0 + base, 5.0, 2.0), Point(5.0, 8.0, 2.0))
    store.candidate_path(1).write_text(_with_scheme(result, scheme).model_dump_json())
    before = _sections(_via_transport(store, registry, contracts, transport, capsys, monkeypatch))["擺位標準檢查表"]
    text = contracts.read_text()
    marker = f'name = "{boundary.name}"'
    head, tail = text.split(marker)
    changed = boundary.value * 32.0
    tail = tail.replace(f"value = {boundary.value}", f"value = {changed}", 1)
    tail = tail.replace(f'display = "{boundary.display}"', f'display = "{changed}"', 1)
    contracts.write_text(head + marker + tail)
    after = _sections(_via_transport(store, registry, contracts, transport, capsys, monkeypatch))["擺位標準檢查表"]
    first = _clause_block(before, entry.id)
    second = _clause_block(after, entry.id)
    assert "搜尋第一名（1 號）：守" in first
    assert "搜尋第一名（1 號）：沒守" in second


def _via_transport(store: SearchStore, registry: Path, contracts: Path, transport: str,
                   capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> str:
    if transport == "build":
        return _text(store, registry, contracts)
    from aosr.search import cli

    monkeypatch.setattr(cli, "config_path", lambda name: registry)
    before = _snapshot(store)
    code = cli.main(["report", "--search", str(store.path), "--contracts", str(contracts)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == "" and _snapshot(store) == before
    return captured.out


def test_both_unreadable_sides_still_list_every_clause(tmp_path: Path) -> None:
    store, registry = _store(tmp_path)
    store.baseline_path.write_text("broken")
    store.candidate_path(1).write_text("broken")
    section = _sections(_text(store, registry))["擺位標準檢查表"]
    source = config_path("placement_standards.toml")
    for entry in load_placement_standards(source).entries:
        block = _clause_block(section, entry.id)
        assert re.findall(re.escape(entry.quote), block) == [entry.quote]
        assert "\n  搜尋第一名（1 號）：結果檔讀不回，沒檢查" in block
        assert "\n  原方案：結果檔讀不回，沒檢查" in block


def test_existing_sections_stay_verbatim_after_refinement(tmp_path: Path) -> None:
    store, registry = _store(tmp_path)
    before = _sections(_text(store, registry))
    _refined(store, ((None, 8.0), (2, 1.0), (1, 2.0)))
    after = _sections(_text(store, registry))
    for heading in ("搜尋停了沒", "品質合不合格", "限制", "尚未評估", "範圍標記", "兩種參考分開寫"):
        assert after[heading] == before[heading]


@pytest.mark.parametrize("change,label", [("missing", "未評估"), ("identity", "不能同表")])
def test_result_ranking_zone_prevents_false_rank(tmp_path: Path, change: str, label: str) -> None:
    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (2, 1.0), (1, 2.0)))
    path = store.refine_result_path(1)
    result = SchemeResult.model_validate_json(path.read_bytes())
    evaluations = tuple(item for item in result.candidate.evaluations if item.category.value != "timbre_balance")
    if change == "identity":
        evaluations = tuple(item.model_copy(update={"evaluator_version": "zz-other"})
                            for item in result.candidate.evaluations)
    updated = result.model_copy(update={"candidate": result.candidate.model_copy(update={"evaluations": evaluations})})
    path.write_text(updated.model_dump_json())
    assert _sections(_text(store, registry))["名次"].splitlines()[0] == f"1 號：{label}。"


def test_search_first_not_yet_refined_preserves_search_rank(tmp_path: Path) -> None:
    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (2, 1.0)))
    assert _sections(_text(store, registry))["名次"].splitlines() == [
        "1 號：搜尋排名第 1；還沒有細算。",
        "2 號：搜尋排名第 2；在已細算的 2 個方案中，排名第 1。另有 4 個方案尚未細算。",
        "原方案：在已細算的 2 個方案中，排名第 2。",
    ]


# 值取自搜尋寫進狀態的那一組（run.py::pin_baseline 的 RankingZone 值與 illegal），不手打。
@pytest.mark.parametrize("zone,label", [(RankingZone.ELIMINATED.value, "淘汰"), (RankingZone.UNASSESSED.value, "未評估"),
                                        (RankingZone.INCOMPARABLE.value, "不能同表"), ("illegal", "不合法")])
def test_unrankable_original_without_refinement_reports_status(tmp_path: Path, zone: str, label: str) -> None:
    store, registry = _store(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    store.status_path.write_text(status.model_copy(update={"baseline_outcome": zone}).model_dump_json())
    assert _sections(_text(store, registry))["名次"].splitlines()[-1] == f"原方案：{label}。"


def test_checklist_uses_search_front_wall(tmp_path: Path) -> None:
    store, registry = _store(tmp_path)
    assert store.settings.layout.front_wall == "x0"
    result = SchemeResult.model_validate_json(store.candidate_path(1).read_bytes())
    scheme = with_points(result.scheme, Point(4.0, 5.0, 2.0), Point(6.0, 5.0, 2.0), Point(1.0, 7.0, 2.0))
    store.candidate_path(1).write_text(_with_scheme(result, scheme).model_dump_json())
    section = _sections(_text(store, registry))["擺位標準檢查表"]
    block = _clause_block(section, "ebu_a1_1_listener_walls")
    best = next(line for line in block.splitlines() if line.startswith("  搜尋第一名（1 號）："))
    # 前牆 x0：只量側牆 y0／yL 和後牆 xL，最近 7 m；誤用 y0 就會量到側牆 x0 的 1 m。
    assert "主位=7.0" in best and "最近面 y0" in best


@pytest.mark.parametrize("conclusion", get_args(OuterConclusion))
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("refine_state", ["not_started", "stopped"])
def test_completion_caveat_only_for_current_complete(
    tmp_path: Path, conclusion: OuterConclusion, stale: bool, refine_state: str,
) -> None:
    store, registry = _store(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    refinement = status.refine.model_copy(update={"state": refine_state,
                                                 "stop_reason": "stable" if refine_state == "stopped" else None})
    status = status.model_copy(update={"refine": refinement})
    outer = OuterStatus(conclusion=conclusion, message=OUTER_MESSAGES[conclusion], snapshot=snapshot_of(status))
    status = status.model_copy(update={"outer": outer, **({"asked": status.asked + 1} if stale else {})})
    store.status_path.write_text(status.model_dump_json())
    sections = _sections(_text(store, registry))
    caveat = "細算完成不代表全域最佳：回饋只在第一名附近找，搜尋取樣沒走到的範圍補不到（#611）"
    assert (caveat in sections["細算做完沒"]) == (conclusion == "complete" and not stale)
    if conclusion == "complete" and not stale:
        assert sections["細算做完沒"].endswith(
            "外圈結論：細算完成\n" + caveat + "\n這個搜尋資料夾沒有時間紀錄（第 585 支之前開的）"
        )
    assert "判定：未判定合格" in sections["品質合不合格"]


def test_failure_note_only_on_side_that_is_not_met(tmp_path: Path) -> None:
    """沒守的附註（例如離牆不足要另外控制早期反射）只掛在沒守的那一邊。房間 6×4×3 m。"""
    store, registry = _store(tmp_path)
    near = SchemeResult.model_validate_json(store.candidate_path(1).read_bytes())
    near_scheme = with_points(near.scheme, Point(0.5, 1.2, 1.5), Point(0.5, 2.8, 1.5), Point(2.5, 2.0, 1.5))
    store.candidate_path(1).write_text(_with_scheme(near, near_scheme).model_dump_json())
    far = SchemeResult.model_validate_json(store.baseline_path.read_bytes())
    far_scheme = with_points(far.scheme, Point(1.5, 1.2, 1.5), Point(1.5, 2.8, 1.5), Point(3.5, 2.0, 1.5))
    store.baseline_path.write_text(_with_scheme(far, far_scheme).model_dump_json())
    entry = next(item for item in load_placement_standards(config_path("placement_standards.toml")).entries
                 if item.failure_note is not None)
    note = entry.failure_note
    assert note is not None
    block = _clause_block(_sections(_text(store, registry))["擺位標準檢查表"], entry.id)
    near_line = next(line for line in block.splitlines() if line.startswith("  搜尋第一名（1 號）："))
    far_line = next(line for line in block.splitlines() if line.startswith("  原方案："))
    assert near_line.startswith("  搜尋第一名（1 號）：沒守；") and near_line.endswith("；" + note)
    assert far_line.startswith("  原方案：守；") and note not in far_line


def test_changed_scoring_settings_do_not_rerank_refined_results(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import report_comparison

    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (2, 1.0), (1, 2.0)))
    text = registry.read_text(encoding="utf-8")
    changed = text.replace("value = -10.0", "value = -11.0", 1)
    assert changed != text
    registry.write_text(changed, encoding="utf-8")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("評分設定不同不准呼叫排名")

    monkeypatch.setattr(report_comparison, "compare_results", forbidden)
    lines = _sections(_text(store, registry))["名次"].splitlines()
    assert lines == [
        "1 號：搜尋排名第 1；在已細算的 3 個方案中，排名第 2。另有 3 個方案尚未細算。評分設定跟搜尋快照不同，尚未重排。",
        "2 號：搜尋排名第 2；在已細算的 3 個方案中，排名第 1。另有 3 個方案尚未細算。評分設定跟搜尋快照不同，尚未重排。",
        "原方案：在已細算的 3 個方案中，排名第 3。評分設定跟搜尋快照不同，尚未重排。",
    ]


def test_refined_results_that_cannot_share_a_table_get_no_rank(tmp_path: Path) -> None:
    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (2, 1.0), (1, 2.0)))
    path = store.refine_result_path(2)
    result = SchemeResult.model_validate_json(path.read_bytes())
    path.write_text(result.model_copy(update={"physics_identity": "phys-v1:" + "c" * 64}).model_dump_json())
    lines = _sections(_text(store, registry))["名次"].splitlines()
    assert lines == ["1 號：不能同表。", "2 號：不能同表。", "原方案：不能同表。"]


def _refined_outcomes(store: SearchStore,
                      spec: tuple[tuple[int | None, Literal["scored", "not_comparable"], float | None], ...]) -> None:
    """照帳上的結果寫細算帳；不可比那幾份的結果檔換掉評估器版本，比較身分跟原方案不同。"""
    store.ensure_refine_dir()
    ledger = RefineLedger.create(store.refine_ledger_path, header_for(store))
    rows = []
    for number, outcome, cost in spec:
        source = store.baseline_path if number is None else store.candidate_path(number)
        result = SchemeResult.model_validate_json(source.read_bytes())
        if outcome == "not_comparable":
            candidate = result.candidate.model_copy(update={"evaluations": tuple(
                item.model_copy(update={"evaluator_version": "zz-other"}) for item in result.candidate.evaluations)})
            result = result.model_copy(update={"candidate": candidate})
        store.refine_result_path(number).write_text(result.model_dump_json())
        row = RefineRow(round=1, trial_number=number, result_file=refine_result_name(number),
                        outcome=outcome, total_cost=cost, seconds=0.0)
        ledger.append(row)
        rows.append(row)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    store.status_path.write_text(status.model_copy(update={"refine": _progress(rows, 1)}).model_dump_json())


def test_incomparable_refined_majority_does_not_unseat_scored_rows(tmp_path: Path) -> None:
    """帳上不可比的一組人數比較多，也不准拿來多數決主表、把有分數的第一名與原方案擠成不能同表。"""
    store, registry = _store(tmp_path)
    _refined_outcomes(store, ((None, "scored", 8.0), (1, "not_comparable", None), (2, "scored", 1.0),
                              (0, "not_comparable", None), (3, "not_comparable", None)))
    assert _sections(_text(store, registry))["名次"].splitlines() == [
        "1 號：不能同表。",
        "2 號：搜尋排名第 2；在已細算的 2 個方案中，排名第 1。另有 1 個方案尚未細算。",
        "原方案：在已細算的 2 個方案中，排名第 2。",
    ]


def test_no_first_place_checklist_says_so(tmp_path: Path) -> None:
    store, registry = _store(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    store.status_path.write_text(status.model_copy(update={"best_trial": None, "best_score": None}).model_dump_json())
    section = _sections(_text(store, registry))["擺位標準檢查表"]
    for entry in load_placement_standards(config_path("placement_standards.toml")).entries:
        block = _clause_block(section, entry.id)
        assert "\n  搜尋第一名：沒有第一名，沒檢查" in block
        assert "結果檔讀不回" not in block


def test_degenerate_original_geometry_keeps_report(tmp_path: Path) -> None:
    """原方案兩支喇叭水平重疊（基寬水平分量為零），檢查表那一邊照實寫算不出來，其他段照印。"""
    store, registry = _store(tmp_path)
    result = SchemeResult.model_validate_json(store.baseline_path.read_bytes())
    scheme = with_points(result.scheme, Point(4.0, 5.0, 2.0), Point(4.0, 5.0, 3.0), Point(5.0, 8.0, 2.0))
    store.baseline_path.write_text(_with_scheme(result, scheme).model_dump_json())
    sections = _sections(_text(store, registry))
    block = _clause_block(sections["擺位標準檢查表"], "itu_8_5_1_1_height")
    original = next(line for line in block.splitlines() if line.startswith("  原方案："))
    assert original.startswith("  原方案：幾何算不出來，沒檢查：") and "退化" in original
    assert "\n  搜尋第一名（1 號）：" in block and "沒檢查" not in block.split("\n  原方案：")[0]
    assert "判定：未判定合格" in sections["品質合不合格"]


def test_ranking_failure_is_reported_verbatim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import report_comparison

    store, registry = _store(tmp_path)
    _refined(store, ((None, 8.0), (2, 1.0), (1, 2.0)))

    def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("登記簿設定矛盾")

    monkeypatch.setattr(report_comparison, "compare_results", broken)
    lines = _sections(_text(store, registry))["名次"].splitlines()
    assert lines and all(line.endswith("排名失敗：登記簿設定矛盾。") for line in lines)
