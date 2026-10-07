"""報告考卷：停止、細算、品質分開；讀報告不能改搜尋資料。"""

from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
import re

import pytest
from pydantic import ValidationError

from aosr.search.cli import main
from aosr.config.frequency_axis import GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ, LowFrequencyAxis
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point
from aosr.physics.report_io import CapabilitySection, ReportOutput, SceneSection, TopFields
from aosr.physics.report_source import SourceModelKind, SourceModelSection, SourceModelSpec
from aosr.physics.reflection_screen import ReflectionScreen, WallPairRow
from aosr.physics.third_octave_decay import ThirdOctaveDecay
from aosr.reporting.result import PairResult, ResultOrigin, SchemeResult, Timings
from aosr.reporting.scheme import Scheme, expected_pairs, pair_input_document
from aosr.scoring.ranking_models import CandidateStatus, ExternalAcceptance
from aosr.scoring.recommendation import NotFinalReason, RecommendationStatus, ReviewStatus
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.sampler import RankingZone
from aosr.search.store import SearchStore
from tests.engine._search_run_cases import FakeCompute, RUN_DATE, make_store, run
from tests.engine._reflection_fixtures import _band
from tests.engine._search_review_cases import with_matching


def _pair(scheme: Scheme, speaker: str, receiver_id: str, role: str) -> PairResult:
    """純合成的完整零件：只填模型，不呼叫任何物理解算或反射路徑計算。"""
    source = scheme.speakers[speaker]
    receiver = Point(*next(point.position_m for point in scheme.receiver_set.points
                           if point.receiver_id == receiver_id))
    scene = SceneSection(scene_fingerprint="a" * 64, source_m=source, receiver_m=receiver,
                         source_model=SourceModelSection.from_spec(
                             SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL), None))
    report = ReportOutput(
        scene=scene, capability=CapabilitySection(frequency_hz=(), outputs=(), status="experimental", evidence=()),
        top=TopFields(f_s_hz=200.0, crossover_lower_hz=100.0, crossover_upper_hz=300.0,
                      capped_by_upper_limit=False, reflection_order_k=3, low_frequency_axis=LowFrequencyAxis.SEARCH,
                      eyring_t60_by_band_s={}, room_volume_m3=72.0, schroeder_band_count=0),
        bands=tuple(_band(center) for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ),
    )
    screen = ReflectionScreen(
        scene_fingerprint=scene.scene_fingerprint, source_m=source, receiver_m=receiver,
        reflection_order_k=3, frequencies_hz=(1000.0,),
        pairs=tuple(WallPairRow.model_validate(dict(pair=axis, faces=faces, distance_m=3.0, round_trip_delay_s=0.02,
                               face_retained_energy=((0.5,), (0.5,)), round_trip_retained_energy=(0.25,)))
                    for axis, faces in (("x", ("x0", "xL")), ("y", ("y0", "yL")), ("z", ("floor", "ceiling")))),
    )
    return PairResult(role=role, speaker_id=speaker, receiver_id=receiver_id,
                      report_id=f"report-{speaker}-{receiver_id}", report=report, screen=screen,
                      third_octave_decay=ThirdOctaveDecay(scene_fingerprint=scene.scene_fingerprint, rows=()),
                      input_document=pair_input_document(scheme, source, receiver, {"kind": "omnidirectional"}))


class _ResultCompute(FakeCompute):
    """保留假搜尋的候選與排名，只把空檔補成真正可驗的結果檔。"""

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for computed in super().__call__(jobs, workers):
            job = computed.job
            # 假計算不評聲源；合成結果明示全向，以便用純資料建零件。
            scheme = job.scheme.model_copy(update={"source_model": "omnidirectional"})
            result = SchemeResult(
                schema_version="aosr.scheme_result.v4", scheme=scheme,
                engine_commit="test", program_fingerprint=self.store.identity.program_fingerprint,
                physics_identity=self.store.identity.physics_identity,
                purpose_settings=self.store.identity.purpose_settings,
                origin=ResultOrigin(kind="search_baseline" if job.trial_number is None else "search_candidate",
                                    search_id=self.store.search_id, trial_number=job.trial_number),
                scope="stage_two_subset", run_date=RUN_DATE,
                quality_targets_fingerprint="a" * 64,
                timings=Timings(solve_s=0.0, output_s=0.0, evaluate_s=0.0, total_s=0.0),
                pairs=tuple(_pair(scheme, speaker, receiver, role)
                            for speaker, receiver, role in expected_pairs(scheme)), candidate=computed.candidate,
            )
            job.result_path.write_text(result.model_dump_json(), encoding="utf-8")
            yield computed


def _finished(tmp_path: Path) -> tuple[SearchStore, Path]:
    # 各題需變更的限制在自己那題傳入；不把測試答案藏在夾具裡。
    store, registry = make_store(tmp_path, budget=3)
    run(store, registry, _ResultCompute(store))
    return store, registry


def _snapshot(store: SearchStore) -> tuple[tuple[str, ...], dict[str, bytes]]:
    paths = tuple(sorted(str(path.relative_to(store.path)) for path in store.path.rglob("*")))
    contents = {str(path.relative_to(store.path)): path.read_bytes()
                for path in store.path.rglob("*") if path.is_file()}
    return paths, contents


def _sections(text: str) -> dict[str, str]:
    # 標題唯一性用切割結果檢查，不以數量斷言鎖死報表長度。
    return {part.splitlines()[0]: "\n".join(part.splitlines()[1:])
            for part in text.strip().split("\n\n")}


def test_three_states_and_stop_snapshot_are_separate(tmp_path: Path) -> None:
    from aosr.search.report import build_report, render_text

    store, registry = _finished(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    for field in ("state", "message", "asked", "computed", "illegal", "illegal_reasons",
                  "excluded", "best_trial", "best_score", "streak", "start_enqueued"):
        assert getattr(report.search, field) == getattr(status, field)
    assert report.search.budget == store.settings.budget
    assert report.search.convergence_run == store.settings.convergence_run
    assert report.refinement.state == "not_started"
    assert report.refinement.message == "細算未開始：還沒有任何候選用驗證軸細算"
    assert report.quality.verdict == "未判定合格"
    assert "合格規則還沒拍板" in report.quality.reason
    sections = _sections(render_text(report))
    assert "合格" not in sections["搜尋停了沒"]
    assert "細算" not in sections["搜尋停了沒"]
    assert status.message in sections["搜尋停了沒"]
    assert "暫行的工程停止設定" in sections["搜尋停了沒"]
    assert "未判定合格" in sections["品質合不合格"]
    # 判定那一行本身要寫未判定；其他行（訊息、原因）出現「未判定合格」不能代替。
    verdicts = [line for line in sections["品質合不合格"].splitlines() if line.startswith("判定：")]
    assert verdicts == ["判定：未判定合格"]
    assert "未開始" in sections["細算做完沒"]


def test_stop_counts_have_chinese_explanations(tmp_path: Path) -> None:
    from aosr.search.report import build_report, render_text

    store, registry = _finished(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes()).model_copy(update={
        "illegal_reasons": {"cabinet_outside_room": 2},
        # 鍵照 run.py::_progress 寫進狀態檔的排名區值（outcome.zone.value）。
        "excluded": {RankingZone.ELIMINATED.value: 1, RankingZone.UNASSESSED.value: 2,
                     RankingZone.INCOMPARABLE.value: 3},
    })
    store.status_path.write_text(status.model_dump_json(), encoding="utf-8")
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    text = _sections(render_text(report))["搜尋停了沒"]
    assert "箱體越界：2" in text
    assert "淘汰：1" in text
    assert "未評估：2" in text
    assert "不能同表：3" in text
    assert "未辨識" not in text


def test_report_sections_are_frozen_and_forbid_extra_fields(tmp_path: Path) -> None:
    from aosr.search.report import build_report

    store, registry = _finished(tmp_path)
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    for model in (report, report.search, report.refinement, report.quality,
                  report.quality.original, report.quality.best, report.restrictions,
                  report.unassessed, report.scope, report.references, report.modal, report.crossover):
        field = next(iter(type(model).model_fields))
        with pytest.raises(ValidationError, match="frozen_instance"):
            setattr(model, field, getattr(model, field))
        with pytest.raises(ValidationError, match="extra_forbidden"):
            type(model).model_validate(model.model_dump() | {"unexpected_field": "不接受"})


def test_quality_preserves_ranking_evidence_and_never_final(tmp_path: Path) -> None:
    from aosr.search.report import build_report, render_text

    store, registry = _finished(tmp_path)
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    # 推法：本用途必評類清單本身是暫定，參與表頭判定；未宣告外部底線，驗收未檢查。
    # 假計算的安靜峰谷無複核警戒，兩類齊全且代價有限，因此兩份均可排名。
    purpose = load_quality_targets(registry).purpose(store.project.purpose)
    mandatory = next(item for item in purpose.qualification if item.key == "ranking.mandatory_categories")
    assert mandatory.status == "baseline"
    header = report.quality.header
    assert header is not None
    assert header.calibration == "baseline"
    assert header.calibration_note == "未校準，不是品質判決"
    assert header.overall_acceptance == ExternalAcceptance.NOT_CHECKED
    assert header.acceptance_note == "外部底線未宣告或未檢查：整體驗收未檢查，不是整體合格"
    for candidate in (report.quality.original, report.quality.best):
        assert candidate.zone == CandidateStatus.RANKABLE
        assert candidate.review_status == ReviewStatus.CLEAR
        assert candidate.recommendation_status == RecommendationStatus.NOT_FINAL
        assert candidate.not_final_reasons == (NotFinalReason.EXTERNAL_NOT_CHECKED,
                                               NotFinalReason.CALIBRATION_BASELINE,
                                               NotFinalReason.NO_FINALIZING_PROCESS)
    assert report.quality.best.message == "不是最終推薦"
    text = render_text(report)
    assert "第一名：不是最終推薦" in text
    assert "暫定" in text
    assert "整體驗收：未檢查" in text
    assert "複核狀態：目前沒有產生複核警戒" in text
    assert "calibration_baseline（品質登記簿條目仍為暫定）" in text


@pytest.mark.parametrize("change,zone,label", [
    ("breached", CandidateStatus.ELIMINATED, "淘汰"),
    ("missing", CandidateStatus.NOT_EVALUATED, "未評估"),
    ("identity", CandidateStatus.NOT_COMPARABLE, "不能同表"),
    ("physics", CandidateStatus.NOT_COMPARABLE, "不能同表"),
])
def test_quality_keeps_each_nonrankable_zone(
    tmp_path: Path, change: str, zone: CandidateStatus, label: str,
) -> None:
    from aosr.search.report import build_report, render_text

    store, registry = _finished(tmp_path)
    original = SchemeResult.model_validate_json(store.baseline_path.read_bytes())
    candidate = original.candidate
    if change == "breached":
        # 正式聲道底線樣本的最差音量差 100，超過登記簿上限，應淘汰。
        candidate = with_matching(candidate, breached=True)
    elif change == "missing":
        # 缺少唯一必評類音色，不可當成可排名或淘汰。
        candidate = candidate.model_copy(update={"evaluations": tuple(
            item for item in candidate.evaluations if item.category.value != "timbre_balance")})
    elif change == "identity":
        # 同票數時取身分字典序較小的主表；z 開頭的原方案比較身分落到不能同表。
        candidate = candidate.model_copy(update={"evaluations": tuple(
            item.model_copy(update={"evaluator_version": "zz-other"}) for item in candidate.evaluations)})
    updated = original.model_copy(update={"candidate": candidate, **(
        {"physics_identity": "phys-v1:" + "c" * 64} if change == "physics" else {})})
    store.baseline_path.write_text(updated.model_dump_json(), encoding="utf-8")
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert report.quality.original.zone == zone
    assert report.quality.original.review_status is None
    assert report.quality.original.recommendation_status is None
    assert report.quality.original.not_final_reasons == ()
    assert f"原方案所在區：{label}" in render_text(report)
    assert report.quality.verdict == "未判定合格"
    if change == "physics":
        # 不能同表要寫出是哪一項固定身分不同（compare.py::comparison_problems 的原文）。
        assert "不能同表" in report.quality.message and "物理" in report.quality.message
        # 原文點名的方案代號換成原方案、第一名；不留以 -baseline 結尾的代號。
        assert "原方案" in report.quality.message and "第一名" in report.quality.message
        assert store.search_id not in report.quality.message
        assert re.search(r"\bbaseline\b", render_text(report), flags=re.I) is None


def test_ranking_error_is_reported_verbatim_not_as_incomparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """排名層自己的錯照原文寫，不冒充不能同表、不猜所在區。"""
    from aosr.reporting import compare
    from aosr.search.report import build_report, render_text

    store, registry = _finished(tmp_path)

    def broken(*args: object, **kwargs: object) -> object:
        raise ValueError("排名層測試原文")

    monkeypatch.setattr(compare, "rank_candidates", broken)
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert "排名層測試原文" in report.quality.message
    assert "不能同表" not in report.quality.message
    assert report.quality.original.zone is None and report.quality.best.zone is None
    assert "不能同表" not in _sections(render_text(report))["品質合不合格"]


BOX = {"x": {"low": 0.0, "high": 6.0}, "y": {"low": 0.0, "high": 6.0},
       "z": {"low": 0.0, "high": 3.0}}
# 參考房 6×4×3、前牆 x0：後段整幅寬的盒子；x、y 不對稱，軸寫反時考卷看得出來。
REAR_BOX = {"x": {"low": 5.0, "high": 6.0}, "y": {"low": 0.0, "high": 4.0},
            "z": {"low": 0.0, "high": 3.0}}


@pytest.mark.parametrize("key,label,value,shown", [
    ("wall_gap_m", "離牆間隙", 0.1, "0.1 公尺"),
    ("keep_out", "禁區", [REAR_BOX], "房間座標 x 5.0～6.0 公尺、y 0.0～4.0 公尺、高度 z 0.0～3.0 公尺"),
    ("speaker_areas", "喇叭可用區", [REAR_BOX], "房間座標 x 5.0～6.0 公尺、y 0.0～4.0 公尺、高度 z 0.0～3.0 公尺"),
    ("listening_range_m", "型號適用聆聽距離（喇叭聲學中心到主位的三維距離）", {"low": 0.1, "high": 9.0}, "0.1～9.0 公尺"),
    ("base_angle_deg", "水平夾角", {"low": 10.0, "high": 120.0}, "10.0～120.0 度"),
])
def test_each_constraint_declared_and_undeclared(
    tmp_path: Path, key: str, label: str, value: object, shown: str,
) -> None:
    from aosr.search.report import build_report, render_text

    for name, changes in (("unset", {}), ("set", {key: value})):
        store, registry = make_store(tmp_path / name, budget=2, layout_changes=changes)
        run(store, registry, _ResultCompute(store))
        text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
        line = next(line for line in _sections(text)["限制"].splitlines() if line.startswith(label + "："))
        if changes:
            assert "未限制" not in line
            assert shown in line
        else:
            assert line == label + "：未限制"
        assert "不限制" not in text


def test_no_scored_candidate_still_reports_original(tmp_path: Path) -> None:
    from aosr.search.report import build_report, render_text

    store, registry = make_store(tmp_path, budget=2, layout_changes={"keep_out": [BOX]})
    status = run(store, registry, _ResultCompute(store))
    assert status.best_trial is None
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert "沒有第一名" in report.quality.best.message
    assert report.quality.best.zone is None
    assert report.quality.original.zone == CandidateStatus.RANKABLE
    assert "沒有任何候選拿到分數" in render_text(report)


@pytest.mark.parametrize("broken", ["missing", "invalid"])
@pytest.mark.parametrize("which", ["original", "best"])
def test_unreadable_result_does_not_break_report(tmp_path: Path, broken: str, which: str) -> None:
    from aosr.search.report import build_report, render_text

    store, registry = _finished(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.best_trial is not None
    path = store.baseline_path if which == "original" else store.candidate_path(status.best_trial)
    if broken == "missing":
        path.unlink()
    else:
        path.write_text("not a result", encoding="utf-8")
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    candidate = report.quality.original if which == "original" else report.quality.best
    assert "結果檔讀不回" in candidate.message
    assert report.scope.message == "第二階段子集"
    assert "未判定合格" in render_text(report)


def test_changed_scoring_settings_are_not_reranked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import report as module

    store, registry = _finished(tmp_path)
    text = registry.read_text(encoding="utf-8")
    changed = text.replace("value = -10.0", "value = -11.0", 1)
    assert changed != text
    registry.write_text(changed, encoding="utf-8")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("評分設定不同不准呼叫排名")

    monkeypatch.setattr(module, "compare_results", forbidden)
    report = module.build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert report.quality.message == "評分設定跟搜尋快照不同，這份報告不重排"
    assert report.quality.header is None
    assert report.quality.original.zone is None and report.quality.best.zone is None
    assert "不重排" in module.render_text(report)


@pytest.mark.parametrize("angle", [None, {"low": 10.0, "high": 120.0}])
def test_scope_unassessed_and_angle_caveat(tmp_path: Path, angle: dict[str, float] | None) -> None:
    from aosr.search.report import build_report, render_text

    store, registry = make_store(tmp_path, budget=2, layout_changes={"base_angle_deg": angle})
    run(store, registry, _ResultCompute(store))
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert report.unassessed.items == ("製作用途", "多人座位", "物件反射", "箱體反射")
    text = render_text(report)
    assert ("夾角限制不代表空間感已評估" in text) == (angle is not None)
    assert "第二階段子集" in text
    assert "低頻拖尾不計分，另有模態診斷報告" in text
    assert "原方案：專案本來的擺法" in text
    assert "暫定：品質登記簿裡還沒校準的條目" in text
    assert re.search(r"\bbaseline\b", text, flags=re.I) is None
    titles = [part.splitlines()[0] for part in text.strip().split("\n\n")]
    for title in ("搜尋停了沒", "細算做完沒", "品質合不合格"):
        first, *remaining = [found for found in titles if found == title]
        assert first == title and not remaining


def test_build_and_cli_leave_every_file_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.search import cli
    from aosr.search.report import build_report, render_text

    store, registry = _finished(tmp_path)
    before = _snapshot(store)
    report = build_report(store, quality_targets_path=registry, run_date=date.today())
    assert _snapshot(store) == before
    monkeypatch.setattr(cli, "config_path", lambda name: registry)
    code = main(["report", "--search", str(store.path)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == render_text(report)
    assert captured.err == ""
    assert _snapshot(store) == before


def test_cli_report_failure_preserves_running_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.search import cli

    store, registry = _finished(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes()).model_copy(update={"state": "running"})
    store.status_path.write_text(status.model_dump_json(), encoding="utf-8")
    before = _snapshot(store)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("報告測試錯誤原文")

    monkeypatch.setattr(cli, "build_report", fail)
    code = main(["report", "--search", str(store.path)])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "報告測試錯誤原文" in captured.err
    assert _snapshot(store) == before


def test_report_missing_search_is_stderr_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """缺少報告入口或誤用搜尋失敗寫狀態的分支，都不能滿足唯讀錯誤回應。"""
    missing = tmp_path / "missing"
    code = main(["report", "--search", str(missing)])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such file or directory" in captured.err
    assert not missing.exists()
