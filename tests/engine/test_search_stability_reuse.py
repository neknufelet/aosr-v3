"""細算帳與交接判舊、內容雜湊、出處核對和兩代清檔的可觀察考卷。"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Point
from aosr.reporting.result import ResultOrigin, SchemeResult
from aosr.search import crossover_record
from aosr.search.placement_stability import SHIFT_NAMES
from aosr.search.placement_stability_record import StabilitySummary, read_summary
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore, refine_scheme_id, refine_result_name
from tests.engine._stability_attach_cases import ShiftCompute, attach, pairs, ready


def change_rows(store: SearchStore) -> None:
    header, rows = RefineLedger.read(store.refine_ledger_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header)
    for row in rows:
        book.append(row.model_copy(update={"total_cost": math.nextafter(row.total_cost, math.inf)})
                    if row.trial_number == 7 and row.total_cost is not None else row)


def move_refined(store: SearchStore, number: int) -> None:
    result = SchemeResult.model_validate_json(store.refine_result_path(number).read_bytes())
    scheme = result.scheme.model_copy(update={"speakers": {
        key: Point(point.x + 0.001, point.y, point.z) for key, point in result.scheme.speakers.items()}})
    result = result.model_copy(update={"scheme": scheme, "pairs": pairs(scheme)})
    store.refine_result_path(number).write_text(result.model_dump_json())
    store.scheme_path_for(store.refine_result_path(number)).write_text(scheme.model_dump_json())
    change_rows(store)


def test_stale_crossover_uses_only_top_three_and_original(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    summary = crossover_record.fresh_summary(store, status).model_copy(update={"state": "done", "completed": True,
        "rows_fingerprint": "old", "variants": (crossover_record.VariantRecord(key="legacy", label="舊", basis="題目",
            ranking=(crossover_record.CostRow(trial_number=999, total_cost=0),)),)})
    crossover_record.write_summary(store.path, summary)
    result = attach(store, registry, status, ShiftCompute(store))
    assert [f.trial_number for f in result.selection.finalists] == [7, 9, None]
    assert all(w.winner is None and "交接摘要舊了" in w.reason_text for w in result.selection.crossover_winners)


def test_refinement_stamp_change_refreshes_but_reuses_all_points(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    change_rows(store)
    compute = ShiftCompute(store)
    second = attach(store, registry, status, compute)
    assert second.rows_fingerprint != first.rows_fingerprint and second.rows != first.rows
    assert not compute.batches and second.completed


def test_new_crossover_winner_only_computes_missing_finalist(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    number = 11
    result = SchemeResult.model_validate_json(store.refine_result_path(None).read_bytes())
    identifier = refine_scheme_id(store.search_id, number)
    scheme = result.scheme.model_copy(update={"scheme_id": identifier})
    from aosr.scoring.contract import CandidateEvaluation
    candidate = CandidateEvaluation.model_validate_json(result.candidate.model_dump_json().replace(result.scheme.scheme_id, identifier))
    result = result.model_copy(update={"scheme": scheme, "candidate": candidate, "pairs": pairs(scheme),
        "origin": ResultOrigin(kind="search_candidate", search_id=store.search_id, trial_number=number)})
    store.refine_result_path(number).write_text(result.model_dump_json())
    store.scheme_path_for(store.refine_result_path(number)).write_text(scheme.model_dump_json())
    baseline_row = RefineLedger.read(store.refine_ledger_path)[1][0]
    RefineLedger(store.refine_ledger_path).append(RefineRow(round=1, trial_number=number, result_file=refine_result_name(number),
        outcome="scored", total_cost=baseline_row.total_cost, seconds=0))
    fresh = crossover_record.fresh_summary(store, status).model_copy(update={"state": "done", "completed": True,
        "official_best": 7, "variants": tuple(crossover_record.VariantRecord(key=key, label=key, basis="題目",
            ranking=(crossover_record.CostRow(trial_number=number, total_cost=0),)) for key in ("legacy", "smooth_150_300", "hard_300"))})
    # 接法代號從第一步公開清單讀，題目刻意讓同一名在各接法贏，去重只算一次。
    from aosr.search.labels import STABILITY_CROSSOVERS
    fresh = fresh.model_copy(update={"variants": tuple(v.model_copy(update={"key": key})
        for v, key in zip(fresh.variants, STABILITY_CROSSOVERS, strict=True))})
    crossover_record.write_summary(store.path, fresh)
    compute = ShiftCompute(store)
    second = attach(store, registry, status, compute)
    assert {j.trial_number for j in compute.jobs} == {number}
    assert {j.shift_name for j in compute.jobs} == set(SHIFT_NAMES)
    assert second.total_points == first.total_points + len(SHIFT_NAMES)


def test_point_reuse_requires_scheme_content_hash(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    move_refined(store, 7)
    compute = ShiftCompute(store)
    second = attach(store, registry, status, compute)
    assert {j.trial_number for j in compute.jobs} == {7}
    assert {j.shift_name for j in compute.jobs} == set(SHIFT_NAMES)
    assert {p.scheme_hash for p in first.points if p.trial_number == 7}.isdisjoint(
        p.scheme_hash for p in second.points if p.trial_number == 7)


@pytest.mark.parametrize("corruption", ["broken", "origin", "content", "identity"])
def test_unreadable_or_wrong_reuse_result_only_recomputes_that_point(tmp_path: Path, corruption: str) -> None:
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    point = first.points[0]
    assert point.result_file is not None
    path = store.path / point.result_file
    result = SchemeResult.model_validate_json(path.read_bytes())
    if corruption == "broken":
        path.write_text("{broken")
    else:
        changes = {"origin": ResultOrigin(kind="search_candidate", search_id=store.search_id, trial_number=7)} if corruption == "origin" else (
            {"program_fingerprint": "calc-v1:" + "c" * 64} if corruption == "identity" else
            {"scheme": result.scheme.model_copy(update={"purpose": "different"})})
        path.write_text(result.model_copy(update=changes).model_dump_json())
    change_rows(store)
    compute = ShiftCompute(store)
    assert attach(store, registry, status, compute).completed
    assert [(j.trial_number, j.shift_name) for j in compute.jobs] == [(point.trial_number, point.shift_name)]


def test_cleanup_preserves_new_and_previous_generation_only(tmp_path: Path) -> None:
    from aosr.search.placement_stability_record import summary_path
    from aosr.search.placement_stability_attach import attach_stability
    from aosr.search.run import Compute
    from tests.engine._search_run_cases import RUN_DATE
    store, registry, status = ready(tmp_path)
    def run() -> StabilitySummary:
        compute = ShiftCompute(store)
        def factory(root: Path) -> Compute:
            root.mkdir()
            (root / "slice.json").write_text("考卷分片")
            return compute
        result = attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
            probe=lambda: store.identity, compute_factory=factory)
        for job in compute.jobs:
            SearchStore.stderr_path_for(job.result_path).write_text("考卷錯誤輸出")
        return result
    older = run()
    for _ in range(4):
        previous = older
        move_refined(store, 7)
        current = run()
        referenced = {store.path / p.result_file for summary in (previous, current) for p in summary.points if p.result_file}
        actual = {p for p in summary_path(store.path).parent.glob("point-*.json") if not p.name.endswith("-scheme.json")}
        assert actual == referenced
        assert all(SearchStore.scheme_path_for(p).is_file() for p in referenced)
        assert all(SearchStore.stderr_path_for(p).is_file() for p in referenced)
        fems = {store.path / p.fem_root for summary in (previous, current) for p in summary.points if p.fem_root}
        assert set(summary_path(store.path).parent.glob("fem-*")) == fems
        assert all((p / "slice.json").is_file() for p in fems)
        older = current


def test_changed_broken_crossover_is_stale_even_without_winners(tmp_path: Path) -> None:
    from aosr.search.placement_stability_attach import fresh_summary
    from aosr.search.placement_stability_record import is_stale
    store, registry, status = ready(tmp_path)
    path = crossover_record.summary_path(store.path)
    path.parent.mkdir()
    path.write_text("{broken-first")
    first = attach(store, registry, status, ShiftCompute(store))
    path.write_text("{broken-second")
    assert is_stale(first, fresh_summary(store, status))
    compute = ShiftCompute(store)
    second = attach(store, registry, status, compute)
    assert second.crossover_stamp != first.crossover_stamp and not compute.batches


def test_protected_helpers_exclude_attachment_but_guard_verification(tmp_path: Path) -> None:
    from tests.engine._crossover_cases import protected as crossover_protected
    from tests.engine._search_modal_cases import protected as modal_protected
    store, _, _ = ready(tmp_path)
    for protected in (crossover_protected, modal_protected):
        folder = store.path / "placement-stability"
        (folder / "owned").unlink(missing_ok=True)
        before = protected(store)
        folder.mkdir(exist_ok=True)
        (folder / "owned").write_text("附件")
        assert protected(store) == before
        path = store.refine_result_path(7)
        old = path.read_bytes()
        path.write_text("細算不能動")
        assert protected(store) != before
        path.write_bytes(old)


def test_completed_cleanup_removes_only_owned_temporary_files_without_following_links(tmp_path: Path) -> None:
    import os
    from tempfile import mkstemp
    from uuid import uuid4
    from aosr.search.placement_stability_record import summary_path
    store, registry, status = ready(tmp_path)
    attach(store, registry, status, ShiftCompute(store))
    root = summary_path(store.path).parent
    descriptor, name = mkstemp(dir=root, prefix=f".point-{uuid4().hex}.json.", suffix=".tmp")
    os.close(descriptor)
    garbage = (root / "summary-write-abcd1234.tmp", Path(name))
    preserved = (root / "summary-write-abcd1234.json", root / "tmpabcd1234", root / "tmpabcd1234.json",
        root / "tmpabc", root / "unknown.tmp", root / ".point-short.json.abcd1234.tmp",
        root / (".point-" + "f" * 32 + ".stderr.abcd1234.tmp"))
    outside = tmp_path / "summary-write-outside.tmp"
    outside.write_text("附件外不刪")
    for path in (*garbage, *preserved):
        path.write_text("暫存樣本")
    link = root / "tmpabcdefgh"
    link.symlink_to(outside)
    directory = root / "tmpijklmnop"
    directory.mkdir()
    point_link = root / ("point-" + "f" * 32 + ".json")
    point_link.symlink_to(outside)
    change_rows(store)
    failing = ShiftCompute(store, fail_after=0)
    # 刻意少掉一點，讓這一輪真進計算；失敗不能提早清檔。
    current = read_summary(store.path)
    assert current is not None and current.points[0].result_file is not None
    (store.path / current.points[0].result_file).unlink()
    with pytest.raises(RuntimeError):
        attach(store, registry, status, failing)
    assert all(p.is_file() for p in garbage)
    assert attach(store, registry, status, ShiftCompute(store)).state == "done"
    assert not any(p.exists() for p in garbage)
    assert all(p.is_file() for p in preserved) and outside.read_text() == "附件外不刪"
    assert link.is_symlink() and point_link.is_symlink() and directory.is_dir()


def test_crossover_validation_reason_is_one_line(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    path = crossover_record.summary_path(store.path)
    path.parent.mkdir()
    path.write_text('{"state":"unknown","completed":"bogus"}')
    summary = attach(store, registry, status, ShiftCompute(store))
    for winner in summary.selection.crossover_winners:
        assert "交接摘要讀不到" in winner.reason_text
        assert "；" in winner.reason_text
        assert "\n" not in winner.reason_text and "\r" not in winner.reason_text


def test_cleanup_does_not_follow_attachment_folder_symlink(tmp_path: Path) -> None:
    from aosr.search import placement_stability_attach as module
    from aosr.search.placement_stability_record import summary_path
    store, _, status = ready(tmp_path)
    outside = tmp_path / "linked"
    outside.mkdir()
    temporary = outside / "tmpabcdefgh"
    temporary.write_text("連結外的檔不刪")
    summary_path(store.path).parent.symlink_to(outside, target_is_directory=True)
    summary = module._empty(store, status).model_copy(update={"state": "skipped", "completed": True})
    module._finish(store.path, summary, None)
    assert temporary.read_text() == "連結外的檔不刪"
