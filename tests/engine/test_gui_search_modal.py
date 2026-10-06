"""搜尋進度新增附件區塊，各態與讀取錯誤只影響附件。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.gui.search_view import build_search_view
from aosr.search.modal_record import ModalRole, ModalSummary, summary_path, write_summary
from tests.engine._search_modal_cases import prepared, protected


@pytest.mark.parametrize("state,label", [
    ("not_started", "未開始"), ("running", "開始過、沒有收尾紀錄"), ("diagnosed_not_scored", "已診斷不計分"),
    ("failed", "失敗"), ("stopped", "已停止"), ("skipped", "跳過"), ("out_of_scope", "範圍外"),
])
def test_modal_block_states_and_reasons(tmp_path: Path, state: str, label: str) -> None:
    store, _, _ = prepared(tmp_path)
    role = ModalRole.model_validate({"role": "baseline", "state": state, "reason_text": "明列原因原文"})
    write_summary(store.path, ModalSummary(cache_dir="cache", completed=state != "running", roles=(role,)))
    before = protected(store)
    view = build_search_view(store.path, server_physics="test", server_program="test")
    block = next(b for b in view.blocks if b.key == "modal")
    text = "\n".join(block.lines)
    assert label in text and "明列原因原文" in text
    assert "不計分、不改名次" in text
    if state == "running":
        assert "上次沒做完（可能進行中或被中斷）" in text
        assert "進行中" not in next(line for line in block.lines if line.startswith("原方案："))
    assert protected(store) == before


def test_bad_modal_summary_does_not_hide_search_or_refine(tmp_path: Path) -> None:
    store, _, _ = prepared(tmp_path)
    summary_path(store.path).parent.mkdir()
    summary_path(store.path).write_text("{")
    view = build_search_view(store.path, server_physics="test", server_program="test")
    blocks = {block.key: block for block in view.blocks}
    assert blocks["modal"].warning and "讀不到" in "\n".join(blocks["modal"].lines)
    assert "搜尋：因預算停止" in "\n".join(blocks["stage"].lines)
    assert not blocks["counts"].warning


@pytest.mark.parametrize("change", ["snapshot", "conclusion", "search_best", "refine_best"])
def test_modal_block_marks_stale_with_same_rule_as_report(tmp_path: Path, change: str) -> None:
    from aosr.config.quality_targets import load_quality_targets
    from aosr.search.modal_record import role_inputs
    from aosr.search.outer_status import snapshot_of
    from aosr.search.report_modal import modal_report
    from tests.engine._search_modal_cases import change_first
    store, registry, status = prepared(tmp_path)
    summary = ModalSummary(cache_dir="cache", completed=True, conclusion=status.outer.conclusion,
                           snapshot=snapshot_of(status), roles=tuple(i.record for i in role_inputs(store, status)))
    write_summary(store.path, summary)
    initial = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "modal")
    note = "這是上次收尾時的診斷，之後搜尋又動過"
    assert note not in initial.lines
    if change in ("search_best", "refine_best"):
        change_first(store, change)
    else:
        summary = summary.model_copy(update={"snapshot": summary.snapshot.model_copy(update={"asked": status.asked + 1})} if change == "snapshot" else {"conclusion": "complete"})
        write_summary(store.path, summary)
    before = protected(store)
    block = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "modal")
    report = modal_report(store, status, load_quality_targets(registry))
    assert note in block.lines and note in report.lines
    assert protected(store) == before


@pytest.mark.parametrize("has_summary", [False, True])
def test_unfinished_modal_with_folder_lock_does_not_guess_progress(tmp_path: Path, has_summary: bool) -> None:
    from aosr.search.cli import _search_lock
    store, _, _ = prepared(tmp_path)
    if has_summary:
        write_summary(store.path, ModalSummary(cache_dir="cache", roles=(ModalRole(role="baseline", state="running"),)))
    with _search_lock(store):
        block = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "modal")
    assert block.lines[0] == "有計算行程拿著這個資料夾；附件尚未收尾"
    assert ("開始過、沒有收尾紀錄" in "\n".join(block.lines)) is has_summary
    assert all("進行中" not in line and "上次沒做完" not in line for line in block.lines)


def test_modal_block_poll_never_reads_whole_result_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """網頁每五秒輪詢只判舊不舊；細算結果一份十幾 MB，整份讀會讓每次輪詢多花半秒。"""
    from aosr.search.modal_record import role_inputs
    from aosr.search.outer_status import snapshot_of
    store, _, status = prepared(tmp_path)
    summary = ModalSummary(cache_dir="cache", completed=True, conclusion=status.outer.conclusion,
                           snapshot=snapshot_of(status), roles=tuple(i.record for i in role_inputs(store, status)))
    write_summary(store.path, summary)
    results = {path.resolve() for path in (store.baseline_path, *(store.path / "candidates").glob("trial-*.json"),
                                           *store.refine_dir.glob("*.json")) if not path.name.endswith("-scheme.json")}
    assert results
    opened: list[Path] = []
    original = Path.read_bytes

    def spy(path: Path) -> bytes:
        opened.append(path.resolve())
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", spy)
    block = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "modal")
    assert summary_path(store.path).resolve() in opened
    assert not results.intersection(opened)
    assert "這是上次收尾時的診斷，之後搜尋又動過" not in block.lines
