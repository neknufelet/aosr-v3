"""網頁輪詢只讀摘要與帳，逐字共用報告投影，舊搜尋不標紅。"""
from pathlib import Path
from typing import BinaryIO, TextIO, cast

import pytest

from aosr.gui.search_view import build_search_view
from aosr.search.placement_stability_record import STALE, summary_path
from aosr.search.report_stability import ABSENT, stability_report
from aosr.search.store import PROJECT_FILE, SETTINGS_FILE, IDENTITY_FILE, PURPOSE_FILE
from tests.engine._stability_report_cases import report_case


def test_stability_block_after_crossover_matches_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, _, _ = report_case(tmp_path, monkeypatch)
    view = build_search_view(store.path, server_physics="test", server_program="test")
    keys = [block.key for block in view.blocks]
    assert keys[keys.index("crossover") + 1] == "stability"
    block = next(b for b in view.blocks if b.key == "stability")
    report = stability_report(store, status)
    assert block.lines == report.lines and block.warning == report.warning


def test_stability_poll_only_reads_summaries_and_ledgers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, _, _ = report_case(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    original_bytes, original_text, original_open = Path.read_bytes, Path.read_text, Path.open
    allowed = {*(store.path / name for name in (PROJECT_FILE, SETTINGS_FILE, IDENTITY_FILE, PURPOSE_FILE)), store.status_path,
               store.ledger_path, store.refine_ledger_path, summary_path(store.path),
               store.path / "crossover-sensitivity" / "summary.json", store.path / "modal-diagnosis" / "summary.json"}
    def guard(path: Path) -> None:
        if path.is_relative_to(store.path):
            assert path in allowed, f"網頁不准讀整份結果：{path}"
    def read_bytes(path: Path) -> bytes:
        guard(path)
        return original_bytes(path)
    def read_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        guard(path)
        return original_text(path, encoding=encoding, errors=errors)
    def opened(path: Path, mode: str = "r", buffering: int = -1, encoding: str | None = None,
               errors: str | None = None, newline: str | None = None) -> BinaryIO | TextIO:
        guard(path)
        return cast(BinaryIO | TextIO, original_open(path, mode, buffering, encoding, errors, newline))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_bytes)
        patch.setattr(Path, "read_text", read_text)
        patch.setattr(Path, "open", opened)
        block = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "stability")
        assert block.lines == stability_report(store, status).lines and not block.warning
        store.status_path.write_text(status.model_copy(update={"asked": status.asked + 1}).model_dump_json())
        updated = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "stability")
        assert STALE in updated.lines and not updated.warning
    assert {str(p): p.read_bytes() for p in store.path.rglob("*") if p.is_file() and p != store.status_path} == {
        key: value for key, value in before.items() if key != str(store.status_path)}


def test_bad_and_absent_summary_only_affect_stability_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, _, _, _ = report_case(tmp_path, monkeypatch)
    before = build_search_view(store.path, server_physics="test", server_program="test")
    summary_path(store.path).write_text("{")
    bad = build_search_view(store.path, server_physics="test", server_program="test")
    blocks = {block.key: block for block in bad.blocks}
    assert blocks["stability"].warning and "讀不到" in "\n".join(blocks["stability"].lines)
    assert [b for b in before.blocks if b.key not in ("stability", "updated")] == [
        b for b in bad.blocks if b.key not in ("stability", "updated")]
    summary_path(store.path).unlink()
    absent = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "stability")
    assert absent.lines == (ABSENT,) and not absent.warning
