"""搜尋進度新增附件區塊，各態與讀取錯誤只影響附件。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.gui.search_view import build_search_view
from aosr.search.modal_record import ModalRole, ModalSummary, summary_path, write_summary
from tests.engine._search_modal_cases import prepared, protected


@pytest.mark.parametrize("state,label", [
    ("not_started", "未開始"), ("running", "進行中"), ("diagnosed_not_scored", "已診斷不計分"),
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
