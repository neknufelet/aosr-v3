"""網頁交接區塊共用報告投影；不讀完整結果，正常提醒不標警示。"""
from pathlib import Path

import pytest

from aosr.gui.search_view import build_search_view
from aosr.search import crossover_sensitivity as module
from aosr.search.crossover_record import INCOMPLETE, STALE, VERDICTS, summary_path, write_summary
from tests.engine._crossover_cases import evaluate, prepared


@pytest.mark.parametrize("f_s,swap", [(200, True), (200, False), (340, False)])
def test_crossover_block_after_refine_with_writer_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      f_s: float, swap: bool) -> None:
    store, registry, status = prepared(tmp_path, f_s=f_s, swap=swap)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    view = build_search_view(store.path, server_physics="test", server_program="test")
    keys = [b.key for b in view.blocks]
    assert keys[keys.index("refine-best") + 1] == "crossover"
    block = next(b for b in view.blocks if b.key == "crossover")
    assert not block.warning and VERDICTS[summary.verdict] in block.lines
    assert any("第一名" in line for line in block.lines)


def test_crossover_poll_only_reads_summary_and_stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    write_summary(store.path, summary.model_copy(update={"completed": False}))
    original = Path.read_bytes
    def read(path: Path) -> bytes:
        assert path.parent != store.refine_dir
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    block = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "crossover")
    assert INCOMPLETE in block.lines and not block.warning
    store.status_path.write_text(status.model_copy(update={"asked": status.asked + 1}).model_dump_json())
    block = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "crossover")
    assert STALE in block.lines and not block.warning


def test_bad_crossover_summary_only_warns_its_block(tmp_path: Path) -> None:
    store, _, _ = prepared(tmp_path)
    path = summary_path(store.path)
    path.parent.mkdir()
    path.write_text("{")
    blocks = {b.key: b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks}
    assert blocks["crossover"].warning and "讀不到" in "\n".join(blocks["crossover"].lines)
    assert not blocks["refine-best"].warning
