"""新原因可由帳本驗證，且搜尋報告與網頁都能顯示中文。"""
from pathlib import Path

import pytest

from aosr.search import constraints, labels, ledger
from aosr.search.run import SearchStatus
from tests.engine._search_run_cases import make_store


REASONS = {
    "direct_path_blocked": "不符合擺位要求：直達路徑被家具擋住",
    "furniture_placement_invalid": "家具擺放不合法：超出房間、間隙不足，或喇叭、座位在家具裡",
    "cabinet_in_furniture": "箱體穿入家具", "cabinet_off_table": "箱體超出桌面",
    "stand_space_occupied": "腳架下方有家具", "furniture_in_keep_out": "家具進入禁區",
}


def test_every_constraint_reason_has_chinese() -> None:
    for reason in constraints.Reason:
        assert reason.value in labels.COUNT_REASONS
        assert any("\u4e00" <= char <= "\u9fff" for char in labels.COUNT_REASONS[reason.value])
    for code, text in REASONS.items():
        assert code in {reason.value for reason in constraints.Reason}
        assert labels.COUNT_REASONS[code] == text


@pytest.mark.parametrize("code", REASONS)
def test_ledger_and_search_page_render_furniture_reason(tmp_path: Path, code: str) -> None:
    from aosr.gui.search_view import build_search_view

    store, _ = make_store(tmp_path)
    params = {"front_distance": 1.0, "spacing": 1.2, "listening_distance": 0.5}
    row = ledger.LedgerRow(batch_index=0, trial_number=0,
                           unit_params_hex={key: (0.5).hex() for key in params}, params_m=params,
                           outcome="illegal", score=None, reason=code, violation_m=0.25,
                           seconds=0.0, result_file=None)
    ledger.create_for(store).append(row)
    store.status_path.write_text(SearchStatus(state="budget_exhausted", asked=1).model_dump_json())
    view = build_search_view(store.path, server_physics="現在", server_program="現在")
    block = next(item for item in view.blocks if item.key == "counts")
    text = "\n".join(block.lines)
    assert REASONS[code] in text and "未辨識原因" not in text


@pytest.mark.parametrize("code", REASONS)
def test_search_report_renders_furniture_reason(tmp_path: Path, code: str) -> None:
    from aosr.search.report import build_report, render_text
    from tests.engine.test_search_report import _finished
    from tests.engine._search_run_cases import RUN_DATE

    store, registry = _finished(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes()).model_copy(update={
        "illegal_reasons": {code: 1},
    })
    store.status_path.write_text(status.model_dump_json())
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert REASONS[code] in text and "未辨識原因" not in text
