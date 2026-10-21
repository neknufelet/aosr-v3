"""第一階段共用投影與命令列報告；壞摘要也不能冒充需要第二階段檢查。"""
from pathlib import Path

import pytest

from aosr.gui.search_view import build_search_view
from aosr.search import cli
from aosr.search.report_crossover import crossover_report
from aosr.search.report_stability import stability_report
from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
from tests.engine._stage_one_auto_cases import invoke, read
from tests.engine._stage_one_post_check_cases import prepared, runner
from tests.engine._stage_one_refine_cases import pool, snapshot
from tests.engine.test_search_stage_one_refine import ROWS

CROSSOVER = "範圍外：第一階段只算有限元素（20～300 Hz），沒有 300 Hz 交接，這一項不適用"
STABILITY = "範圍外：擺位穩定性是第二階段的檢查；第一階段的座位穩健性已在合分裡（主位周圍六點各 10 公分）"


@pytest.mark.parametrize("payload", [None, "broken", "stage_two"])
def test_stage_one_scope_ignores_all_attachment_summaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       payload: str | None) -> None:
    from aosr.search import report_crossover, report_stability
    store = pool(tmp_path, monkeypatch, ROWS)
    if payload == "stage_two":
        from aosr.search.crossover_record import fresh_summary as crossover_summary
        from aosr.search.placement_stability_attach import fresh_summary as stability_summary
        from tests.engine._crossover_cases import prepared as second_stage
        other, _, status = second_stage(tmp_path / "second-stage")
        for name, summary in (("crossover-sensitivity", crossover_summary(other, status)),
                              ("placement-stability", stability_summary(other, status))):
            folder = store.path / name; folder.mkdir(); (folder / "summary.json").write_text(summary.model_dump_json())
    elif payload is not None:
        for name in ("crossover-sensitivity", "placement-stability"):
            folder = store.path / name; folder.mkdir(); (folder / "summary.json").write_text(payload)
    before = snapshot(store)
    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("第一階段投影不可讀摘要或判舊")
    for module in (report_crossover, report_stability):
        for symbol in ("read_summary", "fresh_summary", "is_stale"):
            monkeypatch.setattr(module, symbol, forbidden)
    for report, expected in ((crossover_report(store, read(store)), CROSSOVER),
                             (stability_report(store, read(store)), STABILITY)):
        assert report.lines == (expected,)
        assert not report.warning and not report.has_results
        assert all(word not in "\n".join(report.lines) for word in (
            "（暫時）", "尚未驗證", "讀不到", "舊了", "未開始", "失敗"))
    assert snapshot(store) == before


def test_completed_auto_existing_page_and_cli_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    store = prepared(tmp_path, monkeypatch)
    code = invoke(store, tmp_path, monkeypatch, runner=runner(tmp_path / "child", store))
    assert not code
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
                             server_program=store.identity.program_fingerprint)
    blocks = {block.key: block for block in view.blocks}
    assert blocks["crossover"].lines == (CROSSOVER,) and not blocks["crossover"].warning
    assert blocks["stability"].lines == (STABILITY,) and not blocks["stability"].warning
    assert "外圈結論：細算完成" in blocks["stage"].lines
    text = render_stage_one_text(build_stage_one_report(store))
    assert "外圈結論：細算完成\n" in text
    assert text.endswith(f"300 Hz 交接敏感度（不改名次）\n{CROSSOVER}\n擺位穩定性（±2 公分，不改名次）\n{STABILITY}\n")
    capsys.readouterr()
    code = cli.main(["report", "--search", str(store.path)])
    assert not code and capsys.readouterr().out == text


def test_manual_search_report_has_unjudged_outer_conclusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    text = render_stage_one_text(build_stage_one_report(store))
    assert "外圈結論：未判定\n" in text
