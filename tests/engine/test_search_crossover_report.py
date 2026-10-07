"""報告只讀交接摘要；三句、暫時、未完成與舊摘要的文字來自寫入端。"""
from pathlib import Path

import pytest

from aosr.search import crossover_sensitivity as module
from aosr.search.crossover_record import INCOMPLETE, INTRO, STALE, TITLE, VERDICTS, summary_lines, summary_path, write_summary
from aosr.search.report import build_report, render_text
from aosr.search.report_crossover import crossover_report, crossover_text
from aosr.reporting.display import SPEAKERS
from tests.engine._crossover_cases import evaluate, prepared
from tests.engine._search_run_cases import RUN_DATE


@pytest.mark.parametrize("f_s,swap", [(200, True), (200, False), (340, False)])
def test_report_uses_writer_verdict_and_distance_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      f_s: float, swap: bool) -> None:
    store, _, status = prepared(tmp_path, f_s=f_s, swap=swap)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=tmp_path / "registry")
    report = crossover_report(store, status)
    rendered = crossover_text(report)
    assert report.lines == summary_lines(summary)
    assert rendered.startswith(TITLE + "\n" + INTRO + "\n")
    assert VERDICTS[summary.verdict] in rendered
    assert "\n\n" not in rendered
    for variant in summary.variants:
        assert variant.basis in rendered
        for key, distance in variant.speaker_distance_cm.items():
            assert f"{SPEAKERS[key]}相距 {distance:.1f} 公分" in rendered


@pytest.mark.parametrize("change,note", [("temporary", "（暫時）"), ("incomplete", INCOMPLETE),
    ("snapshot", STALE), ("conclusion", STALE), ("rows", STALE)])
def test_report_marks_temporary_incomplete_and_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     change: str, note: str) -> None:
    from aosr.search.outer_status import OuterStatus
    from aosr.search.refine import RefineLedger
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    if change == "temporary":
        status = status.model_copy(update={"outer": OuterStatus(conclusion="refine_budget")})
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    if change == "incomplete":
        write_summary(store.path, summary.model_copy(update={"completed": False}))
    elif change == "snapshot":
        status = status.model_copy(update={"asked": status.asked + 1})
    elif change == "conclusion":
        status = status.model_copy(update={"outer": OuterStatus(conclusion="refine_budget")})
    elif change == "rows":
        header, rows = RefineLedger.read(store.refine_ledger_path)
        store.refine_ledger_path.unlink()
        book = RefineLedger.create(store.refine_ledger_path, header)
        for row in rows:
            book.append(row.model_copy(update={"total_cost": 20.0}) if row.trial_number == 9 else row)
    assert note in crossover_text(crossover_report(store, status))


def test_build_report_does_not_write_or_reevaluate_and_bad_summary_is_local(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    module.attach_crossover(store, status=status, quality_targets_path=registry)
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("build_report 不准重評或寫檔")
    before = {str(p): p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    monkeypatch.setattr(module, "reevaluate", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert {str(p): p.read_bytes() for p in store.path.rglob("*") if p.is_file()} == before
    with summary_path(store.path).open("w") as stream:
        stream.write("{")
    bad = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    original = render_text(report).split("\n\n")
    changed = render_text(bad).split("\n\n")
    assert [s for s in original if not s.startswith(TITLE)] == [s for s in changed if not s.startswith(TITLE)]
    assert bad.crossover.warning and "讀不到" in crossover_text(bad.crossover)
