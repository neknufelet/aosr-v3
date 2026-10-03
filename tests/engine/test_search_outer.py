"""外圈判準的答案由設計表手推；只使用合成計算。"""

from pathlib import Path

import pytest

from aosr.search.run import RefineStatus, SearchStatus, _write_status
from tests.engine._search_feedback_cases import prepared
from tests.engine._search_run_cases import RUN_DATE


def test_old_status_has_independent_default_outer(tmp_path: Path) -> None:
    path = tmp_path / "old-status"
    path.write_text('{"state":"converged"}')
    status = SearchStatus.model_validate_json(path.read_bytes())
    assert hasattr(status, "outer")
    assert status.outer.conclusion is None
    assert status.outer.message == "未判定"
    with pytest.raises(ValueError):
        type(status.outer).model_validate({"unexpected": True})


@pytest.mark.parametrize("search,reason,best,reference,asked,configured,marker,want", [
    ("converged", "stable", 2, 2, 10, True, False, "complete"),
    ("budget_exhausted", "stable", 2, 2, 40, True, False, "stable_but_search_budget"),
    ("converged", "refine_budget", 2, 1, 10, True, False, "refine_budget"),
    ("converged", "candidates_exhausted", 2, 1, 10, True, False, "refine_candidates_exhausted"),
    ("budget_exhausted", "stable", 2, 1, 10, True, False, "feedback_blocked_by_budget"),
    ("converged", "stable", 2, 1, 40, True, False, "feedback_blocked_by_budget"),
    ("converged", "stable", 2, 1, 10, False, False, "feedback_not_configured"),
    ("converged", "stable", "baseline", 1, 10, True, False, "baseline_first"),
    ("converged", "stable", None, 1, 10, True, False, "no_rankable_first"),
    ("converged", "user_stopped", 2, 1, 10, True, False, "user_stopped"),
    ("failed", "refine_budget", None, 1, 10, False, True, "user_stopped"),
    ("failed", "refine_budget", None, 1, 10, False, False, "search_failed"),
    ("interrupted", "stable", 2, 2, 10, True, False, "search_interrupted"),
    ("converged", "refine_budget", None, 1, 40, False, False, "refine_budget"),
    ("converged", "candidates_exhausted", "baseline", 1, 40, False, False, "refine_candidates_exhausted"),
    ("converged", "stable", 2, 1, 40, False, False, "feedback_blocked_by_budget"),
    ("converged", "stable", 2, 1, 10, True, False, "feedback"),
])
def test_decision_table(tmp_path: Path, search: str, reason: str, best: int | str | None,
                        reference: int, asked: int, configured: bool, marker: bool, want: str) -> None:
    from aosr.search.outer import decide_outer

    status = SearchStatus.model_validate({"state": search, "asked": asked,
        "refine": {"state": "stopped", "stop_reason": reason, "best": best}})
    assert decide_outer(status, reference=reference, budget=40, feedback=configured, stopped=marker) == want
    assert tmp_path.is_dir()


@pytest.mark.parametrize("state,want", [("failed", "refine_failed"), ("interrupted", "refine_interrupted")])
def test_refine_errors_precede_first(tmp_path: Path, state: str, want: str) -> None:
    from aosr.search.outer import decide_outer

    status = SearchStatus(refine=RefineStatus.model_validate({"state": state, "best": "baseline"}), state="converged")
    assert decide_outer(status, reference=1, budget=40, feedback=True) == want
    assert tmp_path.is_dir()


def test_report_conclusion_is_independent_and_expires(tmp_path: Path) -> None:
    from aosr.search.outer import conclude
    from aosr.search.report import build_report, render_text

    store, registry, status = prepared(tmp_path)
    before = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    conclude(store, status, "refine_candidates_exhausted")
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.model_dump(exclude={"outer"}) == status.model_dump(exclude={"outer"})
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    old_sections, new_sections = before.split("\n\n"), text.split("\n\n")
    assert old_sections[0] == new_sections[0]
    assert old_sections[2:] == new_sections[2:]
    assert new_sections[1].splitlines()[:-1] == old_sections[1].splitlines()[:-1]
    assert new_sections[1].endswith("外圈結論：細算候選用完，未完成")
    _write_status(store, after.model_copy(update={"asked": after.asked + 1}))
    expired = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert "外圈結論：未判定（判定之後狀態又變了）" in expired


@pytest.mark.parametrize("field", ["search_state", "search_round", "asked", "refine_state", "refine_round", "refined"])
def test_every_snapshot_field_expires_report(tmp_path: Path, field: str) -> None:
    from aosr.search.outer import conclude
    from aosr.search.report import build_report, render_text

    store, registry, status = prepared(tmp_path)
    current = conclude(store, status, "complete")
    if field == "search_state":
        changed = current.model_copy(update={"state": "budget_exhausted"})
    elif field == "search_round":
        changed = current.model_copy(update={"round": current.round + 1})
    elif field == "asked":
        changed = current.model_copy(update={"asked": current.asked + 1})
    else:
        updates: dict[str, object] = {"state": "running", "stop_reason": None} if field == "refine_state" else {
            "round" if field == "refine_round" else "refined":
                current.refine.round + 1 if field == "refine_round" else current.refine.refined + 1}
        changed = current.model_copy(update={"refine": current.refine.model_copy(update=updates)})
    _write_status(store, changed)
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert "外圈結論：未判定（判定之後狀態又變了）" in text
    assert "外圈結論：細算完成" not in text
