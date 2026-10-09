"""摘要契約與原子換代；不在考卷裡碰真搜尋或真物理。"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.search.placement_stability_record import read_summary, summary_path, write_summary
from tests.engine._stability_attach_cases import ShiftCompute, attach, ready


def test_summary_roundtrip_frozen_forbid_and_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = ready(tmp_path)
    summary = attach(store, registry, status, ShiftCompute(store))
    assert read_summary(store.path) == summary
    with pytest.raises(ValidationError):
        type(summary).model_validate(summary.model_dump() | {"unknown": True})
    with pytest.raises(ValidationError):
        summary.state = "failed"
    document = summary.model_dump()
    document["selection"]["unexpected"] = True
    with pytest.raises(ValidationError):
        type(summary).model_validate(document)
    old = summary_path(store.path).read_bytes()
    seen: list[bytes] = []
    original = Path.replace
    def replacing(source: Path, target: Path) -> Path:
        if target == summary_path(store.path):
            seen.append(target.read_bytes())
            assert source.parent == target.parent and source != target
            assert read_summary(store.path) == summary
            assert type(summary).model_validate_json(source.read_bytes()).reason_text == "下一代"
        return original(source, target)
    monkeypatch.setattr(Path, "replace", replacing)
    changed = summary.model_copy(update={"reason_text": "下一代"})
    write_summary(store.path, changed)
    assert seen == [old] and read_summary(store.path) == changed


def test_atomic_failure_preserves_old_and_removes_own_temporary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = ready(tmp_path)
    summary = attach(store, registry, status, ShiftCompute(store))
    before = {p.name: p.read_bytes() for p in summary_path(store.path).parent.iterdir() if p.is_file()}
    def fail(source: Path, target: Path) -> Path:
        raise OSError("換名失敗原文")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="換名失敗原文"):
        write_summary(store.path, summary.model_copy(update={"reason_text": "下一代"}))
    assert {p.name: p.read_bytes() for p in summary_path(store.path).parent.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("field", ["arithmetic", "boundaries"])
@pytest.mark.parametrize("state", ["running", "skipped", "failed", "stopped"])
def test_only_done_can_carry_report_arithmetic_or_boundaries(tmp_path: Path, state: str, field: str) -> None:
    store, registry, status = ready(tmp_path)
    done = attach(store, registry, status, ShiftCompute(store))
    document = done.model_dump() | {"state": state, "completed": state == "skipped",
                                    "arithmetic": None, "boundaries": ()}
    document[field] = done.model_dump()[field]
    with pytest.raises(ValidationError, match="只有完成"):
        type(done).model_validate(document)


def test_write_validates_before_touching_summary(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    done = attach(store, registry, status, ShiftCompute(store))
    before = summary_path(store.path).read_bytes()
    invalid = done.model_copy(update={"computed_points": done.total_points + 1})
    with pytest.raises(ValidationError):
        write_summary(store.path, invalid)
    assert summary_path(store.path).read_bytes() == before
    assert not list(summary_path(store.path).parent.glob("summary-write-*.tmp"))
