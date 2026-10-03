"""回饋事件逐行驗型別、鎖內追加與末列修復。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aosr.search import layout
from aosr.search.feedback import FeedbackEvent, FeedbackLedger


def event_document() -> dict[str, object]:
    return {"round": 2, "before_batch": 4, "anchor_trial": 3,
            "points": [dict(zip(layout.SEARCH_QUANTITIES, (0.25.hex(), 0.5.hex(), 0.75.hex()), strict=True))]}


@pytest.mark.parametrize("change", [
    {"round": True}, {"round": 1}, {"round": "2"}, {"before_batch": -1}, {"before_batch": False},
    {"anchor_trial": -1}, {"anchor_trial": "3"}, {"points": []}, {"extra": "bad"},
    {"points": [{"wrong": "0x0.0p+0"}]},
    {"points": [dict.fromkeys(layout.SEARCH_QUANTITIES, "nan")]},
    {"points": [dict.fromkeys(layout.SEARCH_QUANTITIES, "0.5")]},
    {"points": [dict.fromkeys(layout.SEARCH_QUANTITIES, 0.5)]},
    {"points": [dict.fromkeys(layout.SEARCH_QUANTITIES, 1.25.hex())]},
])
def test_event_rejects_invalid_types_even_in_last_line(tmp_path: Path, change: dict[str, object]) -> None:
    path = tmp_path / "events"
    path.write_text(json.dumps(event_document() | change) + "\n")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        FeedbackLedger.read(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", [{"round": 4}, {"before_batch": 3}])
def test_append_rejects_round_jump_or_backward_boundary_without_changes(tmp_path: Path, change: dict[str, object]) -> None:
    path = tmp_path / "events"
    first = FeedbackEvent.model_validate(event_document())
    FeedbackLedger.append(path, first)
    before = path.read_bytes()
    next_event = FeedbackEvent.model_validate(event_document() | {"round": 3, "before_batch": 5} | change)
    with pytest.raises(ValueError):
        FeedbackLedger.append(path, next_event)
    assert path.read_bytes() == before


def test_read_drops_only_incomplete_tail_and_append_repairs_it(tmp_path: Path) -> None:
    path = tmp_path / "events"
    first = FeedbackEvent.model_validate(event_document())
    second = FeedbackEvent.model_validate(event_document() | {"round": 3, "before_batch": 5})
    FeedbackLedger.append(path, first)
    complete = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"round":')
    partial = path.read_bytes()
    assert FeedbackLedger.read(path) == (first,)
    assert path.read_bytes() == partial
    FeedbackLedger.append(path, second)
    assert FeedbackLedger.read(path) == (first, second)
    assert path.read_bytes().startswith(complete)
    path.write_bytes(path.read_bytes() + b"broken\n")
    with pytest.raises(ValueError):
        FeedbackLedger.read(path)


def test_first_append_rejects_jump_without_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "events"
    with pytest.raises(ValueError):
        FeedbackLedger.append(path, FeedbackEvent.model_validate(event_document() | {"round": 3}))
    assert not path.exists()


def test_points_cannot_repeat_in_event(tmp_path: Path) -> None:
    point = dict.fromkeys(layout.SEARCH_QUANTITIES, 0.25.hex())
    with pytest.raises(ValueError):
        FeedbackEvent.model_validate(event_document() | {"points": [point, point]})
    assert tmp_path.is_dir()
