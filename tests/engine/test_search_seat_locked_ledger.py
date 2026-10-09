"""兩組單位鍵都合法，但每列仍須核對該帳本表頭；四點清單手算。"""
import json
from pathlib import Path

import pytest

from aosr.search import feedback, layout, ledger
from aosr.search.sampler import Proposal, Scored
from aosr.search.store import candidate_name
from tests.engine._seat_locked_cases import locked_store
from tests.engine._search_run_cases import make_store


def row(*, locked: bool = True) -> ledger.LedgerRow:
    unit = {"front_distance": 0.125, "spacing": 0.5}
    if not locked:
        unit["listening_distance"] = 0.875
    return ledger.row_from_outcome(batch_index=0, proposal=Proposal(0, unit),
        params_m={"front_distance": 0.46875, "spacing": 1.25, "listening_distance": 2.73125},
        outcome=Scored(1.0), seconds=0.0, result_file=candidate_name(0))


def test_locked_header_and_meter_row_keep_derived_distance(tmp_path: Path) -> None:
    store, _ = locked_store(tmp_path)
    header = ledger.header_for(store)
    assert header.search_space == {"front_distance": (0.0, 1.0), "spacing": (0.0, 1.0)}
    book = ledger.create_for(store)
    book.append(row())
    status = ledger.read_for(store)
    assert status.rows == (row(),)
    assert status.rows[0].params_m == {"front_distance": 0.46875, "spacing": 1.25, "listening_distance": 2.73125}
    history, partial = ledger.replay_history(header, status.rows, batch_sizes={0: 1})
    assert not partial
    assert history[0][0][0].params == {"front_distance": 0.125, "spacing": 0.5}


@pytest.mark.parametrize("locked", [True, False])
@pytest.mark.parametrize("operation", ["append", "read", "replay"])
def test_ledger_unit_keys_must_match_header(tmp_path: Path, locked: bool, operation: str) -> None:
    store, _ = locked_store(tmp_path) if locked else make_store(tmp_path)
    header = ledger.header_for(store)
    wrong = row(locked=not locked)
    # 列自己可接受兩組之一；跨欄位核對必須在帳本操作層。
    assert ledger.LedgerRow.model_validate(wrong.model_dump()) == wrong
    book = ledger.create_for(store)
    before = store.ledger_path.read_bytes()
    with pytest.raises(ValueError, match="unit parameter keys do not match ledger search_space"):
        if operation == "append":
            book.append(wrong)
        elif operation == "read":
            with store.ledger_path.open("a") as stream:
                stream.write(json.dumps(wrong.model_dump()) + "\n")
            ledger.Ledger.read(store.ledger_path)
        else:
            ledger.replay_history(header, (wrong,), batch_sizes={0: 1})
    if operation == "append":
        assert store.ledger_path.read_bytes() == before


def test_locked_feedback_has_complete_ordered_four_point_list(tmp_path: Path) -> None:
    center = row()
    expected = ({"front_distance": 0.0, "spacing": 0.5}, {"front_distance": 0.375, "spacing": 0.5},
                {"front_distance": 0.125, "spacing": 0.25}, {"front_distance": 0.125, "spacing": 0.75})
    points = feedback.feedback_points(center, (center,), 0.25)
    assert points == expected
    event = feedback.FeedbackEvent(round=2, before_batch=1, anchor_trial=0,
        points=tuple({key: value.hex() for key, value in point.items()} for point in expected))
    feedback.FeedbackLedger.append(tmp_path / "events", event)
    assert feedback.FeedbackLedger.read(tmp_path / "events") == (event,)
    assert event.unit_points() == expected
    assert layout.SEARCH_QUANTITIES == ("front_distance", "spacing", "listening_distance")
