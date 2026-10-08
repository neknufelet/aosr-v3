"""家具預篩必須在計算前落帳，行程完成順序不改結果。"""
import math
from pathlib import Path

import pytest

from aosr.search import run as search_run
from tests.engine._search_furniture_cases import (
    FurnitureCompute, enqueue_hand_placements, furnished_next_params, furnished_store,
)
from tests.engine._search_run_cases import FakeCompute, make_store, next_params, rows, run


def test_invalid_and_blocked_candidates_are_recorded_before_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_hand_placements(monkeypatch)
    store, registry = furnished_store(tmp_path)
    compute = FurnitureCompute(store)
    status = run(store, registry, compute)
    recorded = {row.trial_number: row for row in rows(store)}
    blocked, outside, valid = recorded[0], recorded[1], recorded[2]
    assert blocked.reason == "direct_path_blocked"
    # left 周圍點 (1.5,1.9,1.25)：盒 x=[1.1,1.4]、y=[1.45,1.75]，
    # x 進入 t=0.2、y 離開 t=0.7，穿盒長度 sqrt(0.5²+0.5²) × 0.5。
    assert blocked.violation_m == pytest.approx(math.sqrt(0.5) / 2.0)
    assert outside.reason == "furniture_placement_invalid"
    assert outside.violation_m == 0.25
    for row in (blocked, outside):
        assert row.outcome == "illegal" and row.score is None
        assert row.result_file is None and row.seconds == 0.0
        assert row.trial_number not in compute.calls
    assert valid.outcome == "scored" and valid.trial_number in compute.calls
    assert status.best_trial == valid.trial_number and status.streak == 0
    assert status.state == "budget_exhausted" and status.asked == store.settings.budget
    assert status.illegal_reasons.keys() == {"direct_path_blocked", "furniture_placement_invalid"}
    assert status.illegal == sum(row.outcome == "illegal" for row in rows(store))
    assert status.computed == sum(row.outcome != "illegal" for row in rows(store))


def test_furnished_single_and_multi_worker_runs_are_bit_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_hand_placements(monkeypatch, extra_legal=True)
    single, registry = furnished_store(tmp_path / "single", budget=6)
    multi, other_registry = furnished_store(tmp_path / "multi", workers=4, budget=6)
    one, four = FurnitureCompute(single), FurnitureCompute(multi)
    first = run(single, registry, one)
    second = run(multi, other_registry, four)
    assert rows(single) == rows(multi)
    assert first == second
    assert furnished_next_params(single) == furnished_next_params(multi)
    assert one.calls != four.calls


def test_plain_search_skips_prefilter_and_contact_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    whole, registry = make_store(tmp_path / "whole")
    expected = run(whole, registry, FakeCompute(whole))

    def forbidden(*args: object, **kwargs: object) -> float:
        raise AssertionError("沒有家具不得讀家具精度登記簿或呼叫預篩")

    monkeypatch.setattr(search_run, "furniture_contact_rel", forbidden)
    monkeypatch.setattr("aosr.search.furniture_prefilter.check", forbidden)
    plain, other_registry = make_store(tmp_path / "plain")
    assert run(plain, other_registry, FakeCompute(plain)) == expected
    assert rows(plain) == rows(whole)
    assert next_params(plain) == next_params(whole)


def test_contact_registry_is_read_once_per_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.engine._precision_contracts import contract_value

    enqueue_hand_placements(monkeypatch)
    # 分三批，第二次讀就丟錯，守的是跨批界只讀一次。
    store, registry = furnished_store(tmp_path, batch=1)
    pending = [contract_value("furniture_geometry_contact")]

    def read_once(path: Path) -> float:
        return pending.pop()

    monkeypatch.setattr(search_run, "furniture_contact_rel", read_once)
    status = run(store, registry, FakeCompute(store))
    assert status.state == "budget_exhausted" and not pending


def test_prefilter_uses_the_registered_contact_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.reporting.scheme import Scheme
    from aosr.search import furniture_prefilter
    from aosr.search.constraints import Violation
    from tests.engine._precision_contracts import contract_value

    enqueue_hand_placements(monkeypatch)
    store, registry = furnished_store(tmp_path)
    seen: list[float] = []
    original = furniture_prefilter.check

    def recording(scheme: Scheme, *, contact_rel: float) -> tuple[Violation, ...]:
        seen.append(contact_rel)
        return original(scheme, contact_rel=contact_rel)

    monkeypatch.setattr(furniture_prefilter, "check", recording)
    run(store, registry, FurnitureCompute(store))
    # 搜尋用的界線就是登記簿那一份（不是乘過或另抄的值）。
    assert seen and set(seen) == {contract_value("furniture_geometry_contact")}
