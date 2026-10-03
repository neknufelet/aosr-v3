"""批界排入與重播；旧 enqueue 的考卷不改。"""

from pathlib import Path

import pytest

from aosr.search.sampler import SamplerAdapter, SamplerSettings, Scored


def test_between_batches_rejects_outstanding_and_preserves_enqueue(tmp_path: Path) -> None:
    adapter = SamplerAdapter({"x": (0.0, 1.0)}, SamplerSettings(3, 2))
    adapter.enqueue({"x": 0.125})
    first = adapter.ask_batch(2)
    with pytest.raises(RuntimeError, match="outstanding"):
        adapter.enqueue_between_batches(({"x": 0.25},))
    adapter.tell_batch({item.trial_number: Scored(1.0) for item in first})
    with pytest.raises(RuntimeError, match="never asked"):
        adapter.enqueue({"x": 0.25})
    adapter.enqueue_between_batches(({"x": 0.25}, {"x": 0.5}, {"x": 0.75}))
    assert adapter.trials_told == len(first)
    next_batch = adapter.ask_batch(2)
    assert [(item.trial_number, dict(item.params)) for item in next_batch] == [
        (len(first), {"x": 0.25}), (len(first) + 1, {"x": 0.5})]
    adapter.tell_batch({item.trial_number: Scored(1.0) for item in next_batch})
    last = adapter.ask_batch(1)
    assert [(item.trial_number, dict(item.params)) for item in last] == [(len(first) + len(next_batch), {"x": 0.75})]
    assert tmp_path.is_dir()


@pytest.mark.parametrize("bad", [{"wrong": 0.1}, {"x": -0.1}, {"x": 1.1}, {"x": float("nan")}, {"x": float("inf")}])
def test_between_batches_validates_all_points_before_enqueue(tmp_path: Path, bad: dict[str, float]) -> None:
    adapter = SamplerAdapter({"x": (0.0, 1.0)}, SamplerSettings(3, 2))
    before = adapter.trial_snapshots
    with pytest.raises(ValueError):
        adapter.enqueue_between_batches(({"x": 0.25}, bad))
    assert adapter.trial_snapshots == before
    assert tmp_path.is_dir()


def test_sampler_replay_inserts_at_the_recorded_batch(tmp_path: Path) -> None:
    settings = SamplerSettings(3, 2)
    adapter = SamplerAdapter({"x": (0.0, 1.0)}, settings)
    history = []
    points = ({"x": 0.25}, {"x": 0.5}, {"x": 0.75})
    for index in range(4):
        if index == 1:
            adapter.enqueue_between_batches(points)
        proposals = adapter.ask_batch(2)
        outcomes = {item.trial_number: Scored(item.params["x"]) for item in proposals}
        adapter.tell_batch(outcomes)
        history.append((proposals, outcomes))
    fresh = SamplerAdapter({"x": (0.0, 1.0)}, settings)
    fresh.replay(history, enqueues={1: points})
    assert [(trial.number, trial.params, trial.state, trial.values) for trial in fresh.trial_snapshots] == [
        (trial.number, trial.params, trial.state, trial.values) for trial in adapter.trial_snapshots]
    assert fresh.ask_batch(2) == adapter.ask_batch(2)
    assert tmp_path.is_dir()
