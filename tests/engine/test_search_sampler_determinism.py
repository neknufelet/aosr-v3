"""批次取樣、回報與帳本重播的逐位決定性考卷。"""

import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import optuna
import pytest
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from aosr.search.sampler import (
    Excluded,
    Illegal,
    Outcome,
    Proposal,
    RankingZone,
    ReplayMismatch,
    SamplerAdapter,
    SamplerSettings,
    Scored,
)

SPACE = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
PROCESS_CODE = """
from aosr.search.sampler import SamplerAdapter, SamplerSettings, Illegal, Scored
adapter = SamplerAdapter({'x': (0.0, 1.0), 'y': (0.0, 1.0)}, SamplerSettings(0, 10))
for _ in range(10):
    outcomes = {}
    for proposal in adapter.ask_batch(4):
        x, y = proposal.params['x'], proposal.params['y']
        print(proposal.trial_number, x.hex(), y.hex())
        outcomes[proposal.trial_number] = (Illegal('wall', x + y - 1.4)
            if x + y > 1.4 else Scored((x - 0.9)**2 + (y - 0.9)**2))
    adapter.tell_batch(outcomes)
"""


@pytest.fixture(autouse=True)
def isolated_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def fingerprint(proposals: Sequence[Proposal]) -> tuple[tuple[int, tuple[tuple[str, str], ...]], ...]:
    return tuple((proposal.trial_number, tuple((key, value.hex()) for key, value in sorted(proposal.params.items())))
                 for proposal in proposals)


def edge_outcomes(proposals: Sequence[Proposal]) -> dict[int, Outcome]:
    outcomes: dict[int, Outcome] = {}
    for proposal in proposals:
        x, y = proposal.params["x"], proposal.params["y"]
        outcomes[proposal.trial_number] = (Illegal("wall", x + y - 1.4) if x + y > 1.4
                                          else Scored((x - 0.9) ** 2 + (y - 0.9) ** 2))
    return outcomes


def mixed_history(adapter: SamplerAdapter) -> list[tuple[Sequence[Proposal], Mapping[int, Outcome]]]:
    history: list[tuple[Sequence[Proposal], Mapping[int, Outcome]]] = []
    for _ in range(3):
        proposals = adapter.ask_batch(3)
        outcomes: dict[int, Outcome] = {
            proposals[0].trial_number: Scored(0.5),
            proposals[1].trial_number: Illegal("wall", 0.125),
            proposals[2].trial_number: Excluded(RankingZone.UNASSESSED),
        }
        adapter.tell_batch(outcomes)
        history.append((proposals, outcomes))
    return history


def test_same_seed_same_k_bit_identical_across_processes(tmp_path: Path) -> None:
    outputs = []
    for hash_seed in ("0", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", PROCESS_CODE], cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONDONTWRITEBYTECODE": "1"},
            check=True, capture_output=True, text=True, timeout=15,
        )
        assert completed.stdout.strip()
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]


def test_tell_order_inside_a_batch_does_not_matter() -> None:
    first = SamplerAdapter(SPACE, SamplerSettings(0, 10))
    second = SamplerAdapter(SPACE, SamplerSettings(0, 10))
    for _ in range(10):
        proposals = first.ask_batch(4)
        assert fingerprint(proposals) == fingerprint(second.ask_batch(4))
        outcomes = edge_outcomes(proposals)
        first.tell_batch(outcomes)
        second.tell_batch(dict(reversed(tuple(outcomes.items()))))
        # TPE（模型取樣）可能不受插入順序影響；完成時間另驗編號順序契約。
        completed = second.trial_snapshots
        assert all(trial.datetime_complete is not None for trial in completed)
        assert [trial.number for trial in sorted(completed, key=lambda trial: str(trial.datetime_complete))] == [
            trial.number for trial in completed
        ]
    assert fingerprint(first.ask_batch(4)) == fingerprint(second.ask_batch(4))


def test_replay_reproduces_the_next_proposal() -> None:
    first = SamplerAdapter(SPACE, SamplerSettings(0, 2))
    history = mixed_history(first)
    second = SamplerAdapter(SPACE, SamplerSettings(0, 2))
    second.replay(history)
    assert second.trials_told == first.trials_told
    assert fingerprint(first.ask_batch(3)) == fingerprint(second.ask_batch(3))


def test_replay_detects_a_changed_record() -> None:
    first = SamplerAdapter(SPACE, SamplerSettings(0, 2))
    history = mixed_history(first)
    proposals, outcomes = history[0]
    proposal = proposals[0]
    changed = Proposal(proposal.trial_number, {**proposal.params, "x": math.nextafter(proposal.params["x"], math.inf)})
    history[0] = ((changed, *proposals[1:]), outcomes)
    second = SamplerAdapter(SPACE, SamplerSettings(0, 2))
    with pytest.raises(ReplayMismatch, match=r"trial 0.*x"):
        second.replay(history)


def test_replay_checks_trial_numbers_parameter_names_and_freshness() -> None:
    first = SamplerAdapter(SPACE, SamplerSettings(0, 2))
    history = mixed_history(first)
    with pytest.raises(RuntimeError):
        first.replay(history)
    proposals, outcomes = history[0]
    proposal = proposals[0]
    for changed in (Proposal(999, proposal.params), Proposal(proposal.trial_number, {"x": proposal.params["x"]})):
        second = SamplerAdapter(SPACE, SamplerSettings(0, 2))
        with pytest.raises(ReplayMismatch):
            second.replay([((changed, *proposals[1:]), outcomes)])
    outstanding = SamplerAdapter(SPACE, SamplerSettings(0, 2))
    outstanding.ask_batch(1)
    with pytest.raises(RuntimeError):
        outstanding.replay(history)


def test_k_greater_than_one_never_repeats_a_proposal_within_a_batch() -> None:
    adapter = SamplerAdapter(SPACE, SamplerSettings(0, 10))
    reference = optuna.create_study(direction="minimize", sampler=TPESampler(
        seed=0, n_startup_trials=10, constant_liar=True,
    ))
    distributions: dict[str, BaseDistribution] = {name: FloatDistribution(*bounds) for name, bounds in SPACE.items()}
    for _ in range(10):
        proposals = adapter.ask_batch(4)
        encoded = [tuple(value.hex() for value in proposal.params.values()) for proposal in proposals]
        assert all(left != right for index, left in enumerate(encoded) for right in encoded[index + 1:])
        trials = [reference.ask(distributions) for _ in proposals]
        expected = tuple(Proposal(trial.number, {key: float(value) for key, value in trial.params.items()}) for trial in trials)
        assert fingerprint(proposals) == fingerprint(expected)
        outcomes = edge_outcomes(proposals)
        adapter.tell_batch(outcomes)
        for trial in trials:
            outcome = outcomes[trial.number]
            trial.set_constraint("geometry", outcome.violation if isinstance(outcome, Illegal) else 0.0)
            trial.set_constraint("ranking", 0.0)
            if isinstance(outcome, Scored):
                reference.tell(trial, outcome.value)
            else:
                reference.tell(trial, state=TrialState.PRUNED)
