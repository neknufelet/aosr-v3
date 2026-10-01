"""真取樣器考卷：限制回饋要少提禁區，無分數結果不能捏造數字。"""

import inspect
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import optuna
import pytest
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.samplers import RandomSampler, TPESampler
from optuna.trial import TrialState

from aosr.search import sampler as sampler_module
from aosr.search.sampler import (
    Excluded,
    Illegal,
    Outcome,
    Proposal,
    RankingZone,
    SamplerAdapter,
    SamplerSettings,
    Scored,
)

SPACE = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
STARTUP = 10


@pytest.fixture(autouse=True)
def isolated_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def edge_violation(x: float, y: float) -> float:
    return x + y - 1.4


def far_violation(x: float, y: float) -> float:
    return 0.25 - math.hypot(x - 0.3, y - 0.3)


def objective(x: float, y: float) -> float:
    return (x - 0.9) ** 2 + (y - 0.9) ** 2


def adapter_illegal_count(seed: int, violation: Callable[[float, float], float]) -> int:
    adapter = SamplerAdapter(SPACE, SamplerSettings(seed, STARTUP))
    illegal_after_startup = 0
    for index in range(80):
        proposal, = adapter.ask_batch(1)
        x, y = proposal.params["x"], proposal.params["y"]
        amount = violation(x, y)
        outcome: Outcome = Illegal("geometry", amount) if amount > 0 else Scored(objective(x, y))
        illegal_after_startup += int(index >= STARTUP and amount > 0)
        adapter.tell_batch({proposal.trial_number: outcome})
    return illegal_after_startup


def control_illegal_count(
    seed: int, violation: Callable[[float, float], float], *, random: bool,
) -> int:
    sampler = RandomSampler(seed=seed) if random else TPESampler(
        seed=seed, n_startup_trials=STARTUP, constant_liar=True,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    distributions: dict[str, BaseDistribution] = {
        name: FloatDistribution(*bounds) for name, bounds in SPACE.items()
    }
    illegal_after_startup = 0
    for index in range(80):
        trial = study.ask(distributions)
        x, y = float(trial.params["x"]), float(trial.params["y"])
        amount = violation(x, y)
        illegal_after_startup += int(index >= STARTUP and amount > 0)
        if amount > 0:
            study.tell(trial, state=TrialState.FAIL)
        else:
            study.tell(trial, objective(x, y))
    return illegal_after_startup


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_constraint_feedback_avoids_edge_region_better_than_fail(seed: int) -> None:
    assert adapter_illegal_count(seed, edge_violation) < control_illegal_count(
        seed, edge_violation, random=False,
    )


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_constraint_feedback_avoids_far_region_better_than_random(seed: int) -> None:
    assert adapter_illegal_count(seed, far_violation) < control_illegal_count(
        seed, far_violation, random=True,
    )


def mixed_batch(adapter: SamplerAdapter) -> dict[int, Outcome]:
    outcomes: tuple[Outcome, ...] = (
        Illegal("wall", 0.125), *(Excluded(zone) for zone in RankingZone), Scored(-2.0),
    )
    proposals = adapter.ask_batch(len(outcomes))
    feedback = {proposal.trial_number: outcome for proposal, outcome in zip(proposals, outcomes, strict=True)}
    adapter.tell_batch(feedback)
    return feedback


def test_illegal_and_excluded_trials_carry_no_value_and_positive_constraint() -> None:
    """唯讀試算快照只供考卷檢查狀態、值、限制；不取得可改動的 study（研究物件）。"""
    adapter = SamplerAdapter(SPACE, SamplerSettings(0, STARTUP))
    outcomes = mixed_batch(adapter)
    for trial in adapter.trial_snapshots:
        outcome = outcomes[trial.number]
        if isinstance(outcome, Illegal):
            assert trial.state == TrialState.PRUNED
            assert trial.value is None
            assert trial.constraints["geometry"] == outcome.violation > 0
            assert trial.constraints["ranking"] == 0.0
        elif isinstance(outcome, Excluded):
            assert trial.state == TrialState.PRUNED
            assert trial.value is None
            assert trial.constraints["ranking"] == 1.0
            assert trial.constraints["geometry"] == 0.0
        else:
            assert trial.state == TrialState.COMPLETE
            assert trial.value == -2.0
            assert all(value <= 0 for value in trial.constraints.values())
            assert set(trial.constraints) == {"geometry", "ranking"}
    assert adapter.trials_told == sum(1 for _ in outcomes)


def test_no_trial_is_ever_failed() -> None:
    adapter = SamplerAdapter(SPACE, SamplerSettings(1, STARTUP))
    mixed_batch(adapter)
    assert all(trial.state in {TrialState.COMPLETE, TrialState.PRUNED} for trial in adapter.trial_snapshots)
    source = inspect.getsource(sampler_module)
    assert "TrialState.FAIL" not in source
    assert "state=FAIL" not in source


def test_fake_numbers_are_refused() -> None:
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            Scored(value)
    for violation in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            Illegal("x", violation)
    with pytest.raises(ValueError):
        Illegal("", 1.0)
    adapter = SamplerAdapter(SPACE, SamplerSettings(0, STARTUP))
    first, second = adapter.ask_batch(2)
    with pytest.raises(TypeError):
        adapter.tell_batch({first.trial_number: Scored(1.0), second.trial_number: cast(Outcome, object())})
    assert adapter.trials_told == 0
    assert all(trial.state == TrialState.RUNNING for trial in adapter.trial_snapshots)
    adapter.tell_batch({first.trial_number: Scored(1.0), second.trial_number: Scored(2.0)})


def test_tell_batch_requires_exactly_the_outstanding_trials() -> None:
    adapter = SamplerAdapter(SPACE, SamplerSettings(0, STARTUP))
    first, second = adapter.ask_batch(2)
    with pytest.raises(ValueError, match=rf"missing.*{second.trial_number}"):
        adapter.tell_batch({first.trial_number: Scored(1.0)})
    with pytest.raises(ValueError, match="extra.*999"):
        adapter.tell_batch({first.trial_number: Scored(1.0), second.trial_number: Scored(2.0), 999: Scored(3.0)})
    with pytest.raises(RuntimeError):
        adapter.ask_batch(1)
    assert adapter.trials_told == 0
    adapter.tell_batch({first.trial_number: Scored(1.0), second.trial_number: Scored(2.0)})
    assert adapter.ask_batch(1)


def test_invalid_settings_space_and_batch_size_are_refused() -> None:
    with pytest.raises(ValueError):
        SamplerSettings(0, 0)
    for space in ({}, {"x": (1.0, 1.0)}, {"x": (2.0, 1.0)}, {"x": (math.nan, 1.0)}, {"x": (0.0, math.inf)}):
        with pytest.raises(ValueError):
            SamplerAdapter(space, SamplerSettings(0, STARTUP))
    adapter = SamplerAdapter(SPACE, SamplerSettings(0, STARTUP))
    for k in (0, -1):
        with pytest.raises(ValueError):
            adapter.ask_batch(k)
    assert adapter.trials_told == 0


def test_proposal_copies_and_freezes_float_parameters() -> None:
    params = {"x": 1.0}
    proposal = Proposal(0, params)
    params["x"] = 2.0
    assert proposal.params["x"].hex() == float(1).hex()
    with pytest.raises(TypeError):
        cast(dict[str, float], proposal.params)["x"] = 3.0
