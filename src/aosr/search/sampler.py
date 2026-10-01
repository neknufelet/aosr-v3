"""Optuna（取樣函式庫）的 TPE（模型取樣）批次要題與回報轉接。

不報 FAIL（失敗）：TPE 不學這些試算，也不把它們算進亂抽期。
PRUNED 原意是「提早停掉」，這裡借用來回報沒有分數的候選，完全不報值。
原因代碼與排名區要由帳本記錄；不能只看 Optuna 的狀態欄判斷排除原因。

限制分 geometry（幾何）與 ranking（排名）兩維：不合法擺法填真實、有限、正的
幾何違反量，排名填零；排名三區填幾何零、排名一；有分數的兩維都填零。
整批算完後按試算編號由小到大回報，固定回報順序才能逐位重播並驗證決定性。
只使用記憶體儲存，一個主行程要題與回報，工作行程只負責計算。
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

import optuna
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.samplers import TPESampler
from optuna.storages import InMemoryStorage
from optuna.trial import FrozenTrial, Trial, TrialState


class RankingZone(StrEnum):
    """沒有可回報篩選分數的排名區。"""

    ELIMINATED = "eliminated"
    UNASSESSED = "unassessed"
    INCOMPARABLE = "incomparable"


@dataclass(frozen=True)
class Scored:
    """有有限篩選分數的候選。"""

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class Illegal:
    """求解前被過濾的不合法擺法，攜帶原因代碼與真實幾何違反量。"""

    reason: str
    violation: float

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a nonempty string")
        if not math.isfinite(self.violation) or self.violation <= 0:
            raise ValueError("geometry violation must be finite and positive")


@dataclass(frozen=True)
class Excluded:
    """已算完但被排名層放進沒有分數的三區之一。"""

    zone: RankingZone


Outcome: TypeAlias = Scored | Illegal | Excluded


@dataclass(frozen=True)
class SamplerSettings:
    """種子與亂抽期由呼叫端指定；constant_liar（暫估在算試算）預設開啟。"""

    seed: int
    n_startup_trials: int
    constant_liar: bool = True

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.n_startup_trials) is not int or self.n_startup_trials < 1:
            raise ValueError("n_startup_trials must be an integer >= 1")
        if type(self.constant_liar) is not bool:
            raise ValueError("constant_liar must be a boolean")


@dataclass(frozen=True)
class Proposal:
    """試算編號與獨立複製、唯讀且收窄成浮點數的參數。"""

    trial_number: int
    params: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType({key: float(value) for key, value in self.params.items()}))


class ReplayMismatch(Exception):
    """重新要到的試算編號或參數與帳本紀錄不同。"""


class SamplerAdapter:
    """一批要題、一批完整回報；支援新轉接器依帳本重播。"""

    def __init__(self, space: Mapping[str, tuple[float, float]], settings: SamplerSettings) -> None:
        if not space:
            raise ValueError("search space must not be empty")
        self._space: dict[str, BaseDistribution] = {}
        for name, (low, high) in space.items():
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"invalid bounds for {name!r}: {low!r}, {high!r}")
            self._space[name] = FloatDistribution(low, high)
        self._study = optuna.create_study(
            sampler=TPESampler(seed=settings.seed, n_startup_trials=settings.n_startup_trials,
                               constant_liar=settings.constant_liar),
            direction="minimize", storage=InMemoryStorage(),
        )
        self._outstanding: dict[int, Trial] = {}
        self._has_asked = False
        self._trials_told = 0

    def ask_batch(self, k: int) -> tuple[Proposal, ...]:
        """按要題順序回傳一批；上一批未完整回報時拒絕要題。"""
        if type(k) is not int or k < 1:
            raise ValueError("batch size must be an integer >= 1")
        if self._outstanding:
            raise RuntimeError("previous batch has outstanding trials")
        self._has_asked = True
        proposals = []
        for _ in range(k):
            trial = self._study.ask(self._space)
            self._outstanding[trial.number] = trial
            proposals.append(Proposal(trial.number, {key: float(value) for key, value in trial.params.items()}))
        return tuple(proposals)

    def tell_batch(self, outcomes: Mapping[int, Outcome]) -> None:
        """先驗完整鍵集合與型別，再按編號回報；無分數候選只填限制並剪枝。"""
        missing = self._outstanding.keys() - outcomes.keys()
        extra = outcomes.keys() - self._outstanding.keys()
        if missing or extra:
            raise ValueError(f"outcome keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        for outcome in outcomes.values():
            if not isinstance(outcome, (Scored, Illegal, Excluded)):
                raise TypeError(f"unsupported outcome type: {type(outcome).__name__}")
        for number in sorted(self._outstanding):
            trial = self._outstanding[number]
            outcome = outcomes[number]
            trial.set_constraint("geometry", outcome.violation if isinstance(outcome, Illegal) else 0.0)
            trial.set_constraint("ranking", 1.0 if isinstance(outcome, Excluded) else 0.0)
            if isinstance(outcome, Scored):
                # 帶明確狀態：不帶狀態時 Optuna 遇到非數字只警告、改標失敗；帶了就直接報錯（Scored 已先擋一道）。
                self._study.tell(trial, outcome.value, state=TrialState.COMPLETE)
            else:
                self._study.tell(trial, state=TrialState.PRUNED)
            self._trials_told += 1
        self._outstanding.clear()

    def replay(self, history: Sequence[tuple[Sequence[Proposal], Mapping[int, Outcome]]]) -> None:
        """新轉接器依批次重播，核對編號與每個參數的浮點十六進位表示後才回報。"""
        if self._has_asked:
            raise RuntimeError("replay requires a fresh adapter that has never asked")
        for recorded, outcomes in history:
            proposed = self.ask_batch(len(recorded))
            for actual, expected in zip(proposed, recorded, strict=True):
                self._check_replayed_proposal(actual, expected)
            self.tell_batch(outcomes)

    @staticmethod
    def _check_replayed_proposal(actual: Proposal, expected: Proposal) -> None:
        if actual.trial_number != expected.trial_number:
            raise ReplayMismatch(f"trial number: proposed={actual.trial_number}, recorded={expected.trial_number}")
        for name in sorted(actual.params.keys() | expected.params.keys()):
            actual_value = actual.params[name].hex() if name in actual.params else "missing"
            expected_value = expected.params[name].hex() if name in expected.params else "missing"
            if actual_value != expected_value:
                raise ReplayMismatch(
                    f"trial {actual.trial_number}, parameter {name!r}: proposed={actual_value}, recorded={expected_value}"
                )

    @property
    def trials_told(self) -> int:
        """已回報的試算數量。"""
        return self._trials_told

    @property
    def trial_snapshots(self) -> tuple[FrozenTrial, ...]:
        """只供考卷檢查的唯讀屬性；深複製快照，修改快照不會改內部研究物件。"""
        return tuple(self._study.get_trials(deepcopy=True))
