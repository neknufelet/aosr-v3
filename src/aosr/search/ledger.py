"""帳本是唯一的進度紀錄，不另用 Optuna（取樣函式庫）的儲存。

參數存 float.hex（浮點十六進位）以保存每一位，重播時能逐位核對要題結果。
每個候選完成就寫一列、flush（沖出緩衝）與 fsync（同步到磁碟）：讀者看得到
不等於斷電後留得住，不能等整批完成才落地。只有一個主行程寫帳本。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aosr.search.constraints import Reason
from aosr.search.layout import UNIT_SPACE
from aosr.search.sampler import Excluded, Illegal, Outcome, Proposal, RankingZone, SamplerSettings, Scored
from aosr.search.store import CANDIDATES_DIR, SearchStore, candidate_name


LEDGER_VERSION: Final = "aosr.search_ledger.v1"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
History: TypeAlias = tuple[tuple[tuple[Proposal, ...], dict[int, Outcome]], ...]


class LedgerHeader(BaseModel):
    """帳本第一列，固定身分、取樣設定、批大小與單位搜尋空間。"""

    model_config = FROZEN
    ledger_version: Literal["aosr.search_ledger.v1"]
    search_id: str = Field(min_length=1)
    settings_fingerprint: str = Field(min_length=1)
    project_fingerprint: str = Field(min_length=1)
    sampler: SamplerSettings
    batch_size: int = Field(ge=1, strict=True)
    search_space: dict[str, tuple[float, float]]
    physics_identity: str = Field(min_length=1)
    program_fingerprint: str = Field(min_length=1)
    purpose_fingerprint: str = Field(min_length=1)

    @field_validator("sampler", mode="before")
    @classmethod
    def _sampler_settings(cls, value: object) -> SamplerSettings:
        if isinstance(value, SamplerSettings):
            return value
        if not isinstance(value, dict) or value.keys() != {"seed", "n_startup_trials", "constant_liar"}:
            raise ValueError("sampler must record exactly the three sampler settings")
        seed: object = value["seed"]
        startup: object = value["n_startup_trials"]
        liar: object = value["constant_liar"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(startup, int) or isinstance(startup, bool) or not isinstance(liar, bool):
            raise ValueError("startup must be an integer and constant_liar a boolean")
        return SamplerSettings(seed, startup, liar)

    @field_validator("search_space")
    @classmethod
    def _fixed_space(cls, value: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        if value != UNIT_SPACE:
            raise ValueError("search_space must match the layout unit space")
        return value


class LedgerRow(BaseModel):
    """候選一列；result_file 是相對搜尋資料夾的 candidates（候選資料夾）內檔名。"""

    model_config = FROZEN
    batch_index: int = Field(ge=0, strict=True)
    trial_number: int = Field(ge=0, strict=True)
    unit_params_hex: dict[str, str]
    params_m: dict[str, float]
    outcome: Literal["scored", "illegal", "excluded"]
    score: float | None
    reason: str | None
    violation_m: float | None = Field(gt=0)
    seconds: float = Field(ge=0)
    result_file: str | None

    @field_validator("unit_params_hex")
    @classmethod
    def _hex_parameters(cls, value: dict[str, str]) -> dict[str, str]:
        if value.keys() != UNIT_SPACE.keys():
            raise ValueError("unit parameters must name exactly the search quantities")
        for name, encoded in value.items():
            number = float.fromhex(encoded)
            low, high = UNIT_SPACE[name]
            if not math.isfinite(number) or not low <= number <= high or number.hex() != encoded:
                raise ValueError("unit parameters must be canonical finite float.hex values in the unit space")
        return value

    @field_validator("params_m")
    @classmethod
    def _meter_parameters(cls, value: dict[str, float]) -> dict[str, float]:
        if value.keys() != UNIT_SPACE.keys() or any(not math.isfinite(v) or v <= 0 for v in value.values()):
            raise ValueError("meter parameters must name the three finite positive distances")
        return value

    @field_validator("result_file")
    @classmethod
    def _candidate_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] != CANDIDATES_DIR
                or path.parts[1] in (".", "..") or path.as_posix() != value or "\\" in value):
            raise ValueError("result_file must name a file directly inside candidates")
        return value

    @model_validator(mode="after")
    def _consistent_outcome(self) -> Self:
        # 結果檔名綁在試算編號上：差一號，日後重新評分就會讀到別的候選的結果。
        if self.result_file is not None and self.result_file != candidate_name(self.trial_number):
            raise ValueError("result_file must be the candidate file of this trial number")
        if self.outcome == "illegal" and self.reason is not None:
            # 不合法的原因只認硬限制的八種原因代碼，依字母排、用 + 連（constraints.to_illegal 的寫法）。
            parts = self.reason.split("+")
            if parts != sorted(set(parts)) or not set(parts) <= {reason.value for reason in Reason}:
                raise ValueError("illegal reason must be sorted constraint reason codes joined by +")
        if self.outcome == "scored":
            if self.score is None or self.reason is not None or self.violation_m is not None or self.result_file is None:
                raise ValueError("scored requires only a score and result file")
        elif self.outcome == "illegal":
            if (self.score is not None or self.reason is None or not self.reason.strip()
                    or self.violation_m is None or self.result_file is not None):
                raise ValueError("illegal requires only a reason and positive violation")
        else:
            if self.score is not None or self.violation_m is not None or self.result_file is None:
                raise ValueError("excluded requires only a ranking zone and result file")
            if self.reason not in tuple(RankingZone):
                raise ValueError("excluded reason must be a ranking zone")
        return self


def row_outcome(row: LedgerRow) -> Outcome:
    """從已驗型別的帳本列換回取樣器回報。"""
    if row.outcome == "scored" and row.score is not None:
        return Scored(row.score)
    if row.outcome == "illegal" and row.reason is not None and row.violation_m is not None:
        return Illegal(row.reason, row.violation_m)
    if row.outcome == "excluded" and row.reason is not None:
        return Excluded(RankingZone(row.reason))
    raise ValueError("inconsistent ledger outcome")


def row_from_outcome(*, batch_index: int, proposal: Proposal, params_m: Mapping[str, float],
                     outcome: Outcome, seconds: float, result_file: str | None) -> LedgerRow:
    """第 4 步完成候選後組列；十六進位參數只取自原始 Proposal（取樣提案）。"""
    kind: Literal["scored", "illegal", "excluded"]
    if isinstance(outcome, Scored):
        kind, score, reason, violation = "scored", outcome.value, None, None
    elif isinstance(outcome, Illegal):
        kind, score, reason, violation = "illegal", None, outcome.reason, outcome.violation
    else:
        kind, score, reason, violation = "excluded", None, outcome.zone.value, None
    return LedgerRow(
        batch_index=batch_index, trial_number=proposal.trial_number,
        unit_params_hex={key: value.hex() for key, value in proposal.params.items()}, params_m=dict(params_m),
        outcome=kind, score=score, reason=reason, violation_m=violation, seconds=seconds, result_file=result_file,
    )


@dataclass(frozen=True)
class LedgerRead:
    """讀回內容、是否捨棄末列，以及完整列的位元組邊界（供再次寫入修復尾端）。"""

    header: LedgerHeader
    rows: tuple[LedgerRow, ...]
    dropped_last_line: bool
    valid_bytes: int


def _check_order(rows: Sequence[LedgerRow]) -> None:
    seen: set[int] = set()
    previous_batch = -1
    for row in rows:
        if row.batch_index < previous_batch or row.trial_number in seen:
            raise ValueError("batch indices must not decrease and trial numbers must be unique")
        previous_batch = row.batch_index
        seen.add(row.trial_number)


def _read_handle(handle: BinaryIO) -> LedgerRead:
    lines = handle.readlines()
    if not lines or not lines[0].endswith(b"\n"):
        raise ValueError("missing or incomplete ledger header")
    header = LedgerHeader.model_validate_json(lines[0])
    rows: list[LedgerRow] = []
    valid_bytes = len(lines[0])
    dropped = False
    for index, line in enumerate(lines[1:], start=1):
        is_last = index == len(lines) - 1
        if is_last and not line.endswith(b"\n"):
            dropped = True
            break
        try:
            document: object = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if not is_last:
                raise ValueError(f"corrupt ledger line {index + 1}") from error
            dropped = True
            break
        # 有效 JSON 但欄位違規不是截斷，連最後一列也不能默默吞掉。
        rows.append(LedgerRow.model_validate(document))
        valid_bytes += len(line)
    _check_order(rows)
    return LedgerRead(header, tuple(rows), dropped, valid_bytes)


def _write_line(handle: BinaryIO, value: LedgerHeader | LedgerRow) -> None:
    handle.write((json.dumps(value.model_dump(mode="json"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())


class Ledger:
    """單一主行程逐列寫；重新開啟或寫入都由磁碟讀帳，不用記憶體冒充進度。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def create(cls, path: Path, header: LedgerHeader) -> Ledger:
        """獨佔建檔、表頭也落地；已有檔案即丟錯。"""
        header = LedgerHeader.model_validate(header.model_dump())
        with path.open("xb") as handle:
            _write_line(handle, header)
        return cls(path)

    @classmethod
    def open(cls, path: Path) -> Ledger:
        """核對既有帳本後接續寫入；截斷末列在下一次成功 append（追加）時修復。"""
        cls.read_status(path)
        return cls(path)

    def append(self, row: LedgerRow) -> None:
        """先核批序與唯一編號，再寫一行並 fsync；已截斷的末列先裁掉再追加。

        整段握排他檔案鎖：設計上只有一個主行程寫帳本，鎖讓「不小心有第二個」變成排隊，
        而不是兩邊照各自算的位置寫、互相蓋掉一列。
        """
        row = LedgerRow.model_validate(row.model_dump())
        with self._path.open("r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            status = _read_handle(handle)
            _check_order((*status.rows, row))
            handle.seek(status.valid_bytes)
            if status.dropped_last_line:
                handle.truncate()
            _write_line(handle, row)

    @staticmethod
    def read(path: Path) -> tuple[LedgerHeader, tuple[LedgerRow, ...]]:
        """逐行驗型別；只捨棄不完整末列，捨棄旗標由 read_status（讀取狀態）提供。"""
        status = Ledger.read_status(path)
        return status.header, status.rows

    @staticmethod
    def read_status(path: Path) -> LedgerRead:
        """末列缺換行或解不開 JSON 時捨棄並設旗標；中間壞列與型別違規一律報錯。"""
        with path.open("rb") as handle:
            return _read_handle(handle)


def header_for(store: SearchStore) -> LedgerHeader:
    """這個搜尋資料夾的帳本表頭應該長什麼樣：八格全由資料夾的快照算出來。"""
    # 修補補上專案方案的正規化指紋，聲源模型、座位與搜尋軸不能在接續時換掉。
    settings, identity = store.settings, store.identity
    document = store.project.model_dump(mode="json")
    if document["furniture"] is None:
        document.pop("furniture")
    project = json.dumps(document, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return LedgerHeader(
        ledger_version=LEDGER_VERSION, search_id=store.search_id, settings_fingerprint=settings.fingerprint,
        project_fingerprint=hashlib.sha256(project.encode("utf-8")).hexdigest(),
        sampler=settings.sampler_settings(), batch_size=settings.batch_size, search_space=dict(UNIT_SPACE),
        physics_identity=identity.physics_identity, program_fingerprint=identity.program_fingerprint,
        purpose_fingerprint=identity.purpose_settings.fingerprint,
    )


def create_for(store: SearchStore) -> Ledger:
    """在搜尋資料夾裡開新帳本，表頭照資料夾的快照寫。"""
    return Ledger.create(store.ledger_path, header_for(store))


def read_for(store: SearchStore) -> LedgerRead:
    """讀搜尋資料夾裡的帳本，表頭必須跟資料夾的快照逐格相同（設計紙第四節：中途任何一樣換了就停）。

    拿錯資料夾的帳本、或資料夾快照被換過，在這裡就報錯，不會拿另一次搜尋的進度重播。
    """
    status = Ledger.read_status(store.ledger_path)
    expected = header_for(store)
    if status.header != expected:
        differing = sorted(name for name in LedgerHeader.model_fields
                           if getattr(status.header, name) != getattr(expected, name))
        raise ValueError(f"帳本表頭跟搜尋資料夾的快照對不上：{differing}")
    return status


def replay_history(header: LedgerHeader, rows: Sequence[LedgerRow], *,
                   batch_sizes: Mapping[int, int] | None = None) -> tuple[History, tuple[LedgerRow, ...]]:
    """回傳（完整批重播歷史，未完成末批的原始列），提案依試算編號排序。

    batch_sizes（原訂批大小）以批號指定，例如預算末批只要兩個時給 {2: 2}；
    未指定批號用 header.batch_size，每格必須介於 1 與 K。完整批才交給取樣器
    replay（重播）；末批已有列供第 4 步重新要題後核對與沿用，只重算沒有結果的。
    """
    header = LedgerHeader.model_validate(header.model_dump())
    sizes = dict(batch_sizes or {})
    for index, size in sizes.items():
        if type(index) is not int or index < 0 or type(size) is not int or not 1 <= size <= header.batch_size:
            raise ValueError("batch_sizes requires nonnegative batch indices and sizes between 1 and K")
    checked = tuple(LedgerRow.model_validate(row.model_dump()) for row in rows)
    _check_order(checked)
    batches = tuple((index, tuple(group)) for index, group in groupby(checked, key=lambda row: row.batch_index))
    history: list[tuple[tuple[Proposal, ...], dict[int, Outcome]]] = []
    partial: tuple[LedgerRow, ...] = ()
    for index, batch in batches:
        expected = sizes.get(index, header.batch_size)
        if len(batch) > expected:
            raise ValueError("batch contains more rows than its planned size")
        if len(batch) < expected:
            if index != batches[-1][0]:
                raise ValueError("only the last batch may be incomplete")
            partial = batch
            break
        ordered = sorted(batch, key=lambda row: row.trial_number)
        proposals = tuple(Proposal(row.trial_number, {key: float.fromhex(value) for key, value in row.unit_params_hex.items()})
                          for row in ordered)
        history.append((proposals, {row.trial_number: row_outcome(row) for row in ordered}))
    return tuple(history), partial
