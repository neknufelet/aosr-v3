"""牆面材料梯度搜尋的步驟帳本（#561）：JSONL，第一列標頭、之後每列一步。

手法照 ``src/aosr/search/ledger.py``（θ 存 float.hex 保存每一位、每列寫完 flush 與 fsync、讀回逐列驗型別），
但列型別是自己的，不套用擺位搜尋帳本的 V1／V2。標頭固定替身身分（輸入雜湊、網格雜湊、物理身分）、
每面的 preset、變數名順序與設定指紋；讀回時順序、θ 長度、停止原因的位置都要對，
給了預期標頭就逐格比，對不上一律拒收。

v2（庚二）：前段照舊是連續下降、取規格、重調（``descent``、``snap``、``retune``），之後是候選確認迴圈一輪一輪：
每一輪「基準重調（``baseline_retune``）→ 每個候選一列跳點（``candidate_jump``）接它的重調（``candidate_retune``）
→ 一列決定（``round_decision``）」；一個候選都沒有的輪只有決定那一列。每段下降的最後一列帶停止原因、
最後一次試的步長與這段求值次數；整場最後一列（決定列）帶迴圈的停止原因。v1 檔一律拒收、不做相容。
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aosr.config.material_gradient import MaterialGradientSettings
from aosr.physics.fem_reduced import ReducedModel
from aosr.gradient.material_gradient import (
    LOOP_STOP_REASONS, STOP_REASONS, DescentResult, SnapResult, SurfaceDesign, design_names,
)


LEDGER_VERSION: Final = "aosr.material_gradient_ledger.v2"
_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
Phase = Literal["descent", "snap", "retune", "baseline_retune", "candidate_jump", "candidate_retune", "round_decision"]
SegmentPhase = Literal["descent", "retune", "baseline_retune", "candidate_retune"]
_PRE_LOOP_ORDER: Final[dict[str, int]] = {"descent": 0, "snap": 1, "retune": 2}
_SEGMENT_PHASES: Final = frozenset(("descent", "retune", "baseline_retune", "candidate_retune"))
_ROUND_PHASES: Final = frozenset(("baseline_retune", "candidate_jump", "candidate_retune", "round_decision"))
_CANDIDATE_ID: Final = re.compile(r"^(?P<surface>.+):(?P<direction>thinner|thicker)$")


class MaterialGradientHeader(BaseModel):
    """帳本第一列：替身身分、每面（面代號, preset id 或 None）、θ 變數名順序、設定指紋。"""

    model_config = _FROZEN
    ledger_version: Literal["aosr.material_gradient_ledger.v2"]
    input_hash: str = Field(min_length=1)
    mesh_hash: str = Field(min_length=1)
    physics_identity: str = Field(min_length=1)
    surfaces: tuple[tuple[str, str | None], ...] = Field(min_length=1)
    variable_names: tuple[str, ...]
    settings_fingerprint: str = Field(min_length=1)

    @field_validator("variable_names")
    @classmethod
    def _unique_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not name.strip() for name in value):
            raise ValueError("變數名必須不重複且非空白")
        return value


class SnapChoice(BaseModel):
    """取規格那一列記的每面取法；剛性面四格都是 None／False。"""

    model_config = _FROZEN
    surface_id: str = Field(min_length=1)
    nearest_thickness_m: float | None
    thickness_m: float | None
    cavity_m: float | None
    cavity_clipped: bool
    thickness_retreated: bool


def _canonical_hex(value: tuple[str, ...]) -> tuple[str, ...]:
    for encoded in value:
        try:
            number = float.fromhex(encoded)
        except (OverflowError, ValueError) as error:
            raise ValueError("必須是有限的 float.hex") from error
        if not math.isfinite(number) or number.hex() != encoded:
            raise ValueError("必須是正規化的有限 float.hex")
    return value


class CandidateJump(BaseModel):
    """跳點那一列：往哪個方向、跳到的厚度與空腔（沒有空腔層是 None）、Δθ（float.hex）、預測變化（梯度 · Δθ）。"""

    model_config = _FROZEN
    direction: Literal["thinner", "thicker"]
    thickness_m: float
    cavity_m: float | None
    delta_hex: tuple[str, ...]
    predicted_change: float

    @field_validator("delta_hex")
    @classmethod
    def _delta(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_hex(value)


class CandidateOutcome(BaseModel):
    """一個候選確認完的結果：代號、預測變化、重調後的值、改善（基準重調後的值減它）。"""

    model_config = _FROZEN
    candidate: str = Field(pattern=_CANDIDATE_ID.pattern)
    predicted_change: float
    value: float
    improvement: float


class RoundDecision(BaseModel):
    """一輪的決定：基準重調後的值（沒開基準是 None）、每個候選的結果（照確認順序）、門檻、接受了誰、
    這一輪與累計的求值次數、下一輪最壞情況要幾次（有算才記；記「求值到上限」時一定有）。"""

    model_config = _FROZEN
    baseline_value: float | None
    outcomes: tuple[CandidateOutcome, ...]
    threshold: float = Field(gt=0.0)
    accepted: str | None
    round_evaluations: int = Field(ge=0, strict=True)
    total_evaluations: int = Field(ge=0, strict=True)
    next_round_bound: int | None = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [outcome.candidate for outcome in self.outcomes]
        if len(set(names)) != len(names):
            raise ValueError("同一輪的候選不可重複")
        if (self.baseline_value is None) != (not self.outcomes):
            raise ValueError("有候選才有基準重調，沒有候選就沒有基準值")
        if self.round_evaluations > self.total_evaluations:
            raise ValueError("這一輪的求值次數不可大於累計")
        if self.baseline_value is not None and any(
                outcome.improvement != self.baseline_value - outcome.value for outcome in self.outcomes):
            raise ValueError("改善必須等於基準重調後的值減候選重調後的值")
        best = max(self.outcomes, key=lambda outcome: outcome.improvement, default=None)
        expected = best.candidate if best is not None and best.improvement > self.threshold else None
        if self.accepted != expected:
            raise ValueError(f"接受的候選必須是改善最大且超過門檻的那一個（{expected}），記的是 {self.accepted}")
        return self


class MaterialGradientRow(BaseModel):
    """一列：全帳本第幾列（從 0 起連號）、階段、第幾輪、哪個候選、θ（float.hex）、值與兩分項、梯度範數、步長、
    秒數、停止原因、最後一次試的步長、這段求值次數、取法、跳點、決定。

    ``snap`` 列沒有梯度範數與步長、要帶每面取法；四種下降段（``descent``、``retune``、``baseline_retune``、
    ``candidate_retune``）帶梯度範數與步長，停止原因、最後一次試的步長與這段求值次數三格同進同出、只在每段的最後一列；
    ``candidate_jump`` 帶跳點、``round_decision`` 帶決定，停止原因只認迴圈的那一組、只在整場最後一列。
    前段三種沒有輪與候選；迴圈四種都有輪，候選只在跳點與候選重調。
    """

    model_config = _FROZEN
    step: int = Field(ge=0, strict=True)
    phase: Phase
    round: int | None = Field(ge=0, strict=True)
    candidate: str | None
    theta_hex: tuple[str, ...]
    value: float
    components: tuple[float, float]
    gradient_norm: float | None = Field(ge=0.0)
    step_size: float | None = Field(ge=0.0)
    seconds: float = Field(ge=0.0)
    stop_reason: str | None
    final_trial_step: float | None = Field(ge=0.0)
    evaluations: int | None = Field(ge=1, strict=True)
    snap: tuple[SnapChoice, ...] | None
    jump: CandidateJump | None
    decision: RoundDecision | None

    @field_validator("theta_hex")
    @classmethod
    def _theta(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_hex(value)

    @model_validator(mode="after")
    def _phase_fields(self) -> Self:
        in_round = self.phase in _ROUND_PHASES
        if (self.round is not None) != in_round:
            raise ValueError(f"{self.phase} 列{'要' if in_round else '不'}帶輪號")
        with_candidate = self.phase in ("candidate_jump", "candidate_retune")
        if (self.candidate is not None) != with_candidate:
            raise ValueError(f"{self.phase} 列{'要' if with_candidate else '不'}帶候選代號")
        if self.candidate is not None and not _CANDIDATE_ID.match(self.candidate):
            raise ValueError(f"候選代號必須是「面代號:thinner|thicker」，收到 {self.candidate}")
        if (self.snap is not None) != (self.phase == "snap") or (self.jump is not None) != (self.phase == "candidate_jump"):
            raise ValueError("取法只在取規格列、跳點只在跳點列")
        if (self.decision is not None) != (self.phase == "round_decision"):
            raise ValueError("決定只在決定列")
        segment = self.phase in _SEGMENT_PHASES
        if (self.gradient_norm is not None) != segment or (self.step_size is not None) != segment:
            raise ValueError("梯度範數與步長只在下降段的列")
        if segment:
            if self.stop_reason is not None and self.stop_reason not in STOP_REASONS:
                raise ValueError(f"下降段的停止原因只認 {list(STOP_REASONS)}")
            if len({self.stop_reason is None, self.final_trial_step is None, self.evaluations is None}) != 1:
                raise ValueError("停止原因、最後一次試的步長與這段求值次數要同在一列")
        else:
            if self.final_trial_step is not None or self.evaluations is not None:
                raise ValueError("最後一次試的步長與求值次數只在下降段的最後一列")
            allowed = LOOP_STOP_REASONS if self.phase == "round_decision" else ()
            if self.stop_reason is not None and self.stop_reason not in allowed:
                raise ValueError(f"{self.phase} 列的停止原因只認 {list(allowed)}")
        if self.jump is not None and self.candidate is not None and not self.candidate.endswith(f":{self.jump.direction}"):
            raise ValueError("候選代號的方向要等於跳點的方向")
        return self

    def theta(self) -> tuple[float, ...]:
        return tuple(float.fromhex(encoded) for encoded in self.theta_hex)


# ---- 造列 ----

def header_for(model: ReducedModel, designs: Sequence[SurfaceDesign],
               settings: MaterialGradientSettings) -> MaterialGradientHeader:
    """這一次搜尋的標頭：全由替身、設計清單與設定算出來。"""
    return MaterialGradientHeader(
        ledger_version=LEDGER_VERSION, input_hash=model.input_hash, mesh_hash=model.mesh_hash,
        physics_identity=model.physics_identity,
        surfaces=tuple((design.surface_id, design.preset_id) for design in designs),
        variable_names=design_names(designs), settings_fingerprint=settings.fingerprint)


def _row(**fields: object) -> MaterialGradientRow:
    blank: dict[str, object] = {
        "round": None, "candidate": None, "gradient_norm": None, "step_size": None, "stop_reason": None,
        "final_trial_step": None, "evaluations": None, "snap": None, "jump": None, "decision": None}
    return MaterialGradientRow.model_validate({**blank, **fields})


def descent_rows(result: DescentResult, phase: SegmentPhase, first_step: int, *, round: int | None = None,
                 candidate: str | None = None) -> tuple[MaterialGradientRow, ...]:
    """一段下降的每一步換成帳本列；停止原因、最後一次試的步長與求值次數只放在這一段的最後一列。"""
    last = len(result.steps) - 1
    return tuple(_row(
        step=first_step + offset, phase=phase, round=round, candidate=candidate,
        theta_hex=tuple(value.hex() for value in item.theta), value=item.value, components=item.components,
        gradient_norm=item.gradient_norm, step_size=item.step_size, seconds=item.seconds,
        stop_reason=result.stop_reason if offset == last else None,
        final_trial_step=result.final_trial_step if offset == last else None,
        evaluations=result.evaluations if offset == last else None,
    ) for offset, item in enumerate(result.steps))


def snap_row(snapped: SnapResult, value: float, components: tuple[float, float], seconds: float,
             step: int) -> MaterialGradientRow:
    """取規格一列：取到的 θ、在那裡重算的值與分項、每面取法。"""
    return _row(step=step, phase="snap", theta_hex=tuple(float(value).hex() for value in snapped.theta),
                value=value, components=components, seconds=seconds,
                snap=tuple(SnapChoice(**choice._asdict()) for choice in snapped.surfaces))


def jump_row(step: int, round: int, candidate: str, theta: Sequence[float], value: float,
             components: tuple[float, float], jump: CandidateJump) -> MaterialGradientRow:
    """跳點一列：跳到的 θ 與那裡的值（就是候選重調的起點那一次求值，不另外求）。"""
    return _row(step=step, phase="candidate_jump", round=round, candidate=candidate,
                theta_hex=tuple(float(value).hex() for value in theta), value=value, components=components,
                seconds=0.0, jump=jump)


def decision_row(step: int, round: int, theta: Sequence[float], value: float, components: tuple[float, float],
                 seconds: float, decision: RoundDecision, stop_reason: str | None) -> MaterialGradientRow:
    """決定一列：這一輪交出去的 θ 與值、決定內容；整場停在這一輪時帶迴圈的停止原因。"""
    return _row(step=step, phase="round_decision", round=round,
                theta_hex=tuple(float(value).hex() for value in theta), value=value, components=components,
                seconds=seconds, stop_reason=stop_reason, decision=decision)


# ---- 讀回驗證 ----

def _check_segments(rows: Sequence[MaterialGradientRow]) -> None:
    """下降段（同階段、同輪、同候選連在一起的列）的停止原因恰在最後一列。

    每一列寫完就 fsync，寫到一半斷掉的帳本最後一段可能還沒寫到停止原因：只有整本帳的最後一列可以缺。
    """
    for index, row in enumerate(rows):
        if row.phase not in _SEGMENT_PHASES:
            continue
        following = rows[index + 1] if index + 1 < len(rows) else None
        if following is None and row.stop_reason is None:
            continue
        segment_ends = following is None or (following.phase, following.round, following.candidate) != (
            row.phase, row.round, row.candidate)
        if (row.stop_reason is not None) != segment_ends:
            raise ValueError(f"第 {index} 列：停止原因只能、而且一定要在每段下降的最後一列")


def _check_pre_loop(rows: Sequence[MaterialGradientRow], surface_ids: tuple[str, ...]) -> int:
    """前段（下降、取規格、重調）階段不倒退；回前段有幾列。迴圈開始後不准再出現前段的列。"""
    count = 0
    while count < len(rows) and rows[count].phase in _PRE_LOOP_ORDER:
        row = rows[count]
        if count and _PRE_LOOP_ORDER[row.phase] < _PRE_LOOP_ORDER[rows[count - 1].phase]:
            raise ValueError(f"第 {count} 列的階段 {row.phase} 倒退，順序必須是 descent、snap、retune")
        if row.snap is not None and tuple(choice.surface_id for choice in row.snap) != surface_ids:
            raise ValueError(f"第 {count} 列的取法面序跟標頭不同")
        count += 1
    if any(row.phase in _PRE_LOOP_ORDER for row in rows[count:]):
        raise ValueError("候選確認迴圈開始之後不准再出現 descent、snap、retune 列")
    return count


def _round_blocks(rows: Sequence[MaterialGradientRow], first: int) -> list[list[MaterialGradientRow]]:
    """迴圈部分照輪切塊；輪號從 0 起、每一輪結束（決定列）後下一列的輪號恰好加一，同一輪裡決定是最後一列。"""
    blocks: list[list[MaterialGradientRow]] = []
    for index in range(first, len(rows)):
        row = rows[index]
        expected = 0 if not blocks else blocks[-1][0].round
        if blocks and blocks[-1][-1].phase == "round_decision":
            expected = (blocks[-1][0].round or 0) + 1
        if row.round != expected:
            raise ValueError(f"第 {index} 列的輪號 {row.round}，這裡應是第 {expected} 輪（輪號不可跳、不可退，"
                             "決定列必須是那一輪的最後一列）")
        if not blocks or blocks[-1][-1].phase == "round_decision":
            blocks.append([row])
        else:
            blocks[-1].append(row)
    return blocks


def _check_round(block: Sequence[MaterialGradientRow], start_theta: tuple[str, ...] | None,
                 is_last_row_of_ledger: bool) -> tuple[str, ...] | None:
    """一輪的列序：（基準重調＋（跳點＋候選重調）×n＋決定）或只有決定；最後一輪可以沒寫完。

    基準的起點、零候選輪的 θ 都要等於上一輪交出去的 θ（或前段最後一列）；跳點的 θ 要等於它的重調起點；
    決定列的基準值、每個候選的值與順序、接受的 θ 要跟前面的列一致。回這一輪交出去的 θ（沒寫完回 None）。
    """
    label = f"第 {block[0].round} 輪"
    decision_row_ = block[-1] if block[-1].phase == "round_decision" else None
    body = block[:-1] if decision_row_ is not None else block
    position = 0
    baseline: list[MaterialGradientRow] = []
    while position < len(body) and body[position].phase == "baseline_retune":
        baseline.append(body[position])
        position += 1
    if baseline and start_theta is not None and baseline[0].theta_hex != start_theta:
        raise ValueError(f"{label}的基準重調起點不是上一輪交出去的 θ")
    if not baseline and position < len(body):
        raise ValueError(f"{label}要先有基準重調才有候選")
    candidates: list[tuple[str, MaterialGradientRow, list[MaterialGradientRow]]] = []
    while position < len(body):
        jump = body[position]
        if jump.phase != "candidate_jump":
            raise ValueError(f"{label}：候選重調前面必須是它的跳點列，第 {jump.step} 列是 {jump.phase}")
        position += 1
        retune: list[MaterialGradientRow] = []
        while position < len(body) and body[position].phase == "candidate_retune":
            if body[position].candidate != jump.candidate:
                raise ValueError(f"{label}：第 {body[position].step} 列的候選跟它前面的跳點不同")
            retune.append(body[position])
            position += 1
        if retune and retune[0].theta_hex != jump.theta_hex:
            raise ValueError(f"{label}：候選 {jump.candidate} 的重調起點不是它的跳點")
        if not retune and (position < len(body) or decision_row_ is not None):
            raise ValueError(f"{label}：候選 {jump.candidate} 有跳點沒有重調")
        candidates.append((str(jump.candidate), jump, retune))
    if decision_row_ is None:
        return None
    _check_decision(label, decision_row_, start_theta, baseline, candidates, is_last_row_of_ledger)
    return decision_row_.theta_hex


def _check_decision(label: str, row: MaterialGradientRow, start_theta: tuple[str, ...] | None,
                    baseline: Sequence[MaterialGradientRow],
                    candidates: Sequence[tuple[str, MaterialGradientRow, Sequence[MaterialGradientRow]]],
                    is_last_row_of_ledger: bool) -> None:
    decision = row.decision
    if decision is None:
        raise ValueError(f"{label}的決定列沒有決定")
    if [outcome.candidate for outcome in decision.outcomes] != [name for name, _, _ in candidates]:
        raise ValueError(f"{label}：決定列記的候選與順序跟前面的跳點列不同")
    if any(outcome.predicted_change != (jump.jump.predicted_change if jump.jump else None)
           or outcome.value != retune[-1].value
           for outcome, (_, jump, retune) in zip(decision.outcomes, candidates, strict=True)):
        raise ValueError(f"{label}：決定列記的預測變化或候選值跟跳點與重調列不同")
    if baseline and decision.baseline_value != baseline[-1].value:
        raise ValueError(f"{label}：決定列的基準值不是基準重調最後一列的值")
    if baseline:
        chosen = next((retune for name, _, retune in candidates if name == decision.accepted), baseline)
        if row.theta_hex != chosen[-1].theta_hex or row.value != chosen[-1].value:
            raise ValueError(f"{label}：決定列交出去的 θ 不是{'接受的候選' if decision.accepted else '基準'}重調後的 θ")
    elif start_theta is not None and row.theta_hex != start_theta:
        raise ValueError(f"{label}：沒有候選的輪交出去的 θ 必須是原樣")
    if row.stop_reason is not None and not is_last_row_of_ledger:
        raise ValueError(f"{label}：帶迴圈停止原因的決定列必須是整本帳的最後一列")
    if row.stop_reason is None and decision.accepted is None:
        raise ValueError(f"{label}：沒有接受任何候選的輪一定要停、要帶停止原因")


def check_rows(header: MaterialGradientHeader, rows: Sequence[MaterialGradientRow]) -> None:
    """全帳本連號、θ 長度等於變數數、前段階段不倒退、每段下降的停止原因恰在最後一列、迴圈一輪一輪的列序與決定一致、
    累計求值次數逐輪相加。最後一輪可以沒寫完（寫到一半斷掉的帳本，接續時丟掉那半輪）。"""
    surface_ids = tuple(surface_id for surface_id, _ in header.surfaces)
    for index, row in enumerate(rows):
        if row.step != index:
            raise ValueError(f"第 {index} 列的步序是 {row.step}，帳本列必須從 0 連號")
        if len(row.theta_hex) != len(header.variable_names):
            raise ValueError(f"第 {index} 列的 θ 長度 {len(row.theta_hex)} 不等於變數數 {len(header.variable_names)}")
        if row.jump is not None and len(row.jump.delta_hex) != len(header.variable_names):
            raise ValueError(f"第 {index} 列的 Δθ 長度不等於變數數")
    _check_segments(rows)
    pre_loop = _check_pre_loop(rows, surface_ids)
    start_theta = rows[pre_loop - 1].theta_hex if pre_loop else None
    total = 0
    for block in _round_blocks(rows, pre_loop):
        start_theta = _check_round(block, start_theta, block[-1] is rows[-1])
        last = block[-1]
        if last.decision is not None:
            spent = 1 + sum(row.evaluations or 0 for row in block) if last.decision.outcomes else (
                last.decision.round_evaluations)
            if last.decision.round_evaluations != spent or last.decision.total_evaluations != total + spent:
                raise ValueError(f"第 {last.round} 輪的求值次數跟各段下降記的對不上（或累計沒有逐輪相加）")
            total = last.decision.total_evaluations


# ---- 寫與讀 ----

def _write_line(handle: BinaryIO, value: BaseModel) -> None:
    handle.write((json.dumps(value.model_dump(mode="json"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())


def write_ledger(path: Path, header: MaterialGradientHeader, rows: Sequence[MaterialGradientRow]) -> None:
    """獨佔建檔（已有檔案即丟錯），先驗整段再逐列寫，每列 flush 與 fsync。"""
    checked_header = MaterialGradientHeader.model_validate(header.model_dump())
    checked = tuple(MaterialGradientRow.model_validate(row.model_dump()) for row in rows)
    check_rows(checked_header, checked)
    with path.open("xb") as handle:
        _write_line(handle, checked_header)
        for row in checked:
            _write_line(handle, row)


def append_rows(path: Path, header: MaterialGradientHeader, rows: Sequence[MaterialGradientRow]) -> None:
    """接在既有帳本後面寫：先讀回整本（標頭要等於 ``header``），連同新列一起驗，再逐列寫、每列 flush 與 fsync。"""
    _, existing = read_ledger(path, expected_header=header)
    checked = tuple(MaterialGradientRow.model_validate(row.model_dump()) for row in rows)
    check_rows(header, (*existing, *checked))
    with path.open("ab") as handle:
        for row in checked:
            _write_line(handle, row)


def truncate_rows(path: Path, header: MaterialGradientHeader, keep: int) -> None:
    """只留標頭與前 ``keep`` 列（接續時丟掉沒寫完的半輪）；截完整本再驗一次。"""
    _, rows = read_ledger(path, expected_header=header)
    if not 0 <= keep <= len(rows):
        raise ValueError(f"要留的列數 {keep} 不在 0～{len(rows)}")
    with path.open("rb") as handle:
        lines = handle.readlines()
    size = sum(len(line) for line in lines[:keep + 1])
    with path.open("r+b") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())
    read_ledger(path, expected_header=header)


def read_ledger(path: Path, *, expected_header: MaterialGradientHeader | None = None,
                ) -> tuple[MaterialGradientHeader, tuple[MaterialGradientRow, ...]]:
    """逐列驗型別與順序；給了預期標頭就逐格比（替身換了、設定換了都拒收）。壞列一律報錯，不捨棄。

    標頭先看版本：不是這一版（例如 v1）就拒收、訊息講明版本，不做舊版相容。
    """
    with path.open("rb") as handle:
        lines = handle.readlines()
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("帳本是空的或有一列沒寫完")
    document = json.loads(lines[0])
    version = document.get("ledger_version") if isinstance(document, dict) else None
    if version != LEDGER_VERSION:
        raise ValueError(f"帳本版本是 {version!r}，這一版只收 {LEDGER_VERSION}；舊版帳本不做相容，要重跑")
    header = MaterialGradientHeader.model_validate_json(lines[0])
    if expected_header is not None and header != expected_header:
        actual, expected = header.model_dump(), expected_header.model_dump()
        differing = sorted(name for name in actual if actual[name] != expected[name])
        raise ValueError(f"帳本標頭跟這一次對不上：{differing}")
    rows = tuple(MaterialGradientRow.model_validate_json(line) for line in lines[1:])
    check_rows(header, rows)
    return header, rows
