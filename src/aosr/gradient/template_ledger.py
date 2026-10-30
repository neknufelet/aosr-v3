"""換樣板的帳本 ``templates.jsonl``（#561 庚三）：JSONL，第一列標頭、之後每列一步；現在的整房配置只住這裡。

手法照庚二帳本（θ 存 float.hex、每列寫完 flush 與 fsync、讀回逐列驗型別與列序、給了預期標頭就逐格比），
但庚二的 v4 帳本 ``ledger.jsonl`` 一個位元都不動：換樣板會換掉一面的構造，θ 的版面（變數名與個數）跟著變，所以這裡的
**每一列都自帶**配置 ``surfaces``（每面面代號與 preset id，不貼是 None）、``variable_names`` 與 ``theta_hex``，讀回不靠標頭的版面。

標頭綁替身身分（輸入雜湊、網格雜湊、物理身分）、完整設定指紋（含 ``[templates]``，``full_fingerprint``）、評分常數指紋、
開始換樣板時 ``ledger.jsonl`` 整份的 SHA-256（庚二帳本已收尾才准開始；它變了就接不上）、起點配置，與六面方案原本那面牆的
材料摘要（任何一面都可能換成「不貼」，所以六面都綁；這一格比派工單多，理由同辛三審查補的 ``kept_digests``）。

每一輪：快速評分（``quick_score``，每個（面, 目標）一列）→ 基準重調（``baseline_retune``）→ 每個進重調的候選一段重調
（``candidate_retune``，第一列就是它快速評分那一點）→ 一列決定（``template_decision``）；沒開的輪只有決定那一列。
每段重調的最後一列帶停止原因、最後一次試的步長與這段求值次數；整段最後一列（決定列）帶換樣板的停止原因。
快速評分列另帶 ``swapped_retunable``：換過去那一面有沒有空腔變數（24K、96K 的樣板沒有空氣層、「不貼」沒有變數，都是 false
＝「無可重調變數」）；讀回時從那一列自己的變數名重算比對，記錯就拒收。
決定列的 ``seconds`` 是整輪的牆鐘秒數，總時間＝各決定列 ``seconds`` 加總（只在一輪寫完決定列時比上限、不看牆鐘）。
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Final, Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aosr.config.material_gradient import MaterialGradientSettings
from aosr.gradient.material_gradient import STOP_REASONS, DescentResult
from aosr.gradient.template_rounds import (
    CANDIDATE_SEPARATOR, NONE_TARGET, TEMPLATE_STOP_MAX_EVALUATIONS, TEMPLATE_STOP_MAX_ROUNDS,
    TEMPLATE_STOP_MAX_SECONDS, TEMPLATE_STOP_NO_ALTERNATIVES, TEMPLATE_STOP_NO_IMPROVEMENT, TEMPLATE_STOP_REASONS,
    QuickScore, Surfaces, alternatives, next_round_bound, preset_for_target, retune_count,
)


TEMPLATE_LEDGER_VERSION: Final = "aosr.material_gradient_templates.v1"
_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_SHA256 = r"^[0-9a-f]{64}$"
TemplatePhase = Literal["quick_score", "baseline_retune", "candidate_retune", "template_decision"]
SegmentPhase = Literal["baseline_retune", "candidate_retune"]
_SEGMENTS: Final = frozenset(("baseline_retune", "candidate_retune"))
_CANDIDATE: Final = re.compile(rf"^(?P<surface>.+){re.escape(CANDIDATE_SEPARATOR)}(?P<target>[a-z][a-z0-9_]*)$")


class TemplateHeader(BaseModel):
    """帳本第一列：替身身分、完整設定指紋、評分常數指紋、庚二帳本摘要、起點配置、六面原牆材料摘要。"""

    model_config = _FROZEN
    ledger_version: Literal["aosr.material_gradient_templates.v1"]
    input_hash: str = Field(min_length=1)
    mesh_hash: str = Field(min_length=1)
    physics_identity: str = Field(min_length=1)
    settings_fingerprint: str = Field(pattern=_SHA256)
    objective_fingerprint: str = Field(pattern=_SHA256)
    base_ledger_sha256: str = Field(pattern=_SHA256)
    initial_surfaces: tuple[tuple[str, str | None], ...] = Field(min_length=1)
    kept_digests: dict[str, str]

    @model_validator(mode="after")
    def _surfaces(self) -> Self:
        ids = [surface_id for surface_id, _ in self.initial_surfaces]
        if len(set(ids)) != len(ids) or sorted(self.kept_digests) != sorted(ids):
            raise ValueError("起點配置的面代號不可重複，原牆摘要要恰好每一面一個")
        return self


class TemplateOutcome(BaseModel):
    """一個進重調的候選：代號、快速評分的值、重調後的值、改善（基準重調後的值減它）。"""

    model_config = _FROZEN
    candidate: str = Field(pattern=_CANDIDATE.pattern)
    quick_value: float
    value: float
    improvement: float


class TemplateDecision(BaseModel):
    """一輪的決定：基準重調後的值（沒開是 None）、快速評分幾個、進重調的候選結果（照重調順序）、門檻、接受了誰、
    這一輪與累計的求值次數、下一輪最壞要幾次（有算才記）。"""

    model_config = _FROZEN
    baseline_value: float | None
    quick_count: int = Field(ge=0, strict=True)
    outcomes: tuple[TemplateOutcome, ...]
    threshold: float = Field(gt=0.0)
    accepted: str | None
    round_evaluations: int = Field(ge=0, strict=True)
    total_evaluations: int = Field(ge=0, strict=True)
    next_round_bound: int | None = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [outcome.candidate for outcome in self.outcomes]
        if len(set(names)) != len(names):
            raise ValueError("同一輪進重調的候選不可重複")
        opened = self.quick_count > 0
        if (self.baseline_value is not None) != opened or bool(self.outcomes) != opened:
            raise ValueError("開了的輪（快速評分至少一個）才有基準與候選結果，沒開的輪都沒有")
        if not 0 <= self.round_evaluations <= self.total_evaluations:
            raise ValueError("這一輪的求值次數不可大於累計")
        if not opened and self.round_evaluations != 0:
            raise ValueError("沒開的輪求值只能是 0 次")
        if len(self.outcomes) > self.quick_count:
            raise ValueError("進重調的候選不可多於快速評分的個數")
        if self.baseline_value is not None and any(
                outcome.improvement != self.baseline_value - outcome.value for outcome in self.outcomes):
            raise ValueError("改善必須等於基準重調後的值減候選重調後的值")
        best = max(self.outcomes, key=lambda outcome: outcome.improvement, default=None)
        expected = best.candidate if best is not None and best.improvement > self.threshold else None
        if self.accepted != expected:
            raise ValueError(f"接受的候選必須是改善最大且超過門檻的那一個（{expected}），記的是 {self.accepted}")
        return self


def _canonical_hex(value: tuple[str, ...]) -> tuple[str, ...]:
    for encoded in value:
        try:
            number = float.fromhex(encoded)
        except (OverflowError, ValueError) as error:
            raise ValueError("必須是有限的 float.hex") from error
        if not math.isfinite(number) or number.hex() != encoded:
            raise ValueError("必須是正規化的有限 float.hex")
    return value


class TemplateRow(BaseModel):
    """一列：全帳本第幾列（從 0 起連號）、階段、第幾輪、哪個候選、這一列的配置與變數名、θ（float.hex）、值與兩分項、
    梯度範數、步長、秒數、停止原因、最後一次試的步長、這段求值次數、決定。

    快速評分列帶候選、不帶梯度範數與步長；兩種重調段帶梯度範數與步長，停止原因、最後一次試的步長與這段求值次數三格
    同進同出、只在每段的最後一列；候選重調帶候選；決定列帶決定，停止原因只認換樣板那一組、只在整段最後一列。
    """

    model_config = _FROZEN
    step: int = Field(ge=0, strict=True)
    phase: TemplatePhase
    round: int = Field(ge=0, strict=True)
    candidate: str | None
    surfaces: tuple[tuple[str, str | None], ...] = Field(min_length=1)
    variable_names: tuple[str, ...]
    theta_hex: tuple[str, ...]
    value: float
    components: tuple[float, float]
    gradient_norm: float | None = Field(ge=0.0)
    step_size: float | None = Field(ge=0.0)
    seconds: float = Field(ge=0.0)
    stop_reason: str | None
    final_trial_step: float | None = Field(ge=0.0)
    evaluations: int | None = Field(ge=1, strict=True)
    decision: TemplateDecision | None
    swapped_retunable: bool | None = Field(strict=True)

    @field_validator("theta_hex")
    @classmethod
    def _theta(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_hex(value)

    @model_validator(mode="after")
    def _phase_fields(self) -> Self:
        with_candidate = self.phase in ("quick_score", "candidate_retune")
        if (self.candidate is not None) != with_candidate:
            raise ValueError(f"{self.phase} 列{'要' if with_candidate else '不'}帶候選代號")
        if (self.decision is not None) != (self.phase == "template_decision"):
            raise ValueError("決定只在決定列")
        if len(self.theta_hex) != len(self.variable_names):
            raise ValueError(f"θ 長度 {len(self.theta_hex)} 不等於變數數 {len(self.variable_names)}")
        _check_layout(self.surfaces, self.variable_names)
        expected = swapped_retunable(self.candidate, self.variable_names) if self.phase == "quick_score" else None
        if self.swapped_retunable != expected:
            raise ValueError(f"{self.phase} 列的「換過去那面有沒有空腔變數」要是 {expected}，記的是 {self.swapped_retunable}")
        segment = self.phase in _SEGMENTS
        if (self.gradient_norm is not None) != segment or (self.step_size is not None) != segment:
            raise ValueError("梯度範數與步長只在重調段的列")
        if segment:
            if self.stop_reason is not None and self.stop_reason not in STOP_REASONS:
                raise ValueError(f"重調段的停止原因只認 {list(STOP_REASONS)}")
            if len({self.stop_reason is None, self.final_trial_step is None, self.evaluations is None}) != 1:
                raise ValueError("停止原因、最後一次試的步長與這段求值次數要同在一列")
            return self
        if self.final_trial_step is not None or self.evaluations is not None:
            raise ValueError("最後一次試的步長與求值次數只在重調段的最後一列")
        allowed = TEMPLATE_STOP_REASONS if self.phase == "template_decision" else ()
        if self.stop_reason is not None and self.stop_reason not in allowed:
            raise ValueError(f"{self.phase} 列的停止原因只認 {list(allowed)}")
        return self

    def theta(self) -> tuple[float, ...]:
        return tuple(float.fromhex(encoded) for encoded in self.theta_hex)


def swapped_retunable(candidate: str | None, names: Sequence[str]) -> bool:
    """候選換過去的那一面有沒有可重調的空腔變數（那一面的變數有兩格：厚度、空腔）；候選代號不合格式就拒收。"""
    match = _CANDIDATE.match(str(candidate))
    if match is None:
        raise ValueError(f"候選代號 {candidate} 不是「面代號->目標」")
    return sum(1 for name in names if name.split(".", 1)[0] == match.group("surface")) == 2


def _check_layout(surfaces: Surfaces, names: Sequence[str]) -> None:
    """變數名的版面：照面序、每面一或兩格（厚度、空腔）、名字以「面代號.」開頭；不貼的面沒有變數。"""
    ids = [surface_id for surface_id, _ in surfaces]
    if len(set(ids)) != len(ids):
        raise ValueError(f"面代號重複：{ids}")
    position = 0
    for surface_id, preset_id in surfaces:
        count = 0
        while position < len(names) and names[position].startswith(f"{surface_id}."):
            position += 1
            count += 1
        if (preset_id is None and count) or (preset_id is not None and not 1 <= count <= 2):
            raise ValueError(f"面 {surface_id}（preset {preset_id}）的變數個數 {count} 不合：不貼的面沒有變數，"
                             "有樣板的面是厚度或厚度加空腔")
    if position != len(names):
        raise ValueError(f"變數名沒有照面序排、或有不屬於任何面的變數：{list(names)}")


class TemplateStart(NamedTuple):
    """換樣板的起點：庚二帳本收尾那一列的配置、θ（float.hex）、值與兩分項。"""

    surfaces: Surfaces
    theta_hex: tuple[str, ...]
    value: float
    components: tuple[float, float]


# ---- 造列 ----

def _row(**fields: object) -> TemplateRow:
    blank: dict[str, object] = {"candidate": None, "gradient_norm": None, "step_size": None, "stop_reason": None,
                                "final_trial_step": None, "evaluations": None, "decision": None,
                                "swapped_retunable": None}
    return TemplateRow.model_validate({**blank, **fields})


def quick_row(step: int, round: int, score: QuickScore) -> TemplateRow:
    """快速評分一列：換過去的配置、變數名、θ、值與兩分項、秒數。"""
    return _row(step=step, phase="quick_score", round=round, candidate=score.candidate_id, surfaces=score.surfaces,
                variable_names=score.variable_names, theta_hex=tuple(float(value).hex() for value in score.theta),
                value=score.value, components=score.components, seconds=score.seconds,
                swapped_retunable=swapped_retunable(score.candidate_id, score.variable_names))


def segment_rows(result: DescentResult, phase: SegmentPhase, first_step: int, *, round: int, surfaces: Surfaces,
                 variable_names: tuple[str, ...], candidate: str | None = None) -> tuple[TemplateRow, ...]:
    """一段重調的每一步換成帳本列；停止原因、最後一次試的步長與求值次數只放在這一段的最後一列。"""
    last = len(result.steps) - 1
    return tuple(_row(
        step=first_step + offset, phase=phase, round=round, candidate=candidate, surfaces=surfaces,
        variable_names=variable_names, theta_hex=tuple(value.hex() for value in item.theta), value=item.value,
        components=item.components, gradient_norm=item.gradient_norm, step_size=item.step_size, seconds=item.seconds,
        stop_reason=result.stop_reason if offset == last else None,
        final_trial_step=result.final_trial_step if offset == last else None,
        evaluations=result.evaluations if offset == last else None,
    ) for offset, item in enumerate(result.steps))


def decision_row(step: int, round: int, *, surfaces: Surfaces, variable_names: tuple[str, ...],
                 theta_hex: tuple[str, ...], value: float, components: tuple[float, float], seconds: float,
                 decision: TemplateDecision, stop_reason: str | None) -> TemplateRow:
    """決定一列：這一輪交出去的配置、θ 與值、整輪秒數、決定內容；整段停在這一輪時帶換樣板的停止原因。"""
    return _row(step=step, phase="template_decision", round=round, surfaces=surfaces, variable_names=variable_names,
                theta_hex=theta_hex, value=value, components=components, seconds=seconds, stop_reason=stop_reason,
                decision=decision)


# ---- 讀回驗證 ----

def _point(row: TemplateRow | TemplateStart) -> tuple[object, ...]:
    return (row.surfaces, row.theta_hex, row.value, row.components)


def _check_segments(rows: Sequence[TemplateRow]) -> None:
    """重調段（同階段、同輪、同候選連在一起的列）的停止原因恰在最後一列（整本帳最後一列可以缺：寫到一半斷掉）；
    同一段的配置與變數名不變。"""
    for index, row in enumerate(rows):
        if row.phase not in _SEGMENTS:
            continue
        following = rows[index + 1] if index + 1 < len(rows) else None
        same = following is not None and (following.phase, following.round, following.candidate) == (
            row.phase, row.round, row.candidate)
        if same and following is not None and (following.surfaces, following.variable_names) != (
                row.surfaces, row.variable_names):
            raise ValueError(f"第 {index + 1} 列：同一段重調的配置或變數名變了")
        if following is None and row.stop_reason is None:
            continue
        if (row.stop_reason is not None) == same:
            raise ValueError(f"第 {index} 列：停止原因只能、而且一定要在每段重調的最後一列")


def _round_blocks(rows: Sequence[TemplateRow]) -> list[list[TemplateRow]]:
    """照輪切塊；輪號從 0 起、每一輪結束（決定列）後下一列的輪號恰好加一。"""
    blocks: list[list[TemplateRow]] = []
    for index, row in enumerate(rows):
        closed = not blocks or blocks[-1][-1].phase == "template_decision"
        expected = (blocks[-1][0].round + 1 if blocks else 0) if closed else blocks[-1][0].round
        if row.round != expected:
            raise ValueError(f"第 {index} 列的輪號 {row.round}，這裡應是第 {expected} 輪（輪號不可跳、不可退，"
                             "決定列必須是那一輪的最後一列）")
        if closed:
            blocks.append([row])
        else:
            blocks[-1].append(row)
    return blocks


def _swapped_surface(start: Surfaces, row: TemplateRow, label: str, settings: MaterialGradientSettings | None) -> None:
    """快速評分列的配置＝起點配置只換候選代號那一面：不貼 ⇔ preset 是 None；給了設定還要是那個家族的樣板。"""
    match = _CANDIDATE.match(str(row.candidate))
    if match is None:
        raise ValueError(f"{label}：候選代號 {row.candidate} 不是「面代號->目標」")
    surface, target = match.group("surface"), match.group("target")
    changed = [index for index, (old, new) in enumerate(zip(start, row.surfaces, strict=True)) if old != new]
    ids = [surface_id for surface_id, _ in start]
    if [item[0] for item in row.surfaces] != ids or surface not in ids or changed != [ids.index(surface)]:
        raise ValueError(f"{label}：候選 {row.candidate} 的配置不是起點配置只換那一面")
    preset = row.surfaces[ids.index(surface)][1]
    if (preset is None) != (target == NONE_TARGET):
        raise ValueError(f"{label}：候選 {row.candidate} 的目標與換上去的 preset {preset} 對不上")
    if settings is not None and preset != preset_for_target(target, settings):
        raise ValueError(f"{label}：候選 {row.candidate} 換上去的 preset {preset} 不是家族 {target} 的樣板")


class _Round(NamedTuple):
    quick: list[TemplateRow]
    baseline: list[TemplateRow]
    candidates: list[tuple[str, list[TemplateRow]]]
    decision: TemplateRow | None


def _split_round(block: Sequence[TemplateRow], label: str) -> _Round:
    """一輪的列序：快速評分×n →（基準重調 →（候選重調）×k）→ 決定，或只有決定；最後一輪可以沒寫完。"""
    decision = block[-1] if block[-1].phase == "template_decision" else None
    body = list(block[:-1] if decision is not None else block)
    quick = []
    while body and body[0].phase == "quick_score":
        quick.append(body.pop(0))
    baseline = []
    while body and body[0].phase == "baseline_retune":
        baseline.append(body.pop(0))
    candidates: list[tuple[str, list[TemplateRow]]] = []
    while body and body[0].phase == "candidate_retune":
        name = str(body[0].candidate)
        segment = []
        while body and body[0].phase == "candidate_retune" and body[0].candidate == name:
            segment.append(body.pop(0))
        candidates.append((name, segment))
    if body:
        raise ValueError(f"{label}：列序必須是快速評分、基準重調、候選重調、決定，第 {body[0].step} 列是 {body[0].phase}")
    if ((baseline or candidates) and not quick) or (candidates and not baseline):
        raise ValueError(f"{label}：要先有快速評分才有基準重調、先有基準重調才有候選重調")
    if decision is not None and quick and not candidates:
        raise ValueError(f"{label}：快速評分過的輪要有候選重調才能決定")
    return _Round(quick, baseline, candidates, decision)


def _check_round(block: Sequence[TemplateRow], start: TemplateRow | TemplateStart | None, start_surfaces: Surfaces,
                 settings: MaterialGradientSettings | None) -> _Round:
    """一輪的內容：快速評分照「可換的目標」（給了設定）、候選是快速評分前 k 名、各段起點對得上。"""
    label = f"第 {block[0].round} 輪"
    parts = _split_round(block, label)
    names = [str(row.candidate) for row in parts.quick]
    if len(set(names)) != len(names):
        raise ValueError(f"{label}：快速評分的候選重複")
    for row in parts.quick:
        _swapped_surface(start_surfaces, row, label, settings)
    decided = parts.decision.decision if parts.decision is not None else None
    complete = (decided is not None and decided.quick_count > 0) or bool(parts.baseline)
    if settings is not None:
        wanted = [f"{surface}{CANDIDATE_SEPARATOR}{target}" for surface, target in alternatives(start_surfaces, settings)]
        if names != (wanted if complete else wanted[:len(names)]):
            raise ValueError(f"{label}：快速評分的候選與順序不是這個配置所有可換的目標")
    if parts.baseline and start is not None and _point(parts.baseline[0])[:2] != _point(start)[:2]:
        raise ValueError(f"{label}：基準重調的起點不是這一輪起點的配置與 θ")
    if parts.baseline and parts.baseline[0].surfaces != start_surfaces:
        raise ValueError(f"{label}：基準重調的配置不是這一輪起點的配置")
    order = sorted(range(len(parts.quick)), key=lambda index: (parts.quick[index].value, index))
    top = [parts.quick[index] for index in order]
    if [name for name, _ in parts.candidates] != [str(row.candidate) for row in top[:len(parts.candidates)]]:
        raise ValueError(f"{label}：進重調的候選不是快速評分照值由小到大的前幾名")
    if settings is not None and parts.decision is not None and parts.quick and len(parts.candidates) != retune_count(
            settings, len(parts.quick)):
        raise ValueError(f"{label}：進重調的候選數不是 max_per_round 與快速評分數取小")
    for (name, segment), quick in zip(parts.candidates, top, strict=False):
        if (segment[0].surfaces, segment[0].variable_names, segment[0].theta_hex) != (
                quick.surfaces, quick.variable_names, quick.theta_hex):
            raise ValueError(f"{label}：候選 {name} 的重調起點不是它快速評分那一點")
    return parts


def _check_decision(label: str, parts: _Round, start: TemplateRow | TemplateStart | None, previous_total: int,
                    is_last: bool) -> TemplateDecision:
    """決定列跟前面的列一致：快速評分數、候選結果、基準值、交出去的點、求值次數、停止原因與下一輪最壞值。"""
    row = parts.decision
    if row is None or row.decision is None:
        raise ValueError(f"{label}的決定列沒有決定")
    decision = row.decision
    if decision.quick_count != len(parts.quick):
        raise ValueError(f"{label}：決定列記的快速評分數跟快速評分列數不同")
    quick_values = {str(item.candidate): item.value for item in parts.quick}
    if [(item.candidate, item.quick_value, item.value) for item in decision.outcomes] != [
            (name, quick_values[name], segment[-1].value) for name, segment in parts.candidates]:
        raise ValueError(f"{label}：決定列記的候選、快速評分值或重調後的值跟前面的列不同")
    if parts.baseline and decision.baseline_value != parts.baseline[-1].value:
        raise ValueError(f"{label}：決定列的基準值不是基準重調最後一列的值")
    if parts.baseline:
        chosen = next((segment for name, segment in parts.candidates if name == decision.accepted), parts.baseline)
        if (*_point(row), row.variable_names) != (*_point(chosen[-1]), chosen[-1].variable_names):
            raise ValueError(f"{label}：決定列交出去的配置、θ、值或分項不是"
                             f"{'接受的候選' if decision.accepted else '基準'}重調末列的")
    elif start is not None and _point(row) != _point(start):
        raise ValueError(f"{label}：沒開的輪交出去的配置、θ、值與分項必須是起點原樣")
    spent = len(parts.quick) + sum(item[-1].evaluations or 0 for item in (
        parts.baseline, *(segment for _, segment in parts.candidates)) if item)
    if decision.round_evaluations != spent or decision.total_evaluations != previous_total + spent:
        raise ValueError(f"{label}：求值次數跟快速評分數加各段重調記的對不上（或累計沒有逐輪相加）")
    _check_stop_reason(label, row, decision)
    if row.stop_reason is not None and not is_last:
        raise ValueError(f"{label}：帶停止原因的決定列必須是整本帳的最後一列")
    return decision


def _check_stop_reason(label: str, row: TemplateRow, decision: TemplateDecision) -> None:
    """停止原因要跟決定內容對得上：沒開的輪只能是「沒有可換的樣板」（不記下一輪最壞值）或「求值到上限」（記）；
    開了沒接受只能是「未找到超過門檻」；接受了只能沒有、輪數、總時間或求值到上限，沒停與停在求值上限要記下一輪最壞值。"""
    reason, opened = row.stop_reason, decision.quick_count > 0
    allowed: dict[str | None, bool]
    if not opened:
        allowed = {TEMPLATE_STOP_NO_ALTERNATIVES: False, TEMPLATE_STOP_MAX_EVALUATIONS: True}
    elif decision.accepted is None:
        allowed = {TEMPLATE_STOP_NO_IMPROVEMENT: False}
    else:
        allowed = {None: True, TEMPLATE_STOP_MAX_ROUNDS: False, TEMPLATE_STOP_MAX_SECONDS: False,
                   TEMPLATE_STOP_MAX_EVALUATIONS: True}
    if reason not in allowed:
        raise ValueError(f"{label}：{'沒開' if not opened else '沒接受' if decision.accepted is None else '接受了候選'}"
                         f"的輪不可記停止原因「{reason}」")
    if (decision.next_round_bound is not None) != allowed[reason]:
        raise ValueError(f"{label}：停止原因「{reason}」{'要' if allowed[reason] else '不'}記下一輪最壞值")


def _check_settings(label: str, row: TemplateRow, decision: TemplateDecision, previous_total: int, seconds: float,
                    settings: MaterialGradientSettings) -> None:
    """給了設定：門檻、停止原因與下一輪最壞值都要照設定重算得出來，帳本跟設定矛盾就拒收。

    接受了候選的輪照「輪數 → 總時間（各決定列秒數加總，含這一輪）→ 下一輪最壞值放不放得進預算」的順序判停，
    該停沒記（或記了不該記的）都拒收；沒開的輪「沒有可換的樣板」要真的沒有、「求值到上限」要真的放不進。
    """
    reason, limits = row.stop_reason, settings.templates
    if decision.threshold != settings.candidates.improvement_threshold:
        raise ValueError(f"{label}：決定列記的門檻不是設定檔的改善門檻")
    count = len(alternatives(row.surfaces, settings))
    bound = next_round_bound(row.surfaces, settings)
    if decision.quick_count == 0:
        expected = TEMPLATE_STOP_NO_ALTERNATIVES if count == 0 else (
            TEMPLATE_STOP_MAX_EVALUATIONS if bound > limits.max_evaluations - previous_total else None)
    elif decision.accepted is None:
        expected = TEMPLATE_STOP_NO_IMPROVEMENT
    elif row.round + 1 >= limits.max_rounds:
        expected = TEMPLATE_STOP_MAX_ROUNDS
    elif seconds > limits.max_seconds:
        expected = TEMPLATE_STOP_MAX_SECONDS
    else:
        expected = TEMPLATE_STOP_MAX_EVALUATIONS if bound > limits.max_evaluations - decision.total_evaluations else None
    if reason != expected:
        raise ValueError(f"{label}：照設定（輪數上限 {limits.max_rounds}、帳本秒數加總 {seconds} 對上限 {limits.max_seconds}、"
                         f"求值上限 {limits.max_evaluations}）這一輪該記「{expected}」，帳本記的是「{reason}」；帳本與設定矛盾")
    if decision.next_round_bound is not None and decision.next_round_bound != bound:
        raise ValueError(f"{label}：記的下一輪最壞求值次數 {decision.next_round_bound} 跟照設定重算的 {bound} 不同")


def check_template_rows(header: TemplateHeader, rows: Sequence[TemplateRow], *,
                        settings: MaterialGradientSettings | None = None, start: TemplateStart | None = None) -> None:
    """全帳本連號、面序跟標頭同、每段重調的停止原因恰在最後一列、一輪一輪的列序與決定一致、累計求值次數逐輪相加。
    最後一輪可以沒寫完（接續時丟掉那半輪）。

    給了 ``settings``：快速評分要恰是起點配置所有可換的目標（照順序）、換上去的 preset 是那個家族的樣板、進重調的數目、
    門檻、停止原因與下一輪最壞值都要跟設定一致。給了 ``start``（庚二收尾那一點）：配置要等於標頭的起點配置，
    第 0 輪的基準起點（或沒開時交出去的點）要等於它。
    """
    ids = [surface_id for surface_id, _ in header.initial_surfaces]
    if start is not None and start.surfaces != header.initial_surfaces:
        raise ValueError("起點配置跟標頭記的不同")
    for index, row in enumerate(rows):
        if row.step != index:
            raise ValueError(f"第 {index} 列的步序是 {row.step}，帳本列必須從 0 連號")
        if [surface_id for surface_id, _ in row.surfaces] != ids:
            raise ValueError(f"第 {index} 列的面序跟標頭的起點配置不同")
    _check_segments(rows)
    previous: TemplateRow | TemplateStart | None = start
    surfaces, total, seconds = header.initial_surfaces, 0, 0.0
    for block in _round_blocks(rows):
        label = f"第 {block[0].round} 輪"
        parts = _check_round(block, previous, surfaces, settings)
        if parts.decision is None:
            continue
        decision = _check_decision(label, parts, previous, total, parts.decision is rows[-1])
        seconds += parts.decision.seconds
        if settings is not None:
            _check_settings(label, parts.decision, decision, total, seconds, settings)
        previous, surfaces, total = parts.decision, parts.decision.surfaces, decision.total_evaluations


def decisions_seconds(rows: Sequence[TemplateRow]) -> float:
    """帳本記的總時間：各決定列的整輪秒數加總（照列序加）。"""
    total = 0.0
    for row in rows:
        if row.decision is not None:
            total += row.seconds
    return total


# ---- 寫與讀 ----

def _write_line(handle: BinaryIO, value: BaseModel) -> None:
    handle.write((json.dumps(value.model_dump(mode="json"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())


def create_template_ledger(path: Path, header: TemplateHeader) -> None:
    """獨佔建檔（已有檔案即丟錯），只寫標頭。"""
    checked = TemplateHeader.model_validate(header.model_dump())
    with path.open("xb") as handle:
        _write_line(handle, checked)


def append_template_rows(path: Path, header: TemplateHeader, rows: Sequence[TemplateRow], *,
                         settings: MaterialGradientSettings | None = None, start: TemplateStart | None = None) -> None:
    """接在既有帳本後面寫：先讀回整本（標頭要等於 ``header``），連同新列一起驗，再逐列寫、每列 flush 與 fsync。"""
    _, existing = read_template_ledger(path, expected_header=header, settings=settings, start=start)
    checked = tuple(TemplateRow.model_validate(row.model_dump()) for row in rows)
    check_template_rows(header, (*existing, *checked), settings=settings, start=start)
    with path.open("ab") as handle:
        for row in checked:
            _write_line(handle, row)


def truncate_template_rows(path: Path, header: TemplateHeader, keep: int) -> None:
    """只留標頭與前 ``keep`` 列（接續時丟掉沒寫完的半輪）；截完整本再驗一次。"""
    _, rows = read_template_ledger(path, expected_header=header)
    if not 0 <= keep <= len(rows):
        raise ValueError(f"要留的列數 {keep} 不在 0～{len(rows)}")
    with path.open("rb") as handle:
        lines = handle.readlines()
    size = sum(len(line) for line in lines[:keep + 1])
    with path.open("r+b") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())
    read_template_ledger(path, expected_header=header)


def read_template_ledger(path: Path, *, expected_header: TemplateHeader | None = None,
                         settings: MaterialGradientSettings | None = None, start: TemplateStart | None = None,
                         ) -> tuple[TemplateHeader, tuple[TemplateRow, ...]]:
    """逐列驗型別與順序；給了預期標頭就逐格比（替身、設定、評分常數、庚二帳本、起點配置、原牆換了都拒收）。
    壞列一律報錯，不捨棄；版本不是這一版就拒收、訊息講明版本。``settings``、``start`` 同 :func:`check_template_rows`。"""
    with path.open("rb") as handle:
        lines = handle.readlines()
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("換樣板帳本是空的或有一列沒寫完")
    document = json.loads(lines[0])
    version = document.get("ledger_version") if isinstance(document, dict) else None
    if version != TEMPLATE_LEDGER_VERSION:
        raise ValueError(f"換樣板帳本版本是 {version!r}，這一版只收 {TEMPLATE_LEDGER_VERSION}")
    header = TemplateHeader.model_validate_json(lines[0])
    if expected_header is not None and header != expected_header:
        actual, expected = header.model_dump(), expected_header.model_dump()
        differing = sorted(name for name in actual if actual[name] != expected[name])
        raise ValueError(f"換樣板帳本標頭跟這一次對不上：{differing}")
    rows = tuple(TemplateRow.model_validate_json(line) for line in lines[1:])
    check_template_rows(header, rows, settings=settings, start=start)
    return header, rows


def template_progress(rows: Sequence[TemplateRow]) -> tuple[int, int, str | None]:
    """帳本記的累計求值次數、做完幾輪、收尾的停止原因（沒收尾是 None）。"""
    decisions = [row.decision for row in rows if row.decision is not None]
    stop = rows[-1].stop_reason if rows and rows[-1].phase == "template_decision" else None
    return (decisions[-1].total_evaluations if decisions else 0), len(decisions), stop
