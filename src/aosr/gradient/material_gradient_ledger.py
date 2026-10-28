"""牆面材料梯度搜尋的步驟帳本（#561）：JSONL，第一列標頭、之後每列一步。

手法照 ``src/aosr/search/ledger.py``（θ 存 float.hex 保存每一位、每列寫完 flush 與 fsync、讀回逐列驗型別），
但列型別是自己的，不套用擺位搜尋帳本的 V1／V2。標頭固定替身身分（輸入雜湊、網格雜湊、物理身分）、
每面的 preset、變數名順序與設定指紋；讀回時順序、θ 長度、停止原因的位置都要對，
給了預期標頭就逐格比，對不上一律拒收。
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aosr.config.material_gradient import MaterialGradientSettings
from aosr.physics.fem_reduced import ReducedModel
from aosr.gradient.material_gradient import STOP_REASONS, DescentResult, SnapResult, SurfaceDesign, design_names


LEDGER_VERSION: Final = "aosr.material_gradient_ledger.v1"
_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
Phase = Literal["descent", "snap", "retune"]
_PHASE_ORDER: Final[dict[str, int]] = {"descent": 0, "snap": 1, "retune": 2}


class MaterialGradientHeader(BaseModel):
    """帳本第一列：替身身分、每面（面代號, preset id 或 None）、θ 變數名順序、設定指紋。"""

    model_config = _FROZEN
    ledger_version: Literal["aosr.material_gradient_ledger.v1"]
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


class MaterialGradientRow(BaseModel):
    """一步：全帳本第幾列（從 0 起連號）、階段、θ（float.hex）、值與兩分項、梯度範數、步長、秒數、停止原因。

    ``snap`` 列沒有梯度範數與步長、要帶每面取法；``descent``／``retune`` 列沒有取法。
    停止原因與最後一次試的步長（``final_trial_step``，線搜尋失敗時就是那個低於下限的試探）只在
    每一段下降（``descent``、``retune``）的最後一列、兩格同進同出，由讀回驗證核位置。
    """

    model_config = _FROZEN
    step: int = Field(ge=0, strict=True)
    phase: Phase
    theta_hex: tuple[str, ...]
    value: float
    components: tuple[float, float]
    gradient_norm: float | None = Field(ge=0.0)
    step_size: float | None = Field(ge=0.0)
    seconds: float = Field(ge=0.0)
    stop_reason: str | None
    final_trial_step: float | None = Field(ge=0.0)
    snap: tuple[SnapChoice, ...] | None

    @field_validator("theta_hex")
    @classmethod
    def _canonical_hex(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for encoded in value:
            try:
                number = float.fromhex(encoded)
            except (OverflowError, ValueError) as error:
                raise ValueError("θ 必須是有限的 float.hex") from error
            if not math.isfinite(number) or number.hex() != encoded:
                raise ValueError("θ 必須是正規化的有限 float.hex")
        return value

    @model_validator(mode="after")
    def _phase_fields(self) -> Self:
        if self.stop_reason is not None and self.stop_reason not in STOP_REASONS:
            raise ValueError(f"停止原因只認 {list(STOP_REASONS)}")
        if (self.stop_reason is None) != (self.final_trial_step is None):
            raise ValueError("停止原因與最後一次試的步長要同在一列")
        if self.phase == "snap":
            if (self.gradient_norm is not None or self.step_size is not None or self.snap is None
                    or self.stop_reason is not None):
                raise ValueError("取規格列只帶取法，不帶梯度範數、步長與停止原因")
        elif self.gradient_norm is None or self.step_size is None or self.snap is not None:
            raise ValueError("下降列要帶梯度範數與步長，不帶取法")
        return self

    def theta(self) -> tuple[float, ...]:
        return tuple(float.fromhex(encoded) for encoded in self.theta_hex)


def header_for(model: ReducedModel, designs: Sequence[SurfaceDesign],
               settings: MaterialGradientSettings) -> MaterialGradientHeader:
    """這一次搜尋的標頭：全由替身、設計清單與設定算出來。"""
    return MaterialGradientHeader(
        ledger_version=LEDGER_VERSION, input_hash=model.input_hash, mesh_hash=model.mesh_hash,
        physics_identity=model.physics_identity,
        surfaces=tuple((design.surface_id, design.preset_id) for design in designs),
        variable_names=design_names(designs), settings_fingerprint=settings.fingerprint)


def descent_rows(result: DescentResult, phase: Literal["descent", "retune"], first_step: int) -> tuple[MaterialGradientRow, ...]:
    """一段下降的每一步換成帳本列；停止原因只放在這一段的最後一列。"""
    last = len(result.steps) - 1
    return tuple(MaterialGradientRow(
        step=first_step + offset, phase=phase, theta_hex=tuple(value.hex() for value in item.theta),
        value=item.value, components=item.components, gradient_norm=item.gradient_norm,
        step_size=item.step_size, seconds=item.seconds,
        stop_reason=result.stop_reason if offset == last else None,
        final_trial_step=result.final_trial_step if offset == last else None, snap=None,
    ) for offset, item in enumerate(result.steps))


def snap_row(snapped: SnapResult, value: float, components: tuple[float, float], seconds: float,
             step: int) -> MaterialGradientRow:
    """取規格一列：取到的 θ、在那裡重算的值與分項、每面取法。"""
    return MaterialGradientRow(
        step=step, phase="snap", theta_hex=tuple(float(value).hex() for value in snapped.theta),
        value=value, components=components, gradient_norm=None, step_size=None, seconds=seconds,
        stop_reason=None, final_trial_step=None, snap=tuple(SnapChoice(**choice._asdict()) for choice in snapped.surfaces))


def check_rows(header: MaterialGradientHeader, rows: Sequence[MaterialGradientRow]) -> None:
    """全帳本連號、階段不倒退、θ 長度等於變數數、停止原因恰在每段下降的最後一列、取法面序等於標頭。"""
    surface_ids = tuple(surface_id for surface_id, _ in header.surfaces)
    for index, row in enumerate(rows):
        if row.step != index:
            raise ValueError(f"第 {index} 列的步序是 {row.step}，帳本列必須從 0 連號")
        if len(row.theta_hex) != len(header.variable_names):
            raise ValueError(f"第 {index} 列的 θ 長度 {len(row.theta_hex)} 不等於變數數 {len(header.variable_names)}")
        if index and _PHASE_ORDER[row.phase] < _PHASE_ORDER[rows[index - 1].phase]:
            raise ValueError(f"第 {index} 列的階段 {row.phase} 倒退，順序必須是 descent、snap、retune")
        if row.snap is not None and tuple(choice.surface_id for choice in row.snap) != surface_ids:
            raise ValueError(f"第 {index} 列的取法面序跟標頭不同")
        segment_ends = index == len(rows) - 1 or rows[index + 1].phase != row.phase
        if row.phase != "snap" and (row.stop_reason is not None) != segment_ends:
            raise ValueError(f"第 {index} 列：停止原因只能、而且一定要在每段下降的最後一列")


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


def read_ledger(path: Path, *, expected_header: MaterialGradientHeader | None = None,
                ) -> tuple[MaterialGradientHeader, tuple[MaterialGradientRow, ...]]:
    """逐列驗型別與順序；給了預期標頭就逐格比（替身換了、設定換了都拒收）。壞列一律報錯，不捨棄。"""
    with path.open("rb") as handle:
        lines = handle.readlines()
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("帳本是空的或有一列沒寫完")
    header = MaterialGradientHeader.model_validate_json(lines[0])
    if expected_header is not None and header != expected_header:
        actual, expected = header.model_dump(), expected_header.model_dump()
        differing = sorted(name for name in actual if actual[name] != expected[name])
        raise ValueError(f"帳本標頭跟這一次對不上：{differing}")
    rows = tuple(MaterialGradientRow.model_validate_json(line) for line in lines[1:])
    check_rows(header, rows)
    return header, rows
