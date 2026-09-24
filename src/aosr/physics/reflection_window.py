"""鞋盒反射評估的時間窗補算，僅供診斷，絕不回填主報表總量。

停止依據是鞋盒的幾何性質：每一軸最近鏡像距離隨該軸撞牆次數不減；任一軸少撞
一次會得到更低階且不更遠的鏡像。因此第 K′+1 階最早路徑已在窗外，所有更高階
也在窗外。非長方形房（票 #465）落地時必須重新證明此性質。

「相對直達」一律是 ``path.delay_s - direct.delay_s``，兩者都取自同一次
``room_paths.image_source_paths`` 的結果；直達是 ``order == 0`` 的那一條。

補到支援上限（``SUPPORTED_MAX_ORDER``）時第 K′+1 階算不了；同一條性質說更高階不會比第 K′ 階
更早到，所以拿第 K′ 階最早那一條當「沒算的路徑最早可能多早到」的下界：它已在窗外就照樣證明完整
（主報表一開始就是上限那一階時會碰到），不在窗外才是證明不了。
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aosr.geometry.shoebox import Point
from aosr.physics.report_io import PathRow, ReportInput, scene_fingerprint, solver_inputs
from aosr.physics.report_path_table import build_path_table
from aosr.physics.room_paths import (
    NUMERICALLY_GUARDED_ORDER_K,
    SUPPORTED_MAX_ORDER,
    SUPPORTED_MIN_ORDER,
    image_source_paths,
)


class ReflectionWindow(BaseModel):
    """與報表場景綁定的額外反射路徑及其覆蓋、數值驗證狀態。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    scene_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_m: Point
    receiver_m: Point
    report_order_k: int = Field(ge=SUPPORTED_MIN_ORDER, le=SUPPORTED_MAX_ORDER)
    window_s: float = Field(gt=0.0)
    direct_delay_s: float = Field(ge=0.0)
    computed_order_k: int = Field(ge=SUPPORTED_MIN_ORDER, le=SUPPORTED_MAX_ORDER)
    coverage: Literal["complete", "not_provable"]
    validation: Literal["validated", "unvalidated"]
    # 沒算的路徑最早可能多早到（相對直達）：一般是第 computed+1 階最早那一條；補到上限時是
    # 第 computed 階最早那一條當下界（只有主報表一開始就在上限才會有值，見模組說明）。
    next_uncomputed_earliest_relative_s: float | None = Field(ge=0.0)
    frequencies_hz: tuple[float, ...] = Field(min_length=1)
    scattering_coefficient: tuple[float, ...]
    rows: tuple[PathRow, ...]

    @field_validator("source_m", "receiver_m")
    @classmethod
    def _point_is_finite(cls, value: Point) -> Point:
        """直接傳進來的 ``Point`` 物件 pydantic 不會再逐格檢查（只有從字典建才會），所以這裡自己擋非有限座標。"""
        if not all(math.isfinite(coordinate) for coordinate in value.as_tuple()):
            raise ValueError("座標必須是有限數")
        return value

    @model_validator(mode="after")
    def _consistent_window(self) -> Self:
        """狀態、階數、截窗列與逐頻軸須互相一致。"""
        if self.computed_order_k < self.report_order_k:
            raise ValueError("computed_order_k 不可小於 report_order_k")
        next_delay = self.next_uncomputed_earliest_relative_s
        if next_delay is None and self.computed_order_k != SUPPORTED_MAX_ORDER:
            raise ValueError("next_uncomputed_earliest_relative_s 只有補到支援上限才可以沒有")
        if (
            next_delay is not None
            and self.computed_order_k == SUPPORTED_MAX_ORDER
            and self.report_order_k != SUPPORTED_MAX_ORDER
        ):
            # 從較低的 K 一路補到上限，表示上限那一階最早那一條已在窗內，下界不可能在窗外
            raise ValueError("補到支援上限的下界只給主報表一開始就在上限的情況")
        complete = next_delay is not None and next_delay > self.window_s
        if (self.coverage == "complete") != complete:
            raise ValueError("coverage 與未算路徑的窗外證明不符")
        if self.coverage == "not_provable" and (
            self.computed_order_k != SUPPORTED_MAX_ORDER or next_delay is not None
        ):
            raise ValueError("not_provable 必須補到支援上限、而且沒有下一階的證明")
        guarded = self.computed_order_k <= NUMERICALLY_GUARDED_ORDER_K
        if (self.validation == "validated") != guarded:
            raise ValueError("validation 與補算階數的數值驗證範圍不符")
        if self.frequencies_hz[0] <= 0.0 or any(
            left >= right
            for left, right in zip(self.frequencies_hz, self.frequencies_hz[1:])
        ):
            raise ValueError("frequencies_hz 必須是遞增正頻率")
        if len(self.scattering_coefficient) != len(self.frequencies_hz) or any(
            not 0.0 <= value <= 1.0 for value in self.scattering_coefficient
        ):
            raise ValueError("scattering_coefficient 與頻率軸或合法範圍不符")
        for row in self.rows:
            if not self.report_order_k < row.order <= self.computed_order_k:
                raise ValueError("rows 的階數不在補算範圍")
            if row.delay_s - self.direct_delay_s > self.window_s:
                raise ValueError("rows 有路徑超過時間窗")
            if len(row.relative_direct_energy) != len(self.frequencies_hz):
                raise ValueError("rows 的逐頻長度與 frequencies_hz 不同")
        return self


def _coverage_from_geometry(
    inputs: ReportInput, window_s: float,
) -> tuple[int, float, float | None]:
    """逐階檢查下一階最早相對直達延遲；只建幾何路徑。"""
    computed = inputs.reflection_order_k
    while True:
        next_order = computed + 1
        paths = image_source_paths(
            inputs.room_m, inputs.source_m, inputs.receiver_m,
            inputs.sound_speed_m_s,
            max_order=min(next_order, SUPPORTED_MAX_ORDER), materials=None,
        )
        direct_delay = next(path.delay_s for path in paths if path.order == 0)
        if next_order > SUPPORTED_MAX_ORDER:
            # 第 K′+1 階算不了：拿第 K′ 階最早那一條當下界（見模組說明）
            bound = min(path.delay_s for path in paths if path.order == computed) - direct_delay
            return computed, direct_delay, bound if bound > window_s else None
        first = min(path.delay_s for path in paths if path.order == next_order)
        relative = first - direct_delay
        if relative > window_s:
            return computed, direct_delay, relative
        computed = next_order


def build_reflection_window(
    inputs: ReportInput, *, frequencies_hz: tuple[float, ...],
    scattering_coefficient: tuple[float, ...], window_s: float,
) -> ReflectionWindow:
    """補算至未算路徑全在窗外；逐頻能量直接沿用路徑表同一支程式。"""
    computed, direct_delay, next_delay = _coverage_from_geometry(inputs, window_s)
    solved = solver_inputs(inputs)
    rows: tuple[PathRow, ...] = ()
    if computed > inputs.reflection_order_k:
        table = build_path_table(
            room=solved.room, source=solved.source, receiver=solved.receiver,
            sound_speed_m_s=solved.sound_speed_m_s,
            rho_c_pa_s_per_m=solved.density_kg_m3 * solved.sound_speed_m_s,
            impedance_by_wall=solved.impedance_by_wall,
            frequencies_hz=frequencies_hz,
            scattering_coefficient=scattering_coefficient,
            reflection_order_k=computed,
        )
        rows = tuple(
            PathRow.model_validate(asdict(row)) for row in table.rows
            if inputs.reflection_order_k < row.order <= computed
            and row.delay_s - direct_delay <= window_s
        )
    return ReflectionWindow(
        scene_fingerprint=scene_fingerprint(inputs),
        source_m=inputs.source_m, receiver_m=inputs.receiver_m,
        report_order_k=inputs.reflection_order_k, window_s=window_s,
        direct_delay_s=direct_delay, computed_order_k=computed,
        coverage="complete" if next_delay is not None else "not_provable",
        validation=("validated" if computed <= NUMERICALLY_GUARDED_ORDER_K else "unvalidated"),
        next_uncomputed_earliest_relative_s=next_delay,
        frequencies_hz=frequencies_hz,
        scattering_coefficient=scattering_coefficient, rows=rows,
    )
