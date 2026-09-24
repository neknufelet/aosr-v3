"""反射評估器的物理資料：平行牆對留存能量與下一階最早幾何到達。

牆對留存是**垂直入射、只算鏡面、逐面各扣自己的散射**（票 #351 第 4 格「兩面的鏡面留存能量相乘、
散射掉的扣掉」）；路徑表與晚期混響用的是整房合成的散射係數，兩邊的「來回剩多少」刻意不同，
不能拿來互相驗證。聲源或接收點貼在牆面、牆邊或角落時，下一階幾何那一段會跟路徑表一樣算不出來而報錯。
"""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aosr.geometry.shoebox import Point
from aosr.materials.response import MATERIAL_SCATTERING_DEFAULT_S
from aosr.physics.amplitude import CANONICAL_WALLS, reflection_coefficient
from aosr.physics.report_io import ReportInput, scene_fingerprint
from aosr.physics.room_paths import (
    SUPPORTED_MAX_ORDER,
    SUPPORTED_MIN_ORDER,
    image_source_paths,
)


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
PairAxis = Literal["x", "y", "z"]
_FACES: dict[PairAxis, tuple[str, str]] = {
    "x": ("x0", "xL"),
    "y": ("y0", "yL"),
    "z": ("floor", "ceiling"),
}


class WallPairRow(BaseModel):
    """一對平行牆的一次來回；能量是逐頻線性比例。"""

    model_config = FROZEN

    pair: PairAxis
    faces: tuple[str, str]
    distance_m: float = Field(gt=0.0)
    round_trip_delay_s: float = Field(gt=0.0)
    face_retained_energy: tuple[tuple[float, ...], tuple[float, ...]]
    round_trip_retained_energy: tuple[float, ...]

    @model_validator(mode="after")
    def _faces_match_pair(self) -> Self:
        """牆名限規範清單，並固定一軸兩面牆的順序。"""
        if any(face not in CANONICAL_WALLS for face in self.faces):
            raise ValueError("faces 有非規範牆名")
        if self.faces != _FACES[self.pair]:
            raise ValueError(f"faces 跟 pair={self.pair} 不相符")
        return self


class ReflectionScreen(BaseModel):
    """與報表場景指紋綁定的反射篩查物理資料。"""

    model_config = FROZEN

    scene_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_m: Point
    receiver_m: Point
    reflection_order_k: int = Field(ge=SUPPORTED_MIN_ORDER, le=SUPPORTED_MAX_ORDER)
    frequencies_hz: tuple[float, ...] = Field(min_length=1)
    next_order_earliest_delay_s: float | None = Field(default=None, ge=0.0)
    pairs: tuple[WallPairRow, ...]

    @field_validator("source_m", "receiver_m")
    @classmethod
    def _point_is_finite(cls, value: Point) -> Point:
        """Point 是普通 dataclass，逐軸擋住非有限座標。"""
        if not all(math.isfinite(coordinate) for coordinate in value.as_tuple()):
            raise ValueError("座標必須是有限數")
        return value

    @model_validator(mode="after")
    def _bands_and_pairs_match(self) -> Self:
        """逐頻列等長，頻率遞增，三軸各只有一對。"""
        if any(
            left <= 0.0 or left >= right
            for left, right in zip(self.frequencies_hz, self.frequencies_hz[1:])
        ) or self.frequencies_hz[0] <= 0.0:
            raise ValueError("frequencies_hz 必須是遞增的正頻率")
        if {row.pair for row in self.pairs} != set(_FACES) or len(self.pairs) != len(_FACES):
            raise ValueError("pairs 必須是 x、y、z 各一對，不可重複")
        for row in self.pairs:
            bands = (*row.face_retained_energy, row.round_trip_retained_energy)
            if any(len(values) != len(self.frequencies_hz) for values in bands):
                raise ValueError(f"{row.pair} 的 face_retained_energy 或 "
                                 "round_trip_retained_energy 與 frequencies_hz 長度不同")
        return self


def _wall_pair(
    inputs: ReportInput, frequencies_hz: tuple[float, ...], pair: PairAxis
) -> WallPairRow:
    """垂直入射反射能量，逐牆扣除散射後再相乘。"""
    rho_c = inputs.density_kg_m3 * inputs.sound_speed_m_s
    scattering = inputs.scattering_by_wall or {}
    faces = _FACES[pair]
    retained = tuple(
        abs(reflection_coefficient(
            complex(inputs.impedance_pa_s_per_m_by_wall[face]), 1.0, rho_c
        )) ** 2 * (1.0 - scattering.get(face, MATERIAL_SCATTERING_DEFAULT_S))
        for face in faces
    )
    distance = getattr(inputs.room_m, f"L{pair}")
    return WallPairRow(
        pair=pair,
        faces=faces,
        distance_m=distance,
        round_trip_delay_s=2.0 * distance / inputs.sound_speed_m_s,
        face_retained_energy=(
            tuple(retained[0] for _frequency in frequencies_hz),
            tuple(retained[1] for _frequency in frequencies_hz),
        ),
        round_trip_retained_energy=tuple(
            retained[0] * retained[1] for _frequency in frequencies_hz
        ),
    )


def _next_order_delay(inputs: ReportInput) -> float | None:
    """只問幾何路徑的 K+1 階；超過實測上限則無資料。"""
    next_order = inputs.reflection_order_k + 1
    if next_order > SUPPORTED_MAX_ORDER:
        return None
    paths = image_source_paths(
        inputs.room_m, inputs.source_m, inputs.receiver_m,
        inputs.sound_speed_m_s, max_order=next_order, materials=None,
    )
    return min(
        (path.delay_s for path in paths if path.order == next_order), default=None
    )


def build_reflection_screen(
    inputs: ReportInput, frequencies_hz: tuple[float, ...]
) -> ReflectionScreen:
    """由同一份報表輸入產生牆對資料及下一階幾何到達。"""
    return ReflectionScreen(
        scene_fingerprint=scene_fingerprint(inputs),
        source_m=inputs.source_m,
        receiver_m=inputs.receiver_m,
        reflection_order_k=inputs.reflection_order_k,
        frequencies_hz=frequencies_hz,
        next_order_earliest_delay_s=_next_order_delay(inputs),
        pairs=tuple(_wall_pair(inputs, frequencies_hz, axis) for axis in _FACES),
    )
