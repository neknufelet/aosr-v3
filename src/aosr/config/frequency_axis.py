"""共用頻率軸與 v3 有限元素路軸設定。

純 Python 產生器住最底下的 ``config``（設定）層，讓 ``materials``（材料）、
``geometry``（幾何）與 ``physics``（物理）都能依分層規矩向下共用一份算法。

``V3_AXIS_START_HZ`` 與 ``V3_AXIS_POINTS_PER_OCTAVE`` 是三路共用細軸的
20 Hz 起點與每八度 24 份解析度。``FEM_GEOMETRIC_CROSSOVER_CAP_HZ`` 同時是
有限元素算到這裡（含）、交接上端與硬切點的 300 Hz。幾何路沿同一條公式軸
接到 4000 Hz 八度帶上緣；頻帶鏡像法另用帶內每 0.5 Hz 整數倍密軸。上緣、
密軸與六個報表中心頻率出自
``docs/decisions/stage-nine-reflection-order-is-a-setting.md`` 第 6 條。這些決策錨定
``docs/decisions/stage-nine-reflection-order-is-a-setting.md`` 與
``docs/decisions/compute-strategy-three-stages-three-lanes-fem-300hz.md``：
20 Hz 起、每八度 24 份；有限元素只收不高於 300 Hz 的公式格點。
"""

from __future__ import annotations

import math


def frequency_axis(
    mode: str,
    f_min_hz: float,
    f_max_hz: float,
    *,
    per_octave: int | None = None,
    step_hz: float | None = None,
) -> tuple[float, ...]:
    """產生含起點、且不超過上限的分數八度或線性頻率軸。"""
    if mode not in ("octave_fraction", "linear"):
        raise ValueError(f"mode 不支援：{mode!r}")
    if not math.isfinite(f_min_hz) or f_min_hz <= 0.0:
        raise ValueError("f_min_hz 必須是大於零的有限值")
    if not math.isfinite(f_max_hz) or f_max_hz <= f_min_hz:
        raise ValueError("f_max_hz 必須是大於 f_min_hz 的有限值")

    if mode == "octave_fraction":
        if (
            isinstance(per_octave, bool)
            or not isinstance(per_octave, int)
            or per_octave <= 0
        ):
            raise ValueError("per_octave 必須是正整數")
        if step_hz is not None:
            raise ValueError("step_hz 只適用於 linear 模式")
        values: list[float] = []
        index = 0
        while (value := f_min_hz * 2 ** (index / per_octave)) <= f_max_hz:
            values.append(value)
            index += 1
        return tuple(values)

    if isinstance(step_hz, bool) or not isinstance(step_hz, int | float):
        raise ValueError("step_hz 必須是正有限值")
    if not math.isfinite(step_hz) or step_hz <= 0.0:
        raise ValueError("step_hz 必須是正有限值")
    if per_octave is not None:
        raise ValueError("per_octave 只適用於 octave_fraction 模式")
    values = []
    index = 0
    while (value := f_min_hz + index * step_hz) <= f_max_hz:
        values.append(value)
        index += 1
    return tuple(values)


V3_AXIS_START_HZ: float = 20.0
V3_AXIS_POINTS_PER_OCTAVE: int = 24
FEM_GEOMETRIC_CROSSOVER_CAP_HZ: float = 300.0
GEOMETRIC_AXIS_UPPER_HZ: float = 4000.0 * math.sqrt(2.0)
GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ: tuple[float, ...] = (
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
)
GEOMETRIC_BAND_FREQUENCY_STEP_HZ: float = 0.5


def geometric_band_frequencies(center_frequency_hz: float) -> tuple[float, ...]:
    """回傳八度帶半開區間內所有具名間距的整數倍頻點。"""
    center = float(center_frequency_hz)
    if not math.isfinite(center) or center <= 0.0:
        raise ValueError("center_frequency_hz 必須是有限正數")
    lower = center / math.sqrt(2.0)
    upper = center * math.sqrt(2.0)
    first_multiple = math.ceil(lower / GEOMETRIC_BAND_FREQUENCY_STEP_HZ)
    stop_multiple = math.ceil(upper / GEOMETRIC_BAND_FREQUENCY_STEP_HZ)
    return tuple(
        multiple * GEOMETRIC_BAND_FREQUENCY_STEP_HZ
        for multiple in range(first_multiple, stop_multiple)
    )


GEOMETRIC_BAND_FREQUENCIES_HZ: tuple[float, ...] = tuple(
    frequency
    for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
    for frequency in geometric_band_frequencies(center)
)
FEM_LANE_FREQUENCIES_HZ: tuple[float, ...] = frequency_axis(
    "octave_fraction",
    V3_AXIS_START_HZ,
    FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
    per_octave=V3_AXIS_POINTS_PER_OCTAVE,
)
GEOMETRIC_LANE_FREQUENCIES_HZ: tuple[float, ...] = frequency_axis(
    "octave_fraction",
    V3_AXIS_START_HZ,
    GEOMETRIC_AXIS_UPPER_HZ,
    per_octave=V3_AXIS_POINTS_PER_OCTAVE,
)
