"""音色平衡第一層評估器（票 #345）：只量、不打分。

吃一份「候選 × 喇叭 × 接收點」的細軸總能量曲線，照票上拍板的五格量法回三件事：
整體傾斜、扣掉傾斜擬合線之後的局部起伏（均方根與峰谷完整清單）、對指定目標曲線的偏差。
輸出照 :mod:`aosr.scoring.contract` 的凍結契約回一條 ``CategoryEvaluation``（類別評估），
狀態只會是 measured（已量未算代價）或 unavailable（不可估），不回總分、不算代價。

三個分支（傾斜、起伏、對目標偏差）各自從原始能量另起，不共用平滑過的中間結果；
入口只在本模組內除掉共同音量（相對量），不回寫輸入。所有範圍、平滑寬度、最小寬度、
最少點數與預設目標都從品質登記簿讀，這裡不寫任何 dB 或八度數字。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.quality_targets import (
    QualityPurpose,
    SettingEntry,
    TargetEntry,
    Unit,
    load_quality_targets,
)
from aosr.physics.report_io import ReportOutput
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CategoryEvaluation,
    EvaluationState,
    Feature,
    Flag,
    FrequencyRange,
    InputProvenance,
    ModelValidationFrequencyRange,
    ModelValidationStatus,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbrePayload,
)
from aosr.scoring.placement import Placement, point_placement


TIMBRE_EVALUATOR_VERSION: Final[str] = "aosr.scoring.timbre.v6"
_PREFIX: Final[str] = "timbre_balance."
_SETTING_UNITS: Final[dict[str, Unit]] = {
    "coverage_range_hz": "Hz",
    "tilt_fit_range_hz": "Hz",
    "ripple_range_hz": "Hz",
    "smoothing_width_octave_tilt": "oct",
    "smoothing_width_octave_ripple": "oct",
    "feature_min_width_octave": "oct",
    "min_points": "1",
}
_TARGET_TILT_KEY: Final[str] = _PREFIX + "target_tilt_db_per_octave"
# 這兩格准填 0：起伏不平滑、窄峰不刪（票 #432）。
_MAY_BE_ZERO: Final[frozenset[str]] = frozenset(
    {"smoothing_width_octave_ripple", "feature_min_width_octave"}
)
_TARGET_TILT_UNIT: Final[Unit] = "dB/oct"
# 分數八度視窗邊界的捨入護欄：1/24 八度細軸上視窗邊緣剛好落在格點，對數頻率的捨入
# 會讓同一個距離一側算進、一側算不進（視窗歪一格，平滑值就跳）。護欄只吸收機器捨入
# （epsilon，機器精度），不是品質門檻。
_WINDOW_ROUNDING_OCT: Final[float] = 64.0 * float(np.finfo(float).eps)

FloatArray = NDArray[np.float64]
_DependencyRanges = tuple[FrequencyRange, FrequencyRange]


class TimbreInput(BaseModel):
    """一份候選 × 喇叭 × 接收點的細軸總能量；原始資料保留共同基準，不在入口扣平均。

    ``total_energy`` 准收非有限值與非正值——那是評估器要回「不可估」的情形，
    不在建構時就把證據擋掉；頻率軸、聲源與接收點座標則必須有限。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=True)

    candidate_id: str = Field(min_length=1)
    scene_fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    speaker_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)
    source_position_m: tuple[float, float, float]
    receiver_position_m: tuple[float, float, float]
    frequencies_hz: tuple[float, ...] = Field(min_length=2)
    total_energy: tuple[float, ...]
    source_reference: str = Field(min_length=1)
    report_flags: tuple[Flag, ...]
    model_validation_status: ModelValidationStatus
    model_validation_frequency_range_hz: ModelValidationFrequencyRange
    provenance: InputProvenance

    @model_validator(mode="after")
    def _capability_range_matches_status(self) -> Self:
        unchecked = self.model_validation_status is ModelValidationStatus.UNCHECKED
        if unchecked != (not self.model_validation_frequency_range_hz):
            raise ValueError("能力範圍是空的若且唯若狀態是 unchecked")
        return self

    @model_validator(mode="after")
    def _axis_is_finite_and_ascending(self) -> Self:
        """頻率軸要有限、為正、嚴格遞增，而且與能量等長；座標要有限。"""
        if len(self.total_energy) != len(self.frequencies_hz):
            raise ValueError("total_energy 必須與 frequencies_hz 等長")
        for name, position in (
            ("source_position_m", self.source_position_m),
            ("receiver_position_m", self.receiver_position_m),
        ):
            if not all(math.isfinite(value) for value in position):
                raise ValueError(f"{name} 必須是有限值")
        axis = self.frequencies_hz
        if not all(math.isfinite(value) and value > 0.0 for value in axis):
            raise ValueError("frequencies_hz 必須是有限的正頻率")
        if any(upper <= lower for lower, upper in zip(axis, axis[1:], strict=False)):
            raise ValueError("frequencies_hz 必須嚴格遞增")
        return self

    @property
    def placement(self) -> Placement:
        """這份單點輸入的真實代號與座標表。"""
        return point_placement(
            self.speaker_id,
            self.source_position_m,
            self.receiver_id,
            self.receiver_position_m,
        )


class TargetCurve(BaseModel):
    """事先指定的目標曲線：flat（水平）或 sloped（對數頻率上的固定傾斜）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["flat", "sloped"]
    tilt_db_per_octave: float

    @model_validator(mode="after")
    def _kind_matches_tilt(self) -> Self:
        """flat 的傾斜必須是零、sloped 必須不是零；一條曲線只有一種寫法。"""
        if (self.kind == "flat") != (self.tilt_db_per_octave == 0.0):
            raise ValueError("flat 目標的傾斜必須為 0，sloped 目標的傾斜必須非 0")
        return self


class _Settings(BaseModel):
    """評估這一次用到的登記簿條目，收成有名字的欄位；只在本模組內用。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    coverage_range_hz: tuple[float, float]
    tilt_fit_range_hz: tuple[float, float]
    ripple_range_hz: tuple[float, float]
    smoothing_width_octave_tilt: float
    smoothing_width_octave_ripple: float
    feature_min_width_octave: float
    min_points: int
    target: TargetCurve
    any_baseline: bool
    fingerprint: str


def timbre_input_from_report(
    report: ReportOutput,
    *,
    candidate_id: str,
    speaker_id: str,
    receiver_id: str,
    source_reference: str,
    provenance: InputProvenance,
) -> TimbreInput:
    """從報表收場景、聲源與接收點座標、細軸頻率、總能量與能力宣告。

    候選、喇叭與接收點代號仍由呼叫端負責真實；場景指紋、聲源與接收點座標只認報表
    ``scene``，不開呼叫端覆寫口。聲源基準與出身仍由呼叫端給，不猜、不填零、不寫
    unknown。報表今天沒有逐點標記，所以 ``report_flags`` 是空的。能力狀態與範圍只從
    報表拿。只讀，不改報表。
    """
    if report.points is None:
        raise ValueError("報表沒有細軸逐點表（產生報表時沒開 --points），收不到音色曲線")
    return TimbreInput(
        candidate_id=candidate_id,
        scene_fingerprint=report.scene.scene_fingerprint,
        speaker_id=speaker_id,
        receiver_id=receiver_id,
        source_position_m=(
            report.scene.source_m.x,
            report.scene.source_m.y,
            report.scene.source_m.z,
        ),
        receiver_position_m=(
            report.scene.receiver_m.x,
            report.scene.receiver_m.y,
            report.scene.receiver_m.z,
        ),
        frequencies_hz=tuple(row.frequency_hz for row in report.points),
        total_energy=tuple(row.total_energy for row in report.points),
        source_reference=source_reference,
        report_flags=(),
        model_validation_status=ModelValidationStatus(report.capability.status),
        model_validation_frequency_range_hz=report.capability.frequency_hz,
        provenance=provenance,
    )


def _setting_entry(
    purpose: QualityPurpose, name: str, expected_unit: Unit
) -> SettingEntry:
    entry = purpose.entry(_PREFIX + name)
    if not isinstance(entry, SettingEntry):
        raise TypeError(f"{_PREFIX + name} 應該是量法設定（setting）")
    if entry.unit != expected_unit:
        raise ValueError(
            f"{entry.key} 單位應為 {expected_unit}，登記簿寫 {entry.unit}"
        )
    return entry


def _range_value(entry: SettingEntry) -> tuple[float, float]:
    value = entry.value
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{entry.key} 必須是兩個數的頻率範圍")
    lower, upper = float(value[0]), float(value[1])
    if not 0.0 < lower < upper:
        raise ValueError(f"{entry.key} 必須是遞增的正頻率範圍")
    return lower, upper


def _scalar_value(entry: SettingEntry | TargetEntry) -> float:
    value = entry.value
    if isinstance(value, tuple):
        raise ValueError(f"{entry.key} 必須是單一數值")
    return float(value)


def _load_settings(
    path: str | Path, purpose_name: str, target: TargetCurve | None
) -> _Settings:
    """讀登記簿裡 timbre_balance.* 那幾條；目標沒給才讀預設目標，也才算「用到」那一條。"""
    registry = load_quality_targets(path)
    purpose = registry.purpose(purpose_name)
    ranges = {
        name: _setting_entry(purpose, name, _SETTING_UNITS[name])
        for name in ("coverage_range_hz", "tilt_fit_range_hz", "ripple_range_hz")
    }
    widths = {
        name: _setting_entry(purpose, name, _SETTING_UNITS[name])
        for name in (
            "smoothing_width_octave_tilt",
            "smoothing_width_octave_ripple",
            "feature_min_width_octave",
        )
    }
    min_points_entry = _setting_entry(
        purpose, "min_points", _SETTING_UNITS["min_points"]
    )
    if not isinstance(min_points_entry.value, int) or min_points_entry.value < 2:
        raise ValueError(f"{min_points_entry.key} 必須是至少 2 的整數（兩點才定得出一條線）")
    used: list[SettingEntry | TargetEntry] = [*ranges.values(), *widths.values(), min_points_entry]
    if target is None:
        target_entry = purpose.entry(_TARGET_TILT_KEY)
        if not isinstance(target_entry, TargetEntry):
            raise TypeError("timbre_balance.target_tilt_db_per_octave 應該是目標（target）")
        if target_entry.unit != _TARGET_TILT_UNIT:
            raise ValueError(
                f"{target_entry.key} 單位應為 {_TARGET_TILT_UNIT}，"
                f"登記簿寫 {target_entry.unit}"
            )
        tilt = _scalar_value(target_entry)
        target = TargetCurve(kind="flat" if tilt == 0.0 else "sloped", tilt_db_per_octave=tilt)
        used.append(target_entry)
    for name, entry in widths.items():
        # 傾斜的平滑寬度必須為正（傾斜是對平滑後的走勢擬合）；起伏的平滑寬度與峰谷最小寬度准填 0
        # ＝看原始曲線、不刪窄峰（老闆 2026-09-22 拍，票 #432）。負的一律紅。
        floor = 0.0 if name in _MAY_BE_ZERO else None
        value = _scalar_value(entry)
        if value < 0.0 or (floor is None and value <= 0.0):
            raise ValueError(f"{entry.key} 必須為{'非負' if floor is not None else '正'}")
    return _Settings(
        coverage_range_hz=_range_value(ranges["coverage_range_hz"]),
        tilt_fit_range_hz=_range_value(ranges["tilt_fit_range_hz"]),
        ripple_range_hz=_range_value(ranges["ripple_range_hz"]),
        smoothing_width_octave_tilt=_scalar_value(widths["smoothing_width_octave_tilt"]),
        smoothing_width_octave_ripple=_scalar_value(widths["smoothing_width_octave_ripple"]),
        feature_min_width_octave=_scalar_value(widths["feature_min_width_octave"]),
        min_points=min_points_entry.value,
        target=target,
        any_baseline=any(entry.status == "baseline" for entry in used),
        fingerprint=registry.fingerprint,
    )


def _in_range(frequencies: FloatArray, bounds: tuple[float, float]) -> NDArray[np.bool_]:
    return (frequencies >= bounds[0]) & (frequencies <= bounds[1])


def _smoothing_half_width_octave(width_octave: float) -> float:
    """回平滑實際使用的半窗；捨入護欄只在這裡算一次，依賴範圍與平滑共用。"""
    return 0.5 * width_octave + _WINDOW_ROUNDING_OCT


def _dependency_ranges(settings: _Settings) -> _DependencyRanges:
    """從本次登記設定現算傾斜與起伏各自真正讀到的頻率範圍。"""

    def expanded(bounds: FrequencyRange, width_octave: float) -> FrequencyRange:
        factor = 2.0 ** _smoothing_half_width_octave(width_octave)
        return bounds[0] / factor, bounds[1] * factor

    return (
        expanded(settings.tilt_fit_range_hz, settings.smoothing_width_octave_tilt),
        expanded(settings.ripple_range_hz, settings.smoothing_width_octave_ripple),
    )


def _smooth_energy(octaves: FloatArray, energy: FloatArray, width_octave: float) -> FloatArray:
    """能量域的分數八度平滑：對數頻率上以每一點為中心、寬 ``width_octave`` 的移動平均。

    視窗碰到資料端點就只平均資料內的點（截短、不外插）。
    """
    if width_octave <= 0.0:
        # 不平滑就是原始曲線本身：不走累積和（累積和相減會把很小的值捨成 0，票 #432 檢查席）。
        return np.asarray(energy, dtype=np.float64)
    half = _smoothing_half_width_octave(width_octave)
    lower = np.searchsorted(octaves, octaves - half, side="left")
    upper = np.searchsorted(octaves, octaves + half, side="right")
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    return np.asarray((cumulative[upper] - cumulative[lower]) / (upper - lower), dtype=np.float64)


def _fit_line(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """最小平方直線 ``y = slope·x + intercept``；平直資料回剛好是零的斜率。"""
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered = x - x_mean
    slope = float(np.sum(centered * (y - y_mean)) / np.sum(centered * centered))
    return slope, y_mean - slope * x_mean


def _half_depth_crossing(
    octaves: FloatArray, residual: FloatArray, index: int, half: float, step: int
) -> float | None:
    """從極值往一側走，找殘差回到 ``half``（半深度）的位置，在對數頻率上線性內插；
    走到評估範圍（或資料）邊上都沒回到就回 None（這一側邊界不完整）。"""
    sign = 1.0 if half > 0.0 else -1.0
    current = index
    while 0 <= current + step < len(residual):
        following = current + step
        if sign * residual[following] <= sign * half:
            span = residual[following] - residual[current]
            fraction = (half - residual[current]) / span
            return float(octaves[current] + fraction * (octaves[following] - octaves[current]))
        current = following
    return None


def _extremum_kind(residual: FloatArray, index: int) -> Literal["peak", "dip"] | None:
    """嚴格局部極大且在線上方＝峰、嚴格局部極小且在線下方＝谷；平台取左端那一點。"""
    left, here, right = residual[index - 1], residual[index], residual[index + 1]
    if here > 0.0 and here > left and here >= right:
        return "peak"
    if here < 0.0 and here < left and here <= right:
        return "dip"
    return None


# 半深度寬度不到相鄰兩點距離的這個倍數，就是軸上點太少、峰高可能沒抓準（票 #432）。
_AXIS_RESOLUTION_POINTS: Final[float] = 2.0


def _features(
    frequencies: FloatArray, residual: FloatArray, min_width_octave: float
) -> tuple[Feature, ...]:
    """起伏評估範圍內殘差的每一個局部極值各成一個特徵；各自照半深度量寬、不做巢狀。

    兩端點只看得到一側，不算局部極值。任一側找不到半深度交點的寬度記 None 並標邊界不完整。
    兩種寬度標記：比登記簿的最小寬度窄的標 ``FEATURE_TOO_NARROW``（代價會略過它；老闆
    2026-09-22 拍板最小寬度設 0、這個標記今天不會出現）；半深度寬度不到相鄰軸點距離的
    ``_AXIS_RESOLUTION_POINTS`` 倍（相鄰指峰左右那兩點、取較大的一邊）的標 ``FEATURE_NARROWER_THAN_AXIS``——照樣列、照樣計分，
    只是提醒這個峰窄到現在的頻率軸可能沒量準峰頂。
    """
    octaves = np.log2(frequencies)
    found: list[Feature] = []
    for index in range(1, len(residual) - 1):
        kind = _extremum_kind(residual, index)
        if kind is None:
            continue
        depth = float(residual[index])
        left = _half_depth_crossing(octaves, residual, index, 0.5 * depth, -1)
        right = _half_depth_crossing(octaves, residual, index, 0.5 * depth, 1)
        flags: tuple[Flag, ...] = ()
        width: float | None = None
        if left is None or right is None:
            flags = (Flag.FEATURE_BOUNDARY_INCOMPLETE,)
        else:
            width = right - left
            marks: list[Flag] = []
            if width < min_width_octave:
                marks.append(Flag.FEATURE_TOO_NARROW)
            # 尺是這個峰左右相鄰兩點的距離（取較大的那一邊），不是整條軸的中位數——軸可以不等距。
            local_step = float(max(octaves[index] - octaves[index - 1], octaves[index + 1] - octaves[index]))
            if width < _AXIS_RESOLUTION_POINTS * local_step:
                marks.append(Flag.FEATURE_NARROWER_THAN_AXIS)
            flags = tuple(marks)
        found.append(
            Feature(
                kind=kind,
                center_frequency_hz=float(frequencies[index]),
                depth_db=depth,
                width_octave=width,
                flags=flags,
            )
        )
    return tuple(found)


def _summary_index(features: Sequence[Feature], kind: Literal["peak", "dip"]) -> int | None:
    """最高峰（深度最大）或最深谷（深度最小）在清單裡的位置；只是摘要。"""
    candidates = [index for index, item in enumerate(features) if item.kind == kind]
    if not candidates:
        return None
    sign = 1.0 if kind == "peak" else -1.0
    return max(candidates, key=lambda index: sign * features[index].depth_db)


def _rms(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def _unavailable(
    data: TimbreInput, settings: _Settings, reason: ReasonCode, flags: tuple[Flag, ...]
) -> CategoryEvaluation:
    """不可估：payload 空、沒有原始量、不填零也不捏造，只帶原因與可回查的身分。"""
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=data.candidate_id,
        scene_fingerprint=data.scene_fingerprint,
        placement=data.placement,
        category=QualityCategory.TIMBRE_BALANCE,
        state=EvaluationState.UNAVAILABLE,
        payload=None,
        raw_quantities=(),
        category_cost=None,
        flags=flags,
        reason_codes=(reason,),
        evaluator_version=TIMBRE_EVALUATOR_VERSION,
        settings_fingerprint=settings.fingerprint,
        provenance=data.provenance,
    )


# 起伏不平滑（寬度 0）時判洞的尺：相鄰兩點的距離超過它左右鄰近間距（取較大的一邊）的這個倍數，
# 就是少了點。少一個軸點距離是鄰近間距的 2 倍，所以 1.5 倍剛好把「少一點」判成洞、又容得下
# 軸自己由疏變密的地方（票 #432）。
_RAW_GAP_STEPS: Final[float] = 1.5


def _has_gap(frequencies: FloatArray, width_octave: float) -> bool:
    """資料軸中間有沒有洞。

    有平滑：相鄰兩點的距離（八度）比視窗寬度還大就是洞（視窗連隔壁一點都收不到，平滑值只是
    單點）。不平滑（寬度 0）：相鄰距離比它左右鄰近間距的較大者還大 ``_RAW_GAP_STEPS`` 倍就是洞
    ——原始曲線少交一點不准變成沒有那個峰，而軸由疏變密不算洞。
    只看首末兩點會把「中間整段缺點」當成完整覆蓋（找碴席實測）。
    """
    steps = np.diff(np.log2(frequencies))
    if len(steps) == 0:
        return False
    if width_octave > 0.0:
        return bool(np.any(steps > width_octave))
    if len(steps) == 1:
        return False
    left = np.concatenate(([steps[1]], steps[:-1]))
    right = np.concatenate((steps[1:], [steps[-2]]))
    neighbours = np.maximum(left, right)
    return bool(np.any(steps > _RAW_GAP_STEPS * neighbours))


def _dependency_range_has_gap(
    frequencies: FloatArray, bounds: tuple[float, float], width_octave: float
) -> bool:
    """資料要完整包住依賴範圍；範圍內再把上下界當成虛擬點判缺段。尺是這一段自己的平滑寬度。"""
    if frequencies[0] > bounds[0] or frequencies[-1] < bounds[1]:
        return True
    scoped = frequencies[_in_range(frequencies, bounds)]
    with_boundaries = np.concatenate(([bounds[0]], scoped, [bounds[1]]))
    return _has_gap(with_boundaries, width_octave)


def _dependency_ranges_have_gap(
    frequencies: FloatArray, dependency_ranges: _DependencyRanges, settings: _Settings
) -> bool:
    """傾斜那一段用傾斜的寬度判、起伏那一段用起伏的寬度判——起伏是 0 時就是「少一點就是洞」。"""
    widths = (settings.smoothing_width_octave_tilt, settings.smoothing_width_octave_ripple)
    return any(
        _dependency_range_has_gap(frequencies, bounds, width)
        for bounds, width in zip(dependency_ranges, widths, strict=True)
    )


def _unique_flags(flags: Sequence[Flag]) -> tuple[Flag, ...]:
    return tuple(dict.fromkeys(flags))


def _model_is_validated(
    data: TimbreInput, dependency_ranges: _DependencyRanges
) -> bool:
    """狀態是驗過，而且報表宣告的範圍把兩段計分依賴範圍各自完整包住（含端點）才成立。

    payload 照抄報表的原始狀態與宣告範圍、不降級；「這一次評估算不算驗過」只看旗標。
    """
    declared = data.model_validation_frequency_range_hz
    if data.model_validation_status is not ModelValidationStatus.VALIDATED or not declared:
        return False
    return all(
        declared[0] <= lower and declared[1] >= upper
        for lower, upper in dependency_ranges
    )


def _tilt_and_line(
    frequencies: FloatArray, relative: FloatArray, settings: _Settings
) -> tuple[float, float]:
    """傾斜分支：能量域平滑 → dB → 在擬合範圍對 log2(f) 擬直線；回斜率與截距。"""
    octaves = np.log2(frequencies)
    smoothed_db = 10.0 * np.log10(
        _smooth_energy(octaves, relative, settings.smoothing_width_octave_tilt)
    )
    mask = _in_range(frequencies, settings.tilt_fit_range_hz)
    return _fit_line(octaves[mask], smoothed_db[mask])


def _ripple(
    frequencies: FloatArray, relative: FloatArray, line: tuple[float, float], settings: _Settings
) -> tuple[float, tuple[Feature, ...]]:
    """起伏分支：從原始能量另起平滑 → dB → 扣傾斜擬合線 → 範圍內均方根與峰谷清單。"""
    octaves = np.log2(frequencies)
    smoothed_db = 10.0 * np.log10(
        _smooth_energy(octaves, relative, settings.smoothing_width_octave_ripple)
    )
    residual = smoothed_db - (line[0] * octaves + line[1])
    mask = _in_range(frequencies, settings.ripple_range_hz)
    features = _features(frequencies[mask], residual[mask], settings.feature_min_width_octave)
    return _rms(residual[mask]), features


def _target_deviation(
    frequencies: FloatArray, relative: FloatArray, settings: _Settings
) -> tuple[float, tuple[tuple[float, float], ...]]:
    """對目標分支：不平滑的原始 dB 減目標曲線，扣掉覆蓋範圍內的平均（不管音量）再算均方根。"""
    mask = _in_range(frequencies, settings.coverage_range_hz)
    covered = frequencies[mask]
    target_db = settings.target.tilt_db_per_octave * np.log2(covered)
    difference = 10.0 * np.log10(relative[mask]) - target_db
    deviation = difference - float(np.mean(difference))
    curve = tuple(
        (float(frequency), float(value)) for frequency, value in zip(covered, deviation, strict=True)
    )
    return _rms(deviation), curve


def _raw_quantities(payload: TimbrePayload) -> tuple[RawQuantity, ...]:
    return (
        RawQuantity(name="tilt_db_per_octave", value=payload.tilt_db_per_octave, unit="dB/oct"),
        RawQuantity(name="target_deviation_rms_db", value=payload.target_deviation_rms_db, unit="dB"),
        RawQuantity(name="residual_rms_db", value=payload.residual_rms_db, unit="dB"),
        RawQuantity(name="feature_count", value=float(len(payload.features)), unit="1"),
    )


def _measured(
    data: TimbreInput,
    settings: _Settings,
    payload: TimbrePayload,
    flags: Sequence[Flag],
) -> CategoryEvaluation:
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=data.candidate_id,
        scene_fingerprint=data.scene_fingerprint,
        placement=data.placement,
        category=QualityCategory.TIMBRE_BALANCE,
        state=EvaluationState.MEASURED,
        payload=payload,
        raw_quantities=_raw_quantities(payload),
        category_cost=None,
        flags=_unique_flags(flags),
        reason_codes=(),
        evaluator_version=TIMBRE_EVALUATOR_VERSION,
        settings_fingerprint=settings.fingerprint,
        provenance=data.provenance,
    )


def evaluate_timbre(
    data: TimbreInput,
    *,
    purpose: str,
    quality_targets_path: str | Path,
    target_curve: TargetCurve | None = None,
) -> CategoryEvaluation:
    """量一份曲線的傾斜、起伏與對目標偏差，回 measured 或 unavailable 的類別評估。

    ``target_curve`` 沒給就用登記簿該用途的預設目標傾斜。登記簿路徑由呼叫端必給。
    """
    settings = _load_settings(quality_targets_path, purpose, target_curve)
    dependency_ranges = _dependency_ranges(settings)
    frequencies = np.asarray(data.frequencies_hz, dtype=np.float64)
    energy = np.asarray(data.total_energy, dtype=np.float64)
    data_range = (float(frequencies[0]), float(frequencies[-1]))
    flags: list[Flag] = list(data.report_flags)
    coverage = settings.coverage_range_hz
    if not _model_is_validated(data, dependency_ranges):
        flags.append(Flag.UNVALIDATED)
    narrowest = min(settings.smoothing_width_octave_tilt, settings.smoothing_width_octave_ripple)
    if data_range[0] > coverage[0] or data_range[1] < coverage[1] or _has_gap(frequencies, narrowest):
        flags.append(Flag.DATA_COVERAGE_SHORT)
    if settings.any_baseline:
        flags.append(Flag.BASELINE_SETTINGS)
    if not bool(np.all(np.isfinite(energy) & (energy > 0.0))):
        return _unavailable(data, settings, ReasonCode.NON_POSITIVE_ENERGY, _unique_flags(flags))
    for bounds in (settings.tilt_fit_range_hz, settings.ripple_range_hz):
        if int(np.count_nonzero(_in_range(frequencies, bounds))) < settings.min_points:
            return _unavailable(
                data, settings, ReasonCode.INSUFFICIENT_COVERAGE, _unique_flags(flags)
            )
    if _dependency_ranges_have_gap(frequencies, dependency_ranges, settings):
        return _unavailable(
            data, settings, ReasonCode.TIMBRE_SCORING_RANGE_GAP, _unique_flags(flags)
        )
    # 只在模組內除掉共同音量（除以最大能量）：相對量不帶絕對級，平直曲線因此剛好是零 dB。
    relative = energy / float(np.max(energy))
    line = _tilt_and_line(frequencies, relative, settings)
    residual_rms, features = _ripple(frequencies, relative, line, settings)
    deviation_rms, deviation_curve = _target_deviation(frequencies, relative, settings)
    payload = TimbrePayload(
        category="timbre_balance",
        tilt_db_per_octave=line[0],
        tilt_fit_range_hz=settings.tilt_fit_range_hz,
        tilt_dependency_range_hz=dependency_ranges[0],
        target_tilt_db_per_octave=settings.target.tilt_db_per_octave,
        target_deviation_rms_db=deviation_rms,
        deviation_curve=deviation_curve,
        residual_rms_db=residual_rms,
        ripple_range_hz=settings.ripple_range_hz,
        ripple_dependency_range_hz=dependency_ranges[1],
        features=features,
        strongest_peak_index=_summary_index(features, "peak"),
        deepest_dip_index=_summary_index(features, "dip"),
        data_range_hz=data_range,
        coverage_range_hz=coverage,
        model_validation_status=data.model_validation_status,
        model_validation_frequency_range_hz=data.model_validation_frequency_range_hz,
    )
    flags.extend(flag for feature in features for flag in feature.flags)
    return _measured(data, settings, payload, flags)
