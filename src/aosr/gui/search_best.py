"""目前最佳的已存結果投影：分圖驗欄位，不求解、不重評、不拿搜尋鎖。"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from pydantic import TypeAdapter

from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.quality_targets import QualityPurpose, SettingEntry, TargetEntry
from aosr.gui.labels import LOW_FREQUENCY_AXES, speaker_label
from aosr.gui.plan_view import plan_for
from aosr.gui.result_view import frequency_responses_for
from aosr.gui.search_view import _document, _reason, best_versions
from aosr.reporting.result import PairResult, RESULT_SCHEMA_VERSION, SchemeResult
from aosr.reporting.display import FURNITURE_MODEL_NOTE, FURNITURE_REFLECTION_NOTE
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import CandidateEvaluation, QualityCategory
from aosr.scoring.reflections_contract import ReflectionChannel, ReflectionsAndEchoPayload

ZONES = {"front": "前方", "lateral": "側方", "rear": "後方", "vertical": "上下"}


class BestCache:
    """保留最近幾份 JSON；互斥僅限伺服器記憶體，同檔並行請求也只剖析一次。"""

    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError("快取容量必須為正整數")
        self.capacity = capacity
        self._items: OrderedDict[tuple[Path, int, int], dict[str, object] | str] = OrderedDict()
        self._lock = Lock()

    def read(self, path: Path) -> dict[str, object]:
        stat = path.stat()
        key = (path.resolve(), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key not in self._items:
                try:
                    document = _document(path)
                    if document.get("schema_version") != RESULT_SCHEMA_VERSION:
                        raise ValueError("結果檔版本不認得")
                    unknown = document.keys() - SchemeResult.model_fields.keys()
                    if unknown:
                        raise ValueError(f"結果檔欄位不認得：{'、'.join(sorted(unknown))}")
                    after = path.stat()
                    if (after.st_mtime_ns, after.st_size) != key[1:]:
                        raise ValueError("結果檔正在更新，請等下一次讀取")
                    self._items[key] = document
                except (OSError, ValueError, UnicodeError) as error:
                    self._items[key] = _reason(error)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            value = self._items[key]
            if isinstance(value, str):
                raise ValueError(value)
            return value


def _chart(draw: Callable[[], dict[str, object]]) -> dict[str, object]:
    try:
        return {"error": "", **draw()}
    except (OSError, ValueError, UnicodeError, KeyError, TypeError) as error:
        return {"error": f"讀不到：{_reason(error)}"}


def _scheme(document: dict[str, object]) -> Scheme:
    return Scheme.model_validate(document["scheme"])


def _primary_pairs(document: dict[str, object], scheme: Scheme) -> tuple[PairResult, ...]:
    pairs = TypeAdapter(tuple[PairResult, ...]).validate_python(document["pairs"])
    primary = scheme.receiver_set.primary.receiver_id
    selected = tuple(pair for pair in pairs if pair.receiver_id == primary)
    expected = {(item.speaker_id, item.role) for item in scheme.channel_group.channels}
    if {(pair.speaker_id, pair.role) for pair in selected} != expected or len(selected) != len(expected):
        raise ValueError("主位報表與方案聲道對不上")
    return selected


def _frequency(document: dict[str, object]) -> dict[str, object]:
    scheme = _scheme(document)
    pairs = _primary_pairs(document, scheme)
    curves = frequency_responses_for(scheme, pairs)
    if not all(curve.points for curve in curves):
        raise ValueError("主位沒有逐點頻響")
    axis = sorted({point.frequency_hz for curve in curves for point in curve.points})
    levels = [{point.frequency_hz: point.level_db for point in curve.points} for curve in curves]
    data = [axis, *[[values.get(x) for x in axis] for values in levels]]
    return {"curves": [{"role": curve.role,
                         "label": f"{'左' if curve.role == 'left' else '右'}聲道 → 主位",
                         "points": [[point.frequency_hz, point.level_db] for point in curve.points]}
                        for curve in curves],
            "data": data,
            "axis_text": "、".join(sorted({LOW_FREQUENCY_AXES[pair.report.top.low_frequency_axis.value] for pair in pairs}))}


def _setting(purpose: QualityPurpose, key: str, unit: str) -> float:
    entry = purpose.entry(f"reflections_and_echo.{key}")
    if not isinstance(entry, TargetEntry | SettingEntry) or entry.unit != unit or not isinstance(entry.value, float | int):
        raise ValueError(f"快照缺少 {key} 的 {unit} 數值")
    return float(entry.value)


def _channel(channel: ReflectionChannel, thresholds: dict[str, float], window: float) -> dict[str, object]:
    # 窗內與否用存下的旗標；只有窗內的點才算超線，窗外的點高於門檻不算違規。
    points = [{"delay_ms": path.relative_direct_delay_s * 1000,
               "level_db": path.broadband_level_db, "zone": path.zone.value,
               "within_window": path.within_window,
               "over_limit": path.within_window and path.broadband_level_db is not None
               and path.broadband_level_db > thresholds[path.zone.value]}
              for path in channel.reflections]
    inside = sum(bool(point["within_window"]) for point in points)
    over = sum(bool(point["over_limit"]) for point in points)
    missing = sum(point["level_db"] is None for point in points)
    summary = (f"窗內 {inside} 條；窗內寬頻聲級超過分區門檻 {over} 條（空心點）。"
               "評分看的是各頻率點的最強反射，條數可能跟這裡不同。")
    if missing:
        summary += f" {missing} 條沒有可畫的聲級，未列入超線數。"
    if channel.coverage == "approximate":
        summary += f" {FURNITURE_MODEL_NOTE}；{FURNITURE_REFLECTION_NOTE}。"
    if channel.coverage not in ("complete", "approximate") or channel.state.value != "measured":
        summary += " 反射資料覆蓋或量測未完成，不能當完整判斷。"
    return {"role": channel.role, "label": f"{speaker_label(channel.role)} → 主位", "points": points,
            "summary_text": summary, "inside_count": inside, "over_count": over}


def _rfz(document: dict[str, object], content: dict[str, object]) -> dict[str, object]:
    purpose = QualityPurpose.model_validate(content)
    thresholds = {zone: _setting(purpose, f"zone_threshold_db.{zone}", "dB") for zone in ZONES}
    window = _setting(purpose, "window_upper_ms", "ms")
    candidate = CandidateEvaluation.model_validate(document["candidate"])
    evaluation = next((item for item in candidate.evaluations if item.category == QualityCategory.REFLECTIONS_AND_ECHO), None)
    if evaluation is None or not isinstance(evaluation.payload, ReflectionsAndEchoPayload):
        raise ValueError("候選評估沒有已存反射資料")
    if evaluation.payload.window_upper_ms != window:
        raise ValueError(f"存下的反射時間窗 {evaluation.payload.window_upper_ms:.2f} 毫秒跟這場搜尋的設定 {window:.2f} 毫秒不同")
    channels = tuple(item for item in evaluation.payload.channels if item.is_primary)
    if not channels:
        raise ValueError("候選評估沒有主位反射資料")
    shared = len(set(thresholds.values())) < len(thresholds)
    return {"thresholds": thresholds, "window_ms": window,
            "window_text": f"時間窗終點：{window:.2f} 毫秒" + ("；門檻相同的分區合畫成一條灰色虛線" if shared else ""),
            "zones": [{"key": zone, "label": label, "threshold_db": thresholds[zone],
                       "text": f"{label}：{thresholds[zone]:.1f} dB"} for zone, label in ZONES.items()],
            "channels": [_channel(channel, thresholds, window) for channel in channels]}


def _axis(document: dict[str, object]) -> str:
    scheme = _scheme(document)
    axis = scheme.scene.low_frequency_axis
    if axis is None:
        pairs = document.get("pairs")
        if not isinstance(pairs, list) or not pairs or not isinstance(pairs[0], dict):
            raise ValueError("缺頻率軸身分")
        report = pairs[0]["report"]
        if not isinstance(report, dict) or not isinstance(report.get("top"), dict):
            raise ValueError("缺頻率軸身分")
        axis = LowFrequencyAxis(report["top"]["low_frequency_axis"])
    return LOW_FREQUENCY_AXES[axis.value]


def build_best_view(path: Path, *, which: str | None, cache: BestCache, directivity: DirectivityDefaults) -> dict[str, object]:
    if which is not None and which not in {"search", "refine"}:
        raise ValueError("which 只接受 search（搜尋）或 refine（細算）")
    store, versions, default = best_versions(path)
    which = which or default
    selected = versions[which]
    name = "搜尋第一名" if which == "search" else "細算第一名"
    number = "原方案" if selected.candidate == "baseline" else f"試算 {selected.candidate}" if selected.candidate else "尚無候選"
    title = f"目前最佳：{name}／{number}"
    try:
        if selected.error or store.value is None:
            raise ValueError(selected.error.removeprefix("讀不到：") or store.error.removeprefix("讀不到："))
        content = store.value.identity.purpose_settings.content
        document = cache.read(path / selected.result_file)
        axis = _chart(lambda: {"text": _axis(document)})
        title += f"／{axis.get('text', axis['error'])}"
        charts = {"frequency": _chart(lambda: _frequency(document)),
                  "rfz": _chart(lambda: _rfz(document, content)),
                  "plan": _chart(lambda: {"data": plan_for(_scheme(document), directivity)})}
    except (OSError, ValueError, UnicodeError) as error:
        charts = {key: {"error": f"讀不到：{_reason(error)}"} for key in ("frequency", "rfz", "plan")}
    return {"which": which, "version": selected.version, "title": title, **charts}
