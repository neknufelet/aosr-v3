"""#558：凍結答案來自改動前 HEAD 5f3b3cf7 實跑，命令與完整值在答案檔。

逐格用第 6 條評分層容差；獨立手算用純量加權最小平方與八度格積分，不呼叫被測函式。
"""
import json
import math
import re
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pytest
from pydantic import ValidationError

from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.scoring import timbre
from aosr.scoring.contract import EvaluationState, Flag, ReasonCode, TimbrePayload
from tests.engine._cross_cpu_tolerance import SCORING_ABS, SCORING_REL, assert_within, cell
from tests.engine._frequency_split_control import (
    PURPOSE, REGISTRY, channel, channel_points, curve_input, listening, measured,
)

_T = TypeVar("_T")

ANSWERS = json.loads((Path(__file__).parents[2] / "blueprint/558-frequency-split-control.json").read_text())


def bounds(key: str) -> tuple[float, float]:
    entry = load_quality_targets(REGISTRY).purpose(PURPOSE).entry(key)
    assert isinstance(entry, SettingEntry)
    value = entry.value
    assert isinstance(value, tuple)
    return float(value[0]), float(value[1])


def compare(actual: object, expected: object, path: str = "") -> None:
    """所有格逐一核對；頻率鍵、狀態、原因、旗標、索引維持逐字。"""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys(), path
        for name, value in expected.items():
            compare(actual[name], value, f"{path}.{name}")
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
            compare(got, want, f"{path}[{i}]")
    elif isinstance(expected, float) and not ("range_hz[" in path and "dependency_range_hz[" not in path or "frequency_hz" in path or
                                             ("deviation_curve[" in path and path.endswith("[0]"))):
        assert isinstance(actual, float)
        assert_within([cell(path, actual, expected, rel=SCORING_REL, abs_=SCORING_ABS)])
    else:
        assert actual == expected, path


def fold_metadata(document: _T) -> _T:
    """只排除刻意改版的身分；數值與結構不摺疊。"""
    if isinstance(document, dict):
        return cast(_T, {k: fold_metadata(v) for k, v in document.items()
                if k not in {"schema_version", "evaluator_version", "settings_fingerprint",
                             "timbre_settings_fingerprint", "cost_settings_fingerprint"}})
    if isinstance(document, list):
        return cast(_T, [fold_metadata(v) for v in document])
    return document


def independent_rms(data: timbre.TimbreInput, score_bounds: tuple[float, float]) -> float:
    """純量手算：原始能量／最大值；傾斜能量平滑；擬線；原始殘差；每八度等權。"""
    purpose = load_quality_targets(REGISTRY).purpose(PURPOSE)
    width_entry = purpose.entry("timbre_balance.smoothing_width_octave_tilt")
    raw_entry = purpose.entry("timbre_balance.smoothing_width_octave_ripple")
    assert isinstance(width_entry, SettingEntry) and isinstance(width_entry.value, float)
    assert isinstance(raw_entry, SettingEntry) and raw_entry.value == 0.0
    width = width_entry.value
    axis = data.frequencies_hz
    x = [math.log2(f) for f in axis]
    peak = max(data.total_energy)
    relative = [e / peak for e in data.total_energy]
    w = [x[1] - x[0], *[(x[i+1] - x[i-1])/2 for i in range(1, len(x)-1)], x[-1]-x[-2]]
    scale = (x[-1]-x[0])/sum(w)
    w = [v*scale for v in w]
    half = width/2 + 64 * np.finfo(float).eps
    smooth_db = []
    for center in x:
        inside = [i for i, v in enumerate(x) if center-half <= v <= center+half]
        smooth_db.append(10*math.log10(sum(w[i]*relative[i] for i in inside)/sum(w[i] for i in inside)))

    def cells(selected: list[int], limits: tuple[float, float]) -> list[float]:
        edges = [math.log2(limits[0]), *[(x[a]+x[b])/2 for a,b in zip(selected, selected[1:])],
                 math.log2(limits[1])]
        return [b-a for a,b in zip(edges, edges[1:])]

    fit_bounds = bounds("timbre_balance.tilt_fit_range_hz")
    fit = [i for i,f in enumerate(axis) if fit_bounds[0] <= f <= fit_bounds[1]]
    weights = cells(fit, fit_bounds)
    mean_x = sum(wi*x[i] for i,wi in zip(fit,weights))/sum(weights)
    mean_y = sum(wi*smooth_db[i] for i,wi in zip(fit,weights))/sum(weights)
    slope = sum(wi*(x[i]-mean_x)*(smooth_db[i]-mean_y) for i,wi in zip(fit,weights)) / sum(
        wi*(x[i]-mean_x)**2 for i,wi in zip(fit,weights))
    intercept = mean_y-slope*mean_x
    scored = [i for i,f in enumerate(axis) if score_bounds[0] <= f <= score_bounds[1]]
    weights = cells(scored, score_bounds)
    return math.sqrt(sum(wi*(10*math.log10(relative[i])-slope*x[i]-intercept)**2
                         for i,wi in zip(scored,weights))/sum(weights))


def test_timbre_split_retains_features_and_uses_independent_rms() -> None:
    data = curve_input()
    evaluation = measured(data)
    payload = evaluation.payload
    assert isinstance(payload, TimbrePayload)
    actual = fold_metadata(evaluation.model_dump(mode="json"))
    expected = fold_metadata(ANSWERS["timbre"])
    for name in ("residual_rms_db", "ripple_range_hz", "ripple_dependency_range_hz"):
        actual["payload"].pop(name)
        expected["payload"].pop(name)
    for name in ("feature_range_hz", "feature_dependency_range_hz"):
        actual["payload"].pop(name)
    actual["raw_quantities"] = [v for v in actual["raw_quantities"] if v["name"] != "residual_rms_db"]
    expected["raw_quantities"] = [v for v in expected["raw_quantities"] if v["name"] != "residual_rms_db"]
    compare(actual, expected)
    want = independent_rms(data, bounds("timbre_balance.ripple_range_hz"))
    compare(payload.residual_rms_db, want)
    assert payload.residual_rms_db != pytest.approx(ANSWERS["timbre"]["payload"]["residual_rms_db"])
    compare(independent_rms(data, bounds("timbre_balance.feature_range_hz")),
            ANSWERS["timbre"]["payload"]["residual_rms_db"])
    tilt = payload.tilt_dependency_range_hz
    ripple = payload.ripple_range_hz  # 登記寬度 0；傾斜依賴仍讀到下界以下。
    assert payload.ripple_dependency_range_hz == (min(tilt[0], ripple[0]), max(tilt[1], ripple[1]))
    assert payload.feature_dependency_range_hz == payload.feature_range_hz


def test_legacy_ripple_range_keeps_old_dependency_and_measurement(tmp_path: Path) -> None:
    path = tmp_path / "quality.toml"
    text = REGISTRY.read_text()
    current = bounds("timbre_balance.ripple_range_hz")
    old = ANSWERS["timbre"]["payload"]["ripple_range_hz"]
    before = f'key = "timbre_balance.ripple_range_hz"\nvalue = {list(current)}'
    after = f'key = "timbre_balance.ripple_range_hz"\nvalue = {old}'
    assert before in text
    # 以文字替換函式改內容；不把登記簿讀出的字串當成 Path.replace 的目標。
    path.write_text(re.sub(re.escape(before), lambda _: after, text))
    got = measured(curve_input(), path).payload
    assert isinstance(got, TimbrePayload)
    compare(got.ripple_dependency_range_hz, tuple(ANSWERS["timbre"]["payload"]["ripple_dependency_range_hz"]))
    compare(got.residual_rms_db, ANSWERS["timbre"]["payload"]["residual_rms_db"])


def test_channel_whole_measurement_and_cost_match_main() -> None:
    got, want = channel().model_dump(mode="json"), ANSWERS["channel"]
    got, want = fold_metadata(got), fold_metadata(want)
    # 內嵌的音色酬載有意改分頻，由上一題逐格守；左右整份量測與代價仍逐格比。
    for document in (got, want):
        for point in document["payload"]["point_sources"]:
            for response in point["channels"]:
                payload = response["payload"]
                for key in ("residual_rms_db", "ripple_range_hz", "ripple_dependency_range_hz",
                            "feature_range_hz", "feature_dependency_range_hz"):
                    payload.pop(key, None)
                response["raw_quantities"] = [q for q in response["raw_quantities"] if q["name"] != "residual_rms_db"]
    compare(got, want)


def test_listening_keeps_other_metrics_and_follows_new_timbre_rms() -> None:
    got, want = fold_metadata(listening().model_dump(mode="json")), fold_metadata(ANSWERS["listening"])
    ripple = got["payload"].pop("ripple_rms_stability")
    old = want["payload"].pop("ripple_rms_stability")
    assert ripple != old
    expected = abs(independent_rms(curve_input("main"), bounds("timbre_balance.ripple_range_hz")) -
                   independent_rms(curve_input("front"), bounds("timbre_balance.ripple_range_hz")))
    compare(ripple["primary_to_surrounding"]["weighted_mean_deviation"], expected)
    # 只有一個周圍點：最差那一格與兩個原始量也都是這個差；指到的座位與參考照舊（#558 找碴席）。
    compare(ripple["primary_to_surrounding"]["worst_deviation"]["value"], expected)
    for side in ("receiver", "reference"):
        assert ripple["primary_to_surrounding"]["worst_deviation"][side] == old["primary_to_surrounding"]["worst_deviation"][side]
    assert {key: value for key, value in ripple.items() if key != "primary_to_surrounding"} == {
        key: value for key, value in old.items() if key != "primary_to_surrounding"}
    # 起伏原始量整筆照改動前對照（名稱、單位與其餘欄位都比），只把預期數值換成獨立算的新值（#558 複查席）。
    ripple_raw = [q for q in want["raw_quantities"] if q["name"].startswith("ripple_rms")]
    assert ripple_raw
    for quantity in ripple_raw:
        quantity["value"] = expected
    compare(got, want)


def test_three_dependency_checks_keep_low_frequency_validation_and_gap() -> None:
    data = curve_input().model_copy(update={"model_validation_frequency_range_hz": bounds("timbre_balance.ripple_range_hz")})
    assert Flag.UNVALIDATED in measured(data).flags
    drop = min(range(len(data.frequencies_hz)), key=lambda i: abs(data.frequencies_hz[i]-48.0))
    holed = data.model_copy(update={
        "frequencies_hz": tuple(f for i,f in enumerate(data.frequencies_hz) if i != drop),
        "total_energy": tuple(e for i,e in enumerate(data.total_energy) if i != drop),
    })
    got = measured(holed)
    assert got.state is EvaluationState.UNAVAILABLE
    assert got.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


@pytest.mark.parametrize("name", ("feature_range_hz", "feature_dependency_range_hz"))
def test_new_payload_ranges_are_validated(name: str) -> None:
    payload = measured(curve_input()).payload
    assert isinstance(payload, TimbrePayload)
    document = payload.model_dump()
    document[name] = tuple(reversed(bounds("timbre_balance.feature_range_hz")))
    with pytest.raises(ValidationError, match=name):
        TimbrePayload.model_validate(document)


def test_registry_split_values_and_sources_are_loaded() -> None:
    purpose = load_quality_targets(REGISTRY).purpose(PURPOSE)
    low_entry = load_quality_targets(REGISTRY).stage_one(PURPOSE).entry("stage_one_low_frequency.scored_range_hz")
    assert isinstance(low_entry, SettingEntry) and isinstance(low_entry.value, tuple)
    low = low_entry.value
    assert bounds("timbre_balance.ripple_range_hz")[0] == low[1]
    for key in ("timbre_balance.ripple_range_hz", "timbre_balance.feature_range_hz", "channel_matching.ripple_range_hz"):
        entry = purpose.entry(key)
        assert isinstance(entry, SettingEntry)
        assert entry.status == "baseline"
        assert entry.source_kind == "product_choice"
        assert "2026-10-25" in entry.source and "#558" in entry.source
    assert bounds("timbre_balance.feature_range_hz") == tuple(ANSWERS["timbre"]["payload"]["ripple_range_hz"])
    assert bounds("timbre_balance.feature_range_hz") == bounds("channel_matching.ripple_range_hz")
    assert bounds("timbre_balance.ripple_range_hz")[1] == bounds("timbre_balance.coverage_range_hz")[1]
    assert bounds("timbre_balance.ripple_range_hz")[0] > bounds("timbre_balance.feature_range_hz")[0]


@pytest.mark.parametrize("side", (0, 1))
def test_channel_rejects_loaded_timbre_settings_fingerprint_mismatch(side: int) -> None:
    points = channel_points()
    response = points[0].responses[side]
    changed = response.model_copy(update={"timbre_evaluation": response.timbre_evaluation.model_copy(update={"settings_fingerprint": "other"})})
    responses = tuple(changed if i == side else item for i, item in enumerate(points[0].responses))
    first = points[0].model_copy(update={"responses": responses})
    with pytest.raises(TypeError, match="音色.*設定"):
        channel(points=(first, *points[1:]))


@pytest.mark.parametrize("damage, reason", (("few", ReasonCode.INSUFFICIENT_COVERAGE),
                                            ("hole", ReasonCode.TIMBRE_SCORING_RANGE_GAP)))
def test_channel_raw_ripple_coverage_is_unavailable(damage: str, reason: ReasonCode) -> None:
    points = channel_points()
    changed = []
    for point in points:
        responses = []
        for response in point.responses:
            axis = response.frequencies_hz
            drop = min(range(len(axis)), key=lambda i: abs(axis[i]-48.0))
            kept = tuple(i for i in range(len(axis)) if (i in (drop, drop+1) if damage == "few" else i != drop))
            responses.append(response.model_copy(update={
                "frequencies_hz": tuple(axis[i] for i in kept),
                "total_energy": tuple(response.total_energy[i] for i in kept),
            }))
        changed.append(point.model_copy(update={"responses": tuple(responses)}))
    result = channel(points=tuple(changed))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE, reason)


def test_feature_dependency_alone_prevents_validation_promotion() -> None:
    payload = measured(curve_input()).payload
    assert isinstance(payload, TimbrePayload)
    # 這個宣告已包住傾斜與起伏外包；只有峰谷的低頻依賴還沒驗過。
    declared = payload.ripple_dependency_range_hz
    result = measured(curve_input().model_copy(update={"model_validation_frequency_range_hz": declared}))
    assert Flag.UNVALIDATED in result.flags


def test_feature_dependency_participates_in_min_points(tmp_path: Path) -> None:
    data = curve_input()
    index = min(range(len(data.frequencies_hz)), key=lambda i: abs(data.frequencies_hz[i]-48.0))
    narrow = list(data.frequencies_hz[index:index+2])
    path = tmp_path / "quality.toml"
    key = 'key = "timbre_balance.feature_range_hz"\nvalue = '
    path.write_text(REGISTRY.read_text().replace(key+str(list(bounds("timbre_balance.feature_range_hz"))), key+str(narrow)))
    result = measured(data, path)
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)


def test_shared_ripple_path_is_bitwise_same_as_timbre_at_same_range() -> None:
    data = curve_input()
    settings = timbre.load_timbre_settings(REGISTRY, PURPOSE)
    direct = timbre.measure_timbre_ripple(np.asarray(data.frequencies_hz), np.asarray(data.total_energy),
                                          settings.ripple_range_hz, settings)
    payload = measured(data).payload
    assert isinstance(payload, TimbrePayload)
    assert direct.residual_rms_db == payload.residual_rms_db
    assert direct.tilt_db_per_octave == payload.tilt_db_per_octave
    assert direct.features == payload.features


def test_expanded_dependency_does_not_replace_minimum_fit_points(tmp_path: Path) -> None:
    data = curve_input()
    axis = data.frequencies_hz
    index = min(range(len(axis)), key=lambda i: abs(axis[i]-1200.0))
    # 名義擬合範圍只有一點，外擴窗內卻有多點；不能拿平滑鄰點冒充擬合點。
    narrow = [axis[index]*0.999, axis[index]*1.001]
    text = REGISTRY.read_text()
    old = bounds("timbre_balance.tilt_fit_range_hz")
    text = text.replace('key = "timbre_balance.tilt_fit_range_hz"\nvalue = '+str(list(old)),
                        'key = "timbre_balance.tilt_fit_range_hz"\nvalue = '+str(narrow))
    import re
    text = re.sub(r'(key = "timbre_balance.min_points"\nvalue = )\d+', r'\g<1>2', text)
    path = tmp_path / "quality.toml"
    path.write_text(text)
    result = measured(data, path)
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)


def test_channel_uses_its_registered_range_and_changes_fingerprint(tmp_path: Path) -> None:
    old = bounds("channel_matching.ripple_range_hz")
    own_range = bounds("timbre_balance.ripple_range_hz")
    path = tmp_path / "quality.toml"
    key = 'key = "channel_matching.ripple_range_hz"\nvalue = '
    path.write_text(REGISTRY.read_text().replace(key+str(list(old)), key+str(list(own_range))))
    original, changed = channel(), channel(path)
    assert original.settings_fingerprint != changed.settings_fingerprint
    assert changed.payload is not None
    expected = independent_rms(curve_input(), own_range) - independent_rms(curve_input(role="right"), own_range)
    actual = changed.model_dump(mode="json")["payload"]["point_results"][0]["ripple_rms_difference_db"]
    compare(actual, expected)
    before = original.model_dump(mode="json")["payload"]["point_results"][0]["ripple_rms_difference_db"]
    assert actual != pytest.approx(before)
