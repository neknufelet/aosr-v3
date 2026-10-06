"""低頻拖尾參考曲線（#669）：答案直接讀登記簿 TOML 另算，不照抄被測的讀值程式。

邊界照決策紙 docs/decisions/low-frequency-decay-reference-curve.md：錨點本身是文獻錨點；錨點之間 T60 對 log f 走直線；
延伸段含下端、不含上端；其餘未評估（t60 是 None），不外推、不當通過。人工線只有錨點與內插。
"""

import math
import tomllib
from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, QualityTargets
from aosr.scoring.low_frequency_decay_reference import (
    SETTING_KEYS,
    LowFrequencyDecayReference,
    ReferenceOrigin,
    load_reference,
)

PURPOSE = "dedicated_two_channel_listening_room"
PREFIX = "low_frequency_decay."
Raw = dict[str, object]


def _raw_purpose() -> Raw:
    with config_path("quality_targets.toml").open("rb") as handle:
        document: Raw = tomllib.load(handle)
    return next(purpose for purpose in cast(list[Raw], document["purpose"]) if purpose["name"] == PURPOSE)


def _settings(raw: Raw) -> list[Raw]:
    return cast(list[Raw], raw["setting"])


def _value(raw: Raw, name: str) -> object:
    return next(entry["value"] for entry in _settings(raw) if entry["key"] == PREFIX + name)


def _floats(raw: Raw, name: str) -> list[float]:
    return [float(value) for value in cast(list[float], _value(raw, name))]


def _purpose(raw: Raw) -> QualityPurpose:
    return QualityTargets.model_validate({"schema_version": 1, "purpose": [raw]}).purpose(PURPOSE)


def _reference() -> LowFrequencyDecayReference:
    return load_reference(_purpose(_raw_purpose()))


def _changed(name: str, **fields: object) -> QualityPurpose:
    raw = _raw_purpose()
    entry = next(item for item in _settings(raw) if item["key"] == PREFIX + name)
    before = dict(entry)
    entry.update(fields)
    assert entry != before
    return _purpose(raw)


def test_reader_declares_exactly_the_registry_keys_it_reads() -> None:
    registered = {str(entry["key"]) for entry in _settings(_raw_purpose()) if str(entry["key"]).startswith(PREFIX)}
    assert set(SETTING_KEYS) == registered


@pytest.mark.parametrize("line,frequencies,t60", [
    ("target", "reference_anchor_frequencies_hz", "reference_anchor_t60_s"),
    ("strict", "strict_reference_frequencies_hz", "strict_reference_t60_s"),
])
def test_anchor_frequencies_return_registered_values_as_literature(line: str, frequencies: str, t60: str) -> None:
    raw, reference = _raw_purpose(), _reference()
    pairs = list(zip(_floats(raw, frequencies), _floats(raw, t60), strict=True))
    assert pairs
    for frequency, expected in pairs:
        value = getattr(reference, f"{line}_at")(frequency)
        assert (value.t60_s, value.origin) == (expected, ReferenceOrigin.LITERATURE_ANCHOR)


@pytest.mark.parametrize("line,frequencies,t60", [
    ("target", "reference_anchor_frequencies_hz", "reference_anchor_t60_s"),
    ("strict", "strict_reference_frequencies_hz", "strict_reference_t60_s"),
])
def test_between_anchors_t60_is_a_straight_line_in_log_frequency(line: str, frequencies: str, t60: str) -> None:
    """幾何中點拿到兩端的算術平均；四分之一處（對數）拿到四分之一的差——直線在 log f，不在 f。"""
    raw, reference = _raw_purpose(), _reference()
    points, values = _floats(raw, frequencies), _floats(raw, t60)
    segments = list(zip(points, values, points[1:], values[1:]))
    assert segments
    for low_hz, low_s, high_hz, high_s in segments:
        for share in (0.25, 0.5):
            frequency = low_hz * (high_hz / low_hz) ** share
            value = getattr(reference, f"{line}_at")(frequency)
            assert value.origin == ReferenceOrigin.ENGINEERING_INTERPOLATION
            assert value.t60_s is not None
            assert math.isclose(value.t60_s, low_s + share * (high_s - low_s), rel_tol=1e-12)


def test_extension_holds_registered_value_from_lower_end_up_to_but_not_including_lowest_anchor() -> None:
    raw, reference = _raw_purpose(), _reference()
    low, high = _floats(raw, "reference_extension_range_hz")
    held = float(cast(float, _value(raw, "reference_extension_t60_s")))
    for frequency in (low, math.sqrt(low * high), math.nextafter(high, 0.0)):
        value = reference.target_at(frequency)
        assert (value.t60_s, value.origin) == (held, ReferenceOrigin.ENGINEERING_EXTENSION)
    assert reference.target_at(high).origin == ReferenceOrigin.LITERATURE_ANCHOR


def test_extension_value_is_its_own_entry_not_borrowed_from_the_lowest_anchor() -> None:
    """決策紙第 2 條：延伸值另登一條。改延伸值只動延伸段；改最低錨點不帶著延伸段改。"""
    raw = _raw_purpose()
    low, high = _floats(raw, "reference_extension_range_hz")
    held = float(cast(float, _value(raw, "reference_extension_t60_s")))
    anchors = _floats(raw, "reference_anchor_t60_s")
    moved_extension = held + 0.07
    reference = load_reference(_changed("reference_extension_t60_s", value=moved_extension))
    assert (reference.target_at(low).t60_s, reference.target_at(low).origin) == (
        moved_extension, ReferenceOrigin.ENGINEERING_EXTENSION)
    assert reference.target_at(high).t60_s == anchors[0]
    moved_anchor = [anchors[0] - 0.06, *anchors[1:]]
    reference = load_reference(_changed("reference_anchor_t60_s", value=moved_anchor))
    assert reference.target_at(low).t60_s == held
    assert reference.target_at(high).t60_s == moved_anchor[0]


@pytest.mark.parametrize("line,lowest,highest", [
    ("target", ("reference_extension_range_hz", 0), ("reference_anchor_frequencies_hz", -1)),
    ("strict", ("strict_reference_frequencies_hz", 0), ("strict_reference_frequencies_hz", -1)),
])
def test_outside_the_line_is_not_assessed_and_has_no_value(
    line: str, lowest: tuple[str, int], highest: tuple[str, int]
) -> None:
    raw, reference = _raw_purpose(), _reference()
    low = _floats(raw, lowest[0])[lowest[1]]
    high = _floats(raw, highest[0])[highest[1]]
    for frequency in (math.nextafter(low, 0.0), low / 2.0, math.nextafter(high, math.inf), high * 2.0):
        value = getattr(reference, f"{line}_at")(frequency)
        assert (value.t60_s, value.origin) == (None, ReferenceOrigin.NOT_ASSESSED)


def test_target_is_stricter_than_the_strict_line_at_the_extension_low_end_and_not_at_the_lowest_anchor() -> None:
    """#669 的更正：照提案規則兩條線在延伸段裡交叉，交點以下目標線反而比人工線嚴。"""
    raw, reference = _raw_purpose(), _reference()
    low, high = _floats(raw, "reference_extension_range_hz")
    for frequency, target_stricter in ((low, True), (high, False)):
        target, strict = reference.target_at(frequency).t60_s, reference.strict_at(frequency).t60_s
        assert target is not None and strict is not None
        assert (target < strict) is target_stricter


@pytest.mark.parametrize("frequency", [0.0, -63.0, math.nan, math.inf])
def test_frequency_must_be_finite_and_positive(frequency: float) -> None:
    with pytest.raises(ValueError, match="有限正數"):
        _reference().target_at(frequency)


@pytest.mark.parametrize("name,fields,error,message", [
    ("reference_anchor_t60_s", {"value": [0.51, 0.30]}, ValueError, "非空且等長"),
    ("reference_anchor_frequencies_hz", {"value": [125.0, 63.0, 250.0]}, ValueError, "嚴格遞增"),
    ("strict_reference_frequencies_hz", {"value": [32.0, 63.0, 63.0, 150.0, 200.0]}, ValueError, "嚴格遞增"),
    ("reference_anchor_t60_s", {"value": [0.51, 0.0, 0.12]}, ValueError, "必須為正"),
    ("reference_anchor_t60_s", {"unit": "ms"}, ValueError, "單位應為 s"),
    ("reference_extension_range_hz", {"value": [32.0, 50.0]}, ValueError, "最低的錨點頻率"),
    ("reference_extension_range_hz", {"value": [63.0, 32.0]}, ValueError, "遞增的正數"),
    ("reference_extension_range_hz", {"value": [32.0, 100.0]}, ValueError, "最低的錨點頻率"),
    ("reference_extension_range_hz", {"value": [32.0, 63.0, 70.0]}, ValueError, "遞增的正數"),
    ("reference_extension_range_hz", {"value": [0.0, 63.0]}, ValueError, "遞增的正數"),
    ("reference_extension_t60_s", {"value": -0.5}, ValueError, "必須為正"),
    ("strict_reference_frequencies_hz", {"value": [-32.0, 63.0, 100.0, 150.0, 200.0]}, ValueError, "必須為正"),
    ("reference_extension_t60_s", {"value": [0.51]}, TypeError, "單一數值"),
    ("strict_reference_t60_s", {"value": 0.9}, TypeError, "數值清單"),
])
def test_malformed_registry_is_rejected_instead_of_guessed(
    name: str, fields: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        load_reference(_changed(name, **fields))
