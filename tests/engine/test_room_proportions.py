"""比例提醒的原文、精確邊界與主對話讀法；手算答案獨立於判定器。"""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import math
from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path


ITU = "ITU-R BS.1116-3 (02/2015)"
EBU = "EBU Tech 3276, 2nd edition (1998)"
ITU_BAND = (
    "The following dimension ratios should be observed to ensure a reasonably uniform distribution "
    "of the low-frequency eigentones of the room: 1.1 w / h ≤ l / h ≤ 4.5 w / h – 4 "
    "where: l : length w : width h : height."
)
EBU_BAND = (
    "To ensure a reasonably uniform distribution of the low–frequency eigentones, the proportions of the room "
    "must lie within controlled limits. The following limits for the length to height and the width to height "
    "ratios should be observed: 1.1w/h ≤ l/h ≤ 4.5 w/h – 4 l <3 h w <3 h "
    "where: l = length (larger dimension of floor plan, irrespective of orientation) "
    "w = width (shorter dimension of floor plan, irrespective of orientation) h = height."
)
ITU_BELOW = "Additionally, the conditions l / h < 3 and w / h < 3 should apply."
EBU_BELOW = "l <3 h w <3 h"
EBU_INTEGER = "In addition, ratios of l, w and h which are within ±5% of integer values should also be avoided."


def test_registry_preserves_sources_and_exact_decimal_thresholds() -> None:
    from aosr.config.room_proportions import load_room_proportions
    path = config_path("room_proportions.toml")
    with path.open("rb") as source:
        raw = tomllib.load(source)
    registry = load_room_proportions(path)
    expected = {
        "band": ((ITU, EBU), ("§8.2.2.3", "Appendix 2 §2"), (15, 10), (ITU_BAND, EBU_BAND)),
        "below": ((ITU, EBU), ("§8.2.2.3", "Appendix 2 §2"), (15, 10), (ITU_BELOW, EBU_BELOW)),
        "near_integer": ((EBU,), ("Appendix 2 §2",), (10,), (EBU_INTEGER,)),
    }
    assert tuple(entry.kind for entry in registry.entries) == tuple(expected)
    for entry, document in zip(registry.entries, raw["entry"], strict=True):
        assert (entry.standard, entry.clause, entry.pdf_page, entry.quote) == expected[entry.kind]
        assert "主對話讀法" in entry.description
        actual = {name: getattr(entry.thresholds, name) for name in document["thresholds"]}
        assert actual == {name: Fraction(value) for name, value in document["thresholds"].items()}
        assert all(isinstance(value, str) for value in document["thresholds"].values())
        assert all(isinstance(value, Fraction) for value in actual.values())
    with pytest.raises(ValidationError):
        setattr(registry.entries[0], "kind", "other")


@pytest.mark.parametrize("damage", [
    "unknown_kind", "float_threshold", "fraction_text", "negative_threshold", "zero_factor",
    "wide_integer_band", "missing_threshold", "extra_threshold", "duplicate_id", "empty_quote",
    "unpaired_sources", "duplicate_source", "unknown_standard", "empty_entries", "extra_field",
])
def test_registry_rejects_malformed_rules(tmp_path: Path, damage: str) -> None:
    from aosr.config.room_proportions import RoomProportions, load_room_proportions
    with config_path("room_proportions.toml").open("rb") as source:
        raw = tomllib.load(source)
    band = next(entry for entry in raw["entry"] if entry["kind"] == "band")
    integer = next(entry for entry in raw["entry"] if entry["kind"] == "near_integer")
    if damage == "unknown_kind":
        band["kind"] = "unknown"
    elif damage == "float_threshold":
        band["thresholds"]["lower_factor"] = float(band["thresholds"]["lower_factor"])
    elif damage == "fraction_text":
        band["thresholds"]["lower_factor"] = "11/10"
    elif damage == "negative_threshold":
        band["thresholds"]["upper_offset"] = "-1"
    elif damage == "zero_factor":
        band["thresholds"]["lower_factor"] = "0"
    elif damage == "wide_integer_band":
        integer["thresholds"]["relative_width"] = "1"
    elif damage == "missing_threshold":
        del band["thresholds"]["upper_factor"]
    elif damage == "extra_threshold":
        band["thresholds"]["upper"] = "1"
    elif damage == "duplicate_id":
        integer["id"] = band["id"]
    elif damage == "empty_quote":
        band["quote"][0] = " "
    elif damage == "unpaired_sources":
        band["pdf_page"].pop()
    elif damage == "duplicate_source":
        band["standard"][1] = band["standard"][0]
    elif damage == "unknown_standard":
        band["standard"][0] = "unknown"
    elif damage == "empty_entries":
        raw["entry"] = []
    else:
        band["extra"] = "unknown"
    with pytest.raises(ValidationError):
        RoomProportions.model_validate(raw)
    broken = tmp_path / "broken.toml"
    broken.write_text("entry = [", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_room_proportions(broken)


def test_reference_room_is_on_closed_band_but_has_integer_height_ratio() -> None:
    from aosr.config.room_proportions import load_room_proportions
    from aosr.search.room_proportions import room_proportion_notice, room_proportion_text
    rules = load_room_proportions(config_path("room_proportions.toml"))
    band = next(entry.thresholds for entry in rules.entries if entry.kind == "band")
    # l/w=3/2，l/h=2，w/h=4/3；上緣正好等於 l/h，含端點所以比例帶不提醒。
    assert band.upper_factor is not None and band.upper_offset is not None
    assert Fraction(6, 3) == band.upper_factor * Fraction(4, 3) - band.upper_offset
    notice = room_proportion_notice(6., 4., 3., rules)
    assert [(hit.rule.kind, hit.ratio, hit.value, hit.near_integer) for hit in notice.hits] == [
        ("near_integer", "l/h", Fraction(2), 2)]
    assert room_proportion_text(notice) == (
        "這個房間的比例可能產生模態重合，請仔細檢查低頻結果；"
        "長高比 2.000000 在整數 2 的 ±5% 內（EBU Tech 3276）")
    with pytest.raises(FrozenInstanceError):
        setattr(notice, "hits", ())
    with pytest.raises(FrozenInstanceError):
        setattr(notice.hits[0], "value", Fraction(0))


@pytest.mark.parametrize("dimensions,kinds", [
    ((6.5, 4.5, 3.), ()),
    ((4.5, 4.25, 3.), ("band",)),
    ((9., 6., 3.), ("below", "near_integer")),
    ((6., 6., 3.), ("band", "near_integer")),
    ((5.5, 5., 3.), ()),
    ((math.nextafter(5.5, -math.inf), 5., 3.), ("band",)),
])
def test_hand_calculated_rooms(dimensions: tuple[float, float, float], kinds: tuple[str, ...]) -> None:
    from aosr.config.room_proportions import load_room_proportions
    from aosr.search.room_proportions import room_proportion_notice, room_proportion_text
    # 全守房：l/w=13/9、l/h=13/6、w/h=3/2；帶內，三比皆遠離正整數。
    # 下緣外：l/h=3/2，小於 lower_factor*(17/12)，三比仍都遠離整數。
    # 上限房：l/h=3，嚴格小於的條件不成立；正方形房 l/w=1 也要提醒。
    # 下緣房：l/h=11/6 正好等於 lower_factor*(5/3)，含端點不提醒；l/w=11/10、l/h、w/h 都離整數夠遠。
    # 長邊少一個最小位就掉出下緣，只提醒比例帶。
    rules = load_room_proportions(config_path("room_proportions.toml"))
    notice = room_proportion_notice(*dimensions, rules)
    assert {hit.rule.kind for hit in notice.hits} == set(kinds)
    assert room_proportion_notice(dimensions[1], dimensions[0], dimensions[2], rules) == notice
    if not kinds:
        assert room_proportion_text(notice) == "無"
    text = room_proportion_text(notice)
    height_ratio = Fraction(max(dimensions[:2])) / Fraction(dimensions[2])
    # 門檻照表上的十進位字串印（不是 11/10 這種分數），從表的原文字串組答案。
    with config_path("room_proportions.toml").open("rb") as source:
        raw = {entry["kind"]: entry["thresholds"] for entry in tomllib.load(source)["entry"]}
    if "band" in kinds:
        band = raw["band"]
        assert (f"長高比 {float(height_ratio):.6f} 在建議比例帶 {band['lower_factor']}·w/h ≤ l/h ≤ "
                f"{band['upper_factor']}·w/h − {band['upper_offset']} 之外（ITU-R BS.1116-3；EBU Tech 3276）") in text
    if "below" in kinds:
        assert (f"長高比 {float(height_ratio):.6f} 達到建議上限 {raw['below']['upper']}（建議小於此值）"
                "（ITU-R BS.1116-3；EBU Tech 3276）") in text
    if dimensions[0] == dimensions[1]:
        assert any(hit.ratio == "l/w" and hit.near_integer == 1 for hit in notice.hits)


def test_threshold_text_keeps_table_decimals() -> None:
    from aosr.search.room_proportions import _decimal
    # 審查反例：浮點的 g 格式會把 1.234567 捨成 1.23457、把 0.0000001 印成 1e-07。
    for text in ("1.1", "4.5", "4", "3", "0.05", "1.234567", "0.0000001", "12345678.125"):
        assert _decimal(Fraction(text)) == text


@pytest.mark.parametrize("length,integers", [
    (5.25, (2,)), (math.nextafter(5.25, math.inf), ()), (5.2, (2,)),
    (4.75, (2,)), (math.nextafter(4.75, -math.inf), ()),
])
def test_integer_edge_and_relative_reading(length: float, integers: tuple[int, ...]) -> None:
    from aosr.config.room_proportions import load_room_proportions
    from aosr.search.room_proportions import room_proportion_notice
    rules = load_room_proportions(config_path("room_proportions.toml"))
    width = next(entry.thresholds.relative_width for entry in rules.entries if entry.kind == "near_integer")
    assert width is not None
    ratio = Fraction(length) / Fraction(2.5)
    if length == 5.25:
        assert abs(ratio - 2) == width * 2  # 二進位精確的兩邊，恰在相對帶邊上。
    elif length == 4.75:
        assert abs(ratio - 2) == width * 2
    elif length > 5.25 or length < 4.75:
        assert abs(ratio - 2) > width * 2
    else:
        assert width < abs(ratio - 2) < width * 2  # 相對讀法提醒，絕對讀法不提醒。
    notice = room_proportion_notice(length, 2.5, 3., rules)
    assert tuple(hit.near_integer for hit in notice.hits if hit.ratio == "l/w") == integers


def test_both_height_ratios_use_strict_upper_limit() -> None:
    from aosr.config.room_proportions import load_room_proportions
    from aosr.search.room_proportions import room_proportion_notice
    rules = load_room_proportions(config_path("room_proportions.toml"))
    upper = next(entry.thresholds.upper for entry in rules.entries if entry.kind == "below")
    assert upper is not None
    notice = room_proportion_notice(9., 9., 3., rules)
    assert [(hit.ratio, hit.value) for hit in notice.hits if hit.rule.kind == "below"] == [
        ("l/h", upper), ("w/h", upper)]


def test_every_positive_integer_in_relative_interval_is_reported() -> None:
    from aosr.config.room_proportions import load_room_proportions
    from aosr.search.room_proportions import room_proportion_notice
    rules = load_room_proportions(config_path("room_proportions.toml"))
    width = next(entry.thresholds.relative_width for entry in rules.entries if entry.kind == "near_integer")
    assert width is not None
    ratio = Fraction(61, 2)
    expected = tuple(n for n in range(1, 62) if abs(ratio - n) <= width * n)
    assert expected == (30, 31, 32)  # 只取最近整數會漏報。
    notice = room_proportion_notice(61., 2., 3., rules)
    assert tuple(hit.near_integer for hit in notice.hits if hit.ratio == "l/w") == expected


def test_rational_verdict_ignores_float_operation_order() -> None:
    from aosr.config.room_proportions import load_room_proportions
    from aosr.search.room_proportions import room_proportion_notice
    rules = load_room_proportions(config_path("room_proportions.toml"))
    band = next(entry.thresholds for entry in rules.entries if entry.kind == "band")
    assert band.upper_factor is not None and band.upper_offset is not None
    l, w, h = map(Fraction, (5.875, 3.75, 2.75))
    assert band.upper_factor * (w / h) - band.upper_offset == (band.upper_factor * w - band.upper_offset * h) / h == l / h
    a = float(band.upper_factor) * (float(w) / float(h)) - float(band.upper_offset)
    b = (float(band.upper_factor) * float(w) - float(band.upper_offset) * float(h)) / float(h)
    # 兩種浮點算式會給相反判決；提醒只照上面獨立核過的精確上緣。
    assert a < float(l / h) <= b
    assert all(hit.rule.kind != "band" for hit in room_proportion_notice(float(l), float(w), float(h), rules).hits)


def test_float_dimensions_keep_their_binary_value() -> None:
    from aosr.config.room_proportions import load_room_proportions
    from aosr.search.room_proportions import room_proportion_notice
    rules = load_room_proportions(config_path("room_proportions.toml"))
    width = next(entry.thresholds.relative_width for entry in rules.entries if entry.kind == "near_integer")
    assert width is not None
    # 浮點 4.2 比十進位字面值大，這組在相對帶外；把尺寸先轉字串會錯判為帶內。
    assert Fraction(4.2) / 2 > (1 + width) * 2 == Fraction(str(4.2)) / 2
    assert all(hit.ratio != "l/w" for hit in room_proportion_notice(4.2, 2., 3., rules).hits)


@pytest.mark.parametrize("dimensions", [(0., 4., 3.), (-1., 4., 3.), (6., math.inf, 3.), (6., 4., math.nan)])
def test_invalid_dimensions_are_rejected(dimensions: tuple[float, float, float]) -> None:
    from aosr.config.room_proportions import load_room_proportions
    from aosr.search.room_proportions import room_proportion_notice
    with pytest.raises(ValueError):
        room_proportion_notice(*dimensions, load_room_proportions(config_path("room_proportions.toml")))
