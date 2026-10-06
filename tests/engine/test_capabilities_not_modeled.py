"""給人看的模型缺項與人工確認清單：逐字內容、預設值與拒收規則。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import CapabilityEntry, CapabilityTable, load_capabilities
from aosr.config.paths import config_path
from aosr.reporting.display import LOW_FREQUENCY_DECAY_NOTE


DISPLAY = {
    "three_lane_report": ((
        "家具、桌面、沙發等大型物件（房間是空的六面盒）",
        "喇叭箱體本身的反射（兩支喇叭互相反射也不算）",
        "空氣吸收", "非長方形房間",
        "隨頻率改變的牆面材料（這一版每面一個與頻率無關的阻抗）",
    ), ()),
    "reflections_and_echo_evaluator": (("家具與桌面的反射", "空氣吸收"), (
        "桌面、控台、螢幕造成的早期反射要現場另外確認",
    )),
    "scheme_pipeline": ((
        LOW_FREQUENCY_DECAY_NOTE, "製作用途與多人座位尚未評估",
    ), ("螢幕視線與觀看角度", "工作姿勢與桌面高度", "門、窗、走道是否被擋")),
}


@pytest.fixture
def table() -> CapabilityTable:
    return load_capabilities(config_path("capabilities.toml"))


def test_declared_display_lists_and_empty_defaults(table: CapabilityTable) -> None:
    for entry in table.entry:
        expected = DISPLAY.get(entry.name, ((), ()))
        assert (entry.not_modeled, entry.manual_checks) == expected


@pytest.mark.parametrize("field", ["not_modeled", "manual_checks"])
@pytest.mark.parametrize("items", [("",), ("   ",), ("同一項", "同一項"), ("同一項", " 同一項 ")])
def test_display_items_reject_empty_or_duplicate(
    table: CapabilityTable, field: str, items: tuple[str, ...],
) -> None:
    document = table.for_entry("three_lane_report").model_dump()
    document[field] = items
    with pytest.raises(ValidationError, match=field):
        CapabilityEntry.model_validate(document)


@pytest.mark.parametrize("field", ["not_modeled", "manual_checks"])
def test_display_items_are_stripped(table: CapabilityTable, field: str) -> None:
    document = table.for_entry("three_lane_report").model_dump()
    document[field] = ["  第一項 ", "\t第二項\n"]
    entry = CapabilityEntry.model_validate(document)
    assert getattr(entry, field) == ("第一項", "第二項")


def test_unknown_fields_remain_forbidden(table: CapabilityTable) -> None:
    document = table.for_entry("three_lane_report").model_dump()
    document["unregistered"] = ()
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CapabilityEntry.model_validate(document)
