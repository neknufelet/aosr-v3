"""卡頂層 ``[[by_design]]``（本來就該這樣的規矩准許，#1014）的共用形狀。

資料跟放行差在沒有到期日，所以載入器與檢查程式共用的那一關只守兩件事：每一筆要有理由、
不准帶 ``expires``（會到期的那一種寫進 ``[[settings.allow]]``，由 exemptions-need-expiry 守）。
各卡自己的欄位與型別由那支檢查讀的時候判；這裡用一組假的欄位證明那一關真的會擋。
"""
from __future__ import annotations

import pytest

from governance.exit_codes import ToolBroken
from governance.loader import by_design_entries, by_design_problems, settings_exemptions

KEYS = {"path": "text", "writes": "count", "cited_in": "texts", "prints": "counts"}
GOOD = {
    "path": "a.md",
    "writes": 2,
    "cited_in": ["b.md"],
    "prints": {"main": 3, "Cli.show": 1},
    "reason": "本來就該這樣：假的理由。",
}


def card(*entries: dict[str, object]) -> dict[str, object]:
    return {"id": "sample", "settings": {"names": ["x"]}, "by_design": list(entries)}


def test_absent_means_no_entries() -> None:
    assert by_design_entries({"id": "sample"}, KEYS, "卡") == []
    assert by_design_problems({"id": "sample"}) == []


def test_good_entry_is_read_back() -> None:
    assert by_design_entries(card(GOOD), KEYS, "卡") == [GOOD]
    assert by_design_problems(card(GOOD)) == []


def test_top_level_data_is_not_an_exemption() -> None:
    # 判準是「[settings] 底下的表陣列才是放行」：頂層的資料不進到期那一關。
    assert settings_exemptions(card(GOOD)) == []


@pytest.mark.parametrize(
    ("broken", "said"),
    [
        ({**GOOD, "expires": "2099-12-31"}, "expires"),
        ({k: v for k, v in GOOD.items() if k != "reason"}, "reason"),
        ({**GOOD, "reason": "  "}, "reason"),
    ],
)
def test_shared_rules_block_both_loader_and_check(broken: dict[str, object], said: str) -> None:
    assert any(said in problem for problem in by_design_problems(card(broken)))
    with pytest.raises(ToolBroken, match=said):
        by_design_entries(card(broken), KEYS, "卡")


@pytest.mark.parametrize(
    ("broken", "said"),
    [
        ({k: v for k, v in GOOD.items() if k != "writes"}, "缺 writes"),
        ({**GOOD, "writes": 0}, "writes 必須是正整數"),
        ({**GOOD, "writes": True}, "writes 必須是正整數"),
        ({**GOOD, "path": ""}, "path 必須是非空字串"),
        ({**GOOD, "cited_in": []}, "cited_in 必須是非空字串的清單"),
        ({**GOOD, "line": 12}, "不認識的鍵"),
        ({**GOOD, "prints": {}}, "prints 必須是「名字→正整數」的非空表"),
        ({**GOOD, "prints": {"main": 0}}, "prints 必須是「名字→正整數」的非空表"),
        ({**GOOD, "prints": {"main": "3"}}, "prints 必須是「名字→正整數」的非空表"),
        ({**GOOD, "prints": ["main"]}, "prints 必須是「名字→正整數」的非空表"),
    ],
)
def test_card_specific_fields_are_checked_by_the_reader(broken: dict[str, object], said: str) -> None:
    with pytest.raises(ToolBroken, match=said):
        by_design_entries(card(broken), KEYS, "卡")


@pytest.mark.parametrize("raw", [[], {"path": "a.md"}, ["a.md"]])
def test_written_but_not_a_table_array_is_broken(raw: object) -> None:
    data = {"id": "sample", "by_design": raw}
    assert by_design_problems(data)
    with pytest.raises(ToolBroken):
        by_design_entries(data, KEYS, "卡")
