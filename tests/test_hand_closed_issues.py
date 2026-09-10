"""規矩卡 issues-closed-only-by-merged-pr 判斷那一半的測試：全部餵假回應，不上網。

為什麼可以這樣測：那支檢查把「問 GitHub」與「判哪幾張是人手關的」切成兩半，判斷那一半
（:func:`read_page`、:func:`read_ticket`、:func:`hand_closed`）吃的是一段已經解好的圖查詢
回應，不碰網路也不碰子程序。上網那一半（:func:`read_closed_tickets`）走的是狀態頁那一層的
`collect.Shell`，後設測試六回合會拿真的樹與樣本樹去跑它。

假的 PR（合併請求）節點組裝共用 `tests/test_build_status.py` 那一份（`pull_node`），
不抄第二份——兩份假資料會各自漂。

斷言刻意不鎖死數量（規矩卡 assertions-not-pinned-to-counts）：比的是「哪幾張被點名」、
「兩句話一不一樣」，不是「必須剛好幾筆」。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import pytest

from governance.checks import issues_closed_only_by_merged_pr as card
from governance.exit_codes import ToolBroken
from governance.status import collect
from tests.test_build_status import FAKE_LATE_PULL, pull_node

# 假的主線分支名。刻意不寫 "main"：主線叫什麼是卡上登記的一格，判斷那一半只認參數傳進來的
# 那個值——寫死 "main" 的版本會讓「樣本把分支名改掉」那一招整條失效。
FAKE_MAIN = "fixture-main"
FAKE_SIDE_BRANCH = "fixture-side"
FAKE_NUMBER = 401
FAKE_TITLE = "假的：一張關掉的票"
# 放行到期日兩種：一個在「今天」之後、一個之前。「今天」也由測試自己給，不問系統時鐘。
TODAY = date(2026, 9, 10)
LIVE = "2026-12-08"
DEAD = "2026-01-01"


def issue_row(
    number: int,
    title: str,
    nodes: Sequence[Mapping[str, object]],
    *,
    more: bool = False,
) -> dict[str, object]:
    """圖查詢回的一張關掉的票（只留這一層會讀的欄位），底下掛著那幾個關它的 PR。"""
    return {
        "number": number,
        "title": title,
        "url": f"https://x.invalid/{number}",
        collect.CLOSED_BY_FIELD: {
            "pageInfo": {"hasNextPage": more, "endCursor": "假的游標" if more else ""},
            "nodes": [dict(node) for node in nodes],
        },
    }


def issues_page(
    rows: Sequence[Mapping[str, object]], *, more: bool = False, cursor: str = ""
) -> dict[str, object]:
    """一頁圖查詢回應：關掉的票那一段。"""
    return {
        "data": {
            "repository": {
                "issues": {
                    "pageInfo": {"hasNextPage": more, "endCursor": cursor},
                    "nodes": [dict(row) for row in rows],
                }
            }
        }
    }


def tickets_of(body: Mapping[str, object], main_branch: str = FAKE_MAIN) -> list[card.Ticket]:
    """把一頁回應判成那幾張票（游標另外測）。"""
    tickets, _ = card.read_page(body, main_branch, "假的：問關掉的票")
    return tickets


# ── 落地判斷：哪一張算「有 PR 做掉它」────────────────────────────────────────


def test_a_ticket_closed_by_a_pull_that_landed_on_main_is_not_hand_closed() -> None:
    """關它的 PR 合進了卡上登記的主線：不是人手關的，一句 HIT 都不出。"""
    body = issues_page([issue_row(FAKE_NUMBER, FAKE_TITLE, [pull_node(FAKE_LATE_PULL, base=FAKE_MAIN)])])
    assert card.hand_closed(tickets_of(body), {}) == []


def test_a_pull_that_landed_on_another_branch_does_not_count() -> None:
    """合了，可是合去別的分支：不算落地，那張票照樣被點名。"""
    body = issues_page(
        [issue_row(FAKE_NUMBER, FAKE_TITLE, [pull_node(FAKE_LATE_PULL, base=FAKE_SIDE_BRANCH)])]
    )
    hits = card.hand_closed(tickets_of(body), {})
    assert [f"#{FAKE_NUMBER}" in hit for hit in hits] == [True]


def test_the_main_branch_comes_from_the_card_not_from_the_code() -> None:
    """同一段回應、換一個主線分支名，答案就不一樣——主線叫什麼不是寫死在檢查程式裡的。

    這一針就是必紅樣本那一招（把卡上的主線分支名改成一條永遠不存在的分支）的單元版：
    分支名一旦被寫死，那份樣本會安靜地回綠。
    """
    body = issues_page([issue_row(FAKE_NUMBER, FAKE_TITLE, [pull_node(FAKE_LATE_PULL, base=FAKE_MAIN)])])
    assert card.hand_closed(tickets_of(body, FAKE_MAIN), {}) == []
    assert card.hand_closed(tickets_of(body, "aosr-fixture-branch-that-never-exists"), {})


def test_no_closing_pull_at_all_reads_differently_from_one_that_never_landed() -> None:
    """兩種紅要分得開：根本沒有 PR（人手關的），跟有 PR 但沒有一個合進主線。

    混成同一句，看紀錄的人就分不出下一步該催誰——一種是有人開了還沒合，一種是根本沒人做。
    """
    bare = card.hand_closed(tickets_of(issues_page([issue_row(FAKE_NUMBER, FAKE_TITLE, [])])), {})
    hanging = card.hand_closed(
        tickets_of(
            issues_page(
                [issue_row(FAKE_NUMBER, FAKE_TITLE, [pull_node(FAKE_LATE_PULL, merged=False)])]
            )
        ),
        {},
    )
    assert bare and hanging
    assert bare != hanging


def test_hits_name_the_tickets_and_carry_their_links() -> None:
    """HIT 那幾行點名的就是沒落地的那幾張，而且帶得出它的網址（下一步要去重開它）。"""
    landed = issue_row(11, "落地了", [pull_node(FAKE_LATE_PULL, base=FAKE_MAIN)])
    hand = issue_row(22, "人手關的", [])
    hits = card.hand_closed(tickets_of(issues_page([landed, hand])), {})
    named = {"#22" in hit for hit in hits}
    assert named == {True}
    assert [str(hand["url"]) in hit for hit in hits] == [True]


# ── 具名放行：有牙，過期就沒有 ────────────────────────────────────────────────


def test_a_named_exemption_lets_one_ticket_through() -> None:
    """具名放過的那一張不算違規，別張照咬。"""
    rows = [issue_row(11, "放過這張", []), issue_row(22, "這張不放", [])]
    hits = card.hand_closed(tickets_of(issues_page(rows)), {11: "假的：說得出為什麼"})
    assert [f"#{11}" in hit for hit in hits] == [False]
    assert [f"#{22}" in hit for hit in hits] == [True]


def test_an_expired_named_exemption_is_no_exemption() -> None:
    """過期的具名放行讀出來就不在名單裡——放行不是寫上去就永久有效。"""
    settings = {
        "allow": [
            {"network": "假的", "reason": "假的：准上網", "expires": LIVE},
            {"issue": 83, "reason": "假的：過期的具名放行", "expires": DEAD},
            {"issue": 84, "reason": "假的：還沒過期的具名放行", "expires": LIVE},
        ]
    }
    excused = card.excused_tickets(settings, "假的卡", TODAY)
    assert sorted(excused) == [84]


def test_a_named_exemption_without_a_reason_stops_the_run() -> None:
    """放行少一格（沒有理由）不是「當它不存在」，是這一跑不算數。"""
    settings = {"allow": [{"issue": 83, "expires": LIVE}]}
    with pytest.raises(ToolBroken):
        card.excused_tickets(settings, "假的卡", TODAY)


def test_an_issue_number_that_is_not_an_integer_stops_the_run() -> None:
    """票號寫成字串（或 true）讀不成一張票——形狀壞掉就回 2，不安靜放過。"""
    settings = {"allow": [{"issue": "83", "reason": "假的", "expires": LIVE}]}
    with pytest.raises(ToolBroken):
        card.excused_tickets(settings, "假的卡", TODAY)


# ── 准上網那一筆放行：缺席或過期就不上網 ──────────────────────────────────────


def test_no_network_exemption_means_no_network() -> None:
    """沒有登記的放行就不准上網：ToolBroken（外殼翻成離開碼 2）。"""
    with pytest.raises(ToolBroken):
        card.network_allowed({"allow": [{"issue": 83, "reason": "假的", "expires": LIVE}]}, "假的卡", TODAY)


def test_an_expired_network_exemption_means_no_network() -> None:
    """准上網那一筆過期了也不准上網——放行有牙，不是裝飾。"""
    settings = {"allow": [{"network": "假的", "reason": "假的：准上網", "expires": DEAD}]}
    with pytest.raises(ToolBroken):
        card.network_allowed(settings, "假的卡", TODAY)


def test_a_live_network_exemption_hands_back_its_reason() -> None:
    """還沒過期的那一筆讀得出來，理由要原樣交回去（輸出裡要寫得出憑什麼上網）。"""
    reason = "假的：決策紙說可以"
    settings = {"allow": [{"network": "假的", "reason": reason, "expires": LIVE}]}
    assert card.network_allowed(settings, "假的卡", TODAY) == reason


# ── 翻頁與看不懂的形狀 ────────────────────────────────────────────────────────


def test_a_full_page_hands_back_the_cursor_to_keep_going() -> None:
    """票那一層還有下一頁：把游標交回去（只問第一頁等於「多出來的那些不存在」）。"""
    cursor = "假的游標"
    _, nxt = card.read_page(
        issues_page([issue_row(FAKE_NUMBER, FAKE_TITLE, [])], more=True, cursor=cursor),
        FAKE_MAIN,
        "假的：問關掉的票",
    )
    assert nxt == cursor


def test_the_last_page_hands_back_no_cursor() -> None:
    """翻完了就不再翻：游標是空的。"""
    _, nxt = card.read_page(
        issues_page([issue_row(FAKE_NUMBER, FAKE_TITLE, [])]), FAKE_MAIN, "假的：問關掉的票"
    )
    assert nxt == ""


def test_a_ticket_whose_closing_pulls_do_not_fit_one_page_stops_the_run() -> None:
    """一張票掛著的關票 PR 一頁看不完：回 2，不猜——看不完就不出結論。"""
    body = issues_page([issue_row(FAKE_NUMBER, FAKE_TITLE, [], more=True)])
    with pytest.raises(ToolBroken):
        card.read_page(body, FAKE_MAIN, "假的：問關掉的票")


def test_a_shape_we_do_not_understand_stops_the_run() -> None:
    """GitHub 回一段看不懂的（票那一段是 null）：ToolBroken，不是「零張人手關的」。"""
    with pytest.raises(ToolBroken):
        card.read_page({"data": {"repository": {"issues": None}}}, FAKE_MAIN, "假的：問關掉的票")


def test_a_ticket_without_a_number_stops_the_run() -> None:
    """沒有票號的一筆讀不成一張票——點不出名字的違規等於沒有違規，回 2。"""
    row = issue_row(FAKE_NUMBER, FAKE_TITLE, [])
    row.pop("number")
    with pytest.raises(ToolBroken):
        card.read_page(issues_page([row]), FAKE_MAIN, "假的：問關掉的票")


# ── 跟狀態頁那一層共用同一份判準 ──────────────────────────────────────────────


def test_the_landing_rule_is_the_one_shared_with_the_status_page() -> None:
    """「這個 PR 算不算落地」只有一支判準，狀態頁那一格與這張卡共用它。

    這一針不是形式：兩邊各寫一份的時候，改了一邊另一邊會安靜地繼續用舊判準。
    """
    node = pull_node(FAKE_LATE_PULL, base=FAKE_MAIN)
    assert collect.landed_pull(node, FAKE_MAIN) == FAKE_LATE_PULL
    assert collect.landed_pull(node, FAKE_SIDE_BRANCH) is None
