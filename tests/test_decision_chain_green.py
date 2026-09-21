"""合法的取代鏈不准被誤咬：餵一棵綠樹下去，必須回 0、而且一筆違規都沒有。

後設測試的六回合裡沒有「餵一棵樹必須回 0」這一格：第 1 回的乾淨樹是真 repo，而真 repo
的決策紙一條取代鏈都沒有（立卡當天；2026-09-15 起被取代的紙住 docs/archive/）；控制樣本那一回合的正確答案是 1（已知會咬的最小輸入），
所以「走鏈那一段不會誤咬合法的多跳鏈」在那六回合裡證不出來。

找碴席（blueprint/cards-38.json 的 critic_v2）點的正是這個洞：逐跳要求「還在生效」會誤咬
合法的三跳鏈，而第一輪的對照組沒有覆蓋多跳鏈，所以那個誤咬不會被測出來。這一支就是那個
對照組：綠樹裡有一條三跳鏈（鏈尾停在一份 accepted，中途每一跳都是 superseded、雙向指標
都對），加一張沒有取代關係的紙。

跑法刻意跟後設測試共用同一支 ``_run``（同一份環境衛生：不帶 PYTHONPATH、不寫 .pyc），
斷言離開碼與那一行收據——回 0 而印不出收據，等於「什麼都沒掃到卻說乾淨」。
"""
from __future__ import annotations

from pathlib import Path

from governance.checks.decision_paper_structure import (
    FIELD_STATUS,
    FIELD_SUPERSEDED_BY,
    FIELD_SUPERSEDES,
    _card_settings,
    _components,
    _frontmatter,
    _link_problems,
)
from governance.exit_codes import CLEAN
from governance.loader import Card, load_all_cards, setting_text
from tests.test_fixture_runner import _assert_report_line, _run, _tail

REPO = Path(__file__).resolve().parents[1]

CARD_ID = "decision-paper-structure"
GREEN_TREE = "governance/fixtures/decision-paper-structure-green/legal-multi-hop-chain"
REPORT_PREFIX = "scan_root="
NO_HITS = "hits=0"


def _card() -> Card:
    """卡是從 governance/rules/ 讀出來的，不是這裡抄一份。"""
    cards = [c for c in load_all_cards(REPO) if c.id == CARD_ID]
    assert cards, f"governance/rules/ 裡找不到 id 是 {CARD_ID} 的卡——這一支就沒有對象可測"
    return cards[0]


def test_legal_supersede_chain_is_green() -> None:
    """綠樹（合法的三跳取代鏈）餵下去必須回 0，而且收據行的 hits 是零。"""
    tree = REPO / GREEN_TREE
    assert tree.is_dir(), f"{GREEN_TREE} 不在——沒有樹可餵，這一支證不了任何事"
    proc = _run(_card(), tree)
    _assert_report_line(proc)
    assert proc.returncode == CLEAN, (
        f"合法的取代鏈被咬了（回 {proc.returncode}，應為 0）：{_tail(proc)}"
    )
    report = [ln for ln in proc.stdout.splitlines() if ln.startswith(REPORT_PREFIX)][-1]
    assert report.split()[-1] == NO_HITS, f"回 0 但收據行說有違規：{report!r}"


def test_real_tree_keeps_superseded_papers_only_in_archive() -> None:
    """真樹：docs/decisions/ 裡沒有標 superseded 的紙，docs/archive/ 裡全部都是（票 #313 的第 8 條，直接數）。"""
    settings = _card_settings(REPO, [p for p in (REPO / "governance" / "rules").glob("*.toml")])
    live_dir = REPO / setting_text(settings, "decisions_prefix").rstrip("/")
    archive_dir = REPO / setting_text(settings, "archive_prefix").rstrip("/")
    superseded = setting_text(settings, "status_superseded")
    live = sorted(live_dir.glob("*.md"))
    archived = sorted(archive_dir.glob("*.md"))
    assert live, "活的目錄一份決策紙都沒有"
    assert archived, "封存區一份都沒有——第二條斷言會變成永遠成立，證不了東西"

    def status(path: Path) -> str:
        front = _frontmatter(path.read_text(encoding="utf-8"))
        return (front or {}).get(FIELD_STATUS, "").strip()

    assert [p.name for p in live if status(p) == superseded] == []
    assert [p.name for p in archived if status(p) != superseded] == []


NEW = "merged.md"
OLD_A = "old-a.md"
OLD_B = "old-b.md"


def _merged_fronts(listed: str, b_points_to: str) -> dict[str, dict[str, str]]:
    """一張新紙加兩張舊紙：新紙的取代誰那一格、第二張舊紙指回誰，由呼叫端給。"""
    return {
        NEW: {FIELD_SUPERSEDES: listed, FIELD_SUPERSEDED_BY: ""},
        OLD_A: {FIELD_SUPERSEDES: "", FIELD_SUPERSEDED_BY: NEW},
        OLD_B: {FIELD_SUPERSEDES: "", FIELD_SUPERSEDED_BY: b_points_to},
    }


def _all_link_problems(fronts: dict[str, dict[str, str]]) -> list[str]:
    return [line for name, front in fronts.items() for line in _link_problems(name, front, fronts)]


def test_one_paper_superseding_two_is_symmetric_and_one_group() -> None:
    """票 #412：一張取代兩張、兩張都指回來——第 5 條一筆都不咬，三張連成同一群。"""
    fronts = _merged_fronts(f"{OLD_A}, {OLD_B}", NEW)
    assert _all_link_problems(fronts) == []
    assert _components(fronts) == [sorted(fronts)]


def test_second_listed_paper_not_pointing_back_is_named() -> None:
    """清單第二項沒指回來：被點名的是第二張，第一張不准被連坐。"""
    problems = _all_link_problems(_merged_fronts(f"{OLD_A}, {OLD_B}", ""))
    assert problems, "清單第二項沒指回來卻沒有違規——只查了清單第一項"
    assert all(OLD_B in line for line in problems)
    assert not any(OLD_A in line for line in problems)


def test_old_paper_missing_from_successor_list_is_named() -> None:
    """舊紙說被新紙接手、新紙的清單沒列它：反方向要在清單裡找。"""
    problems = _all_link_problems(_merged_fronts(OLD_A, NEW))
    assert problems, "舊紙指過去、新紙沒列它，卻沒有違規"
    assert all(line.startswith(OLD_B) for line in problems)


def test_malformed_list_reports_empty_item_and_repeat() -> None:
    """清單有空項、同一張列兩次：各咬一筆，指標本身對得上所以沒有別的違規。"""
    fronts = _merged_fronts(f"{OLD_A}, {OLD_B}, {OLD_A},", NEW)
    problems = _all_link_problems(fronts)
    assert any("空的一項" in line for line in problems)
    assert any("不只一次" in line and OLD_A in line for line in problems)
    assert not any("沒指回來" in line or "沒把這一張列進去" in line for line in problems)


def test_listed_paper_joins_group_even_without_pointing_back() -> None:
    """清單第二項沒指回來時，它仍然靠新紙那一頭的邊連進同一群——第 7 條才咬得到「兩份生效」。"""
    fronts = _merged_fronts(f"{OLD_A}, {OLD_B}", "")
    assert _components(fronts) == [sorted(fronts)]
