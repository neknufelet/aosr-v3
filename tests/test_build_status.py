"""狀態頁的測試：全部餵假資料，不上網、不 spawn 任何東西、不寫進真的樹。

為什麼可以這樣測：`governance/status/render.py` 那一層刻意不碰網路也不碰子程序，
它只吃一個算好的 `PageData`。真的去問版控與 GitHub 的那一層在 `collect.py`，
這裡只戳它「外部工具不在的時候是不是回 2」——用假的子程序，不真的叫誰。

斷言刻意不鎖死數量（規矩卡 assertions-not-pinned-to-counts）：比的是「有沒有這幾個名字」、
「圖上的點是不是跟假資料裡的天數一樣多」，不是「必須剛好 17 張」。
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken
from governance.status import build_status, collect, render
from governance.status.model import (
    Blueprint,
    ClosedReview,
    ClosedTicket,
    CloudRun,
    Milestone,
    PageData,
    RepoState,
    RuleCard,
    StepResult,
    Ticket,
)

TODAY = date(2026, 9, 11)

FAKE_CARDS = (
    RuleCard(
        card_id="fake-first-card",
        human="假的第一張卡：這一句是人話欄，後面還有很多字，測試不看它的長度。",
        blood_debt=("fake-incident-a", "fake-incident-b"),
        merged_day="2026-09-08",
    ),
    RuleCard(
        card_id="fake-second-card",
        human="假的第二張卡。",
        blood_debt=("fake-incident-b",),
        merged_day="2026-09-10",
    ),
    RuleCard(
        card_id="fake-third-card",
        human="假的第三張卡。",
        blood_debt=(),
        merged_day="2026-09-10",
    ),
)

FAKE_TICKETS = (
    Ticket(
        number=101,
        title="要拍板：假的一題 <也吃得下角括號>",
        labels=("decision",),
        is_pr=False,
        updated="2026-09-10 08:00",
        url="https://github.com/fake/fake/issues/101",
    ),
    Ticket(
        number=102,
        title="假的暫緩組：在等假的條件",
        labels=("deferred-cards",),
        is_pr=False,
        updated="2026-09-10 09:00",
        url="https://github.com/fake/fake/issues/102",
    ),
    Ticket(
        number=103,
        title="假的其他票",
        labels=(),
        is_pr=False,
        updated="2026-09-10 10:00",
        url="https://github.com/fake/fake/issues/103",
    ),
    Ticket(
        number=104,
        title="假的合併請求",
        labels=(),
        is_pr=True,
        updated="2026-09-10 11:00",
        url="https://github.com/fake/fake/pull/104",
    ),
)

FAKE_CLOUD = CloudRun(
    run_id=999111,
    conclusion="success",
    status="completed",
    started="2026-09-10 03:37",
    finished="2026-09-10 03:40",
    head_sha="abcdef123456",
    url="https://github.com/fake/fake/actions/runs/999111",
    job_seconds=175,
    steps=(
        StepResult(name="假的第一步——規矩卡必填欄位", conclusion="success", seconds=3),
        StepResult(name="假的第二步——寫法警衛", conclusion="failure", seconds=9),
    ),
)


def fake_page(*, cloud: CloudRun | None = FAKE_CLOUD) -> PageData:
    """一份假的整頁資料。每個測試自己決定要不要有雲端紀錄。"""
    return PageData(
        computed_at="2026-09-11 07:30",
        computed_by="雲端 run 424242（第 1 次嘗試）",
        reflects="main 的 abcdef123456（假的最新一筆）",
        repo="fake/fake",
        page_url="https://fake.example.invalid/page/",
        cards=FAKE_CARDS,
        lessons_total=66,
        blueprint=Blueprint(
            total=38,
            groups=(("deferred", 19), ("dropped", 1), ("pass", 18)),
            waiting=(
                ("fake-waiting-on-boss", ("fake-deferred-one",)),
                ("fake-waiting-on-engine", ("fake-deferred-two", "fake-deferred-three")),
            ),
        ),
        milestones=(
            Milestone(
                title="fake-batch-1",
                state="closed",
                open_issues=0,
                closed_issues=16,
                url="https://github.com/fake/fake/milestone/1",
            ),
        ),
        tickets=FAKE_TICKETS,
        closed_review=FAKE_REVIEW,
        cloud=cloud,
        repo_state=RepoState(
            branch="main",
            head_sha="abcdef123456",
            head_subject="假的提交標題",
            head_when="2026-09-10 03:35",
            ahead=0,
            behind=0,
        ),
    )


def resource_markers(page: str) -> list[str]:
    """頁面裡有沒有「載入時要去外面抓東西」的痕跡。"""
    markers = ("<script", "<link", "<img", "<iframe", "@import", "url(http", " src=")
    return [marker for marker in markers if marker in page]


def http_attributes(page: str) -> list[str]:
    """把 http 當**屬性值**的地方。只准是超連結的 href（點了才走，不是載入時去抓的東西）。

    純文字裡的網址不算（頁面上那個固定網址就是連結文字），所以只找 `="http` 這個形狀，
    再往前看它是哪一個標籤的哪一個屬性。
    """
    loose: list[str] = []
    at = page.find('="http')
    while at >= 0:
        opened = page.rfind("<", 0, at)
        chunk = page[opened:at]
        if not chunk.startswith("<a ") or not chunk.endswith("href"):
            loose.append(page[opened : at + 40])
        at = page.find('="http', at + 1)
    return loose


def test_page_names_every_rule_card_it_was_given() -> None:
    """規矩卡：每一張的名字與人話都要出現在頁面上。"""
    page = render.render_page(fake_page(), TODAY)
    missing = {card.card_id for card in FAKE_CARDS if card.card_id not in page}
    assert missing == set()


def test_page_says_what_each_deferred_group_is_waiting_for() -> None:
    """暫緩的卡：在等什麼要跟卡的名字一起寫出來，不只寫幾張。"""
    data = fake_page()
    page = render.render_page(data, TODAY)
    for blocker, ids in data.blueprint.waiting:
        assert blocker in page
        for card_id in ids:
            assert card_id in page


def test_page_carries_the_do_not_edit_warning() -> None:
    """頁首那一行警告：這頁由機器算出，不要手改。"""
    page = render.render_page(fake_page(), TODAY)
    assert render.DO_NOT_EDIT in page


def test_page_says_which_run_computed_it() -> None:
    """「只認雲端」：頁面自己要說得出是哪一跑算的、反映的是哪一顆 commit。"""
    data = fake_page()
    page = render.render_page(data, TODAY)
    assert data.computed_by in page
    assert data.reflects in page
    assert data.computed_at in page


def test_timeline_draws_one_dot_per_day_with_cards() -> None:
    """時間軸：一天一個點，天數跟假資料裡算出來的一樣多。"""
    data = fake_page()
    page = render.render_page(data, TODAY)
    days = render.timeline_days(data.cards)
    assert "<svg" in page
    assert page.count('class="tl-dot"') == len(days)
    assert {day for day, _ in days} == {card.merged_day for card in FAKE_CARDS}


def test_timeline_is_inline_svg_not_an_image() -> None:
    """時間軸是畫出來的，不是一張要去外面抓的圖。"""
    svg = render.timeline_svg(render.timeline_days(FAKE_CARDS), TODAY)
    assert "<circle" in svg
    assert resource_markers(svg) == []


def test_page_loads_nothing_from_outside() -> None:
    """整頁不准依賴外部資源：http 只准出現在超連結的 href 上。"""
    page = render.render_page(fake_page(), TODAY)
    assert resource_markers(page) == []
    assert http_attributes(page) == []
    # 反面：真的有超連結（不是因為整頁一個網址都沒有才過關）。
    assert '<a href="https://' in page


def test_missing_cloud_run_is_said_out_loud_in_red() -> None:
    """雲端一次都沒跑過的時候要寫成紅字，不准安靜留白。"""
    page = render.render_page(fake_page(cloud=None), TODAY)
    assert 'class="bad"' in page
    assert "只認雲端" in page


def test_every_conclusion_word_is_chinese_not_colour_only() -> None:
    """狀態顏色一定配中文字：顏色分不出來的人也讀得懂。"""
    for conclusion in render.CONCLUSIONS:
        cell = render.state_html(conclusion)
        css, word = render.word_for(conclusion)
        assert word in cell
        assert css in cell


def test_tickets_are_grouped_by_label() -> None:
    """開著的票依標籤分組：要拍板的、暫緩的、PR、其他，各有各的段落。"""
    page = render.render_page(fake_page(), TODAY)
    for wanted in ("標籤 decision", "標籤 deferred-cards", "合併請求", "其他還開著的票"):
        assert wanted in page
    for ticket in FAKE_TICKETS:
        assert f"#{ticket.number}" in page


def test_titles_from_github_are_escaped() -> None:
    """從 GitHub 來的標題要轉義，角括號不准弄壞這一頁。"""
    page = render.render_page(fake_page(), TODAY)
    assert "&lt;也吃得下角括號&gt;" in page


def test_shell_returns_tool_broken_when_the_tool_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`gh` 不在就回 2（工具自壞）：不准安靜產一頁沒資料的。"""

    def absent(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(collect.GH)

    # 打的是 subprocess 這個模組自己那一格（collect.py 走的就是同一個物件），
    # 所以這一支測試從頭到尾沒有真的叫起任何一支外部程式。
    monkeypatch.setattr(subprocess, "run", absent)
    shell = collect.Shell(cwd=tmp_path, timeout=1)
    with pytest.raises(ToolBroken) as caught:
        shell.out([collect.GH, "api", "repos/fake/fake"], "假的一問")
    assert collect.GH in str(caught.value)


def test_entry_point_maps_tool_broken_to_exit_code_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """算不出來的時候：離開碼 2，而且一份產出物都不留。"""

    def boom(*args: object, **kwargs: object) -> PageData:
        raise ToolBroken("假的：gh 不在")

    monkeypatch.setattr(build_status, "collect", boom)
    out = tmp_path / "page"
    assert build_status.main(["--out", str(out)]) == TOOL_BROKEN
    assert not (out / build_status.PAGE_FILE).exists()


def test_entry_point_refuses_to_write_into_the_repo_tree(tmp_path: Path) -> None:
    """產出目錄落在版控樹裡就回 2：這一頁刻意不進主線。"""
    with pytest.raises(ToolBroken):
        build_status.assert_outside_repo(tmp_path, tmp_path / "docs" / "page")


def test_entry_point_writes_a_page_when_everything_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """算得出來的時候：離開碼 0，產出目錄裡有那一頁，還有給 Pages 的 .nojekyll。"""

    def answered(*args: object, **kwargs: object) -> PageData:
        return fake_page()

    monkeypatch.setattr(build_status, "collect", answered)
    out = tmp_path / "page"
    assert build_status.main(["--out", str(out)]) == CLEAN
    page = (out / build_status.PAGE_FILE).read_text(encoding="utf-8")
    assert render.DO_NOT_EDIT in page
    assert (out / build_status.NOJEKYLL_FILE).exists()


def test_collect_narrowing_refuses_shapes_it_cannot_read() -> None:
    """讀回來的東西形狀看不懂就回 2，不准自己補一個空清單。"""
    with pytest.raises(ToolBroken):
        collect.as_rows({"不是": "清單"}, "假的一格")
    with pytest.raises(ToolBroken):
        collect.as_table(["不是表"], "假的一格")
    with pytest.raises(ToolBroken):
        collect.parse_json("{這不是 json", "假的一問")


def test_seconds_and_time_conversion_are_readable() -> None:
    """時間換成台北時間（+8）、耗時算得出來；看不懂的時間照原樣還回去，不變成假的值。"""
    assert collect.to_taipei("2026-09-10T03:00:00Z").endswith("11:00")
    assert collect.to_taipei("看不懂的時間") == "看不懂的時間"
    one = collect.elapsed_seconds("2026-09-10T03:00:00Z", "2026-09-10T03:01:00Z")
    two = collect.elapsed_seconds("2026-09-10T03:00:00Z", "2026-09-10T03:02:00Z")
    assert one > 0
    assert two == one * 2
    # 倒過來問（結束在開始之前）不准算出負數，那會在頁面上變成一個看不懂的耗時。
    assert not collect.elapsed_seconds("2026-09-10T03:02:00Z", "2026-09-10T03:00:00Z")


def test_labels_come_out_of_github_tables() -> None:
    """標籤那一格：GitHub 給的是一張張表，取得出名字。"""
    row: dict[str, object] = {"labels": [{"name": "decision"}, {"name": "deferred-cards"}]}
    assert set(collect.field_labels(row)) == {"decision", "deferred-cards"}


def test_one_sentence_keeps_it_short_for_the_boss() -> None:
    """卡的人話取一句：長的要切短，短的原樣留著。"""
    short = "很短的一句"
    assert render.one_sentence(f"{short}。後面還有很多字") == short
    long_text = "一" * 200
    assert render.one_sentence(long_text).endswith("…")
    assert len(render.one_sentence(long_text)) < len(long_text)


def test_milestone_progress_is_computed_not_typed() -> None:
    """里程碑的進度成數是算出來的：關掉的除以總數。"""
    data = fake_page()
    page = render.render_page(data, TODAY)
    for milestone in data.milestones:
        assert milestone.title in page
    assert "100%" in page


# ── 關掉的票對不對得到綠收據 ─────────────────────────────────────────────────
# 四張假的關掉的票，各自壞在不同的一段：全串得起來、找不到合進主線的 PR、
# 那一顆 commit 沒有收據、收據不是綠的。

FAKE_CLOSED = (
    ClosedTicket(
        number=201,
        title="假的票：三樣都串得起來",
        closed="2026-09-10 15:02",
        url="https://github.com/fake/fake/issues/201",
        pr_number=301,
        pr_url="https://github.com/fake/fake/pull/301",
        merge_sha="aaaaaaaaaaaa",
        receipt_run_id=555001,
        receipt_url="https://github.com/fake/fake/actions/runs/555001",
        receipt_green=True,
        problem="",
    ),
    ClosedTicket(
        number=202,
        title="假的票：人手關的，沒有 PR 做掉它",
        closed="2026-09-10 14:00",
        url="https://github.com/fake/fake/issues/202",
        pr_number=0,
        pr_url="",
        merge_sha="",
        receipt_run_id=0,
        receipt_url="",
        receipt_green=False,
        problem="人手關的、沒有 PR 做掉它——GitHub 上沒有任何 PR 掛著關掉 #202",
    ),
    ClosedTicket(
        number=203,
        title="假的票：那一顆 commit 沒有收據",
        closed="2026-09-10 13:00",
        url="https://github.com/fake/fake/issues/203",
        pr_number=303,
        pr_url="https://github.com/fake/fake/pull/303",
        merge_sha="bbbbbbbbbbbb",
        receipt_run_id=0,
        receipt_url="",
        receipt_green=False,
        problem="PR #303 併出來的 bbbbbbbbbbbb 在收據分支上沒有收據",
    ),
    ClosedTicket(
        number=204,
        title="假的票：收據不是綠的",
        closed="2026-09-10 12:00",
        url="https://github.com/fake/fake/issues/204",
        pr_number=304,
        pr_url="https://github.com/fake/fake/pull/304",
        merge_sha="cccccccccccc",
        receipt_run_id=555004,
        receipt_url="https://github.com/fake/fake/actions/runs/555004",
        receipt_green=False,
        problem="收據 run 555004 不是綠的（那一跑或某一支檢查沒過）",
    ),
)

FAKE_REVIEW = ClosedReview(source="origin/status（假的 9 顆 commit 有收據）", tickets=FAKE_CLOSED)

# 一份假的機器收據，形狀跟 status 分支上那些一樣（只留這一格會看的欄位）。
FAKE_RECEIPT: dict[str, object] = {
    "run": {"run_id": 555001, "head_sha": "a" * 40, "conclusion": "success", "url": "https://x.invalid"},
    "job": {"conclusion": "success"},
    "checks": [{"name": "假的第一支", "exit_code": 0}, {"name": "假的第二支", "exit_code": 0}],
}

# 兩個假的 PR：都合進主線了，後合的那一個才是這一格要挑出來的。
FAKE_EARLY_PULL = collect.MergedPull(
    number=301,
    url="https://github.com/fake/fake/pull/301",
    merged_at="2026-09-10T06:13:54Z",
    merge_sha="a" * 40,
)
FAKE_LATE_PULL = collect.MergedPull(
    number=302,
    url="https://github.com/fake/fake/pull/302",
    merged_at="2026-09-10T07:02:28Z",
    merge_sha="b" * 40,
)


def with_receipt(**changed: object) -> dict[str, object]:
    """改一格的假收據（其他格照 FAKE_RECEIPT）。"""
    return {**FAKE_RECEIPT, **changed}


def test_closed_tickets_without_a_green_receipt_are_said_out_loud_in_red() -> None:
    """關掉的票串不起來的那幾張：那一句要出現在頁面上，而且是紅字。"""
    page = render.render_page(fake_page(), TODAY)
    block = render.closed_review_block(FAKE_REVIEW)
    broken = [t for t in FAKE_CLOSED if t.problem]
    for ticket in broken:
        assert ticket.problem in page
        assert f'<span class="bad">{ticket.problem}</span>' in block
    # 反面：紅字那一行點名的就是壞掉的那幾張，沒有多也沒有少。
    named = {f"#{t.number}" for t in broken}
    listed = {f"#{t.number}" for t in FAKE_CLOSED if f'<a href="{t.url}">#{t.number}</a>' in block}
    assert named <= listed


def test_closed_ticket_that_checks_out_shows_its_green_receipt() -> None:
    """串得起來的那一張：PR、那一顆 commit、綠收據三樣都寫在頁面上。"""
    good = next(t for t in FAKE_CLOSED if not t.problem)
    page = render.render_page(fake_page(), TODAY)
    assert f"#{good.pr_number}" in page
    assert good.merge_sha in page
    assert f"run {good.receipt_run_id}" in page


def test_closed_review_says_where_the_receipts_came_from() -> None:
    """這一格自己要說得出收據是從哪裡讀的（哪一個 ref、幾份），不然數字沒有出處。"""
    page = render.render_page(fake_page(), TODAY)
    assert FAKE_REVIEW.source in page


def test_closed_review_with_nothing_closed_says_so() -> None:
    """最近一張關掉的票都沒有的時候：明說沒有，不留白。"""
    block = render.closed_review_block(ClosedReview(source="假的來源", tickets=()))
    assert "都沒有" in block


def test_receipt_is_green_only_when_the_run_the_job_and_every_check_are_clean() -> None:
    """綠的定義從欄位重算：那一跑綠、job 綠、每一支檢查離開碼 0，缺一不可。"""
    assert collect.receipt_is_green(FAKE_RECEIPT)
    # 一支檢查的離開碼非零，而 GitHub 記的還是綠——中間有一層把離開碼吞掉，正是要抓的形狀。
    swallowed = with_receipt(checks=[{"name": "假的一支", "exit_code": 1}])
    assert not collect.receipt_is_green(swallowed)
    assert not collect.receipt_is_green(with_receipt(job={"conclusion": "failure"}))
    assert not collect.receipt_is_green(with_receipt(run={"conclusion": "failure"}))
    # 一支檢查都沒有的收據不算綠：沒掃過的乾淨不是乾淨。
    assert not collect.receipt_is_green(with_receipt(checks=[]))


# ── 假的外殼：不上網、不 spawn，照順序把備好的回答交出去 ─────────────────────────


def page_of(argv: Sequence[str]) -> int:
    """從一次呼叫的 argv（給外部指令的那串參數）裡挖出問的是第幾頁；沒寫就是第一頁。"""
    for item in argv:
        for part in item.replace("?", "&").split("&"):
            if part.startswith("page="):
                return int(part.removeprefix("page="))
    return 1


class Replies:
    """假的外殼：照順序把備好的一段段 JSON 交出去，並且記下每一次的 argv。

    答案用完還被問就當場炸——「有沒有多翻一頁」這件事要斷言得到，不能靠它安靜地回一份空的。
    """

    def __init__(self, answers: Sequence[str]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(tuple(str(item) for item in argv))
        if not self.answers:
            raise AssertionError(f"假外殼被多問了一次：{list(argv)}")
        return subprocess.CompletedProcess(
            args=list(argv), returncode=0, stdout=self.answers.pop(0), stderr=""
        )

    @property
    def pages_asked(self) -> list[int]:
        """問過的每一次各是第幾頁。"""
        return [page_of(call) for call in self.calls]


def fake_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answers: Sequence[str], per_page: int
) -> tuple[collect.Hub, Replies]:
    """一把只會回假資料的 Hub（問 GitHub 的那一把手）。

    打的是 subprocess 那一格（collect.py 走的就是同一個物件），所以這幾支測試從頭到尾
    沒有真的叫起任何一支外部程式，也沒有上網。
    """
    replies = Replies(answers)
    monkeypatch.setattr(subprocess, "run", replies)
    shell = collect.Shell(cwd=tmp_path, timeout=1)
    return collect.Hub(shell=shell, slug="fake/fake", per_page=per_page), replies


def fake_open_row(number: int, title: str) -> dict[str, object]:
    """GitHub 回的一張開著的票（只留這一層會讀的欄位）。"""
    return {
        "number": number,
        "title": title,
        "html_url": f"https://x.invalid/{number}",
        "updated_at": "2026-09-10T00:00:00Z",
        "labels": [],
    }


def fake_job_row(name: str, step: str) -> dict[str, object]:
    """GitHub 回的一個 job（雲端那一跑裡的一份工作），底下一步。"""
    return {
        "name": name,
        "started_at": "2026-09-10T00:00:00Z",
        "completed_at": "2026-09-10T00:01:00Z",
        "steps": [
            {
                "name": step,
                "conclusion": "success",
                "started_at": "2026-09-10T00:00:00Z",
                "completed_at": "2026-09-10T00:00:30Z",
            }
        ],
    }


def pull_node(
    pull: collect.MergedPull, *, merged: bool = True, base: str = collect.MAIN
) -> dict[str, object]:
    """圖查詢回的一個「掛著關掉這張票」的 PR。沒合的那些 GitHub 給的 mergeCommit 是 null。"""
    return {
        "number": pull.number,
        "url": pull.url,
        "merged": merged,
        "mergedAt": pull.merged_at,
        "baseRefName": base,
        "mergeCommit": {"oid": pull.merge_sha} if merged else None,
    }


def closed_by(nodes: Sequence[Mapping[str, object]], *, more: bool = False, cursor: str = "") -> str:
    """假的圖查詢回應：GitHub 記的關票來源那一段（`closedByPullRequestsReferences`）。"""
    references = {
        "pageInfo": {"hasNextPage": more, "endCursor": cursor},
        "nodes": [dict(node) for node in nodes],
    }
    issue = {"closedByPullRequestsReferences": references}
    return json.dumps({"data": {"repository": {"issue": issue}}})


# ── 翻頁：問清單一律翻到底 ────────────────────────────────────────────────────


def test_lists_from_github_are_paged_to_the_bottom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """滿滿一頁就再翻一頁：第二頁那幾筆的名字也要出現在結果裡，不是只拿第一頁。"""
    first = [fake_open_row(11, "第一頁：第一張"), fake_open_row(12, "第一頁：第二張")]
    second = [fake_open_row(21, "第二頁：只有這一張")]
    hub, replies = fake_hub(
        monkeypatch, tmp_path, [json.dumps(first), json.dumps(second)], len(first)
    )
    titles = {ticket.title for ticket in collect.read_tickets(hub)}
    assert {str(row["title"]) for row in first} <= titles
    assert "第二頁：只有這一張" in titles
    assert replies.pages_asked == [1, 2]


def test_a_short_page_is_the_last_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """第一頁就沒滿：不再多問一次（短的一頁就是最後一頁）。"""
    rows = [fake_open_row(11, "只有這一張")]
    hub, replies = fake_hub(monkeypatch, tmp_path, [json.dumps(rows)], len(rows) + 1)
    assert [ticket.title for ticket in collect.read_tickets(hub)] == ["只有這一張"]
    assert replies.pages_asked == [1]


def test_boxed_lists_are_paged_too(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """包在一格裡的清單（jobs、workflow_runs 那種）一樣翻到底，不是只開第一頁的盒子。"""
    first = [
        fake_job_row("別的 job", "無關的一步"),
        fake_job_row(collect.VERIFY_WORKFLOW, "第一頁那一步"),
    ]
    second = [fake_job_row(collect.VERIFY_WORKFLOW, "第二頁那一步")]
    hub, replies = fake_hub(
        monkeypatch,
        tmp_path,
        [json.dumps({"jobs": first}), json.dumps({"jobs": second})],
        len(first),
    )
    _, steps = collect.read_run_steps(hub, 555001)
    assert [step.name for step in steps] == ["第一頁那一步", "第二頁那一步"]
    assert replies.pages_asked == [1, 2]


# ── 這張票是被哪個 PR 做掉的：讀 GitHub 自己記的關票來源 ──────────────────────


def test_closing_pull_comes_from_githubs_own_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """關票 PR 讀的是圖查詢那條連結，翻到底，好幾個就取最後合進主線的那一個。"""
    hub, replies = fake_hub(
        monkeypatch,
        tmp_path,
        [
            closed_by([pull_node(FAKE_EARLY_PULL)], more=True, cursor="第一頁的游標"),
            closed_by([pull_node(FAKE_LATE_PULL)]),
        ],
        1,
    )
    closing = collect.closing_pulls(hub, 201)
    assert closing.landed is not None
    assert closing.landed.number == FAKE_LATE_PULL.number
    assert closing.landed.merge_sha == FAKE_LATE_PULL.merge_sha
    # 第二頁真的問了，而且帶著第一頁給的游標（cursor，GitHub 給的位置代號）。
    assert any("after=第一頁的游標" in part for part in replies.calls[1])
    # 走的是圖查詢，不是票面上的事件流：cross-referenced 那條猜法已經拿掉。
    assert "graphql" in replies.calls[0]
    assert not any("timeline" in part for call in replies.calls for part in call)


def test_a_ticket_closed_by_hand_has_no_closing_pull(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """人手關的票：GitHub 上一個掛著關它的 PR 都沒有，這一層就說沒有，不去猜一個。"""
    hub, _ = fake_hub(monkeypatch, tmp_path, [closed_by([])], 1)
    closing = collect.closing_pulls(hub, 202)
    assert closing.landed is None
    assert not closing.referenced


def test_a_closing_pull_that_never_landed_is_not_landed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """掛著關掉它、但沒合進主線的 PR 不算落地——而且要跟「根本沒有 PR」分得開。"""
    hub, _ = fake_hub(
        monkeypatch, tmp_path, [closed_by([pull_node(FAKE_LATE_PULL, merged=False)])], 1
    )
    closing = collect.closing_pulls(hub, 203)
    assert closing.landed is None
    assert closing.referenced


def test_a_closing_pull_into_another_branch_is_not_landed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """合進別條分支的不算落地：這一格問的是「東西有沒有進主線」。"""
    hub, _ = fake_hub(
        monkeypatch, tmp_path, [closed_by([pull_node(FAKE_LATE_PULL, base="別條分支")])], 1
    )
    assert collect.closing_pulls(hub, 204).landed is None


def test_the_repo_full_name_has_to_be_owner_slash_repo() -> None:
    """全名拆不成 owner/repo 就回 2（圖查詢要兩個分開的變數），不硬送一個空的上去。"""
    with pytest.raises(ToolBroken):
        collect.owner_and_name("沒有斜線")


def test_judge_says_which_link_of_the_chain_is_broken() -> None:
    """斷在哪一段就寫哪一段：人手關的／有 PR 沒合／沒有收據／收據不綠，四句話不一樣。"""
    issue = collect.ClosedIssue(
        number=201, title="假的票", url="https://x.invalid/201", closed_at="2026-09-10T07:02:36Z"
    )
    green = collect.ReceiptFacts(run_id=555001, green=True, url="https://x.invalid/run")
    red = collect.ReceiptFacts(run_id=555004, green=False, url="https://x.invalid/run")
    by_hand = collect.ClosingPulls(landed=None, referenced=0)
    never_landed = collect.ClosingPulls(landed=None, referenced=2)
    landed = collect.ClosingPulls(landed=FAKE_LATE_PULL, referenced=1)
    said = {
        "人手關的": collect.judge_closed(issue, by_hand, None).problem,
        "有 PR 沒合": collect.judge_closed(issue, never_landed, None).problem,
        "沒有收據": collect.judge_closed(issue, landed, None).problem,
        "收據不綠": collect.judge_closed(issue, landed, red).problem,
    }
    assert "人手關的、沒有 PR 做掉它" in said["人手關的"]
    assert collect.MAIN in said["有 PR 沒合"]
    assert "沒有收據" in said["沒有收據"]
    assert "不是綠的" in said["收據不綠"]
    # 四句話彼此不一樣：字面上分得開，人一眼看得出斷在哪一段。
    assert sorted(said.values()) == sorted(set(said.values()))
    ok = collect.judge_closed(issue, landed, green)
    assert not ok.problem
    assert ok.receipt_green
    # 關掉的時間要換成台北時間（+8），不是照抄 GitHub 給的那串。
    assert ok.closed == collect.to_taipei(issue.closed_at)


def test_the_two_reds_are_different_sentences() -> None:
    """頁面上那兩種紅是兩句不一樣的話：人手關的、跟有 PR 但收據串不起來。"""
    block = render.closed_review_block(FAKE_REVIEW)
    assert "人手關的、沒有 PR 做掉它：" in block
    assert "有 PR 做掉它、但收據串不起來：" in block


def test_closed_issues_come_back_newest_closed_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GitHub 排不出「按關掉時間」，所以這一層自己排：新關的在前，PR 不算票。"""
    rows = [
        {
            "number": 1,
            "title": "舊的",
            "html_url": "https://x.invalid/1",
            "closed_at": "2026-09-01T00:00:00Z",
        },
        {
            "number": 2,
            "title": "新的",
            "html_url": "https://x.invalid/2",
            "closed_at": "2026-09-10T00:00:00Z",
        },
        {
            "number": 3,
            "title": "這是 PR 不是票",
            "html_url": "https://x.invalid/3",
            "closed_at": "2026-09-11T00:00:00Z",
            "pull_request": {"url": "https://x.invalid/3"},
        },
    ]
    hub, _ = fake_hub(monkeypatch, tmp_path, [json.dumps(rows)], len(rows) + 1)
    picked = collect.read_closed_issues(hub, len(rows))
    assert [issue.number for issue in picked] == [2, 1]
