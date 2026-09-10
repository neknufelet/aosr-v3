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
        title="假的票：沒有合進主線的 PR",
        closed="2026-09-10 14:00",
        url="https://github.com/fake/fake/issues/202",
        pr_number=0,
        pr_url="",
        merge_sha="",
        receipt_run_id=0,
        receipt_url="",
        receipt_green=False,
        problem="沒有一個合進主線的 PR 提到它——票關了，東西不知道有沒有進主線",
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


def test_judge_says_which_link_of_the_chain_is_broken() -> None:
    """斷在哪一段就寫哪一段：沒有 PR／沒有收據／收據不綠，三句話不一樣。"""
    issue = collect.ClosedIssue(
        number=201, title="假的票", url="https://x.invalid/201", closed_at="2026-09-10T07:02:36Z"
    )
    green = collect.ReceiptFacts(run_id=555001, green=True, url="https://x.invalid/run")
    red = collect.ReceiptFacts(run_id=555004, green=False, url="https://x.invalid/run")
    assert "沒有一個合進主線的 PR" in collect.judge_closed(issue, None, None).problem
    assert "沒有收據" in collect.judge_closed(issue, FAKE_LATE_PULL, None).problem
    assert "不是綠的" in collect.judge_closed(issue, FAKE_LATE_PULL, red).problem
    ok = collect.judge_closed(issue, FAKE_LATE_PULL, green)
    assert not ok.problem
    assert ok.receipt_green
    # 關掉的時間要換成台北時間（+8），不是照抄 GitHub 給的那串。
    assert ok.closed == collect.to_taipei(issue.closed_at)


def test_closed_issue_takes_the_last_merged_pull_request_that_mentions_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """票面上有好幾個合進主線的 PR 提到它，就取最後合進去的那一個。"""
    timeline = json.dumps(
        [
            {"event": "commented"},
            {"event": "cross-referenced", "source": {"issue": {"number": FAKE_EARLY_PULL.number}}},
            {"event": "cross-referenced", "source": {"issue": {"number": FAKE_LATE_PULL.number}}},
            {"event": "cross-referenced", "source": {"issue": {"number": 999}}},
        ]
    )

    def answered(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=timeline, stderr="")

    # 一樣打 subprocess 那一格：這一支測試沒有真的叫起任何一支外部程式，也沒有上網。
    monkeypatch.setattr(subprocess, "run", answered)
    shell = collect.Shell(cwd=tmp_path, timeout=1)
    merged = {FAKE_EARLY_PULL.number: FAKE_EARLY_PULL, FAKE_LATE_PULL.number: FAKE_LATE_PULL}
    picked = collect.linked_merged_pull(shell, "fake/fake", 201, merged)
    assert picked is not None
    assert picked.number == FAKE_LATE_PULL.number


def test_closed_issue_with_no_merged_pull_request_comes_back_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """票面上提到的那幾個都還沒合進主線：這一格回「沒有」，不隨便挑一個充數。"""
    timeline = json.dumps(
        [{"event": "cross-referenced", "source": {"issue": {"number": 999}}}, {"event": "closed"}]
    )

    def answered(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=timeline, stderr="")

    monkeypatch.setattr(subprocess, "run", answered)
    shell = collect.Shell(cwd=tmp_path, timeout=1)
    merged = {FAKE_LATE_PULL.number: FAKE_LATE_PULL}
    assert collect.linked_merged_pull(shell, "fake/fake", 202, merged) is None


def test_only_merged_pull_requests_count_as_landed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """關掉但沒合的 PR 不算落地：只有帶 merged_at 的才進得了那張對照表。"""
    pulls = json.dumps(
        [
            {
                "number": FAKE_LATE_PULL.number,
                "html_url": FAKE_LATE_PULL.url,
                "merged_at": FAKE_LATE_PULL.merged_at,
                "merge_commit_sha": FAKE_LATE_PULL.merge_sha,
            },
            {"number": 998, "html_url": "https://x.invalid/998", "merged_at": None},
        ]
    )

    def answered(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=pulls, stderr="")

    monkeypatch.setattr(subprocess, "run", answered)
    shell = collect.Shell(cwd=tmp_path, timeout=1)
    landed = collect.read_merged_pulls(shell, "fake/fake")
    assert set(landed) == {FAKE_LATE_PULL.number}


def test_closed_issues_come_back_newest_closed_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GitHub 排不出「按關掉時間」，所以這一層自己排：新關的在前，PR 不算票。"""
    issues = json.dumps(
        [
            {"number": 1, "title": "舊的", "html_url": "https://x.invalid/1", "closed_at": "2026-09-01T00:00:00Z"},
            {"number": 2, "title": "新的", "html_url": "https://x.invalid/2", "closed_at": "2026-09-10T00:00:00Z"},
            {
                "number": 3,
                "title": "這是 PR 不是票",
                "html_url": "https://x.invalid/3",
                "closed_at": "2026-09-11T00:00:00Z",
                "pull_request": {"url": "https://x.invalid/3"},
            },
        ]
    )

    def answered(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=issues, stderr="")

    monkeypatch.setattr(subprocess, "run", answered)
    shell = collect.Shell(cwd=tmp_path, timeout=1)
    picked = collect.read_closed_issues(shell, "fake/fake", len(json.loads(issues)))
    assert [issue.number for issue in picked] == [2, 1]
