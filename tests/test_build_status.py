"""狀態頁的測試：全部餵假資料，不上網、不 spawn 任何東西、不寫進真的樹。

為什麼可以這樣測：`governance/status/render.py` 那一層刻意不碰網路也不碰子程序，
它只吃一個算好的 `PageData`。真的去問版控與 GitHub 的那一層在 `collect.py`，
這裡只戳它「外部工具不在的時候是不是回 2」——用假的子程序，不真的叫誰。

斷言刻意不鎖死數量（規矩卡 assertions-not-pinned-to-counts）：比的是「有沒有這幾個名字」、
「圖上的點是不是跟假資料裡的天數一樣多」，不是「必須剛好 17 張」。
"""
from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken
from governance.status import build_status, collect, render
from governance.status.model import (
    Blueprint,
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
