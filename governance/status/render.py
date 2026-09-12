"""把算好的資料排成一頁 HTML。

**這一層不碰網路、不碰子程序、不讀檔**，所以測得動：測試餵假資料進來，看出來的那份 HTML
有沒有該有的東西。

**頁面不准依賴外部資源。** 樣式、圖、時間軸全部寫在這一份檔裡（inline CSS ＋ inline SVG），
一個 `<script>`／`<link>`／`<img>` 都沒有：這一頁掛在 GitHub Pages 上給老闆看，網路上任何
一個第三方掛掉都不准讓它變成一片空白。同樣的道理寫在規矩卡 refs-and-links-resolve 與決策紙
「文件不准依賴外部資源」那一題底下。頁面裡唯一的 `http` 是連到 GitHub 的**超連結**
（點了才走），不是這一頁載入時要抓的東西。

全中文、白話、給老闆看：每個英文名字旁邊同一句要有中文說它在做什麼。

分段的順序是「先講老闆今天要做什麼，再講最近發生什麼，機器細節折起來」：

1. 今天一句話——一行字總結現在站在哪。
2. 要你回的——等著老闆拍板的那幾題。
3. 最近做完的——最近關掉的票，落地證據對不對得起來。
4. 正在做——開著的合併請求（PR）與其他票。
5. 健康燈——三盞，每盞一句人話，細節折在 `<details>` 裡。
6. 路線走到哪——里程碑的進度條。
7. 規矩與藍圖——折起來的完整清單、時間軸、藍圖去向。
8. 頁尾——這一頁怎麼來的。
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from html import escape

from governance.status.model import (
    ClosedReview,
    ClosedTicket,
    CloudRun,
    Milestone,
    PageData,
    ReceiptGaps,
    RuleCard,
    Ticket,
)

# 頁首那一行警告。手改這一頁沒有意義：下一次合併會整份蓋掉。
DO_NOT_EDIT = "這頁由機器算出，不要手改；手改會被下一次合併蓋掉"

# 兩個要單獨分組的標籤（決策紙「待辦全走 GitHub issue」定的那兩組）。
LABEL_DECISION = "decision"
LABEL_DEFERRED = "deferred-cards"

# 結論字樣 → （顏色類別, 中文說法）。狀態顏色一律配上中文字，不准只靠顏色說話。
CONCLUSIONS = {
    "success": ("ok", "綠（過了）"),
    "failure": ("bad", "紅（沒過）"),
    "cancelled": ("warn", "被取消"),
    "skipped": ("warn", "跳過"),
    "timed_out": ("bad", "逾時"),
    "startup_failure": ("bad", "沒跑起來"),
    "": ("warn", "還在跑或沒有結論"),
}

# 時間軸的畫布。單位是 viewBox 的座標，不是像素（頁面寬度由 CSS 拉滿）。
VIEW_WIDTH = 1000
VIEW_HEIGHT = 150
AXIS_Y = 92
AXIS_LEFT = 46
AXIS_RIGHT = 954

STYLE = """
:root{--ink:#1b1c1e;--dim:#5b6068;--line:#d9dce1;--bg:#fbfbfc;--card:#ffffff;
--ok:#1a7f52;--bad:#b3261e;--warn:#8a6100;--dot:#2b5d9b;--bad-bg:#fbe9e7;--warn-bg:#fff3e0;}
@media (prefers-color-scheme:dark){:root{--ink:#e8eaed;--dim:#a4abb5;--line:#3a3f47;
--bg:#15171a;--card:#1e2126;--ok:#4bbd85;--bad:#f2857c;--warn:#e0b155;--dot:#7fb0e8;
--bad-bg:#351f22;--warn-bg:#342a18;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:17px;line-height:1.7;
font-family:-apple-system,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:32px 18px 72px}
h1{font-size:28px;margin:0 0 8px}
h2{font-size:22px;margin:40px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:18px;margin:26px 0 8px}
p{margin:8px 0}
.meta{color:var(--dim);font-size:14px}
.today{margin:12px 0 20px;padding:14px 16px;border:1px solid var(--line);background:var(--card);
border-radius:12px;font-size:18px}
.today.bad{background:var(--bad-bg);border-color:var(--bad);color:var(--bad);font-weight:700}
.today.warn{background:var(--warn-bg);border-color:var(--warn);color:var(--warn);font-weight:700}
.say{font-size:18px;margin:8px 0}
ul{margin:6px 0;padding-left:24px}
li{margin:5px 0}
a{color:inherit;text-decoration:underline;text-underline-offset:2px}
.ok{color:var(--ok);font-weight:700}
.bad{color:var(--bad);font-weight:700}
.warn{color:var(--warn);font-weight:700}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:600;font-size:13px}
.scroll{overflow-x:auto}
details{margin:10px 0;border:1px solid var(--line);border-radius:10px;background:var(--card)}
summary{cursor:pointer;padding:10px 14px;color:var(--dim);font-size:14px;font-weight:600}
details .fold{padding:0 14px 12px}
.light{display:block;border:1px solid var(--line);border-left:5px solid var(--line);
border-radius:10px;background:var(--card);padding:10px 14px;margin:10px 0;font-size:17px}
.light.ok{border-left-color:var(--ok)}
.light.bad{border-left-color:var(--bad)}
.light.warn{border-left-color:var(--warn)}
.bar{height:10px;border-radius:5px;background:var(--line);overflow:hidden;margin-top:6px}
.bar span{display:block;height:100%;background:var(--ok)}
.ms{margin:12px 0}
.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;
padding:0 8px;font-size:12px;color:var(--dim);margin-right:4px}
.foot{color:var(--dim);font-size:14px;margin-top:36px;border-top:1px solid var(--line);
padding-top:14px}
svg{width:100%;height:auto;display:block;background:var(--card);
border:1px solid var(--line);border-radius:10px}
.tl-axis{stroke:var(--line);stroke-width:2}
.tl-dot{fill:var(--dot);stroke:var(--card);stroke-width:2}
.tl-day{fill:var(--ink);font-size:13px}
.tl-count{fill:var(--dim);font-size:12px}
@media (max-width:600px){body{font-size:16px}h1{font-size:24px}h2{font-size:20px}}
"""


def esc(text: str) -> str:
    """所有從 GitHub 或版控來的字都要過這一手：標題裡有 `<` 也不會弄壞這一頁。"""
    return escape(text, quote=True)


def word_for(conclusion: str) -> tuple[str, str]:
    """一個結論字樣的（顏色類別, 中文說法）。沒見過的字樣照原樣寫出來。"""
    known = CONCLUSIONS.get(conclusion)
    if known:
        return known
    return ("warn", f"{conclusion}（沒見過的結論字樣）")


def state_html(conclusion: str) -> str:
    """一個結論的那一格 HTML。顏色之外一定有中文字。"""
    css, word = word_for(conclusion)
    return f'<span class="{css}">{esc(word)}</span>'


def one_sentence(text: str) -> str:
    """卡的人話取一句。太長就切掉，後面加刪節號——這一頁是給老闆看的，不是規格書。"""
    limit = 58
    head = text.split("。", 1)[0].strip()
    if len(head) <= limit:
        return head
    return f"{head[:limit]}…"


def cloud_short(cloud: CloudRun | None) -> tuple[str, str]:
    """主線最近一次雲端檢查的（顏色類別, 中文說法）。沒跑過就是紅。

    結論有三種顏色，不是兩種：綠（過了）、紅（沒過）、琥珀（被取消／跳過／還在跑）。
    琥珀不准當綠——這一跑沒有結論，不算數。
    """
    if cloud is None:
        return "bad", "紅（還沒跑過）"
    return word_for(cloud.conclusion)


def decision_tickets(tickets: Sequence[Ticket]) -> list[Ticket]:
    """開著的、要老闆拍板的那些票（標籤 decision）。"""
    return [t for t in tickets if LABEL_DECISION in t.labels]


def tags_html(labels: Sequence[str]) -> str:
    """標籤藥丸那一格。一個標籤都沒有就寫一個破折號，不留空格。"""
    tags = [f'<span class="tag">{esc(name)}</span>' for name in labels]
    return "".join(tags) if tags else "—"


# ── 1. 今天一句話 ───────────────────────────────────────────────────────────


def today_block(data: PageData) -> str:
    """一行總結：雲端檢查綠紅、最新合進去的是哪一筆、有幾題等你拍板。

    紅的那一天整行用紅底、琥珀的那一天整行用琥珀底——不能被老闆掃過去漏看。
    琥珀（被取消／跳過／還在跑）不是綠：這一跑沒有結論。
    """
    css, word = cloud_short(data.cloud)
    n_decision = len(decision_tickets(data.tickets))
    state = data.repo_state
    extra = f" {css}" if css != "ok" else ""
    caveat = "這一跑沒有結論，不算綠。" if css == "warn" else ""
    return (
        '<div class="today' + extra + '">'
        f"今天：主線最近一次雲端檢查（verify）{esc(word)}。{caveat}"
        f"最新合進去的是「{esc(state.head_subject)}」（{esc(state.head_when)}）。"
        f"有 {n_decision} 題要老闆拍板。</div>"
    )


# ── 2. 要你回的 ─────────────────────────────────────────────────────────────


def decisions_block(tickets: Sequence[Ticket]) -> str:
    """要老闆拍板的題（標籤 decision）一票一行：號碼、標題、最後動過、標籤。"""
    rows = decision_tickets(tickets)
    note = "要老闆拍板的題（標籤 decision）"
    if not rows:
        return (
            "<h2>要你回的</h2>\n"
            f'<p class="say">現在沒有要你回的——{note}一題都沒有。</p>'
        )
    lines = "\n".join(
        f'<li><a href="{esc(t.url)}">#{t.number}</a>　{esc(t.title)}　'
        f'<span class="meta">最後動過 {esc(t.updated)}</span>　{tags_html(t.labels)}</li>'
        for t in rows
    )
    return f"<h2>要你回的</h2>\n<p>{note}：</p>\n<ul>{lines}</ul>"


# ── 3. 最近做完的 ───────────────────────────────────────────────────────────


def closed_line(ticket: ClosedTicket) -> str:
    """一張關掉的票一行。串得起綠收據就寫「由哪個 PR 合進主線」，紅的寫問題那一句。"""
    base = f'<a href="{esc(ticket.url)}">#{ticket.number}</a>　{esc(ticket.title)}'
    if ticket.problem:
        return f'<li class="bad">{base}　<span class="bad">{esc(ticket.problem)}</span></li>'
    return (
        f"<li>{base} — 由 PR "
        f'<a href="{esc(ticket.pr_url)}">#{ticket.pr_number}</a>'
        f"（合併請求）合進主線，關掉時間 {esc(ticket.closed)}</li>"
    )


def closed_summary(review: ClosedReview) -> str:
    """「最近做完的」那兩個總結紅字：人手關的、跟有 PR 但收據串不起來，各列票號。"""
    by_hand = [t for t in review.tickets if t.problem and t.pr_number == 0]
    with_pr = [t for t in review.tickets if t.problem and t.pr_number != 0]
    lines: list[str] = []
    if by_hand:
        names = "、".join(f"#{t.number}" for t in by_hand)
        lines.append(
            f'<p class="bad">人手關的、沒有 PR 做掉它：{esc(names)}'
            "——票關了，GitHub 上沒有一個 PR 掛著關掉它，東西有沒有進主線只能靠人記得。</p>"
        )
    if with_pr:
        names = "、".join(f"#{t.number}" for t in with_pr)
        lines.append(
            f'<p class="bad">有 PR 做掉它、但收據串不起來：{esc(names)}'
            "——落地的那一顆 commit 對不到一份綠收據。</p>"
        )
    return "\n".join(lines)


def recently_closed_block(review: ClosedReview) -> str:
    """最近關掉的票（新的在前）一票一行。對不上綠收據的那幾張用紅字寫問題那一句。"""
    head = (
        "<h2>最近做完的</h2>\n"
        "<p>最近關掉的票（issue，待辦票），新的在前。一張票要串得起三樣東西：一個把它關掉的"
        "PR（合併請求）、那個 PR 併出來的那一顆 commit、以及那一顆 commit 的雲端檢查（verify）"
        '在機器分支上留下的一份綠收據（雲端那一跑留下的成績單）。串不起來的用紅字寫問題。</p>\n'
    )
    if not review.tickets:
        return head + '<p class="say">最近一張關掉的票都沒有。</p>'
    summary = closed_summary(review)
    body = head
    if summary:
        body += summary + "\n"
    return body + "<ul>" + "\n".join(closed_line(t) for t in review.tickets) + "</ul>"


# ── 4. 正在做 ───────────────────────────────────────────────────────────────


def in_progress_block(tickets: Sequence[Ticket]) -> str:
    """開著的合併請求（PR）一組、其他開著的票一組；等條件的卡併進其他那一組。"""
    prs = [t for t in tickets if t.is_pr]
    deferred = [t for t in tickets if not t.is_pr and LABEL_DEFERRED in t.labels]
    deferred_nums = {t.number for t in deferred}
    decision_nums = {t.number for t in decision_tickets(tickets)}
    others = [
        t
        for t in tickets
        if not t.is_pr and t.number not in deferred_nums and t.number not in decision_nums
    ]

    def group(title: str, rows: Sequence[Ticket]) -> str:
        count = f"（{len(rows)} 張）" if rows else "：沒有。"
        if not rows:
            return f"<p>{esc(title)}{'：沒有。'}</p>"
        lines = "\n".join(
            f'<li><a href="{esc(t.url)}">#{t.number}</a>　{esc(t.title)}　'
            f'<span class="meta">最後動過 {esc(t.updated)}</span>　{tags_html(t.labels)}</li>'
            for t in rows
        )
        return f"<p>{esc(title)}{count}</p>\n<ul>{lines}</ul>"

    deferred_with_others = [*others, *deferred]
    return (
        "<h2>正在做</h2>\n"
        + group("開著的合併請求（PR，等人看等人合）", prs)
        + group("其他開著的票（含等條件才立得起來的卡，標籤 deferred-cards）", deferred_with_others)
    )


# ── 5. 健康燈 ───────────────────────────────────────────────────────────────


def cloud_details(cloud: CloudRun | None) -> str:
    """第一盞燈折起來的細節：雲端檢查最近一跑每一步的結論表。"""
    if cloud is None:
        return (
            '<details><summary>沒有雲端紀錄可看</summary><div class="fold">'
            "<p>主線上還沒有任何一次雲端檢查（verify）的紀錄。</p></div></details>"
        )
    steps = "\n".join(
        "<tr>"
        f"<td>{esc(step.name)}</td>"
        f"<td>{state_html(step.conclusion)}</td>"
        f"<td>{step.seconds} 秒</td>"
        "</tr>"
        for step in cloud.steps
    )
    return (
        "<details>"
        '<summary>看那一跑每一步（中文那幾步就是一支一支的檢查）</summary>'
        '<div class="fold"><div class="scroll"><table>\n'
        "<tr><th>那一步</th><th>結論</th><th>跑了多久</th></tr>\n"
        f"{steps}\n</table></div>\n"
        f'<p class="meta">那一跑是 <a href="{esc(cloud.url)}">雲端那一跑（run {cloud.run_id}）</a>，'
        f"狀態 {esc(cloud.status)}　開始 {esc(cloud.started)}　"
        f"對著提交（commit）{esc(cloud.head_sha)}　收工 {esc(cloud.finished)}。"
        "這一頁只認雲端那一跑的結論，本機跑綠不算數。</p></div></details>"
    )


def cloud_light(cloud: CloudRun | None) -> str:
    """第一盞健康燈：雲端檢查最近一跑 綠／紅／琥珀，跑了多久。"""
    if cloud is None:
        return (
            '<div class="light bad">主線最近一次雲端檢查（verify）：'
            "紅——主線上還沒有任何一次紀錄。沒有雲端紀錄就沒有綠可言（只認雲端）。</div>\n"
            + cloud_details(cloud)
        )
    css, word = cloud_short(cloud)
    minutes = cloud.job_seconds // 60
    duration = f"約 {minutes} 分鐘" if minutes else f"{cloud.job_seconds} 秒"
    if css == "bad":
        verdict = f"紅——{word}。"
    elif css == "warn":
        verdict = f"琥珀——{word}。這一跑沒有結論，不算綠。"
    else:
        verdict = f"綠——{word}。"
    return (
        f'<div class="light {css}">主線最近一次雲端檢查（verify）：{verdict}'
        f"那一跑的工作（job）跑了 {duration}（{cloud.job_seconds} 秒）。</div>\n"
        + cloud_details(cloud)
    )


def receipts_details(gaps: ReceiptGaps) -> str:
    """第二盞燈折起來的細節：缺收據的那幾跑一張表。"""
    if not gaps.missing:
        inner = '<p class="ok">每一跑都有收據。</p>'
    else:
        rows = "\n".join(
            "<tr>"
            f'<td><a href="{esc(run.url)}">雲端那一跑（run {run.run_id}）</a></td>'
            f"<td>{state_html(run.conclusion)}</td>"
            f"<td>{esc(run.started)}</td>"
            f"<td>{esc(run.head_sha)}</td>"
            "</tr>"
            for run in gaps.missing
        )
        inner = (
            '<div class="scroll"><table>\n'
            '<tr><th>那一跑（run）</th><th>結論</th><th>開始</th><th>對著的提交（commit）</th></tr>\n'
            f"{rows}\n</table></div>"
        )
    return (
        "<details>"
        "<summary>看缺收據的那幾跑</summary>"
        '<div class="fold">'
        f'<p class="meta">收據讀自 {esc(gaps.source)}。收據是雲端那一跑留下的成績單。'
        "比對的是收據裡的 run id；還在跑的那一跑不算（它本來就還沒有收據）。</p>\n"
        f"{inner}</div></details>"
    )


def receipts_light(gaps: ReceiptGaps) -> str:
    """第二盞健康燈：主線最近那幾跑都是不是每一跑都有收據。"""
    if gaps.missing:
        names = "、".join(f"run {run.run_id}" for run in gaps.missing)
        return (
            f'<div class="light bad">收據（雲端那一跑留下的成績單）不齊：最近 {gaps.looked} '
            f"跑裡，有 {len(gaps.missing)} 跑在收據分支上沒有收據——{names}。"
            "那一跑等於沒有留下每一步的離開碼，收據不是掉了就是根本沒寫成。</div>\n"
            + receipts_details(gaps)
        )
    return (
        f'<div class="light ok">收據（雲端那一跑留下的成績單）齊全：最近 {gaps.looked} '
        "跑每一跑都有收據。</div>\n"
        + receipts_details(gaps)
    )


def broken_tickets(review: ClosedReview) -> list[ClosedTicket]:
    """最近關掉的票裡，串不起綠收據的（`problem` 非空，含人手關的那幾張）。"""
    return [t for t in review.tickets if t.problem]


def closed_details(review: ClosedReview) -> str:
    """第三盞燈折起來的細節：逐票三格（做掉的 PR、那顆 commit、綠收據）加關掉的時間。"""
    if not review.tickets:
        inner = "<p>最近一張關掉的票都沒有。</p>"
    else:
        rows = "\n".join(
            "<tr>"
            f'<td><a href="{esc(t.url)}">#{t.number}</a></td>'
            f"<td>{esc(t.title)}</td>"
            f"<td>{esc(t.closed)}</td>"
            f"{closed_cells(t)}"
            "</tr>"
            for t in review.tickets
        )
        inner = (
            '<div class="scroll"><table>\n'
            "<tr><th>票</th><th>標題</th><th>關掉的時間</th><th>做掉它的 PR</th>"
            "<th>併出來那一顆提交（commit）</th><th>綠收據</th></tr>\n"
            f"{rows}\n</table></div>"
        )
    return (
        "<details>"
        "<summary>看逐票三格（做掉的 PR、那顆提交（commit）、綠收據）</summary>"
        '<div class="fold">'
        f'<p class="meta">收據讀自 {esc(review.source)}。綠是從收據的欄位重算的：那一跑綠、'
        "verify 那個工作（job）綠、而且每一支檢查的離開碼都是 0——不看收據自報的那一欄。</p>\n"
        f"{inner}</div></details>"
    )


def closed_light(review: ClosedReview) -> str:
    """第三盞健康燈：關掉的票串不起綠收據的（`problem` 非空）有沒有。

    句子分兩段：人手關的幾張、有 PR 但收據不綠的幾張，各列票號；兩種都 0 才綠。
    """
    broken = broken_tickets(review)
    if broken:
        by_hand = [t for t in broken if t.pr_number == 0]
        with_pr = [t for t in broken if t.pr_number != 0]
        parts: list[str] = []
        if by_hand:
            names = "、".join(f"#{t.number}" for t in by_hand)
            parts.append(f"人手關的有 {len(by_hand)} 張（{names}）")
        if with_pr:
            names = "、".join(f"#{t.number}" for t in with_pr)
            parts.append(f"有 PR 做掉它、但收據串不起來的有 {len(with_pr)} 張（{names}）")
        sentence = "；".join(parts)
        return (
            f'<div class="light bad">關掉的票串不起綠收據的有 {len(broken)} 張：'
            f"{sentence}——票關了，東西有沒有進主線只能靠人記得。</div>\n"
            + closed_details(review)
        )
    return (
        f'<div class="light ok">關掉的票串不起綠收據的有 {len(broken)} 張——'
        "最近關掉的每一張票都有一個合併請求（PR）做掉它、而且收據是綠的。</div>\n"
        + closed_details(review)
    )


def health_block(data: PageData) -> str:
    """三盞健康燈，每盞一句人話，細節折在 `<details>` 裡。"""
    return (
        "<h2>健康燈</h2>\n"
        + cloud_light(data.cloud)
        + receipts_light(data.receipt_gaps)
        + closed_light(data.closed_review)
    )


# ── 6. 路線走到哪 ───────────────────────────────────────────────────────────


def milestone_state_word(state: str) -> str:
    """里程碑的 state 那一格：closed／open 印成人話「關了」「開著」。"""
    return {"closed": "關了", "open": "開著"}.get(state, state)


def milestones_block(rows: Sequence[Milestone]) -> str:
    """里程碑進度條：一條一行，寬度用純 CSS 的百分比，不靠圖片。"""
    head = "<h2>路線走到哪（里程碑 milestone，一段路線）</h2>\n"
    if not rows:
        return head + "<p>GitHub 上一條里程碑都沒有。</p>"
    lines: list[str] = []
    for ms in rows:
        total = ms.open_issues + ms.closed_issues
        done = round(100 * ms.closed_issues / total) if total else 0
        link = f'<a href="{esc(ms.url)}">{esc(ms.title)}</a>' if ms.url else esc(ms.title)
        lines.append(
            f'<div class="ms">{link}　'
            f'<span class="meta">{esc(milestone_state_word(ms.state))}</span>　'
            f"關掉 {ms.closed_issues}／還開著 {ms.open_issues}　"
            f'<span class="meta">{done}%</span>'
            f'<div class="bar"><span style="width:{done}%"></span></div></div>'
        )
    return head + "\n".join(lines)


# ── 7. 規矩與藍圖（折起來） ─────────────────────────────────────────────────


def cards_block(cards: Sequence[RuleCard], lessons_total: int) -> str:
    """立了幾張卡、每張卡人話第一句、以及總共對到上一代幾件事故。"""
    debts = {debt for card in cards for debt in card.blood_debt}
    lines = "\n".join(
        f"<li><strong>{esc(card.card_id)}</strong>：{esc(one_sentence(card.human))}</li>"
        for card in cards
    )
    return (
        f"<h3>規矩卡（{len(cards)} 張，每一張都有檢查程式與必紅樣本）</h3>\n"
        f"<p>這 {len(cards)} 張卡總共對到上一代 {len(debts)} 件事故"
        f"（v2 事故庫一共 {lessons_total} 件）。</p>\n"
        f"<ul>{lines}</ul>"
    )


def blueprint_block(data: PageData) -> str:
    """當初規劃的那幾張卡各自的去向：可行性分組，以及暫緩的卡在等什麼。"""
    bp = data.blueprint
    groups = "".join(
        f'<span class="tag">{esc(name)}：{count} 張</span>' for name, count in bp.groups
    )
    waiting = "".join(
        f"<li><strong>{esc(blocker)}</strong>：{esc('、'.join(ids))}</li>"
        for blocker, ids in bp.waiting
    )
    tickets = [t for t in data.tickets if LABEL_DEFERRED in t.labels]
    return (
        f"<h3>當初規劃的 {bp.total} 張卡走到哪</h3>\n"
        f"<p>{groups}</p>\n"
        f'<p class="meta">分組來自 blueprint/cards-38.json 的 feasibility_v2（pass＝可以立、'
        f"deferred＝暫緩、dropped＝丟掉）與 blocked_on（在等什麼），這一頁只讀不寫。"
        f"同一組也各有一張 issue（待辦票），標籤 {esc(LABEL_DEFERRED)}，"
        f"現在有 {len(tickets)} 張。</p>\n"
        f"<p>暫緩的卡在等什麼：</p>\n<ul>{waiting}</ul>"
    )


def rules_block(data: PageData, today: date) -> str:
    """規矩與當初規劃的卡整節：折在 `<details>` 裡的完整清單、時間軸、藍圖去向。"""
    return (
        f"<h2>規矩與當初規劃的 {data.blueprint.total} 張卡</h2>\n"
        "<details>\n"
        "<summary>展開：立好的規矩卡、時間軸、與當初規劃的卡的去向</summary>\n"
        '<div class="fold">\n'
        + cards_block(data.cards, data.lessons_total)
        + "\n"
        + timeline_block(data.cards, today)
        + "\n"
        + blueprint_block(data)
        + "\n</div>\n</details>"
    )


# ── 頁首與頁尾 ─────────────────────────────────────────────────────────────


def head_block() -> str:
    """頁首：標題與警告那一行。算出時間等瑣碎放到頁尾。"""
    return (
        f"<h1>aosr-v3 現況</h1>\n"
        f'<div class="meta">{esc(DO_NOT_EDIT)}</div>'
    )


def mainline_block(data: PageData) -> str:
    """主線那一節（放在頁尾上面）；落後 0 筆以上就紅字說「這一頁是舊的」。"""
    state = data.repo_state
    stale = ""
    if state.behind > 0:
        stale = '<p class="bad">這一頁是舊的——算它的那棵樹落後主線。</p>\n'
    return (
        f"<h2>主線（repo：{esc(data.repo)}）</h2>\n"
        f"{stale}"
        f"<p>算這一頁的那棵樹在 <strong>{esc(state.branch)}</strong>："
        f"比 origin/main（GitHub 上的主線）領先 {state.ahead} 筆、落後 {state.behind} 筆。</p>\n"
        f"<p>最新一筆（提交）：<code>{esc(state.head_sha)}</code>　{esc(state.head_subject)}"
        f"　（{esc(state.head_when)}）</p>"
    )


def foot_block(data: PageData) -> str:
    """頁尾：算出時間、算這一頁的是哪一跑、反映主線哪一筆。"""
    return (
        '<p class="foot">'
        f"算出時間：{esc(data.computed_at)}（台北時間）。"
        f"算這一頁的是：{esc(data.computed_by)}。"
        f"這一頁反映的是：{esc(data.reflects)}。"
        f"{esc(DO_NOT_EDIT)}。"
        "這一頁由機器算出，每次併入主線由雲端重算重推；"
        "算不出資料的時候它回離開碼 2（工具自壞），不產一頁沒資料的。"
        '固定網址：<a href="' + esc(data.page_url) + '">' + esc(data.page_url) + "</a></p>"
    )


# ── 共用：時間軸 ───────────────────────────────────────────────────────────


def timeline_days(cards: Sequence[RuleCard]) -> list[tuple[str, int]]:
    """時間軸要畫的點：哪一天、那天有幾張卡進主線（算不出日期的卡不畫）。"""
    tally: dict[str, int] = {}
    for card in cards:
        if card.merged_day:
            tally[card.merged_day] = tally.get(card.merged_day, 0) + 1
    return sorted(tally.items())


def _x_of(day: str, first: date, span: int) -> int:
    """一天在時間軸上的橫座標。"""
    try:
        moment = date.fromisoformat(day)
    except ValueError:
        return AXIS_LEFT
    offset = (moment - first).days
    return AXIS_LEFT + round((AXIS_RIGHT - AXIS_LEFT) * offset / span)


def timeline_svg(days: Sequence[tuple[str, int]], today: date) -> str:
    """一條時間線：每一天一個點，點上寫那天進了幾張卡。純 inline SVG，不載外部資源。"""
    if not days:
        return "<p>還沒有一張卡有算得出來的進主線日期。</p>"
    first = date.fromisoformat(days[0][0])
    last = max(date.fromisoformat(days[-1][0]), today)
    span = max((last - first).days, 1)
    parts = [
        f'<svg viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" role="img" '
        f'aria-label="規矩卡進主線的時間軸">',
        f'<line class="tl-axis" x1="{AXIS_LEFT}" y1="{AXIS_Y}" x2="{AXIS_RIGHT}" y2="{AXIS_Y}"/>',
    ]
    placed = -999
    for day, count in days:
        x = _x_of(day, first, span)
        parts.append(f'<circle class="tl-dot" cx="{x}" cy="{AXIS_Y}" r="7"/>')
        if x - placed < 90:  # 太近就只畫點不寫字，免得字疊在一起
            continue
        placed = x
        parts.append(
            f'<text class="tl-day" x="{x}" y="{AXIS_Y - 22}" text-anchor="middle">{esc(day)}</text>'
        )
        parts.append(
            f'<text class="tl-count" x="{x}" y="{AXIS_Y + 28}" text-anchor="middle">'
            f"{count} 張</text>"
        )
    parts.append(
        f'<text class="tl-count" x="{AXIS_RIGHT}" y="{AXIS_Y + 46}" text-anchor="end">'
        f"今天 {today.isoformat()}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def timeline_block(cards: Sequence[RuleCard], today: date) -> str:
    """時間軸那一節：圖，加一份按日期分組的明細（圖上塞不進 17 個名字）。"""
    days = timeline_days(cards)
    detail: list[str] = []
    for day, _ in days:
        names = "、".join(card.card_id for card in cards if card.merged_day == day)
        detail.append(f"<li>{esc(day)}：{esc(names)}</li>")
    return (
        "<h3>時間軸：卡是哪一天進主線的</h3>\n"
        + timeline_svg(days, today)
        + f"\n<ul>{''.join(detail)}</ul>"
    )


# ── 關票三格（細節表共用） ─────────────────────────────────────────────────


def closed_cells(ticket: ClosedTicket) -> str:
    """一張關掉的票在表上的後三格：做掉它的 PR、那一顆 commit、綠收據。"""
    pull = (
        f'<a href="{esc(ticket.pr_url)}">#{ticket.pr_number}</a>'
        if ticket.pr_number
        else '<span class="bad">沒有</span>'
    )
    sha = f"<code>{esc(ticket.merge_sha)}</code>" if ticket.merge_sha else "—"
    if ticket.receipt_green:
        receipt = (
            f'<span class="ok">綠</span>　'
            f'<a href="{esc(ticket.receipt_url)}">雲端那一跑（run {ticket.receipt_run_id}）</a>'
        )
    else:
        receipt = f'<span class="bad">{esc(ticket.problem)}</span>'
    return f"<td>{pull}</td><td>{sha}</td><td>{receipt}</td>"


def render_page(data: PageData, today: date) -> str:
    """整頁 HTML。這一支只排版，不算任何東西。"""
    body = "\n".join(
        [
            head_block(),
            today_block(data),
            decisions_block(data.tickets),
            recently_closed_block(data.closed_review),
            in_progress_block(data.tickets),
            health_block(data),
            milestones_block(data.milestones),
            rules_block(data, today),
            mainline_block(data),
            foot_block(data),
        ]
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>aosr-v3 現況（{esc(data.computed_at)}）</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n"
        f'<div class="wrap">\n{body}\n</div>\n</body>\n</html>\n'
    )
