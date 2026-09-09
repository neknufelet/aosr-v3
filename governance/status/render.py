"""把算好的資料排成一頁 HTML。

**這一層不碰網路、不碰子程序、不讀檔**，所以測得動：測試餵假資料進來，看出來的那份 HTML
有沒有該有的東西。

**頁面不准依賴外部資源。** 樣式、圖、時間軸全部寫在這一份檔裡（inline CSS ＋ inline SVG），
一個 `<script>`／`<link>`／`<img>` 都沒有：這一頁掛在 GitHub Pages 上給老闆看，網路上任何
一個第三方掛掉都不准讓它變成一片空白。同樣的道理寫在規矩卡 refs-and-links-resolve 與決策紙
「文件不准依賴外部資源」那一題底下。頁面裡唯一的 `http` 是連到 GitHub 的**超連結**
（點了才走），不是這一頁載入時要抓的東西。

全中文、白話、給老闆看：每個英文名字旁邊同一句要有中文說它在做什麼。
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from html import escape

from governance.status.model import CloudRun, Milestone, PageData, RuleCard, Ticket

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
--ok:#1a7f52;--bad:#b3261e;--warn:#8a6100;--dot:#2b5d9b;}
@media (prefers-color-scheme:dark){:root{--ink:#e8eaed;--dim:#a4abb5;--line:#3a3f47;
--bg:#15171a;--card:#1e2126;--ok:#4bbd85;--bad:#f2857c;--warn:#e0b155;--dot:#7fb0e8;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:15px;line-height:1.7;
font-family:-apple-system,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:28px 18px 64px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:19px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
p{margin:6px 0}
.warnbar{margin:10px 0 18px;padding:10px 14px;border-left:4px solid var(--warn);
background:var(--card);color:var(--ink);font-weight:700}
.meta{color:var(--dim);font-size:13px}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.tile{flex:1 1 190px;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px 14px}
.tile .n{font-size:28px;font-weight:700;line-height:1.2}
.tile .k{color:var(--dim);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:600;font-size:13px}
.ok{color:var(--ok);font-weight:700}
.bad{color:var(--bad);font-weight:700}
.warn{color:var(--warn);font-weight:700}
.scroll{overflow-x:auto}
ul{margin:6px 0;padding-left:22px}
li{margin:3px 0}
a{color:inherit;text-decoration:underline;text-underline-offset:2px}
.bar{height:9px;border-radius:5px;background:var(--line);overflow:hidden;margin-top:5px}
.bar span{display:block;height:100%;background:var(--ok)}
.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;
padding:0 8px;font-size:12px;color:var(--dim);margin-right:4px}
.foot{color:var(--dim);font-size:13px;margin-top:30px}
svg{width:100%;height:auto;display:block;background:var(--card);
border:1px solid var(--line);border-radius:10px}
.tl-axis{stroke:var(--line);stroke-width:2}
.tl-dot{fill:var(--dot);stroke:var(--card);stroke-width:2}
.tl-day{fill:var(--ink);font-size:13px}
.tl-count{fill:var(--dim);font-size:12px}
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


def tile(number: str, label: str) -> str:
    """一格數字卡。"""
    return f'<div class="tile"><div class="n">{esc(number)}</div><div class="k">{esc(label)}</div></div>'


def head_block(data: PageData) -> str:
    """頁首：警告那一行、算出時間、是哪一跑算的。"""
    return (
        f"<h1>aosr-v3 現況</h1>\n"
        f'<div class="warnbar">{esc(DO_NOT_EDIT)}</div>\n'
        f'<p class="meta">算出時間：{esc(data.computed_at)}（台北時間）</p>\n'
        f'<p class="meta">算這一頁的是：{esc(data.computed_by)}</p>\n'
        f'<p class="meta">這一頁反映的是：{esc(data.reflects)}</p>\n'
        f'<p class="meta">repo：{esc(data.repo)}　固定網址：'
        f'<a href="{esc(data.page_url)}">{esc(data.page_url)}</a></p>'
    )


def summary_block(data: PageData) -> str:
    """一眼看：四格數字。"""
    debts = {debt for card in data.cards for debt in card.blood_debt}
    cloud = data.cloud
    verdict = word_for(cloud.conclusion)[1] if cloud else "還沒有雲端紀錄"
    return (
        "<h2>一眼看</h2>\n"
        '<div class="tiles">\n'
        + tile(str(len(data.cards)), "已經立在主線上的規矩卡")
        + tile(f"{len(debts)} / {data.lessons_total}", "血債：對到的 v2 事故件數")
        + tile(str(dict(data.blueprint.groups).get("deferred", 0)), "藍圖裡還暫緩的卡")
        + tile(verdict, "主線最近一次 verify（雲端檢查）")
        + "\n</div>"
    )


def cards_block(cards: Sequence[RuleCard]) -> str:
    """規矩卡一張一行。"""
    rows = "\n".join(
        "<tr>"
        f"<td>{esc(card.card_id)}</td>"
        f"<td>{esc(one_sentence(card.human))}</td>"
        f"<td>{len(card.blood_debt)}</td>"
        f"<td>{esc(card.merged_day) or '算不出來'}</td>"
        "</tr>"
        for card in cards
    )
    return (
        f"<h2>規矩卡（{len(cards)} 張，每一張都有檢查程式與必紅樣本）</h2>\n"
        '<div class="scroll"><table>\n'
        "<tr><th>卡</th><th>它在管什麼</th><th>認領的 v2 事故</th><th>進主線那天</th></tr>\n"
        f"{rows}\n</table></div>"
    )


def blueprint_block(data: PageData) -> str:
    """藍圖 38 張卡的分組，以及暫緩的是哪幾張。"""
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
        f"<h2>藍圖：{bp.total} 張卡走到哪</h2>\n"
        f"<p>{groups}</p>\n"
        f'<p class="meta">分組來自 blueprint/cards-38.json 的 feasibility_v2（pass＝可以立、'
        f"deferred＝暫緩、dropped＝丟掉）與 blocked_on（在等什麼），這一頁只讀不寫。"
        f"同一組也各有一張 issue（待辦票），標籤 {esc(LABEL_DEFERRED)}，"
        f"現在有 {len(tickets)} 張。</p>\n"
        f"<p>暫緩的卡在等什麼：</p>\n<ul>{waiting}</ul>"
    )


def milestones_block(rows: Sequence[Milestone]) -> str:
    """里程碑進度線。"""
    if not rows:
        return "<h2>里程碑（milestone，一段路線）</h2>\n<p>GitHub 上一條里程碑都沒有。</p>"
    body: list[str] = []
    for ms in rows:
        total = ms.open_issues + ms.closed_issues
        done = round(100 * ms.closed_issues / total) if total else 0
        link = f'<a href="{esc(ms.url)}">{esc(ms.title)}</a>' if ms.url else esc(ms.title)
        body.append(
            "<tr>"
            f"<td>{link}</td>"
            f"<td>{esc(ms.state)}</td>"
            f"<td>關掉 {ms.closed_issues}／還開著 {ms.open_issues}"
            f'<div class="bar"><span style="width:{done}%"></span></div></td>'
            f"<td>{done}%</td>"
            "</tr>"
        )
    return (
        "<h2>里程碑（milestone，一段路線）走到哪</h2>\n"
        '<div class="scroll"><table>\n'
        "<tr><th>里程碑</th><th>開著還是關了</th><th>票數</th><th>做完幾成</th></tr>\n"
        + "\n".join(body)
        + "\n</table></div>"
    )


def tags_html(labels: Sequence[str]) -> str:
    """標籤那一格。一個標籤都沒有就寫一個破折號，不留空格。"""
    tags = [f'<span class="tag">{esc(name)}</span>' for name in labels]
    return "".join(tags) if tags else "—"


def ticket_rows(rows: Sequence[Ticket]) -> str:
    """一組票的表格內容。"""
    return "\n".join(
        "<tr>"
        f'<td><a href="{esc(t.url)}">#{t.number}</a></td>'
        f"<td>{esc(t.title)}</td>"
        f"<td>{tags_html(t.labels)}</td>"
        f"<td>{esc(t.updated)}</td>"
        "</tr>"
        for t in rows
    )


def group_block(title: str, rows: Sequence[Ticket]) -> str:
    """一組票（開著的 PR、要拍板的題、暫緩的卡、其他）。"""
    if not rows:
        return f"<p><strong>{esc(title)}</strong>：沒有。</p>"
    return (
        f"<p><strong>{esc(title)}</strong>（{len(rows)} 張）</p>\n"
        '<div class="scroll"><table>\n'
        "<tr><th>號碼</th><th>標題</th><th>標籤</th><th>最後動過</th></tr>\n"
        + ticket_rows(rows)
        + "\n</table></div>"
    )


def tickets_block(rows: Sequence[Ticket]) -> str:
    """開著的 PR（合併請求）與 issue（待辦票），依標籤分組。"""
    prs = [t for t in rows if t.is_pr]
    issues = [t for t in rows if not t.is_pr]
    decision = [t for t in issues if LABEL_DECISION in t.labels]
    deferred = [t for t in issues if LABEL_DEFERRED in t.labels]
    named = {t.number for t in decision} | {t.number for t in deferred}
    others = [t for t in issues if t.number not in named]
    return (
        "<h2>開著的 PR（合併請求）與 issue（待辦票）</h2>\n"
        + group_block("開著的 PR（合併請求，等人看等人合）", prs)
        + group_block("要老闆拍板的題（標籤 decision）", decision)
        + group_block("暫緩的卡在等什麼（標籤 deferred-cards，一組一張）", deferred)
        + group_block("其他還開著的票", others)
    )


def cloud_block(cloud: CloudRun | None) -> str:
    """每支檢查最近一次雲端結果：verify 那一跑的每一步。"""
    if cloud is None:
        return (
            "<h2>每支檢查最近一次雲端結果</h2>\n"
            '<p class="bad">主線上還沒有任何一次 verify（雲端檢查）的紀錄——'
            "沒有雲端紀錄就沒有綠可言（只認雲端）。</p>"
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
        "<h2>每支檢查最近一次雲端結果</h2>\n"
        f"<p>主線最近一次 verify（雲端檢查）：{state_html(cloud.conclusion)}"
        f"　狀態 {esc(cloud.status)}　開始 {esc(cloud.started)}　收工 {esc(cloud.finished)}"
        f"　整個 job 跑了 {cloud.job_seconds} 秒</p>\n"
        f'<p class="meta">那一跑是 <a href="{esc(cloud.url)}">run {cloud.run_id}</a>，'
        f"對著 commit {esc(cloud.head_sha)}。這一頁只認雲端那一跑的結論，本機跑綠不算數。</p>\n"
        '<div class="scroll"><table>\n'
        "<tr><th>那一跑的每一步（中文那幾步就是一支一支的檢查）</th><th>結論</th>"
        "<th>跑了多久</th></tr>\n"
        f"{steps}\n</table></div>"
    )


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
        "<h2>時間軸：卡是哪一天進主線的</h2>\n"
        + timeline_svg(days, today)
        + f"\n<ul>{''.join(detail)}</ul>"
    )


def repo_block(data: PageData) -> str:
    """主線領先落後、最新一筆。"""
    state = data.repo_state
    return (
        "<h2>主線</h2>\n"
        f"<p>算這一頁的那棵樹在 <strong>{esc(state.branch)}</strong>："
        f"比 origin/main（GitHub 上的主線）領先 {state.ahead} 筆、落後 {state.behind} 筆。</p>\n"
        f"<p>最新一筆：<code>{esc(state.head_sha)}</code>　{esc(state.head_subject)}"
        f"　（{esc(state.head_when)}）</p>"
    )


def foot_block(data: PageData) -> str:
    """頁尾：這一頁怎麼來的。"""
    return (
        '<p class="foot">'
        f"{esc(DO_NOT_EDIT)}。這一頁由 governance/status/build_status.py 從版控與 GitHub 現算，"
        "每次併入主線由雲端重算重推；算不出資料的時候它會回離開碼 2（工具自壞），"
        "不會產一頁沒資料的。決策紙：狀態頁不進主線、機器算出來的狀態頁掛 GitHub Pages。"
        f"　算這一頁的是：{esc(data.computed_by)}</p>"
    )


def render_page(data: PageData, today: date) -> str:
    """整頁 HTML。這一支只排版，不算任何東西。"""
    body = "\n".join(
        [
            head_block(data),
            summary_block(data),
            cloud_block(data.cloud),
            repo_block(data),
            milestones_block(data.milestones),
            tickets_block(data.tickets),
            cards_block(data.cards),
            timeline_block(data.cards, today),
            blueprint_block(data),
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
