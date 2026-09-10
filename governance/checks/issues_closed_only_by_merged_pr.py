#!/usr/bin/env python3
"""規矩卡 issues-closed-only-by-merged-pr：人手關的票超過零張就紅。

票（issue，GitHub 上的待辦票）只准由**合進主線**的 PR（合併請求）關掉：PR 內文寫
``Closes #n``，合併的時候 GitHub 自己關票，順手把兩邊連起來。那條連結在 GraphQL
（GitHub 的圖查詢介面）上叫 ``closedByPullRequestsReferences``，是機器記下來的事實，
不是「記得在 PR 裡提票號」那種習慣。這支檢查把所有關掉的票問回來，每一張看有沒有一個
``merged`` 為真、而且 ``baseRefName``（合去哪一條分支）等於卡上登記的主線分支名的 PR；
沒有的就是人手關的，列成 HIT，一張都不准有。

**判準不抄第二份。** 欄位名、要讀哪幾格、以及「這個 PR 算不算落地」那一支判斷，
全部從 ``governance/status/collect.py`` 借來——狀態頁那一格（關掉的票對不對得到綠收據）
與票務守衛 ``governance/status/issue_guard.py`` 用的是同一份。這裡只換一樣東西：主線分支名
從卡上的 ``[settings]`` 來，不是模組常數，樣本樹才改得動它（見那幾份必紅樣本）。

**這是第二支准上網的檢查。** 決策紙 ``docs/decisions/hand-closed-issues-block-merge.md``：
只准讀、放行登記在卡的 ``[[settings.allow]]`` 並帶到期日。放行有牙——這裡先讀那一筆放行，
缺席或過期就回 2、不上網。同一張表裡帶票號那幾筆是**具名放過某幾張票**，過期的當沒放行。

**回 2 的路（這一跑不算數，不准當綠）：** ``gh`` 不在 PATH 上、沒 token、API 回不了話或
逾時、回來的不是 JSON、形狀跟 GitHub 文件對不上、某一張票掛著的關票 PR 一頁看不完、
卡讀不到或准上網那筆放行缺席／過期。

**刻意不判的：** 這張票該不該被關（機器判不出來）；關它的那個 PR 有沒有一份綠收據
（那是狀態頁那一格的事，鏈更長，而且不擋合併）。門檻與期望只住在卡上，這裡沒有任何數字。
"""
from __future__ import annotations

import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from governance import gh_replay
from governance.exit_codes import ToolBroken, note, run
from governance.loader import (
    EXEMPTION_KEYS,
    RULES_DIR,
    exemption_field_problems,
    setting_int,
    setting_tables,
    setting_text,
)
from governance.status.collect import (
    CLOSED_BY_FIELD,
    CLOSED_BY_NODES,
    Hub,
    MergedPull,
    Shell,
    as_rows,
    as_table,
    field_int,
    field_text,
    graphql,
    landed_pull,
    nested,
    owner_and_name,
)

CHECK_REL = "governance/checks/issues_closed_only_by_merged_pr.py"
ALLOW_KEY = "allow"
# 放行條目裡「准上網」那一格的鍵。找的是帶這一格的那一筆，不是任何一筆 allow。
NETWORK_KEY = "network"
# 放行條目裡「具名放過這一張票」那一格的鍵（值是票號）。
ISSUE_KEY = "issue"

# 一句圖查詢就把「關掉的票」與「每一張掛著關它的那幾個 PR」一起問回來。
# 為什麼不是一張票問一次：那樣一次掃描要打幾十支 API，四個回合的後設測試加起來就是幾百支，
# 而這一句一頁抓得完。裡外兩層共用同一個 `$size`（一頁幾筆，卡上登記）。
# 欄位名與欄位清單是從狀態頁那一層借來的常數，不是這裡另抄的字串。
CLOSED_ISSUES_QUERY = f"""
query($owner:String!,$name:String!,$size:Int!,$after:String){{
  repository(owner:$owner,name:$name){{
    issues(states:CLOSED, first:$size, after:$after){{
      pageInfo{{ hasNextPage endCursor }}
      nodes{{ number title url
        {CLOSED_BY_FIELD}(first:$size, includeClosedPrs:true){{ {CLOSED_BY_NODES} }}
      }}
    }}
  }}
}}
"""


@dataclass(frozen=True)
class Expected:
    """卡上登記的期望：問哪個 repo、主線叫什麼、等多久、一頁幾筆。"""

    repo: str
    main_branch: str
    timeout: int
    per_page: int


@dataclass(frozen=True)
class Ticket:
    """一張關掉的票，加上 GitHub 記的關票來源收窄之後的結果。

    ``landed`` 是合進主線、而且掛著關掉它的那個 PR（好幾個就留最後看到的一個——這一格問的是
    「有沒有落地」，不是「哪一筆落地」）；``referenced`` 是掛著關掉它的 PR 總數（含還開著、
    關掉沒合的）。兩個都要，才分得出「根本沒有 PR」跟「有 PR 但沒合進主線」這兩種不一樣的斷法。
    """

    number: int
    title: str
    url: str
    landed: MergedPull | None
    referenced: int


# ── 讀卡 ──────────────────────────────────────────────────────────────────────


def _card_path(scan_root: Path, files: list[Path]) -> Path:
    """掃描根自己的 governance/rules/ 底下宣告 check 是這支的那張卡。要剛好一張。"""
    rules_dir = scan_root / RULES_DIR
    mine: list[Path] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append(path)
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——期望只寫在卡上，讀不到卡這一跑就不算數"
        )
    return mine[0]


def _settings(card: Path, scan_root: Path) -> dict[str, object]:
    data = tomllib.loads(card.read_bytes().decode("utf-8"))
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{card.relative_to(scan_root)} 沒有 [settings] 表（期望全寫在那裡）")
    return settings


def read_expected(settings: Mapping[str, object]) -> Expected:
    """把卡上的期望讀成一個具體型別。少一格、形狀不對，loader 的讀值函式自己就回 2。"""
    return Expected(
        repo=setting_text(settings, "repo"),
        main_branch=setting_text(settings, "main_branch"),
        timeout=setting_int(settings, "api_timeout_seconds"),
        per_page=setting_int(settings, "per_page"),
    )


def network_allowed(settings: Mapping[str, object], where: str, today: date) -> str:
    """找那一筆「准上網」的放行。缺席、欄位不齊、過期，一律回 2——不上網。回那一筆的 reason。"""
    entries = [e for e in setting_tables(settings, ALLOW_KEY) if NETWORK_KEY in e]
    if len(entries) != 1:
        raise ToolBroken(
            f"{where} 的 [[settings.allow]] 裡帶 {NETWORK_KEY} 那一格的放行有 {len(entries)} 筆，"
            "要剛好 1 筆——沒有登記的放行就不准上網（決策紙 hand-closed-issues-block-merge）"
        )
    entry = entries[0]
    problems = exemption_field_problems(entry, f"{where} 的准上網放行")
    if problems:
        raise ToolBroken("；".join(problems))
    expires = date.fromisoformat(str(entry["expires"]).strip())
    if expires < today:
        raise ToolBroken(
            f"{where} 的准上網放行 {expires} 已到期（今天 {today}）——到期就不准上網，"
            "要續就重新看它還需不需要、範圍有沒有變寬，改卡走 PR"
        )
    return str(entry[EXEMPTION_KEYS[0]])


def excused_tickets(settings: Mapping[str, object], where: str, today: date) -> dict[int, str]:
    """具名放過的那幾張票（票號 → 理由）。過期的當沒放行，而且要說出來。"""
    out: dict[int, str] = {}
    for entry in setting_tables(settings, ALLOW_KEY):
        if ISSUE_KEY not in entry:
            continue
        number = entry[ISSUE_KEY]
        if isinstance(number, bool) or not isinstance(number, int):
            raise ToolBroken(f"{where} 的具名放行 {ISSUE_KEY} 必須是整數票號，實際是 {number!r}")
        problems = exemption_field_problems(entry, f"{where} 具名放過 #{number} 那一筆")
        if problems:
            raise ToolBroken("；".join(problems))
        expires = date.fromisoformat(str(entry["expires"]).strip())
        if expires < today:
            note(f"具名放過 #{number} 的那一筆放行 {expires} 已過期（今天 {today}），當沒放行")
            continue
        out[number] = str(entry[EXEMPTION_KEYS[0]])
    return out


# ── 判斷那一半（不上網，測得動）────────────────────────────────────────────────


def read_ticket(row: Mapping[str, object], main_branch: str, what: str) -> Ticket:
    """一張關掉的票（連同它掛著的那幾個關票 PR）收窄成 :class:`Ticket`。"""
    number = field_int(row, "number")
    if not number:
        raise ToolBroken(f"{what} 回來的一張票沒有票號：{row!r}"[:300])
    refs = as_table(row.get(CLOSED_BY_FIELD), f"#{number} 的關票來源")
    info = as_table(refs.get("pageInfo"), f"#{number} 的關票來源翻頁資訊")
    if info.get("hasNextPage") is True:
        raise ToolBroken(
            f"#{number} 掛著關它的 PR 一頁看不完（{what}）——看不完就不出結論，"
            "要嘛把卡上那一頁幾筆調大，要嘛這張票的形狀本來就沒看過"
        )
    landed: MergedPull | None = None
    seen = 0
    for item in as_rows(refs.get("nodes"), f"#{number} 掛著的那幾個 PR"):
        node = as_table(item, "一個關票的 PR")
        seen += 1
        pull = landed_pull(node, main_branch)
        if pull is not None:
            landed = pull
    return Ticket(
        number=number,
        title=field_text(row, "title"),
        url=field_text(row, "url"),
        landed=landed,
        referenced=seen,
    )


def read_page(body: Mapping[str, object], main_branch: str, what: str) -> tuple[list[Ticket], str]:
    """一頁圖查詢回應 → 那一頁的每一張關掉的票，加上下一頁的游標（空字串就是翻完了）。

    這一半刻意不碰網路也不碰子程序（吃的是已經解好的一段回應），所以測得動：
    測試餵一段假的圖查詢回應進來，全程不上網。
    """
    block = nested(body, ("data", "repository", "issues"), what)
    tickets = [
        read_ticket(as_table(item, "一張關掉的票"), main_branch, what)
        for item in as_rows(block.get("nodes"), f"{what} 的那幾張票")
    ]
    info = as_table(block.get("pageInfo"), f"{what} 的翻頁資訊")
    cursor = field_text(info, "endCursor")
    return tickets, cursor if info.get("hasNextPage") is True and cursor else ""


def hand_closed(tickets: Sequence[Ticket], excused: Mapping[int, str]) -> list[str]:
    """哪幾張是人手關的。具名放過的那幾張不算違規，但要記一行說出來。

    兩種紅的話刻意不一樣：掛著關它的 PR 但沒有一個合進主線（有人開了還沒合），
    跟一個掛著關它的 PR 都沒有（根本沒人做）。混成同一句，看紀錄的人就分不出下一步該催誰。
    """
    hits: list[str] = []
    for ticket in sorted(tickets, key=lambda t: t.number):
        if ticket.landed is not None:
            continue
        if ticket.number in excused:
            note(f"具名放過 #{ticket.number}（{ticket.title}）：{excused[ticket.number]}")
            continue
        how = (
            f"掛著 {ticket.referenced} 個關它的 PR，但沒有一個合進主線"
            if ticket.referenced
            else "一個掛著關它的 PR 都沒有（人手關的）"
        )
        hits.append(
            f"#{ticket.number}（{ticket.title}）{how}——票只准由合進主線的 PR 關，"
            f"重開它再由 PR 關掉：{ticket.url}"
        )
    return hits


# ── 問伺服器 ───────────────────────────────────────────────────────────────────


class ReplayShell(Shell):
    """跟 :class:`Shell` 一樣去跑指令，只是先問一次這一跑的錄音（:mod:`governance.gh_replay`）。

    為什麼要有它：這張卡的後設測試每張卡六回合裡有三回合真的會問伺服器，再乘上「產收據那一
    跑」，同一句問題一次 ``uv run pytest`` 會問十幾次，答案卻是同一份。錄音只在後設測試設了
    ``AOSR_GH_REPLAY_DIR`` 的時候才有東西；雲端那一跑身上沒有那一格，每一句都真的去問伺服器。

    **判準沒有放寬**：重播之前一樣要求那支工具真的在 ``PATH`` 上（見 :func:`gh_replay.replay`），
    所以第 4 回合（把 ``gh`` 抽掉必須回 2）在有錄音的時候照樣回 2。

    只掛在這支檢查上、不掛進共用的 :class:`Shell`：狀態頁那條線與票務守衛的測試自己餵假回應，
    多一層快取會蓋掉它們餵的那一份。
    """

    def out(self, argv: Sequence[str], what: str) -> str:
        recorded = gh_replay.replay(argv, what)
        if recorded is not None:
            return recorded
        answer = super().out(argv, what)
        gh_replay.record(argv, answer)
        return answer


def read_closed_tickets(hub: Hub, main_branch: str) -> list[Ticket]:
    """把所有關掉的票（只有 issue，不含 PR）翻到底問回來。

    圖查詢的 ``issues`` 本來就只給 issue，PR 是另一個欄位——不必像 REST 那樣事後分掉。
    """
    owner, name = owner_and_name(hub.slug)
    what = f"問 {hub.slug} 關掉的票是被哪個 PR 關的"
    tickets: list[Ticket] = []
    cursor = ""
    while True:
        text = {"owner": owner, "name": name}
        if cursor:
            text["after"] = cursor
        body = graphql(hub, CLOSED_ISSUES_QUERY, what, text, {"size": hub.per_page})
        page, cursor = read_page(body, main_branch, what)
        tickets.extend(page)
        if not cursor:
            return tickets


# ── 外殼接線 ──────────────────────────────────────────────────────────────────


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：只有自己那張卡。GitHub 上那些票不是檔案。"""
    return [_card_path(scan_root, files)]


def check(scan_root: Path, files: list[Path]) -> list[str]:
    card = _card_path(scan_root, files)
    where = str(card.relative_to(scan_root))
    settings = _settings(card, scan_root)
    want = read_expected(settings)
    today = date.today()
    reason = network_allowed(settings, where, today)
    excused = excused_tickets(settings, where, today)
    note(f"准上網的放行在（{reason[:60]}…）；只讀 {want.repo} 關掉的票，主線是 {want.main_branch}")
    shell = ReplayShell(cwd=scan_root, timeout=want.timeout)
    hub = Hub(shell=shell, slug=want.repo, per_page=want.per_page)
    tickets = read_closed_tickets(hub, want.main_branch)
    hits = hand_closed(tickets, excused)
    note(f"關掉的票 {len(tickets)} 張，具名放過 {len(excused)} 張，人手關的 {len(hits)} 張")
    return hits


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="關掉的票都必須有一個合進主線的關票 PR：人手關的超過零張就紅，問不出來回 2",
            targets=targets,
        )
    )
