#!/usr/bin/env python3
"""票只准由 PR 關：人手關掉的票，這一支當場重開並留一句人話。

跑法（唯一一種，環境交給 uv 管——規矩卡 uv-single-entrypoint）：

    uv run python -m governance.status.issue_guard --issue 83

雲端由 `.github/workflows/issue-guard.yml` 掛 `on: issues: [closed]` 叫起來：票
（issue，GitHub 上的待辦票）一被關就跑一次。決策紙
`docs/decisions/issues-closed-only-by-merged-pr.md`：票只准被**合進主線**的 PR
（合併請求）關掉，PR 內文寫 `Closes #n`，合併時 GitHub 自己關票並記下那條連結。

**判準讀的是 GitHub 自己記的關票來源，不是猜的。** 那條連結在 GraphQL（GitHub 的圖查詢
介面）上叫 `closedByPullRequestsReferences`，狀態頁那一格（「關掉的票對不對得到綠收據」）
已經有一支 :func:`governance.status.collect.closing_pulls` 在問它，這裡共用同一支，不抄第二份
——兩份查詢會各自漂，而「哪個 PR 做掉它」這件事只該有一個答案。

**離開碼只有兩種：0 與 2。沒有 1。** 這支程式不下判決（它不是規矩卡的檢查程式），
它只做兩件事之一：有合進主線的 PR 關掉這張票就什麼都不做，沒有就重開加留言，兩條路都回 0。
`gh`（GitHub 的命令列工具）不在、沒登入、GitHub 回不了話、回來的形狀看不懂，一律回 2，
而且**不重開也不留言**——分不清的時候亂重開，比漏掉一張更糟。

**它為什麼住在 `governance/status/` 而不是 `governance/checks/`。** 那一層是「准上網讀
GitHub」的一層（狀態頁與收據都住那裡），檢查程式那一層刻意不上網。這一支不但要上網讀，
還要寫回去（重開票），是這個 repo 裡第一支會回頭改 GitHub 上東西的機器；權限
（`issues: write`）只給它自己那一份 workflow，其餘不給。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken, note, repo_root
from governance.status.collect import GH, ClosingPulls, Hub, Shell, closing_pulls, read_slug

# 重開時留的那一句人話。給的是下一步怎麼做，不是罵人：關票的正路只有一條。
REOPEN_COMMENT = "票只准由 PR 關：PR 內文寫 Closes #n，合併時 GitHub 會自動關。這張已重開。"


@dataclass(frozen=True)
class Verdict:
    """該不該重開，以及為什麼。`why` 那一句會原樣寫進這一跑的輸出（給人看的紀錄）。"""

    reopen: bool
    why: str


def judge(number: int, closing: ClosingPulls) -> Verdict:
    """給 GitHub 記的關票來源，回「該不該重開」。

    這一半刻意不碰網路也不碰子程序（吃的是已經讀回來、收窄過的 `ClosingPulls`），
    所以測得動：測試餵一段假的圖查詢回應進 :func:`governance.status.collect.closing_pulls`，
    再把結果交給這裡，全程不上網。

    三種結局的話不一樣，因為它們是三件不同的事：合進主線了（不動它）、
    掛著關它的 PR 但沒有一個合進主線（重開）、根本沒有 PR（人手關的，重開）。
    """
    landed = closing.landed
    if landed is not None:
        return Verdict(
            reopen=False,
            why=f"#{number} 是 PR #{landed.number} 關掉的，那個 PR 合進主線了（{landed.url}）——不動它",
        )
    if closing.referenced:
        return Verdict(
            reopen=True,
            why=f"#{number} 上掛著 {closing.referenced} 個關它的 PR，但沒有一個合進主線——重開",
        )
    return Verdict(
        reopen=True,
        why=f"#{number} 上一個掛著關它的 PR 都沒有（人手關的）——重開",
    )


def verdict_for(hub: Hub, number: int) -> Verdict:
    """問 GitHub 這張票是被哪個 PR 關掉的，再判該不該重開。"""
    return judge(number, closing_pulls(hub, number))


def reopen(shell: Shell, slug: str, number: int, comment: str) -> None:
    """重開一張票，順手留那一句人話。`gh` 回非零就 ToolBroken（這一跑不算數）。"""
    shell.out(
        [GH, "issue", "reopen", str(number), "--repo", slug, "--comment", comment],
        f"重開 #{number} 並留言",
    )


def guard(
    root: Path, env: Mapping[str, str], number: int, timeout: int, per_page: int, comment: str
) -> Verdict:
    """一張票走完一輪：問、判、必要時重開。回那一次的判詞。"""
    shell = Shell(cwd=root, timeout=timeout)
    slug = read_slug(shell, env)
    hub = Hub(shell=shell, slug=slug, per_page=per_page)
    verdict = verdict_for(hub, number)
    note(verdict.why)
    if verdict.reopen:
        reopen(shell, slug, number, comment)
        note(f"已重開 {slug}#{number}，留言：{comment}")
    return verdict


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """命令列參數。逾時秒數與一頁幾筆也在這裡宣告，不寫成程式裡的常數。"""
    parser = argparse.ArgumentParser(
        description="票只准由合進主線的 PR 關：人手關掉的當場重開並留言（問不出答案就回 2，不動手）",
    )
    parser.add_argument("--issue", required=True, type=int, help="剛被關掉的那張票的號碼")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="每一個外部指令的看門狗秒數；逾時就回 2（這一跑不算數，也不重開）",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="問 GitHub 清單的時候一頁抓幾筆；不管填多少都會翻到底，這個數字只決定翻幾次",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """入口。ToolBroken 一律翻成離開碼 2，而且那條路上一張票都不會被重開。"""
    args = parse_args(argv)
    try:
        guard(
            repo_root(),
            os.environ,
            args.issue,
            args.timeout,
            args.per_page,
            REOPEN_COMMENT,
        )
    except ToolBroken as exc:
        note(f"問不出這張票是被誰關的，離開碼 2（工具自壞），不重開也不留言：{exc}")
        return TOOL_BROKEN
    return CLEAN


if __name__ == "__main__":
    sys.exit(main())
