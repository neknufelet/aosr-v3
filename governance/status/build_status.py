#!/usr/bin/env python3
"""狀態頁的命令列入口。

跑法（唯一一種，環境交給 uv 管——規矩卡 uv-single-entrypoint）：

    uv run python -m governance.status.build_status --out <一個暫存目錄>

**產出物刻意不進版控。** 本機跑就寫到暫存目錄（老闆開工時看一眼），雲端那一跑寫完之後由
`.github/workflows/status.yml` 推到只有機器寫的 `status` 分支，GitHub Pages 對著那條分支掛。
理由在決策紙 `docs/decisions/status-page-not-in-main.md`：算得出來的東西存成檔就會過期，
而放在眼前就會有人手改。同一件事也由規矩卡 status-page-computed-not-typed 從另一頭守著
（手寫的進度檔不准進版控）。

**離開碼只有兩種。** 0 算出來了、2 算不出來（`gh` 不在、沒登入、GitHub 回不了話、
讀回來的東西形狀看不懂）。**沒有 1**：這支程式不下判決，所以沒有「抓到違規」這個結局。
算不出來的時候它不產檔——一頁沒資料的狀態頁看起來像「真的什麼都沒發生」，那比沒有頁更糟。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken, note, repo_root
from governance.status.collect import Shell, collect
from governance.status.render import render_page

# 這一頁的固定網址（GitHub Pages 對著 status 分支掛）。頁面自己要寫得出來，
# 不然老闆看到的那一份不知道自己該住在哪裡。
PAGE_URL = "https://neknufelet.github.io/aosr-v3/"

# 產出物的檔名。index.html 是 Pages 的門面；.nojekyll 叫 Pages 不要拿 Jekyll 再處理一次
# （這一頁是現成的 HTML，不需要它，而它會把開頭是底線的檔名吃掉）。
PAGE_FILE = "index.html"
NOJEKYLL_FILE = ".nojekyll"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """命令列參數。逾時秒數也在這裡宣告，不寫成程式裡的常數。"""
    parser = argparse.ArgumentParser(
        description="從版控與 GitHub 現算一頁狀態頁（算不出來就回 2，不產一頁沒資料的）",
    )
    parser.add_argument("--out", required=True, help="產出目錄（不准指到版控樹裡）")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="每一個外部指令的看門狗秒數；逾時就回 2（這一跑不算數）",
    )
    parser.add_argument("--page-url", default=PAGE_URL, help="這一頁掛出來的固定網址")
    return parser.parse_args(argv)


def assert_outside_repo(root: Path, out: Path) -> None:
    """產出目錄不准落在版控樹裡。

    落進去就是在版控樹裡留殘留：`file-placement-allowlist` 會咬根層多出來的目錄，
    `status-page-computed-not-typed` 會咬手寫的狀態檔——而這一頁本來就不該進主線。
    """
    target = out.expanduser().resolve()
    if target == root or root in target.parents:
        raise ToolBroken(
            f"產出目錄 {target} 落在版控樹（{root}）裡面——這一頁刻意不進主線，"
            "本機請寫到暫存目錄，雲端那一跑寫到 runner 的暫存目錄再推到 status 分支"
        )


def build(root: Path, out: Path, timeout: int, page_url: str) -> Path:
    """算出來、寫下去，回那份 HTML 的路徑。"""
    assert_outside_repo(root, out)
    shell = Shell(cwd=root, timeout=timeout)
    data = collect(root, shell, os.environ, page_url)
    page = render_page(data, date.today())
    out.mkdir(parents=True, exist_ok=True)
    target = out / PAGE_FILE
    target.write_text(page, encoding="utf-8")
    (out / NOJEKYLL_FILE).write_text("", encoding="utf-8")
    note(f"卡 {len(data.cards)} 張、開著的票 {len(data.tickets)} 張、里程碑 {len(data.milestones)} 條")
    note(f"算這一頁的是：{data.computed_by}")
    note(f"寫好了：{target}（{len(page)} 個字元）")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """入口。ToolBroken 一律翻成離開碼 2，而且不留下半份產出物。"""
    args = parse_args(argv)
    try:
        root = repo_root()
        build(root, Path(args.out), args.timeout, args.page_url)
    except ToolBroken as exc:
        note(f"算不出這一頁，離開碼 2（工具自壞）：{exc}")
        return TOOL_BROKEN
    return CLEAN


if __name__ == "__main__":
    sys.exit(main())
