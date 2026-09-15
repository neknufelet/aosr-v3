#!/usr/bin/env python3
"""合併請求內文的驗收結論要指到要合的那顆提交。

守的形狀：驗收沒過、工人修正之後**沒再驗就合**（票 #297）。機器抓得到的是「自報要對得上」：
合併請求的內文必須有一行「通過的前綴 ＋ 完整提交 id」（前綴登記在卡的 ``verdict_pass_prefix``），
那顆提交必須等於這支合併請求此刻的 head；驗收之後又推了新提交，head 變了、那一行就對不上，紅。抓不到「沒驗卻寫通過」——那要靠
驗收席有自己的身分（第二階段，卡面照實寫）。

**讀哪裡**
* 真的 git 工作樹：只在 ``GITHUB_EVENT_NAME=pull_request`` 時判，事件檔從 ``GITHUB_EVENT_PATH``
  讀（GitHub 給每一步的 JSON，不上網）；head 用 ``pull_request.head.sha``，**不拿**
  ``GITHUB_SHA``——那是 GitHub 合出來的假想合併提交，永遠不等於內文寫的 head。
  不是 pull_request 事件（push 到 main、本機）沒有內文可判，回 0 並記一行；空範圍是有效範圍。
* 必紅樣本：樣本樹裡放一份卡上 ``fixture_event_file`` 登記的事件檔（跟 GitHub 事件檔同形狀），
  檢查照它判。真的 git 工作樹的根出現那份檔一律回 2——不然把它擺進主線就能繞過真事件。

**判準**（全部從卡的 ``[settings]`` 讀，程式裡沒有預設值）
* 內文裡以 ``verdict_pass_prefix`` 開頭的行要剛好一行；零行紅（沒宣告驗收），兩行以上紅
  （兩個宣告不知道信哪個）。
* 那一行的第二個字段必須是 ``sha_hex_length`` 位十六進位，而且等於 head（比對前轉小寫）。
* 以 ``verdict_fail_prefix`` 開頭的行出現就紅——最後一次驗收不通過不准合。

**回 2 的情況**：讀不到卡的 ``[settings]``、事件檔不存在或剖不開、事件裡沒有 ``pull_request``、
head 不是完整提交 id、git 不在 PATH（判「這是不是真的工作樹」要用它）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

from governance.exit_codes import ToolBroken, note, run

CARD_ID = "acceptance-verdict-points-at-head"
RULES_DIR = "governance/rules"
EVENT_PATH_ENV = "GITHUB_EVENT_PATH"
EVENT_NAME_ENV = "GITHUB_EVENT_NAME"
PR_EVENT_NAME = "pull_request"
SETTINGS_KEYS = (
    "verdict_pass_prefix",
    "verdict_fail_prefix",
    "sha_hex_length",
    "fixture_event_file",
)


class Settings(NamedTuple):
    """判準。**全部從卡的 ``[settings]`` 讀。**"""

    pass_prefix: str
    fail_prefix: str
    sha_hex_length: int
    fixture_event_file: str


def _card_path(scan_root: Path, files: list[Path]) -> Path:
    """掃描根底下 id 是這張卡的那一張。找不到、找到多張、讀不開，一律 ToolBroken。"""
    mine: list[Path] = []
    for path in sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("id") == CARD_ID:
            mine.append(path)
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好一張"
            "——判準只寫在卡上，讀不到這一跑就不算數"
        )
    return mine[0]


def read_settings(scan_root: Path, files: list[Path]) -> Settings:
    """從卡的 ``[settings]`` 讀判準：兩個前綴、提交 id 幾位、樣本事件檔住哪。"""
    data = tomllib.loads(_card_path(scan_root, files).read_bytes().decode("utf-8"))
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"id={CARD_ID} 的卡沒有 [settings] 表")
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    missing = [k for k in SETTINGS_KEYS if k not in settings]
    if extra or missing:
        raise ToolBroken(
            f"id={CARD_ID} 的卡 [settings] 鍵不對：多了 {sorted(extra)}、少了 {missing}，只認 {list(SETTINGS_KEYS)}"
        )
    pass_prefix = settings["verdict_pass_prefix"]
    fail_prefix = settings["verdict_fail_prefix"]
    length = settings["sha_hex_length"]
    fixture = settings["fixture_event_file"]
    if not (isinstance(pass_prefix, str) and pass_prefix.strip()):
        raise ToolBroken(f"verdict_pass_prefix 必須是非空字串，實際 {pass_prefix!r}")
    if not (isinstance(fail_prefix, str) and fail_prefix.strip()):
        raise ToolBroken(f"verdict_fail_prefix 必須是非空字串，實際 {fail_prefix!r}")
    if pass_prefix.strip() == fail_prefix.strip():
        raise ToolBroken("通過與不通過的前綴一樣，分不出判決")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ToolBroken(f"sha_hex_length 必須是正整數，實際 {length!r}")
    if not (isinstance(fixture, str) and fixture.strip()):
        raise ToolBroken(f"fixture_event_file 必須是非空字串，實際 {fixture!r}")
    return Settings(pass_prefix.strip(), fail_prefix.strip(), length, fixture.strip())


def _toplevel(scan_root: Path) -> Path | None:
    """這棵樹是哪個 git 工作樹的根？不是 git 工作樹就回 None；git 叫不動回 2。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=scan_root,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ToolBroken(f"外部工具 git 不在 PATH（問這棵樹是不是 git 工作樹）：{exc}") from exc
    except OSError as exc:
        raise ToolBroken(f"叫不動 git：{exc}") from exc
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return Path(proc.stdout.strip()).resolve()


def _load_event(path: Path, what: str) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolBroken(f"{what} {path} 讀不開或不是 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ToolBroken(f"{what} {path} 頂層不是物件")
    return data


def _resolve_event(scan_root: Path, settings: Settings) -> dict[str, object] | None:
    """決定判哪一份事件。回 None 代表這一跑沒有合併請求可判（不是 pull_request 事件）。"""
    decl = scan_root / settings.fixture_event_file
    top = _toplevel(scan_root)
    if top is not None and top == scan_root:
        if decl.exists():
            raise ToolBroken(
                f"這是真的 git 工作樹的根，卻放著樣本用的 {settings.fixture_event_file}"
                "——那條路只給必紅樣本用；真的工作樹裡出現它，等於真事件被一個檔案繞過"
            )
        event_name = os.environ.get(EVENT_NAME_ENV, "").strip()
        if event_name != PR_EVENT_NAME:
            return None
        raw = os.environ.get(EVENT_PATH_ENV, "").strip()
        if not raw:
            raise ToolBroken(f"這是 {PR_EVENT_NAME} 事件，卻沒有 {EVENT_PATH_ENV}——沒有事件檔就沒有內文可判")
        return _load_event(Path(raw), "事件檔")
    if decl.is_file():
        return _load_event(decl, "樣本事件檔")
    raise ToolBroken(
        f"{scan_root} 既不是 git 工作樹的根，也沒有 {settings.fixture_event_file}——我拿不到事件，這一跑不算數"
    )


def _is_hex(text: str, length: int) -> bool:
    return len(text) == length and all(ch in "0123456789abcdef" for ch in text)


def _head_and_body(event: dict[str, object], settings: Settings) -> tuple[str, str]:
    pr = event.get(PR_EVENT_NAME)
    if not isinstance(pr, dict):
        raise ToolBroken(f"事件裡沒有 {PR_EVENT_NAME} 這一節——這不是合併請求的事件")
    head = pr.get("head")
    sha = head.get("sha") if isinstance(head, dict) else None
    if not isinstance(sha, str) or not _is_hex(sha.strip().lower(), settings.sha_hex_length):
        raise ToolBroken(f"事件的 head.sha 不是完整提交 id：{sha!r}")
    body = pr.get("body")
    if body is None:
        body = ""
    if not isinstance(body, str):
        raise ToolBroken(f"事件的 body 不是字串：{type(body).__name__}")
    return sha.strip().lower(), body


def judge(head: str, body: str, settings: Settings) -> list[str]:
    """純判準：內文對 head。抽出來是為了讓考卷不用造事件檔就能餵。"""
    bad: list[str] = []
    passes: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        fields = line.split()
        if not fields:
            continue
        if fields[0] == settings.fail_prefix:
            bad.append(f"內文寫著「{line}」——最後一次驗收不通過，不准合")
        elif fields[0] == settings.pass_prefix:
            passes.append(line)
    if not passes:
        bad.append(
            f"內文沒有「{settings.pass_prefix} <完整提交 id>」那一行——沒宣告驗收過哪一顆，就不知道驗的是不是要合的這一顆"
        )
        return bad
    if len(passes) > 1:
        bad.append(f"內文有 {len(passes)} 行驗收通過的宣告，不知道信哪一行；只留最後一次那一行")
        return bad
    fields = passes[0].split()
    if len(fields) < 2:
        bad.append(f"「{passes[0]}」後面沒有提交 id——驗收通過要指到哪一顆")
        return bad
    claimed = fields[1].strip().lower()
    if not _is_hex(claimed, settings.sha_hex_length):
        bad.append(f"「{passes[0]}」的提交 id 不是完整的十六進位提交 id（短 id 也不收，抄短了會撞）")
        return bad
    if claimed != head:
        bad.append(
            f"驗收通過的是 {claimed[:9]}，要合的 head 是 {head[:9]}——驗過之後又推了提交，修完沒再驗"
        )
    return bad


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面：這張卡自己（判準住在它的 [settings]）。事件檔在 repo 外，不算版控裡的掃描面。"""
    return [_card_path(scan_root, files)]


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = read_settings(scan_root, files)
    event = _resolve_event(scan_root, settings)
    if event is None:
        note(f"不是 {PR_EVENT_NAME} 事件（{EVENT_NAME_ENV}={os.environ.get(EVENT_NAME_ENV, '')!r}），沒有內文可判")
        return []
    head, body = _head_and_body(event, settings)
    bad = judge(head, body, settings)
    note(f"head={head[:9]} body_lines={len(body.splitlines())} hits={len(bad)}")
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="合併請求內文的驗收結論要指到要合的那顆提交",
            targets=targets,
        )
    )
