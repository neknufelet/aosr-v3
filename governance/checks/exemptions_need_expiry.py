#!/usr/bin/env python3
"""每一筆放行都要有理由與到期日；過期即紅，關鍵字當不了免死金牌。

掃描面明確列舉，只有三類算「放行條目」（名單與樣式全部只寫在卡的 ``[settings]``，
讀不到就回 2，不回 0）：

1. **規矩卡的表陣列**——``[settings]`` 底下的表陣列（今天全叫 ``[[settings.allow]]``），
   一筆就是一個具名放過。兩格寫成 toml 的鍵。
2. **測試裡的 skip 標記**——``pytest.skip``、``pytest.mark.skip``／``skipif``／``xfail``，
   由 AST 認（不是字串比對）。
3. **原始碼註解裡的抑制標記**——卡上登記的那幾種（少寫一個字母的抑制標記本來就抑制不了東西，
   所以逐字比對註解原文就夠）。由 ``tokenize`` 拿註解，剖不開就回 2。

第 2、3 類的兩格寫成**標記那一行行尾的註解**，形如 ``expires=<日期> reason=<一句話>``：
理由取到行尾，所以到期日寫在前面。順序寫反不會整條讀不到（到期日那一段會先被挖掉再取理由），
但理由那一格就可能把後面的字一起吃進去——照卡上的順序寫。
刻意只認註解這一個來源，不順便讀 ``pytest.mark.skip(reason=...)`` 自己那一格——
兩個來源就是兩份會各自漂的宣告，而漂掉的那一份正是這個 repo 最常出事的形狀。

**什麼會紅**

* 缺 ``reason`` 或缺 ``expires``（第 2、3 類：那一行行尾根本沒有註解，或註解裡挖不出那一格）。
* ``expires`` 看不懂（不是「四位數年-兩位數月-兩位數日」）——看不懂的日期等於沒有到期日。
* ``expires`` 早於今天。今天從系統時間讀，不寫死；所以同一棵樹今天綠、到期那天自己會紅。
* ``reason`` 正規化之後整串只由卡上登記的關鍵字組成（「暫時」「legacy」「TODO」這種字樣）。
  正規化＝丟掉標點與空白、拉成小寫。只在**放行條目本身**判，不掃全樹散文——掃散文會誤咬
  正常內文（第一輪找碴席就是在這裡點的洞）。

**什麼不會紅（刻意的，留在卡面不要當成漏掉）**

* 到期日填得很遠等於沒有到期日，這支檢查咬不到；它咬的是「連日期都不寫」與「日期到了還在」。
* 執行期才觸發的 ``pytest.skip()`` 呼叫、或 ``getattr`` 繞法拿到同一支函式，AST 看不到。
* 白名單與名單檔（``[[allowlist]]``、提交者名單、required check 名單）是**資料**不是放行：
  它們是規矩的正面定義，拿掉就沒有規矩，不是「放過某一筆已經算出來的違規」。判準與今天的
  分類逐條寫在卡的人話欄。

**沒掃到東西就回 2**

卡讀不到、``[settings]`` 形狀不對、某張卡或某支 .py 剖不開、掃描面上一個對象都沒有，
一律 raise :class:`ToolBroken`。這一跑沒量到東西，「沒問題」這句話就不算數。

**為什麼靠 ``id`` 認卡、不靠 ``check`` 欄。** 必紅樣本是一棵棵獨立的迷你掃描根，每棵樹裡都要
放一張卡（名單在卡上），而 ``check`` 欄的值本身是一個路徑 token——樣本樹裡沒有那支程式，
寫了那一欄，每份樣本都會多一筆跟它要證的事無關的違規。所以樣本的迷你卡只帶 ``id``
與 ``[settings]``，這支檢查照 ``id`` 找卡（跟 refs-and-links-resolve 同一個理由）。
"""
from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
import tomllib
from datetime import date
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.loader import (
    EXEMPTION_DATE_SHAPE,
    EXEMPTION_KEYS,
    RULES_DIR,
    exemption_field_problems,
    exemption_problems,
    settings_exemptions,
)

# 這張卡的 id。名單只從「id 是這個」的那張卡讀（為什麼不用 check 欄，見模組說明）。
CARD_ID = "exemptions-need-expiry"

PYTHON_SUFFIX = ".py"
CARD_SUFFIX = ".toml"

# 名單的形狀。打錯字的名單等於沒有名單，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = ("comment_markers", "skip_markers", "keyword_reasons", "scan_exempt_prefixes")
NONEMPTY_LIST_KEYS = ("comment_markers", "skip_markers", "keyword_reasons")
SETTINGS_KEYS = LIST_KEYS

# 從註解裡挖那兩格。到期日取到下一個空白，理由取到行尾（所以到期日寫在前面）。
# 兩個樣式都從 EXEMPTION_KEYS 拼出來，欄位名不在這裡再寫一次。
EXPIRES_KEY, REASON_KEY = "expires", "reason"
EXPIRES_RE = re.compile(EXPIRES_KEY + r"\s*=\s*(\S+)")
REASON_RE = re.compile(REASON_KEY + r"\s*=\s*(.*)$")


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（第 1 類的對象，名單也住在其中一張上）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == CARD_SUFFIX)


def _load_toml(path: Path, rel: str) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"{rel} 讀不開或解不開（{exc}）——我沒看懂就不出結論") from exc


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這張卡的名單。

    找不到、找到多張、或名單的形狀不對，一律 raise ToolBroken——沒有判準就不出結論。
    """
    mine: list[tuple[str, dict[str, object]]] = []
    for path in _card_files(scan_root, files):
        rel = path.relative_to(scan_root).as_posix()
        data = _load_toml(path, rel)
        if data.get("id") == CARD_ID:
            mine.append((rel, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——名單只寫在卡上，讀不到判準這一跑就不算數"
        )
    rel, data = mine[0]
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（標記樣式與關鍵字名單都寫在那裡）")
    _assert_settings(settings, rel)
    return settings


def _assert_settings(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的名單等於沒名單）")
    for key in LIST_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}（沒有也要明寫 {key} = []）")
        elif not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif key in NONEMPTY_LIST_KEYS and not value:
            bad.append(f"{key} 不准是空 list——空的名單等於這一條沒在管")
    if settings_exemptions({"settings": settings}):
        bad.append(
            "這張卡自己的 [settings] 底下有表陣列——這張卡就是在罰「表陣列的每一筆都要有到期日」，"
            "名單一律寫成純陣列（真的需要一筆放行，就跟別人一樣帶 reason 與 expires）"
        )
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _names(settings: dict[str, object], key: str) -> list[str]:
    return [str(s) for s in settings[key]]  # 形狀已由 _assert_settings 驗過


def _scanned_python(scan_root: Path, files: list[Path], exempt: list[str]) -> list[Path]:
    return sorted(
        f
        for f in files
        if f.suffix == PYTHON_SUFFIX
        and not any(f.relative_to(scan_root).as_posix().startswith(prefix) for prefix in exempt)
    )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀／會判的檔：所有規矩卡 ＋ 卡上沒被扣掉的每一支 .py。

    樣本樹裡的道具不算（卡上登記的前綴扣掉）：那些是別的掃描根，由後設測試單獨餵。
    """
    settings = _card_settings(scan_root, files)
    python = _scanned_python(scan_root, files, _names(settings, "scan_exempt_prefixes"))
    return sorted({*_card_files(scan_root, files), *python})


def _normalise(reason: str) -> str:
    """理由的正規化：只留字母、數字與文字（丟掉空白與標點），再拉成小寫。"""
    return "".join(ch for ch in reason if ch.isalnum()).casefold()


def _keyword_only(reason: str, keywords: list[str]) -> list[str]:
    """理由整串是不是只由關鍵字組成。回傳吃掉它的那幾個關鍵字（空 list 就是不只關鍵字）。"""
    rest = _normalise(reason)
    if not rest:
        return []
    ordered = sorted((_normalise(k) for k in keywords), key=len, reverse=True)
    eaten: list[str] = []
    while rest:
        for word in ordered:
            if word and rest.startswith(word):
                eaten.append(word)
                rest = rest[len(word) :]
                break
        else:
            return []
    return eaten


def _judgement(fields: dict[str, object], where: str, keywords: list[str], today: date) -> list[str]:
    """形狀之外的判斷：到期了沒、理由是不是只是一個推託的字樣。"""
    bad: list[str] = []
    expires = fields.get(EXPIRES_KEY)
    if isinstance(expires, str) and EXEMPTION_DATE_SHAPE.match(expires.strip()):
        when = date.fromisoformat(expires.strip())
        if when < today:
            bad.append(
                f"{where} 的 expires={when.isoformat()} 早於今天（{today.isoformat()}）"
                "——過期的放行不是放行，是沒人回頭看的已知債"
            )
    reason = fields.get(REASON_KEY)
    if isinstance(reason, str) and reason.strip():
        eaten = _keyword_only(reason, keywords)
        if eaten:
            bad.append(
                f"{where} 的 reason={reason.strip()!r} 整串只由卡上登記的關鍵字組成（{eaten}）"
                "——那不是理由，是把問題往後推的說法；要寫的是「為什麼這一筆可以放過」"
            )
    return bad


def _valid_until(fields: dict[str, object], today: date) -> date | None:
    """這一筆放行目前有效到哪天。形狀壞掉或已經過期就回 None。"""
    expires = fields.get(EXPIRES_KEY)
    if not isinstance(expires, str) or not EXEMPTION_DATE_SHAPE.match(expires.strip()):
        return None
    when = date.fromisoformat(expires.strip())
    return when if when >= today else None


def _card_hits(
    path: Path, rel: str, keywords: list[str], today: date, live: list[tuple[date, str]]
) -> list[str]:
    """第 1 類：規矩卡 ``[settings]`` 底下每一個表陣列的每一筆。"""
    bad: list[str] = []
    for name, index, entry in settings_exemptions(_load_toml(path, rel)):
        where = f"{rel} 的 [[settings.{name}]] 第 {index + 1} 條"
        problems = exemption_problems(entry, where)
        if problems:
            # 一筆放行報一筆違規（兩格都缺就一句話講完），不是一格一筆——
            # 數字要對得上「這棵樹上有幾筆放行」，不然「補了幾筆」跟「紅了幾筆」讀起來不一樣。
            bad.append("；".join(problems))
        if not isinstance(entry, dict):
            continue
        bad += _judgement(entry, where, keywords, today)
        until = _valid_until(entry, today)
        if until is not None and not problems:
            live.append((until, where))
    return bad


def _comments(text: str, rel: str) -> dict[int, str]:
    """一支 .py 裡每一行的註解（一行最多一個註解 token）。剖不開就回 2。"""
    out: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise ToolBroken(f"{rel} 斷不出 token（{exc}）——我沒看懂就不出結論") from exc
    return out


def _dotted(node: ast.expr) -> str:
    """把 ``pytest.mark.skipif`` 這種運算式攤成 ``"pytest.mark.skipif"``。認不出來回空字串。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _skip_marks(text: str, rel: str, markers: list[str]) -> list[tuple[int, str]]:
    """AST 認出來的 skip 標記：``(行號, 標記名)``，同一行同一個名字只算一次。"""
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行剖不開（{exc.msg}）——我沒看懂就不出結論") from exc
    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Name)):
            name = _dotted(node)
            if name in markers:
                found.add((node.lineno, name))
    return sorted(found)


def _annotation(comment: str) -> dict[str, object]:
    """從標記那一行的註解裡挖出那兩格。挖不到的那一格就不在回傳的表裡（= 缺）。

    先把到期日那一段挖掉再取理由，所以順序寫反也不會整條讀不到；但理由取到行尾，
    寫反的時候後面的字會被一起吃進理由，所以照卡上的順序寫（到期日在前）。
    """
    found: dict[str, object] = {}
    rest = comment
    hit = EXPIRES_RE.search(rest)
    if hit:
        found[EXPIRES_KEY] = hit.group(1)
        rest = rest[: hit.start()] + rest[hit.end() :]
    hit = REASON_RE.search(rest)
    if hit and hit.group(1).strip():
        found[REASON_KEY] = hit.group(1).strip()
    return found


def _python_hits(
    path: Path,
    rel: str,
    settings: dict[str, object],
    keywords: list[str],
    today: date,
    live: list[tuple[date, str]],
) -> list[str]:
    """第 2、3 類：一支 .py 裡的 skip 標記與註解裡的抑制標記。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}（{exc}）——我沒看懂就不出結論") from exc

    comments = _comments(text, rel)
    marked: list[tuple[int, str]] = []
    for lineno, comment in sorted(comments.items()):
        for marker in _names(settings, "comment_markers"):
            if marker in comment:
                marked.append((lineno, marker))
                break  # 一行的註解只算一筆：一組兩格就交代得完它
    marked += _skip_marks(text, rel, _names(settings, "skip_markers"))

    bad: list[str] = []
    for lineno, marker in sorted(marked):
        where = f"{rel}:{lineno} 的 {marker} 標記"
        fields = _annotation(comments.get(lineno, ""))
        problems = exemption_field_problems(fields, where)
        if problems:
            bad.append(
                "；".join(problems)
                + "（兩格寫成標記那一行行尾的註解，形如"
                f" {EXPIRES_KEY}=<四位數年-兩位數月-兩位數日> {REASON_KEY}=<一句話>）"
            )
        bad += _judgement(fields, where, keywords, today)
        until = _valid_until(fields, today)
        if until is not None and not problems:
            live.append((until, where))
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    picked = targets(scan_root, files)
    if not picked:
        raise ToolBroken(
            f"{scan_root} 底下一個放行條目的載體都沒有（規矩卡與卡上沒被扣掉的 .py 都是空的）"
            "——這一跑沒掃到東西，「沒問題」這句話不算數"
        )

    keywords = _names(settings, "keyword_reasons")
    today = date.today()
    live: list[tuple[date, str]] = []
    bad: list[str] = []
    for path in picked:
        rel = path.relative_to(scan_root).as_posix()
        if not path.is_file():
            # git 認得、檔案系統上不在（剛被刪掉還沒 commit）。不猜內容，跳過。
            continue
        if path.suffix == CARD_SUFFIX:
            bad += _card_hits(path, rel, keywords, today, live)
        else:
            bad += _python_hits(path, rel, settings, keywords, today, live)

    _note(live, today)
    return bad


def _note(live: list[tuple[date, str]], today: date) -> None:
    """把「目前有效的放行幾筆、最近到期的是哪一筆」印出來，讓清單長大這件事看得見。"""
    if not live:
        note(f"今天（{today.isoformat()}）這棵樹上沒有任何一筆有效的放行。")
        return
    when, where = min(live)
    note(
        f"今天（{today.isoformat()}）有效的放行共 {len(live)} 筆，"
        f"最近到期的是 {where}，到 {when.isoformat()}。"
    )


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description=f"每一筆放行都要有 {list(EXEMPTION_KEYS)}，過期即紅，理由不准只是關鍵字",
            targets=targets,
        )
    )
