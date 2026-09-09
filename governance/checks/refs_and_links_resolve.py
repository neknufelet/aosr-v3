#!/usr/bin/env python3
"""文件裡的路徑與連結都要解析得到，不准指到 repo 外。

掃描面是 ``git ls-files``（含未被忽略的未追蹤檔）的全集，先扣掉卡上登記的前綴，再只留卡上
登記的副檔名。兩條規則，門檻、掃描面與放行清單全部只寫在卡的 ``[settings]``，讀不到就回 2
（工具自壞），不回 0：

1. **解析得到**——形如 ``目錄/…/檔名.副檔名`` 的路徑 token，加上 md 的 markdown link 目標，
   都必須在列舉集合裡找得到。glob 算解析得到（``governance/checks/*.py`` 要真的有東西命中，
   命中零個就是死引用），markdown link 指到目錄也算（有檔住在那個前綴底下）。
2. **不准指到 repo 外**——``../`` 逃出樹、開頭 ``/`` 的絕對路徑、開頭 ``~/`` 的家目錄路徑，
   一律紅。這一條只判形狀：**檢查程式不會去讀那些路徑**，repo 外的更不會。

每一種檔按自己的形狀取文字，不是一律當純文字讀：

* ``.md``／``.toml``／``.yaml``／``.yml``——整份逐行掃（設定檔裡的路徑也是引用）。
* ``.py``——只掃字串與註解（含 f-string 的字面段落），會跑的程式碼本身不掃。用 ``tokenize``，
  解不開就回 2，不是跳過。
* ``.json``——只掃字串值，鍵名與數字不掃。用 ``json.loads``，解不開就回 2。

**為什麼靠 ``id`` 認卡、不靠 ``check`` 欄。** 別支檢查是找「``check`` 欄指到自己」的那張卡；
這支不行。必紅樣本是一棵棵獨立的迷你掃描根，每棵樹裡都得放一張卡（門檻在卡上），而 ``check``
欄的值本身就是一個路徑 token——樣本樹裡沒有 ``governance/checks/`` 那支程式，寫了那一欄，
每份樣本都會多一筆跟它要證的事無關的違規。所以樣本的迷你卡只帶 ``id`` 與 ``[settings]``，
這支檢查照 ``id`` 找卡。

``[[settings.allow]]`` 是明文放行，一條一個路徑（照它在文件裡寫的樣子逐字比對），**每條必須
寫 ``reason``**。放行只是「不判紅」，不是「去讀它」。今天四條，各一種刻意的形狀：repo 外的
備份座標、刻意不進版控的產出物、只活在必紅樣本樹裡的道具、v2 事故現場的路徑。哪幾條真的被
用到、哪幾條沒被用到（可能已經過期），跑完印 ``NOTE:`` 說清楚。

血債 ``moved-files-leave-dead-references``（``v2-audit/lessons.json``）：那筆事故自己的對策
幾乎逐字寫了這道閘。找碴席對它判 lesson_bites = false，理由與指揮為什麼仍記成血債，寫在卡的
註解裡，這裡不重複。

已知的縫（照實寫，不遮）：沒有副檔名的裸目錄 token（``docs/``、``.github/``）不當引用看，
不然乾淨樹當場回 1；md 的小節錨點（``#小節``）與 http(s) 連結這一版不管（後者要老闆先拍板
「文件准不准依賴外部資源」）；``<占位>/`` 這種前綴會先脫掉再當相對掃描根的路徑解析；副檔名
不在卡登記清單上的檔（``.txt``／``.lock``）整份不掃。放行清單今天沒有到期日，等
exemptions-need-expiry 那張卡立起來再補。
"""
from __future__ import annotations

import fnmatch
import io
import json
import posixpath
import re
import sys
import tokenize
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from governance.exit_codes import ToolBroken, run  # noqa: E402
from governance.loader import RULES_DIR  # noqa: E402

# 這張卡的 id。門檻只從「id 是這個」的那張卡讀（為什麼不用 check 欄，見模組說明）。
CARD_ID = "refs-and-links-resolve"

# 門檻的形狀。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = ("text_suffixes", "scan_exempt_prefixes")
NONEMPTY_LIST_KEYS = ("text_suffixes",)
ALLOW_KEY = "allow"
ALLOW_ENTRY_KEYS = ("path", "reason")
SETTINGS_KEYS = (*LIST_KEYS, ALLOW_KEY)

# 先從行裡挖掉的東西：http(s) 之類的 URL（這一版不管外部連結），以及 `<scan_root>/` 這種
# 占位前綴（脫掉之後剩下的當相對掃描根的路徑，不然它會被誤判成絕對路徑）。
URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
PLACEHOLDER_RE = re.compile(r"<[A-Za-z0-9_]+>/")

# markdown link／圖片：`[標題](目標)`。只取目標那一段，取完把它塗白，
# 免得同一個死目標又被下面的 token 掃出來、同一件事報兩次。
LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*(?P<target>[^)\s]+)")

# 路徑 token：形如 `目錄/…/檔名.副檔名`。
#   前面那一段兩種寫法——逃出去的前綴（`../`、`./`、`~/`、`/`）後面接零個以上目錄，
#   或是至少一個目錄（沒有前綴的相對路徑至少要有一個斜線，不然「檔名.副檔名」滿地都是）。
#   兩頭的 lookbehind／lookahead 是為了不把一個 token 截半：`>` 在裡面，`<占位>/a/b.c`
#   脫不掉的時候不會被讀成絕對路徑；`-` 在裡面，`blueprint/.engine-foo/` 不會被截成 `.engine`。
SEGMENT = r"[A-Za-z0-9_.*?-]"
TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_./~>-])"
    r"("
    rf"(?:(?:\.\.?/|~/|/)(?:{SEGMENT}+/)*|(?:{SEGMENT}+/)+)"
    rf"{SEGMENT}*\.[A-Za-z0-9*?]{{1,8}}"
    r")"
    r"(?![A-Za-z0-9_./*?-])"
)

GLOB_CHARS = "*?["

MARKDOWN_SUFFIX = ".md"
PYTHON_SUFFIX = ".py"
JSON_SUFFIX = ".json"

# py 裡算「不是會跑的程式碼」的 token 型別：字串、註解，以及 f-string 的字面段落
# （3.12 起 f-string 被拆成 FSTRING_START／MIDDLE／END，字面文字在 MIDDLE）。
PY_TEXT_TYPES = frozenset(
    {tokenize.STRING, tokenize.COMMENT} | {t for t in (getattr(tokenize, "FSTRING_MIDDLE", None),) if t}
)

KIND_TOKEN = "路徑 token"
KIND_LINK = "markdown link"


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這張卡的門檻。

    找不到、找到多張、或門檻的形狀不對，一律 raise ToolBroken——沒有門檻就不出結論。
    """
    rules_dir = scan_root / RULES_DIR
    mine: list[tuple[str, dict[str, object]]] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        rel = path.relative_to(scan_root).as_posix()
        raw = _read_text(path, rel)
        try:
            data = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            raise ToolBroken(f"{rel} 解不開（{exc}）——我沒看懂就不出結論") from exc
        if data.get("id") == CARD_ID:
            mine.append((rel, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——門檻只寫在卡上，讀不到門檻這一跑就不算數"
        )
    rel, data = mine[0]
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（掃描面、副檔名清單與放行清單都寫在那裡）")
    _assert_settings(settings, rel)
    return settings


def _assert_settings(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒門檻）")
    for key in LIST_KEYS:
        value = settings.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif key in NONEMPTY_LIST_KEYS and not value:
            bad.append(f"{key} 不准是空 list——空的清單等於這張卡什麼都不掃")
    suffixes = settings.get("text_suffixes")
    if isinstance(suffixes, list):
        for suffix in suffixes:
            if isinstance(suffix, str) and not suffix.startswith("."):
                bad.append(f"text_suffixes 裡的 {suffix!r} 要從點開始寫（例如 \".md\"）")
    allow = settings.get(ALLOW_KEY, [])
    if not isinstance(allow, list):
        bad.append(f"{ALLOW_KEY} 寫了就必須是表的 list（一條一個放行的路徑），實際是 {allow!r}")
    else:
        for index, entry in enumerate(allow):
            where = f"{ALLOW_KEY}[{index}]"
            if not isinstance(entry, dict):
                bad.append(f"{where} 必須是一張表（path ＋ reason），實際是 {entry!r}")
                continue
            for key in ALLOW_ENTRY_KEYS:
                value = entry.get(key)
                if not isinstance(value, str) or not value.strip():
                    bad.append(
                        f"{where} 的 {key} 必須是非空字串"
                        f"——放行不准沒有理由，不然清單會長成沒人記得為什麼的豁免（實際是 {value!r}）"
                    )
            unknown = [k for k in entry if k not in ALLOW_ENTRY_KEYS]
            if unknown:
                bad.append(f"{where} 多了不認識的鍵 {sorted(unknown)}，只認 {list(ALLOW_ENTRY_KEYS)}")
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _md_segments(text: str) -> list[tuple[int, str]]:
    """md／toml／yaml：整份逐行。"""
    return list(enumerate(text.splitlines(), 1))


def _py_segments(text: str, rel: str) -> list[tuple[int, str]]:
    """py：只留字串與註解（含 f-string 的字面段落），會跑的程式碼本身不看。"""
    out: list[tuple[int, str]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in PY_TEXT_TYPES:
                continue
            for offset, line in enumerate(tok.string.splitlines()):
                out.append((tok.start[0] + offset, line))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise ToolBroken(f"{rel} 斷不出 token（{exc}）——我沒看懂就不出結論") from exc
    return out


def _json_strings(node: object, out: list[str]) -> None:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for value in node.values():  # 只看值，不看鍵名
            _json_strings(value, out)
    elif isinstance(node, list):
        for value in node:
            _json_strings(value, out)


def _json_segments(text: str, rel: str) -> list[tuple[int, str]]:
    """json：只留字串值。行號用「在原文裡找得到那段字」回推，找不到就記 0。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolBroken(f"{rel} 解不開（{exc}）——我沒看懂就不出結論") from exc
    strings: list[str] = []
    _json_strings(data, strings)
    out: list[tuple[int, str]] = []
    for value in strings:
        index = text.find(value)
        out.append((text.count("\n", 0, index) + 1 if index >= 0 else 0, value))
    return out


def _segments(path: Path, rel: str, suffix: str) -> list[tuple[int, str]]:
    text = _read_text(path, rel)
    if suffix == PYTHON_SUFFIX:
        return _py_segments(text, rel)
    if suffix == JSON_SUFFIX:
        return _json_segments(text, rel)
    return _md_segments(text)


def _refs(segment: str, *, links: bool) -> list[tuple[str, str]]:
    """一段文字裡的引用。回傳 [(種類, 原文寫法)]。"""
    out: list[tuple[str, str]] = []
    cleaned = URL_RE.sub(" ", segment)
    if links:
        chars = list(cleaned)
        for match in LINK_RE.finditer(cleaned):
            out.append((KIND_LINK, match.group("target")))
            start, end = match.span("target")
            for index in range(start, end):
                chars[index] = " "
        cleaned = "".join(chars)
    cleaned = PLACEHOLDER_RE.sub("", cleaned)
    for match in TOKEN_RE.finditer(cleaned):
        out.append((KIND_TOKEN, match.group(1)))
    return out


def _link_target(raw: str) -> str:
    """連結目標：削掉 `#錨點` 與 `?查詢`。純錨點、mailto: 這種回空字串（這一版不管）。"""
    target = raw.split("#", 1)[0].split("?", 1)[0]
    head = target.split("/", 1)[0]
    if ":" in head:
        return ""
    return target


def _outside_reason(raw: str, normalized: str) -> str:
    if raw.startswith("~/"):
        return "指到家目錄（`~/`）"
    if raw.startswith("/"):
        return "是絕對路徑"
    if normalized == ".." or normalized.startswith("../"):
        return "用 `../` 逃出這棵樹"
    return ""


def _resolves(normalized: str, names: list[str], nameset: set[str], *, allow_dir: bool) -> bool:
    if any(char in normalized for char in GLOB_CHARS):
        return bool(fnmatch.filter(names, normalized))
    if normalized in nameset:
        return True
    if allow_dir:
        prefix = normalized.rstrip("/") + "/"
        return any(name.startswith(prefix) for name in names)
    return False


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    suffixes = {str(s).casefold() for s in settings["text_suffixes"]}  # type: ignore[union-attr]
    exempt = [str(p) for p in settings["scan_exempt_prefixes"]]  # type: ignore[union-attr]
    allow = {str(entry["path"]): str(entry["reason"]) for entry in settings.get(ALLOW_KEY, [])}  # type: ignore[union-attr,index]

    names = sorted(path.relative_to(scan_root).as_posix() for path in files)
    nameset = set(names)
    targets = [
        name
        for name in names
        if not any(name.startswith(prefix) for prefix in exempt)
        and Path(name).suffix.casefold() in suffixes
    ]
    if not targets:
        raise ToolBroken(
            f"{scan_root} 扣掉 {exempt} 之後，一份卡上登記的文字檔（{sorted(suffixes)}）都不剩"
            "——這一跑沒掃到東西，「沒問題」這句話不算數"
        )

    used: set[str] = set()
    bad: list[str] = []
    for rel in targets:
        path = scan_root / rel
        if not path.is_file():
            # git 認得、檔案系統上不在（剛被刪掉還沒 commit）。不猜內容，跳過。
            continue
        suffix = Path(rel).suffix.casefold()
        is_markdown = suffix == MARKDOWN_SUFFIX
        for lineno, segment in _segments(path, rel, suffix):
            for kind, raw in _refs(segment, links=is_markdown):
                if raw in allow:
                    used.add(raw)
                    continue
                target = _link_target(raw) if kind == KIND_LINK else raw
                if not target:
                    continue
                base = posixpath.dirname(rel) if kind == KIND_LINK else ""
                normalized = posixpath.normpath(posixpath.join(base, target))
                outside = _outside_reason(target, normalized)
                if outside:
                    bad.append(
                        f"{rel}:{lineno} {kind} {raw!r} {outside}"
                        "——引用不准指到 repo 外，換一台機器就不存在了；真的必要（例如 repo 外的"
                        "備份座標）就進卡的 [[settings.allow]] 並寫理由"
                    )
                    continue
                if not _resolves(normalized, names, nameset, allow_dir=(kind == KIND_LINK)):
                    bad.append(
                        f"{rel}:{lineno} {kind} {raw!r} 在這棵樹裡解析不到"
                        "（`git ls-files` 的集合裡沒有，glob 也命中零個）"
                        "——刪檔或搬目錄要順手改引用，v2 就是這樣讓死引用活了 53 天"
                    )

    stale = sorted(set(allow) - used)
    if stale:
        print(
            f"NOTE: 卡上有 {len(stale)} 條放行沒被用到（可能已經過期，該從卡上刪掉）：{stale}",
            file=sys.stderr,
        )
    if used:
        print(f"NOTE: 這一跑用到的放行：{sorted(used)}", file=sys.stderr)
    return bad


if __name__ == "__main__":
    sys.exit(run(check, description="文件裡的路徑與連結都要解析得到，不准指到 repo 外"))
