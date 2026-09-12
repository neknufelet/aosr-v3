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
* ``blueprint/*.py``——更窄：只掃 **docstring 與註解**（程式碼字串是執行期的事，一律不看），
  而且只認「長得像這一棵 repo 的相對路徑」的 token：前綴限 ``docs/``／``governance/``／
  ``blueprint/``／``src/``／``tests/``、副檔名限 ``.md``／``.py``／``.json``／``.toml``。
  那批檔是產生器／case 表，docstring 與註解裡滿是上一代（donor 樹）的座標，不這樣收窄，
  乾淨樹當場回 1。``../``／絕對路徑／``~/`` 這種指到 repo 外的形狀不受這層收窄影響。
* ``.json``——只掃字串值，鍵名與數字不掃。用 ``json.loads``，解不開就回 2。

**為什麼靠 ``id`` 認卡、不靠 ``check`` 欄。** 別支檢查是找「``check`` 欄指到自己」的那張卡；
這支不行。必紅樣本是一棵棵獨立的迷你掃描根，每棵樹裡都得放一張卡（門檻在卡上），而 ``check``
欄的值本身就是一個路徑 token——樣本樹裡沒有 ``governance/checks/`` 那支程式，寫了那一欄，
每份樣本都會多一筆跟它要證的事無關的違規。所以樣本的迷你卡只帶 ``id`` 與 ``[settings]``，
這支檢查照 ``id`` 找卡。

``[[settings.allow]]`` 是明文放行，一條一個路徑（照它在文件裡寫的樣子逐字比對），**每條必須
寫 ``reason`` 與 ``expires``**（到期日；過期即紅那一關由 exemptions-need-expiry 判）。放行只是「不判紅」，不是「去讀它」。今天四條，各一種刻意的形狀：repo 外的
備份座標、刻意不進版控的產出物、只活在必紅樣本樹裡的道具、v2 事故現場的路徑。哪幾條真的被
用到、哪幾條沒被用到（可能已經過期），跑完印 ``NOTE:`` 說清楚。

血債 ``moved-files-leave-dead-references``（``v2-audit/lessons.json``）：那筆事故自己的對策
幾乎逐字寫了這道閘。找碴席對它判 lesson_bites = false，理由與指揮為什麼仍記成血債，寫在卡的
註解裡，這裡不重複。

已知的縫（照實寫，不遮）：沒有副檔名的裸目錄 token（``docs/``、``.github/``）不當引用看，
不然乾淨樹當場回 1；md 的小節錨點（``#小節``）與 http(s) 連結這一版不管（後者要老闆先拍板
「文件准不准依賴外部資源」）；``<占位>/`` 這種前綴會先脫掉再當相對掃描根的路徑解析；副檔名
不在卡登記清單上的檔（``.txt``／``.lock``）整份不掃。放行清單的每一條從今天起要有 ``expires``
（到期日），過期即紅——那道閘在規矩卡 exemptions-need-expiry，形狀在 governance/loader.py。
"""
from __future__ import annotations

import ast
import fnmatch
import io
import json
import posixpath
import re
import sys
import tokenize
import tomllib
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.loader import EXEMPTION_KEYS, RULES_DIR, setting_strings, setting_tables

# 這張卡的 id。門檻只從「id 是這個」的那張卡讀（為什麼不用 check 欄，見模組說明）。
CARD_ID = "refs-and-links-resolve"

# 門檻的形狀。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = ("text_suffixes", "scan_exempt_prefixes", "scan_exempt_suffixes")
NONEMPTY_LIST_KEYS = ("text_suffixes",)
ALLOW_KEY = "allow"
# 一筆放行三格：放行哪個路徑，加上放行條目共用的兩格（reason ＋ expires，形狀定義在
# governance/loader.py 的 EXEMPTION_KEYS，由規矩卡 exemptions-need-expiry 統一）。
ALLOW_ENTRY_KEYS = ("path", *EXEMPTION_KEYS)
SETTINGS_KEYS = (*LIST_KEYS, ALLOW_KEY)
# scan_exempt_suffixes 一筆一條「前綴:副檔名」：只在那個前綴底下、只有那個副檔名被扣掉
# （其餘副檔名照掃）。用冒號分隔，不用斜線——斜線加副檔名會長得像路徑 token，被這支檢查
# 自己掃成死引用。今天只有 blueprint 底下的 dot-json 一條：那些 .json 是資料不是引用。
SUFFIX_EXEMPT_SEP = ":"
SUFFIX_EXEMPT_PARTS = 2

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

# blueprint/*.py 的收窄判準：那批檔是產生器／case 表，docstring 與註解裡滿是上一代（donor 樹）
# 的座標（``run/…``、``lib/…``）與示範性死路徑。所以只解析「長得像這一棵 repo 的相對路徑」的
# 字串：前綴限這五個目錄，副檔名限這四種。程式碼字串（``BATCH1 = REPO / \"…\"``）更是不看——
# 那是執行期的事。理由整段寫在卡面的 scan_exempt_prefixes 註解與 human。
BLUEPRINT_PATH_HEADS = ("docs", "governance", "blueprint", "src", "tests")
BLUEPRINT_PATH_SUFFIXES = (".md", ".py", ".json", ".toml")

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
    suffix_exempt = settings.get("scan_exempt_suffixes")
    if isinstance(suffix_exempt, list):
        for entry in suffix_exempt:
            if not isinstance(entry, str):
                bad.append(f"scan_exempt_suffixes 裡的 {entry!r} 必須是字串")
                continue
            parts = entry.split(SUFFIX_EXEMPT_SEP)
            if len(parts) != SUFFIX_EXEMPT_PARTS or not parts[0] or not parts[1].startswith("."):
                bad.append(
                    f"scan_exempt_suffixes 裡的 {entry!r} 要寫成「前綴:副檔名」"
                    "（冒號分隔、副檔名從點開始，例如某目錄下的 .json）"
                )
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


def _py_docstring_segments(text: str, rel: str) -> list[tuple[int, str]]:
    """py 的 docstring（module／class／function 的第一個字串常數）。程式碼字串不算。

    用 :mod:`ast` 找每一個 docstring 節點，行號用「docstring 開頭那一行＋值裡第幾行」回推——
    夠報給人看位置，正不正确只影響第幾行，不影響判不判紅。
    """
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行解不開（{exc.msg}）——我沒看懂就不出結論") from exc
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            continue
        for offset, line in enumerate(first.value.value.splitlines()):
            out.append((first.lineno + offset, line))
    return out


def _blueprint_py_segments(text: str, rel: str) -> list[tuple[int, str]]:
    """blueprint 的 .py：只留 docstring 與註解（程式碼字串是執行期的事，不看）。"""
    out = _py_docstring_segments(text, rel)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out.append((tok.start[0], tok.string))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise ToolBroken(f"{rel} 斷不出 token（{exc}）——我沒看懂就不出結論") from exc
    return out


def _blueprint_ref_kept(raw: str) -> bool:
    """blueprint .py 的引用只認 repo 相對路徑：前綴在前述五個目錄、副檔名在前述四種。

    ``../``／``/``／``~/`` 這種指到 repo 外的形狀不受這層收窄影響——照舊交給第②條判。
    """
    if raw.startswith(("/", "~/")) or raw == ".." or raw.startswith("../"):
        return True
    head = raw.split("/", 1)[0]
    if head not in BLUEPRINT_PATH_HEADS:
        return False
    return raw.endswith(BLUEPRINT_PATH_SUFFIXES)


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


def _segments(path: Path, rel: str, suffix: str, *, blueprint_py: bool) -> list[tuple[int, str]]:
    text = _read_text(path, rel)
    if suffix == PYTHON_SUFFIX:
        return _blueprint_py_segments(text, rel) if blueprint_py else _py_segments(text, rel)
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


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（掃描面、副檔名清單與放行清單只寫在卡上）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：卡上登記的那幾種文字檔（扣掉卡上登記的前綴）＋ 所有規矩卡。

    全樹的名字清單另有用途——它是「這個引用解不解析得到」的對照表，不是被讀的檔，
    所以不算掃描面。
    """
    settings = _card_settings(scan_root, files)
    suffixes = {s.casefold() for s in setting_strings(settings, "text_suffixes")}
    exempt = setting_strings(settings, "scan_exempt_prefixes")
    suffix_exempt = [
        tuple(entry.split(SUFFIX_EXEMPT_SEP))
        for entry in setting_strings(settings, "scan_exempt_suffixes")
    ]
    picked = [
        f
        for f in files
        if not any(f.relative_to(scan_root).as_posix().startswith(prefix) for prefix in exempt)
        and not any(
            f.relative_to(scan_root).as_posix().startswith(prefix)
            and f.suffix.casefold() == suffix
            for prefix, suffix in suffix_exempt
        )
        and f.suffix.casefold() in suffixes
    ]
    return sorted(set(picked) | set(_card_files(scan_root, files)))


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    suffixes = {s.casefold() for s in setting_strings(settings, "text_suffixes")}
    exempt = setting_strings(settings, "scan_exempt_prefixes")
    allow = {
        str(entry["path"]): str(entry["reason"])
        for entry in setting_tables(settings, ALLOW_KEY)
    }

    names = sorted(path.relative_to(scan_root).as_posix() for path in files)
    nameset = set(names)
    picked = [p.relative_to(scan_root).as_posix() for p in targets(scan_root, files)]
    if not picked:
        raise ToolBroken(
            f"{scan_root} 扣掉 {exempt} 之後，一份卡上登記的文字檔（{sorted(suffixes)}）都不剩"
            "——這一跑沒掃到東西，「沒問題」這句話不算數"
        )

    used: set[str] = set()
    bad: list[str] = []
    for rel in picked:
        path = scan_root / rel
        if not path.is_file():
            # git 認得、檔案系統上不在（剛被刪掉還沒 commit）。不猜內容，跳過。
            continue
        suffix = Path(rel).suffix.casefold()
        is_markdown = suffix == MARKDOWN_SUFFIX
        blueprint_py = suffix == PYTHON_SUFFIX and rel.startswith("blueprint/")
        for lineno, segment in _segments(path, rel, suffix, blueprint_py=blueprint_py):
            for kind, raw in _refs(segment, links=is_markdown):
                if blueprint_py and not _blueprint_ref_kept(raw):
                    continue
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
        note(f"卡上有 {len(stale)} 條放行沒被用到（可能已經過期，該從卡上刪掉）：{stale}")
    if used:
        note(f"這一跑用到的放行：{sorted(used)}")
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="文件裡的路徑與連結都要解析得到，不准指到 repo 外",
            targets=targets,
        )
    )
