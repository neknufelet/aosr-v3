#!/usr/bin/env python3
"""密碼、金鑰、``.env`` 一進版控就紅，歷史也要掃。

**這支檢查的第一要求不是「掃得到金鑰」，是咬得到 v2 那件事故的形狀。** 那筆事故
（``v2-audit/lessons.json`` 的 secrets-committed-and-never-scanned）洩漏的是低熵的樣板值
密碼——容器的資料庫密碼是一個八位十六進位字串、跟公開樣板一模一樣，另一個 env 檔裡是另一個
八位十六進位字串。事故原文自己寫著當時**已經有一支通用 secret-scan，而它對這個案例假陰性**。
所以這裡不看熵、不引外部工具，改成三層各咬一種形狀：

1. **存在性**——檔名命中卡上登記的 env 樣式（``.env``、``.env.*``）就紅，**不看內容**。
   v2 放過那四個檔的理由正是「裡面只是樣板值」，所以這一層刻意不給自己看內容的機會。
2. **賦值語境**——``<NAME>=<value>``／``<NAME>: <value>``／``<NAME> = "<value>"``，名字帶到
   卡上登記的字樣（比對大寫、前後要是非英數字元的邊界），而值不是佔位符，就紅。
   **這一層是專為 v2 那個八位十六進位樣板值寫的：不管熵、不管長度、不管像不像金鑰。**
3. **格式金鑰**——卡上登記的已知前綴樣式（AWS、GitHub、PEM 私鑰開頭那些）。

三層同時套在兩個集合上：

* **現在這棵樹**——``git ls-files`` 的全集（含未被忽略的未追蹤檔），扣掉卡上登記的前綴。
* **全部歷史**——``git rev-list --objects --all`` 走得到的每一顆 blob，用 ``git cat-file
  --batch`` 讀內容，路徑取 rev-list 記的那一個。刪掉檔案不會讓歷史裡那一顆消失，所以
  歷史那一筆**紅不掉**（主線 ruleset 不准改寫歷史），只能具名放行。

**淺 clone 一律回 2。** ``git rev-parse --is-shallow-repository`` 是 true，就代表這一跑根本
看不到舊提交——實測過：一個「加進來又刪掉」的金鑰，在完整的樹裡 ``rev-list --objects --all``
列得出那顆 blob，在 ``--depth 1`` 的淺 clone 裡整顆不見。這種時候「掃過全歷史」是假話，
不准回 0。CI 的 checkout 因此必須帶 ``fetch-depth: 0``。

**必紅樣本怎麼有歷史。** 樣本是一棵進版控的迷你掃描根，裡面塞不進一個真的 git 目錄
（巢狀 repo 進不了外層版控），所以每份樣本用樹根的 ``fixture-history.txt`` 宣告要有哪幾筆
提交，檢查在暫存目錄照它建一段**真的**歷史再掃。要證「淺 clone 回 2」的樣本在宣告裡多寫一行
``shallow``，建完之後多做一次 ``--depth 1`` 的本機 clone（走 ``file://`` 才吃得到那個旗標，
裸路徑會被忽略）。真的 git 工作樹的根若出現那個宣告檔一律回 2——不然把它擺進主線就能繞過真歷史。
宣告檔自己不進內容掃描（它是這支檢查的道具，裡面的假金鑰是刻意的），但它算在掃描面裡：
這支檢查真的會讀它。

**為什麼靠 ``id`` 認卡、不靠 ``check`` 欄。** 每份樣本都是獨立的掃描根、都要帶一張迷你卡
（名單在卡上），而 ``check`` 欄的值本身是一個路徑 token——樣本樹裡沒有那支程式，寫了那一欄
每份樣本都會多一筆跟它要證的事無關的違規。所以樣本的迷你卡只帶 ``id`` 與 ``[settings]``。

**為什麼不用 gitleaks**（理由也寫在卡面）：事故原文自承通用工具對這個案例假陰性；引入外部
二進位又多一個要釘版本、本機沒裝就回 2 讓每個開發者本機紅的東西；三層自寫的規則加起來已經
覆蓋事故形狀與常見金鑰前綴，判準全在卡上、改寬要走 PR。將來要把它當第四層再開票。

已知的縫（照實寫，不遮）：卡上的佔位字用「值切成英數字段之後整段相等」比對，所以把值取成
以佔位字當一整段的名字就放得過去；第②層靠「值的形狀」把程式碼濾掉（帶括號的一律不當值），
所以 ``FOO = BAR`` 這種把機密指給一個裸名字的寫法會被當成值來判——那個方向是過嚴不是過鬆，
留著；歷史那一層的路徑取 rev-list 記的第一個，同一顆 blob 曾住過兩個路徑時只認一個。
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from governance.exit_codes import ToolBroken, run
from governance.loader import EXEMPTION_KEYS, RULES_DIR

# 這張卡的 id。名單與樣式只從「id 是這個」的那張卡讀（為什麼不用 check 欄，見模組說明）。
CARD_ID = "secrets-never-committed"

# 樣本樹宣告歷史用的檔，放在樣本樹的根。刻意不放在子目錄裡：名字裡沒有斜線，
# refs-and-links-resolve 就不會把它當成一個要解析的路徑 token（它只活在樣本樹裡，
# 外層樹解析不到），這樣不必為它在那張卡上開一條放行。
HISTORY_DECL = "fixture-history.txt"
# 宣告檔裡認得的動作與旗標。
DECL_COMMIT = "commit"
DECL_ADD = "add"
DECL_DELETE = "delete"
DECL_SHALLOW = "shallow"

# 卡上 [settings] 的形狀。打錯字的名單等於沒有名單，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = (
    "env_file_globs",
    "name_words",
    "placeholder_words",
    "key_patterns",
    "scan_exempt_prefixes",
)
# 這幾張清單空掉就等於那一層整條關掉，所以不准空。
# placeholder_words 與 scan_exempt_prefixes 准空（空的只會更嚴，不會更鬆）。
NONEMPTY_LIST_KEYS = ("env_file_globs", "name_words", "key_patterns")
WRAPPERS_KEY = "placeholder_wrappers"
ALLOW_KEY = "allow"
# 一筆放行三格：放行哪個路徑前綴，加上放行條目共用的兩格（reason ＋ expires，形狀定義在
# governance/loader.py 的 EXEMPTION_KEYS，由規矩卡 exemptions-need-expiry 統一）。
ALLOW_ENTRY_KEYS = ("prefix", *EXEMPTION_KEYS)
SETTINGS_KEYS = (*LIST_KEYS, WRAPPERS_KEY, ALLOW_KEY)

# 第②層的賦值形狀。名字是識別字，接 `=` 或 `:`，值取到空白或註解符號為止。
ASSIGN_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_.\-]*)[ \t]*(?::|=)[ \t]*(?P<value>[^\s#]*)")
# 值長得像一個值（不是一段程式）。刻意排掉括號與方括號：`re.compile(...)`、`dict(...)`、
# `x[0]` 這種是程式不是機密，咬它們只會變成誤咬機。
VALUE_SHAPE_RE = re.compile(r"^[A-Za-z0-9+/=_.~:@!#%^&*?,;|${}<>-]+$")
# 值兩頭要剝掉的東西：引號、反引號，以及尾巴的逗號與分號。
VALUE_TRIM = "\"'`,;"
# 把值切成英數字段（比對佔位字用）。
SEGMENT_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
# 值裡到底有沒有英數字元。一個都沒有（空值、`***`、`---`）就是佔位符。
ALNUM_RE = re.compile(r"[A-Za-z0-9]")

# 建樣本歷史時要丟掉的 git 環境變數：留著會讓暫存 repo 指到別人的版控目錄。
DROPPED_GIT_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)
FIXTURE_DATE = "2026-01-01T00:00:00+00:00"

# 報告用的標籤：這一筆命中是在現在這棵樹，還是在歷史裡。
WHERE_TREE = "這棵樹"


class Rules(NamedTuple):
    """從卡上讀出來的三層判準。檢查程式自己沒有預設值。"""

    env_globs: tuple[str, ...]
    name_words: tuple[str, ...]
    placeholder_words: tuple[str, ...]
    wrappers: tuple[tuple[str, str], ...]
    key_patterns: tuple[tuple[str, re.Pattern[str]], ...]


class Step(NamedTuple):
    """樣本宣告裡的一筆提交。"""

    action: str
    path: str
    body: str


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _decoded(path: Path, rel: str) -> str:
    """讀一個被掃的檔。不是合法 UTF-8 的位元組換成替代字元，**不跳過整份檔**
    ——「這份檔我讀不懂所以不掃」正是機密最好的藏身處。"""
    try:
        return path.read_bytes().decode("utf-8", "replace")
    except OSError as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


# ── 卡上的名單 ─────────────────────────────────────────────────────────────────


def _card_data(scan_root: Path, files: list[Path]) -> tuple[str, dict[str, object]]:
    rules_dir = scan_root / RULES_DIR
    mine: list[tuple[str, dict[str, object]]] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        rel = path.relative_to(scan_root).as_posix()
        try:
            data = tomllib.loads(_read_text(path, rel))
        except tomllib.TOMLDecodeError as exc:
            raise ToolBroken(f"{rel} 解不開（{exc}）——我沒看懂就不出結論") from exc
        if data.get("id") == CARD_ID:
            mine.append((rel, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好一張"
            "——名單與樣式只寫在卡上，讀不到這一跑就不算數"
        )
    return mine[0]


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    rel, data = _card_data(scan_root, files)
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（三層的名單、樣式與放行清單都寫在那裡）")
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
            bad.append(f"缺 {key}")
        elif not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif key in NONEMPTY_LIST_KEYS and not value:
            bad.append(f"{key} 不准是空 list——空的清單等於那一層整條關掉")
    wrappers = settings.get(WRAPPERS_KEY)
    if WRAPPERS_KEY not in settings:
        bad.append(f"缺 {WRAPPERS_KEY}（沒有也要明寫 {WRAPPERS_KEY} = []）")
    elif not isinstance(wrappers, list):
        bad.append(f"{WRAPPERS_KEY} 必須是 list，實際是 {wrappers!r}")
    else:
        for index, pair in enumerate(wrappers):
            ok = (
                isinstance(pair, list)
                and len(pair) == len(("head", "tail"))
                and all(isinstance(part, str) and part for part in pair)
            )
            if not ok:
                bad.append(f"{WRAPPERS_KEY} 第 {index + 1} 條要寫成「開頭、結尾」兩格非空字串，實際是 {pair!r}")
    allow = settings.get(ALLOW_KEY, [])
    if not isinstance(allow, list) or not all(isinstance(x, dict) for x in allow):
        bad.append(
            f"{ALLOW_KEY} 必須是 [[settings.{ALLOW_KEY}]] 表陣列"
            f"（沒有也要明寫 {ALLOW_KEY} = []），實際是 {allow!r}"
        )
    else:
        for index, entry in enumerate(allow):
            where = f"[[settings.{ALLOW_KEY}]] 第 {index + 1} 條"
            missing = [k for k in ALLOW_ENTRY_KEYS if k not in entry]
            if missing:
                bad.append(f"{where} 缺 {missing}——放行要說清楚放過哪個前綴、為什麼、什麼時候失效")
            odd = [k for k in entry if k not in ALLOW_ENTRY_KEYS]
            if odd:
                bad.append(f"{where} 多了不認識的鍵 {sorted(odd)}，只認 {list(ALLOW_ENTRY_KEYS)}")
            for key in ALLOW_ENTRY_KEYS:
                value = entry.get(key)
                if key in entry and (not isinstance(value, str) or not value.strip()):
                    bad.append(f"{where} 的 {key} 必須是非空字串，實際是 {value!r}")
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _str_list(settings: dict[str, object], key: str) -> list[str]:
    """讀一張已經驗過形狀的字串清單。刻意寫成函式而不是就地取用：
    這樣不需要在取用點掛型別忽略註解（那種註解每一條都要帶到期日）。"""
    value = settings.get(key, [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _allow_entries(settings: dict[str, object]) -> list[tuple[str, str]]:
    """放行清單：(前綴, 理由)。形狀已經由 _assert_settings 驗過。"""
    raw = settings.get(ALLOW_KEY, [])
    out: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                out.append((str(entry.get("prefix", "")), str(entry.get("reason", ""))))
    return out


def _wrappers(settings: dict[str, object]) -> tuple[tuple[str, str], ...]:
    raw = settings.get(WRAPPERS_KEY, [])
    out: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for pair in raw:
            if isinstance(pair, list):
                out.append((str(pair[0]), str(pair[1])))
    return tuple(out)


def _rules(settings: dict[str, object], rel: str) -> Rules:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for raw in _str_list(settings, "key_patterns"):
        try:
            patterns.append((raw, re.compile(raw)))
        except re.error as exc:
            raise ToolBroken(
                f"{rel} 的 key_patterns 裡 {raw!r} 不是一個編得起來的正則（{exc}）"
                "——樣式編不起來就等於那一條沒在掃，不准當乾淨"
            ) from exc
    return Rules(
        env_globs=tuple(_str_list(settings, "env_file_globs")),
        name_words=tuple(word.upper() for word in _str_list(settings, "name_words")),
        placeholder_words=tuple(word.lower() for word in _str_list(settings, "placeholder_words")),
        wrappers=_wrappers(settings),
        key_patterns=tuple(patterns),
    )


# ── 三層判準 ───────────────────────────────────────────────────────────────────


def _secret_name(name: str, words: tuple[str, ...]) -> str:
    """名字裡有沒有卡上登記的機密字樣。前後要落在非英數字元的邊界上，
    所以 `DB_PASSWORD` 命中、`PASSWORDLESS` 不命中。沒命中回空字串。"""
    upper = name.upper()
    for word in words:
        start = upper.find(word)
        while start >= 0:
            before = upper[start - 1] if start else ""
            after = upper[start + len(word)] if start + len(word) < len(upper) else ""
            if not before.isalnum() and not after.isalnum():
                return word
            start = upper.find(word, start + 1)
    return ""


def _placeholder(value: str, rules: Rules) -> str:
    """值是不是佔位符。是的話回一句人話理由，不是回空字串。"""
    if not ALNUM_RE.search(value):
        return "整個值裡一個英數字元都沒有（空值、`***` 這種）"
    for head, tail in rules.wrappers:
        if value.startswith(head) and value.endswith(tail):
            return f"值被 {head}…{tail} 包起來，那是範本不是值"
    segments = [seg for seg in SEGMENT_SPLIT_RE.split(value.lower()) if seg]
    for word in rules.placeholder_words:
        if word in segments:
            return f"值裡有一段剛好等於卡上登記的佔位字 {word!r}"
    return ""


def _env_file_hits(where: str, rel: str, rules: Rules) -> list[str]:
    """第①層：檔名命中 env 樣式就紅，不看內容。"""
    name = rel.rsplit("/", maxsplit=1)[-1]
    for glob in rules.env_globs:
        if fnmatch.fnmatch(name, glob):
            return [
                f"[①存在性] {where} 的 {rel} 命中 env 樣式 {glob!r}"
                "——這種檔一進版控就紅，**不看內容**：v2 那四個檔就是靠「裡面只是樣板值」"
                "被放過的。機密只從環境變數或家目錄底下的設定讀，不進版控"
            ]
    return []


def _assignment_hits(where: str, rel: str, text: str, rules: Rules) -> list[str]:
    """第②層：賦值語境。名字帶機密字樣、值不是佔位符就紅（不看熵）。"""
    bad: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in ASSIGN_RE.finditer(line):
            word = _secret_name(match.group("name"), rules.name_words)
            if not word:
                continue
            value = match.group("value").strip().strip(VALUE_TRIM)
            if not VALUE_SHAPE_RE.match(value):
                continue
            why = _placeholder(value, rules)
            if why:
                continue
            bad.append(
                f"[②賦值語境] {where} 的 {rel}:{lineno} 把 {match.group('name')!r}"
                f"（名字帶了卡上登記的 {word!r}）指定成一個不是佔位符的值"
                "——這一層不看熵、不看長度：v2 洩漏的就是一個八位十六進位的樣板值，"
                "通用規則對它天生盲。值要嘛拿掉、要嘛換成範本寫法"
            )
    return bad


def _key_hits(where: str, rel: str, text: str, rules: Rules) -> list[str]:
    """第③層：格式金鑰的已知前綴樣式。"""
    bad: list[str] = []
    for raw, pattern in rules.key_patterns:
        match = pattern.search(text)
        if match:
            lineno = text.count("\n", 0, match.start()) + 1
            bad.append(
                f"[③格式金鑰] {where} 的 {rel}:{lineno} 命中卡上登記的金鑰樣式 {raw!r}"
                "——這是一把有固定前綴的金鑰。在版控裡出現就當它已經洩漏：先在發行方那邊輪替，"
                "再處理這棵樹"
            )
    return bad


def _all_layers(where: str, rel: str, text: str, rules: Rules) -> list[str]:
    return [
        *_env_file_hits(where, rel, rules),
        *_assignment_hits(where, rel, text, rules),
        *_key_hits(where, rel, text, rules),
    ]


# ── git ───────────────────────────────────────────────────────────────────────


def _git(
    argv: list[str],
    cwd: Path,
    *,
    what: str,
    env: dict[str, str] | None = None,
    allow: tuple[int, ...] = (0,),
    stdin: str = "",
) -> subprocess.CompletedProcess[bytes]:
    """跑一次 git。叫不動、或退出碼不在 ``allow`` 裡，一律 ToolBroken（回 2，不是回 0）。

    刻意收 bytes 不收 text：歷史裡的 blob 不保證是合法 UTF-8，交給呼叫端自己決定怎麼解碼。
    """
    try:
        proc = subprocess.run(
            ["git", *argv],
            cwd=cwd,
            env=env,
            input=stdin.encode("utf-8"),
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise ToolBroken(f"外部工具 git 不在 PATH（{what}）：{exc}") from exc
    except OSError as exc:
        raise ToolBroken(f"叫不動 git（{what}）：{exc}") from exc
    if proc.returncode not in allow:
        tail = proc.stderr.decode("utf-8", "replace").strip()[:300]
        raise ToolBroken(f"git {' '.join(argv)} 回 {proc.returncode}（{what}）：{tail}")
    return proc


def _git_out(argv: list[str], cwd: Path, *, what: str) -> str:
    """跑一次 git，把 stdout 當 UTF-8 讀出來（讀不懂的位元組換成替代字元）。"""
    return _git(argv, cwd, what=what).stdout.decode("utf-8", "replace")


def _toplevel(scan_root: Path) -> Path | None:
    """這棵樹是哪個 git 工作樹的根？不是 git 工作樹就回 None。"""
    proc = _git(
        ["rev-parse", "--show-toplevel"],
        scan_root,
        what="問這棵樹是不是 git 工作樹的根",
        allow=(0, 128),
    )
    line = proc.stdout.decode("utf-8", "replace").strip()
    if proc.returncode != 0 or not line:
        return None
    return Path(line).resolve()


def _assert_not_shallow(work_tree: Path) -> None:
    """淺 clone 一律 ToolBroken：看不到舊提交，「掃過全歷史」就是假話。"""
    answer = _git_out(
        ["rev-parse", "--is-shallow-repository"], work_tree, what="問這棵樹是不是淺 clone"
    ).strip()
    if answer not in ("true", "false"):
        raise ToolBroken(f"問「是不是淺 clone」得到我看不懂的答案 {answer!r}，這一跑不算數")
    if answer == "true":
        raise ToolBroken(
            f"{work_tree} 是淺 clone（rev-parse --is-shallow-repository 回 true）"
            "——舊提交根本不在這個物件庫裡，一個「加進來又刪掉」的金鑰在這種樹上列不出來。"
            "在淺 clone 上宣稱掃過全歷史是假話，一律回 2：CI 的 checkout 要帶 fetch-depth: 0"
        )


def _history_blobs(work_tree: Path) -> list[tuple[str, str, bytes]]:
    """歷史裡每一顆帶路徑的 blob：(sha, 路徑, 內容)。同一顆只算一次。"""
    listing = _git_out(
        ["rev-list", "--objects", "--all"], work_tree, what="列舉歷史裡走得到的每一顆物件"
    )
    paths: dict[str, str] = {}
    for line in listing.splitlines():
        sha, _, rest = line.partition(" ")
        rel = rest.strip()
        if not sha.strip() or not rel:
            continue
        paths.setdefault(sha.strip(), rel)
    if not paths:
        raise ToolBroken(
            f"{work_tree} 的歷史裡一顆帶路徑的物件都沒有"
            "——沒掃到任何歷史，「歷史裡沒有機密」這句話不算數"
        )
    order = sorted(paths)
    raw = _git(
        ["cat-file", "--batch"],
        work_tree,
        what="讀歷史裡那些物件的內容",
        stdin="\n".join(order) + "\n",
    ).stdout
    out: list[tuple[str, str, bytes]] = []
    position = 0
    for sha in order:
        newline = raw.find(b"\n", position)
        if newline < 0:
            raise ToolBroken(f"讀 {sha[:9]} 的時候 cat-file 的輸出斷在半路，我沒看懂就不出結論")
        header = raw[position:newline].decode("utf-8", "replace").split()
        position = newline + 1
        if len(header) < len(("sha", "type", "size")):
            raise ToolBroken(f"cat-file 這一行我沒看懂：{header!r}（要 sha、型別、大小三欄）")
        try:
            size = int(header[2])
        except ValueError as exc:
            raise ToolBroken(f"cat-file 說 {sha[:9]} 的大小是 {header[2]!r}，我沒看懂（{exc}）") from exc
        body = raw[position : position + size]
        position += size + len("\n")
        if header[1] == "blob":
            out.append((sha, paths[sha], body))
    return out


# ── 樣本樹的歷史 ───────────────────────────────────────────────────────────────


def _read_plan(decl: Path) -> tuple[list[Step], bool]:
    """讀樣本宣告的歷史。格式看不懂就 ToolBroken——樣本建不出來就等於什麼都沒掃到。"""
    text = _read_text(decl, str(decl))
    steps: list[Step] = []
    shallow = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == DECL_SHALLOW:
            shallow = True
            continue
        tokens = line.split(maxsplit=len(("commit", "action", "path")))
        if tokens[0] != DECL_COMMIT:
            raise ToolBroken(
                f"{decl}:{lineno} 開頭只認 {DECL_COMMIT!r} 或單獨一行 {DECL_SHALLOW!r}，"
                f"實際是 {tokens[0]!r}"
            )
        if len(tokens) < len(("commit", "action", "path")):
            raise ToolBroken(
                f"{decl}:{lineno} 一筆要寫滿 {DECL_COMMIT} <{DECL_ADD}|{DECL_DELETE}> <路徑> "
                f"[內容取到行尾]：{line!r}"
            )
        action, rel = tokens[1], tokens[2]
        if action not in (DECL_ADD, DECL_DELETE):
            raise ToolBroken(f"{decl}:{lineno} 動作只認 {DECL_ADD!r} 與 {DECL_DELETE!r}，實際是 {action!r}")
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ToolBroken(f"{decl}:{lineno} 的路徑 {rel!r} 往樣本樹外面指，不准")
        body = tokens[3] if len(tokens) > len(("commit", "action", "path")) else ""
        if action == DECL_ADD and not body:
            raise ToolBroken(f"{decl}:{lineno} 是 {DECL_ADD} 卻沒寫內容——建出來的 blob 會是空的")
        steps.append(Step(action=action, path=rel, body=body))
    if not steps:
        raise ToolBroken(f"{decl} 裡一筆提交都沒有——這份樣本建不出歷史")
    return steps, shallow


def _fixture_env(tmp: Path) -> dict[str, str]:
    """建樣本歷史用的乾淨環境：不吃外面的 git 設定，也不吃外面的 GIT_DIR。"""
    env = {k: v for k, v in os.environ.items() if k not in DROPPED_GIT_ENV}
    nowhere = str(tmp / "no-such-gitconfig")
    env["GIT_CONFIG_GLOBAL"] = nowhere
    env["GIT_CONFIG_SYSTEM"] = nowhere
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GIT_AUTHOR_NAME"] = "Fixture Author"
    env["GIT_AUTHOR_EMAIL"] = "fixture@aosr.invalid"
    env["GIT_AUTHOR_DATE"] = FIXTURE_DATE
    env["GIT_COMMITTER_NAME"] = "Fixture Committer"
    env["GIT_COMMITTER_EMAIL"] = "fixture@aosr.invalid"
    env["GIT_COMMITTER_DATE"] = FIXTURE_DATE
    return env


def _nothing() -> None:
    """真的工作樹不用清什麼。"""


def _materialize(scan_root: Path, decl: Path) -> tuple[Path, Callable[[], None]]:
    """照樣本宣告，在暫存目錄裡建一段真的 git 歷史。回傳（要掃的工作樹, 清乾淨的函式）。"""
    plan, shallow = _read_plan(decl)
    tmp = Path(tempfile.mkdtemp(prefix="aosr-secrets-", dir=scan_root))

    def cleanup() -> None:
        # 清不掉自己開的暫存目錄就是這一跑被汙染了，讓它回 2，不要吞。
        try:
            shutil.rmtree(tmp)
        except OSError as exc:
            raise ToolBroken(f"清不掉自己開的暫存目錄 {tmp}：{exc}") from exc

    work = tmp / "repo"
    try:
        work.mkdir()
        env = _fixture_env(tmp)
        _git(
            ["-c", "init.defaultBranch=main", "init", "--quiet"],
            work,
            what="開樣本用的暫存 repo",
            env=env,
        )
        for index, step in enumerate(plan, start=1):
            target = work / step.path
            if step.action == DECL_ADD:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(step.body + "\n", encoding="utf-8")
            else:
                if not target.is_file():
                    raise ToolBroken(f"{decl} 要刪 {step.path}，但那個檔在暫存 repo 裡不存在")
                target.unlink()
            _git(["add", "--all", "--", step.path], work, what=f"把 {step.path} 的變動放進索引", env=env)
            _git(
                ["commit", "--no-verify", "--quiet", "-m", f"樣本歷史第 {index} 筆（{step.action}）"],
                work,
                what=f"在暫存 repo 補一筆提交（{step.action} {step.path}）",
                env=env,
            )
        if shallow:
            deep = work
            work = tmp / "shallow"
            # `--depth` 只有走 file:// 才吃得到；裸路徑會被 git 忽略，clone 出來是完整的。
            _git(
                ["clone", "--quiet", "--depth", "1", f"file://{deep}", str(work)],
                tmp,
                what="照宣告做一次淺 clone（用來證「淺 clone 一律回 2」）",
                env=env,
            )
    except Exception:
        cleanup()
        raise
    return work, cleanup


def _resolve_history(scan_root: Path) -> tuple[Path, Callable[[], None]]:
    """決定要掃哪棵樹的歷史。兩條路：真的 git 工作樹，或樣本宣告出來的歷史。"""
    decl = scan_root / HISTORY_DECL
    top = _toplevel(scan_root)
    if top is not None and top == scan_root:
        if decl.exists():
            raise ToolBroken(
                f"這是真的 git 工作樹的根，卻放著樣本用的 {HISTORY_DECL}"
                "——那條路只給必紅樣本用；真的工作樹裡出現它，等於真歷史被一個檔案繞過"
            )
        return scan_root, _nothing
    if decl.is_file():
        return _materialize(scan_root, decl)
    raise ToolBroken(
        f"{scan_root} 既不是 git 工作樹的根，也沒有 {HISTORY_DECL}"
        "——我拿不到歷史，「歷史裡沒有機密」這句話不算數"
    )


# ── 掃描面 ─────────────────────────────────────────────────────────────────────


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：列舉集合的全部，扣掉卡上登記的前綴。

    歷史裡的 blob 不算掃描面上的檔（它們沒有對應的工作樹路徑，也不在列舉集合裡）；
    樣本樹根的宣告檔算，這支檢查真的會讀它（只是不對它的內容下判斷）。
    """
    settings = _card_settings(scan_root, files)
    exempt = _str_list(settings, "scan_exempt_prefixes")
    return sorted(
        f
        for f in files
        if not any(f.relative_to(scan_root).as_posix().startswith(prefix) for prefix in exempt)
    )


def _matched_allow(rel: str, allow: list[tuple[str, str]]) -> str:
    for prefix, _ in allow:
        if rel.startswith(prefix):
            return prefix
    return ""


def check(scan_root: Path, files: list[Path]) -> list[str]:
    rel_card, _ = _card_data(scan_root, files)
    settings = _card_settings(scan_root, files)
    rules = _rules(settings, rel_card)
    exempt = _str_list(settings, "scan_exempt_prefixes")
    allow = _allow_entries(settings)

    picked = targets(scan_root, files)
    if not picked:
        raise ToolBroken(
            f"{scan_root} 扣掉 {exempt} 之後一個檔都不剩"
            "——這一跑沒掃到東西，「沒有機密」這句話不算數"
        )

    used: set[str] = set()
    bad: list[str] = []
    for path in picked:
        rel = path.relative_to(scan_root).as_posix()
        if rel == HISTORY_DECL:
            # 這支檢查自己的道具：裡面的假金鑰是刻意的宣告，不是這棵樹的機密。
            continue
        if not path.is_file():
            # git 認得、檔案系統上不在（剛被刪掉還沒提交）。不猜內容，跳過。
            continue
        hits = _all_layers(WHERE_TREE, rel, _decoded(path, rel), rules)
        matched = _matched_allow(rel, allow)
        if hits and matched:
            used.add(matched)
            continue
        bad += hits

    work_tree, cleanup = _resolve_history(scan_root)
    try:
        _assert_not_shallow(work_tree)
        blobs = _history_blobs(work_tree)
        for sha, rel, body in blobs:
            if any(rel.startswith(prefix) for prefix in exempt):
                continue
            where = f"歷史 blob {sha[:9]}"
            hits = _all_layers(where, rel, body.decode("utf-8", "replace"), rules)
            matched = _matched_allow(rel, allow)
            if hits and matched:
                used.add(matched)
                continue
            bad += [
                f"{hit}。歷史裡這一筆**刪不掉**（主線 ruleset 不准改寫歷史）："
                "先在發行方那邊輪替那把憑證，再具名放行"
                for hit in hits
            ]
    finally:
        cleanup()

    print(f"tree_files={len(picked)} history_blobs={len(blobs)} allow_used={len(used)}")
    stale = sorted({prefix for prefix, _ in allow} - used)
    if stale:
        print(
            f"NOTE: 卡上有 {len(stale)} 條放行這一跑沒有被用到（今天沒有對象，可能該從卡上刪掉）：{stale}",
            file=sys.stderr,
        )
    if used:
        print(f"NOTE: 這一跑用到的放行：{sorted(used)}", file=sys.stderr)
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="密碼、金鑰、.env 一進版控就紅，歷史也要掃",
            targets=targets,
        )
    )
