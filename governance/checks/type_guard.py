#!/usr/bin/env python3
"""型別警衛：型別對不上就紅、公開函式一律要有標註、逃生門一律紅。

掃描面是版控裡的每一支 ``.py``（扣掉卡上登記的前綴：必紅樣本樹與本機 agent 工具目錄），
加上所有規矩卡（門檻與放行清單住在卡上，這支檢查要打開每一張去找自己那一張），
以及卡上登記的那個設定檔——第①層真的會讀它，讀了就要宣告。

**第①層 mypy 嚴格模式**

把掃描面上那一整組 ``.py`` 一次餵給 ``mypy``，設定檔用卡上登記的那個（今天是
``pyproject.toml`` 的 ``[tool.mypy]``）。這支程式自己不帶任何嚴格度旗標：嚴格度只有一個家，
它被改寬了要在那個 diff 上看得見，不是藏在檢查程式裡。

三種情況 raise :class:`ToolBroken` 讓外殼回 2，不回 0：設定檔不在（沒有尺）、``mypy``
不在 PATH 上（第①層整層沒跑）、``mypy`` 回一個既不是「乾淨」也不是「找到錯誤」的離開碼
（它自己崩掉了，這一跑量不到型別）。刻意叫裸的 ``mypy`` 走 PATH（``uv run`` 會把它放進來），
不走直譯器的 ``-m``：後者抽不掉，「抽掉外部工具就要回 2」那一關會變成空話。

快取刻意放在系統的暫存目錄、跑完刪掉。mypy 預設在 cwd 底下開快取目錄，而 cwd 是掃描根
——必紅樣本樹也是掃描根，那些目錄會變成版控樹裡的殘留（測試收尾的守衛會抓到）。

**第②層 逃生門掃描（完全不看 mypy 的旗標與輸出）**

用 :mod:`ast` 與 :mod:`tokenize` 判三件事：

1. 標註位置（參數、回傳、變數標註、型別別名）出現 ``Any``。巢狀的也算——
   ``dict[str, Any]`` 一樣咬。可以在卡的 ``[[settings.allow]]`` 具名放行一個檔。
2. ``cast(Any, …)`` 這個呼叫本身。沒有放行的寫法。
3. 抑制型別檢查的行尾註解沒帶錯誤碼（方括號裡沒寫明是哪一種錯）。沒有放行的寫法。

**為什麼第②層不能靠 mypy。** 嚴格模式那一包不含 ``disallow_any_explicit``，所以
``-> Any`` 在純嚴格模式底下合法通過；``warn_unused_ignores`` 也只抓「多餘的抑制」，
抓不到「真的在壓一個錯誤」的那種。這兩件事各有一份必紅樣本盯著：那兩棵樹餵給 mypy 是
乾淨的，第②層要是偷懶靠旗標，它們會靜靜回綠。

抑制註解的**理由與到期日不歸這張卡管**——那兩格由規矩卡 ``exemptions-need-expiry`` 判，
這裡只多守一格（錯誤碼）。兩張卡看同一行的不同格，刻意不重疊：兩張卡去判同一格，改壞了
不知道是哪一張在咬、改對了也不知道是誰放的，那是 v2 事故 ``guard-teeth-shadow-each-other``
的形狀。

**沒掃到東西就回 2**

讀不到卡的 ``[settings]``、``[settings]`` 形狀壞掉、掃描面上一支 ``.py`` 都沒有、
某支 ``.py`` 剖不開，一律 raise :class:`ToolBroken`。這一跑沒量到東西，
「沒問題」這句話就不算數。

血債：沒有。這張卡是好習慣卡，理由寫在 ``governance/rules/type-guard.toml`` 的檔頭，
它刻意沒管的七件事也在那張卡的檔尾。
"""
from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tempfile
import tokenize
import tomllib
from collections.abc import Iterator
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.loader import (
    EXEMPTION_KEYS,
    RULES_DIR,
    setting_int,
    setting_strings,
    setting_tables,
    setting_text,
)

# 這張卡的 id。門檻與放行清單只從「id 是這個」的那張卡讀。為什麼靠 id 認卡而不靠 check 欄：
# 必紅樣本是一棵棵獨立的迷你掃描根，每棵樹裡都得放一張卡（門檻在卡上），而樣本樹裡沒有
# governance/checks/ 底下那支程式——寫了 check 欄，refs-and-links-resolve 會把它當引用去解析。
CARD_ID = "type-guard"

PYTHON_SUFFIX = ".py"
CARD_SUFFIX = ".toml"

# 卡上 [settings] 的形狀。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
TEXT_KEYS = ("config_file",)
INT_KEYS = ("mypy_timeout_seconds",)
LIST_KEYS = ("scan_exempt_prefixes",)
ALLOW_KEY = "allow"
# 一筆放行三格：放行哪個檔（path），加上放行條目共用的兩格（reason ＋ expires，形狀定義在
# governance/loader.py 的 EXEMPTION_KEYS，由規矩卡 exemptions-need-expiry 統一）。
ALLOW_ENTRY_KEYS = ("path", *EXEMPTION_KEYS)
SETTINGS_KEYS = (*TEXT_KEYS, *INT_KEYS, *LIST_KEYS, ALLOW_KEY)

# ① 怎麼叫 mypy。裸的名字走 PATH（理由見模組說明）。
MYPY_ARGV = ("mypy",)
MYPY_CONFIG_FLAG = "--config-file"
MYPY_CACHE_FLAG = "--cache-dir"
# 只調輸出的形狀，不調嚴格度（嚴格度全部住在卡上登記的設定檔裡）。
MYPY_OUTPUT_FLAGS = ("--show-error-codes", "--no-color-output", "--no-error-summary")
CACHE_PREFIX = "aosr-mypy-cache-"

# mypy 自己的離開碼協定：乾淨／找到錯誤。別的值一律當「它自己崩掉」。
# 這兩個數字剛好跟離開碼約定重疊，但它們是 mypy 的協定不是我們的，所以另外命名、不共用。
MYPY_EXIT_CLEAN = 0
MYPY_EXIT_ERRORS_FOUND = 1

# mypy 的一行錯誤：`路徑:行號[:欄號]: error: 訊息  [錯誤碼]`。note: 那幾行不算違規。
MYPY_ERROR_RE = re.compile(r"^[^:]+:\d+(?::\d+)?: error: ")

# ② Any 這個名字。`typing.Any`／`t.Any` 這種帶前綴的寫法靠最後一段認。
ANY_NAME = "Any"
CAST_NAME = "cast"
# 字串形式的標註（`def f(x: "dict[str, Any]")`）：只認整詞，`AnyStr`／`MyAny` 不算。
STRING_ANY_RE = re.compile(rf"(?<![0-9A-Za-z_]){ANY_NAME}(?![0-9A-Za-z_])")

# ② 抑制型別檢查的行尾註解。比對的是 mypy 自己認得的每一種寫法（冒號前後的空白可有可無），
# 不是一個固定字面值——只認一種拼法的話，少打一個空白就整條繞過去了。
IGNORE_MARKER_RE = re.compile(r"#\s*type:\s*ignore")
# 帶錯誤碼＝那個標記後面緊接一個方括號。
IGNORE_CODE_RE = re.compile(IGNORE_MARKER_RE.pattern + r"\s*\[")

# 自己一個作用域、以及「有簽章」的那些節點。
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


# ── 卡上的門檻與名單 ────────────────────────────────────────────────────────


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（門檻與放行清單住在其中一張上）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == CARD_SUFFIX)


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這張卡的門檻與放行清單。

    找不到、找到多張、或形狀不對，一律 raise ToolBroken——沒有尺就不出結論。
    """
    mine: list[tuple[str, dict[str, object]]] = []
    for path in _card_files(scan_root, files):
        rel = path.relative_to(scan_root).as_posix()
        try:
            data = tomllib.loads(_read_text(path, rel))
        except tomllib.TOMLDecodeError as exc:
            raise ToolBroken(f"{rel} 解不開（{exc}）——我沒看懂就不出結論") from exc
        if data.get("id") == CARD_ID:
            mine.append((rel, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——門檻與放行清單只寫在卡上，讀不到這一跑就不算數"
        )
    rel, data = mine[0]
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（看門狗秒數、設定檔位置與放行清單都寫在那裡）")
    _assert_settings(settings, rel)
    return settings


def _allow_problems(settings: dict[str, object]) -> list[str]:
    if ALLOW_KEY not in settings:
        return [
            f"缺 {ALLOW_KEY}（{ANY_NAME} 放行清單，沒有也要明寫 {ALLOW_KEY} = []）"
            "——省略跟「空的」不是同一件事：省略讀起來像忘了寫，這一條就不知道自己有沒有清單"
        ]
    allow = settings[ALLOW_KEY]
    if not isinstance(allow, list) or not all(isinstance(x, dict) for x in allow):
        return [
            f"{ALLOW_KEY} 必須是 [[settings.{ALLOW_KEY}]] 表陣列"
            f"（沒有也要明寫 {ALLOW_KEY} = []），實際是 {allow!r}"
        ]
    bad: list[str] = []
    for index, entry in enumerate(allow):
        where = f"[[settings.{ALLOW_KEY}]] 第 {index + 1} 條"
        missing = [k for k in ALLOW_ENTRY_KEYS if k not in entry]
        if missing:
            bad.append(f"{where} 缺 {missing}——放行要說清楚是哪個檔、為什麼、到什麼時候")
        odd = [k for k in entry if k not in ALLOW_ENTRY_KEYS]
        if odd:
            bad.append(f"{where} 多了不認識的鍵 {sorted(odd)}，只認 {list(ALLOW_ENTRY_KEYS)}")
        for key in ALLOW_ENTRY_KEYS:
            value = entry.get(key)
            if key in entry and (not isinstance(value, str) or not value.strip()):
                bad.append(f"{where} 的 {key} 必須是非空字串，實際是 {value!r}")
    return bad


def _assert_settings(settings: dict[str, object], rel: str) -> None:
    """[settings] 的形狀。任何一格不對就 raise ToolBroken——尺壞掉的時候不出結論。"""
    bad: list[str] = []
    extra = sorted(k for k in settings if k not in SETTINGS_KEYS)
    if extra:
        bad.append(f"多了不認識的鍵 {extra}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒有門檻）")
    for key in TEXT_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}（第①層的尺住在哪，相對掃描根的路徑）")
        elif not isinstance(value, str) or not value.strip():
            bad.append(f"{key} 必須是非空字串，實際是 {value!r}")
    for key in INT_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}（看門狗秒數，正整數）")
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            bad.append(f"{key} 必須是 1 以上的整數，實際是 {value!r}——看門狗寫 0 等於沒有看門狗")
    for key in LIST_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}（沒有也要明寫 {key} = []）")
        elif not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
    bad += _allow_problems(settings)
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _allow_paths(settings: dict[str, object]) -> list[str]:
    """放行清單上的那幾個檔（相對掃描根）。"""
    return sorted({str(entry["path"]) for entry in setting_tables(settings, ALLOW_KEY)})


# ── 掃描面 ──────────────────────────────────────────────────────────────────


def _scanned_python(scan_root: Path, files: list[Path], exempt: list[str]) -> list[Path]:
    return sorted(
        f
        for f in files
        if f.suffix == PYTHON_SUFFIX
        and not any(f.relative_to(scan_root).as_posix().startswith(prefix) for prefix in exempt)
    )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀／會判的檔：所有規矩卡 ＋ 第①層的設定檔 ＋ 卡上沒被扣掉的每一支 .py。

    ``check()`` 自己也是叫這一支拿掃描面，所以 ``--list-files`` 印出來的清單就是真的被掃的
    那一組——不是第二份會各自漂的宣告。
    """
    settings = _card_settings(scan_root, files)
    python = _scanned_python(scan_root, files, setting_strings(settings, "scan_exempt_prefixes"))
    picked = {*_card_files(scan_root, files), *python}
    config = scan_root / setting_text(settings, "config_file")
    if config in files:
        picked.add(config)
    return sorted(picked)


# ── 第①層：mypy 嚴格模式 ───────────────────────────────────────────────────


def _tail(proc: subprocess.CompletedProcess[str]) -> str:
    rows = (proc.stdout + proc.stderr).strip().splitlines()
    return " ／ ".join(row.strip() for row in rows[-3:])


def _run_mypy(
    scan_root: Path, config: Path, rels: list[str], timeout: int
) -> subprocess.CompletedProcess[str]:
    """跑一次 mypy。快取放系統的暫存目錄、跑完刪掉（不准在被掃的樹裡留東西）。"""
    with tempfile.TemporaryDirectory(prefix=CACHE_PREFIX) as cache:
        argv = [
            *MYPY_ARGV,
            MYPY_CONFIG_FLAG,
            str(config),
            MYPY_CACHE_FLAG,
            cache,
            *MYPY_OUTPUT_FLAGS,
            *rels,
        ]
        try:
            return subprocess.run(
                argv, cwd=scan_root, capture_output=True, text=True, timeout=timeout
            )
        except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
            raise ToolBroken(
                f"mypy 跑不起來（{exc}）——第①層整層就是它在判，它不在的時候「型別沒問題」"
                "這句話不算數。它登記在 pyproject.toml 的 dev 群組，`uv sync --locked` 之後"
                "走 uv run 就在 PATH 上"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolBroken(
                f"mypy 跑超過卡上登記的看門狗秒數（{timeout}）還沒跑完"
                "——量到一半死掉，這一跑不算數。看門狗刻意比 job 那一層的分鐘上限早"
                "，不然整個 job 會被硬砍，收據上看不出是誰卡住的"
            ) from exc


def _mypy_hits(scan_root: Path, settings: dict[str, object], picked: list[Path]) -> list[str]:
    """第①層：整組 .py 一次餵給 mypy，把它印的每一行錯誤轉成一筆違規。"""
    config = scan_root / setting_text(settings, "config_file")
    if not config.is_file():
        raise ToolBroken(
            f"卡上登記的 mypy 設定檔不在：{config}"
            "——第①層的嚴格度只住在那個檔，讀不到就不知道自己拿的是哪一把尺，這一跑不算數"
        )
    rels = [p.relative_to(scan_root).as_posix() for p in picked]
    proc = _run_mypy(scan_root, config, rels, setting_int(settings, "mypy_timeout_seconds"))
    if proc.returncode == MYPY_EXIT_CLEAN:
        note(f"第①層：{len(rels)} 支 .py 餵給 mypy（設定檔 {config.name}），嚴格模式回乾淨")
        return []
    if proc.returncode != MYPY_EXIT_ERRORS_FOUND:
        raise ToolBroken(
            f"mypy 回 {proc.returncode}，既不是「乾淨」也不是「找到錯誤」——它自己崩掉了，"
            f"這一跑量不到型別，不出結論：{_tail(proc)}"
        )
    hits = [
        f"{line.strip()}（第①層 mypy 嚴格模式）"
        for line in proc.stdout.splitlines()
        if MYPY_ERROR_RE.match(line.strip())
    ]
    if not hits:
        raise ToolBroken(
            f"mypy 說它找到錯誤（回 {proc.returncode}），但它的輸出裡一行錯誤都剖不出來"
            f"——我沒看懂就不出結論：{_tail(proc)}"
        )
    return hits


# ── 第②層：逃生門掃描 ──────────────────────────────────────────────────────


def _parse(text: str, rel: str) -> ast.Module:
    try:
        return ast.parse(text, filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行剖不開（{exc.msg}）——我沒看懂就不出結論") from exc


def _comments(text: str, rel: str) -> dict[int, str]:
    """一支 .py 裡每一行的註解（一行最多一個註解 token）。斷不出 token 就回 2。"""
    out: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise ToolBroken(f"{rel} 斷不出 token（{exc}）——我沒看懂就不出結論") from exc
    return out


def _last_name(node: ast.expr) -> str:
    """``typing.Any`` 這種運算式的最後一段名字。認不出來就回空字串。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _has_any(expr: ast.expr) -> bool:
    """這一段運算式裡有沒有 ``Any``（巢狀的、以及字串形式的標註都算）。"""
    for node in ast.walk(expr):
        if isinstance(node, (ast.Name, ast.Attribute)) and _last_name(node) == ANY_NAME:
            return True
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and STRING_ANY_RE.search(node.value)
        ):
            return True
    return False


def _signature_args(args: ast.arguments) -> list[ast.arg]:
    """一個簽章裡的每一個參數（含 ``*args``／``**kwargs``）。"""
    out = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            out.append(extra)
    return out


def _annotations(tree: ast.Module) -> Iterator[tuple[str, ast.expr]]:
    """一份檔裡每一個標註位置：(位置的人話, 那一段運算式)。"""
    for node in ast.walk(tree):
        if isinstance(node, FUNCTION_NODES):
            if node.returns is not None:
                yield f"函式 {node.name}() 的回傳標註", node.returns
            for arg in _signature_args(node.args):
                if arg.annotation is not None:
                    yield f"函式 {node.name}() 的參數 {arg.arg} 的標註", arg.annotation
        elif isinstance(node, ast.AnnAssign):
            yield f"{ast.unparse(node.target)} 的變數標註", node.annotation
        elif isinstance(node, ast.TypeAlias):
            yield f"型別別名 {ast.unparse(node.name)}", node.value


def _any_count(tree: ast.Module) -> int:
    """這份檔裡有幾個標註位置出現 ``Any``（放行清單那幾個檔用這個數字報告）。"""
    return sum(1 for _, annotation in _annotations(tree) if _has_any(annotation))


def _any_hits(tree: ast.Module, rel: str) -> list[str]:
    return [
        f"{rel}:{annotation.lineno} {where} 出現 {ANY_NAME}"
        f"（`{ast.unparse(annotation)}`）——{ANY_NAME} 不是一個型別，是「這裡不要檢查」："
        f"標成 {ANY_NAME} 的值之後怎麼用都不會紅，型別檢查在那一格整個讓開。"
        f"寫得出具體型別就寫（連 object 都比 {ANY_NAME} 強，它什麼都不能做但還是型別安全的）；"
        f"真的是讀進來的動態資料，就在卡 {CARD_ID} 的 [[settings.{ALLOW_KEY}]] 具名放行那個檔，"
        "寫理由與到期日"
        for where, annotation in _annotations(tree)
        if _has_any(annotation)
    ]


def _cast_hits(tree: ast.Module, rel: str) -> list[str]:
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _last_name(node.func) != CAST_NAME:
            continue
        if node.args and _has_any(node.args[0]):
            bad.append(
                f"{rel}:{node.lineno} {CAST_NAME}({ANY_NAME}, …) 把型別檢查繞過去"
                f"——{CAST_NAME} 是對檢查器說「相信我，這個值是這個型別」，"
                f"轉成 {ANY_NAME} 就是「相信我，它什麼都是」：那不是標註，"
                "是把那個名字之後的每一次使用都放生。這一條沒有放行的寫法，"
                "要嘛轉成具體型別、要嘛用 isinstance 真的收窄"
            )
    return bad


def _ignore_hits(text: str, rel: str) -> list[str]:
    bad: list[str] = []
    for lineno, comment in sorted(_comments(text, rel).items()):
        if not IGNORE_MARKER_RE.search(comment) or IGNORE_CODE_RE.search(comment):
            continue
        bad.append(
            f"{rel}:{lineno} 抑制型別檢查的註解沒帶錯誤碼"
            "——裸的抑制不是「放過這一個錯誤」，是把那一行的型別檢查整個關掉："
            "明天那一行改成別的寫法，新長出來的錯誤也一起被壓住，而且沒有人會知道。"
            "方括號裡寫明是哪一種錯（檢查器的訊息尾巴就印著那個碼），就只放過那一種。"
            "這一行的理由與到期日由規矩卡 exemptions-need-expiry 守，這裡只守錯誤碼那一格"
        )
    return bad


# ── 主流程 ──────────────────────────────────────────────────────────────────


def _note_allowed(counts: dict[str, int], allowed: list[str]) -> None:
    """放行清單這一跑的實況：哪個檔放過幾處，以及哪幾筆今天沒有對象。"""
    live = sorted(f"{rel}（{count} 處）" for rel, count in counts.items() if count)
    if live:
        note(f"第②層的 {ANY_NAME} 放行清單這一跑放過 {live}——清單長大這件事要看得見")
    stale = sorted(set(allowed) - {rel for rel, count in counts.items() if count})
    if stale:
        note(f"卡上這幾筆放行這一跑沒有對象，可能該從卡上刪掉：{stale}")


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    picked = _scanned_python(scan_root, files, setting_strings(settings, "scan_exempt_prefixes"))
    if not picked:
        raise ToolBroken(
            f"{scan_root} 底下一支要判的 .py 都沒有（扣掉卡上登記的前綴之後是空集合）"
            "——這一跑沒讀到任何程式，「沒問題」這句話不算數"
        )

    # 第①層先跑：mypy 缺席、崩掉、或設定檔不在的時候要回 2，不准被第②層抓到的違規
    # 蓋成 1（回 1 讀起來是「量過了，有問題」，那是說謊）。
    bad = _mypy_hits(scan_root, settings, picked)

    allowed = _allow_paths(settings)
    counts: dict[str, int] = {rel: 0 for rel in allowed}
    for path in picked:
        rel = path.relative_to(scan_root).as_posix()
        if not path.is_file():
            # 版控認得、檔案系統上不在（剛被刪掉還沒 commit）。不猜內容，跳過。
            continue
        text = _read_text(path, rel)
        tree = _parse(text, rel)
        if rel in counts:
            counts[rel] = _any_count(tree)
        else:
            bad += _any_hits(tree, rel)
        bad += _cast_hits(tree, rel)
        bad += _ignore_hits(text, rel)

    _note_allowed(counts, allowed)
    return sorted(bad)


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="型別警衛：型別對不上就紅、公開函式一律要有標註、逃生門一律紅",
            targets=targets,
        )
    )
