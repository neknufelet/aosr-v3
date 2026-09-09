#!/usr/bin/env python3
"""環境交給 uv 管，``uv run`` 是唯一入口。

血債 ``interpreter-environment-never-machine-bound``（v2-audit/lessons.json，legacy_id L22，
3 次復發、嚴重度 4）。事故原文兩句就是這支檢查要擋的東西：

* 「再抓 canonical venv editable .pth 把 lib 導向 main checkout(4a01377b、682cd8c2)」
  ——同一個 worktree 裡跑起來的直譯器，import 到的是**另一個 checkout** 的程式碼。
* 「08-22 R4 direct-file 啟動 module root 落 scripts/ import 不到同 worktree fixture(exit 1)；
  module-mode 被外部同名 scripts package shadow exit 0」——同一份程式碼，跑法不同、答案不同。

事故自己的 ``v3_countermeasure`` 逐字寫了對策：「單一 lockfile＋src layout＋``uv run`` 為唯一
入口（exact sync 會清掉指向別 checkout 的 editable .pth）」。三條各擋一段：

**第一條 命令位置不准直接叫直譯器與工具**

掃三種面：``<scan_root>/.github/workflows/*.yml`` 的每個 ``run:``（含 ``run: |`` 區塊純量）、
版控裡的腳本（卡上登記的後綴 ``*.sh`` 與檔名 ``Makefile``）、``pyproject.toml`` 裡表名最後一段
是 ``scripts``／``tasks`` 的表。每一段原文拆成一個個命令（在 ``&&``／``||``／``|``／``;``／換行
斷開，剝掉行首的環境變數指派與 Makefile 的 ``@``／``-`` 前綴），只看**命令位置那個字**的
basename：是 ``uv`` 就合法（``uv run …``、``uv sync``、``uv lock`` 都算），落在卡上登記的
``bare_commands``（``python``／``python3``／``pytest``／``mypy``／``ruff``）就紅。

刻意不改成「整行出現 python 就紅」：那會咬到 ``echo python``、咬到路徑裡含 python 的檔名，
變成誤咬機。代價是 ``exec python``／``env python``／``bash -c "python x"``／Makefile 的
``$(PYTHON)`` 間接繞得過（卡面第 1 條有記）。

**第二條 Python 原始碼不准硬插模組搜尋路徑（AST 判，不是字串比對）**

抓 ``sys.path.append``／``insert``／``extend``，以及對 ``sys.path`` 本身的指派與 ``+=``。
**一律紅，沒有放行的寫法。** 立這張卡的時候放行過「路徑完全由 ``__file__`` 推出來」的自我
定位，那不是因為需要，是因為當時有別的工人同時在立卡、樹裡到處是那個形狀，一刀切會讓後
合併的 PR 在主線上變紅（issue #34）。收緊的時候那個形狀已經長成 13 行（11 支檢查 ＋ tests
兩支），全部拿掉之後這一條才跟著收成一律紅——
走 ``uv run python -m governance.checks.<x>``（cwd 在 repo 根）本來就 import 得到，那一行
是多餘的；``pytest`` 那邊要的是 ``pyproject.toml`` 的 ``[tool.pytest.ini_options]``
``pythonpath``，那是**設定**（走 PR 看得到、只有一份），不是程式裡自己插。

判法還是把被插的那個運算式攤平（模組層的名字轉一手，例如
``REPO = Path(__file__).resolve().parents[1]``），只是攤平的結果現在只用來說「中的是哪一種」：
卡上登記的 ``ambient_names``（``environ``／``getenv``／``getcwd``／``cwd``／``argv``／
``prefix``／``PYTHONPATH``）＞字串字面值（``".."`` 這種相對爬升就是 v2「module root 落
scripts/」的來源）＞``__file__`` 自我定位＞看不懂的寫法。四種都紅，只是訊息不同。

**第三條 鎖檔要在、同步要鎖死**

``uv.lock`` 必須在版控裡；上面掃到的命令裡至少要有一處 ``uv sync``；每一處 ``uv sync`` 都得帶
``--locked``。``--frozen`` 不算——它整個跳過鎖檔新鮮度檢查，鎖檔可以跟 pyproject 漂開，
就沒有事故對策要的 exact sync。

**什麼情況回 2（工具自壞，這一跑不算數）**

掃描根底下沒有一張卡宣告這支檢查、卡的 ``[settings]`` 形狀不對、``.github/workflows``
底下一份 workflow 都沒有、要看的檔讀不開或解不開。一律不准回 0。

門檻與名單只寫在卡的 ``[settings]``，這支程式沒有預設值。
"""
from __future__ import annotations

import ast
import re
import shlex
import sys
import tomllib
from pathlib import Path

from governance.exit_codes import ToolBroken, run
from governance.loader import RULES_DIR, setting_strings

# 這支檢查在卡裡的名字。門檻只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/uv_single_entrypoint.py"

WORKFLOW_DIR = ".github/workflows"
WORKFLOW_SUFFIXES = (".yml", ".yaml")
PYPROJECT = "pyproject.toml"
PYTHON_SUFFIX = ".py"

# 第二條的判定對象。ANCHOR 以前是放行錨點，現在只用來分辨「這一筆是自我定位那個形狀」，
# 好在訊息裡指出正確的做法（設定寫在 pyproject.toml，不是程式裡插）。
SYS_PATH = "sys.path"
ANCHOR = "__file__"

# workflow 的 run:（含 ``run: |``）。跟 green_must_be_real_green 用同一個樣式。
RUN_RE = re.compile(r"^(?P<pre>\s*(?:-\s+)?)run\s*:\s*(?P<rest>.*)$")
# 一行原文裡把命令斷開的地方。
SEGMENT_RE = re.compile(r"&&|\|\||\||;")
# 行首的環境變數指派（``FOO=bar cmd``）。
ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Makefile recipe 的前綴字元（@ 安靜、- 忽略錯誤、+ 一定執行）。
RECIPE_PREFIXES = "@-+"

# 門檻的形狀。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = (
    "bare_commands",
    "script_suffixes",
    "script_names",
    "script_tables",
    "sys_path_methods",
    "ambient_names",
    "scan_exempt_prefixes",
)
NONEMPTY_LIST_KEYS = ("bare_commands", "script_tables", "sys_path_methods", "ambient_names")
STR_KEYS = ("entrypoint", "sync_subcommand", "sync_required_flag", "lockfile")
SETTINGS_KEYS = (*LIST_KEYS, *STR_KEYS)


# ── 卡上的門檻 ──────────────────────────────────────────────────────────────


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這支檢查的門檻。

    找不到、找到多張、或門檻的形狀不對，一律 raise ToolBroken——沒有門檻就不出結論。
    """
    rules_dir = scan_root / RULES_DIR
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——門檻只寫在卡上，讀不到門檻這一跑就不算數"
        )
    path, data = mine[0]
    rel = path.relative_to(scan_root)
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（名單與門檻都寫在那裡）")
    _settings_problems(settings, str(rel))
    return settings


def _settings_problems(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(
            f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒門檻）"
        )
    for key in LIST_KEYS:
        value = settings.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif key in NONEMPTY_LIST_KEYS and not value:
            bad.append(f"{key} 不准是空 list——空的名單等於這一條沒在管")
    for key in STR_KEYS:
        value = settings.get(key)
        if not isinstance(value, str) or not value.strip():
            bad.append(f"{key} 必須是非空字串，實際是 {value!r}")
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _read(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


# ── 第一條：把掃描面拆成一個個命令 ──────────────────────────────────────────


def _run_bodies(text: str) -> list[str]:
    """把一份 workflow 裡每個 ``run:`` 的內容挖出來（含 ``run: |`` 那種區塊純量）。

    刻意只挖 run: 的內容、不掃整份 yaml：step 的名字是給人看的中文，掃進去會誤咬。
    """
    lines = text.splitlines()
    bodies: list[str] = []
    i = 0
    while i < len(lines):
        match = RUN_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        indent = len(match.group("pre"))
        rest = match.group("rest").strip()
        i += 1
        if rest and rest[0] not in "|>":
            bodies.append(rest)
            continue
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            body.append(line)
            i += 1
        bodies.append("\n".join(body))
    return bodies


def _tokenize(segment: str) -> list[str]:
    """把一個命令拆成 token。拆不開（引號不成對）就退回空白切，不猜也不吞。"""
    try:
        return shlex.split(segment, comments=True)
    except ValueError:
        return segment.split()


def _command_word(segment: str) -> tuple[str, list[str]]:
    """一個命令的「命令位置那個字」與整串 token。

    剝掉 Makefile recipe 的 ``@``／``-``／``+`` 前綴，以及行首的環境變數指派。
    """
    tokens = _tokenize(segment)
    if tokens and len(tokens[0]) > 1 and tokens[0][0] in RECIPE_PREFIXES:
        head = tokens[0].lstrip(RECIPE_PREFIXES)
        if head and not head.startswith("-"):
            tokens = [head, *tokens[1:]]
    while tokens and ENV_ASSIGN_RE.match(tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return "", []
    return tokens[0].rsplit("/", 1)[-1], tokens


def _segments(text: str, where: str) -> list[tuple[str, str]]:
    """把一段 shell 原文拆成 [(位置, 一個命令)]。反斜線續行接回同一個命令。"""
    out: list[tuple[str, str]] = []
    pending = ""
    start = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not pending:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            start = lineno
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        whole = pending + line
        pending = ""
        for piece in SEGMENT_RE.split(whole):
            piece = piece.strip()
            if piece and not piece.startswith("#"):
                out.append((f"{where} 第 {start} 行", piece))
    tail = pending.strip()
    if tail:
        out.append((f"{where} 第 {start} 行", tail))
    return out


def _table_commands(data: object, path: str, table_names: list[str]) -> list[tuple[str, str]]:
    """``pyproject.toml`` 裡表名最後一段落在 ``script_tables`` 的表，底下的字串值。"""
    out: list[tuple[str, str]] = []
    if not isinstance(data, dict):
        return out
    for key, value in data.items():
        here = f"{path}.{key}" if path else str(key)
        if str(key) in table_names:
            out += _strings_under(value, here)
        else:
            out += _table_commands(value, here, table_names)
    return out


def _strings_under(value: object, path: str) -> list[tuple[str, str]]:
    """一個表底下所有的字串值（poe 那種 ``cmd = "..."`` 也在內）。"""
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, inner in value.items():
            out += _strings_under(inner, f"{path}.{key}")
        return out
    if isinstance(value, list):
        out = []
        for i, inner in enumerate(value):
            out += _strings_under(inner, f"{path}[{i}]")
        return out
    return []


def _keep(scan_root: Path, files: list[Path], exempt: list[str]) -> list[Path]:
    """列舉集合扣掉卡上登記的前綴。"""
    return [
        f
        for f in sorted(files)
        if not any(f.relative_to(scan_root).as_posix().startswith(p) for p in exempt)
    ]


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（三條的名單與門檻只寫在卡上）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀／會判的檔：workflow ＋ 卡上登記的腳本（副檔名或檔名）＋ 每一支
    ``.py`` ＋ ``pyproject.toml`` ＋ 鎖檔 ＋ 所有規矩卡，全部扣掉卡上登記的前綴。

    鎖檔算在裡面是因為第三條對它下判斷（「版控裡沒有鎖檔」是一筆違規）。
    """
    settings = _card_settings(scan_root, files)
    suffixes = setting_strings(settings, "script_suffixes")
    names = setting_strings(settings, "script_names")
    exempt = setting_strings(settings, "scan_exempt_prefixes")
    keep = _keep(scan_root, files, exempt)

    picked = [
        f for f in keep if f.suffix == PYTHON_SUFFIX or f.suffix in suffixes or f.name in names
    ]
    picked += [
        f for f in keep if f.parent == scan_root / WORKFLOW_DIR and f.suffix in WORKFLOW_SUFFIXES
    ]
    picked += _card_files(scan_root, files)
    for name in (PYPROJECT, str(settings["lockfile"])):
        extra = scan_root / name
        if extra in files:
            picked.append(extra)
    return sorted(set(picked))


def _gather_commands(
    scan_root: Path, files: list[Path], settings: dict[str, object], keep: list[Path]
) -> list[tuple[str, str]]:
    """掃描面上所有的命令：[(位置, 一個命令的原文)]。"""
    suffixes = setting_strings(settings, "script_suffixes")
    names = setting_strings(settings, "script_names")
    tables = setting_strings(settings, "script_tables")

    workflows = sorted(
        f for f in files if f.parent == scan_root / WORKFLOW_DIR and f.suffix in WORKFLOW_SUFFIXES
    )
    if not workflows:
        raise ToolBroken(
            f"{WORKFLOW_DIR} 底下一份 workflow 都沒有——這張卡的第一條與第三條都要看 CI 怎麼跑，"
            "看不到就不出結論"
        )

    out: list[tuple[str, str]] = []
    for wf in workflows:
        rel = wf.relative_to(scan_root).as_posix()
        for body in _run_bodies(_read(wf, rel)):
            out += _segments(body, f"{rel} 的 run:")

    for path in keep:
        rel = path.relative_to(scan_root).as_posix()
        if path.suffix in suffixes or path.name in names:
            out += _segments(_read(path, rel), rel)

    pyproject = scan_root / PYPROJECT
    if pyproject in files and pyproject.is_file():
        try:
            data = tomllib.loads(_read(pyproject, PYPROJECT))
        except tomllib.TOMLDecodeError as exc:
            raise ToolBroken(f"{PYPROJECT} 解不開（{exc}）——我沒看懂就不出結論") from exc
        for where, command in _table_commands(data, "", tables):
            out.append((f"{PYPROJECT} 的 [{where}]", command))
    return out


# ── 第二條：AST 判 sys.path ─────────────────────────────────────────────────


def _dotted(expr: ast.expr) -> str:
    """把 ``sys.path`` 這種運算式攤成 ``"sys.path"``。認不出來就回空字串。"""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted(expr.value)
        return f"{base}.{expr.attr}" if base else ""
    return ""


def _module_names(tree: ast.Module) -> dict[str, ast.expr]:
    """模組層的 ``名字 = 運算式``。給「``REPO = Path(__file__)…``」那種轉一手的寫法用。"""
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.value is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            out[node.target.id] = node.value
    return out


def _leaves(
    expr: ast.expr, names: dict[str, ast.expr], ambient: list[str], seen: set[str]
) -> list[tuple[str, str]]:
    """把運算式攤平成一串 ``(種類, 名字)``。模組層的名字轉一手再攤。

    種類：``anchor``（``__file__``）／``ambient``（環境、cwd、參數）／``str``（字串字面值）
    ／``other``（函式名之類，不影響判定）。
    """
    out: list[tuple[str, str]] = []
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            if node.id == ANCHOR:
                out.append(("anchor", ANCHOR))
            elif node.id in ambient:
                out.append(("ambient", node.id))
            elif node.id in names and node.id not in seen:
                out += _leaves(names[node.id], names, ambient, seen | {node.id})
            else:
                out.append(("other", node.id))
        elif isinstance(node, ast.Attribute) and node.attr in ambient:
            out.append(("ambient", node.attr))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(("str", node.value))
    return out


def _verdict(
    expr: ast.expr, names: dict[str, ast.expr], ambient: list[str]
) -> str:
    """這個被插的路徑為什麼不准。一律不准，這裡只負責說中的是哪一種。

    回傳永遠是非空字串——這一條沒有放行的寫法（含由 ``__file__`` 推出來的自我定位）。
    """
    leaves = _leaves(expr, names, ambient, set())
    bad_ambient = sorted({name for kind, name in leaves if kind == "ambient"})
    literals = sorted({name for kind, name in leaves if kind == "str"})
    if bad_ambient:
        return f"路徑是從 {bad_ambient} 來的——那是環境／cwd／參數，跑在哪台機器就 import 到什麼"
    if literals:
        return (
            f"路徑裡有字串字面值 {literals}——相對路徑爬升（\"..\"）或寫死的絕對路徑"
            "都會隨 cwd 與機器改變，v2 的「module root 落 scripts/」就是這樣來的"
        )
    if any(kind == "anchor" for kind, _ in leaves):
        return (
            f"路徑是由 {ANCHOR} 推出來的自我定位——這一條以前放行它，現在不放行了："
            "走 uv run python -m（cwd 在 repo 根）本來就 import 得到，那一行是多餘的，"
            "而新來的檢查會照抄它。要宣告模組搜尋路徑就寫進 pyproject.toml 的 "
            "[tool.pytest.ini_options] pythonpath，那是設定、走 PR 看得到，不是程式裡自己插"
        )
    return "路徑推不出是從哪裡來的——搜尋路徑不准在程式裡自己決定，看不懂的寫法也不放行"


def _sys_path_problems(path: Path, rel: str, settings: dict[str, object]) -> list[str]:
    methods = setting_strings(settings, "sys_path_methods")
    ambient = setting_strings(settings, "ambient_names")
    text = _read(path, rel)
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行解不開（{exc.msg}）——我沒看懂就不出結論") from exc

    names = _module_names(tree)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in methods or _dotted(node.func.value) != SYS_PATH:
                continue
            index = 1 if node.func.attr == "insert" else 0
            if len(node.args) <= index:
                bad.append(
                    f"{rel}:{node.lineno} {SYS_PATH}.{node.func.attr}(…) 看不出插的是什麼路徑"
                    "——看不懂的寫法不放行"
                )
                continue
            why = _verdict(node.args[index], names, ambient)
            bad.append(
                f"{rel}:{node.lineno} `{ast.unparse(node)}` 硬插模組搜尋路徑：{why}。"
                "環境交給 uv 管，入口只有 uv run"
            )
        elif isinstance(node, ast.AugAssign) and _dotted(node.target) == SYS_PATH:
            bad.append(
                f"{rel}:{node.lineno} {SYS_PATH} += … 直接接上別的搜尋路徑"
                "——跟 append 是同一件事，環境交給 uv 管"
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if _dotted(target) == SYS_PATH or (
                    isinstance(target, ast.Subscript) and _dotted(target.value) == SYS_PATH
                ):
                    bad.append(
                        f"{rel}:{node.lineno} 整個換掉 {SYS_PATH}"
                        "——比硬插更徹底，搜尋路徑不准在程式裡自己決定"
                    )
    return bad


# ── 主檢查 ──────────────────────────────────────────────────────────────────


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    bare = setting_strings(settings, "bare_commands")
    entrypoint = str(settings["entrypoint"])
    sync_sub = str(settings["sync_subcommand"])
    sync_flag = str(settings["sync_required_flag"])
    lockfile = str(settings["lockfile"])
    exempt = setting_strings(settings, "scan_exempt_prefixes")

    keep = _keep(scan_root, files, exempt)

    bad: list[str] = []

    # 第一條 ＋ 第三條的後半（uv sync 有沒有帶 --locked）。
    syncs = 0
    for where, command in _gather_commands(scan_root, files, settings, keep):
        word, tokens = _command_word(command)
        if not word:
            continue
        if word == entrypoint:
            if len(tokens) > 1 and tokens[1] == sync_sub:
                syncs += 1
                if sync_flag not in tokens:
                    bad.append(
                        f"{where} 的 `{command}` 沒帶 {sync_flag}"
                        f"——{entrypoint} {sync_sub} 在鎖檔跟宣告對不上時會自己重解一次，"
                        "裝出來的環境可以跟鎖檔不一樣；--frozen 也不算，它整個跳過新鮮度檢查"
                    )
            continue
        if word in bare:
            bad.append(
                f"{where} 直接叫 `{word}`（`{command}`）而不是經 {entrypoint} run"
                f"——跑的是那台機器當下 PATH 前面的那一個，跟鎖檔沒關係。"
                "v2 事故裡同一份程式碼 direct-file 啟動 exit 1、module-mode 被同名 package "
                "shadow 而 exit 0，就是這種「跑法不同、答案不同」"
            )

    # 第三條的前半：鎖檔要在，而且真的有一步在同步。
    lock = scan_root / lockfile
    if lock not in files or not lock.is_file():
        bad.append(
            f"版控裡沒有 {lockfile}——事故對策的第一句是「單一 lockfile」，"
            "沒有鎖檔就沒有任何東西釘住環境，--locked 也無從比對"
        )
    if syncs == 0:
        bad.append(
            f"掃描面上沒有任何一處 `{entrypoint} {sync_sub}`"
            "——沒有人在照鎖檔把環境裝出來，鎖檔存在也只是擺著"
        )

    # 第二條：Python 原始碼裡的搜尋路徑。
    for path in keep:
        if path.suffix != PYTHON_SUFFIX or not path.is_file():
            continue
        bad += _sys_path_problems(path, path.relative_to(scan_root).as_posix(), settings)

    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="環境交給 uv 管、uv run 是唯一入口",
            targets=targets,
        )
    )
