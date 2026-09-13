#!/usr/bin/env python3
"""檢查程式的離開碼要誠實：0 乾淨／1 違規／2 工具自壞。

掃描面是 ``<scan_root>/governance/checks/*.py``（那一層，不含底下樣本樹裡的道具）
加上 ``governance/exit_codes.py``（離開碼約定本身，存在才驗）。三層：

**第一層 靜態（AST）**

1. 離開碼只准 0／1／2 三個字面值——``sys.exit``／``exit``／``os._exit``／``SystemExit``
   帶的整數，以及入口函式 ``main()`` 回的整數，都只能是這三個。多一個值 CI 只看得到
   「非零」，沒人知道那是違規還是沒跑成。
2. 子程序的退出碼不准被布林化。``if rc:``／``bool(rc)``／``not proc.returncode`` 這種寫法
   把 1（有違規）與 2（工具自壞）都當真值，兩種結局被摺成一種。要跟具體的值比對。
3. 不准把失敗吞掉：把離開碼或錯誤導掉的 shell 字樣（``|`` 兩個接 true、把 stderr 導進
   ``dev/null`` 之類）、``stderr=DEVNULL``、``contextlib.suppress``、內容只有 ``pass``
   的 ``except``。字樣只掃會跑的字串，**文件字串（docstring）刻意不掃**——它是給人看的
   說明，不是會跑的指令；這一段本身就是例子。
4. 離開碼約定自己不准被改寫：``governance/exit_codes.py`` 在的時候，它的
   ``CLEAN``／``VIOLATION``／``TOOL_BROKEN`` 必須是頂層的 0／1／2 字面值。尺可以被改，
   就等於沒有尺。

**第二層 每支檢查都要有一張卡宣告它**

``governance/checks/`` 底下的每一支都必須被某張讀得出來的卡宣告，那張卡才帶得出
控制樣本。沒有卡的檢查等於沒人證明過它咬得到已知案例。

**第三層 動態探針（四個，每一個都順便驗報告行）**

① 掃描根換成不存在的路徑 → 2；② 抽掉卡宣告的外部工具 → 2，另外把列舉子程序塞成
非零退出（``GIT_DIR`` 指向不存在的目錄）→ 2；③ 掃描集合是空的 → 2，**不准回 0**；
④ 卡宣告的控制樣本（已知會咬）→ 1，回 0 代表尺自己壞了。
每一次探針都要在 stdout 抓到 ``scan_root=<路徑> files=<整數> hits=<整數>`` 那一行，
而且不准出現「files=0 卻回 0」。

**第四層 單獨戳共用外殼的空集合守門**

第三層的探針③走的是「整支檢查」這條路，而每支檢查通常還有自己的「這棵樹裡沒有我要管
的東西」守門——掃描根是空的時候那道守門也回 2，**把外殼那道守門的死活整個遮住**。
所以要單獨戳：把掃描根那棵樹自己的 ``governance/exit_codes.py`` import 進來，拿一個
獨立暫存 Git 樹中的空子目錄餵 ``enumerate_files``，它必須 raise ``ToolBroken``。回得出一個空 list
就是「空集合當乾淨」，判違規。這一層是實測 PR #24 時被找出來的洞：外殼的空集合守門被
拆掉，四個探針全綠。

掃描面外的那一半：``measuring-tool-itself-was-wrong`` 那筆事故 20 次復發裡，多數是稽核
當下臨時打的 shell 指令，不是進版控的檢查程式，這張卡管不到（見卡的 related_lessons_why）。
共用外殼 ``governance/exit_codes.py`` 不在 ``governance/checks/`` 底下，它的行為由上面四個
探針間接管到——探針戳的是每支檢查跑起來的實際離開碼，外殼壞了每一支都會被抓到。
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken, note, run
from governance.loader import CHECKS_DIR, RULES_DIR, Card, card_problems, load_card

EXIT_CODES_FILE = "governance/exit_codes.py"

# 離開碼約定：這三個名字必須是頂層的整數字面值，而且就是這三個數。
EXPECTED_CONSTANTS = (("CLEAN", 0), ("VIOLATION", 1), ("TOOL_BROKEN", 2))

ALLOWED_EXIT_VALUES = (0, 1, 2)
EXIT_CALLS = ("sys.exit", "exit", "os._exit", "SystemExit")
ENTRY_FUNCTIONS = ("main",)

# 被當成「子程序退出碼」的名字。屬性一律認 .returncode。
RC_NAMES = frozenset({"rc", "retcode", "returncode", "exit_code", "exitcode"})

REPORT_RE = re.compile(r"^scan_root=(?P<root>.*?) files=(?P<files>\d+) hits=(?P<hits>\d+)\s*$")
EXPECTED_WHY = {
    VIOLATION: "1（抓到違規）",
    TOOL_BROKEN: "2（工具自壞，這一跑不算數）",
}

# 探針會遞迴（這支檢查自己也在掃描面裡）。深度到 1 就只跑靜態層，不然沒完沒了。
DEPTH_ENV = "AOSR_EXIT_PROBE_DEPTH"
MAX_PROBE_DEPTH = 1
PROBE_TIMEOUT = 300
MISSING_ROOT_NAME = "no-such-scan-root-EXIT-PROBE"
SABOTAGE_GIT_DIR = "/nonexistent-aosr-probe/no-such-git-dir"

# 「把失敗吞掉」的字樣。刻意用片段拼出來：這支檢查的掃描面包含 governance/checks/ 底下
# 的每一支，包括它自己，樣式表寫成完整字面值它會咬到自己。
_NULL = "/dev" + "/null"
_OR = "|" + "|"
SWALLOW_SNIPPETS = (
    _OR + " true",
    _OR + "true",
    _OR + " :",
    ";" + " true",
    "set" + " +e",
    "2>" + _NULL,
    "2> " + _NULL,
    ">" + _NULL + " 2>&1",
)


# 單獨戳共用外殼的探針。這一段餵給 ``python -c``，cwd 設在掃描根，所以 import 到的是
# 那棵樹自己的 governance/exit_codes.py，不是這個 repo 的。
# 離開碼：0 = 空集合有 fail closed（對）；3 = 空集合被當成正常結果（違規）；
# 4／5 = 約定檔沒有那兩個名字、或炸出別的錯（我沒看懂，外層回 2）。
SHELL_FAIL_CLOSED = 0
SHELL_NOT_FAIL_CLOSED = 3
SHELL_PROBE_CODE = """
import sys
from pathlib import Path

sys.path.insert(0, ".")
try:
    from governance.exit_codes import ToolBroken, enumerate_files
except Exception as exc:
    print(f"約定檔裡 import 不到 ToolBroken／enumerate_files：{exc}")
    sys.exit(4)
try:
    got = enumerate_files(Path(sys.argv[1]))
except ToolBroken:
    sys.exit(0)
except Exception as exc:
    print(f"空集合上炸出別的錯：{exc!r}")
    sys.exit(5)
print(f"空集合沒有 fail closed，回了 {got!r}")
sys.exit(3)
"""


# 開一顆獨立空 repo 要丟掉的 ambient git 環境變數：留著會把 git init 指到別人的 .git 目錄。
DROPPED_GIT_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_TEMPLATE_DIR",
)

# 複製「被測掃描根」自己的 governance 套件時要跳過的內容：大資料與本跑產物，不是這支檢查要判的。
COPY_GOVERNANCE_IGNORE = {"__pycache__", "receipts", "fixtures"}


def _empty_repo_env(tmp: Path) -> dict[str, str]:
    """開空 repo 的乾淨環境：不吃 ambient 的 GIT_* 指向，也不吃 global／system 設定、注入的 -c 設定與 template。"""
    env = {k: v for k, v in os.environ.items() if k not in DROPPED_GIT_ENV}
    # GIT_CONFIG_COUNT=0..N、GIT_CONFIG_KEY_0=…、GIT_CONFIG_VALUE_0=… 這種「第 N 格」會回頭
    # 灌進 git init 的 -c，單清 GIT_DIR 那幾個抓不到它；連前綴一起剝掉。只動這顆 sandbox 的
    # 環境，故意 sabotage 的原探針（_probe 裡 GIT_DIR 那一針）走的是另一條路、不受影響。
    for key in list(env):
        if key.startswith("GIT_CONFIG_"):
            env.pop(key, None)
    nowhere = str(tmp / "no-such-gitconfig")
    env["GIT_CONFIG_GLOBAL"] = nowhere
    env["GIT_CONFIG_SYSTEM"] = nowhere
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _git_init(repo: Path, env: dict[str, str], template: Path) -> None:
    """在 ``repo`` 開一顆空 repo。git 不在、超時、或 init 失敗一律 ToolBroken，不吞 stderr 與退出碼。"""
    # --template 押一個空目錄：把 ambient 的 init.templateDir（環境變數或 -c 注入）擋在門外，
    # 不讓外面的 hooks／樣板跑進這顆乾淨的探針 repo。global／system 設定已由 _empty_repo_env 擋掉。
    try:
        proc = subprocess.run(
            [
                "git",
                "-c",
                "init.defaultBranch=main",
                "init",
                "--quiet",
                "--template",
                str(template),
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise ToolBroken(f"外部工具 git 不在 PATH（開探針用的空 repo）：{exc}") from exc
    except OSError as exc:
        raise ToolBroken(f"叫不動 git（開探針用的空 repo）：{exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolBroken(f"git init 超過 {PROBE_TIMEOUT} 秒沒跑完（開探針用的空 repo）") from exc
    if proc.returncode != 0:
        raise ToolBroken(
            f"git init 回 {proc.returncode}（開探針用的空 repo）：{proc.stderr.strip()[:300]}"
        )


def _empty_git_repo(
    prefix: str, *, copy_governance_from: Path | None = None
) -> tuple[Path, Callable[[], None], Path]:
    """在系統暫存區開一顆「真的空」git repo，回傳（repo 裡一個空的子目錄, 清乾淨的函式, repo 根）。

    為什麼要一顆真的空 repo：動態探針戳的是「git ls-files 成功回空集合時，外殼／檢查有沒有
    fail closed」。量的前提正是 git ls-files 能**成功回空集合**（離開碼 0）；把探針目錄開在
    一棵不是 git 工作樹的地方，git ls-files 回 128 而不是空集合，就把「空集合那道守門」遮住，
    探針量到的變成「非零退出」那一件別的事——假綠。所以探針目錄必須住在一顆真的空 repo 裡，
    而探針目錄本身要是 repo 裡一個真正空的子目錄。

    ``copy_governance_from`` 有給時，把那一棵掃描根自己的 ``governance/`` 套件原封不動複製進
    這顆 repo（保持 bytes；跳過 fixtures／receipts／__pycache__ 這些大資料）。整支檢查
    跑起來（``python -m``）時 import 到的 ``governance.exit_codes`` 是這份副本，它的 ``repo_root()``
    會往上找到**這顆**獨立樹的 ``.git``，空集合子目錄才過得了「掃描根要在 repo 裡」那道門、
    真正量到「列舉出來是空集合」而不是「落在 repo 外面」——兩個都回 2，但量的是兩件不同的事。
    沒給（共用外殼直戳那一針）就不複製：那一針直接呼叫列舉函式、不走 resolve，繼續 import 原來源。
    """
    try:
        tmp = Path(tempfile.mkdtemp(prefix=prefix))
    except OSError as exc:
        # 暫存容器開不起來就是工具自壞：沒有樹可探，「沒抓到」這句話不算數。
        raise ToolBroken(f"開不了暫存 git 容器（mkdtemp）：{exc}") from exc
    repo = tmp / "repo"
    probe = repo / "probe"
    template = tmp / "template"

    def cleanup() -> None:
        # 先 rmdir 那個空子目錄：它一被寫進東西就刪不掉，那才是更根本的錯（被寫入）。不管
        # rmdir 成不成，容器本身都要 rmtree 清掉；「被寫入」的 ToolBroken 留到最後再報。
        probe_error: ToolBroken | None = None
        try:
            probe.rmdir()
        except OSError as exc:
            probe_error = ToolBroken(f"探針子目錄 {probe} 清不掉（被寫入？）：{exc}")
        try:
            shutil.rmtree(tmp)
        except OSError as exc:
            rmtree_error = ToolBroken(f"清不掉自己開的暫存 git 容器 {tmp}：{exc}")
            if probe_error is not None:
                raise probe_error from rmtree_error
            raise rmtree_error from exc
        if probe_error is not None:
            raise probe_error

    try:
        repo.mkdir()
        template.mkdir()
        env = _empty_repo_env(tmp)
        if copy_governance_from is not None:
            src = copy_governance_from / "governance"
            if not src.is_dir():
                raise ToolBroken(
                    f"被測掃描根 {copy_governance_from} 底下沒有 governance/，探針副本建不出來"
                )

            def _ignore(_directory: str, names: list[str]) -> set[str]:
                return {n for n in names if n in COPY_GOVERNANCE_IGNORE}

            shutil.copytree(src, repo / "governance", ignore=_ignore)
        _git_init(repo, env, template)
        probe.mkdir()
    except Exception as orig:
        # init 半路失敗也要把暫存目錄清掉，不留在系統暫存區；清不掉照樣報，不再用 ignore_errors 吞。
        try:
            shutil.rmtree(tmp)
        except OSError as exc:
            raise ToolBroken(
                f"開空 repo 失敗（{orig}）之後，清暫存容器 {tmp} 也失敗：{exc}"
            ) from orig
        # 容器已清掉之後才轉離開碼：建樹途中炸出的系統錯誤（含 copytree 的 shutil.Error，
        # 它繼承 OSError）是工具自壞，要轉 ToolBroken 讓外殼回 2；已經是 ToolBroken 的
        # 原樣往外丟（連訊息都不換），其餘例外也不吞。
        if isinstance(orig, OSError):
            raise ToolBroken(f"開空 repo 失敗（建樹的系統錯誤）：{orig}") from orig
        raise
    return probe, cleanup, repo


def _callee(func: ast.expr) -> str:
    """把 ``sys.exit`` 這種呼叫對象攤成 ``"sys.exit"``。認不出來就回空字串。"""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _callee(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _bad_exit_values(expr: ast.expr, where: str, rel: str) -> list[str]:
    """離開碼裡的整數字面值只准 0／1／2。值是呼叫結果（動態）就不猜。"""
    if isinstance(expr, ast.Call):
        return []
    bad: list[str] = []
    for node in ast.walk(expr):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool):
            bad.append(
                f"{rel}:{node.lineno} {where} 拿布林 {node.value} 當離開碼"
                "——只准 0（乾淨）／1（違規）／2（工具自壞）三個值"
            )
        elif isinstance(node.value, int) and node.value not in ALLOWED_EXIT_VALUES:
            bad.append(
                f"{rel}:{node.lineno} {where} 的離開碼是 {node.value}"
                "——只准 0（乾淨）／1（違規）／2（工具自壞）；多一個值 CI 只看得到「非零」"
            )
    return bad


def _truth_tests(tree: ast.AST) -> Iterator[ast.expr]:
    """所有「被當成真假值」的位置。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            yield node.test
        elif isinstance(node, ast.Assert):
            yield node.test
        elif isinstance(node, ast.BoolOp):
            yield from node.values
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield node.operand
        elif isinstance(node, ast.Call) and _callee(node.func) == "bool":
            yield from node.args
        elif isinstance(node, ast.comprehension):
            yield from node.ifs


def _is_returncode(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Attribute) and expr.attr == "returncode":
        return True
    return isinstance(expr, ast.Name) and expr.id in RC_NAMES


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """模組／類別／函式的文件字串節點。它們是人話說明，不是會跑的指令，不掃字樣。"""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            out.add(id(first.value))
    return out


def _swallows_exceptions(node: ast.ExceptHandler) -> bool:
    """``except ...:`` 底下只有 pass／... ——例外被吞掉。"""
    for stmt in node.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
            continue
        return False
    return True


def _parse(path: Path, rel: str) -> ast.Module:
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc
    try:
        return ast.parse(src, filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行解不開（{exc.msg}）——我沒看懂就不出結論") from exc


def _static_problems(path: Path, rel: str) -> list[str]:
    tree = _parse(path, rel)
    bad: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee(node.func) in EXIT_CALLS:
            for arg in node.args:
                bad += _bad_exit_values(arg, f"{_callee(node.func)}(...)", rel)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ENTRY_FUNCTIONS:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and inner.value is not None:
                    bad += _bad_exit_values(inner.value, f"入口 {node.name}() 的 return", rel)

    for test in _truth_tests(tree):
        if _is_returncode(test):
            bad.append(
                f"{rel}:{test.lineno} 子程序的退出碼被布林化（{ast.unparse(test)}）"
                "——1（有違規）與 2（工具自壞）都是真值，這樣寫把兩種結局摺成一種；要跟具體的值比對"
            )

    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            for snip in SWALLOW_SNIPPETS:
                if snip in node.value:
                    bad.append(f"{rel}:{node.lineno} 字串裡有把失敗吞掉的寫法 {snip!r}——失敗被吞掉就變成假綠")
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("stderr", "stdout") and _callee(kw.value).endswith("DEVNULL"):
                    bad.append(
                        f"{rel}:{node.lineno} {kw.arg}=DEVNULL 把子程序的輸出導進黑洞"
                        "——工具自壞的證據就看不見了"
                    )
            if _callee(node.func).endswith("suppress"):
                bad.append(f"{rel}:{node.lineno} suppress 把例外吞掉——工具自壞會被當成乾淨")
        if isinstance(node, ast.ExceptHandler) and _swallows_exceptions(node):
            bad.append(f"{rel}:{node.lineno} except 底下只有 pass——例外被吞掉，工具自壞會被當成乾淨")
    return bad


def _constant_problems(path: Path, rel: str) -> list[str]:
    """離開碼約定自己：CLEAN／VIOLATION／TOOL_BROKEN 必須是頂層的 0／1／2 字面值。"""
    tree = _parse(path, rel)
    seen: dict[str, int] = {}
    for node in tree.body:
        targets: list[ast.Name] = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        value = getattr(node, "value", None)
        if not targets or not isinstance(value, ast.Constant):
            continue
        if isinstance(value.value, int) and not isinstance(value.value, bool):
            for target in targets:
                seen[target.id] = value.value
    bad: list[str] = []
    for name, want in EXPECTED_CONSTANTS:
        if name not in seen:
            bad.append(f"{rel} 沒有在頂層把 {name} 定成整數字面值——離開碼約定必須寫死在版控裡")
        elif seen[name] != want:
            bad.append(f"{rel} 的 {name} = {seen[name]}，約定是 {want}——尺被改寫，等於沒有尺")
    return bad


def _tail(proc: subprocess.CompletedProcess[str]) -> str:
    rows = (proc.stdout + proc.stderr).strip().splitlines()
    return " ／ ".join(row.strip() for row in rows[-3:])


def _path_without(tool: str) -> str:
    """把裝著某個外部工具的目錄從 PATH 拿掉（不是清空 PATH）。"""
    kept = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and not (Path(d) / tool).exists()]
    # 路徑走 Path 組，不用 os.path.join：規矩卡 style-guard 那半（ruff 的 PTH）在守這件事。
    return os.pathsep.join(kept) or str(Path(os.sep, "nonexistent-aosr-probe", "no-such-bin-dir"))


def _probe_env(depth: int, *, path: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # 不要在樣本樹裡留 __pycache__：那些是掃描面上的垃圾，管路徑的檢查會看到。
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env[DEPTH_ENV] = str(depth + 1)
    # 不寫 .pyc：改了程式卻拿到 __pycache__ 裡的舊位元碼，是實測時撞過的坑。
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if path is not None:
        env["PATH"] = path
    if extra:
        env.update(extra)
    return env


def _probe(
    label: str,
    card: Card,
    scan_root: Path,
    target: Path,
    env: dict[str, str],
    want: int,
    rel: str,
    *,
    cwd: Path | None = None,
) -> list[str]:
    """跑一次探針：驗離開碼，順便驗報告行。

    ``cwd`` 給的時候就在那棵樹跑 ``python -m``（空集合那一針在複製了 governance 的獨立樹上跑，
    讓檢查的 ``repo_root()`` 找到獨立樹的 ``.git``）；沒給就照舊在原掃描根 cwd 跑。
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", card.check_module, "--scan-root", str(target)],
            cwd=cwd if cwd is not None else scan_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolBroken(f"探針「{label}」跑 {card.check_module} 超過 {PROBE_TIMEOUT} 秒") from exc

    bad: list[str] = []
    matched = [m for m in (REPORT_RE.match(ln.strip()) for ln in proc.stdout.splitlines()) if m]
    if not matched:
        bad.append(
            f"{rel} 探針「{label}」沒印出 scan_root=<路徑> files=<整數> hits=<整數> 那一行"
            f"——沒有這一行就分不出「回綠是因為乾淨」還是「因為什麼都沒掃到」：{_tail(proc)}"
        )
    elif int(matched[-1]["files"]) == 0 and proc.returncode == CLEAN:
        bad.append(
            f"{rel} 探針「{label}」印 files=0 卻回 0"
            "——空集合當乾淨。一個檔都沒掃到，「沒問題」這句話就不算數，要回 2"
        )
    if proc.returncode != want:
        bad.append(
            f"{rel} 探針「{label}」回 {proc.returncode}，應為 {EXPECTED_WHY[want]}：{_tail(proc)}"
        )
    return bad


def _probe_problems(card: Card, scan_root: Path, rel: str, depth: int) -> list[str]:
    if depth >= MAX_PROBE_DEPTH:
        note(f"動態探針在深度 {depth} 停止遞迴，{rel} 這一層只跑靜態掃描")
        return []

    bad: list[str] = []
    missing = scan_root / "governance" / MISSING_ROOT_NAME
    if missing.exists():
        raise ToolBroken(f"探針用的假路徑 {missing} 竟然真的存在，這個探針證明不了什麼")
    bad += _probe("掃描根不存在", card, scan_root, missing, _probe_env(depth), TOOL_BROKEN, rel)

    if card.external_tools:
        for tool in card.external_tools:
            bad += _probe(
                f"抽掉外部工具 {tool}",
                card,
                scan_root,
                scan_root,
                _probe_env(depth, path=_path_without(tool)),
                TOOL_BROKEN,
                rel,
            )
        if "git" in card.external_tools:
            bad += _probe(
                "列舉子程序非零退出（GIT_DIR 指向不存在的目錄）",
                card,
                scan_root,
                scan_root,
                _probe_env(depth, extra={"GIT_DIR": SABOTAGE_GIT_DIR}),
                TOOL_BROKEN,
                rel,
            )
    else:
        note(f"{rel} 的卡宣告 external_tools = []，沒有工具可抽，跳過那個探針")

    empty, empty_cleanup, empty_repo = _empty_git_repo("aosr-empty-probe-", copy_governance_from=scan_root)
    try:
        bad += _probe(
            "掃描集合是空的", card, scan_root, empty, _probe_env(depth), TOOL_BROKEN, rel, cwd=empty_repo
        )
    finally:
        empty_cleanup()

    control = card.control_path(scan_root)
    if not control.is_dir():
        return bad + [
            f"{rel} 的卡宣告的控制樣本 {card.control_fixture} 不存在"
            "——沒有已知會咬的輸入，就沒有人證明過這把尺咬得到東西"
        ]
    bad += _probe("控制樣本（已知會咬）", card, scan_root, control, _probe_env(depth), VIOLATION, rel)
    return bad


def _shell_problems(scan_root: Path, rel: str, depth: int) -> list[str]:
    """單獨戳共用外殼：空集合的時候它自己有沒有 fail closed。

    為什麼要跟探針③分開：每支檢查通常還有自己的「這棵樹裡沒有我要管的東西」守門，
    掃描根是空的時候那道守門也會回 2，把外殼那道守門的死活遮住——兩邊都回 2，從離開碼
    看不出外殼還活著沒有。這一針直接呼叫外殼的列舉函式，繞過每支檢查自己的守門。
    """
    empty, empty_cleanup, _empty_repo = _empty_git_repo("aosr-shell-probe-")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", SHELL_PROBE_CODE, str(empty)],
            cwd=scan_root,
            env=_probe_env(depth),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolBroken(f"戳 {rel} 的空集合守門超過 {PROBE_TIMEOUT} 秒") from exc
    finally:
        empty_cleanup()

    if proc.returncode == SHELL_FAIL_CLOSED:
        return []
    if proc.returncode == SHELL_NOT_FAIL_CLOSED:
        return [
            f"{rel} 的列舉在空集合上沒有 fail closed（{proc.stdout.strip()}）"
            "——空集合當乾淨，「沒找到違規」這句話就不算數，要 raise 讓外殼回 2。"
            "每支檢查自己的守門會把這個洞遮住，所以這一針直接戳外殼"
        ]
    raise ToolBroken(
        f"戳 {rel} 的空集合守門時我沒看懂（離開碼 {proc.returncode}）：{_tail(proc)}"
    )


def _check_programs(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面第一組：``governance/checks/`` 那一層的檢查程式。

    ``__init__.py`` 這種底線開頭的不算——它是套件標記，不是檢查程式（也沒有卡宣告它，
    掃了會被第二層判成「沒有卡的檢查」）。這件事同時寫在卡的 scope 裡（明寫扣掉它）。
    """
    checks_dir = scan_root / CHECKS_DIR
    return sorted(
        f for f in files if f.parent == checks_dir and f.suffix == ".py" and not f.name.startswith("__")
    )


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面第二組：所有規矩卡。每一張都要讀（第二層要問「這支檢查有卡宣告嗎」）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：檢查程式 ＋ 離開碼約定本身 ＋ 所有規矩卡。

    必紅樣本樹不在裡面：動態探針是把**另一支程式**餵給那些樹，這支檢查自己不讀它們的內容。
    """
    picked = [*_check_programs(scan_root, files), *_card_files(scan_root, files)]
    convention = scan_root / EXIT_CODES_FILE
    if convention in files:
        picked.append(convention)
    return sorted(set(picked))


def check(scan_root: Path, files: list[Path]) -> list[str]:
    depth = int(os.environ.get(DEPTH_ENV, "0"))
    programs = _check_programs(scan_root, files)
    if not programs:
        raise ToolBroken(
            f"{scan_root}/{CHECKS_DIR} 底下一支版控裡的檢查程式都沒有"
            "——這一跑沒掃到東西，「沒問題」這句話不算數"
        )

    by_check: dict[str, Card] = {}
    for path in _card_files(scan_root, files):
        if card_problems(path, scan_root):
            # 卡自己壞掉是第一張卡（rule-card-required-fields）的事，這裡只是讀不出控制樣本。
            continue
        declaring = load_card(path, scan_root)
        by_check.setdefault(declaring.check, declaring)

    bad: list[str] = []
    convention = scan_root / EXIT_CODES_FILE
    if convention in files:
        bad += _constant_problems(convention, EXIT_CODES_FILE)
        bad += _static_problems(convention, EXIT_CODES_FILE)
        bad += _shell_problems(scan_root, EXIT_CODES_FILE, depth)

    for path in programs:
        rel = str(path.relative_to(scan_root))
        bad += _static_problems(path, rel)
        card = by_check.get(rel)
        if card is None:
            bad.append(
                f"{rel} 沒有任何一張讀得出來的卡宣告它"
                "——沒有卡就沒有控制樣本，沒人證明過這支尺咬得到已知案例"
            )
            continue
        bad += _probe_problems(card, scan_root, rel, depth)
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="檢查程式的離開碼要誠實：0 乾淨／1 違規／2 工具自壞",
            targets=targets,
        )
    )
