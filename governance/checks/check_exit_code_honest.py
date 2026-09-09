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
repo 內的空目錄餵 ``enumerate_files``，它必須 raise ``ToolBroken``。回得出一個空 list
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
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken, run  # noqa: E402
from governance.loader import CHECKS_DIR, RULES_DIR, Card, card_problems, load_card  # noqa: E402

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
    return os.pathsep.join(kept) or os.path.join(os.sep, "nonexistent-aosr-probe", "no-such-bin-dir")


def _probe_env(depth: int, *, path: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env[DEPTH_ENV] = str(depth + 1)
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
) -> list[str]:
    """跑一次探針：驗離開碼，順便驗報告行。"""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", card.check_module, "--scan-root", str(target)],
            cwd=scan_root,
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
        print(f"NOTE: 動態探針在深度 {depth} 停止遞迴，{rel} 這一層只跑靜態掃描", file=sys.stderr)
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
        print(f"NOTE: {rel} 的卡宣告 external_tools = []，沒有工具可抽，跳過那個探針", file=sys.stderr)

    empty = Path(tempfile.mkdtemp(prefix="aosr-empty-probe-", dir=scan_root))
    try:
        bad += _probe("掃描集合是空的", card, scan_root, empty, _probe_env(depth), TOOL_BROKEN, rel)
    finally:
        # 只刪得掉空目錄。刪不掉就是有人往探針目錄裡寫東西，讓它炸出來，不要吞。
        empty.rmdir()

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
    empty = Path(tempfile.mkdtemp(prefix="aosr-shell-probe-", dir=scan_root))
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
        # 只刪得掉空目錄。刪不掉就是有人往探針目錄裡寫東西，讓它炸出來，不要吞。
        empty.rmdir()

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


def check(scan_root: Path, files: list[Path]) -> list[str]:
    depth = int(os.environ.get(DEPTH_ENV, "0"))
    checks_dir = scan_root / CHECKS_DIR
    targets = sorted(
        f for f in files if f.parent == checks_dir and f.suffix == ".py" and not f.name.startswith("__")
    )
    if not targets:
        raise ToolBroken(
            f"{scan_root}/{CHECKS_DIR} 底下一支版控裡的檢查程式都沒有"
            "——這一跑沒掃到東西，「沒問題」這句話不算數"
        )

    by_check: dict[str, Card] = {}
    for path in sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml"):
        if card_problems(path, scan_root):
            # 卡自己壞掉是第一張卡（rule-card-required-fields）的事，這裡只是讀不出控制樣本。
            continue
        card = load_card(path, scan_root)
        by_check.setdefault(card.check, card)

    bad: list[str] = []
    convention = scan_root / EXIT_CODES_FILE
    if convention in files:
        bad += _constant_problems(convention, EXIT_CODES_FILE)
        bad += _static_problems(convention, EXIT_CODES_FILE)
        bad += _shell_problems(scan_root, EXIT_CODES_FILE, depth)

    for path in targets:
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
    sys.exit(run(check, description="檢查程式的離開碼要誠實：0 乾淨／1 違規／2 工具自壞"))
