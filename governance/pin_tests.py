"""推之前跑「釘住數值的考卷」並留收據；``gate`` 判一行指令裡的 git push 此刻能不能推。

**為什麼要有它。** 2026-11-28 一天裡三輪雲端紅都是同一種：改了程式，考卷裡釘住的雜湊（分區預覽、物理身分、
模態身分、模態程式摘要）沒跟著改完，推上去等 20～30 分鐘才知道。這些值本機就算得出來。

**清單不手寫。** 釘值考卷＝``tests/`` 底下的 ``test_*.py``，自己含 64 位十六進位字串，或匯入了含這種字串的
共用答案檔（``_*.py``）。加一支釘值考卷、拿掉一支，清單自動跟著變。瀏覽器考卷不進清單（理由見 ``BROWSER_SUFFIX``）。

**兩個用法。**

* ``uv run python -m governance.pin_tests``：跑清單上的考卷，把「哪一顆提交、樹乾不乾淨、離開碼」寫進
  ``governance/receipts/local`` 底下的 ``pins.json``（被忽略的目錄）。``--list`` 只印清單不跑，``--jobs N`` 調考卷工人數。
* ``uv run python -m governance.pin_tests gate --command '<整行指令>'``：指令裡沒有 git push 回 0；有的話，
  那棵樹相對 ``origin/main`` 動過 ``GATED`` 列的路徑，就要有一張指到此刻 HEAD、乾淨、離開碼 0 的收據，否則回 1
  並說要跑哪一行；``--cwd`` 給那一行指令執行時所在的目錄。施工機器上的 push 掛鉤（推之前自動執行的小程式）
  叫的就是這個，而且拿的是主線上的這一支，不是要推的那條分支自己帶的。gate 只用標準程式庫與 git。

**這一支不是規矩卡**：本機跑出來的只是宣稱，雲端 ``verify`` 才算數；它只讓「本機就紅得出來的」不必等雲端。
離開碼照 :mod:`governance.exit_codes`：0 可以、1 不行、2 這一跑沒量到東西。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken, note, repo_root

RECEIPT_DIR = "governance/receipts/local"
RECEIPT_NAME = "pins.json"
SELF = "governance/pin_tests.py"
TESTS_DIR = "tests"
BASE_REF = "origin/main"
# SHA-256 的十六進位長度；前後不准再接十六進位字元（更長的字串不算）。
PIN = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
# 瀏覽器考卷不進清單：它們受機器負載影響、偶發紅（#1231），擋 push 的東西不能偶發紅；雲端照跑。
BROWSER_SUFFIX = "_browser.py"
# 相對主線動到這些路徑才要收據：程式、考卷、答案檔，以及會改執行環境身分的依賴鎖檔。
GATED = ("src/", "tests/", "blueprint/", "pyproject.toml", "uv.lock")
# 同一行指令裡先動提交再推：掛鉤在整行執行前就判，看不到之後才出現的那顆提交，所以整行擋下。
REWRITE_VERBS = frozenset({"commit", "merge", "rebase", "cherry-pick", "pull", "am", "reset", "revert",
                           "checkout", "switch"})
# 不會把提交送上去的 push：刪遠端分支、演練、只印說明；短旗標合寫（-nv）也算。
HARMLESS_PUSH_FLAGS = frozenset({"--delete", "--dry-run", "--help"})
HARMLESS_SHORT_FLAGS = re.compile(r"-[A-Za-z]*[ndh][A-Za-z]*")
# 後面跟著一個值的 push 旗標：那個值不是遠端也不是來源。
PUSH_FLAGS_WITH_VALUE = frozenset({"-o", "--push-option", "--repo", "--receive-pack", "--exec"})
# 一次推很多條、看不出是哪幾顆的寫法：不猜，擋下請人寫明來源。
UNREADABLE_PUSH_FLAGS = frozenset({"--all", "--mirror", "--branches"})
OPERATOR_CHARS = frozenset(";&|()")
# 包在指令前面、不改變「後面那個才是真的指令」的字（含殼層的 then／do／else 這類關鍵字）；以及用 -c 帶一整段指令的殼層。
WRAPPERS = frozenset({"command", "env", "time", "nice", "exec", "sudo", "noglob",
                      "if", "then", "elif", "else", "while", "until", "do", "!", "{"})
SHELLS = frozenset({"bash", "sh", "zsh"})
RUN_LINE = "uv run python -m governance.pin_tests"

type Git = Callable[[Path, Sequence[str]], str]


@dataclass(frozen=True)
class GitCall:
    """一段指令裡的一次 git 呼叫：在哪個目錄、哪個子命令、後面的引數。"""

    directory: str
    verb: str
    args: tuple[str, ...]


def _git(root: Path, args: Sequence[str]) -> str:
    done = subprocess.run(("git", "-C", str(root), *args), capture_output=True, text=True)
    if done.returncode != 0:
        raise ToolBroken(f"git {' '.join(args)} 回 {done.returncode}：{done.stderr.strip()}")
    return done.stdout.strip()


def _parsed(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as exc:
        raise ToolBroken(f"剖不開 {path}：{exc}") from exc


def has_pin(tree: ast.Module) -> bool:
    """程式結構裡有沒有 64 位十六進位的字串常數（註解不算）。"""
    return any(isinstance(node, ast.Constant) and isinstance(node.value, str) and PIN.search(node.value)
               for node in ast.walk(tree))


def imported(tree: ast.Module, module: str = "") -> set[str]:
    """這支檔匯入的模組點記法；``from a.b import c`` 兩種都記（``a.b`` 與 ``a.b.c``）。

    ``module`` 是這支檔自己的點記法，用來把相對匯入（``from ._answers import X``）換成完整名字。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            package = module.split(".")[:-node.level] if node.level else []
            base = ".".join([*package, *([node.module] if node.module else [])])
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _dotted(name: str) -> str:
    return name.removesuffix(".py").replace("/", ".")


def pin_files(root: Path) -> list[str]:
    """釘值考卷清單（相對 ``root``、排序過）。一支都沒有就是這一跑沒量到東西。"""
    trees = {path.relative_to(root).as_posix(): _parsed(path)
             for path in sorted((root / TESTS_DIR).rglob("*.py"))}
    answers = {_dotted(name) for name, tree in trees.items() if Path(name).name.startswith("_") and has_pin(tree)}
    chosen = [name for name, tree in trees.items()
              if Path(name).name.startswith("test_") and not name.endswith(BROWSER_SUFFIX)
              and (has_pin(tree) or imported(tree, _dotted(name)) & answers)]
    if not chosen:
        raise ToolBroken(f"{TESTS_DIR}/ 底下一支釘值考卷都沒找到，沒有東西可跑")
    return chosen


def tree_state(root: Path, git: Git = _git) -> tuple[str, bool]:
    """此刻的 HEAD，以及 ``GATED`` 那些路徑有沒有還沒提交的改動（含沒被忽略的新檔）。"""
    head = git(root, ("rev-parse", "HEAD"))
    dirty = bool(git(root, ("status", "--porcelain", "--untracked-files=normal", "--", *GATED)))
    return head, dirty


def run_pytest(root: Path, files: Sequence[str], jobs: int | None) -> int:
    """跑考卷，輸出直接給叫的人看；回 pytest 的離開碼。

    環境裡的 ``PYTEST_ADDOPTS`` 不帶進去：它可以讓 pytest 只清點不執行（離開碼照樣 0），那種收據不算驗過。"""
    workers = () if jobs is None else ("-n", str(jobs))
    argv = (sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *workers, *files)
    return subprocess.run(argv, cwd=root, env=pytest_env(os.environ)).returncode


def pytest_env(environ: Mapping[str, str]) -> dict[str, str]:
    return {**{key: value for key, value in environ.items() if key != "PYTEST_ADDOPTS"}, "PYTHONDONTWRITEBYTECODE": "1"}


def receipt_path(root: Path) -> Path:
    return root / RECEIPT_DIR / RECEIPT_NAME


def write_receipt(root: Path, *, head: str, dirty: bool, code: int, files: Sequence[str], seconds: float) -> None:
    path = receipt_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema": 1, "head": head, "dirty": dirty, "exit": code, "files": list(files), "seconds": round(seconds, 1)}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(root: Path, *, jobs: int | None, git: Git = _git,
        pytest_call: Callable[[Path, Sequence[str], int | None], int] = run_pytest) -> int:
    """跑清單並留收據。樹不乾淨照跑、照留，但收據記不乾淨，``gate`` 不收。"""
    head, dirty = tree_state(root, git)  # 先記提交再列清單：列的途中提交變了，收尾那一次比對看得到。
    files = pin_files(root)
    start = time.perf_counter()
    code = pytest_call(root, files, jobs)
    after_head, after_dirty = tree_state(root, git)
    seconds = time.perf_counter() - start
    moved = after_head != head
    write_receipt(root, head=head, dirty=dirty or after_dirty or moved, code=code, files=files, seconds=seconds)
    note(f"釘值考卷 {len(files)} 支檔，{seconds:.1f} 秒，pytest 離開碼 {code}；收據 {receipt_path(root)} 指到 {head}")
    if dirty or after_dirty or moved:
        note(f"{'、'.join(GATED)} 有還沒提交的改動，或跑的途中提交變了：這張收據不算，提交之後再跑一次")
    return CLEAN if code == 0 else VIOLATION


def needs_receipt(root: Path, rev: str = "HEAD", git: Git = _git) -> bool:
    """``rev`` 相對主線有沒有動到 ``GATED`` 的路徑。"""
    base = git(root, ("merge-base", rev, BASE_REF))
    changed = git(root, ("diff", "--name-only", base, rev)).splitlines()
    return any(name == prefix or (prefix.endswith("/") and name.startswith(prefix))
               for name in changed for prefix in GATED)


def receipt_problem(root: Path, head: str) -> str:
    """收據哪裡不對；沒問題回空字串。"""
    path = receipt_path(root)
    if not path.is_file():
        return "沒有釘值考卷的收據"
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        seen, dirty, code = body["head"], body["dirty"], body["exit"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"收據讀不懂（{type(exc).__name__}）"
    if seen != head:
        return f"收據是 {str(seen)[:12]} 那顆的，現在要推的是 {head[:12]}"
    if dirty is not False:
        return "收據是在還有沒提交的改動時跑的"
    if type(code) is not int or code != 0:
        return f"釘值考卷上次沒過（pytest 離開碼 {code!r}）"
    return ""


def check(root: Path, rev: str = "HEAD", git: Git = _git) -> str:
    """把 ``rev`` 那顆推出去之前還缺什麼；可以推回空字串。"""
    if not needs_receipt(root, rev, git):
        return ""
    problem = receipt_problem(root, git(root, ("rev-parse", f"{rev}^{{commit}}")))
    return problem and f"{root}：{problem}。先在這棵樹跑 `{RUN_LINE}`（過了再推）"


def _segments(command: str) -> Iterator[tuple[list[str], str]]:
    """把一行殼層指令切成（一段的字、結束它的運算子）；引號裡的字不切，引號沒收好就退回用空白與運算子粗切。"""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        tokens = re.sub(r"(&&|\|\||[;|&()\n])", r" \1 ", command).split()
    segment: list[str] = []
    for token in tokens:
        if set(token) <= OPERATOR_CHARS:
            yield segment, token
            segment = []
        else:
            segment.append(token)
    yield segment, ""


def _unwrapped(segment: Sequence[str]) -> list[str]:
    """拿掉開頭的環境變數設定與包裝字（``env``、``command``…），剩下真的指令。"""
    words = list(segment)
    while words and (words[0] in WRAPPERS or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0])):
        words.pop(0)
    return words


def _inner_script(segment: Sequence[str]) -> str:
    """``bash -c '<一整段>'`` 這種寫法裡的那一整段；不是這種寫法回空字串。"""
    words = _unwrapped(segment)
    if not words or Path(words[0]).name not in SHELLS:
        return ""
    for index, word in enumerate(words[1:-1], start=1):
        if re.fullmatch(r"-[A-Za-z]*c", word):
            return words[index + 1]
    return ""


def _git_call(segment: Sequence[str], directory: str) -> GitCall | None:
    """這一段是不是 git 呼叫；是的話取出 ``-C`` 的目錄、子命令與引數。前面的環境變數設定略過。"""
    words = _unwrapped(segment)
    if not words or Path(words[0]).name != "git":
        return None
    index = 1
    while index < len(words) and words[index].startswith("-"):
        if words[index] in ("-C", "-c") and index + 1 < len(words):
            if words[index] == "-C":
                directory = str(Path(directory) / words[index + 1])
            index += 1
        index += 1
    if index >= len(words):
        return None
    return GitCall(directory, words[index], tuple(words[index + 1:]))


def git_calls(command: str, cwd: str) -> list[GitCall]:
    """整行指令裡的每一次 git 呼叫，照出現順序；``cd <目錄>`` 之後的呼叫算在那個目錄，出了括號（子殼層）就還原。"""
    calls: list[GitCall] = []
    directory = cwd
    outer: list[str] = []
    for segment, operator in _segments(command.replace("\n", " ; ")):
        if len(segment) >= 2 and segment[0] == "cd":
            directory = str(Path(directory) / Path(segment[1]).expanduser())
        inner = _inner_script(segment)
        if inner:
            calls.extend(git_calls(inner, directory))
        call = _git_call(segment, directory)
        if call is not None:
            calls.append(call)
        for char in operator:
            if char == "(":
                outer.append(directory)
            elif char == ")" and outer:
                directory = outer.pop()
    return calls


def push_sources(call: GitCall) -> list[str] | None:
    """這次 push 要送出的來源。空清單＝不會送出提交（演練、說明、全是刪除）；``None``＝看不出是哪幾顆。

    沒寫來源（``git push``、``git push <遠端>``）當成 HEAD。遠端設定裡另外指定來源的寫法這裡不讀。"""
    args = tuple(takewhile(lambda arg: not arg.startswith("#"), call.args))  # 行尾註解不是引數。
    if HARMLESS_PUSH_FLAGS & set(args) or any(HARMLESS_SHORT_FLAGS.fullmatch(arg) for arg in args):
        return []
    if UNREADABLE_PUSH_FLAGS & set(args):
        return None
    plain: list[str] = []
    skip = False
    for arg in args:
        if skip or arg.startswith("-"):
            skip = arg in PUSH_FLAGS_WITH_VALUE
            continue
        plain.append(arg)
    refspecs = [arg.lstrip("+") for arg in plain[1:]]
    if not refspecs:
        return ["HEAD"]
    sources = [refspec.split(":")[0] for refspec in refspecs if not refspec.startswith(":")]
    return None if any("*" in source for source in sources) else sources


def gate(command: str, cwd: str, git: Git = _git) -> list[str]:
    """整行指令不准執行的理由（空的就是可以）。"""
    calls = git_calls(command, cwd)
    problems: list[str] = []
    for index, call in enumerate(calls):
        sources = push_sources(call) if call.verb == "push" else []
        if sources == []:
            continue
        if any(earlier.verb in REWRITE_VERBS for earlier in calls[:index]):
            problems.append("git push 要自己一行下：同一行裡先提交（或合併、切換）再推，推之前看不到那顆新提交。"
                            f"先提交，跑 `{RUN_LINE}`，再單獨 git push")
        else:
            problems.extend(_push_problems(call.directory, sources, git))
    return list(dict.fromkeys(problems))


def _push_problems(directory: str, sources: Sequence[str] | None, git: Git) -> list[str]:
    try:
        root = Path(git(Path(directory), ("rev-parse", "--show-toplevel")))
    except ToolBroken:
        return []  # 不是 git 工作樹：讓 git 自己報錯，這裡不擋。
    if not (root / SELF).is_file():
        return []  # 這棵樹的分支比這支工具還舊，跑不了那一行：不擋，跟上主線之後才開始管。
    if sources is None:
        return [f"{root}：看不出這一行要推哪幾顆提交（一次推全部，或來源帶萬用字元）。"
                "改成寫明來源，例如 git push <遠端> HEAD:<分支>"]
    problems: list[str] = []
    for rev in dict.fromkeys(sources):
        try:
            git(root, ("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"))
        except ToolBroken:
            problems.append(f"{root}：解析不到要推的來源 {rev}。寫成 HEAD 或這棵樹認得的分支名")
            continue
        problems.extend(filter(None, [check(root, rev, git)]))
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m governance.pin_tests", description="推之前跑釘住數值的考卷")
    parser.add_argument("mode", nargs="?", choices=("run", "gate"), default="run")
    parser.add_argument("--jobs", type=int, default=None, help="考卷工人數（不給就照 pyproject 的設定）")
    parser.add_argument("--list", action="store_true", help="只印清單，不跑")
    parser.add_argument("--command", default="", help="gate：要判的那一整行殼層指令")
    parser.add_argument("--cwd", default="", help="gate：那一行指令會在哪個目錄執行（不給就是現在這個目錄）")
    args = parser.parse_args(argv)
    try:
        if args.mode == "gate":
            problems = gate(args.command, args.cwd or str(Path.cwd()))
            for problem in problems:
                note(problem)
            return VIOLATION if problems else CLEAN
        root = repo_root()
        if args.list:
            for name in pin_files(root):
                note(name)
            return CLEAN
        return run(root, jobs=args.jobs)
    except ToolBroken as exc:
        note(str(exc))
        return TOOL_BROKEN


if __name__ == "__main__":
    sys.exit(main())
