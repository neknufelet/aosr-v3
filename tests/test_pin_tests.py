"""推之前跑釘值考卷：清單從考卷裡找、收據指到提交、push 那一行缺收據就擋。"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from governance import pin_tests as subject
from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken

REPO = Path(__file__).resolve().parents[1]
PIN = "ab" * 32
HEAD = "1" * 40
# 要判的那幾行指令。這份考卷不執行版控工具：git 的回答全由 FakeGit 給。
PUSH = "git push"
PUSH_NAMED = "git push origin HEAD:feat"
PUSH_TYPO = "git push origin typo"
COMMIT_THEN_PUSH = "git commit -am x && git push"
# 真樹裡各類路徑各取一個當「相對主線動到的檔」。
SOURCE_FILE = "src/aosr/__init__.py"
UNGATED_FILES = ("docs/decisions/five-seats.md", "governance/rules/style-guard.toml")
GATED_FILES = (SOURCE_FILE, "tests/engine/test_zone_range.py", "blueprint/precision_contracts.toml", "uv.lock",
               "pyproject.toml")


@dataclass
class FakeGit:
    """照子命令回固定答案的 git；``top`` 是 None 代表那個目錄不是工作樹。"""

    top: Path | None
    head: str = HEAD
    changed: tuple[str, ...] = (SOURCE_FILE,)
    status: str = ""
    heads: list[str] = field(default_factory=list)

    def __call__(self, root: Path, args: Sequence[str]) -> str:
        if args[0] == "rev-parse" and args[1] == "--show-toplevel":
            if self.top is None:
                raise ToolBroken("不是工作樹")
            return str(self.top)
        if args[0] == "rev-parse":
            return self.heads.pop(0) if self.heads else self.head
        return {"merge-base": "base", "diff": "\n".join(self.changed), "status": self.status}[args[0]]


def tree(tmp_path: Path) -> Path:
    """一棵有這支工具的假樹（gate 只管有這支工具的樹）。"""
    (tmp_path / "governance").mkdir()
    (tmp_path / subject.SELF).write_text("", encoding="utf-8")
    return tmp_path


def receipt(root: Path, *, head: str = HEAD, dirty: object = False, code: object = 0) -> None:
    path = subject.receipt_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"head": head, "dirty": dirty, "exit": code}), encoding="utf-8")


def test_real_tree_list_holds_the_pins_that_went_red_and_no_browser_or_helper_files() -> None:
    """抓清單漏掉 2026-11-28 紅過的那幾支：分區預覽、物理身分、模態摘要（經共用答案檔）、家具考卷的模態身分。"""
    files = subject.pin_files(REPO)
    assert {"tests/engine/test_zone_range.py", "tests/engine/test_physics_identity_pinned.py",
            "tests/engine/test_modal_lookup.py", "tests/engine/test_furniture_layout.py"} <= set(files)
    assert files == sorted(files)
    assert all(Path(name).name.startswith("test_") and not name.endswith("_browser.py") for name in files)


def test_list_is_found_from_literals_and_answer_file_imports(tmp_path: Path) -> None:
    """抓手寫清單會漂：自己含 64 位雜湊的、匯入共用答案檔的才進；註解、太長的字串、瀏覽器考卷、慢卷不進。"""
    folder = tmp_path / "tests" / "engine"
    folder.mkdir(parents=True)
    sources = {
        "_answers.py": f'DIGEST = "{PIN}"\n',
        "_plain.py": "VALUE = 1\n",
        "test_literal.py": f'def test_a() -> None:\n    assert "modal-v1:{PIN}"\n',
        "test_from_import.py": "from tests.engine._answers import DIGEST\n",
        "test_module_import.py": "from tests.engine import _answers\n",
        "test_plain_import.py": "from tests.engine._plain import VALUE\n",
        "test_comment_only.py": f"# {PIN}\nVALUE = 1\n",
        "test_too_long.py": f'VALUE = "{PIN}ab"\n',
        "test_page_browser.py": f'VALUE = "{PIN}"\n',
        "slow_search.py": f'VALUE = "{PIN}"\n',
    }
    for name, text in sources.items():
        (folder / name).write_text(text, encoding="utf-8")
    assert subject.pin_files(tmp_path) == [f"tests/engine/test_{name}.py"
                                           for name in ("from_import", "literal", "module_import")]


def test_no_pin_file_is_tool_broken(tmp_path: Path) -> None:
    """抓清單空了還回綠：一支都沒找到是這一跑沒量到東西。"""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_plain.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ToolBroken, match="一支釘值考卷都沒找到"):
        subject.pin_files(tmp_path)


def test_unparsable_test_file_is_tool_broken(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_broken.py").write_text("def (:\n", encoding="utf-8")
    with pytest.raises(ToolBroken, match="剖不開"):
        subject.pin_files(tmp_path)


@pytest.mark.parametrize(("command", "expected"), [
    ("git push", [(".", "push", ())]),
    ("git -C ../other -c a=b push origin HEAD:x", [("../other", "push", ("origin", "HEAD:x"))]),
    ("cd tree && FOO=1 git push -u origin feat", [("tree", "push", ("-u", "origin", "feat"))]),
    ("git add -A; git commit -m 'git push later' && git push", [(".", "add", ("-A",)), (".", "commit", ("-m", "git push later")),
                                                                (".", "push", ())]),
    ("git status\ngit push", [(".", "status", ()), (".", "push", ())]),
    ("bash -lc 'cd w && git push'", [("w", "push", ())]),
    ("env X=1 command git push # 註解", [(".", "push", ("#", "註解"))]),
    ("echo git push", []),
    ("git commit -F - <<EOF\nit's about git push\nEOF\ngit push", [(".", "commit", ("-F", "-", "<<EOF")), (".", "push", ())]),
])
def test_git_calls_are_read_from_the_whole_line(command: str, expected: list[tuple[str, str, tuple[str, ...]]]) -> None:
    """抓只看行首：cd、-C、環境變數、包裝字、殼層 -c、換行、引號裡的字都要讀對；引號沒收好（內嵌文件裡的撇號）退回逐行粗切。"""
    found = [(str(Path(call.directory)), call.verb, call.args) for call in subject.git_calls(command, ".")]
    assert found == expected


@pytest.mark.parametrize("command", [
    "git status", "git push origin --delete old", "git push --dry-run", "git push origin :old", "git log --grep push",
])
def test_lines_that_send_no_commits_pass(tmp_path: Path, command: str) -> None:
    """抓擋到不相干的指令：沒有 push、刪遠端分支、演練都不必有收據。"""
    assert subject.gate(command, str(tree(tmp_path)), FakeGit(tmp_path)) == []


def test_push_without_receipt_is_blocked_and_says_what_to_run(tmp_path: Path) -> None:
    """抓沒跑就推：動過程式、沒有收據，擋下並說要跑哪一行。"""
    problems = subject.gate(PUSH_NAMED, str(tree(tmp_path)), FakeGit(tmp_path))
    assert len(problems) == 1
    assert "沒有釘值考卷的收據" in problems[0]
    assert subject.RUN_LINE in problems[0]


def test_push_with_matching_receipt_passes(tmp_path: Path) -> None:
    receipt(tree(tmp_path))
    assert subject.gate(PUSH, str(tmp_path), FakeGit(tmp_path)) == []


@pytest.mark.parametrize(("stored", "word"), [
    ({"head": "2" * 40}, "現在要推的是"),
    ({"dirty": True}, "沒提交的改動"),
    ({"dirty": None}, "沒提交的改動"),
    ({"code": 1}, "沒過"),
    ({"code": None}, "沒過"),
])
def test_stale_dirty_or_red_receipt_is_blocked(tmp_path: Path, stored: dict[str, object], word: str) -> None:
    """抓拿舊收據過關：收據不是這顆提交的、跑的時候樹不乾淨、考卷沒過，都擋。"""
    root = tree(tmp_path)
    path = subject.receipt_path(root)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"head": HEAD, "dirty": False, "exit": 0} | {
        {"code": "exit"}.get(key, key): value for key, value in stored.items()}), encoding="utf-8")
    assert word in subject.gate(PUSH, str(root), FakeGit(root))[0]


def test_unreadable_receipt_is_blocked(tmp_path: Path) -> None:
    root = tree(tmp_path)
    subject.receipt_path(root).parent.mkdir(parents=True)
    subject.receipt_path(root).write_text("{", encoding="utf-8")
    assert "讀不懂" in subject.gate(PUSH, str(root), FakeGit(root))[0]


def test_push_that_touches_no_gated_path_needs_no_receipt(tmp_path: Path) -> None:
    """抓只改文件也要等四分鐘：相對主線沒動到程式、考卷、答案檔、依賴鎖檔，就不必有收據。"""
    git = FakeGit(tree(tmp_path), changed=UNGATED_FILES)
    assert subject.gate(PUSH, str(tmp_path), git) == []
    for name in GATED_FILES:
        assert subject.gate(PUSH, str(tmp_path), FakeGit(tmp_path, changed=(name,))) != []


def test_commit_and_push_on_one_line_is_blocked_even_with_a_receipt(tmp_path: Path) -> None:
    """抓同一行先提交再推：判的時候新提交還不存在，舊收據會放行沒驗過的那一顆，所以整行擋。"""
    receipt(tree(tmp_path))
    problems = subject.gate(COMMIT_THEN_PUSH, str(tmp_path), FakeGit(tmp_path))
    assert len(problems) == 1
    assert "要自己一行下" in problems[0]


def test_trees_without_this_tool_and_non_trees_are_left_alone(tmp_path: Path) -> None:
    """抓舊分支的樹全部推不出去：樹裡還沒有這支工具（合併前開的分支）或根本不是工作樹，不擋。"""
    assert subject.gate(PUSH, str(tmp_path), FakeGit(tmp_path)) == []
    assert subject.gate(PUSH, str(tmp_path), FakeGit(None)) == []


def test_source_this_tree_does_not_know_is_left_to_git(tmp_path: Path) -> None:
    """抓來源打錯字時回一串看不懂的錯：那一顆解析不到就不擋，讓 git 自己說 src refspec 對不上。"""
    class Unknown(FakeGit):
        def __call__(self, root: Path, args: Sequence[str]) -> str:
            if "--verify" in args:
                raise ToolBroken("解析不到")
            return super().__call__(root, args)

    assert subject.gate(PUSH_TYPO, str(tree(tmp_path)), Unknown(tmp_path)) == []


def test_named_source_ref_is_the_commit_that_is_checked() -> None:
    """抓推別條分支卻拿 HEAD 的收據過關：只有一個來源時查那一顆，其餘當 HEAD。"""
    def rev(*args: str) -> str:
        return subject.pushed_rev(subject.GitCall(".", "push", args))

    assert rev() == rev("origin") == rev("-u", "origin", "a", "b") == "HEAD"
    assert rev("origin", "feat") == rev("-u", "origin", "+feat:claude/x") == "feat"
    assert rev("origin", "HEAD:claude/x") == "HEAD"


def run_with(tmp_path: Path, git: FakeGit, code: int) -> tuple[int, dict[str, object], list[str]]:
    folder = tmp_path / "tests"
    folder.mkdir()
    (folder / "test_pin_tests.py").write_text(f'VALUE = "{PIN}"\n', encoding="utf-8")
    seen: list[str] = []

    def fake_pytest(root: Path, files: Sequence[str], jobs: int | None) -> int:
        seen.extend(files)
        return code

    result = subject.run(tmp_path, jobs=None, git=git, pytest_call=fake_pytest)
    return result, json.loads(subject.receipt_path(tmp_path).read_text(encoding="utf-8")), seen


def test_run_writes_a_receipt_that_the_gate_accepts(tmp_path: Path) -> None:
    """抓收據與門口對不上：乾淨樹跑過，收據指到那顆提交，同一棵樹接著推就放行。"""
    git = FakeGit(tree(tmp_path))
    code, body, seen = run_with(tmp_path, git, 0)
    assert code == CLEAN
    assert seen == ["tests/test_pin_tests.py"]
    assert (body["head"], body["dirty"], body["exit"], body["files"]) == (HEAD, False, 0, seen)
    assert subject.gate(PUSH, str(tmp_path), git) == []


def test_red_run_returns_violation_and_gate_blocks(tmp_path: Path) -> None:
    git = FakeGit(tree(tmp_path))
    code, body, _seen = run_with(tmp_path, git, 1)
    assert (code, body["exit"]) == (VIOLATION, 1)
    assert "沒過" in subject.gate(PUSH, str(tmp_path), git)[0]


@pytest.mark.parametrize("git", [
    FakeGit(None, status=f" M {SOURCE_FILE}"),
    FakeGit(None, heads=[HEAD, "3" * 40]),
])
def test_run_on_dirty_or_moving_tree_leaves_a_receipt_that_does_not_count(tmp_path: Path, git: FakeGit) -> None:
    """抓改到一半跑的收據拿去推：樹不乾淨、或跑的途中提交變了，收據記成不乾淨。"""
    _code, body, _seen = run_with(tmp_path, git, 0)
    assert body["dirty"] is True


def test_command_line_gate_and_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """抓命令列接錯：gate 把指令與目錄原樣交下去、有理由回 1；清單空的回 2 不回 0。"""
    monkeypatch.chdir(tmp_path)
    assert subject.main(["gate", "--command", "ls -la"]) == CLEAN
    asked: list[tuple[str, str]] = []

    def fake_gate(command: str, cwd: str) -> list[str]:
        asked.append((command, cwd))
        return ["擋下的理由"]

    monkeypatch.setattr(subject, "gate", fake_gate)
    assert subject.main(["gate", "--command", PUSH, "--cwd", "elsewhere"]) == VIOLATION
    assert subject.main(["gate", "--command", PUSH]) == VIOLATION
    assert asked == [(PUSH, "elsewhere"), (PUSH, str(tmp_path))]
    assert "擋下的理由" in capsys.readouterr().err
    monkeypatch.setattr(subject, "repo_root", lambda: tmp_path)
    assert subject.main(["--list"]) == TOOL_BROKEN
    assert "一支釘值考卷都沒找到" in capsys.readouterr().err
