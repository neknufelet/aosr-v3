"""規矩卡 tests-isolated-from-real-env 的聚焦反例：參數陣列第一格用 ``shutil.which`` ／
``os.environ.get`` 找出版控工具，也要咬。

原始偵測器只認「參數陣列第一格是字面 ``"git"``」或「先指給名字的字面陣列」；第一格改成
``shutil.which("git")``、``shutil.which(GIT_TOOL)``、``os.environ.get("GIT", "git")``，
又或者把這些呼叫先指給名字再餵下去，``_names_git`` 都回 False——同一種「沒經過 sandbox
fixture 就 spawn 版控工具」，換個寫法就繞過去了。

這些測試餵的是**給 AST 掃描器的字串**，不是真的執行：不 import 不跑任何子程序，
``shutil.which(...)`` ／ ``os.environ.get(...)`` 那幾行是輸入的一部分，從來沒有真的
去找 git、也沒有人真的去 spawn。受測的是 ``checker.check(tmp_path, files)`` 對一棵
只帶最小卡設定與一支產出來的測試源的暫存樹的判決。

判的是「spawn 違規有沒有被報出來」，不是「有沒有任何一筆違規」——後者會把
「同名 fixture 缺定義」那一類別的違規誤當成目標 RED，那正是這支檢查看不懂
which 與 environ.get 的證據之外的雜訊。所以這裡用 spawn 那一行專屬的幾個字當判據。

而且「spawn 那一筆」的判據必須是**外層真的呼叫了 subprocess**，不是「訊息裡有 spawn
字樣」：受測的舊偵測器把任何第一格是 git 的呼叫都當 spawn，``subprocess.run([shutil.which("git"),
"status"])`` 裡的內層 ``shutil.which("git")`` 因此各自也報一筆 spawn 字樣的違規。只看字樣
的話，那一筆冒充成目標 RED，4 支真反例加 4 筆內層查找一起紅成 8，看起來全中、其實漏抓。
所以 :func:`_spawn` 要求同一列同時帶 spawn 字樣與被呼叫的 ``subprocess.run(``。

反例、正向控制與解不開的名字一律餵**真的參數陣列**：``subprocess.run([EXPRESSION, "status"])``
（第一格才是要找的版控工具），只有「先指給名字的字面陣列」那支照舊寫成 ``subprocess.run(ARGV)``。

還有一組是同名重新指定（``tool = "git"`` 之後 ``tool = shutil.which(tool)``）：外層（spawn
第一格）與內層（which 的 cmd／environ.get 的 default）是兩個互相獨立的解析領域，內層
進來時帶全新的 visited，所以外層走過的名字不會讓內層誤判循環。真循環照舊要停，見
:func:`test_true_cycle_still_terminates_after_domain_split`。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance.checks import tests_isolated_from_real_env as checker

# 最小卡：照 governance/fixtures/tests-isolated-from-real-env/case-git-argv-behind-a-name
# 那份樣本卡的形狀抄來——只帶 id、check 與 [settings] 的 sandbox_fixture。門檻只從掃描根
# 自己的 governance/rules/ 讀，讀不到就回 2，所以這裡要自帶一份。
CARD = (
    'id = "tests-isolated-from-real-env"\n'
    f'check = "{checker.CHECK_REL}"\n'
    "\n"
    "[settings]\n"
    'sandbox_fixture = "git_sandbox"\n'
)

# spawn 違規那一行專屬的字樣。缺 fixture（sandbox_fixture 沒人定義、或要了不存在的）
# 是別的訊息，這張考卷用這幾個字把「真的要抓的那一筆」撈出來，不把雜訊算進 RED。
SPAWN_MARK = "就 spawn 版控工具"
# 受測的那一跑，`_look_at_call` 把**任何**第一格是 git 的呼叫都算成 spawn——連
# `shutil.which("git")` 這種「只是去找在哪」的內層呼叫也一樣，而那一筆的訊息裡
# 會出現 `shutil.which('git')` 這幾個字。真正的 spawn 那一筆，訊息裡一定有被呼叫的
# `subprocess.run(`。所以 oracle 要求「同一列同時出現兩者」：只有外層真的 spawn
# 才算數，內層的 which 查找不會被誤當成目標 RED（否則 8 支反例只會紅 4 支，
# 兩支 which 反例與兩支 env.get 反例靠內層查找就假綠了）。
SPAWN_CALL = "subprocess.run("


def _run(tmp_path: Path, source: str) -> list[str]:
    """在暫存樹上跑一次 check：一份最小卡＋一支產出來的測試源，回違規清單。"""
    card = tmp_path / "governance" / "rules" / "tests-isolated-from-real-env.toml"
    card.parent.mkdir(parents=True)
    card.write_text(CARD, encoding="utf-8")
    test = tmp_path / "tests" / "test_generated.py"
    test.parent.mkdir(parents=True)
    test.write_text(source, encoding="utf-8")
    return checker.check(tmp_path, [card, test])


def _spawn(report: list[str]) -> list[str]:
    """只撈「真的 spawn」那一筆：同一列要同時帶 spawn 字樣與被呼叫的 ``subprocess.run(``。

    只比對 SPAWN_MARK 不夠——受測的舊偵測器把任何第一格是 git 的呼叫都當 spawn，
    連 ``subprocess.run(shutil.which("git"))`` 裡那層 ``shutil.which("git")`` 都會各報一筆，
    只看字樣的話「which 內層查找」就會冒充成目標 RED。要求同一列出現
    ``subprocess.run(`` 之後，撈到的才是外層那次呼叫被 ast.unparse 印出來的那一筆。
    """
    return [row for row in report if SPAWN_MARK in row and SPAWN_CALL in row]


# ── 反例：第一格用 which 與 environ.get 找版控工具，不經 fixture，必須報 spawn ────
# 每份源都是一支不帶 sandbox fixture 的測試函式，直接 subprocess.run。
# 原始偵測器對這一欄回 False（Supervisor 已實跑證實），所以每一支都該是「漏抓」的 RED。

NEGATIVES = [
    "direct-which-literal",
    "which-module-level-git-tool",
    "local-name-bound-to-which",
    "name-alias-chain",
    "argv-list-bound-to-variable",
]


@pytest.mark.parametrize(
    "source",
    [
        (
            "import subprocess\n"
            "import shutil\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([shutil.which(\"git\"), \"status\"])\n"
        ),
        (
            "import subprocess\n"
            "import shutil\n"
            "\n"
            "GIT_TOOL = \"git\"\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([shutil.which(GIT_TOOL), \"status\"])\n"
        ),
        (
            "import subprocess\n"
            "import shutil\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    tool = shutil.which(\"git\")\n"
            "    subprocess.run([tool, \"status\"])\n"
        ),
        (
            "import subprocess\n"
            "import shutil\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    tool = shutil.which(\"git\")\n"
            "    alias = tool\n"
            "    subprocess.run([alias, \"status\"])\n"
        ),
        (
            "import subprocess\n"
            "import shutil\n"
            "\n"
            "ARGV = [shutil.which(\"git\"), \"status\", \"--porcelain\"]\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run(ARGV)\n"
        ),
    ],
    ids=NEGATIVES,
)
def test_which_spawn_is_a_violation(tmp_path: Path, source: str) -> None:
    """第一格是 which(...) 的版控工具，不經 fixture，要報 spawn 違規。"""
    report = _run(tmp_path, source)
    assert _spawn(report), f"漏抓 spawn 違規：{report}"


@pytest.mark.parametrize(
    "source",
    [
        (
            "import os\n"
            "import subprocess\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([os.environ.get(\"GIT\", \"git\"), \"status\"])\n"
        ),
        (
            "import os\n"
            "import subprocess\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    tool = os.environ.get(\"GIT\", \"git\")\n"
            "    subprocess.run([tool, \"status\"])\n"
        ),
        (
            "import os\n"
            "import subprocess\n"
            "\n"
            "DEFAULT_GIT = \"git\"\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([os.environ.get(\"GIT\", DEFAULT_GIT), \"status\"])\n"
        ),
    ],
    ids=[
        "direct-env-get-default-literal",
        "bound-env-get",
        "env-get-default-bound-name",
    ],
)
def test_env_get_spawn_is_a_violation(tmp_path: Path, source: str) -> None:
    """第一格是 environ.get(..., "git") 的版控工具，不經 fixture，要報 spawn 違規。"""
    report = _run(tmp_path, source)
    assert _spawn(report), f"漏抓 spawn 違規：{report}"


# ── 正向對照：不是 git、或經 fixture，都不報 spawn；舊的裸字面照舊咬 ──────────

def test_which_python_is_not_a_violation(tmp_path: Path) -> None:
    """which('python') 不是版控工具，不報。"""
    source = (
        "import subprocess\n"
        "import shutil\n"
        "\n"
        "\n"
        "def test_python():\n"
        "    subprocess.run([shutil.which(\"python\"), \"--version\"])\n"
    )
    assert _run(tmp_path, source) == []


def test_env_get_python_is_not_a_violation(tmp_path: Path) -> None:
    """environ.get('PYTHON', 'python') 不是版控工具，不報。"""
    source = (
        "import os\n"
        "import subprocess\n"
        "\n"
        "\n"
        "def test_python():\n"
        "    subprocess.run([os.environ.get(\"PYTHON\", \"python\"), \"--version\"])\n"
    )
    assert _run(tmp_path, source) == []


@pytest.mark.parametrize(
    "call",
    [
        "subprocess.run([shutil.which(\"git\"), \"status\"])",
        "subprocess.run([os.environ.get(\"GIT\", \"git\"), \"status\"])",
    ],
    ids=["guarded-which", "guarded-env-get"],
)
def test_the_same_spawn_under_the_fixture_is_clean(tmp_path: Path, call: str) -> None:
    """同一句 spawn，只要經唯一那支 git_sandbox fixture，就該回乾淨。"""
    implib = "shutil" if "which" in call else "os"
    source = (
        f"import {implib}\n"
        "import subprocess\n"
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.fixture\n"
        "def git_sandbox(tmp_path):\n"
        "    yield tmp_path\n"
        "\n"
        "\n"
        "def test_git(git_sandbox):\n"
        f"    {call}\n"
    )
    assert _run(tmp_path, source) == []


def test_plain_literal_git_is_still_a_violation(tmp_path: Path) -> None:
    """原本就咬得到的裸 ``[\"git\", ...]`` 字面，補新判準之後不准漏掉。"""
    source = (
        "import subprocess\n"
        "\n"
        "\n"
        "def test_git():\n"
        "    subprocess.run([\"git\", \"status\", \"--porcelain\"])\n"
    )
    report = _run(tmp_path, source)
    assert _spawn(report), f"漏抓既有的裸字面 spawn：{report}"


# ── 不能收斂的名字要終止，不准遞迴爆棧、也不准硬猜成 git ──────────────────────
# 解析器只做「同一作用域裡普通的名字指派」，不做整支 Python 的執行推演；
# 遇上一圈互相指的名字、或指不到任何值的名字，要停下來當「不知道」，而不是
# 沿著圈一直追到 RecursionError，也不是把解不開的名字硬當成版控工具。

@pytest.mark.parametrize(
    "source",
    [
        (
            "import shutil\n"
            "import subprocess\n"
            "\n"
            "tool = alias\n"
            "alias = tool\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([shutil.which(tool), \"status\"])\n"
        ),
        (
            "import shutil\n"
            "import subprocess\n"
            "\n"
            "TOOL = TOOL\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([shutil.which(TOOL), \"status\"])\n"
        ),
        (
            "import shutil\n"
            "import subprocess\n"
            "\n"
            "TOOL = somewhere_undefined\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([shutil.which(TOOL), \"status\"])\n"
        ),
    ],
    ids=["mutual-name-cycle", "self-reference", "unresolved-name"],
)
def test_unresolvable_tool_name_terminates_without_false_positive(
    tmp_path: Path, source: str
) -> None:
    """解不開的工具名要終止，而且不准硬猜成 git——回空，不爆棧。"""
    report = _run(tmp_path, source)
    assert report == [], f"解不開的名字被誤判：{report}"


# ── 關鍵字形：which(cmd=...) ／ environ.get(..., default=...) 一樣要咬 ──────────

@pytest.mark.parametrize(
    "source",
    [
        (
            "import subprocess\n"
            "import shutil\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([shutil.which(cmd=\"git\"), \"status\"])\n"
        ),
        (
            "import os\n"
            "import subprocess\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([os.environ.get(\"GIT\", default=\"git\"), \"status\"])\n"
        ),
    ],
    ids=["which-cmd-keyword", "env-get-default-keyword"],
)
def test_keyword_locator_forms_are_still_a_violation(tmp_path: Path, source: str) -> None:
    """``which(cmd=...)``／``environ.get(..., default=...)`` 的關鍵字形不經 fixture 也要報 spawn。"""
    report = _run(tmp_path, source)
    assert _spawn(report), f"漏抓關鍵字形的 spawn 違規：{report}"


# ── 混合條件綁定：名字在 if／else 兩邊各綁一次，任一綁定算得出 git 就咬 ──────────

def test_mixed_conditional_binding_is_a_violation(tmp_path: Path) -> None:
    """同一個名字在 if／else 兩邊各綁一次，其中一邊是 git——任一綁定算出 git 就該咬。"""
    source = (
        "import shutil\n"
        "import subprocess\n"
        "\n"
        "\n"
        "def test_git():\n"
        "    if False:\n"
        "        tool = \"python\"\n"
        "    else:\n"
        "        tool = \"git\"\n"
        "    subprocess.run([shutil.which(tool), \"status\"])\n"
    )
    report = _run(tmp_path, source)
    assert _spawn(report), f"漏抓混合條件綁定：{report}"


# ── 同名重新指定：同一個名字先綁字串再綁定位呼叫，兩層解析領域不准互相當成循環 ─────
# 外層（spawn 第一格）追名別鏈，內層（which 的 cmd／environ.get 的 default）追純字串，
# 是兩個獨立的領域。修好之前，外層走過的 tool 進了 visited，內層的 which(tool) 再碰到
# tool 就被當成循環而回 False——這不是真的循環（內層永遠繞不回外層），一筆都不報。

@pytest.mark.parametrize(
    "source",
    [
        (
            "import shutil\n"
            "import subprocess\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    tool = \"git\"\n"
            "    tool = shutil.which(tool)\n"
            "    subprocess.run([tool, \"status\"])\n"
        ),
        (
            "import os\n"
            "import subprocess\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    tool = \"git\"\n"
            "    tool = os.environ.get(\"GIT\", tool)\n"
            "    subprocess.run([tool, \"status\"])\n"
        ),
        (
            "import shutil\n"
            "import subprocess\n"
            "\n"
            "tool = \"git\"\n"
            "alias = shutil.which(tool)\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run([alias, \"status\"])\n"
        ),
    ],
    ids=["reassigned-to-which", "reassigned-to-env-get", "alias-then-which-same-name"],
)
def test_same_name_reassigned_to_locator_is_a_violation(tmp_path: Path, source: str) -> None:
    """同一名字先綁字串、再綁定位呼叫，外層那次 spawn 要被咬。

    這條釘住「兩個解析領域互相獨立」：外層的 visited 不准讓內層誤判循環。
    """
    report = _run(tmp_path, source)
    assert _spawn(report), f"同名重新指定漏抓 spawn：{report}"


def test_true_cycle_still_terminates_after_domain_split(tmp_path: Path) -> None:
    """拆開兩個領域之後，真正的名字循環照樣要停——不准為了修洞把循環偵測也拆掉。"""
    source = (
        "import shutil\n"
        "import subprocess\n"
        "\n"
        "tool = alias\n"
        "alias = tool\n"
        "\n"
        "\n"
        "def test_git():\n"
        "    subprocess.run([shutil.which(tool), \"status\"])\n"
    )
    report = _run(tmp_path, source)
    assert report == [], f"真循環沒停下來或硬猜成 git：{report}"


# ── 回歸對照：名字綁到純字串／f-string 不解、參數與本層指派遮蔽上層綁定 ──────────

@pytest.mark.parametrize(
    "source",
    [
        (
            "import subprocess\n"
            "\n"
            "CMD = \"git\"\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run(CMD)\n"
        ),
        (
            "import subprocess\n"
            "\n"
            "CHECK = \"uv run pytest\"\n"
            "CMD = f\"git -c alias.rc='!{CHECK}' rc\"\n"
            "\n"
            "\n"
            "def test_git():\n"
            "    subprocess.run(CMD)\n"
        ),
    ],
    ids=["plain-string-name", "fstring-name"],
)
def test_name_bound_to_plain_string_stays_unresolved(tmp_path: Path, source: str) -> None:
    """名字綁到純字串／f-string 不解——``CMD = \"git\"`` 今天抓不到，是已知的覆蓋限制，不是保證安全。

    這條守住不回頭誤咬：``Row(f\"git -c alias.rc=...\", ...)`` 那種「整句命令字串存進名字」
    的寫法若照外層解到字面，就會把 tests/test_ci_command_segments.py 的 ALIAS_VCS 誤報違規
    （它名字綁到 f-string），所以外層只追 list/tuple、定位呼叫與別名鏈，不追純字串。
    這裡斷言的是「不誤報」，不等於這種寫法保證乾淨、或保證有乾淨的 Git 行為——純字串名字
    綁定是本次未覆蓋的已知限制。
    """
    report = _run(tmp_path, source)
    assert report == [], f"純字串／f-string 名字綁定被誤當 spawn：{report}"


def test_parameter_shadows_module_binding(tmp_path: Path) -> None:
    """函式參數覆蓋同名的上層綁定——``which(tool)`` 的 tool 是參數，不再吃模組層的 \"git\"。"""
    source = (
        "import shutil\n"
        "import subprocess\n"
        "\n"
        "tool = \"git\"\n"
        "\n"
        "\n"
        "def test_git(tool):\n"
        "    subprocess.run([shutil.which(tool), \"status\"])\n"
    )
    report = _run(tmp_path, source)
    assert report == [], f"參數遮蔽沒生效，誤咬到模組層的 tool：{report}"


def test_local_assignment_shadows_module_binding(tmp_path: Path) -> None:
    """函式內重新指定同名名字，覆蓋上層綁定——本層指成 \"python\"，不再吃模組層的 \"git\"。"""
    source = (
        "import shutil\n"
        "import subprocess\n"
        "\n"
        "TOOL = \"git\"\n"
        "\n"
        "\n"
        "def test_git():\n"
        "    TOOL = \"python\"\n"
        "    subprocess.run([shutil.which(TOOL), \"status\"])\n"
    )
    report = _run(tmp_path, source)
    assert report == [], f"本層重新指定沒覆蓋上層綁定：{report}"
