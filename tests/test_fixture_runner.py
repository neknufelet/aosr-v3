"""後設測試：每張規矩卡都要跑完八回合，任一回合不符就紅。

八回合（退出碼約定見 governance/exit_codes.py）：
  1. 乾淨樹（真 repo 根）        = 0（宣告了 [junit] 的卡在「產收據那一跑」裡改為 = 1，見下）
  2. 卡宣告的每一份必紅樣本      = 1
  3. 掃描根換成不存在的路徑      = 2
  4. 抽掉卡宣告的外部工具        = 2（卡宣告 external_tools = [] 時改為斷言「宣告為空」並記一行，不 skip）
  5. 控制樣本（已知會咬的最小輸入）= 1
  6. 卡宣告的每一份「該回 2」樣本 = 2（沒宣告 tool_broken_fixture 的卡改為斷言它是空的並記一行，不 skip）
  7. 新候選的必紅樣本不被既有檢查咬到 = 沒有未宣告重疊
  8. 每份必紅樣本至少有一個檔落在卡的 scope

正式八回合不認識任何一張卡的內容，全部從卡的欄位讀；只有第七回控制組刻意拿真卡證明裁判會紅、且回 2 不算命中。

第 1 回有一個由卡的欄位決定的分支：卡宣告了 `[junit]`（要判一份 pytest 收據）時，
`tests/conftest.py` 會在開跑前先跑一次真的全套把收據產出來。**在那一跑裡**收據還不存在，
這時候正確答案是紅不是綠，所以第 1 回改為斷言「檢查必須回 1」——比平常更凶，不是放水。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION
from governance.loader import Card, expand_scope, load_all_cards
from tests.conftest import SEED_ENV, GitSandbox

REPO = Path(__file__).resolve().parents[1]
GIT_TOOL = "git"

CARDS = load_all_cards(REPO)
CARD_IDS = [c.id for c in CARDS]


def _mainline_card_ids(scan_root: Path) -> tuple[set[str] | None, str]:
    """從 CI base（本機則 origin/main）現算主線卡名；讀不到就明說這回不判。"""
    base = os.environ.get("AOSR_RANGE_BASE") or "origin/main"
    git = shutil.which(GIT_TOOL)
    if git is None:
        return None, "這一跑沒有候選名單，判準三不跑：找不到 git"
    try:
        proc = subprocess.run(
            [git, "ls-tree", "--name-only", base, "governance/rules/"],
            cwd=scan_root,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError) as exc:
        return None, f"這一跑沒有候選名單，判準三不跑：git ls-tree 跑不起來（{exc}）"
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        tail = detail[-1] if detail else f"離開碼 {proc.returncode}"
        return None, f"這一跑沒有候選名單，判準三不跑：讀不到 base={base}（{tail}）"
    names = {
        Path(line).stem
        for line in proc.stdout.splitlines()
        if line.startswith("governance/rules/") and line.endswith(".toml")
    }
    return names, f"這一跑沒有候選（base={base}）"


def _guards_rule_cards(card: Card) -> bool:
    """scope 含規矩卡目錄的卡是守卡者，不拿來被餵。"""
    return any(item == "governance/rules" or item.startswith("governance/rules") for item in card.scope)


MAINLINE_CARD_IDS, ROUND7_EMPTY_NOTE = _mainline_card_ids(REPO)
ROUND7_CANDIDATES = (
    [card for card in CARDS if card.id not in MAINLINE_CARD_IDS]
    if MAINLINE_CARD_IDS is not None
    else []
)
ROUND7_ELIGIBLE_CHECKERS = [card for card in CARDS if not _guards_rule_cards(card)]
ROUND7_CHECKERS = (
    [card for card in ROUND7_ELIGIBLE_CHECKERS if card.id in MAINLINE_CARD_IDS]
    if MAINLINE_CARD_IDS is not None
    else []
)
ROUND7_EXCLUDED = [card.id for card in CARDS if _guards_rule_cards(card)]
ROUND7_PARAMS: list[Card | None] = [*ROUND7_CANDIDATES] or [None]
ROUND7_PARAM_IDS = [card.id if card else "候選集合為空" for card in ROUND7_PARAMS]


def _run(
    card: Card,
    scan_root: Path | str,
    *,
    path: str | None = None,
    module_root: Path = REPO,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # 不要在被掃的樹裡留 __pycache__——樣本樹的檔案清單就是證據，不該被跑測試這件事改變。
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["AOSR_BITE_DEPTH"] = "0"
    # 子程序不要寫 .pyc：探針會在被掃的樹裡開暫存目錄再刪掉，多出來的 __pycache__ 會讓它刪不掉。
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if path is not None:
        env["PATH"] = path
    if module_root != REPO:
        env["PYTHONPATH"] = str(module_root)
    return subprocess.run(
        [sys.executable, "-m", card.check_module, "--scan-root", str(scan_root)],
        cwd=module_root,
        env=env,
        capture_output=True,
        text=True,
    )


def _path_without(tool: str) -> str:
    """把某個外部工具從 PATH 裡拿掉（只拿掉裝著它的那幾個目錄，不是清空 PATH）。"""
    kept = [
        d
        for d in os.environ.get("PATH", "").split(os.pathsep)
        if d and not (Path(d) / tool).exists()
    ]
    return os.pathsep.join(kept) or str(REPO / "governance" / "no-such-bin-dir")


def _tail(proc: subprocess.CompletedProcess[str]) -> str:
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return " | ".join(out[-6:])


def _assert_report_line(proc: subprocess.CompletedProcess[str]) -> None:
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("scan_root=")]
    assert lines, f"檢查沒有印出 scan_root=... files=... hits=... 那一行：{_tail(proc)}"
    parts = lines[-1].split()
    # 逐項具名比對，不寫 len(parts) == 3——鎖死數量的斷言由 assertions-not-pinned-to-counts 咬。
    # 這樣寫也比較有用：格式錯的時候直接印出「實際是哪幾個欄位」。
    keys = [p.split("=", 1)[0] for p in parts]
    assert keys == ["scan_root", "files", "hits"], f"報告行的欄位不對：{keys}（{lines[-1]!r}）"
    int(parts[1].removeprefix("files="))
    int(parts[2].removeprefix("hits="))


def test_there_is_at_least_one_card() -> None:
    """第 0 關：governance/rules/ 裡至少要有一張卡，不然這支後設測試會變成永遠回綠的空跑。"""
    assert CARDS, "governance/rules/ 裡一張卡都沒有——後設測試會變成永遠回綠的空跑"


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round1_clean_tree_is_green(card: Card) -> None:
    """第 1 回：乾淨樹（真 repo 根）必須回 0。

    例外：卡宣告了 `[junit]`、而且這一跑就是 `tests/conftest.py` 派去產收據的那一跑——
    收據此刻還不存在，正確答案是紅。那一回合改為斷言「檢查必須回 1」。
    """
    if card.junit and os.environ.get(SEED_ENV):
        receipt = REPO / card.junit_path
        assert not receipt.exists(), (
            f"產收據那一跑裡 {card.junit_path} 竟然已經存在，這個斷言就證明不了什麼了"
        )
        proc = _run(card, REPO)
        _assert_report_line(proc)
        assert proc.returncode == VIOLATION, (
            f"{card.id} 在收據不存在時回 {proc.returncode}，應為 1"
            f"——沒有收據就是沒有綠，不准當乾淨：{_tail(proc)}"
        )
        return
    proc = _run(card, REPO)
    _assert_report_line(proc)
    assert proc.returncode == CLEAN, f"{card.id} 在乾淨樹回 {proc.returncode}，應為 0：{_tail(proc)}"


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round2_negative_fixtures_are_red(card: Card) -> None:
    """第 2 回：卡宣告的每一份必紅樣本都必須回 1。"""
    cases = card.negative_cases(REPO)
    assert cases, f"{card.id} 的 negative_fixture 底下沒有任何樣本目錄"
    for case in cases:
        proc = _run(card, case)
        _assert_report_line(proc)
        assert proc.returncode == VIOLATION, (
            f"{card.id} 餵樣本 {case.relative_to(REPO)} 回 {proc.returncode}，應為 1：{_tail(proc)}"
        )


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round3_missing_scan_root_is_tool_broken(card: Card) -> None:
    """第 3 回：掃描根不存在必須回 2（工具自壞），不准回 0。"""
    missing = REPO / "governance" / "no-such-scan-root-ROUND3"
    assert not missing.exists()
    proc = _run(card, missing)
    _assert_report_line(proc)
    assert proc.returncode == TOOL_BROKEN, (
        f"{card.id} 掃描根不存在時回 {proc.returncode}，應為 2（工具自壞）：{_tail(proc)}"
    )


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round4_external_tool_removed_is_tool_broken(card: Card, capsys: pytest.CaptureFixture[str]) -> None:
    """第 4 回：把卡宣告的外部工具從 PATH 拿掉必須回 2；卡宣告 external_tools = [] 時斷言宣告為空並記一行，不 skip。"""
    if not card.external_tools:
        # 不 skip：卡必須明寫 external_tools = []，這裡斷言它真的是空的並記一行。
        assert card.external_tools == [], f"{card.id} 的 external_tools 不是空 list"
        with capsys.disabled():
            print(f"\n[第4回] {card.id} 宣告 external_tools = []（無外部工具可抽，已斷言宣告為空）")
        return
    for tool in card.external_tools:
        proc = _run(card, REPO, path=_path_without(tool))
        _assert_report_line(proc)
        assert proc.returncode == TOOL_BROKEN, (
            f"{card.id} 抽掉外部工具 {tool} 後回 {proc.returncode}，應為 2：{_tail(proc)}"
        )


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round5_control_fixture_is_red(card: Card) -> None:
    """第 5 回：控制樣本（已知會咬的最小輸入）必須回 1。"""
    control = card.control_path(REPO)
    proc = _run(card, control)
    _assert_report_line(proc)
    assert proc.returncode == VIOLATION, (
        f"{card.id} 餵控制樣本 {control.relative_to(REPO)} 回 {proc.returncode}，應為 1：{_tail(proc)}"
    )


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round6_tool_broken_fixtures_are_tool_broken(
    card: Card, capsys: pytest.CaptureFixture[str]
) -> None:
    """第 6 回：卡宣告的每一份「該回 2」樣本都必須回 2；沒宣告的卡斷言它是空的並記一行，不 skip。

    為什麼要另開一回合：必紅樣本（第 2 回）每一份都必須回 1，所以「名單檔不存在」「範圍算出來
    是空的」這種「這一跑不算數」的樣本放不進那個目錄——放進去等於要求同一份樣本同時回 1 又回 2。
    """
    cases = card.tool_broken_cases(REPO)
    if not card.tool_broken_fixture:
        # 不 skip：斷言這張卡真的沒宣告，並記一行，免得「跳過」被當成通過。
        assert cases == [], f"{card.id} 沒宣告 tool_broken_fixture，卻找出樣本 {cases}"
        with capsys.disabled():
            print(f"\n[第6回] {card.id} 沒宣告 tool_broken_fixture（沒有該回 2 的樣本，已斷言為空）")
        return
    assert cases, f"{card.id} 的 tool_broken_fixture 底下沒有任何樣本目錄"
    for case in cases:
        proc = _run(card, case)
        _assert_report_line(proc)
        assert proc.returncode == TOOL_BROKEN, (
            f"{card.id} 餵該回 2 的樣本 {case.relative_to(REPO)} 回 {proc.returncode}，"
            f"應為 2（工具自壞，這一跑不算數）：{_tail(proc)}"
        )


def _fake_card(tmp_path: Path, *, scope: list[str]) -> Card:
    source = tmp_path / "governance" / "rules" / "fake-card.toml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        '\n'.join(
            [
                'id = "fake-card"',
                'human = "只給後設測試控制組使用"',
                "scope = [" + ", ".join(f'\"{item}\"' for item in scope) + "]",
                'check = "governance/checks/check_exit_code_honest.py"',
                'negative_fixture = "negative"',
                'control_fixture = "negative/control"',
                'declared_level = "blocking"',
                'enforcer = "pytest-meta-test"',
                'job = "verify"',
                'external_tools = []',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return Card(
        id="fake-card",
        human="只給後設測試控制組使用",
        scope=scope,
        scope_kind="files",
        check="governance/checks/check_exit_code_honest.py",
        negative_fixture="negative",
        control_fixture="negative/control",
        declared_level="blocking",
        enforcer="pytest-meta-test",
        job="verify",
        external_tools=[],
        mountpoint={},
        source=source,
    )


def _assert_round7(
    card: Card,
    other_cards: list[Card],
    scan_root: Path,
    *,
    module_root: Path = REPO,
) -> None:
    """新候選的非控制必紅樣本不准被既有檢查咬到。"""
    observed = _round7_overlaps(card, other_cards, scan_root, module_root=module_root)
    actual = set(observed)
    details = "\n\n".join(item for records in observed.values() for item in records)
    assert not actual, (
        f"{card.id} 第 7 回：未宣告卻咬到={sorted(actual)}"
        + (f"\n\n{details}" if details else "")
    )


def _assert_round8(card: Card, scan_root: Path) -> None:
    """每份非控制樣本至少一個檔要落在卡宣告的 scope。"""
    if card.scope_kind == "commits":
        return
    control = card.control_path(scan_root)
    for case in card.negative_cases(scan_root):
        if case == control:
            continue
        names = sorted(
            path.relative_to(case).as_posix() for path in case.rglob("*") if path.is_file()
        )
        matched = expand_scope(card.scope, names)
        assert matched, (
            f"{card.id} 第 8 回：{case.relative_to(scan_root)} 沒有任何檔落在 scope={card.scope!r}"
        )


def _round7_overlaps(
    card: Card,
    other_cards: list[Card],
    scan_root: Path,
    *,
    module_root: Path = REPO,
) -> dict[str, list[str]]:
    """逐個非控制 case 走真子行程餵其他卡；只記離開碼 1。"""
    observed: dict[str, list[str]] = {}
    control = card.control_path(scan_root)
    for case in card.negative_cases(scan_root):
        if case == control:
            continue
        for other in other_cards:
            if other.id == card.id:
                continue
            proc = _run(other, case, module_root=module_root)
            if proc.returncode != VIOLATION:
                continue
            output = proc.stdout + proc.stderr
            record = (
                f"X={card.id}\ncase={case.relative_to(scan_root)}\n"
                f"Y={other.id}\noutput:\n{output}"
            )
            observed.setdefault(other.id, []).append(record)
    return observed


@pytest.mark.parametrize("card", ROUND7_PARAMS, ids=ROUND7_PARAM_IDS)
def test_round7_new_candidates_have_no_overlap(
    card: Card | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """第 7 回：只餵主線名單外的新候選；空集合也收成一題明白的通過。"""
    with capsys.disabled():
        print(f"\n[第7回] 排除守卡檢查：{ROUND7_EXCLUDED}")
    if card is None:
        assert not ROUND7_CANDIDATES
        with capsys.disabled():
            print(f"[第7回] 候選集合為空：{ROUND7_EMPTY_NOTE}")
        return
    _assert_round7(card, ROUND7_CHECKERS, REPO)


def test_round7_control_detects_a_real_other_checker(git_sandbox: GitSandbox) -> None:
    """第 7 回控制組：假候選的樣本走真子行程，必須被第一張非守卡真卡抓到。"""
    target = ROUND7_ELIGIBLE_CHECKERS[0]
    shutil.copytree(REPO / "governance", git_sandbox.root / "governance")
    case = git_sandbox.root / "negative" / "case-bitten-by-real-card"
    shutil.copytree(target.control_path(REPO), case)
    (git_sandbox.root / "negative" / "control").mkdir()
    fake = _fake_card(git_sandbox.root, scope=["."])
    git_sandbox.git("add", ".")

    proc = _run(target, case, module_root=git_sandbox.root)
    assert proc.returncode == VIOLATION, (
        f"控制組複製 {target.id} 的真控制樣本後回 {proc.returncode}，應為 1：{_tail(proc)}"
    )
    with pytest.raises(
        AssertionError,
        match=rf"fake-card 第 7 回：未宣告卻咬到=.*{re.escape(target.id)}",
    ):
        _assert_round7(fake, [target], git_sandbox.root, module_root=git_sandbox.root)


def test_round7_control_does_not_count_tool_broken_as_overlap(git_sandbox: GitSandbox) -> None:
    """第 7 回控制組：假候選不帶真卡需要的 settings，真卡回 2 時不算命中。"""
    target = next(
        card
        for card in ROUND7_ELIGIBLE_CHECKERS
        if "\n[settings]" in card.source.read_text(encoding="utf-8")
    )
    shutil.copytree(REPO / "governance", git_sandbox.root / "governance")
    case = git_sandbox.root / "negative" / "case-without-real-card-settings"
    case.mkdir(parents=True)
    (case / "README.md").write_text("# 沒有真卡的 settings\n", encoding="utf-8")
    (git_sandbox.root / "negative" / "control").mkdir()
    fake = _fake_card(git_sandbox.root, scope=["."])
    git_sandbox.git("add", ".")

    proc = _run(target, case, module_root=git_sandbox.root)
    assert proc.returncode == TOOL_BROKEN, (
        f"控制組挑到的 {target.id} 在缺 settings 時沒有回 2，而是 {proc.returncode}：{_tail(proc)}"
    )
    _assert_round7(fake, [target], git_sandbox.root, module_root=git_sandbox.root)


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round8_negative_fixtures_contain_a_scoped_file(
    card: Card, capsys: pytest.CaptureFixture[str]
) -> None:
    """第 8 回：每份必紅樣本至少一個檔要落在卡宣告的 scope。"""
    if card.scope_kind == "commits":
        with capsys.disabled():
            print(f"\n[第8回] {card.id} 的 scope_kind=commits（掃描面不是路徑，不比路徑，視為通過）")
    _assert_round8(card, REPO)


def test_round8_control_treats_commit_scope_as_not_path_comparable(tmp_path: Path) -> None:
    """第 8 回控制組：提交 metadata 的 scope 不拿樣本檔路徑硬比。"""
    case = tmp_path / "negative" / "case-only-markdown"
    case.mkdir(parents=True)
    (case / "README.md").write_text("# 提交掃描面不比路徑\n", encoding="utf-8")
    (tmp_path / "negative" / "control").mkdir()
    fake = _fake_card(tmp_path, scope=["src/**/*.py"])
    commits = replace(fake, scope_kind="commits")
    _assert_round8(commits, tmp_path)


def test_round8_control_rejects_fixture_without_a_scoped_file(tmp_path: Path) -> None:
    """第 8 回控制組：scope 只收 src/**/*.py、樣本只有 Markdown 時必須紅。"""
    case = tmp_path / "negative" / "case-only-markdown"
    case.mkdir(parents=True)
    (case / "README.md").write_text("# 不在假卡掃描面\n", encoding="utf-8")
    fake = _fake_card(tmp_path, scope=["src/**/*.py"])
    with pytest.raises(AssertionError, match="fake-card.*case-only-markdown"):
        _assert_round8(fake, tmp_path)
