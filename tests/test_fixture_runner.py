"""後設測試：每張規矩卡都要跑完八回合，任一回合不符就紅。

八回合（退出碼約定見 governance/exit_codes.py）：
  1. 乾淨樹（真 repo 根）        = 0（宣告了 [junit] 的卡固定改為 = 1，見下）
  2. 卡宣告的每一份必紅樣本      = 1
  3. 掃描根換成不存在的路徑      = 2
  4. 抽掉卡宣告的外部工具        = 2（卡宣告 external_tools = [] 時改為斷言「宣告為空」並記一行，不 skip）
  5. 控制樣本（已知會咬的最小輸入）= 1
  6. 卡宣告的每一份「該回 2」樣本 = 2（沒宣告 tool_broken_fixture 的卡改為斷言它是空的並記一行，不 skip）
  7. 新候選的必紅樣本不被既有檢查咬到 = 沒有未宣告重疊
  8. 每份必紅樣本至少有一個檔落在卡的 scope

正式八回合不認識任何一張卡的內容，全部從卡的欄位讀；只有第七回控制組刻意拿真卡證明裁判會紅、且回 2 不算命中。

第 1 回有一個由卡的欄位決定的分支：卡宣告了 `[junit]`（要判一份 pytest 收據）時，固定斷言
「收據此刻不存在，而且檢查必須回 1」——沒有收據就是沒有綠。這一跑真的收據由
`.github/workflows/verify.yml` 在 pytest 之後的「綠必須是真的綠」那一步判；後設測試裡這張卡
的第 1 回只證明缺收據不能回 0，也不自己造收據。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from governance.checks import secrets_never_committed
from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION
from governance.loader import Card, expand_scope, load_all_cards
from governance.mainline_cards import mainline_card_ids
from tests.conftest import GitSandbox

REPO = Path(__file__).resolve().parents[1]

CARDS = load_all_cards(REPO)
CARD_IDS = [c.id for c in CARDS]

def _guards_rule_cards(card: Card) -> bool:
    """scope 含規矩卡目錄的卡是守卡者，不拿來被餵。"""
    # 刻意只認這個字面前綴：scope 可能寫到目錄下的檔案或 glob，這條窄縫不擴成所有 toml。
    return any(item.startswith("governance/rules") for item in card.scope)


RANGE_BASE = os.environ.get("AOSR_RANGE_BASE")
MAINLINE_CARDS = mainline_card_ids(REPO, RANGE_BASE or "origin/main")
MAINLINE_CARD_IDS = MAINLINE_CARDS.ids
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
if MAINLINE_CARD_IDS is not None:
    ROUND7_EMPTY_PARAM_ID = "候選集合為空"
elif MAINLINE_CARDS.reason == "找不到 git":
    ROUND7_EMPTY_PARAM_ID = "沒有候選名單-找不到git"
else:
    ROUND7_EMPTY_PARAM_ID = "沒有候選名單-讀不到base"
ROUND7_PARAM_IDS = [card.id if card else ROUND7_EMPTY_PARAM_ID for card in ROUND7_PARAMS]


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

    例外：卡宣告了 `[junit]` 時，固定斷言收據不存在、檢查必須回 1。真的收據留給 CI
    在 pytest 之後判；這裡只證明「沒有收據就是沒有綠」。
    """
    if card.junit:
        receipt = REPO / card.junit_path
        assert not receipt.exists(), (
            f"後設測試第 1 回裡 {card.junit_path} 竟然已經存在，這個斷言就證明不了什麼了"
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


def test_secrets_fixture_history_is_materialized_outside_scan_root(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """檢查程式建樣本歷史時，暫存 repo 不准成為被掃樣本樹的一部分。"""
    temp_root = git_sandbox.root / "checker-temp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    case = (
        REPO
        / "governance"
        / "fixtures"
        / "secrets-never-committed-tool-broken"
        / "case-shallow-history"
    )

    work_tree, cleanup = secrets_never_committed._resolve_history(case)
    try:
        assert work_tree.is_relative_to(temp_root), (
            f"樣本歷史建在 {work_tree}，沒有落在沙箱暫存根 {temp_root}"
        )
    finally:
        cleanup()


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
    observed, fed_cases = _round7_overlaps(
        card, other_cards, scan_root, module_root=module_root
    )
    assert fed_cases > 0, f"{card.id} 第 7 回：沒有非控制必紅樣本可餵"
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
) -> tuple[dict[str, list[str]], int]:
    """逐個非控制 case 走真子行程餵其他卡；回報 case 數，只記離開碼 1。"""
    observed: dict[str, list[str]] = {}
    fed_cases = 0
    control = card.control_path(scan_root)
    for case in card.negative_cases(scan_root):
        if case == control:
            continue
        fed_cases += 1
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
    return observed, fed_cases


def _copy_round7_runtime(sandbox_root: Path) -> None:
    """只複製控制組子行程可匯入的治理模組與卡，不碰其他卡的樣本樹。"""
    source = REPO / "governance"
    destination = sandbox_root / "governance"
    destination.mkdir()
    for module in sorted(source.glob("*.py")):
        shutil.copy2(module, destination / module.name)
    for package in ("checks", "rules", "status"):
        shutil.copytree(
            source / package,
            destination / package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )


@pytest.mark.parametrize("card", ROUND7_PARAMS, ids=ROUND7_PARAM_IDS)
def test_round7_new_candidates_have_no_overlap(
    card: Card | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """第 7 回：只餵主線名單外的新候選；空集合也收成一題明白的通過。"""
    with capsys.disabled():
        print(f"\n[第7回] 排除守卡檢查：{ROUND7_EXCLUDED}")
    if card is None:
        assert not RANGE_BASE or MAINLINE_CARD_IDS is not None, (
            f"第 7 回：明確指定的 {MAINLINE_CARDS.reason}，不准當成沒有候選而通過"
        )
        with capsys.disabled():
            if MAINLINE_CARD_IDS is None:
                print(f"[第7回] 沒有候選名單：{MAINLINE_CARDS.reason}")
            else:
                assert not ROUND7_CANDIDATES
                print(f"[第7回] 候選集合為空：base={MAINLINE_CARDS.base}")
        return
    _assert_round7(card, ROUND7_CHECKERS, REPO)


def test_round7_control_detects_a_real_other_checker(git_sandbox: GitSandbox) -> None:
    """第 7 回控制組：假候選的樣本走真子行程，必須被第一張非守卡真卡抓到。"""
    target = ROUND7_ELIGIBLE_CHECKERS[0]
    _copy_round7_runtime(git_sandbox.root)
    # 守住「沙箱只複製需要的子樹」這一半：整棵 governance/ 複製會把 28 張卡的樣本樹（含別題
    # 正在寫的暫存物）一起搬進來，正是先前偶紅的根因；fixtures 不該出現在沙箱裡。
    assert not (git_sandbox.root / "governance" / "fixtures").exists()
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
    """第 7 回控制組：case 目錄沒有真卡 toml，真卡回 2 時不算命中。"""
    target = next(
        card
        for card in ROUND7_ELIGIBLE_CHECKERS
        if "\n[settings]" in card.source.read_text(encoding="utf-8")
    )
    _copy_round7_runtime(git_sandbox.root)
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
