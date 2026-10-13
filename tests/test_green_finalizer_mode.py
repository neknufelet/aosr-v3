"""收尾接線逐棵驗訊息：底子壞而碰巧紅不能冒充守到了指定的那一格。"""
from __future__ import annotations

import re
import shutil
import tomllib
from pathlib import Path

import pytest

from governance.checks import green_must_be_real_green as subject
from governance.exit_codes import CLEAN, ToolBroken
from governance.loader import load_all_cards
from tests.test_fixture_runner import _assert_report_line, _run, _tail

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / 'governance/fixtures'
CONTROL = FIXTURES / 'green-must-be-real-green-green/control'
PARTS = 'governance/receipts/parts'
PART_A = f'{PARTS}/receipts-123-2-shard-a'
MANIFEST_A = f'片 shard-a {Path(PART_A) / "shard" / "manifest.json"}'
MERGED = "2 片合併 ['shard-a', 'shard-b'] junit.xml"
WANTED_PACKAGES = "['receipts-123-2-shard-a', 'receipts-123-2-shard-b']"
WANTED_CASES = [(f'tests.test_{letter}', f'test_{index}') for letter in ('a', 'b') for index in range(9)]
TEST_A = str(Path('tests') / 'test_a.py')
TEST_B = str(Path('tests') / 'test_b.py')
ARGS_A = (TEST_A + '\n').encode()
ARGS_BOTH = (TEST_A + '\n' + TEST_B + '\n').encode()
RED_WORDS = {
    'case-shard-result-failure': ("片 shard-a needs.result='failure'，預期 'success'",),
    'case-shard-result-cancelled': ("片 shard-a needs.result='cancelled'，預期 'success'",),
    'case-shard-result-skipped': ("片 shard-a needs.result='skipped'，預期 'success'",),
    'case-needs-lacks-registered-shard': ("needs.鍵集合=['shard-a']，預期 ['shard-a', 'shard-b']",),
    'case-part-missing': (f"片 shard-a {PARTS}.缺包=None，預期 'receipts-123-2-shard-a'",),
    'case-part-only-previous-attempt': (
        f"{PARTS}.未登記的包='receipts-123-1-shard-a'，預期 {WANTED_PACKAGES}",
        f"片 shard-a {PARTS}.缺包=None，預期 'receipts-123-2-shard-a'"),
    'case-part-unregistered-extra': (f"{PARTS}.未登記的包='unknown'，預期 {WANTED_PACKAGES}",),
    'case-part-two-junit-files': (f"片 shard-a {PART_A}.額外 XML='extra.xml'，預期 'junit.xml'",),
    'case-part-tested-sha-is-pr-head': (f"{MANIFEST_A}.tested_sha={'b' * 40!r}，預期 {'a' * 40!r}",),
    'case-part-other-run-id': (f"{MANIFEST_A}.run_id='456'，預期 '123'",),
    'case-part-other-attempt-in-manifest': (f"{MANIFEST_A}.run_attempt='1'，預期 '2'",),
    'case-part-job-mismatch': (f"{MANIFEST_A}.job='shard-b'，預期 'shard-a'",),
    'case-fragment-other-attempt': (f"片 shard-a {PART_A}/steps/pytest.json.run_attempt='1'，預期 '2'",),
    'case-part-junit-replaced-after-seal': (
        f"{MANIFEST_A}.junit.sha256='58fb357dbc7a400f8c7faf2ef26fe3c79882c015376106c2c333eed22cc15f35'，"
        "預期 'cb700b474e1dc001c3698062e14fd0584d22e6479980204df972ca50e33fe7d2'",
        f"{MANIFEST_A}.junit.bytes=553，預期 554"),
    'case-governance-step-fragment-missing': (
        f"片 shard-a {PART_A}/steps.缺片段=['style_guard']，"
        "預期 ['prepare', 'pytest', 'refs_and_links_resolve', 'seal', 'style_guard']",),
    'case-step-fragment-nonzero-while-job-success': (f"片 shard-a {PART_A}/steps/style_guard.json.exit_code=1，預期 0",),
    'case-test-missing': (f"片 shard-a junit.xml.testcase('tests.test_a', 'test_8').漏題=[]，預期 '在 {TEST_A} 恰好一次'",),
    'case-test-duplicated-same-total': (
        f"片 shard-a junit.xml.testcase('tests.test_a', 'test_8').漏題=[]，預期 '在 {TEST_A} 恰好一次'",
        "片 shard-a junit.xml.testcase('tests.test_a', 'test_0').片內重複=2，預期 '恰好一次'"),
    'case-test-run-in-both-shards': (
        f"片 shard-b junit.xml.testcase('tests.test_a', 'test_0').不屬本片={TEST_A!r}，預期 {[TEST_B]!r}",
        "片 ['shard-a', 'shard-b'] junit.xml.testcase('tests.test_a', 'test_0').跨片重複=['shard-a', 'shard-b']，預期 '僅一片'"),
    'case-test-unknown-extra': (f"片 shard-a junit.xml.不明題=('tests.test_a', 'test_unknown')，預期 {WANTED_CASES!r}",),
    'case-part-skipped': (
        "片 shard-a junit.xml.testcase('tests.test_a', 'test_0').skipped='<skipped />'，預期 '沒有跳過'",
        f"{MERGED} 的 skipped=1，必須是 0（收集 18 個）"),
    'case-part-failure': (
        "片 shard-a junit.xml.testcase('tests.test_a', 'test_0').failure='<failure />'，預期 '沒有失敗／錯誤'",
        f"{MERGED} 裡 failures=1、errors=0，卻被當成綠"),
    'case-union-below-floor': (f"{MERGED} 的籃子 'tests.' 只收集到 18 個測試，卡 minimal-card 上登記的地板是 19",),
    'case-part-tests-attr-mismatch': ("片 shard-a junit.xml.tests=10，預期 9", "片 shard-b junit.xml.tests=8，預期 9"),
    'case-parts-dir-absent': (f"片 shard-a {PARTS}: 缺包 receipts-123-2-shard-a", f"片 shard-b {PARTS}: 缺包 receipts-123-2-shard-b"),
    'case-shard-args-not-own': (f"片 shard-a pytest.args.位元組={ARGS_BOTH!r}，預期 {ARGS_A!r}",),
    'case-finalizer-formal-receipt-present': ("卡 minimal-card governance/receipts/pytest.junit.xml: 收尾的正式收據在場",),
}
BROKEN_WORDS = {
    'broken-needs-json-unparseable': re.escape('needs') + '.*JSON',
    'broken-manifest-unparseable': re.escape(str(Path('shard') / 'manifest.json')) + '.*JSON',
    'broken-fixture-run-file-missing-field': re.escape('fixture-finalizer-run.json.env') + '.*' + re.escape("缺 ['GITHUB_RUN_ATTEMPT']；多 []"),
    'broken-expected-lists-differ': re.escape('expected') + '.*逐位元組',
    'broken-expected-mapping-collision': re.escape('expected.nodeid') + '.*撞名',
    'broken-expected-list-empty': re.escape('expected') + '.*' + re.escape("清單是空的，實際 b''"),
    'broken-expected-list-unparseable': re.escape('expected') + '.*UTF-8',
    'broken-part-junit-unparseable': re.escape('junit.xml') + '.*解不開',
    'broken-settings-missing-key': 'settings.*' + re.escape("缺 ['parts_dir']；多 []"),
}


def files(root: Path) -> list[Path]:
    """樣本是獨立掃描根；不把外層工作樹檔案帶進去。"""
    return sorted(path for path in root.rglob('*') if path.is_file())


@pytest.fixture
def control(tmp_path: Path) -> Path:
    """有寫入的題目只動此題的控制副本。"""
    return Path(shutil.copytree(CONTROL, tmp_path / 'control'))


def test_finalizer_green_control_check() -> None:
    """漏接收尾模式會退回單機，控制樹沒有正式收據就紅。"""
    assert subject.check(CONTROL, files(CONTROL)) == []


def test_finalizer_green_control_cli() -> None:
    """綠控制由正式外殼回 0 並印判決行。"""
    card = next(card for card in load_all_cards(REPO) if card.check_module == subject.__name__)
    proc = _run(card, CONTROL)
    _assert_report_line(proc)
    assert proc.returncode == CLEAN, _tail(proc)


@pytest.mark.parametrize('case,words', RED_WORDS.items())
def test_finalizer_red_fixture_only_its_own_fault(case: str, words: tuple[str, ...]) -> None:
    """每條問題至少命中一個指定字樣，且各指定字樣全部出現。"""
    root = FIXTURES / 'green-must-be-real-green' / case
    bad = [re.sub(re.escape(str(root) + '/'), '', item) for item in subject.check(root, files(root))]
    assert bad, case
    assert all(any(word in item for word in words) for item in bad), bad
    assert all(any(word in item for item in bad) for word in words), bad
    if case == 'case-union-below-floor':
        assert all('地板過期' not in item for item in bad), bad


@pytest.mark.parametrize('case,word', BROKEN_WORDS.items())
def test_finalizer_broken_fixture_has_specific_reason(case: str, word: str) -> None:
    """未知输入必須回 ToolBroken，不能空集合或誤判紅。"""
    root = FIXTURES / 'green-must-be-real-green-tool-broken' / case
    with pytest.raises(ToolBroken, match=word):
        subject.check(root, files(root))


def test_every_finalizer_fixture_has_its_words() -> None:
    """帶樣本身分檔的樹都要列在字樣表；新加一棵沒列，它紅的理由就沒人驗。"""
    run_file = tomllib.loads((CONTROL / 'governance' / 'rules' / 'minimal-card.toml').read_text())['settings']['fixture_finalizer_run_file']
    found = {(root.parent.name, root.name) for root in FIXTURES.glob('green-must-be-real-green*/*') if (root / run_file).is_file()}
    listed = {('green-must-be-real-green', case) for case in RED_WORDS}
    listed |= {('green-must-be-real-green-tool-broken', case) for case in BROKEN_WORDS}
    assert found - {('green-must-be-real-green-green', 'control')} == listed


def test_wrapper_command_has_same_authority_in_both_cards() -> None:
    """兩張真卡的抄寫員命令漂移時不能讓片段守門悄悄失聯。"""
    green = tomllib.loads((REPO / 'governance/rules/green-must-be-real-green.toml').read_text())
    ci = tomllib.loads((REPO / 'governance/rules/ci-jobs-cannot-die-quietly.toml').read_text())
    assert green['settings']['wrapper_command'] == ci['settings']['wrapper_command']


def test_fixture_tree_has_no_empty_directories() -> None:
    """git 不保存空目錄：依賴空包才能紅的樣本必須先在本機被攔下；只讀樣本樹。"""
    empty = {str(path.relative_to(FIXTURES)) for path in FIXTURES.rglob('*')
             if path.is_dir() and not any(path.iterdir())}
    assert empty == set(), sorted(empty)
