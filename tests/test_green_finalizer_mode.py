"""收尾接線逐棵驗訊息：底子壞而碰巧紅不能冒充守到了指定的那一格。"""
from __future__ import annotations

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
RED_WORDS = {
    'case-shard-result-failure': ('needs.result',),
    'case-shard-result-cancelled': ('needs.result',),
    'case-shard-result-skipped': ('needs.result',),
    'case-needs-lacks-registered-shard': ('needs.鍵集合',),
    'case-part-missing': ('缺包',),
    'case-part-only-previous-attempt': ('未登記的包', '缺包'),
    'case-part-unregistered-extra': ('未登記的包',),
    'case-part-two-junit-files': ('額外 XML',),
    'case-part-tested-sha-is-pr-head': ('.tested_sha',),
    'case-part-other-run-id': ('.run_id',),
    'case-part-other-attempt-in-manifest': ('.run_attempt',),
    'case-part-job-mismatch': ('.job',),
    'case-fragment-other-attempt': ('.run_attempt',),
    'case-part-junit-replaced-after-seal': ('junit.sha256', 'junit.bytes'),
    'case-governance-step-fragment-missing': ('缺片段',),
    'case-step-fragment-nonzero-while-job-success': ('.exit_code',),
    'case-test-missing': ('漏題',),
    'case-test-duplicated-same-total': ('漏題', '片內重複'),
    'case-test-run-in-both-shards': ('不屬本片', '跨片重複'),
    'case-test-unknown-extra': ('不明題',),
    'case-part-skipped': ('skipped=',),
    'case-part-failure': ('.failure', 'failures='),
    'case-union-below-floor': ('地板',),
    'case-part-tests-attr-mismatch': ('junit.xml.tests=',),
    'case-parts-dir-absent': ('缺包',),
    'case-shard-args-not-own': ('pytest.args.位元組',),
    'case-finalizer-formal-receipt-present': ('正式收據',),
}
BROKEN_WORDS = {
    'broken-needs-json-unparseable': 'needs.*JSON',
    'broken-manifest-unparseable': 'manifest.json.*JSON',
    'broken-fixture-run-file-missing-field': 'GITHUB_RUN_ATTEMPT',
    'broken-expected-lists-differ': 'expected.*逐位元組',
    'broken-expected-mapping-collision': 'expected.nodeid.*撞名',
    'broken-expected-list-empty': 'expected.*空',
    'broken-expected-list-unparseable': 'expected.*UTF-8',
    'broken-part-junit-unparseable': 'junit.xml.*解不開',
    'broken-settings-missing-key': 'settings.*parts_dir',
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
    bad = subject.check(root, files(root))
    assert bad, case
    assert all(any(word in item for word in words) for item in bad), bad
    assert all(any(word in item for item in bad) for word in words), bad


@pytest.mark.parametrize('case,word', BROKEN_WORDS.items())
def test_finalizer_broken_fixture_has_specific_reason(case: str, word: str) -> None:
    """未知输入必須回 ToolBroken，不能空集合或誤判紅。"""
    root = FIXTURES / 'green-must-be-real-green-tool-broken' / case
    with pytest.raises(ToolBroken, match=word):
        subject.check(root, files(root))


def test_every_finalizer_fixture_has_its_words() -> None:
    """帶樣本身分檔的樹都要列在字樣表；新加一棵沒列，它紅的理由就沒人驗。"""
    run_file = tomllib.loads((CONTROL / 'governance/rules/minimal-card.toml').read_text())['settings']['fixture_finalizer_run_file']
    found = {(root.parent.name, root.name) for root in FIXTURES.glob('green-must-be-real-green*/*') if (root / run_file).is_file()}
    listed = {('green-must-be-real-green', case) for case in RED_WORDS}
    listed |= {('green-must-be-real-green-tool-broken', case) for case in BROKEN_WORDS}
    assert found - {('green-must-be-real-green-green', 'control')} == listed


def test_wrapper_command_has_same_authority_in_both_cards() -> None:
    """兩張真卡的抄寫員命令漂移時不能讓片段守門悄悄失聯。"""
    green = tomllib.loads((REPO / 'governance/rules/green-must-be-real-green.toml').read_text())
    ci = tomllib.loads((REPO / 'governance/rules/ci-jobs-cannot-die-quietly.toml').read_text())
    assert green['settings']['wrapper_command'] == ci['settings']['wrapper_command']
