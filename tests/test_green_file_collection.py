"""#137：整支考卷未收集要紅；兩種收據入口不能漏接檔案名單。"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from governance.checks.green_must_be_real_green import (
    _assert_default_python_files,
    _collection_files,
    _file_collection_problems,
    _test_files,
    check,
)
from governance.exit_codes import ToolBroken

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / 'governance' / 'fixtures'
MISSING = str(Path('tests') / 'test_c.py')
TEST_A = str(Path('tests') / 'test_a.py')
RECEIPT = '單機 junit'


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob('*') if path.is_file())


def _receipt(*classnames: str) -> ET.Element:
    root = ET.Element('testsuite')
    for classname in classnames:
        ET.SubElement(root, 'testcase', classname=classname, name='test_ok')
    return root


def test_nested_classes_count_for_the_module_file() -> None:
    """若只比完整 classname，類別底下的題會被誤當整支沒收到。"""
    root = _receipt('tests.test_a.TestThing.TestInner')
    assert _file_collection_problems(RECEIPT, root, (TEST_A,)) == []


def test_similar_module_name_does_not_cover_a_shorter_name() -> None:
    """若改成字串前綴，test_ab 會替沒收到的 test_a 冒名。"""
    bad = _file_collection_problems(RECEIPT, _receipt('tests.test_ab'), (TEST_A,))
    assert bad
    assert all(TEST_A in item for item in bad), bad


def test_empty_classname_covers_no_file() -> None:
    """收集錯誤的空 classname 不准當成任一考卷收到的題。"""
    bad = _file_collection_problems(RECEIPT, _receipt(''), (TEST_A,))
    assert bad
    assert all(TEST_A in item for item in bad), bad


def test_only_the_longest_matching_file_gets_the_testcase() -> None:
    """若一題算給所有前綴檔，較短的空考卷也會漂綠。"""
    deeper = str(Path('tests') / 'test_a' / 'test_b.py')
    bad = _file_collection_problems(RECEIPT, _receipt('tests.test_a.test_b.TestThing'), (TEST_A, deeper))
    assert bad
    assert all(TEST_A in item and deeper not in item for item in bad), bad


def test_helpers_and_conftest_are_not_exam_files(tmp_path: Path) -> None:
    """篩選不能把所有 Python 檔都當考卷，否則輔助檔永遠沒題。"""
    sources = [tmp_path / 'tests' / (name + '.py') for name in ('helpers', 'conftest')]
    assert _collection_files(tmp_path, _test_files(tmp_path, sources)) == ()


def test_default_patterns_include_both_names_at_any_depth(tmp_path: Path) -> None:
    """漏掉 *_test 或只認 tests 根層，都會放走預設收集面上的空考卷。"""
    suffix_test = str(Path('tests') / 'nested' / 'feature_test.py')
    listed = (TEST_A, suffix_test)
    sources = [tmp_path / rel for rel in listed]
    assert set(_collection_files(tmp_path, _test_files(tmp_path, sources))) == set(listed)


def test_zero_testcases_does_not_report_each_file() -> None:
    """零題只留既有零收集訊息；不能逐支印幾百行。"""
    assert _file_collection_problems(RECEIPT, _receipt(), (TEST_A, MISSING)) == []


def test_partial_loss_inside_one_file_is_not_detected_by_this_rule() -> None:
    """兩題變一題仍有收到，這條抓不到的那格必須照實釘住。"""
    full = _receipt('tests.test_a', 'tests.test_a')
    assert _file_collection_problems(RECEIPT, full, (TEST_A,)) == []
    full.remove(next(full.iter('testcase')))
    assert _file_collection_problems(RECEIPT, full, (TEST_A,)) == []


@pytest.mark.parametrize('table', ('[tool.pytest.ini_options]', '[tool.pytest]'))
@pytest.mark.parametrize('value', ('[]', '["test_*.py", "*_test.py"]'))
def test_even_empty_or_default_python_files_setting_is_not_silently_accepted(
    tmp_path: Path, table: str, value: str,
) -> None:
    """只看鍵有沒有設；空值或照抄預設也不能略過判準變動守門。pytest 9 的原生 [tool.pytest] 那一層也算。"""
    home = tmp_path / 'pyproject.toml'
    home.write_text(table + '\npython_files = ' + value + '\n', encoding='utf-8')
    with pytest.raises(ToolBroken, match='名單跟著改之前不判'):
        _assert_default_python_files(tmp_path, [home])


@pytest.mark.parametrize('text', ('[tool.pytest', 'tool = "broken"'))
def test_unreadable_pytest_configuration_is_tool_broken(tmp_path: Path, text: str) -> None:
    """新讀設定的路不能把無法判定的形狀當成綠或直接崩潰。"""
    home = tmp_path / 'pyproject.toml'
    home.write_text(text, encoding='utf-8')
    with pytest.raises(ToolBroken):
        _assert_default_python_files(tmp_path, [home])


def test_single_receipt_reports_the_uncollected_file() -> None:
    """單機若仍只呼叫舊判官，這棵沒有其他違規的樣本會漂綠。"""
    root = FIXTURES / 'green-must-be-real-green' / 'case-test-file-collected-nowhere-single'
    bad = check(root, _files(root))
    assert bad, '單機漏接整支考卷檔未收集的判準'
    assert all(MISSING in item and '一題都沒有' in item for item in bad), bad
    assert all('governance/receipts/pytest.junit.xml' in item for item in bad), bad


def test_finalizer_receipt_reports_the_uncollected_file() -> None:
    """收尾若交給舊判官，expected 清點看不到的空考卷就會漂綠。"""
    root = FIXTURES / 'green-must-be-real-green' / 'case-test-file-collected-nowhere'
    bad = check(root, _files(root))
    assert bad, '收尾漏接整支考卷檔未收集的判準'
    assert all(MISSING in item and '一題都沒有' in item for item in bad), bad
    assert all("2 片合併 ['shard-a', 'shard-b'] junit.xml" in item for item in bad), bad


def test_python_files_override_is_tool_broken() -> None:
    """改了 pytest 的檔名模式，固定名單不能照舊判綠。"""
    root = FIXTURES / 'green-must-be-real-green-tool-broken' / 'broken-python-files-overridden'
    with pytest.raises(ToolBroken, match='考卷檔的檔名樣式被改了'):
        check(root, _files(root))
