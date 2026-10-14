"""產品代寫的靜態判準：只解析輸入字串，不載入或執行產品。"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance.checks import tests_isolated_from_real_env as checker
from governance.exit_codes import ToolBroken

CARD = (
    'id = "tests-isolated-from-real-env"\n'
    f'check = "{checker.CHECK_REL}"\n'
    '[settings]\n'
    'sandbox_fixture = "git_sandbox"\n'
    'product_writers = ["regenerate_schema_files"]\n'
    'committed_constants = ["_SCHEMA_DIR"]\n'
    'committed_paths = ["blueprint/schemas"]\n'
)
WRITE_MARK = "代寫函式收到版控路徑"
TEST_REL = Path("tests") / "test_generated.py"


def _run(tmp_path: Path, source: str, card_text: str = CARD) -> list[str]:
    """造最小掃描樹，回真的檢查判決；來源從來不被執行。"""
    card = tmp_path / "governance" / "rules" / "tests-isolated-from-real-env.toml"
    card.parent.mkdir(parents=True)
    card.write_text(card_text, encoding="utf-8")
    test = tmp_path / TEST_REL
    test.parent.mkdir(parents=True)
    test.write_text(source, encoding="utf-8")
    ignore = tmp_path / ".gitignore"
    ignore.write_text("/blueprint/schemas/\n", encoding="utf-8")
    return checker.check(tmp_path, [card, test, ignore])


def test_committed_constant_attribute_is_red(tmp_path: Path) -> None:
    report = _run(tmp_path, "report_io.regenerate_schema_files(report_io._SCHEMA_DIR)")
    assert any(WRITE_MARK in row for row in report)


def test_bare_committed_name_is_red(tmp_path: Path) -> None:
    source = "from aosr.physics.report_io import _SCHEMA_DIR\nregenerate_schema_files(_SCHEMA_DIR)"
    assert any(WRITE_MARK in row for row in _run(tmp_path, source))


def test_tree_anchored_path_by_keyword_is_red(tmp_path: Path) -> None:
    source = (
        "from pathlib import Path\n"
        "REPO = Path(__file__).resolve().parents[1]\n"
        "regenerate_schema_files(directory=REPO / 'blueprint' / 'schemas')\n"
    )
    assert any(WRITE_MARK in row for row in _run(tmp_path, source))


@pytest.mark.parametrize(
    "path", ["blueprint/schemas", "blueprint/schemas/nested", "/blueprint/schemas/", "blueprint/schemas-extra"]
)
def test_committed_literal_path_is_red(tmp_path: Path, path: str) -> None:
    source = f"regenerate_schema_files(Path({path!r}))"
    assert any(WRITE_MARK in row for row in _run(tmp_path, source))


def test_same_call_under_git_sandbox_is_still_red(tmp_path: Path) -> None:
    source = (
        "def git_sandbox():\n    pass\n"
        "def test_write(git_sandbox):\n"
        "    report_io.regenerate_schema_files(report_io._SCHEMA_DIR)\n"
    )
    assert any(WRITE_MARK in row for row in _run(tmp_path, source))


def test_allow_entry_cannot_excuse_committed_write(tmp_path: Path) -> None:
    source = "def test_write():\n    report_io.regenerate_schema_files(report_io._SCHEMA_DIR)\n"
    card = CARD + (
        f'[[settings.allow]]\nfile = "{TEST_REL.as_posix()}"\nfunction = "test_write"\n'
        'path = "blueprint/schemas"\nreason = "負控制：放行不能免掉產品代寫"\n'
        'expires = "2026-12-08"\n'
    )
    assert any(WRITE_MARK in row for row in _run(tmp_path, source, card))


def test_tmp_path_is_clean(tmp_path: Path) -> None:
    assert _run(tmp_path, "def test_write(tmp_path):\n    regenerate_schema_files(tmp_path)") == []


def test_reading_committed_dir_is_clean(tmp_path: Path) -> None:
    assert _run(tmp_path, "json.loads((report_io._SCHEMA_DIR / name).read_text())") == []


def test_unregistered_callee_with_committed_arg_is_clean(tmp_path: Path) -> None:
    assert _run(tmp_path, "print(report_io._SCHEMA_DIR)") == []


def test_malformed_product_writers_is_tool_broken(tmp_path: Path) -> None:
    card = CARD.replace('["regenerate_schema_files"]', '"regenerate_schema_files"')
    with pytest.raises(ToolBroken, match="product_writers"):
        _run(tmp_path, "pass", card)


@pytest.mark.parametrize("key", ["product_writers", "committed_constants", "committed_paths"])
@pytest.mark.parametrize("value", ['[]', '[""]', '["   "]', '[1]', '["ok", 1]'])
def test_optional_list_must_contain_nonempty_strings(tmp_path: Path, key: str, value: str) -> None:
    lines = [f"{key} = {value}" if row.startswith(f"{key} = ") else row for row in CARD.splitlines()]
    with pytest.raises(ToolBroken, match=key):
        _run(tmp_path, "pass", "\n".join(lines))


@pytest.mark.parametrize("name", ["bad-name", "two names"])
def test_constant_must_be_identifier(tmp_path: Path, name: str) -> None:
    with pytest.raises(ToolBroken, match="committed_constants"):
        _run(tmp_path, "pass", CARD.replace('"_SCHEMA_DIR"', f'"{name}"'))


@pytest.mark.parametrize("path", ["/blueprint/schemas", "../schemas", "blueprint/../schemas"])
def test_committed_path_must_stay_relative(tmp_path: Path, path: str) -> None:
    with pytest.raises(ToolBroken, match="committed_paths"):
        _run(tmp_path, "pass", CARD.replace('"blueprint/schemas"', f'"{path}"'))
