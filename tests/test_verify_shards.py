"""分片登記簿的格式契約；壞輸入必須說出欄位與實際值。"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import re

import pytest

from governance.exit_codes import ToolBroken
from governance.verify_shards import load_registry

REGISTRY = '''schema = 1
shards = ["shard-a", "shard-b"]
default_shard = "shard-b"
[evidence]
dir = "governance/receipts/shard"
expected = "expected.txt"
args = "pytest.args"
junit = "junit.xml"
manifest = "manifest.json"
[[group]]
name = "control"
shard = "shard-a"
why = "共用控制組"
files = ["tests/test_verify_shards.py"]
'''


def write_registry(tmp_path: Path, text: str = REGISTRY) -> Path:
    """每題只把登記簿寫在自己的暫存樹。"""
    path = tmp_path / "verify-shards.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_legal_registry_is_immutable(tmp_path: Path) -> None:
    registry = load_registry(write_registry(tmp_path))
    assert registry.schema == 1
    assert registry.shards == ("shard-a", "shard-b")
    assert registry.default_shard == "shard-b"
    assert registry.evidence.dir == "governance/receipts/shard"
    assert registry.evidence.expected == "expected.txt"
    assert registry.evidence.args == "pytest.args"
    assert registry.evidence.junit == "junit.xml"
    assert registry.evidence.manifest == "manifest.json"
    assert registry.groups[0].files == ("tests/test_verify_shards.py",)
    with pytest.raises(FrozenInstanceError):
        setattr(registry, "schema", 2)
    with pytest.raises(FrozenInstanceError):
        setattr(registry.evidence, "args", "other")
    with pytest.raises(FrozenInstanceError):
        setattr(registry.groups[0], "shard", "shard-b")


BAD_FIELDS = [
    ("top-extra", "schema = 1", "schema = 1\nextra = 7", "registry"),
    ("top-missing", "schema = 1\n", "", "registry"),
    ("schema-wrong", "schema = 1", "schema = 2", "schema"),
    ("schema-bool", "schema = 1", "schema = true", "schema"),
    ("schema-string", "schema = 1", 'schema = "1"', "schema"),
    ("shards-scalar", '["shard-a", "shard-b"]', '"shard-a"', "shards"),
    ("shards-empty", '["shard-a", "shard-b"]', "[]", "shards"),
    ("shards-duplicate", '["shard-a", "shard-b"]', '["shard-a", "shard-b", "shard-a"]', "shards"),
    ("shards-empty-item", '["shard-a", "shard-b"]', '["shard-a", ""]', "shards"),
    ("shards-blank-item", '["shard-a", "shard-b"]', '["shard-a", "  "]', "shards"),
    ("shards-nonstring", '["shard-a", "shard-b"]', '["shard-a", 7]', "shards"),
    ("default-unknown", 'default_shard = "shard-b"', 'default_shard = "missing"', "default_shard"),
    ("default-nonstring", 'default_shard = "shard-b"', 'default_shard = 7', "default_shard"),
    ("evidence-extra", '[evidence]', '[evidence]\nextra = "x"', "evidence"),
    ("evidence-missing", 'args = "pytest.args"\n', '', "evidence"),
    ("evidence-not-table", '[evidence]', '[evidence]', "evidence"),
    ("dir-absolute", 'dir = "governance/receipts/shard"', 'dir = "/tmp/shard"', "evidence.dir"),
    ("dir-parent", 'dir = "governance/receipts/shard"', 'dir = "governance/../shard"', "evidence.dir"),
    ("group-name-duplicate", 'name = "control"', 'name = "control"', "group"),
    ("group-shard-unknown", 'shard = "shard-a"', 'shard = "missing"', "group"),
    ("group-extra", 'name = "control"', 'name = "control"\nextra = "x"', "group"),
    ("group-missing", 'why = "共用控制組"\n', '', "group"),
    ("group-name-empty", 'name = "control"', 'name = ""', "group"),
    ("group-why-empty", 'why = "共用控制組"', 'why = ""', "group"),
    ("files-empty", '["tests/test_verify_shards.py"]', '[]', "files"),
    ("files-scalar", '["tests/test_verify_shards.py"]', '"tests/test_verify_shards.py"', "files"),
    ("files-nonstring", '["tests/test_verify_shards.py"]', '[7]', "files"),
    ("files-duplicate", '["tests/test_verify_shards.py"]', '["tests/test_verify_shards.py", "tests/test_verify_shards.py"]', "files"),
    ("files-outside", '["tests/test_verify_shards.py"]', '["governance/exit_codes.py"]', "files"),
    ("files-helper", '["tests/test_verify_shards.py"]', '["tests/conftest.py"]', "files"),
    ("files-extension", '["tests/test_verify_shards.py"]', f'["{Path("tests") / "test_probe.txt"}"]', "files"),
    ("files-parent", '["tests/test_verify_shards.py"]', f'["{Path("tests") / ".." / "test_verify_shards.py"}"]', "files"),
    ("files-absolute", '["tests/test_verify_shards.py"]', f'["{Path("/") / "tests" / "test_verify_shards.py"}"]', "files"),
    ("files-two-groups", 'name = "control"', 'name = "control"', "files"),
]


@pytest.mark.parametrize("case,old,new,field", BAD_FIELDS, ids=[item[0] for item in BAD_FIELDS])
def test_bad_registry_field(tmp_path: Path, case: str, old: str, new: str, field: str) -> None:
    text = REGISTRY.replace(old, new)
    if case == "evidence-not-table":
        text = REGISTRY.split("[evidence]")[0] + 'evidence = "x"\n' + "[[group]]" + REGISTRY.split("[[group]]")[1]
    if case in {"group-name-duplicate", "files-two-groups"}:
        name, file = ("control", "tests/test_shard_seal.py") if case == "group-name-duplicate" else ("other", "tests/test_verify_shards.py")
        text += f'\n[[group]]\nname = "{name}"\nshard = "shard-b"\nwhy = "另一群"\nfiles = ["{file}"]\n'
    if field == "files":
        field = "group[1].files" if case == "files-two-groups" else "group[0].files"
    with pytest.raises(ToolBroken, match="^" + re.escape(field)) as error:
        load_registry(write_registry(tmp_path, text))
    assert ":" in str(error.value)


@pytest.mark.parametrize("field", ["dir", "expected", "args", "junit", "manifest"])
@pytest.mark.parametrize("value", ['""', '"  "', '7'])
def test_evidence_requires_nonempty_strings(tmp_path: Path, field: str, value: str) -> None:
    lines = [f"{field} = {value}" if line.startswith(f"{field} =") else line for line in REGISTRY.splitlines()]
    with pytest.raises(ToolBroken, match=f"evidence.{field}"):
        load_registry(write_registry(tmp_path, "\n".join(lines)))


@pytest.mark.parametrize("field", ["expected", "args", "junit", "manifest"])
def test_evidence_filename_cannot_have_directory(tmp_path: Path, field: str) -> None:
    lines = [f'{field} = "nested/file"' if line.startswith(f"{field} =") else line for line in REGISTRY.splitlines()]
    with pytest.raises(ToolBroken, match=f"evidence.{field}"):
        load_registry(write_registry(tmp_path, "\n".join(lines)))


def test_real_registry_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_registry(root / "governance" / "verify-shards.toml")
    for group in registry.groups:
        for file in group.files:
            assert (root / file).is_file(), file


@pytest.mark.parametrize("text", [None, "not toml = ["])
def test_registry_unreadable(tmp_path: Path, text: str | None) -> None:
    path = tmp_path / "missing.toml" if text is None else write_registry(tmp_path, text)
    with pytest.raises(ToolBroken):
        load_registry(path)


def test_group_must_be_list(tmp_path: Path) -> None:
    text = REGISTRY.split('[evidence]')[0] + 'group = "x"\n[evidence]' + REGISTRY.split('[evidence]')[1].split('[[group]]')[0]
    with pytest.raises(ToolBroken, match='^group:'):
        load_registry(write_registry(tmp_path, text))


@pytest.mark.parametrize('field', ['dir', 'expected', 'args', 'junit', 'manifest'])
def test_each_evidence_key_is_required(tmp_path: Path, field: str) -> None:
    text = '\n'.join(line for line in REGISTRY.splitlines() if not line.startswith(f'{field} ='))
    with pytest.raises(ToolBroken, match='^evidence:'):
        load_registry(write_registry(tmp_path, text))
