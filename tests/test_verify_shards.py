"""分片登記簿的格式契約；壞輸入必須說出欄位與實際值。"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import re

import pytest

from governance.exit_codes import ToolBroken
from governance.verify_shards import load_registry

REGISTRY = '''schema = 2
shards = ["shard-a", "shard-b"]
unregistered = "path-hash"
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
    assert registry.schema == 2
    assert registry.shards == ("shard-a", "shard-b")
    assert registry.unregistered == "path-hash"
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
    ("top-extra", "schema = 2", "schema = 2\nextra = 7", "registry"),
    ("top-missing", "schema = 2\n", "", "registry"),
    ("schema-wrong", "schema = 2", "schema = 1", "schema"),
    ("schema-bool", "schema = 2", "schema = true", "schema"),
    ("schema-string", "schema = 2", 'schema = "1"', "schema"),
    ("shards-scalar", '["shard-a", "shard-b"]', '"shard-a"', "shards"),
    ("shards-empty", '["shard-a", "shard-b"]', "[]", "shards"),
    ("shards-duplicate", '["shard-a", "shard-b"]', '["shard-a", "shard-b", "shard-a"]', "shards"),
    ("shards-empty-item", '["shard-a", "shard-b"]', '["shard-a", ""]', "shards"),
    ("shards-blank-item", '["shard-a", "shard-b"]', '["shard-a", "  "]', "shards"),
    ("shards-nonstring", '["shard-a", "shard-b"]', '["shard-a", 7]', "shards"),
    ("unregistered-missing", 'unregistered = "path-hash"\n', '', "registry"),
    ("unregistered-unknown", 'unregistered = "path-hash"', 'unregistered = "missing"', "unregistered"),
    ("unregistered-nonstring", 'unregistered = "path-hash"', 'unregistered = 7', "unregistered"),
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
    ("expected-nul", 'expected = "expected.txt"', 'expected = "expected\\u0000.txt"', "evidence.expected"),
    ("dir-double-slash", 'dir = "governance/receipts/shard"', 'dir = "governance//receipts/shard"', "evidence.dir"),
    ("dir-dot-prefix", 'dir = "governance/receipts/shard"', 'dir = "./governance/receipts/shard"', "evidence.dir"),
    ("files-double-slash", '["tests/test_verify_shards.py"]', '["tests//test_verify_shards.py"]', "files"),
    ("files-dot-prefix", '["tests/test_verify_shards.py"]', '["./tests/test_verify_shards.py"]', "files"),
    ("files-trailing-slash", '["tests/test_verify_shards.py"]', '["tests/test_verify_shards.py/"]', "files"),
    ("files-nul", '["tests/test_verify_shards.py"]', '["tests/test_verify_shards\\u0000.py"]', "files"),
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


def test_registry_path_with_nul_is_tool_broken(tmp_path: Path) -> None:
    with pytest.raises(ToolBroken, match="讀不開"):
        load_registry(tmp_path / "bad\x00registry.toml")


def test_group_must_be_list(tmp_path: Path) -> None:
    text = REGISTRY.split('[evidence]')[0] + 'group = "x"\n[evidence]' + REGISTRY.split('[evidence]')[1].split('[[group]]')[0]
    with pytest.raises(ToolBroken, match='^group:'):
        load_registry(write_registry(tmp_path, text))


@pytest.mark.parametrize('field', ['dir', 'expected', 'args', 'junit', 'manifest'])
def test_each_evidence_key_is_required(tmp_path: Path, field: str) -> None:
    text = '\n'.join(line for line in REGISTRY.splitlines() if not line.startswith(f'{field} ='))
    with pytest.raises(ToolBroken, match='^evidence:'):
        load_registry(write_registry(tmp_path, text))


def test_unregistered_hash_matches_independent_digest() -> None:
    """抓摘要截取、端序、片序或編碼植錯。"""
    import hashlib
    from governance import verify_shards

    shards = ('shard-a', 'shard-b', 'shard-c')
    for file in (str(Path('tests') / 'test_new.py'), str(Path('tests') / 'engine' / 'test_new.py'), str(Path('tests') / 'test_新.py')):
        expected = shards[int.from_bytes(hashlib.sha256(file.encode('utf-8')).digest()[:8], 'big') % len(shards)]
        assert verify_shards.unregistered_shard(file, shards) == expected


def test_unregistered_hash_partitions_and_group_overrides(tmp_path: Path) -> None:
    """抓新檔全落同片、群登記失效與改名沿用舊片。"""
    import hashlib
    from governance.verify_shards import render_inputs

    files = [f'tests/test_probe_{index}.py' for index in range(64)]
    shards = ('shard-a', 'shard-b')
    expected = {file: shards[int.from_bytes(hashlib.sha256(file.encode()).digest()[:8], 'big') % len(shards)] for file in files}
    pinned = next(file for file in files if expected[file] == 'shard-b')
    registry = load_registry(write_registry(tmp_path, REGISTRY.replace('tests/test_verify_shards.py', pinned)))
    nodes = [f'{file}::test_one' for file in files]
    expected[pinned] = 'shard-a'
    for shard in shards:
        selected = set(render_inputs(nodes, registry, shard).args.splitlines())
        assert selected == {file for file, assigned in expected.items() if assigned == shard}
        assert selected
    original = next(file for file in files if file != pinned and expected[file] == 'shard-a')
    renamed = next(file for file in files if file != pinned and expected[file] == 'shard-b')
    renamed_nodes = [node for node in nodes if not node.startswith((original + '::', renamed + '::'))]
    renamed_nodes.append(renamed + '::test_one')
    assert original not in render_inputs(renamed_nodes, registry, 'shard-a').args.splitlines()
    assert renamed in render_inputs(renamed_nodes, registry, 'shard-b').args.splitlines()
