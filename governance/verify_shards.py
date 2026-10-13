"""分片工具與未來收尾裁判共用的不可變登記簿；不依賴 status 產證據層。"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from governance.exit_codes import ToolBroken


@dataclass(frozen=True)
class Evidence:
    """每片證據的相對目錄與四個檔名。"""
    dir: str
    expected: str
    args: str
    junit: str
    manifest: str


@dataclass(frozen=True)
class Group:
    """互斥的考卷檔群；why 保留人核過的共片理由。"""
    name: str
    shard: str
    why: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class Registry:
    """schema 1；所有容器都是不可變的 tuple 或 frozen 資料物件。"""
    schema: int
    shards: tuple[str, ...]
    default_shard: str
    evidence: Evidence
    groups: tuple[Group, ...]


def _table(raw: object, field: str, keys: frozenset[str]) -> dict[str, object]:
    """表格的鍵必須恰好等於契約。"""
    if not isinstance(raw, dict) or raw.keys() != keys:
        raise ToolBroken(f"{field}: 預期鍵 {sorted(keys)!r}，實際 {raw!r}")
    return cast(dict[str, object], raw)


def _string(raw: object, field: str) -> str:
    """非空字串；禁止換行，避免文字證據一格變兩行。"""
    if not isinstance(raw, str) or not raw.strip() or '\n' in raw or '\r' in raw:
        raise ToolBroken(f"{field}: 預期非空單行字串，實際 {raw!r}")
    return raw


def _strings(raw: object, field: str) -> tuple[str, ...]:
    """非空、不重複的字串清單。"""
    if not isinstance(raw, list) or not raw:
        raise ToolBroken(f"{field}: 預期非空字串清單，實際 {raw!r}")
    items = tuple(_string(value, field) for value in raw)
    if len(set(items)) != len(items):
        raise ToolBroken(f"{field}: 不准重複，實際 {raw!r}")
    return items


def _member(raw: object, field: str, shards: tuple[str, ...]) -> str:
    """分片 id 必須出自唯一的 shards 清單。"""
    value = _string(raw, field)
    if value not in shards:
        raise ToolBroken(f"{field}: 必須在 shards {shards!r}，實際 {raw!r}")
    return value


def _relative(raw: str, field: str) -> PurePosixPath:
    """只認 repo 內的 POSIX 相對路徑，不許父目錄或反斜線。"""
    path = PurePosixPath(raw)
    if path.is_absolute() or '..' in path.parts or '\\' in raw or path == PurePosixPath('.'):
        raise ToolBroken(f"{field}: 預期 repo 內相對路徑，實際 {raw!r}")
    return path


def _evidence(raw: object) -> Evidence:
    """目錄可以分層，四個檔名不可以。"""
    table = _table(raw, 'evidence', frozenset({'dir', 'expected', 'args', 'junit', 'manifest'}))
    values = {key: _string(value, f'evidence.{key}') for key, value in table.items()}
    _relative(values['dir'], 'evidence.dir')
    for key in ('expected', 'args', 'junit', 'manifest'):
        value = values[key]
        if '/' in value or '\\' in value or value in {'.', '..'}:
            raise ToolBroken(f"evidence.{key}: 預期單一檔名，實際 {value!r}")
    return Evidence(**values)


def _files(raw: object, field: str) -> tuple[str, ...]:
    """每個檔必須在 tests/ 下，以 test_ 開頭且以 .py 結尾。"""
    files = _strings(raw, field)
    for file in files:
        path = _relative(file, field)
        if path.parts[0] != 'tests' or not path.name.startswith('test_') or path.suffix != '.py':
            raise ToolBroken(f"{field}: 預期 tests/ 底下的 test_*.py，實際 {file!r}")
    return files


def _groups(raw: object, shards: tuple[str, ...]) -> tuple[Group, ...]:
    """群名與檔案都互斥；空群清單可用補集覆蓋全套。"""
    if not isinstance(raw, list):
        raise ToolBroken(f"group: 預期群清單，實際 {raw!r}")
    groups: list[Group] = []
    names: set[str] = set()
    registered: set[str] = set()
    for index, value in enumerate(raw):
        field = f'group[{index}]'
        table = _table(value, field, frozenset({'name', 'shard', 'why', 'files'}))
        name = _string(table['name'], f'{field}.name')
        shard = _member(table['shard'], f'{field}.shard', shards)
        why = _string(table['why'], f'{field}.why')
        files = _files(table['files'], f'{field}.files')
        if name in names:
            raise ToolBroken(f"{field}.name: 群名重複，實際 {name!r}")
        if registered.intersection(files):
            raise ToolBroken(f"{field}.files: 同一檔在兩群，實際 {sorted(registered.intersection(files))!r}")
        names.add(name)
        registered.update(files)
        groups.append(Group(name, shard, why, files))
    return tuple(groups)


def load_registry(path: Path) -> Registry:
    """讀不開、TOML 不合法、形狀錯誤都由 ToolBroken 報成 2。"""
    try:
        raw = tomllib.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"registry {path}: 讀不開，實際 {exc}") from exc
    table = _table(raw, 'registry', frozenset({'schema', 'shards', 'default_shard', 'evidence', 'group'}))
    schema = table['schema']
    if type(schema) is not int or schema != 1:
        raise ToolBroken(f"schema: 預期整數 1，實際 {schema!r}")
    shards = _strings(table['shards'], 'shards')
    return Registry(schema, shards, _member(table['default_shard'], 'default_shard', shards),
                    _evidence(table['evidence']), _groups(table['group'], shards))
