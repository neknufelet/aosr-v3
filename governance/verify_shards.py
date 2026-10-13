"""分片工具與未來收尾裁判共用的不可變登記簿；不依賴 status 產證據層。"""
from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Iterable, Mapping
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
    """非空字串；禁止換行（文字證據一格變兩行）與 NUL（拿去開檔會漏成未捕捉的 ValueError）。"""
    if not isinstance(raw, str) or not raw.strip() or any(char in raw for char in '\n\r\x00'):
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
    """只認 repo 內的標準 POSIX 相對路徑：不許父目錄、反斜線，也不許 `//`、`./`、結尾 `/` 這類會被正規化掉的拼法。"""
    path = PurePosixPath(raw)
    if (path.is_absolute() or '..' in path.parts or '\\' in raw or path == PurePosixPath('.')
            or str(path) != raw):
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
    except (OSError, ValueError) as exc:  # ValueError 含解碼錯、TOML 錯與路徑裡的 NUL
        raise ToolBroken(f"registry {path}: 讀不開，實際 {exc}") from exc
    table = _table(raw, 'registry', frozenset({'schema', 'shards', 'default_shard', 'evidence', 'group'}))
    schema = table['schema']
    if type(schema) is not int or schema != 1:
        raise ToolBroken(f"schema: 預期整數 1，實際 {schema!r}")
    shards = _strings(table['shards'], 'shards')
    return Registry(schema, shards, _member(table['default_shard'], 'default_shard', shards),
                    _evidence(table['evidence']), _groups(table['group'], shards))


ENV_KEYS = ('GITHUB_REPOSITORY', 'GITHUB_WORKFLOW_REF', 'GITHUB_EVENT_NAME', 'GITHUB_RUN_ID',
            'GITHUB_RUN_ATTEMPT', 'GITHUB_JOB', 'GITHUB_SHA')


def run_identity(env: Mapping[str, str], head: Callable[[], str]) -> dict[str, object]:
    """環境與 HEAD 均由呼叫端提供，考卷不碰真環境或真 git。"""
    for key in ENV_KEYS:
        if not env.get(key, '').strip():
            raise ToolBroken(f'{key}: 必填非空環境變數，實際 {env.get(key)!r}')
    for key in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'):
        if re.fullmatch(r'[1-9][0-9]*', env[key]) is None:
            raise ToolBroken(f'{key}: 必須是正整數字串，實際 {env[key]!r}')
    tested_sha = head()
    for key, value in (('GITHUB_SHA', env['GITHUB_SHA']), ('tested_sha', tested_sha)):
        if re.fullmatch(r'[0-9a-f]{40}', value) is None:
            raise ToolBroken(f'{key}: 必須是 40 位小寫十六進位，實際 {value!r}')
    # 路徑到 .yml／.yaml 為止、不含 @；ref 一律 refs/ 開頭，裡面可以有合法的 @（例如分支 review@format）。
    workflow = re.fullmatch(r'([^@\n]+\.ya?ml)@(refs/[^\n]+)', env['GITHUB_WORKFLOW_REF'])
    if workflow is None:
        raise ToolBroken(f'GITHUB_WORKFLOW_REF: 必須是 <workflow 路徑>.yml@refs/…，實際 {env["GITHUB_WORKFLOW_REF"]!r}')
    return {'repository': env['GITHUB_REPOSITORY'], 'workflow_path': workflow[1], 'workflow_ref': workflow[2],
            'event': env['GITHUB_EVENT_NAME'], 'run_id': env['GITHUB_RUN_ID'],
            'run_attempt': env['GITHUB_RUN_ATTEMPT'], 'job': env['GITHUB_JOB'],
            'github_sha': env['GITHUB_SHA'], 'tested_sha': tested_sha}


@dataclass(frozen=True)
class Prepared:
    """兩份 UTF-8 文字：全套 nodeid 與本片考卷檔，都排序且以單一換行結尾。"""
    expected: str
    args: str


def _collected_file(nodeid: str) -> str:
    """文字格式是一行一題；只接 pytest 的 tests/test_*.py::題目地址。"""
    if '\n' in nodeid or '\r' in nodeid:
        raise ToolBroken(f'nodeid 不准含換行: {nodeid!r}')
    file, separator, item = nodeid.partition('::')
    path = PurePosixPath(file)
    if (not separator or not item or path.is_absolute() or '..' in path.parts
            or not path.parts or path.parts[0] != 'tests' or not path.name.startswith('test_')
            or path.suffix != '.py' or '\\' in file):
        raise ToolBroken(f'nodeid 必須來自 tests/test_*.py 的題目: {nodeid!r}')
    return file


def render_inputs(nodeids: Iterable[str], registry: Registry, shard: str) -> Prepared:
    """不套分片篩選的全套 expected；群檔加預設片補集形成 pytest.args。"""
    if shard not in registry.shards:
        raise ToolBroken(f'shard: 未登記 {shard!r}，實際 shards={registry.shards!r}')
    nodes = tuple(nodeids)
    if not nodes:
        raise ToolBroken('nodeid: 全套清點是空集合')
    files = {_collected_file(nodeid) for nodeid in nodes}
    if len(set(nodes)) != len(nodes):
        raise ToolBroken('nodeid: 全套清點出現重複題目')
    registered = {file for group in registry.groups for file in group.files}
    missing = registered - files
    if missing:
        raise ToolBroken(f'group.files: 登記了收集不到的考卷 {sorted(missing)!r}')
    selected = {file for group in registry.groups if group.shard == shard for file in group.files}
    if shard == registry.default_shard:
        selected.update(files - registered)
    if not selected:
        raise ToolBroken(f'shard {shard!r}: 本片考卷集合是空的，空參數檔會跑全套')
    return Prepared('\n'.join(sorted(nodes)) + '\n', '\n'.join(sorted(selected)) + '\n')
