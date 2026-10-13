"""分片收尾的純裁判：違規回問題清單，讀不開或形狀不明拋 ToolBroken。

不讀真環境、不跑 git、不寫收據。呼叫端提供設定、收尾身分、登記簿、下載樹與已剖開的 workflow。
manifest 依封存契約，片段依下一版抄寫員契約；所有版本均由設定指定。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shlex
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import cast

from governance.checks.ci_command_segments import SEGMENT_SEP_RE, _command_lines, _command_text, _is_wrapped, _split_outside_quotes
from governance.exit_codes import ToolBroken
from governance.verify_shards import Registry, _relative, _string, _table, render_inputs, run_identity

TestKey = tuple[str, str]
IDENTITY_KEYS = frozenset({'repository', 'workflow_path', 'workflow_ref', 'event', 'run_id', 'run_attempt',
                           'job', 'github_sha', 'tested_sha'})
FILE_KEYS = ('expected', 'args', 'junit')
MANIFEST_KEYS = IDENTITY_KEYS | frozenset({'schema', 'shard', *FILE_KEYS})
FRAGMENT_KEYS = frozenset({'schema', 'name', 'argv', 'started', 'seconds', 'exit_code', 'signal', 'report',
                           'stdout_tail', 'stderr_tail', 'run_id', 'run_attempt', 'job'})


@dataclass(frozen=True)
class FinalizerSettings:
    """由卡設定注入；本庫沒有路徑、包名前綴或版本預設值。"""
    artifact_root: str
    artifact_prefix: str
    steps_subdir: str
    manifest_schema: int
    fragment_schema: int
    pytest_version: str
    wrapper_command: str

    def __post_init__(self) -> None:
        """錯設定不能變成空掃描面或逃出包根。"""
        for key in ('artifact_root', 'steps_subdir'):
            _relative(_string(getattr(self, key), f'settings.{key}'), f'settings.{key}')
        for key in ('artifact_prefix', 'pytest_version', 'wrapper_command'):
            _string(getattr(self, key), f'settings.{key}')
        if '/' in self.artifact_prefix or '\\' in self.artifact_prefix:
            raise ToolBroken(f'settings.artifact_prefix: 預期檔名前綴，實際 {self.artifact_prefix!r}')
        for key in ('manifest_schema', 'fragment_schema'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ToolBroken(f'settings.{key}: 預期正整數，實際 {value!r}')


@dataclass(frozen=True)
class Expected:
    """完整 nodeid 與不撞名的 junit 地址到原考卷檔對照。"""
    nodeids: tuple[str, ...]
    files: tuple[tuple[TestKey, str], ...]


def _problem(where: str, field: str, actual: object, wanted: object) -> str:
    """每條違規帶片、檔、格、實際值與預期值。"""
    return f'{where}.{field}={actual!r}，預期 {wanted!r}'


def _json(raw: bytes | str, where: str) -> object:
    """JSON 與 UTF-8 讀不懂就是工具自壞；重複鍵也不能被最後一格悄悄蓋掉。"""
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ToolBroken(f'{where}: JSON 鍵 {key!r} 重複，實際 {pairs!r}')
            result[key] = value
        return result
    try:
        return cast(object, json.loads(raw, object_pairs_hook=unique))
    except (ValueError, TypeError) as exc:
        raise ToolBroken(f'{where}: JSON 解不開，實際 {exc}') from exc


def needs_problems(raw: str, registry: Registry) -> list[str]:
    """上游逐片必須 success；缺／多片是違規，result 無法判讀是工具自壞。"""
    value = _json(raw, 'needs')
    if not isinstance(value, dict):
        raise ToolBroken(f'needs: 頂層須物件，實際 {value!r}')
    bad: list[str] = []
    for shard, row in value.items():
        if not isinstance(row, dict) or not isinstance(row.get('result'), str):
            raise ToolBroken(f'needs[{shard!r}].result: 須字串，實際 {row!r}')
        if row['result'] != 'success':
            bad.append(_problem(f'片 {shard} needs', 'result', row['result'], 'success'))
    if set(value) != set(registry.shards):
        bad.append(_problem('needs', '鍵集合', sorted(value), sorted(registry.shards)))
    return bad


def _no_symlink(path: Path, root: Path) -> None:
    """從 path 往上到 root（含）每一層的符號連結都拒收；root 以上是跑者的路徑，不歸證據管。"""
    try:
        for candidate in (path, *path.parents):
            if candidate.is_symlink():
                raise ToolBroken(f'{path}: 符號連結不准跟隨，實際 {candidate}')
            if candidate == root:
                break
    except (OSError, ValueError) as exc:
        raise ToolBroken(f'{path}: 檔案種類讀不開，實際 {exc}') from exc


def _kind(path: Path, query: Callable[[Path], bool]) -> bool:
    """查檔案種類讀不開（例如權限）不改成「不存在」或「不是目錄」。"""
    try:
        return query(path)
    except (OSError, ValueError) as exc:
        raise ToolBroken(f'{path}: 檔案種類讀不開，實際 {exc}') from exc


def _children(path: Path, root: Path) -> list[Path]:
    """缺目錄與不可讀目錄不改成空集合。"""
    _no_symlink(path, root)
    try:
        return sorted(path.iterdir())
    except (OSError, ValueError) as exc:
        raise ToolBroken(f'{path}: 目錄讀不開，實際 {exc}') from exc


def _read_bytes(path: Path, root: Path) -> bytes:
    """不可讀或連結的證據絕不替成空位元組。"""
    _no_symlink(path, root)
    try:
        return path.read_bytes()
    except (OSError, ValueError) as exc:
        raise ToolBroken(f'{path}: 檔案讀不開，實際 {exc}') from exc


def _walk(path: Path, root: Path) -> list[Path]:
    """只列舉，不跟連結；其他子樹仍要檢查連結與額外 XML。"""
    result: list[Path] = []
    for child in _children(path, root):
        _no_symlink(child, root)
        result.append(child)
        if _kind(child, Path.is_dir):
            result.extend(_walk(child, root))
    return result


def _evidence_relative(registry: Registry, settings: FinalizerSettings) -> PurePosixPath:
    """artifact 根上傳的是內容，所以去掉根前綴才能找分片證據。"""
    path = _relative(registry.evidence.dir, 'evidence.dir')
    try:
        return path.relative_to(PurePosixPath(settings.artifact_root))
    except ValueError as exc:
        raise ToolBroken(f'evidence.dir={registry.evidence.dir!r} 不在 artifact_root={settings.artifact_root!r} 底下') from exc


def _inventory(part: Path, relative: PurePosixPath, registry: Registry, shard: str) -> list[str]:
    """每包四個證據檔；包內其他 XML 一律紅，非 XML 的其他子樹不判。"""
    entries = _walk(part, part)
    evidence = part / relative
    wanted = {getattr(registry.evidence, key) for key in (*FILE_KEYS, 'manifest')}
    actual = {p.name for p in _children(evidence, part)} if _kind(evidence, Path.exists) else set()
    where = f'片 {shard} {evidence}'
    bad = [_problem(where, '缺證據', sorted(wanted - actual), sorted(wanted))] if wanted - actual else []
    if actual - wanted:
        bad.append(_problem(where, '多證據', sorted(actual - wanted), sorted(wanted)))
    for path in entries:
        if path.parent == evidence and path.name in wanted and not _kind(path, Path.is_file):
            raise ToolBroken(f'片 {shard} {path}: 證據須一般檔案，實際非檔案')
        if path.suffix == '.xml' and path != evidence / registry.evidence.junit:
            bad.append(_problem(f'片 {shard} {part}', '額外 XML', str(path.relative_to(part)), registry.evidence.junit))
    return bad


def part_dirs(download: Path, registry: Registry, run_id: str, run_attempt: str,
              settings: FinalizerSettings) -> tuple[dict[str, Path], list[str]]:
    """回傳本次各包根與違規；缺包不造假目錄，下載樹不可讀則拋 ToolBroken。"""
    relative = _evidence_relative(registry, settings)
    wanted = {f'{settings.artifact_prefix}{run_id}-{run_attempt}-{shard}': shard for shard in registry.shards}
    parts: dict[str, Path] = {}
    bad: list[str] = []
    for path in _children(download, download):
        _no_symlink(path, download)
        shard = wanted.get(path.name)
        if shard is None or not _kind(path, Path.is_dir):
            bad.append(_problem(str(download), '未登記的包', path.name, sorted(wanted)))
            if _kind(path, Path.is_dir):
                _walk(path, path)
            continue
        parts[shard] = path
        bad.extend(_inventory(path, relative, registry, shard))
    for name, shard in wanted.items():
        if shard not in parts:
            bad.append(_problem(f'片 {shard} {download}', '缺包', None, name) + '；請整個 workflow 重跑')
    return parts, bad


def finalizer_identity(env: Mapping[str, str], head: Callable[[], str]) -> dict[str, object]:
    """與封存共用唯一身分驗法；HEAD／GITHUB_SHA 不一致由出處裁判報違規。"""
    return run_identity(env, head)


def _schema(row: Mapping[str, object], field: str, wanted: int) -> None:
    """只認設定指定的一版；舊／新版本與 bool 均不可判。"""
    value = row['schema']
    if type(value) is not int or value != wanted:
        raise ToolBroken(f'{field}.schema: 預期整數 {wanted!r}，實際 {value!r}')


def _manifest(raw: bytes, where: str, settings: FinalizerSettings) -> dict[str, object]:
    """manifest 頂層與三份檔案格的鍵、型別全部嚴格核對。"""
    row = _table(_json(raw, where), where, MANIFEST_KEYS)
    _schema(row, where, settings.manifest_schema)
    for key in IDENTITY_KEYS | {'shard'}:
        _string(row[key], f'{where}.{key}')
    for key in FILE_KEYS:
        file = _table(row[key], f'{where}.{key}', frozenset({'path', 'sha256', 'bytes'}))
        for label in ('path', 'sha256'):
            _string(file[label], f'{where}.{key}.{label}')
        if type(file['bytes']) is not int or file['bytes'] < 0:
            raise ToolBroken(f'{where}.{key}.bytes: 預期非負整數，實際 {file["bytes"]!r}')
    return row


def _own_identity(own: Mapping[str, object]) -> dict[str, object]:
    """記憶體的收尾身分也要完整；原始環境的驗法仍只有 run_identity。"""
    identity = _table(dict(own), '收尾身分', IDENTITY_KEYS)
    for key in IDENTITY_KEYS:
        _string(identity[key], f'收尾身分.{key}')
    return identity


def provenance_problems(own: Mapping[str, object], shard: str, evidence: Path, registry: Registry,
                        settings: FinalizerSettings) -> list[str]:
    """每片身分必須等於收尾，job／shard 指向包名；三份證據重新算位元組。"""
    own = _own_identity(own)
    path = evidence / registry.evidence.manifest
    where = f'片 {shard} {path}'
    row = _manifest(_read_bytes(path, evidence), where, settings)
    bad: list[str] = []
    if own['tested_sha'] != own['github_sha']:
        bad.append(_problem(f'片 {shard} 收尾 HEAD', 'tested_sha', own['tested_sha'], own['github_sha']))
    for key in (*sorted(IDENTITY_KEYS), 'shard'):
        wanted = shard if key in {'job', 'shard'} else own[key]
        if row[key] != wanted:
            bad.append(_problem(where, key, row[key], wanted))
    for key in FILE_KEYS:
        filename = getattr(registry.evidence, key)
        content = _read_bytes(evidence / filename, evidence)
        actual = cast(dict[str, object], row[key])
        wanted_file = {'path': filename, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        for field, wanted in wanted_file.items():
            if actual[field] != wanted:
                bad.append(_problem(where, f'{key}.{field}', actual[field], wanted))
    return bad


def _fragment(raw: bytes, where: str, settings: FinalizerSettings) -> dict[str, object]:
    """下一版片段完整形狀；離開碼不是整數不能誤讀為零。"""
    row = _table(_json(raw, where), where, FRAGMENT_KEYS)
    _schema(row, where, settings.fragment_schema)
    for key in ('name', 'started', 'run_id', 'run_attempt', 'job'):
        _string(row[key], f'{where}.{key}')
    for key in ('argv', 'stdout_tail', 'stderr_tail'):
        value = row[key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ToolBroken(f'{where}.{key}: 預期字串清單，實際 {value!r}')
    if type(row['exit_code']) is not int:
        raise ToolBroken(f'{where}.exit_code: 預期整數，實際 {row["exit_code"]!r}')
    signal = row['signal']
    if signal is not None and (type(signal) is not int or signal < 1):
        raise ToolBroken(f'{where}.signal: 預期正整數或 null，實際 {signal!r}')
    seconds = row['seconds']
    # 整數不轉浮點（10**400 轉不過去會漏成 OverflowError）；浮點才核有限。
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0 or (
            isinstance(seconds, float) and not math.isfinite(seconds)):
        raise ToolBroken(f'{where}.seconds: 預期非負有限數，實際 {seconds!r}')
    if row['report'] is not None and not isinstance(row['report'], str):
        raise ToolBroken(f'{where}.report: 預期字串或 null，實際 {row["report"]!r}')
    return row


def _recorder_names(steps: object, shard: str, settings: FinalizerSettings) -> set[str]:
    """共用 ci 第⑤條的斷行、切段與開頭判讀；只從抄寫員自己的參數取 --name。"""
    if not isinstance(steps, list):
        raise ToolBroken(f'片 {shard} workflow.steps: 預期清單，實際 {steps!r}')
    names: set[str] = set()
    for index, step in enumerate(steps):
        where = f'片 {shard} workflow.steps[{index}]'
        if not isinstance(step, dict) or ('run' not in step and 'uses' not in step):
            raise ToolBroken(f'{where}: 預期 run／uses 物件，實際 {step!r}')
        if 'run' not in step:
            continue
        if not isinstance(step['run'], str):
            raise ToolBroken(f'{where}.run: 預期字串，實際 {step["run"]!r}')
        for line in _command_lines(step['run']):
            for command in _split_outside_quotes(_command_text(line), SEGMENT_SEP_RE):
                if not _is_wrapped(command, settings.wrapper_command):
                    continue
                name = _recorder_name(command, settings.wrapper_command, where)
                if name in names:
                    raise ToolBroken(f'{where}.name: 同名抄寫步驟無法區分，實際 {name!r}')
                names.add(name)
    return names


def _recorder_name(command: str, wrapper: str, where: str) -> str:
    """只剖 wrapper 之後、子程序 -- 之前；不把子程序裡的字樣當抄寫員。"""
    try:
        tokens = shlex.split(command[len(wrapper):])
    except ValueError as exc:
        raise ToolBroken(f'{where}: 抄寫員參數解不開，實際 {command!r}') from exc
    parent = tokens[:tokens.index('--')] if '--' in tokens else tokens
    names: list[str] = []
    for index, token in enumerate(parent):
        if token == '--name' and index + 1 < len(parent):
            names.append(parent[index + 1])
        elif token.startswith('--name='):
            names.append(token.partition('=')[2])
    if len(names) != 1 or re.fullmatch(r'[A-Za-z0-9_.-]+', names[0]) is None:
        raise ToolBroken(f'{where}.name: 預期一個合法 --name，實際 {names!r}，命令 {command!r}')
    return names[0]


def step_fragment_problems(steps: object, part: Path, shard: str, own: Mapping[str, object],
                           settings: FinalizerSettings) -> list[str]:
    """yaml 每個抄寫步驟都有恰好一片、身分正確、離開碼零的證據。"""
    own = _own_identity(own)
    wanted = _recorder_names(steps, shard, settings)
    directory = part / settings.steps_subdir
    _no_symlink(directory, part)
    paths = _children(directory, part) if _kind(directory, Path.exists) else []
    bad: list[str] = []
    found: set[str] = set()
    for path in paths:
        where = f'片 {shard} {path}'
        _no_symlink(path, part)
        if path.suffix != '.json' or not _kind(path, Path.is_file):
            raise ToolBroken(f'{where}: 片段須 JSON 一般檔案，實際 {path.name!r}')
        name = path.stem
        found.add(name)
        row = _fragment(_read_bytes(path, part), where, settings)
        for key, expected in (('name', name), ('exit_code', 0), ('run_id', own['run_id']),
                              ('run_attempt', own['run_attempt']), ('job', shard)):
            if row[key] != expected:
                bad.append(_problem(where, key, row[key], expected))
        if row['signal'] is not None:
            bad.append(_problem(where, 'signal', row['signal'], None) + f'；離開碼記 {row["exit_code"]!r}，被訊號殺掉不是跑完')
    if wanted - found:
        bad.append(_problem(f'片 {shard} {directory}', '缺片段', sorted(wanted - found), sorted(wanted)))
    if found - wanted:
        bad.append(_problem(f'片 {shard} {directory}', '多片段', sorted(found - wanted), sorted(wanted)))
    return bad


def _expected_nodes(raw: bytes, where: str) -> tuple[str, ...]:
    """清點契約是 UTF-8、有最後 LF、非空、不重複且排序的一行一題。"""
    try:
        text = raw.decode('utf-8')
    except (UnicodeError, AttributeError) as exc:
        raise ToolBroken(f'{where}: 預期 UTF-8 位元組，實際 {raw!r}') from exc
    if not text:
        raise ToolBroken(f'{where}: 清單是空的，實際 {raw!r}')
    if '\r' in text:
        raise ToolBroken(f'{where}: 不准 CR，實際 {raw!r}')
    if not text.endswith('\n'):
        raise ToolBroken(f'{where}: 結尾缺換行，實際 {raw!r}')
    nodes = tuple(text[:-1].split('\n'))
    if any(not node for node in nodes):
        raise ToolBroken(f'{where}: 不准空行，實際 {nodes!r}')
    if len(set(nodes)) != len(nodes):
        raise ToolBroken(f'{where}: 不准重複，實際 {nodes!r}')
    if nodes != tuple(sorted(nodes)):
        raise ToolBroken(f'{where}: 未排序，實際 {nodes!r}')
    return nodes


def _pytest_version(settings: FinalizerSettings) -> None:
    """私有換算函式只對人重核過的 pytest 版本算數。"""
    try:
        actual = metadata.version('pytest')
    except metadata.PackageNotFoundError as exc:
        raise ToolBroken('pytest: 版本讀不到，實際套件不存在') from exc
    except (OSError, ValueError) as exc:
        raise ToolBroken(f'pytest: 版本讀不到，實際 {exc}') from exc
    if actual != settings.pytest_version:
        raise ToolBroken(f'pytest.version: 預期 {settings.pytest_version!r}，實際 {actual!r}；升版須重核 nodeid 換算')


def expected_set(lists: Mapping[str, bytes], settings: FinalizerSettings, registry: Registry) -> Expected:
    """鍵集合等於登記片、各片逐位元組一致，再按 pytest 的同一條路換算並拒收撞名。"""
    _pytest_version(settings)
    if set(lists) != set(registry.shards):
        raise ToolBroken(f'expected: 鍵集合須全部登記片 {sorted(registry.shards)!r}，實際 {sorted(lists)!r}')
    first = next(iter(lists.values()))
    for shard, raw in lists.items():
        if raw != first:
            raise ToolBroken(f'片 {shard} expected: 兩片未逐位元組相同，實際 {raw!r}，基準 {first!r}')
    nodes = _expected_nodes(first, f'片 {tuple(lists)} expected')
    return Expected(nodes, _mapped_files(nodes))


def _mapped_files(nodes: tuple[str, ...]) -> tuple[tuple[TestKey, str], ...]:
    """classname 直接組合、只有 name 做 bin_xml_escape；對照 record_testreport。"""
    from _pytest.junitxml import bin_xml_escape, mangle_test_address
    mapped: dict[TestKey, str] = {}
    for node in nodes:
        file, separator, item = node.partition('::')
        if not separator or not item:
            raise ToolBroken(f'expected.nodeid: 預期考卷檔與題目，實際 {node!r}')
        names = mangle_test_address(node)
        key = ('.'.join(names[:-1]), bin_xml_escape(names[-1]))
        if key in mapped:
            raise ToolBroken(f'expected.nodeid: 換算撞名 {key!r}，實際 {node!r} 與檔 {mapped[key]!r}')
        mapped[key] = file
    return tuple(mapped.items())


def read_junit(raw: bytes, shard: str, filename: str) -> ET.Element:
    """記憶體的位元組剖 XML；未解開、沒有 testsuite、或 testsuite 外面還有節點都回 2。

    合併只抄 testsuite：根層的 failure／error／skipped 一抄就不見、套疊的 testsuite 會抄兩次，
    舊判準看不到原樣，所以這兩種形狀在合併前就判不出來。
    """
    where = f'片 {shard} {filename}'
    try:
        root = ET.fromstring(raw.decode('utf-8'))
    except (ValueError, ET.ParseError) as exc:
        raise ToolBroken(f'{where}: junit 解不開，實際 {exc}') from exc
    suites = _suites(root, where)
    outside = [] if root.tag == 'testsuite' else [child.tag for child in root if child.tag != 'testsuite']
    nested = any(inner is not suite for suite in suites for inner in suite.iter('testsuite'))
    if root.tag not in {'testsuites', 'testsuite'} or outside or nested:
        raise ToolBroken(f'{where}: 預期根是 testsuites 或 testsuite、testsuite 只有一層且外面沒有別的節點，'
                         f'實際根 {root.tag!r}、外面 {outside!r}、套疊 {nested}')
    return root


def _suites(root: ET.Element, where: str) -> list[ET.Element]:
    """記憶體樹也要有 testsuite，不把未知形狀當零題。"""
    suites = list(root.iter('testsuite'))
    if not suites:
        raise ToolBroken(f'{where}: 沒有 testsuite，實際 {ET.tostring(root)!r}')
    return suites


def _cases(root: ET.Element, where: str) -> tuple[Counter[TestKey], list[str]]:
    """逐題取多重集合；failure／error／skipped 逐片逐題先記，避免重複題遮住真正失敗、也讓合併後的跳過指得出是哪一片。

    收集期爆掉的檔，pytest 交 classname 是空字串的 testcase：那是紅（不明題＋error），不是讀不懂。
    """
    _suites(root, where)
    counts: Counter[TestKey] = Counter()
    bad: list[str] = []
    for case in root.iter('testcase'):
        classname = case.get('classname')
        if classname is None:
            raise ToolBroken(f'{where}.testcase.classname: 缺屬性，實際 {ET.tostring(case, encoding="unicode")!r}')
        key = (classname, _string(case.get('name'), f'{where}.testcase.name'))
        counts[key] += 1
        for tag, wanted in (('failure', '沒有失敗／錯誤'), ('error', '沒有失敗／錯誤'), ('skipped', '沒有跳過')):
            for child in case.iter(tag):
                bad.append(_problem(where, f'testcase{key!r}.{tag}', ET.tostring(child, encoding='unicode'), wanted))
    return counts, bad


def _coverage_inputs(expected: Expected, roots: Mapping[str, ET.Element], args: Mapping[str, bytes], registry: Registry,
                     settings: FinalizerSettings) -> tuple[dict[str, Counter[TestKey]], list[str]]:
    """逐片重算參數檔，逐題核所屬考卷檔；失敗訊息先於所有覆蓋率訊息。"""
    if set(roots) != set(registry.shards) or set(args) != set(registry.shards):
        raise ToolBroken(f'coverage: 必須提供全部登記片，實際 junit={sorted(roots)!r} args={sorted(args)!r}')
    counts: dict[str, Counter[TestKey]] = {}
    failures: list[str] = []
    problems: list[str] = []
    files = dict(expected.files)
    for shard in registry.shards:
        wanted = render_inputs(expected.nodeids, registry, shard).args.encode('utf-8')
        try:
            args[shard].decode('utf-8')
        except (UnicodeError, AttributeError) as exc:
            raise ToolBroken(f'片 {shard} {registry.evidence.args}: 預期 UTF-8 位元組，實際 {args[shard]!r}') from exc
        if args[shard] != wanted:
            problems.append(_problem(f'片 {shard} {registry.evidence.args}', '位元組', args[shard], wanted))
        counts[shard], failed = _cases(roots[shard], f'片 {shard} {registry.evidence.junit}')
        failures.extend(failed)
        allowed = set(wanted.decode('utf-8').splitlines()) & set(args[shard].decode('utf-8').splitlines())
        for key in counts[shard]:
            if key in files and files[key] not in allowed:
                problems.append(_problem(f'片 {shard} {registry.evidence.junit}', f'testcase{key!r}.不屬本片', files[key], sorted(allowed)))
    return counts, failures + problems


def coverage_problems(expected: Expected, roots: Mapping[str, ET.Element], args: Mapping[str, bytes], registry: Registry,
                      settings: FinalizerSettings) -> list[str]:
    """漏題、不明題、片內重複、跨片重複分開報，實跑篩題靠此關擋。"""
    counts, bad = _coverage_inputs(expected, roots, args, registry, settings)
    files = dict(expected.files)
    wanted = set(files)
    actual = set().union(*(set(part) for part in counts.values()))
    owner = {file: shard for shard in registry.shards
             for file in render_inputs(expected.nodeids, registry, shard).args.splitlines()}
    for key in sorted(wanted - actual):
        bad.append(_problem(f'片 {owner.get(files[key], registry.shards)} {registry.evidence.junit}', f'testcase{key!r}.漏題', [],
                            f'在 {files[key]} 恰好一次'))
    for shard, part in counts.items():
        where = f'片 {shard} {registry.evidence.junit}'
        for key, count in sorted(part.items()):
            if key not in wanted:
                bad.append(_problem(where, '不明題', key, sorted(wanted)))
            if count > 1:
                bad.append(_problem(where, f'testcase{key!r}.片內重複', count, '恰好一次'))
    for key in sorted(actual):
        shards = [shard for shard, part in counts.items() if key in part]
        if len(shards) > 1:
            bad.append(_problem(f'片 {shards!r} {registry.evidence.junit}', f'testcase{key!r}.跨片重複', shards, '僅一片'))
    return bad


def merged_receipt(roots: Mapping[str, ET.Element]) -> ET.Element:
    """複製原樣 testsuite 到新的 testsuites，不修改輸入，也不寫正式收據。"""
    if not roots:
        raise ToolBroken(f'junit: 缺分片樹，實際 {roots!r}')
    merged = ET.Element('testsuites')
    for shard, root in roots.items():
        for suite in _suites(root, f'片 {shard} junit'):
            merged.append(copy.deepcopy(suite))
    return merged


def part_attr_problems(roots: Mapping[str, ET.Element], filename: str) -> list[str]:
    """逐片的 tests 加總必須等於實際 testcase 數，不能讓兩片誤差互相抵消。"""
    bad: list[str] = []
    if not roots:
        raise ToolBroken(f'junit: 缺分片樹，實際 {roots!r}')
    for shard, root in roots.items():
        where = f'片 {shard} {filename}'
        declared = 0
        for suite in _suites(root, where):
            value = suite.get('tests')
            if value is None or re.fullmatch(r'[0-9]+', value) is None:
                raise ToolBroken(f'{where}.tests: 預期非負整數屬性，實際 {value!r}')
            try:
                declared += int(value)
            except ValueError as exc:
                raise ToolBroken(f'{where}.tests: 數字讀不懂，實際 {exc}') from exc
        actual = sum(1 for _ in root.iter('testcase'))
        if declared != actual:
            bad.append(_problem(where, 'tests', declared, actual))
    return bad
