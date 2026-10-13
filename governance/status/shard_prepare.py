"""在 repo 根獨立清點全套，再寫出本片的 @參數檔；不印裁判的判決行。"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

from governance.exit_codes import TOOL_BROKEN, ToolBroken, note
from governance.verify_shards import Registry, load_registry


@dataclass(frozen=True)
class Prepared:
    """兩份 UTF-8 文字：全套 nodeid 與本片考卷檔，都排序且以單一換行結尾。"""
    expected: str
    args: str


class Collector:
    """只在收集完成時取 session.items；不依據收到的 JUnit 反推預期清單。"""
    def __init__(self, root: Path | None = None) -> None:
        self.nodeids: tuple[str, ...] = ()
        self.deselected = 0
        self.root = root

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        """清點不能悄悄採用上層 repo 的設定，也不准縮小收集面：篩過的清單兩片一樣、實跑也一樣少，收尾看不出來。"""
        config = session.config
        if self.root is not None and config.rootpath.resolve() != self.root.resolve():
            raise pytest.UsageError(f'只准在 repo 根清點：cwd={self.root}，pytest root={config.rootpath}')
        if config.args_source is not pytest.Config.ArgsSource.TESTPATHS:
            raise pytest.UsageError(f'全套清點只准走設定的 testpaths，實際來源 {config.args_source.name}、參數 {config.args!r}')
        if config.getoption('ignore') or config.getoption('ignore_glob'):
            raise pytest.UsageError(f"全套清點不准帶 --ignore／--ignore-glob：{config.getoption('ignore')!r} {config.getoption('ignore_glob')!r}")

    def pytest_deselected(self, items: Sequence[pytest.Item]) -> None:
        """-k、-m、--deselect、--sw 都走這個鉤子；記下來，清點完一律拒收。"""
        self.deselected += len(items)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """pytest 的原始 nodeid 是唯一來源，排序與分組留給純函式。"""
        self.nodeids = tuple(item.nodeid for item in session.items)


def collect_nodeids(*, runner: Callable[..., int] = pytest.main, root: Path | None = None) -> tuple[str, ...]:
    """行程內清點；捕捉 pytest 輸出，避免把全套代號灌進雲端日誌。"""
    collector = Collector(root)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = runner(['--collect-only', '-n', '0', '-q', '-p', 'no:cacheprovider'], plugins=[collector])
    except OSError as exc:
        raise ToolBroken(f'pytest 全套清點起不來: {exc}') from exc
    if code != 0:
        raise ToolBroken(f'pytest 全套清點離開碼 {code}: {log.getvalue()[-2000:]}')
    if collector.deselected:
        raise ToolBroken(f'全套清點被篩掉 {collector.deselected} 題（-k／-m／--deselect 之類）：篩過的不是全套')
    return collector.nodeids


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


def prepare(nodeids: Iterable[str], registry: Registry, shard: str, out: Path) -> Prepared:
    """自行建目錄；非空目錄與不可寫輸出回 2，絕不覆蓋舊證據。"""
    result = render_inputs(nodeids, registry, shard)
    try:
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise ToolBroken(f'out: 證據目錄非空 {str(out)!r}')
        for name, text in ((registry.evidence.expected, result.expected), (registry.evidence.args, result.args)):
            with (out / name).open('x', encoding='utf-8', newline='\n') as target:
                target.write(text)
    except (OSError, ValueError) as exc:  # ValueError：路徑裡有 NUL
        raise ToolBroken(f'out {out}: 寫不了證據，實際 {exc}') from exc
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """命令列只產證據；ToolBroken 由同一個退出碼約定回 2。"""
    parser = argparse.ArgumentParser(description='獨立清點全套與分片考卷')
    parser.add_argument('--shard', required=True)
    parser.add_argument('--registry', type=Path, default=Path('governance/verify-shards.toml'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        root = Path.cwd()
        if not (root / 'pyproject.toml').is_file() or not (root / 'tests').is_dir():
            raise ToolBroken(f'cwd: 只准在 repo 根跑，實際 {root}')
        if os.environ.get('PYTEST_ADDOPTS', '').strip():
            raise ToolBroken(f"PYTEST_ADDOPTS: 全套清點不收環境帶進來的選項，實際 {os.environ['PYTEST_ADDOPTS']!r}")
        registry = load_registry(args.registry)
        if args.shard not in registry.shards:
            raise ToolBroken(f'shard: 未登記 {args.shard!r}')
        out = args.out if args.out is not None else root / registry.evidence.dir
        prepare(collect_nodeids(root=root), registry, args.shard, out)
    except ToolBroken as exc:
        note(f'分片清點回 2（工具自壞）：{exc}')
        return TOOL_BROKEN
    return 0


if __name__ == '__main__':
    sys.exit(main())
