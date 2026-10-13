"""在 repo 根獨立清點全套，再寫出本片的 @參數檔；不印裁判的判決行。"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pytest

from governance.exit_codes import TOOL_BROKEN, ToolBroken, note
from governance.verify_shards import Prepared, Registry, load_registry, render_inputs as render_inputs


# 環境帶進來的選項（可夾 -k、-o testpaths=…）與外掛（可在收集後悄悄刪題、不發 deselect）都會縮小全套。
ENVIRONMENT_OPTIONS = ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS')


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
    except (OSError, ImportError) as exc:  # ImportError：外掛載入失敗，pytest 不轉成回傳碼
        raise ToolBroken(f'pytest 全套清點起不來: {exc}') from exc
    if code != 0:
        raise ToolBroken(f'pytest 全套清點離開碼 {code}: {log.getvalue()[-2000:]}')
    if collector.deselected:
        raise ToolBroken(f'全套清點被篩掉 {collector.deselected} 題（-k／-m／--deselect 之類）：篩過的不是全套')
    return collector.nodeids


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
        for name in ENVIRONMENT_OPTIONS:
            if os.environ.get(name, '').strip():
                raise ToolBroken(f'{name}: 全套清點不收環境帶進來的選項或外掛，實際 {os.environ[name]!r}')
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
