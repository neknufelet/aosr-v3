"""#928：只快取瀏覽器安裝新增的 Ubuntu 安裝包；apt 仍負責核對、下載與安裝。"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from urllib.parse import quote

from governance.exit_codes import CLEAN, VIOLATION, note

CACHE_SUBDIR = 'browser-apt-archives'
BASELINE_FILENAME = 'browser-apt-baseline.json'
ARCHIVES_DIR = Path('/var/cache/apt/archives')
DPKG_FORMAT = '${Package}\t${Architecture}\t${Version}\t${Status}\n'
NOT_INSTALLED = '-\t沒裝（archives 裡有這支安裝包，dpkg 沒裝上；命中時多半是快取放回的舊包這一跑用不到）'


def _required(value: str, field: str) -> str:
    if not value.strip():
        raise ValueError(f'缺少 {field}，不准使用較寬的快取鑰匙或猜測路徑')
    return value


def _run_command(argv: Sequence[str]) -> str:
    """唯一外部命令入口；考卷注入替身，不執行 sudo 或套件查詢。"""
    return subprocess.run(list(argv), check=True, capture_output=True, text=True).stdout


def cache_key(environ: Mapping[str, str], architecture: Callable[[], str],
              package_version: Callable[[str], str]) -> str:
    """環境與實裝依賴版本精確分家，任一格缺席就失敗。"""
    values = [
        _required(environ.get('ImageOS', ''), 'ImageOS'),
        _required(environ.get('ImageVersion', ''), 'ImageVersion'),
        _required(architecture(), 'CPU 架構'),
        _required(package_version('playwright'), 'playwright'),
    ]
    # 分隔符號用 /，值裡的 / 會編碼，避免不同欄位組合得到同一把鑰匙。
    return 'browser-apt-v1/' + '/'.join(quote(value, safe='') for value in values)


def restore(runner_temp: Path, archives: Path, run_command: Callable[[Sequence[str]], str]) -> None:
    """先排除 gmsh 等早先下載的包，再把相符快取交給 apt 的 archives。"""
    runner_temp.mkdir(parents=True, exist_ok=True)
    baseline = sorted(path.name for path in archives.glob('*.deb'))
    (runner_temp / BASELINE_FILENAME).write_text(json.dumps(baseline), encoding='utf-8')
    cached = sorted((runner_temp / CACHE_SUBDIR).glob('*.deb'))
    if not cached:
        note('沒有快取，照常下載')
        return
    run_command(['sudo', 'cp', '--', *(str(path) for path in cached), str(archives)])
    note(f'放回了 {len(cached)} 支，{sum(path.stat().st_size for path in cached)} 位元組')
    for path in cached:
        note(f'RESTORED\t{path.name}\t{path.stat().st_size}')


def _browser_archives(runner_temp: Path, archives: Path) -> list[Path]:
    baseline_path = runner_temp / BASELINE_FILENAME
    if not baseline_path.is_file():
        raise ValueError('缺少 restore 寫的 archives 清單，不能判定瀏覽器安裝包')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if not isinstance(baseline, list) or not all(isinstance(name, str) for name in baseline):
        raise ValueError('restore 的 archives 清單格式錯誤')
    packages = sorted(path for path in archives.glob('*.deb') if path.name not in baseline)
    if not packages:
        raise ValueError('安裝後 archives 沒有安裝包，快取存不了（可能是映像的 apt 設定會刪掉）')
    return packages


def _package_key(path: Path) -> tuple[str, str]:
    """deb 的標準檔名帶套件、版本、架構；字型這類不分架構的包，架構就是 all。"""
    fields = path.stem.rsplit('_', 2)
    if len(fields) != 3 or not all(fields):
        raise ValueError(f'無法從安裝包檔名辨認套件：{path.name}')
    package, _, architecture = fields
    return package, architecture


def _installed(run_command: Callable[[Sequence[str]], str]) -> dict[tuple[str, str], str]:
    """整份 dpkg 資料庫按（套件, 架構）查版本與狀態。

    不帶套件名：帶名字時只要一支沒裝，dpkg-query 就整條非零。命中那一跑，快取放回的舊包裡
    可能有一支這一跑已經用不到（同一版映像裡套件站換了相依），那不是安裝失敗。
    """
    installed: dict[tuple[str, str], str] = {}
    for row in run_command(['dpkg-query', '-W', f'-f={DPKG_FORMAT}']).splitlines():
        fields = row.split('\t')
        # 只核格數：資料庫裡跟瀏覽器無關的套件哪一格空著，不該讓這一步紅。
        if len(fields) != 4:
            raise ValueError(f'dpkg-query 傳回的一行不是套件、架構、版本、狀態四格：{row!r}')
        package, architecture, version, status = fields
        installed[(package, architecture)] = f'{version}\t{status}'
    if not installed:
        raise ValueError('dpkg-query 沒有傳回任何套件')
    return installed


def stash(runner_temp: Path, archives: Path, run_command: Callable[[Sequence[str]], str]) -> None:
    """只留瀏覽器安裝包，印排序的檔案與實裝版本，讓雲端兩跑逐行比。"""
    packages = _browser_archives(runner_temp, archives)
    wanted = sorted({_package_key(path) for path in packages})
    installed = _installed(run_command)
    cache = runner_temp / CACHE_SUBDIR
    cache.mkdir(parents=True, exist_ok=True)
    for stale in cache.glob('*.deb'):
        stale.unlink()
    for path in packages:
        shutil.copyfile(path, cache / path.name)
        note(f'DEB\t{path.name}\t{path.stat().st_size}')
    for package, architecture in wanted:
        note(f'PACKAGE\t{package}:{architecture}\t{installed.get((package, architecture), NOT_INSTALLED)}')


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         architecture: Callable[[], str] = platform.machine,
         package_version: Callable[[str], str] = metadata.version,
         archives: Path = ARCHIVES_DIR,
         run_command: Callable[[Sequence[str]], str] = _run_command) -> int:
    """三條命令共用明確失敗出口；錯誤交給抄寫員記錄，不重試或存空快取。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('key', 'restore', 'stash'))
    args = parser.parse_args(argv)
    env = os.environ if environ is None else environ
    try:
        if args.command == 'key':
            key = cache_key(env, architecture, package_version)
            output = Path(_required(env.get('GITHUB_OUTPUT', ''), 'GITHUB_OUTPUT'))
            with output.open('a', encoding='utf-8') as stream:
                stream.write(f'key={key}\n')
            note(f'快取鑰匙：{key}')
        else:
            runner_temp = Path(_required(env.get('RUNNER_TEMP', ''), 'RUNNER_TEMP'))
            if args.command == 'restore':
                restore(runner_temp, archives, run_command)
            else:
                stash(runner_temp, archives, run_command)
    except (OSError, ValueError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        detail = f'；{exc.stderr}' if isinstance(exc, subprocess.CalledProcessError) else ''
        note(f'apt archives {args.command} 失敗：{exc}{detail}')
        return VIOLATION
    return CLEAN


if __name__ == '__main__':
    raise SystemExit(main())
