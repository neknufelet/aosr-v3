"""#928：只快取瀏覽器安裝新增的 Ubuntu 安裝包；放回前逐支跟 apt 索引核雜湊，下載與安裝照舊交給 apt。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from urllib.parse import quote, unquote

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
    # GitHub 的主鑰匙沒有完全相符時，還會找開頭相同的鑰匙（官方文件 Cache key matching 一節），
    # 例如 1.63.0 會配到 1.63.0.post1 存的那份；尾巴補一段固定的 end，兩把不同的鑰匙就不會一把是另一把的開頭。
    return 'browser-apt-v1/' + '/'.join(quote(value, safe='') for value in values) + '/end'


def _index_sha256(path: Path, run_command: Callable[[Sequence[str]], str]) -> set[str]:
    """apt 索引裡這一版（套件、版本、架構都照檔名）登記的 SHA256；索引裡沒有這一版就是空的。"""
    package, version, architecture = _deb_fields(path)
    output = run_command(['apt-cache', 'show', '--', f'{package}:{architecture}={version}'])
    return {line.split(':', 1)[1].strip() for line in output.splitlines() if line.startswith('SHA256:')}


def restore(runner_temp: Path, archives: Path, run_command: Callable[[Sequence[str]], str]) -> None:
    """先記下 gmsh 等早先下載的包；快取裡的包逐支跟 apt 索引核過雜湊，才交給 apt 的 archives。

    apt 對 archives 裡已有的檔只比檔名與大小就直接用（apt 原始碼 apt-pkg 那一層的 acquire-item.cc，
    pkgAcqArchive::QueueNext），不核雜湊；所以雜湊在這裡核。對不上、或索引裡沒有這一版就不放回，
    逐支印出原因，apt 照常下載。
    """
    runner_temp.mkdir(parents=True, exist_ok=True)
    baseline = sorted(path.name for path in archives.glob('*.deb'))
    (runner_temp / BASELINE_FILENAME).write_text(json.dumps(baseline), encoding='utf-8')
    cached = sorted((runner_temp / CACHE_SUBDIR).glob('*.deb'))
    if not cached:
        note('沒有快取，照常下載')
        return
    usable: list[Path] = []
    for path in cached:
        try:
            expected = _index_sha256(path, run_command)
        except subprocess.CalledProcessError as exc:
            # 整個套件不在索引裡時 apt-cache 回 100（「沒有這一版」則回 0、什麼都不印）。
            note(f'SKIPPED\t{path.name}\t索引裡查不到（apt-cache 回 {exc.returncode}：{str(exc.stderr).strip()}）')
            continue
        if not expected:
            note(f'SKIPPED\t{path.name}\t索引裡沒有這一版')
        elif hashlib.sha256(path.read_bytes()).hexdigest() not in expected:
            note(f'SKIPPED\t{path.name}\t雜湊跟索引不同')
        else:
            usable.append(path)
    if not usable:
        note(f'快取裡 {len(cached)} 支都不放回，照常下載')
        return
    run_command(['sudo', 'cp', '--', *(str(path) for path in usable), str(archives)])
    note(f'放回了 {len(usable)} 支，{sum(path.stat().st_size for path in usable)} 位元組；'
         f'不放回 {len(cached) - len(usable)} 支')
    for path in usable:
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


def _deb_fields(path: Path) -> tuple[str, str, str]:
    """deb 的標準檔名是「套件_版本_架構.deb」；版本裡的 : 存檔時寫成 %3a，查索引要解回來。"""
    fields = path.stem.rsplit('_', 2)
    if len(fields) != 3 or not all(fields):
        raise ValueError(f'無法從安裝包檔名辨認套件：{path.name}')
    package, version, architecture = fields
    return package, unquote(version), architecture


def _package_key(path: Path) -> tuple[str, str]:
    """（套件, 架構）；字型這類不分架構的包，架構就是 all。"""
    package, _, architecture = _deb_fields(path)
    return package, architecture


def _installed(run_command: Callable[[Sequence[str]], str]) -> dict[tuple[str, str], tuple[str, str]]:
    """整份 dpkg 資料庫按（套件, 架構）查版本與狀態。

    不帶套件名：帶名字時只要一支沒裝，dpkg-query 就整條非零。命中那一跑，快取放回的舊包裡
    可能有一支這一跑已經用不到（同一版映像裡套件站換了相依），那不是安裝失敗。
    """
    installed: dict[tuple[str, str], tuple[str, str]] = {}
    for row in run_command(['dpkg-query', '-W', f'-f={DPKG_FORMAT}']).splitlines():
        fields = row.split('\t')
        # 跟瀏覽器無關的套件哪一格空著不管（瀏覽器那幾支在 stash 另核）；套件名空著就不是 dpkg 的一筆紀錄。
        if len(fields) != 4 or not fields[0]:
            raise ValueError(f'dpkg-query 傳回的一行不是套件、架構、版本、狀態四格：{row!r}')
        package, architecture, version, status = fields
        installed[(package, architecture)] = (version, status)
    if not installed:
        raise ValueError('dpkg-query 沒有傳回任何套件')
    return installed


def stash(runner_temp: Path, archives: Path, run_command: Callable[[Sequence[str]], str]) -> None:
    """只留瀏覽器安裝包，印排序的檔案與實裝版本，讓雲端兩跑逐行比。"""
    packages = _browser_archives(runner_temp, archives)
    installed = _installed(run_command)
    rows: list[str] = []
    for package, architecture in sorted({_package_key(path) for path in packages}):
        found = installed.get((package, architecture))
        if found is None:
            rows.append(f'{package}:{architecture}\t{NOT_INSTALLED}')
            continue
        version, status = found
        if not version or not status:
            raise ValueError(f'dpkg 對 {package}:{architecture} 傳回空的版本或狀態，版本紀錄不完整')
        rows.append(f'{package}:{architecture}\t{version}\t{status}')
    cache = runner_temp / CACHE_SUBDIR
    cache.mkdir(parents=True, exist_ok=True)
    for stale in cache.glob('*.deb'):
        stale.unlink()
    for path in packages:
        shutil.copyfile(path, cache / path.name)
        note(f'DEB\t{path.name}\t{path.stat().st_size}')
    for row in rows:
        note(f'PACKAGE\t{row}')


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
