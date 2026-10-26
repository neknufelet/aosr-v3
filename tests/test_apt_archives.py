"""#928 的下載快取契約；所有套件命令用替身，檔案只寫 pytest 暫存樹。"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN
from governance.status.apt_archives import BASELINE_FILENAME, CACHE_SUBDIR, NOT_INSTALLED, _run_command, main


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, str]:
    return {'ImageOS': 'ubuntu24', 'ImageVersion': '20261025.1.0',
            'GITHUB_OUTPUT': str(tmp_path / 'output'), 'RUNNER_TEMP': str(tmp_path / 'runner')}


def installed_version(distribution: str) -> str:
    assert distribution == 'playwright'
    return '1.2.3'


def forbidden_command(argv: Sequence[str]) -> str:
    raise AssertionError(f'這條路不該叫套件命令：{argv}')


def invoke(command: str, environment: dict[str, str], archives: Path,
           run_command: Callable[[Sequence[str]], str] = forbidden_command) -> int:
    """不讓測試碰真 archives 或環境版本。"""
    return main([command], environ=environment, archives=archives,
                architecture=lambda: 'x86_64', package_version=installed_version,
                run_command=run_command)


def read_key(environment: dict[str, str]) -> str:
    return Path(environment['GITHUB_OUTPUT']).read_text(encoding='utf-8').splitlines()[-1]


def test_key_contains_environment_architecture_and_installed_version(
    environment: dict[str, str], tmp_path: Path,
) -> None:
    assert invoke('key', environment, tmp_path) == CLEAN
    key = read_key(environment)
    assert key.startswith('key=')
    assert all(value in key for value in (*[environment[name] for name in ('ImageOS', 'ImageVersion')],
                                          'x86_64', installed_version('playwright')))
    assert invoke('key', environment, tmp_path) == CLEAN
    assert read_key(environment) == key


@pytest.mark.parametrize('field', ['ImageOS', 'ImageVersion', 'CPU 架構', 'playwright'])
@pytest.mark.parametrize('value', ['', '   '])
def test_key_missing_field_fails_with_its_name(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, value: str,
) -> None:
    if field in environment:
        environment[field] = value
    result = main(['key'], environ=environment, archives=tmp_path,
                  architecture=lambda: value if field == 'CPU 架構' else 'x86_64',
                  package_version=lambda _: value if field == 'playwright' else '1.2.3',
                  run_command=forbidden_command)
    assert result != CLEAN
    assert field in capsys.readouterr().err
    assert not Path(environment['GITHUB_OUTPUT']).exists()


@pytest.mark.parametrize('field', ['ImageOS', 'ImageVersion', 'CPU 架構', 'playwright'])
def test_key_changes_for_each_dependency_dimension(
    environment: dict[str, str], tmp_path: Path, field: str,
) -> None:
    assert invoke('key', environment, tmp_path) == CLEAN
    before = read_key(environment)
    if field in environment:
        environment[field] = 'changed'
    assert main(['key'], environ=environment, architecture=lambda: 'arm64' if field == 'CPU 架構' else 'x86_64',
                package_version=lambda _: '2.0' if field == 'playwright' else '1.2.3',
                run_command=forbidden_command) == CLEAN
    after = read_key(environment)
    assert after != before
    assert not after.startswith(before) and not before.startswith(after)


@pytest.mark.parametrize('field', ['ImageOS', 'ImageVersion', 'CPU 架構', 'playwright'])
def test_key_is_never_a_prefix_of_a_longer_value(
    environment: dict[str, str], tmp_path: Path, field: str,
) -> None:
    """GitHub 的主鑰匙沒完全相符時會找開頭相同的鑰匙：1.2.3 不准配到 1.2.3.post1 存的那份。"""
    keys = []
    for suffix in ('', '.post1'):
        env = dict(environment)
        if field in env:
            env[field] += suffix
        assert main(['key'], environ=env, archives=tmp_path,
                    architecture=lambda: 'x86_64' + (suffix if field == 'CPU 架構' else ''),
                    package_version=lambda _: '1.2.3' + (suffix if field == 'playwright' else ''),
                    run_command=forbidden_command) == CLEAN
        keys.append(read_key(env).removeprefix('key='))
    short, longer = keys
    assert short != longer
    assert not longer.startswith(short)


def test_key_uninstalled_playwright_fails(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    def missing(distribution: str) -> str:
        raise PackageNotFoundError(distribution)

    assert main(['key'], environ=environment, package_version=missing,
                architecture=lambda: 'x86_64', run_command=forbidden_command) != CLEAN
    assert 'playwright' in capsys.readouterr().err


@pytest.mark.parametrize('command,field', [('key', 'GITHUB_OUTPUT'), ('restore', 'RUNNER_TEMP'), ('stash', 'RUNNER_TEMP')])
def test_required_output_paths_are_not_guessed(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, field: str,
) -> None:
    del environment[field]
    assert invoke(command, environment, tmp_path) != CLEAN
    assert field in capsys.readouterr().err


@pytest.mark.parametrize('empty_directory', [False, True])
def test_restore_without_cache_records_baseline_and_continues(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str], empty_directory: bool,
) -> None:
    archives = tmp_path / 'archives'
    archives.mkdir()
    (archives / 'libglu1-mesa_1_amd64.deb').write_bytes(b'gmsh')
    runner = Path(environment['RUNNER_TEMP'])
    if empty_directory:
        (runner / CACHE_SUBDIR).mkdir(parents=True)
    assert invoke('restore', environment, archives) == CLEAN
    assert '沒有快取' in capsys.readouterr().err
    assert (runner / BASELINE_FILENAME).is_file()
    assert not (runner / CACHE_SUBDIR / BASELINE_FILENAME).exists()


def stanza(data: bytes) -> str:
    """apt-cache show 一節裡跟這支檢查有關的那一行。"""
    return f'Package: x\nSHA256: {hashlib.sha256(data).hexdigest()}\n'


class PackageCommands:
    """真複製檔案、假裝 sudo、apt 索引與 dpkg 資料庫；不執行外部命令。"""
    def __init__(self, database: str = (
        'zlib\tamd64\t2.0\tinstall ok installed\n'
        'zlib\ti386\t1.9\tinstall ok installed\n'
        'unrelated\tamd64\t\t\n'
        'fonts-a\tall\t1.0\tinstall ok installed\n'
    ), index: dict[str, str] | None = None, unknown: frozenset[str] = frozenset()) -> None:
        self.database = database
        self.index = {} if index is None else index
        self.unknown = unknown
        self.shows: list[str] = []
        self.copied: list[str] = []
        self.queries: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> str:
        if list(argv[:3]) == ['sudo', 'cp', '--']:
            for source in argv[3:-1]:
                shutil.copyfile(source, Path(argv[-1]) / Path(source).name)
                self.copied.append(Path(source).name)
            return ''
        if list(argv[:3]) == ['apt-cache', 'show', '--']:
            self.shows.append(argv[3])
            if argv[3] in self.unknown:
                raise subprocess.CalledProcessError(100, list(argv), stderr='E: No packages found\n')
            return self.index.get(argv[3], '')
        self.queries.append(tuple(argv))
        return self.database


def test_restore_and_stash_keep_only_browser_packages_and_sorted_versions(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    archives = tmp_path / 'archives'
    archives.mkdir()
    (archives / 'libglu1-mesa_1_amd64.deb').write_bytes(b'gmsh')
    cache = Path(environment['RUNNER_TEMP']) / CACHE_SUBDIR
    cache.mkdir(parents=True)
    cached = {'zlib_2.0_amd64.deb': b'zlib', 'fonts-a_1.0_all.deb': b'font'}
    for name, data in cached.items():
        (cache / name).write_bytes(data)
    commands = PackageCommands(index={'zlib:amd64=2.0': stanza(b'zlib'), 'fonts-a:all=1.0': stanza(b'font')})
    assert invoke('restore', environment, archives, commands) == CLEAN
    restored = capsys.readouterr().err
    assert f'放回了 {len(cached)} 支' in restored
    assert f'{sum(map(len, cached.values()))} 位元組' in restored
    for name, data in cached.items():
        assert name in restored
        assert (archives / name).read_bytes() == data
        (cache / name).unlink()
    assert commands.shows == ['fonts-a:all=1.0', 'zlib:amd64=2.0']
    (cache / 'stale_0_amd64.deb').write_bytes(b'stale')
    assert invoke('stash', environment, archives, commands) == CLEAN
    assert {path.name: path.read_bytes() for path in cache.glob('*.deb')} == cached
    log = capsys.readouterr().err
    assert 'libglu1-mesa' not in log
    # 不帶套件名查整份資料庫；同名別架構（i386）與無關的套件不印。格式照抄字面，不從產品匯入，兩邊才不會一起錯。
    assert commands.queries == [('dpkg-query', '-W', '-f=${Package}\t${Architecture}\t${Version}\t${Status}\n')]
    assert [line for line in log.splitlines() if '\t' in line] == [
        *[f'NOTE: DEB\t{name}\t{len(cached[name])}' for name in sorted(cached)],
        'NOTE: PACKAGE\tfonts-a:all\t1.0\tinstall ok installed',
        'NOTE: PACKAGE\tzlib:amd64\t2.0\tinstall ok installed',
    ]


def test_stash_reports_a_restored_package_this_run_did_not_install(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """命中時放回的舊包這一跑沒裝：照實印「沒裝」那一行，不把它當安裝失敗、也不假裝有版本。"""
    assert invoke('restore', environment, tmp_path) == CLEAN
    for name in ('fonts-a_1.0_all.deb', 'dropped_3.0_amd64.deb'):
        (tmp_path / name).write_bytes(b'deb')
    capsys.readouterr()
    assert invoke('stash', environment, tmp_path, PackageCommands()) == CLEAN
    rows = [line for line in capsys.readouterr().err.splitlines() if line.startswith('NOTE: PACKAGE')]
    assert rows == ['NOTE: PACKAGE\tdropped:amd64\t' + NOT_INSTALLED,
                    'NOTE: PACKAGE\tfonts-a:all\t1.0\tinstall ok installed']


@pytest.mark.parametrize('database,words', [
    ('', '沒有傳回任何套件'),
    # 一行好的加一行三格的：只有空的那道守門在時，這一題要紅。
    ('fonts-a\tall\t1.0\tinstall ok installed\nzlib\tamd64\t2.0\n', '四格'),
])
def test_stash_refuses_an_empty_or_malformed_package_database(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str], database: str, words: str,
) -> None:
    assert invoke('restore', environment, tmp_path) == CLEAN
    (tmp_path / 'fonts-a_1.0_all.deb').write_bytes(b'deb')
    capsys.readouterr()
    assert invoke('stash', environment, tmp_path, PackageCommands(database)) != CLEAN
    assert words in capsys.readouterr().err
    assert not (Path(environment['RUNNER_TEMP']) / CACHE_SUBDIR).exists()


def test_stash_collects_fresh_download_after_cache_miss(
    environment: dict[str, str], tmp_path: Path,
) -> None:
    assert invoke('restore', environment, tmp_path) == CLEAN
    name = 'fonts-a_1.0_all.deb'
    (tmp_path / name).write_bytes(b'downloaded')
    assert invoke('stash', environment, tmp_path, PackageCommands()) == CLEAN
    assert (Path(environment['RUNNER_TEMP']) / CACHE_SUBDIR / name).read_bytes() == b'downloaded'


@pytest.mark.parametrize('baseline_only', [False, True])
def test_stash_without_browser_archives_fails_instead_of_saving_empty_cache(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str], baseline_only: bool,
) -> None:
    if baseline_only:
        (tmp_path / 'libglu1-mesa_1_amd64.deb').write_bytes(b'gmsh')
    assert invoke('restore', environment, tmp_path) == CLEAN
    capsys.readouterr()
    assert invoke('stash', environment, tmp_path) != CLEAN
    assert '安裝後 archives 沒有安裝包，快取存不了（可能是映像的 apt 設定會刪掉）' in capsys.readouterr().err


def test_stash_without_restore_baseline_fails(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / 'fonts-a_1_all.deb').write_bytes(b'font')
    assert invoke('stash', environment, tmp_path) != CLEAN
    # 比整句：pytest 給的暫存資料夾名字裡本來就有 restore 這個字。
    assert '缺少 restore 寫的 archives 清單' in capsys.readouterr().err


@pytest.mark.parametrize('command', ['restore', 'stash'])
def test_failed_package_command_is_nonzero_and_reports_failure(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str,
) -> None:
    assert invoke('restore', environment, tmp_path) == CLEAN
    if command == 'restore':
        cache = Path(environment['RUNNER_TEMP']) / CACHE_SUBDIR
        cache.mkdir()
        (cache / 'fonts-a_1_all.deb').write_bytes(b'font')
    else:
        (tmp_path / 'fonts-a_1_all.deb').write_bytes(b'font')

    def failed(argv: Sequence[str]) -> str:
        if argv[0] == 'apt-cache':
            return stanza(b'font')
        raise subprocess.CalledProcessError(23, list(argv), stderr='package command failed')

    assert invoke(command, environment, tmp_path, failed) != CLEAN
    assert 'package command failed' in capsys.readouterr().err


def test_restore_puts_back_only_packages_whose_hash_matches_the_index(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """apt 對已有的檔只比檔名與大小：雜湊不同、索引裡沒有這一版、整個套件不在索引裡，都不放回、逐支印原因。"""
    archives = tmp_path / 'archives'
    archives.mkdir()
    cache = Path(environment['RUNNER_TEMP']) / CACHE_SUBDIR
    cache.mkdir(parents=True)
    for name in ('good_1%3a1.0_all.deb', 'tampered_1.0_amd64.deb', 'gone_1.0_amd64.deb', 'dropped_1.0_amd64.deb'):
        (cache / name).write_bytes(name.encode())
    commands = PackageCommands(index={'good:all=1:1.0': stanza(b'good_1%3a1.0_all.deb'),
                                      'tampered:amd64=1.0': stanza(b'other bytes')},
                               unknown=frozenset({'dropped:amd64=1.0'}))
    assert invoke('restore', environment, archives, commands) == CLEAN
    log = capsys.readouterr().err
    assert commands.copied == ['good_1%3a1.0_all.deb']
    assert sorted(path.name for path in archives.glob('*.deb')) == ['good_1%3a1.0_all.deb']
    assert 'SKIPPED\tdropped_1.0_amd64.deb\t索引裡查不到（apt-cache 回 100：E: No packages found）' in log
    assert 'SKIPPED\tgone_1.0_amd64.deb\t索引裡沒有這一版' in log
    assert 'SKIPPED\ttampered_1.0_amd64.deb\t雜湊跟索引不同' in log
    assert '放回了 1 支' in log and '不放回 3 支' in log


def test_restore_with_nothing_usable_downloads_as_usual(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    cache = Path(environment['RUNNER_TEMP']) / CACHE_SUBDIR
    cache.mkdir(parents=True)
    (cache / 'gone_1.0_amd64.deb').write_bytes(b'old')

    def index_only(argv: Sequence[str]) -> str:
        assert argv[0] == 'apt-cache', f'沒有能放回的包時不該叫 {argv[0]}'
        return ''

    assert invoke('restore', environment, tmp_path, index_only) == CLEAN
    assert '快取裡 1 支都不放回，照常下載' in capsys.readouterr().err
    assert not list(tmp_path.glob('*.deb'))


@pytest.mark.parametrize('database', [
    'fonts-a\tall\t\tinstall ok installed\n',
    'fonts-a\tall\t1.0\t\n',
])
def test_stash_refuses_a_browser_package_without_version_or_status(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str], database: str,
) -> None:
    """瀏覽器那幾支的版本或狀態是空的，就不算版本紀錄完成；跟瀏覽器無關的套件空著不管。"""
    assert invoke('restore', environment, tmp_path) == CLEAN
    (tmp_path / 'fonts-a_1.0_all.deb').write_bytes(b'deb')
    capsys.readouterr()
    assert invoke('stash', environment, tmp_path, PackageCommands('unrelated\tamd64\t\t\n' + database)) != CLEAN
    assert 'dpkg 對 fonts-a:all 傳回空的版本或狀態' in capsys.readouterr().err
    assert not (Path(environment['RUNNER_TEMP']) / CACHE_SUBDIR).exists()


def test_stash_refuses_a_row_without_package_name(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert invoke('restore', environment, tmp_path) == CLEAN
    (tmp_path / 'fonts-a_1.0_all.deb').write_bytes(b'deb')
    capsys.readouterr()
    database = 'fonts-a\tall\t1.0\tinstall ok installed\n\t\t\t\n'
    assert invoke('stash', environment, tmp_path, PackageCommands(database)) != CLEAN
    assert '四格' in capsys.readouterr().err


def test_stash_refuses_a_malformed_baseline(
    environment: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    runner = Path(environment['RUNNER_TEMP'])
    runner.mkdir()
    (runner / BASELINE_FILENAME).write_text(json.dumps({'a': 1}), encoding='utf-8')
    (tmp_path / 'fonts-a_1.0_all.deb').write_bytes(b'deb')
    assert invoke('stash', environment, tmp_path, PackageCommands()) != CLEAN
    assert 'restore 的 archives 清單格式錯誤' in capsys.readouterr().err


def test_real_command_boundary_raises_on_nonzero_and_returns_stdout() -> None:
    """產品那一支真的叫子程序：非零要丟出來（main 才回非零），成功時交回標準輸出。"""
    assert _run_command([sys.executable, '-c', 'print("ok")']) == 'ok\n'
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _run_command([sys.executable, '-c', 'import sys; sys.stderr.write("boom"); sys.exit(3)'])
    assert caught.value.stderr == 'boom'
