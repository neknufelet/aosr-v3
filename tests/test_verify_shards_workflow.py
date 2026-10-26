"""兩台分攤的 verify 契約：兩片與收尾同一份環境、每支治理檢查恰好跑一次、pytest 命令及收尾登記對帳。"""
from __future__ import annotations

import shlex
import tomllib
from collections import Counter
from pathlib import Path
from typing import cast

import yaml

from governance.loader import load_all_cards

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github' / 'workflows' / 'verify.yml'
SETUP_PREFIXES = ('actions/checkout@', 'astral-sh/setup-uv@')


def _jobs() -> dict[str, dict[str, object]]:
    """只讀版控設定，不執行 workflow 或寫進工作樹。"""
    workflow = cast(dict[object, object], yaml.safe_load(WORKFLOW.read_text(encoding='utf-8')))
    assert workflow['name'] == 'verify'
    return cast(dict[str, dict[str, object]], workflow['jobs'])


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    """工作步驟保留所有欄位，供逐格對帳。"""
    return cast(list[dict[str, object]], job['steps'])


def _recorded(job: dict[str, object]) -> dict[str, dict[str, object]]:
    """用抄寫員的名字找原步驟；不得同一工作內重複登記。"""
    result: dict[str, dict[str, object]] = {}
    for step in _steps(job):
        argv = shlex.split(str(step.get('run', '')))
        if '--name' in argv:
            name = argv[argv.index('--name') + 1]
            assert name not in result, f'重複抄寫步驟 {name}'
            result[name] = step
    return result


def _setup(job: dict[str, object]) -> list[dict[str, object]]:
    """checkout、setup-uv、系統函式庫與鎖定安裝那幾步（不含 if，收尾每步都帶 always）。"""
    picked = [step for step in _steps(job) if str(step.get('uses', '')).startswith(SETUP_PREFIXES)
              or str(step.get('run', '')).startswith(('sudo ', 'uv sync'))]
    return [{key: value for key, value in step.items() if key != 'if'} for step in picked]


def _shards() -> list[str]:
    registry = tomllib.loads((ROOT / 'governance/verify-shards.toml').read_text(encoding='utf-8'))
    return cast(list[str], registry['shards'])


def test_shards_and_finalizer_share_one_environment() -> None:
    """兩片清點的環境不一致時兩份清單會對不上而回 2；三個 job 用同一份 env 與同一份鎖定環境。"""
    jobs = _jobs()
    first, *others = _shards()
    for shard in others:
        assert jobs[shard]['env'] == jobs[first]['env']
        assert _setup(jobs[shard]) == _setup(jobs[first])
    assert jobs['verify']['env'] == jobs[first]['env']
    finalizer = _setup(jobs['verify'])
    assert [step for step in _setup(jobs[first]) if not str(step.get('run', '')).startswith('sudo ')] == finalizer


def test_every_verify_card_check_runs_in_exactly_one_shard() -> None:
    """掛在 verify 的每張卡，它的檢查在兩片之間恰好跑一次；綠燈卡只在收尾判。"""
    jobs = _jobs()
    shards = _shards()
    runs = [str(step.get('run', '')) for shard in shards for step in _steps(jobs[shard])]
    cards = [card for card in load_all_cards(ROOT) if card.job == 'verify']
    assert cards, '一張掛在 verify 的卡都沒讀到：讀法壞了，這一題就沒在考'
    for card in cards:
        module = f'-m {card.check_module} '
        where = sum(module in f'{run} ' for run in runs)
        if card.id == 'green-must-be-real-green':
            assert where == 0, '綠燈卡要在收尾判，不在分片'
            assert module in f"{_recorded(jobs['verify'])['green_must_be_real_green']['run']} "
        else:
            assert where == 1, f'{card.id} 在兩片共跑了 {where} 次'
    names: Counter[str] = Counter(name for shard in shards for name in _recorded(jobs[shard]))
    assert names['ruff'] == 1 and names['mirror_receipts'] == 1 and names['install_browser'] == 1


def test_receipt_cards_read_the_mirror_in_the_same_shard_and_browser_lives_in_a() -> None:
    """鏡像排在四張收據卡前面、同一片；瀏覽器只在跑瀏覽器考卷的 shard-a。"""
    jobs = _jobs()
    readers = ('receipt_schema_complete', 'receipt_authority_is_the_cloud_run', 'four_roles_different_actors',
               'identity_strings_generated')
    home = next(shard for shard in _shards() if 'mirror_receipts' in _recorded(jobs[shard]))
    order = list(_recorded(jobs[home]))
    assert all(order.index('mirror_receipts') < order.index(name) for name in readers)
    browser = list(_recorded(jobs['shard-a']))
    assert browser.index('install_browser') < browser.index('pytest')


def test_shard_pytest_and_finalizer_match_registries() -> None:
    jobs = _jobs()
    settings = tomllib.loads((ROOT / 'governance/rules/green-must-be-real-green.toml').read_text(encoding='utf-8'))
    registry = tomllib.loads((ROOT / 'governance/verify-shards.toml').read_text(encoding='utf-8'))
    evidence = registry['evidence']
    junit = f"{evidence['dir']}/{evidence['junit']}"
    args = f"{evidence['dir']}/{evidence['args']}"
    for shard in registry['shards']:
        records = _recorded(jobs[shard])
        argv = shlex.split(str(records['pytest']['run']))
        command = shlex.split(settings['settings']['pytest_command'])
        assert argv[argv.index('--') + 1:] == command + [f'@{args}', f'--junitxml={junit}']
        assert argv[argv.index('--output') + 1] == junit
        assert list(records).index('prepare') < list(records).index('pytest') < list(records).index('seal')
        assert not records['pytest'].get('env')
    ci = tomllib.loads((ROOT / 'governance/rules/ci-jobs-cannot-die-quietly.toml').read_text(encoding='utf-8'))
    finalizer = jobs['verify']
    registration = ci['settings']['finalizer_jobs']['verify']
    assert set(cast(list[str], finalizer['needs'])) == set(registry['shards'])
    assert finalizer['if'] == registration['job_if']
    assert all(step['if'] == registration['step_if'] for step in _steps(finalizer))
    verdict = _recorded(finalizer)['green_must_be_real_green']
    assert verdict['run'] == registration['verdict_run']
    assert cast(dict[str, str], verdict['env'])[registration['verdict_env_name']] == registration['verdict_env_value']


def test_browser_archive_cache_order_and_exact_restore() -> None:
    """#928：只在 a 片、安裝前放回，考卷前存好；命中仍跑原來的安裝命令。"""
    jobs = _jobs()
    steps = _steps(jobs['shard-a'])
    records = _recorded(jobs['shard-a'])
    restore = next(step for step in steps if step.get('uses') == 'actions/cache/restore@v4')
    save = next(step for step in steps if step.get('uses') == 'actions/cache/save@v4')
    sequence = [records['apt_archives_key'], restore, records['apt_archives_restore'],
                records['install_browser'], records['apt_archives_stash'], save]
    start = steps.index(sequence[0])
    assert steps[start:steps.index(save) + 1] == sequence
    assert steps.index(save) < steps.index(records['pytest'])
    key = records['apt_archives_key']
    assert key['id'] == 'apt_archives_key'
    assert restore['id'] == 'apt_archives_cache'
    for step in (restore, save):
        options = cast(dict[str, str], step['with'])
        assert options['key'] == '${{ steps.apt_archives_key.outputs.key }}'
        assert 'restore-keys' not in options
    assert save['with'] == restore['with']
    assert save['if'] == "steps.apt_archives_cache.outputs.cache-hit != 'true'"
    ci = tomllib.loads((ROOT / 'governance/rules/ci-jobs-cannot-die-quietly.toml').read_text(encoding='utf-8'))
    assert save['if'] == ci['settings']['allowed_required_step_ifs']['shard-a'][save['name']]
    assert records['install_browser']['run'] == (
        'uv run python -m governance.status.record_step --name install_browser '
        '--out-dir governance/receipts/steps -- uv run playwright install --with-deps --only-shell chromium'
    )
    for command in ('key', 'restore', 'stash'):
        step = records[f'apt_archives_{command}']
        assert step['run'] == (
            f'uv run python -m governance.status.record_step --name apt_archives_{command} '
            f'--out-dir governance/receipts/steps -- uv run python -m governance.status.apt_archives {command}'
        )
        assert 'if' not in step
    assert all('apt_archives' not in str(step) and 'actions/cache/' not in str(step)
               for step in _steps(jobs['shard-b']))


def test_browser_archive_cache_path_matches_module() -> None:
    """#928：快取 action 與模組必須讀寫同一個暫存子目錄。"""
    from governance.status.apt_archives import CACHE_SUBDIR

    steps = _steps(_jobs()['shard-a'])
    cache_steps = [step for step in steps if str(step.get('uses', '')).startswith('actions/cache/')]
    assert cache_steps
    for step in cache_steps:
        assert cast(dict[str, str], step['with'])['path'] == f'${{{{ runner.temp }}}}/{CACHE_SUBDIR}'


def test_browser_archive_records_are_registered_without_verdict_lines() -> None:
    """#928：實際抄寫名單與收據規矩同步，新增的套件紀錄不冒充治理判決。"""
    settings = tomllib.loads((ROOT / 'governance/rules/receipt-schema-complete.toml').read_text(encoding='utf-8'))
    names = set(settings['settings']['report_optional_names'])
    for command in ('key', 'restore', 'stash'):
        name = f'apt_archives_{command}'
        assert name in _recorded(_jobs()['shard-a'])
        assert name in names
