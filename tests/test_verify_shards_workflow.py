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
