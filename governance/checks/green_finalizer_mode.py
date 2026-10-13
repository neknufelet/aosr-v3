"""green 收尾模式的設定、來源選擇與裁判編排；舊判樹函式由主檔注入。"""
from __future__ import annotations

import os
import re
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from governance.checks import green_shard_evidence as evidence
from governance.checks.acceptance_verdict_points_at_head import _toplevel
from governance.checks.ci_jobs_cannot_die_quietly import YAML_IMPORT_ERROR, _workflow
from governance.exit_codes import ToolBroken
from governance.loader import Card
from governance.verify_shards import ENV_KEYS, Registry, _relative, _string, _table, load_registry

EVIDENCE_KEYS = tuple(evidence.FinalizerSettings.__dataclass_fields__)
MODE_KEYS = ('finalizer_needs_env', 'fixture_finalizer_run_file', 'shard_registry', 'parts_dir')
JudgeReceipt = Callable[[Card, str, ET.Element], list[str]]


@dataclass(frozen=True)
class Settings:
    """所有座標與版本都來自被掃的卡，沒有默認判準。"""
    finalizer_needs_env: str
    fixture_finalizer_run_file: str
    shard_registry: str
    parts_dir: str
    evidence: evidence.FinalizerSettings


@dataclass(frozen=True)
class Run:
    """只含這次收尾的環境快照與已驗形狀的身分。"""
    env: Mapping[str, str]
    own: Mapping[str, object]


def read_settings(card: Card) -> Settings | None:
    """没有 settings 保持單機；表存在則每個鍵與型別嚴格核對。"""
    try:
        raw = tomllib.loads(card.source.read_bytes().decode('utf-8'))
    except (OSError, ValueError) as exc:
        raise ToolBroken(f'{card.source}: 卡讀不開，實際 {exc}') from exc
    if 'settings' not in raw:
        return None
    table = _table(raw['settings'], f'{card.source} settings 表', frozenset((*MODE_KEYS, *EVIDENCE_KEYS)))
    mode = {key: _string(table[key], f'settings.{key}') for key in MODE_KEYS}
    for key in MODE_KEYS[1:]:
        _relative(mode[key], f'settings.{key}')
    strings = {key: _string(table[key], f'settings.{key}') for key in EVIDENCE_KEYS if not key.endswith('_schema')}
    versions: dict[str, int] = {}
    for key in ('manifest_schema', 'fragment_schema'):
        value = table[key]
        if type(value) is not int or value < 1:
            raise ToolBroken(f'settings.{key}: 預期正整數，實際 {value!r}')
        versions[key] = value
    settings = evidence.FinalizerSettings(strings['artifact_root'], strings['artifact_prefix'], strings['steps_subdir'],
                                          versions['manifest_schema'], versions['fragment_schema'],
                                          strings['pytest_version'], strings['wrapper_command'])
    return Settings(**mode, evidence=settings)


def process_env() -> Mapping[str, str]:
    """唯一讀取行程環境的位置；樣本樹不會叫它。"""
    return dict(os.environ)


def git_head(scan_root: Path) -> str:
    """只讀被掃根的真 HEAD，git 不可用與非零都回 2。"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=scan_root, capture_output=True, text=True)
    except OSError as exc:
        raise ToolBroken(f'tested_sha: git rev-parse HEAD 叫不動，實際 {exc}') from exc
    if result.returncode != 0:
        raise ToolBroken(f'tested_sha: git rev-parse HEAD 回 {result.returncode}，實際 {result.stderr!r}')
    return result.stdout.strip()


def fixture_run(path: Path, settings: Settings) -> Run:
    """樣本 env 恰好列必要鍵；空 needs 留給 JSON 裁判，head 仍要合法。"""
    raw = evidence._json(evidence._read_bytes(path, path.parent), str(path))
    if not isinstance(raw, dict):
        raise ToolBroken(f'{path}: 頂層須物件，實際 {raw!r}')
    row = _table(raw, str(path), frozenset({'env', 'head'}))
    table = _table(row['env'], f'{path}.env', frozenset((*ENV_KEYS, settings.finalizer_needs_env)))
    env: dict[str, str] = {}
    for key, value in table.items():
        if not isinstance(value, str):
            raise ToolBroken(f'{path}.env.{key}: 預期字串，實際 {value!r}')
        env[key] = value
    head = row['head']
    if not isinstance(head, str) or re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise ToolBroken(f'{path}.head: 預期 40 位小寫十六進位，實際 {head!r}')
    return Run(env, evidence.finalizer_identity(env, lambda: head))


def resolve_run(scan_root: Path, settings: Settings) -> Run | None:
    """只有真 git 根採用環境；其他掃描根只採樣本檔，結果不隨跑者環境變化。"""
    path = scan_root / settings.fixture_finalizer_run_file
    present = evidence._kind(path, Path.exists) or evidence._kind(path, Path.is_symlink)
    top = _toplevel(scan_root)
    if top is not None and top == scan_root.resolve():
        if present:
            raise ToolBroken(f'這是真的 git 工作樹的根，卻放著樣本身分檔 {path}；樣本檔只准活在樣本樹')
        env = process_env()
        if settings.finalizer_needs_env in env:
            return Run(env, evidence.finalizer_identity(env, lambda: git_head(scan_root)))
        return None
    if present:
        return fixture_run(path, settings)
    return None


def workflow_jobs(scan_root: Path, own: Mapping[str, object], registry: Registry) -> dict[str, object]:
    """共用 ci 的 safe_load 讀法，依收尾身分去掉 repository 前綴後找逐片步驟。"""
    prefix = _string(own['repository'], 'repository') + '/'
    path = _string(own['workflow_path'], 'workflow_path')
    if not path.startswith(prefix):
        raise ToolBroken(f'workflow_path: 預期 repository 前綴 {prefix!r}，實際 {path!r}')
    rel = str(_relative(path[len(prefix):], 'workflow_path'))
    if YAML_IMPORT_ERROR:
        raise ToolBroken(f'workflow_path {rel}: YAML 剖析器不可用，實際 {YAML_IMPORT_ERROR}')
    jobs = _workflow(scan_root / rel, rel)['jobs']
    if not isinstance(jobs, dict):
        raise ToolBroken(f'{rel}.jobs: 預期表，實際 {jobs!r}')
    result: dict[str, object] = {}
    for shard in registry.shards:
        job = jobs.get(shard)
        if not isinstance(job, dict) or 'steps' not in job:
            raise ToolBroken(f'片 {shard} {rel}.jobs.{shard}.steps: 找不到分片工作與步驟，實際 {job!r}')
        result[shard] = job['steps']
    return result


def downloaded_parts(download: Path, registry: Registry, own: Mapping[str, object],
                     settings: evidence.FinalizerSettings) -> tuple[dict[str, Path], list[str]]:
    """下載目錄不在是逐片缺包紅；其他種類與連結是未知輸入。"""
    evidence._no_symlink(download, download)
    if not evidence._kind(download, Path.exists):
        return {}, [f'片 {shard} {download}: 缺包 {settings.artifact_prefix}{own["run_id"]}-{own["run_attempt"]}-{shard}；'
                    '請整個 workflow 重跑' for shard in registry.shards]
    if not evidence._kind(download, Path.is_dir):
        raise ToolBroken(f'{download}: parts_dir 須目錄，實際不是目錄')
    return evidence.part_dirs(download, registry, str(own['run_id']), str(own['run_attempt']), settings)


def part_problems(scan_root: Path, parts: Mapping[str, Path], registry: Registry, settings: Settings,
                  own: Mapping[str, object]) -> list[str]:
    """先核封存出處，再核每片 workflow 列的全部抄寫步驟片段。"""
    relative = evidence._evidence_relative(registry, settings.evidence)
    bad: list[str] = []
    for shard, part in parts.items():
        bad.extend(evidence.provenance_problems(own, shard, part / relative, registry, settings.evidence))
    steps = workflow_jobs(scan_root, own, registry)
    for shard, part in parts.items():
        bad.extend(evidence.step_fragment_problems(steps[shard], part, shard, own, settings.evidence))
    return bad


def receipt_problems(card: Card, parts: Mapping[str, Path], registry: Registry, settings: Settings,
                     judge_receipt: JudgeReceipt) -> list[str]:
    """原始位元組逐片入裁判；覆蓋與逐片對帳收齊，最後才合併給舊判準。"""
    relative = evidence._evidence_relative(registry, settings.evidence)
    roots = {shard: part / relative for shard, part in parts.items()}
    lists = {shard: evidence._read_bytes(root / registry.evidence.expected, root) for shard, root in roots.items()}
    expected = evidence.expected_set(lists, settings.evidence, registry)
    trees: dict[str, ET.Element] = {}
    args: dict[str, bytes] = {}
    for shard, root in roots.items():
        trees[shard] = evidence.read_junit(evidence._read_bytes(root / registry.evidence.junit, root), shard, registry.evidence.junit)
        args[shard] = evidence._read_bytes(root / registry.evidence.args, root)
    bad = evidence.coverage_problems(expected, trees, args, registry, settings.evidence)
    bad.extend(evidence.part_attr_problems(trees, registry.evidence.junit))
    name = f'{len(trees)} 片合併 {sorted(trees)} {registry.evidence.junit}'
    bad.extend(judge_receipt(card, name, evidence.merged_receipt(trees)))
    return bad


def judge(card: Card, scan_root: Path, settings: Settings, own_run: Run, judge_receipt: JudgeReceipt) -> list[str]:
    """編排次序固定；包清點有紅就停，之後各組違規全部收齊。"""
    registry = load_registry(scan_root / settings.shard_registry)
    bad = evidence.needs_problems(own_run.env[settings.finalizer_needs_env], registry)
    parts, inventory = downloaded_parts(scan_root / settings.parts_dir, registry, own_run.own, settings.evidence)
    bad.extend(inventory)
    if inventory:
        return bad
    bad.extend(part_problems(scan_root, parts, registry, settings, own_run.own))
    bad.extend(receipt_problems(card, parts, registry, settings, judge_receipt))
    formal = scan_root / card.junit_path
    if evidence._kind(formal, Path.exists) or evidence._kind(formal, Path.is_symlink):
        bad.append(f'卡 {card.id} {card.junit_path}: 收尾的正式收據在場；收尾不跑 pytest，不准帶入單機收據')
    return bad


def finalizer_problems(card: Card, scan_root: Path, judge_receipt: JudgeReceipt) -> list[str] | None:
    """None 是單機模式；空清單則是收尾判綠，兩者不混用。"""
    settings = read_settings(card)
    if settings is None:
        return None
    own_run = resolve_run(scan_root, settings)
    if own_run is None:
        return None
    return judge(card, scan_root, settings, own_run, judge_receipt)
