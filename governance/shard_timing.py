"""分片共用的純分配器：整群最大先放、路徑平手、缺值用中位數。"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from statistics import median
from typing import cast

from governance.exit_codes import ToolBroken
from governance.verify_shards import Registry, unregistered_shard


def history_sha(event: Mapping[str, object]) -> str:
    """合併請求釘住目標提交；主線推送釘住推送前提交。"""
    pull = event.get('pull_request')
    base = pull.get('base') if isinstance(pull, dict) else None
    sha = base.get('sha') if isinstance(base, dict) else event.get('before')
    if not isinstance(sha, str) or re.fullmatch('[0-9a-f]{40}', sha) is None:
        raise ToolBroken(f'歷史提交不完整：{sha!r}')
    return sha


def fallback(sha: str, reason: str) -> dict[str, object]:
    """共同快照明記退回與原因，不能把壞資料冒充零秒。"""
    return {'timing_source': 'fallback', 'run_id': None, 'sha': sha, 'reason': reason, 'times': {}}


def select_run(rows: Sequence[Mapping[str, object]], sha: str, branch: str) -> str:
    """只選這顆提交已完成成功的主線推送；多跑取最早跑號，不取啟動時的最新一跑。"""
    ids = [row['databaseId'] for row in rows if row.get('headSha') == sha and row.get('headBranch') == branch
           and row.get('event') == 'push' and row.get('status') == 'completed' and row.get('conclusion') == 'success'
           and type(row.get('databaseId')) is int]
    if not ids:
        raise ToolBroken('該提交沒有已完成而且成功的主線驗證')
    return str(min(cast(list[int], ids)))


def first_parent_chain(commits: Sequence[Mapping[str, object]], sha: str, limit: int) -> list[str]:
    """從起點沿第一父鏈往回，最多 limit 顆；清單外的父提交就停，不猜。"""
    parents: dict[str, str | None] = {}
    for row in commits:
        raw = row.get('parents')
        first = raw[0] if isinstance(raw, list) and raw else None
        parent = first.get('sha') if isinstance(first, dict) else None
        if isinstance(row.get('sha'), str):
            parents[cast(str, row['sha'])] = parent if isinstance(parent, str) else None
    if sha not in parents:
        raise ToolBroken('提交清單沒有起點那一顆')
    chain = [sha]
    while len(chain) < limit and parents.get(chain[-1]) in parents:
        chain.append(cast(str, parents[chain[-1]]))
    return chain


def nearest_success(chain: Sequence[str], rows: Sequence[Mapping[str, object]], branch: str) -> str:
    """起點沒有成功主線驗證時用最近一顆有的祖先（#1059）；整條都沒有才交給退回。"""
    for sha in chain:
        try:
            select_run(rows, sha, branch)
        except ToolBroken:
            continue
        return sha
    raise ToolBroken(f'起點連祖先共 {len(chain)} 顆主線提交都沒有已完成而且成功的主線驗證')


def resolve_snapshot(event: Mapping[str, object], reader: Callable[[str], tuple[str, dict[str, float]]],
                     locate: Callable[[str], str] = lambda sha: sha) -> dict[str, object]:
    """歷史讀不到時產生整跑共用的退回決定；分片不再獨自碰網路。快照的提交記實際讀秒數的那一顆。"""
    base = history_sha(event)
    try:
        sha = locate(base)
        run_id, times = reader(sha)
        return validate_snapshot({'timing_source': 'history', 'run_id': run_id, 'sha': sha, 'reason': '', 'times': times})
    except (ToolBroken, OSError, ValueError) as exc:
        return fallback(base, str(exc))


def assign(files: Sequence[str], registry: Registry, times: Mapping[str, float]) -> dict[str, str]:
    """每群或單檔是一個單位；秒數相同按完整路徑，負載相同先比檔數、再比片名。"""
    if not files or len(set(files)) != len(files):
        raise ToolBroken('考卷清單不得空白或重複')
    if any(isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in times.values()):
        raise ToolBroken('歷史秒數必須有限且非負')
    known = [times[file] for file in sorted(files) if file in times]
    if not known:
        raise ToolBroken('歷史秒數沒有現存考卷')
    default = median(known)
    weights = {file: times.get(file, default) for file in files}
    grouped: set[str] = set()
    units: list[tuple[str, ...]] = []
    for group in registry.groups:
        # 舊 balance 群只是手動配平，並非共用前置；退回時才沿用。
        if group.name.startswith('balance-'):
            continue
        if not set(group.files) <= set(files) or grouped.intersection(group.files):
            raise ToolBroken(f'群組考卷缺失或重複：{group.name}')
        grouped.update(group.files)
        units.append(tuple(sorted(group.files)))
    units.extend((file,) for file in sorted(set(files) - grouped))
    loads = dict.fromkeys(sorted(registry.shards), 0.0)
    counts = dict.fromkeys(registry.shards, 0)
    assigned: dict[str, str] = {}
    for unit in sorted(units, key=lambda unit: (-sum(weights[file] for file in unit), unit)):
        shard = min(loads, key=lambda name: (loads[name], counts[name], name))
        assigned.update((file, shard) for file in unit)
        loads[shard] += sum(weights[file] for file in unit)
        counts[shard] += len(unit)
    return dict(sorted(assigned.items()))


def validate_snapshot(raw: object) -> dict[str, object]:
    """共用輸入必須完整；不准單片因壞快照自行退回而形成混用。"""
    keys = {'timing_source', 'run_id', 'sha', 'reason', 'times'}
    if not isinstance(raw, dict) or raw.keys() != keys:
        raise ToolBroken('耗時快照欄位不完整')
    if not isinstance(raw['sha'], str) or re.fullmatch('[0-9a-f]{40}', raw['sha']) is None:
        raise ToolBroken('耗時快照提交不完整')
    if not isinstance(raw['reason'], str) or not isinstance(raw['times'], dict):
        raise ToolBroken('耗時快照原因或秒數格式錯')
    if raw['timing_source'] == 'fallback':
        if raw['run_id'] is not None or raw['times'] or not raw['reason'].strip():
            raise ToolBroken('退回快照必須有原因且沒有歷史秒數')
    elif raw['timing_source'] == 'history':
        if not isinstance(raw['run_id'], str) or re.fullmatch('[1-9][0-9]*', raw['run_id']) is None or raw['reason']:
            raise ToolBroken('歷史快照必須有唯一跑號且沒有退回原因')
        if not raw['times'] or any(not isinstance(key, str) or isinstance(value, bool)
                                   or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                                   for key, value in raw['times'].items()):
            raise ToolBroken('歷史快照秒數必須有限且非負')
    else:
        raise ToolBroken('未知耗時來源')
    return cast(dict[str, object], raw)


def from_snapshot(files: Sequence[str], registry: Registry, raw: object) -> dict[str, str]:
    """五片與本機預覽共用；整跑退回才採原登記簿與路徑雜湊。"""
    snapshot = validate_snapshot(raw)
    if snapshot['timing_source'] == 'history':
        return assign(files, registry, cast(dict[str, float], snapshot['times']))
    assigned = {file: unregistered_shard(file, registry.shards) for file in files}
    for group in registry.groups:
        assigned.update((file, group.shard) for file in group.files)
    return dict(sorted(assigned.items()))


def assignment_hash(assigned: Mapping[str, str]) -> str:
    """整份分配摘要只由排序後的檔案與片名決定。"""
    return hashlib.sha256(json.dumps(dict(assigned), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
