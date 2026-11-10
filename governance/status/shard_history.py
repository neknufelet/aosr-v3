"""整跑只讀一次固定提交的成功收據，輸出不可分歧的耗時快照。"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from governance.exit_codes import TOOL_BROKEN, ToolBroken, note
from governance.shard_timing import first_parent_chain, history_sha, nearest_success, resolve_snapshot, select_run
from governance.status.shard_balance import read_timing_junits


def command(argv: list[str]) -> str:
    """只讀平台指令；網路、逾時與非零交給整跑退回處理。"""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolBroken(f'歷史收據平台讀取失敗：{exc}') from exc
    if result.returncode:
        raise ToolBroken(f'歷史收據平台讀取回 {result.returncode}：{result.stderr[:300]}')
    return result.stdout


# 起點那顆主線驗證沒成功時，連起點看幾顆（#1059）：11-10 主線往回第二十顆是 11-06 合的，約四天；再往前的秒數多半已經過時。
ANCESTOR_LIMIT = 20


def locate(sha: str, repository: str, branch: str, *, call: Callable[[list[str]], str] = command) -> str:
    """起點沒有成功主線驗證時，沿第一父鏈找最近一顆有的祖先；找不到交給整跑退回。"""
    commits = json.loads(call(['gh', 'api', f'repos/{repository}/commits?sha={sha}&per_page={ANCESTOR_LIMIT}']))
    rows = json.loads(call(['gh', 'run', 'list', '--repo', repository, '--workflow', 'verify.yml', '--branch', branch,
                            '--event', 'push', '--status', 'success', '--limit', '100',
                            '--json', 'databaseId,headSha,headBranch,event,status,conclusion']))
    if (not isinstance(commits, list) or any(not isinstance(row, dict) for row in commits)
            or not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)):
        raise ToolBroken('祖先提交或主線跑清單格式錯')
    return nearest_success(first_parent_chain(commits, sha, ANCESTOR_LIMIT), rows, branch)


def read_cloud(sha: str, repository: str, branch: str, files: tuple[str, ...], *,
               call: Callable[[list[str]], str] = command) -> tuple[str, dict[str, float]]:
    """跑號、嘗試號與分片清單全部固定；漏一片或失敗都不採用局部秒數。"""
    rows = json.loads(call(['gh', 'run', 'list', '--repo', repository, '--workflow', 'verify.yml', '--commit', sha,
                            '--branch', branch, '--event', 'push', '--status', 'success', '--limit', '100',
                            '--json', 'databaseId,headSha,headBranch,event,status,conclusion']))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ToolBroken('歷史收據跑清單格式錯')
    run_id = select_run(rows, sha, branch)
    endpoint = f'repos/{repository}/actions/runs/{run_id}'
    run = json.loads(call(['gh', 'api', endpoint]))
    if not isinstance(run, dict) or any(run.get(key) != value for key, value in
                                       [('head_sha', sha), ('head_branch', branch), ('event', 'push'),
                                        ('status', 'completed'), ('conclusion', 'success')]):
        raise ToolBroken('歷史收據那一跑已非該提交完成成功的主線推送')
    attempt = run.get('run_attempt')
    if type(attempt) is not int or attempt < 1:
        raise ToolBroken('歷史收據嘗試號格式錯')
    jobs = json.loads(call(['gh', 'api', f'{endpoint}/attempts/{attempt}/jobs?per_page=100']))
    if not isinstance(jobs, dict) or not isinstance(jobs.get('jobs'), list):
        raise ToolBroken('歷史收據工作清單格式錯')
    shards = sorted(row['name'] for row in jobs['jobs'] if isinstance(row, dict)
                    and isinstance(row.get('name'), str) and row['name'].startswith('shard-') and row.get('conclusion') == 'success')
    if not shards or len(set(shards)) != len(shards):
        raise ToolBroken('歷史收據缺成功分片或片名重複')
    with TemporaryDirectory(prefix='aosr-shard-history-') as temporary:
        root = Path(temporary)
        for shard in shards:
            out = root / 'download' / shard
            call(['gh', 'run', 'download', run_id, '--repo', repository,
                  '--name', f'receipts-{run_id}-{attempt}-{shard}', '--dir', str(out)])
            check_binding(out, run_id, attempt, shard)
            target = root / 'history' / shard
            target.mkdir(parents=True)
            try:
                shutil.copyfile(out / 'shard' / 'junit.xml', target / 'junit.xml')
            except OSError as exc:
                raise ToolBroken(f'歷史收據缺分片考卷：{shard}：{exc}') from exc
        return run_id, read_timing_junits(root / 'history', files)


def check_binding(out: Path, run_id: str, attempt: int, shard: str) -> None:
    """歷史秒數也必須是正式考卷結束那份位元組；合法但被改秒數的文件同樣拒收。"""
    try:
        row = json.loads((out / 'steps' / 'pytest.json').read_text(encoding='utf-8'))
        content = (out / 'shard' / 'junit.xml').read_bytes()
    except (OSError, ValueError) as exc:
        raise ToolBroken(f'歷史收據綁定讀不開：{exc}') from exc
    if (not isinstance(row, dict) or row.get('exit_code') != 0 or row.get('run_id') != run_id
            or row.get('run_attempt') != str(attempt) or row.get('job') != shard
            or not isinstance(row.get('output'), dict)):
        raise ToolBroken('歷史收據綁定身分不正確')
    output = row['output']
    if output.get('bytes') != len(content) or output.get('sha256') != hashlib.sha256(content).hexdigest():
        raise ToolBroken('歷史收據綁定位元組不同')


def main(argv: Sequence[str] | None = None) -> int:
    """歷史不可用回零並明記退回；事件或共同輸出通道壞掉則回二，不造假快照。"""
    argparse.ArgumentParser(description='固定本跑的共用歷史耗時快照').parse_args(argv)
    try:
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text(encoding='utf-8'))
        if not isinstance(event, dict):
            raise ToolBroken('事件必須是物件')
        repository = os.environ['GITHUB_REPOSITORY']
        branch = os.environ['AOSR_DEFAULT_BRANCH']
        files = tuple(sorted(path.as_posix() for path in Path('tests').rglob('test_*.py')))
        def reader(sha: str) -> tuple[str, dict[str, float]]:
            return read_cloud(sha, repository, branch, files, call=command)
        def ancestor(sha: str) -> str:
            return locate(sha, repository, branch, call=command)
        snapshot = resolve_snapshot(cast(dict[str, object], event), reader, ancestor)
        encoded = json.dumps(snapshot, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as stream:
            stream.write(f'snapshot={encoded}\n')
        note(f"耗時來源 {snapshot['timing_source']}（全片共用）：{snapshot['reason'] or snapshot['run_id']}")
        base = history_sha(cast(dict[str, object], event))
        if snapshot['timing_source'] == 'history' and snapshot['sha'] != base:
            note(f"起點 {base} 沒有已完成而且成功的主線驗證，秒數改用最近一顆有的祖先 {snapshot['sha']}")
    except (KeyError, OSError, ValueError, ToolBroken) as exc:
        note(f'共同耗時快照回 2（工具自壞）：{exc}')
        return TOOL_BROKEN
    return 0


if __name__ == '__main__':
    sys.exit(main())
