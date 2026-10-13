"""分片考完後封存來源與證據位元組，供下一步裁判核對；不印判決行。

schema 1 的 manifest 頂層為身分欄位，另有 expected／args／junit 三格；
每格是 {path: 登記的檔名, sha256: 小寫雜湊, bytes: 實際位元組數}。
workflow_path 保留 owner/repo/ 前綴，workflow_ref 只存 @ 後的 ref。
run_id 與 run_attempt 保留正整數字串，github_sha 與 tested_sha 分開記。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from governance.exit_codes import TOOL_BROKEN, ToolBroken, note
from governance.verify_shards import Registry, load_registry

ENV_KEYS = ('GITHUB_REPOSITORY', 'GITHUB_WORKFLOW_REF', 'GITHUB_EVENT_NAME', 'GITHUB_RUN_ID',
            'GITHUB_RUN_ATTEMPT', 'GITHUB_JOB', 'GITHUB_SHA')


def _identity(env: Mapping[str, str], head: Callable[[], str]) -> dict[str, object]:
    """環境與 HEAD 均由呼叫端提供，考卷不碰真環境或真 git。"""
    for key in ENV_KEYS:
        if not env.get(key, '').strip():
            raise ToolBroken(f'{key}: 必填非空環境變數，實際 {env.get(key)!r}')
    for key in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'):
        if re.fullmatch(r'[1-9][0-9]*', env[key]) is None:
            raise ToolBroken(f'{key}: 必須是正整數字串，實際 {env[key]!r}')
    tested_sha = head()
    for key, value in (('GITHUB_SHA', env['GITHUB_SHA']), ('tested_sha', tested_sha)):
        if re.fullmatch(r'[0-9a-f]{40}', value) is None:
            raise ToolBroken(f'{key}: 必須是 40 位小寫十六進位，實際 {value!r}')
    workflow = env['GITHUB_WORKFLOW_REF'].split('@')
    if len(workflow) != 2 or not all(workflow):
        raise ToolBroken(f'GITHUB_WORKFLOW_REF: 必須是 path@ref，實際 {env["GITHUB_WORKFLOW_REF"]!r}')
    return {'repository': env['GITHUB_REPOSITORY'], 'workflow_path': workflow[0], 'workflow_ref': workflow[1],
            'event': env['GITHUB_EVENT_NAME'], 'run_id': env['GITHUB_RUN_ID'],
            'run_attempt': env['GITHUB_RUN_ATTEMPT'], 'job': env['GITHUB_JOB'],
            'github_sha': env['GITHUB_SHA'], 'tested_sha': tested_sha}


def _file_evidence(out: Path, name: str) -> dict[str, object]:
    """對實際位元組封存，檔案不存在或讀不開不能留下假 manifest。"""
    path = out / name
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ToolBroken(f'evidence {path}: 讀不開，實際 {exc}') from exc
    return {'path': name, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


def seal(registry: Registry, shard: str, out: Path, *, env: Mapping[str, str], head: Callable[[], str]) -> None:
    """全部身分與三份證據驗好才建 manifest；已存在就回 2。"""
    if shard not in registry.shards:
        raise ToolBroken(f'shard: 未登記 {shard!r}')
    target = out / registry.evidence.manifest
    if target.exists():
        raise ToolBroken(f'manifest: 已經存在 {str(target)!r}')
    manifest: dict[str, object] = {'schema': 1, 'shard': shard, **_identity(env, head)}
    for key in ('expected', 'args', 'junit'):
        manifest[key] = _file_evidence(out, getattr(registry.evidence, key))
    try:
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    except OSError as exc:
        raise ToolBroken(f'manifest {target}: 寫不開，實際 {exc}') from exc


def git_head() -> str:
    """正式入口讀被測樹的 HEAD；子程序故障與非零都回 2。"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    except OSError as exc:
        raise ToolBroken(f'tested_sha: 取 HEAD 失敗，實際 {exc}') from exc
    if result.returncode != 0:
        raise ToolBroken(f'tested_sha: git rev-parse HEAD 回 {result.returncode}，實際 {result.stderr!r}')
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None,
         head: Callable[[], str] = git_head) -> int:
    """雲端傳環境，本機考卷可注入環境與 HEAD；兩者不隱藏在純函式裡。"""
    parser = argparse.ArgumentParser(description='封存分片證據來源與雜湊')
    parser.add_argument('--shard', required=True)
    parser.add_argument('--registry', type=Path, default=Path('governance/verify-shards.toml'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        root = Path.cwd()
        if not (root / 'pyproject.toml').is_file() or not (root / 'tests').is_dir():
            raise ToolBroken(f'cwd: 只准在 repo 根跑，實際 {root}')
        registry = load_registry(args.registry)
        out = args.out if args.out is not None else root / registry.evidence.dir
        seal(registry, args.shard, out, env=os.environ if env is None else env, head=head)
    except ToolBroken as exc:
        note(f'分片封存回 2（工具自壞）：{exc}')
        return TOOL_BROKEN
    return 0


if __name__ == '__main__':
    sys.exit(main())
