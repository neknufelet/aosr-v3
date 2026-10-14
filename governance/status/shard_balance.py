"""由離線雲端收據建議兩片配平；保留共用前置群，不做即時搶工。"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import tomllib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import cast
from xml.etree import ElementTree as ET

from governance.exit_codes import TOOL_BROKEN, ToolBroken, note
from governance.verify_shards import Group, Registry, _groups, load_registry, render_inputs, unregistered_shard


@dataclass(frozen=True)
class Sample:
    """一份讀得懂的 run×shard 收據。"""
    run: str
    shard: str
    cpu: str
    wall: float
    fixed: float
    total: float
    times: dict[str, float]


@dataclass(frozen=True)
class Balance:
    """現在、雜湊基準與建議的分片，以及計算來源與匯入提醒。"""
    shards: tuple[str, ...]
    weights: dict[str, float]
    fixed: dict[str, float]
    ratio: float
    actual: dict[str, str]
    hashed: dict[str, str]
    suggested: dict[str, str]
    accepted: tuple[tuple[str, str, str], ...]
    skipped: tuple[str, ...]
    unmapped: tuple[str, ...]
    unmeasured: tuple[str, ...]
    hints: tuple[tuple[str, str, tuple[str, ...]], ...]

    def predictions(self, assignments: dict[str, str]) -> dict[str, float]:
        """固定成本與逐題權重共同預測 job 完成時間。"""
        return _predictions(assignments, self.weights, self.fixed, self.ratio)


def _seconds(raw: object, field: str) -> float:
    """收據秒數必須是有限非負數，bool 不是秒數。"""
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not math.isfinite(raw) or raw < 0:
        raise ToolBroken(f'{field}: 預期有限非負秒數，實際 {raw!r}')
    return float(raw)


def _step(path: Path) -> dict[str, object]:
    """每步收據必須可讀，name 與檔名一致。"""
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ToolBroken(f'步驟收據 {path}: {exc}') from exc
    if not isinstance(raw, dict) or raw.get('name') != path.stem:
        raise ToolBroken(f'步驟收據 {path}: name 格式錯')
    tail = raw.get('stdout_tail')
    if isinstance(tail, list) and all(isinstance(line, str) for line in tail):
        raw['stdout_tail'] = '\n'.join(tail)
    elif not isinstance(tail, str):
        raise ToolBroken(f'步驟收據 {path}: name／stdout_tail 格式錯')
    return cast(dict[str, object], raw)


def _class_file(classname: str, modules: dict[str, str]) -> str | None:
    """取能映射到現存考卷的最長 classname 前綴。"""
    parts = classname.split('.')
    while parts:
        candidate = modules.get('.'.join(parts))
        if candidate is not None:
            return candidate
        parts.pop()
    return None


def _junit(path: Path, modules: dict[str, str]) -> tuple[float, dict[str, float], list[str]]:
    """同份收據中同檔所有 testcase 加總，未知檔仍計入比例分母。"""
    tree = ET.parse(path)
    times: dict[str, float] = {}
    unmapped: list[str] = []
    total = 0.0
    for case in tree.iter('testcase'):
        seconds = _seconds(float(case.get('time', '0')), f'{path} testcase.time')
        total += seconds
        classname = case.get('classname', '')
        file = _class_file(classname, modules)
        if file is None:
            unmapped.append(f'{path.parent.parent.name}/{path.parent.name}: {classname!r} 無法對回現存考卷')
        else:
            times[file] = times.get(file, 0.0) + seconds
    return total, times, unmapped


def _history(history: Path, registry: Registry, files: tuple[str, ...]) -> tuple[list[Sample], list[str], list[str]]:
    """讀全部機型；不能解讀的機器行與 junit 明列跳過原因。"""
    if not history.is_dir():
        raise ToolBroken(f'history: 目錄不存在 {history}')
    modules = {file.removesuffix('.py').replace('/', '.'): file for file in files}
    samples: list[Sample] = []
    skipped: list[str] = []
    unmapped: list[str] = []
    for run in sorted(history.iterdir()):
        if not run.is_dir():
            continue
        for shard in sorted(run.iterdir()):
            if not shard.is_dir():
                continue
            if shard.name not in registry.shards:
                raise ToolBroken(f'{run.name}/{shard.name}: 未登記的片')
            label = f'{run.name}/{shard.name}'
            try:
                pytest = _step(shard / 'steps' / 'pytest.json')
            except ToolBroken as exc:
                skipped.append(f'{label}: 機器行讀不到（{exc}）')
                continue
            machines = re.findall(r'^機器：([^；\n]+)；avx512f：[^\n]*$', cast(str, pytest['stdout_tail']), re.MULTILINE)
            if len(machines) != 1:
                skipped.append(f'{label}: 機器行缺或不唯一')
                continue
            try:
                total, times, unknown = _junit(shard / 'junit.xml', modules)
            except (OSError, ValueError, ET.ParseError, ToolBroken) as exc:
                skipped.append(f'{label}: junit 讀不開（{exc}）')
                continue
            wall = _seconds(pytest.get('seconds'), f'{label} pytest.seconds')
            fixed = sum(_seconds(_step(path).get('seconds'), f'{path}.seconds')
                        for path in sorted((shard / 'steps').glob('*.json')) if path.stem != 'pytest')
            samples.append(Sample(run.name, shard.name, machines[0], wall, fixed, total, times))
            unmapped.extend(unknown)
    return samples, skipped, unmapped


def _assignments(files: tuple[str, ...], registry: Registry, *, discard_balance: bool) -> dict[str, str]:
    """兩種基準都只呼叫登記簿共用的路徑雜湊函式。"""
    assigned = {file: unregistered_shard(file, registry.shards) for file in files}
    for group in registry.groups:
        if discard_balance and group.name.startswith('balance-'):
            continue
        for file in group.files:
            if file not in assigned:
                raise ToolBroken(f'group.files: 現存考卷缺 {file}')
            assigned[file] = group.shard
    return assigned


def _predictions(assignments: dict[str, str], weights: dict[str, float], fixed: dict[str, float], ratio: float) -> dict[str, float]:
    """調整內部共用與報告相同的預測式。"""
    return {shard: fixed[shard] + ratio * sum(weights[file] for file, owner in assignments.items() if owner == shard)
            for shard in fixed}


def _adjust(hashed: dict[str, str], weights: dict[str, float], fixed: dict[str, float], ratio: float, movable: set[str]) -> dict[str, str]:
    """每回合挑最大完成時間下降最多的一檔；至少下降一秒才搬。"""
    assigned = hashed.copy()
    while True:
        predicted = _predictions(assigned, weights, fixed, ratio)
        heavy = max(predicted, key=lambda shard: (predicted[shard], shard))
        light = min(predicted, key=lambda shard: (predicted[shard], shard))
        best_file: str | None = None
        best_gain = 0.0
        for file in sorted(movable):
            if assigned[file] != heavy:
                continue
            cost = ratio * weights[file]
            after = max(predicted[heavy] - cost, predicted[light] + cost)
            gain = predicted[heavy] - after
            if gain > best_gain:
                best_gain, best_file = gain, file
        if best_file is None or best_gain < 1:
            return assigned
        assigned[best_file] = light


def _module_paths(module: str, root: Path) -> tuple[str, ...]:
    """只解析 tests 範圍內存在的模組，含套件初始化檔。"""
    if module != 'tests' and not module.startswith('tests.'):
        return ()
    path = Path(*module.split('.'))
    candidates = (path.with_suffix('.py'), path / '__init__.py')
    return tuple(candidate.as_posix() for candidate in candidates if (root / candidate).is_file())


def _imports(file: str, root: Path) -> tuple[str, ...]:
    """AST 解析 import 與 from，含 from 套件 import 輔助模組。"""
    tree = ast.parse((root / file).read_text(encoding='utf-8'), filename=file)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.update(_module_paths(alias.name, root))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                parent = list(Path(file).parent.parts)
                module = '.'.join(parent[:len(parent) - node.level + 1] + ([module] if module else []))
            found.update(_module_paths(module, root))
            for alias in node.names:
                found.update(_module_paths(f'{module}.{alias.name}', root))
    return tuple(sorted(found))


def _hints(root: Path, movable: set[str], registry: Registry) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """廣度優先追共用前置，迴圈只走一次，每群留下最短可重播路徑。"""
    pinned = {file: group.name for group in registry.groups if not group.name.startswith('balance-') for file in group.files}
    cache: dict[str, tuple[str, ...]] = {}
    hints: list[tuple[str, str, tuple[str, ...]]] = []
    for start in sorted(movable):
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(start, (start,))])
        visited = {start}
        groups: set[str] = set()
        while queue:
            file, chain = queue.popleft()
            group = pinned.get(file)
            if group is not None and group not in groups:
                hints.append((start, group, chain))
                groups.add(group)
            if file not in cache:
                cache[file] = _imports(file, root)
            for dependency in cache[file]:
                if dependency not in visited:
                    visited.add(dependency)
                    queue.append((dependency, (*chain, dependency)))
    return tuple(hints)


def analyze(history: Path, registry_path: Path, root: Path, reference_cpu: str) -> Balance:
    """固定群保留，權重與固定成本只用相同處理器，比例使用全部型號。"""
    registry = load_registry(registry_path)
    if len(registry.shards) != 2:
        raise ToolBroken('配平工具只處理兩片')
    files = tuple(sorted(path.relative_to(root).as_posix() for path in (root / 'tests').rglob('test_*.py')))
    if not files:
        raise ToolBroken('現存考卷是空集合')
    samples, skipped, unmapped = _history(history, registry, files)
    reference = [sample for sample in samples if sample.cpu == reference_cpu]
    if not reference:
        raise ToolBroken(f'參考型號 {reference_cpu!r} 零份；' + '；'.join(skipped))
    fixed: dict[str, float] = {}
    for shard in registry.shards:
        values = [sample.fixed for sample in reference if sample.shard == shard]
        if not values:
            raise ToolBroken(f'參考型號在 {shard} 零份，固定成本不足')
        fixed[shard] = median(values)
    ratios = [sample.wall / sample.total for sample in samples if sample.total > 0]
    if not ratios:
        raise ToolBroken('逐題總和全是零，牆鐘比例不足')
    ratio = median(ratios)
    observations = {file: [sample.times[file] for sample in reference if file in sample.times] for file in files}
    weights = {file: median(values) if values else 0.0 for file, values in observations.items()}
    actual = _assignments(files, registry, discard_balance=False)
    hashed = _assignments(files, registry, discard_balance=True)
    pinned = {file for group in registry.groups if not group.name.startswith('balance-') for file in group.files}
    movable = set(files) - pinned
    return Balance(registry.shards, weights, fixed, ratio, actual, hashed,
                   _adjust(hashed, weights, fixed, ratio, movable),
                   tuple((sample.run, sample.shard, sample.cpu) for sample in reference),
                   tuple(skipped), tuple(unmapped), tuple(file for file, values in observations.items() if not values),
                   _hints(root, movable, registry))


BALANCE_WHY = '照 governance/status/shard_balance 依雲端歷史耗時印出的建議貼上；重算後整群換掉，改動走合併請求。'


def suggested_groups(result: Balance) -> tuple[Group, ...]:
    """跟雜湊落片不同的檔，按片名各成一個 balance 群（空群不列）；群裡不帶會過期的數字。"""
    groups = []
    for shard in result.shards:
        files = tuple(sorted(file for file, assigned in result.suggested.items()
                             if assigned == shard and assigned != result.hashed[file]))
        if files:
            groups.append(Group(f'balance-{shard}', shard, BALANCE_WHY, files))
    return tuple(groups)


def check_suggestion(result: Balance, registry: Registry, text: str) -> None:
    """只在記憶體裡驗印出來的那段原文：讀得回同一批群、跟固定群一起過登記簿的群規則、兩片互斥且合起來是全部考卷。不寫任何檔。"""
    try:
        parsed = tomllib.loads(text)
    except ValueError as exc:
        raise ToolBroken(f'建議原文讀不回來：{exc}') from exc
    if set(parsed) - {'group'}:
        raise ToolBroken(f'建議原文最上層只准有群，實際多了 {sorted(set(parsed) - {"group"})!r}')
    pasted = parsed.get('group', [])
    if not isinstance(pasted, list):
        raise ToolBroken(f'建議原文的 group 不是群清單：{pasted!r}')
    pinned = [{'name': group.name, 'shard': group.shard, 'why': group.why, 'files': list(group.files)}
              for group in registry.groups if not group.name.startswith('balance-')]
    groups = _groups(pinned + pasted, registry.shards)
    if groups[len(pinned):] != suggested_groups(result):
        raise ToolBroken('建議原文讀回來跟算出來的群不一樣')
    candidate = replace(registry, groups=groups)
    nodes = [f'{file}::test_balance_validation' for file in result.weights]
    parts = [set(render_inputs(nodes, candidate, shard).args.splitlines()) for shard in candidate.shards]
    if parts[0] & parts[1] or parts[0] | parts[1] != set(result.weights):
        raise ToolBroken('建議分片不互斥或未覆蓋全部考卷')


def check_registry_shape(source: str, registry: Registry) -> None:
    """建議只照一種貼法給：登記簿至少留一個固定群、群寫成 [[group]] 表；別種寫法回 2，請人手改。

    換掉 balance 群之後沒有固定群，檔裡就沒有 group 鍵、讀不回；群寫成根層一行清單（例如
    group = [...]）時，尾巴接不上 [[group]] 表。試接一張空的 [[group]] 表就知道是哪一種。
    """
    if not any(not group.name.startswith('balance-') for group in registry.groups):
        raise ToolBroken('登記簿沒有固定群：換掉 balance 群後檔裡就沒有 group 鍵，這種寫法不出建議，請手改登記簿')
    try:
        tomllib.loads(source + '\n[[group]]\n')
    except ValueError as exc:
        raise ToolBroken(f'登記簿的群沒有寫成 [[group]] 表（例如根層 group = [...]），檔尾接不上建議的群，'
                         f'這種寫法不出建議，請手改登記簿：{exc}') from exc


def suggestion_text(groups: tuple[Group, ...]) -> str:
    """建議貼上的群原文（TOML 的 [[group]] 表）；沒有群就是空字串。"""
    return '\n'.join(
        f'[[group]]\nname = {json.dumps(group.name)}\nshard = {json.dumps(group.shard)}\nwhy = {json.dumps(group.why, ensure_ascii=False)}\n'
        'files = [\n' + ''.join(f'  {json.dumps(file)},\n' for file in group.files) + ']\n'
        for group in groups)


def same_as_registry(registry: Registry, groups: tuple[Group, ...]) -> bool:
    """登記簿現有的 balance 群跟建議的是不是同一批（只比群名、片、檔；說明是人話不比）。"""
    def shape(items: Sequence[Group]) -> list[tuple[str, str, tuple[str, ...]]]:
        return sorted((group.name, group.shard, tuple(sorted(group.files))) for group in items)
    return shape([group for group in registry.groups if group.name.startswith('balance-')]) == shape(groups)


def format_report(result: Balance, reference_cpu: str, text: str = '', applied: bool = False) -> str:
    """完整中文報告；每個預測帶推測標記，尾巴是建議貼上的 balance 群原文。"""
    lines = [f'參考型號：{reference_cpu}', '算數的跑（權重與固定成本）：']
    lines.extend(f'  {run}/{shard} 型號：{cpu}' for run, shard, cpu in result.accepted)
    lines.append('跳過的與原因：')
    lines.extend(f'  {item}' for item in result.skipped or ('無',))
    lines.append('無法映射的 classname：')
    lines.extend(f'  {item}' for item in result.unmapped or ('無',))
    lines.append(f'牆鐘比例（全部型號中位數）：{result.ratio:.6f}')
    lines.append('每片固定成本（參考型號中位數）：')
    lines.extend(f'  {shard}：{result.fixed[shard]:.3f} 秒' for shard in result.shards)
    for title, assignments in [('照今天實際落片', result.actual), ('只用雜湊', result.hashed), ('建議', result.suggested)]:
        lines.append(f'{title}預測：')
        lines.extend(f'  {shard}：{seconds:.3f} 秒／{seconds / 60:.3f} 分鐘（推測）'
                     for shard, seconds in result.predictions(assignments).items())
    lines.append('相較雜湊搬動的檔與逐題權重：')
    moved = [file for file in sorted(result.weights) if result.suggested[file] != result.hashed[file]]
    lines.extend(f'  {file}：{result.hashed[file]} → {result.suggested[file]}，{result.weights[file]:.3f} 秒' for file in moved)
    if not moved:
        lines.append('  無')
    lines.append('未量的檔（權重記零）：')
    lines.extend(f'  {file}' for file in result.unmeasured or ('無',))
    lines.append('共用前置提醒（僅報告，不自動併群）：')
    lines.extend(f'  {file}：建議併入群 {group}（經 {" → ".join(chain)}）' for file, group, chain in result.hints)
    if not result.hints:
        lines.append('  無')
    lines.append(f'登記簿現有的 balance 群：{"跟建議相同，不用改" if applied else "跟建議不同，要換成下面這段"}')
    lines.append('建議貼上的 balance 群（換掉登記簿裡現有的 balance-* 群；改登記簿走合併請求，這支不改檔）：')
    return '\n'.join(lines) + '\n' + (text if text else '  無（登記簿裡不用 balance 群）\n')


def main(argv: Sequence[str] | None = None) -> int:
    """輸入壞掉或資料不足依治理退出碼回二。"""
    parser = argparse.ArgumentParser(description='依雲端歷史耗時提出兩台配平建議')
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=Path('governance/verify-shards.toml'))
    parser.add_argument('--reference-cpu', required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze(args.history, args.registry, Path.cwd(), args.reference_cpu)
        registry = load_registry(args.registry)
        check_registry_shape(args.registry.read_text(encoding='utf-8'), registry)
        groups = suggested_groups(result)
        text = suggestion_text(groups)
        check_suggestion(result, registry, text)
        sys.stdout.write(format_report(result, args.reference_cpu, text, same_as_registry(registry, groups)))
    except (ToolBroken, OSError, ValueError, SyntaxError) as exc:
        note(f'配平建議回 2（工具自壞）：{exc}')
        return TOOL_BROKEN
    return 0


if __name__ == '__main__':
    sys.exit(main())
