"""兩片各自綁定正式考卷結果；非考卷片段的輸出沒有判決權。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from governance.checks import green_must_be_real_green as subject
from governance.exit_codes import ToolBroken
from tests.test_green_finalizer_mode import control as control
from tests.test_green_finalizer_mode import files
from tests.test_green_finalizer_inputs import part_path

BINDING = '正式考卷那一跑結束時的 junit 跟封存的不是同一份：中間有別的步驟重跑或改寫'


def output_path(root: Path, shard: str) -> Path:
    return part_path(root, shard) / 'steps' / 'pytest.json'


def recorded_output(root: Path, shard: str) -> dict[str, object]:
    raw = (part_path(root, shard) / 'shard' / 'junit.xml').read_bytes()
    return {'path': str(Path('governance') / 'receipts' / 'shard' / 'junit.xml'), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
@pytest.mark.parametrize('fault', ['null', 'missing-key', 'sha-null', 'path', 'sha256', 'bytes', 'bytes-bool', 'bytes-float', 'extra-key'])
def test_each_shard_requires_exact_official_output(control: Path, shard: str, fault: str) -> None:
    path = output_path(control, shard)
    row = json.loads(path.read_text())
    output = recorded_output(control, shard)
    if fault == 'null':
        row['output'] = None
    else:
        if fault == 'missing-key':
            output.pop('bytes')
        elif fault == 'bytes-float':
            output['bytes'] = float(str(output['bytes']))
        elif fault == 'extra-key':
            output['extra'] = 'x'
        else:
            field, value = {'sha-null': ('sha256', None), 'path': ('path', 'other.xml'),
                            'sha256': ('sha256', '0' * 64), 'bytes': ('bytes', 0), 'bytes-bool': ('bytes', False)}[fault]
            output[field] = value
        row['output'] = output
    path.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert bad and all(f'片 {shard} {path}.pytest.output' in item and BINDING in item for item in bad), bad
    assert all(str(recorded_output(control, shard)['sha256']) in item for item in bad), bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_fragment_output_key_is_required(control: Path, shard: str) -> None:
    path = output_path(control, shard)
    row = json.loads(path.read_text())
    row.pop('output', None)
    path.write_text(json.dumps(row))
    with pytest.raises(ToolBroken, match="缺 \\['output'\\]"):
        subject.check(control, files(control))


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
@pytest.mark.parametrize('output', [None, {'path': 'unrelated', 'sha256': None, 'bytes': None}])
def test_non_pytest_output_has_no_verdict_authority(control: Path, shard: str, output: object) -> None:
    for path in (part_path(control, shard) / 'steps').glob('*.json'):
        row = json.loads(path.read_text())
        row['output'] = recorded_output(control, shard) if path.name == 'pytest.json' else output
        path.write_text(json.dumps(row))
    assert subject.check(control, files(control)) == []


def test_binding_uses_registry_paths(control: Path) -> None:
    from governance.checks import green_finalizer_mode as mode
    from governance.checks import green_shard_evidence as evidence
    from governance.loader import load_card
    from governance.verify_shards import load_registry
    card = load_card(control / 'governance' / 'rules' / 'minimal-card.toml', control)
    settings = mode.read_settings(card)
    assert settings is not None
    registry = load_registry(control / settings.shard_registry)
    registry = replace(registry, evidence=replace(registry.evidence, dir=str(Path('governance') / 'receipts' / 'custom'), junit='result.xml'))
    part = part_path(control, 'shard-b')
    (part / 'shard').rename(part / 'custom')
    (part / 'custom' / 'junit.xml').rename(part / 'custom' / 'result.xml')
    path = output_path(control, 'shard-b')
    row = json.loads(path.read_text())
    raw = (part / 'custom' / 'result.xml').read_bytes()
    row['output'] = {'path': str(Path('governance') / 'receipts' / 'custom' / 'result.xml'), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    row['argv'] = ['uv', 'run', 'pytest', f'@{Path("governance") / "receipts" / "custom" / "pytest.args"}',
                   f'--junitxml={Path("governance") / "receipts" / "custom" / "result.xml"}']
    path.write_text(json.dumps(row))
    run = mode.fixture_run(control / settings.fixture_finalizer_run_file, settings)
    steps = [{'run': settings.evidence.wrapper_command + ' --name pytest --out-dir steps -- uv run pytest'}]
    for other in (part / 'steps').glob('*.json'):
        if other != path:
            other.unlink()
    assert evidence.step_fragment_problems(steps, part, 'shard-b', run.own, settings.evidence, registry) == []


def test_binding_message_comes_from_scanned_card(control: Path) -> None:
    from tests.test_green_finalizer_inputs import rewrite_setting
    message = '本題卡指定的結果綁定說明'
    rewrite_setting(control, 'junit_binding_message', json.dumps(message, ensure_ascii=False))
    path = output_path(control, 'shard-a')
    row = json.loads(path.read_text())
    row['output'] = None
    path.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert bad and all(message in item and BINDING not in item for item in bad), bad
