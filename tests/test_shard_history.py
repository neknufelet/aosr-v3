"""共同耗時快照的雲端邊界；只注入合成的平台回覆與暫存檔。"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from governance.status import shard_history as subject
from governance.exit_codes import ToolBroken

SHA = 'a' * 40
FILES = tuple(f'tests/test_{letter}.py' for letter in 'ab')


def transport(argv: list[str]) -> str:
    """固定成功一跑、兩片；模擬下載內容，不碰網路。"""
    if argv[:3] == ['gh', 'run', 'list']:
        return json.dumps([{'databaseId': 42, 'headSha': SHA, 'headBranch': 'main', 'event': 'push',
                            'status': 'completed', 'conclusion': 'success'}])
    if argv[:2] == ['gh', 'api']:
        if argv[2].endswith('/jobs?per_page=100'):
            return json.dumps({'jobs': [{'name': name, 'conclusion': 'success'} for name in ('shard-a', 'shard-b', 'verify')]})
        return json.dumps({'run_attempt': 2, 'head_sha': SHA, 'status': 'completed', 'conclusion': 'success',
                           'head_branch': 'main', 'event': 'push'})
    if argv[:3] == ['gh', 'run', 'download']:
        folder = Path(argv[argv.index('--dir') + 1]) / 'shard'
        folder.mkdir(parents=True)
        file = 'a' if 'shard-a' in argv[argv.index('--name') + 1] else 'b'
        (folder / 'junit.xml').write_text(f'<testsuite><testcase classname="tests.test_{file}" name="test_one" time="3"/></testsuite>')
        content = (folder / 'junit.xml').read_bytes()
        steps = folder.parent / 'steps'
        steps.mkdir()
        (steps / 'pytest.json').write_text(json.dumps({'exit_code': 0, 'run_id': '42', 'run_attempt': '2',
                                                      'job': f'shard-{file}', 'output': {'bytes': len(content),
                                                      'sha256': hashlib.sha256(content).hexdigest()}}))
        return ''
    raise AssertionError(argv)


def test_cloud_receipts_are_pinned_to_one_successful_attempt() -> None:
    assert subject.read_cloud(SHA, 'owner/repo', 'main', FILES, call=transport) == ('42', dict.fromkeys(FILES, 3.0))


@pytest.mark.parametrize('broken', ['network', 'download', 'failed-run', 'missing-shard'])
def test_partial_cloud_history_is_never_accepted(broken: str) -> None:
    def call(argv: list[str]) -> str:
        if broken == 'network':
            raise ToolBroken('網路失敗')
        if broken == 'download' and argv[:3] == ['gh', 'run', 'download']:
            raise ToolBroken('收據下載失敗')
        raw = transport(argv)
        if broken == 'failed-run' and argv[:2] == ['gh', 'api'] and '/jobs?' not in argv[2]:
            return raw.replace('"success"', '"failure"')
        if broken == 'missing-shard' and argv[:2] == ['gh', 'api'] and '/jobs?' in argv[2]:
            return json.dumps({'jobs': [{'name': 'verify', 'conclusion': 'success'}]})
        return raw
    with pytest.raises(ToolBroken):
        subject.read_cloud(SHA, 'owner/repo', 'main', FILES, call=call)


def test_cli_network_failure_writes_one_fallback_and_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    event = tmp_path / 'event.json'
    event.write_text(json.dumps({'before': SHA}))
    output = tmp_path / 'output'
    output.write_text('')
    for key, value in {'GITHUB_EVENT_PATH': str(event), 'GITHUB_OUTPUT': str(output),
                       'GITHUB_REPOSITORY': 'owner/repo', 'AOSR_DEFAULT_BRANCH': 'main'}.items():
        monkeypatch.setenv(key, value)
    def broken(argv: list[str]) -> str:
        raise ToolBroken('網路失敗')
    monkeypatch.setattr(subject, 'command', broken)
    monkeypatch.chdir(tmp_path)
    assert subject.main([]) == 0
    snapshot = json.loads(output.read_text().removeprefix('snapshot='))
    assert snapshot['timing_source'] == 'fallback'
    assert '網路失敗' in snapshot['reason']
    assert 'fallback' in capsys.readouterr().err


def test_valid_xml_with_changed_seconds_is_corrupt_history() -> None:
    def tamper(argv: list[str]) -> str:
        result = transport(argv)
        if argv[:3] == ['gh', 'run', 'download']:
            path = Path(argv[argv.index('--dir') + 1]) / 'shard' / 'junit.xml'
            path.write_text(path.read_text().replace('time="3"', 'time="300"'))
        return result
    with pytest.raises(ToolBroken, match='綁定'):
        subject.read_cloud(SHA, 'owner/repo', 'main', FILES, call=tamper)
