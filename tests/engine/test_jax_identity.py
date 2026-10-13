"""正式入口先定精度；真子行程與靜態身分共用設定，變異不得漏掉。"""
from __future__ import annotations

import importlib.metadata as metadata
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import aosr
from aosr import runtime
from aosr.reporting import import_closure

ENTRIES = (
    'aosr.physics.three_lane_report_cli', 'aosr.reporting.scheme_cli', 'aosr.search.cli',
    'aosr.gui.__main__', 'aosr.physics.fem_rigid', 'aosr.physics.late_energy_cli',
    'aosr.physics.room_paths',
)
_CPU_PIN_VARS = ('NPY_DISABLE_CPU_FEATURES', 'OPENBLAS_CORETYPE', runtime.XLA_FLAGS_VAR)
CONFIGURED_FUNCTIONS = tuple((module, 'main') for module in ENTRIES) + (('aosr.search.cli', '_identity'),)


def _env(directory: Path) -> dict[str, str]:
    env = runtime.child_process_env(threads=1)
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    runtime.configure_jax(platforms='deliberately-invalid', enable_x64=False, env=env)
    env['TMPDIR'] = str(directory)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def _run(directory: Path, source: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, '-B', '-c', source, *args], cwd=directory,
                          env=_env(directory), capture_output=True, text=True, timeout=45)


def _entry_probe() -> str:
    return '''
import importlib, sys
from pathlib import Path
import aosr
from aosr import runtime
assert Path(aosr.__file__).resolve() == Path(sys.argv[2]).resolve()
module = importlib.import_module(sys.argv[1])
assert 'jax' not in sys.modules, '入口匯入過早載入 JAX'
sys.argv = [sys.argv[1], '--help']
try:
    module.main() if sys.argv[0] == 'aosr.gui.__main__' else module.main(['--help'])
except SystemExit as error:
    assert error.code == 0
import jax
assert jax.config.jax_platforms == 'cpu', '入口沒有先指定 CPU'
assert jax.config.jax_enable_x64 is True, '入口没有先開雙精度'
assert runtime.jax_settings(env={}) == ('cpu', True)
'''


def _first_statement(source: str, function: str) -> ast.stmt:
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == function)
    first = node.body[0]
    has_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
        and isinstance(first.value.value, str)
    return node.body[1] if has_doc else first


def _assert_configuration_first(statement: ast.stmt) -> None:
    assert isinstance(statement, ast.Expr), '正式設定必須是第一個語句'
    call = statement.value
    assert isinstance(call, ast.Call), '第一個語句必須呼叫正式設定'
    assert isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
    assert (call.func.value.id, call.func.attr) == ('runtime', 'configure_calculation_jax')
    assert not call.args and not call.keywords, '正式設定不得改成注入字典或其他參數'


@pytest.mark.parametrize(('module', 'function'), CONFIGURED_FUNCTIONS)
def test_configuration_is_first_statement(module: str, function: str) -> None:
    path = Path(str(aosr.__file__)).parent.joinpath(*module.split('.')[1:]).with_suffix('.py')
    _assert_configuration_first(_first_statement(path.read_text(), function))


@pytest.mark.parametrize(('module', 'function'), CONFIGURED_FUNCTIONS)
@pytest.mark.parametrize('change', ['second', 'conditional', 'other-call'])
def test_first_statement_guard_rejects_mutations(module: str, function: str, change: str) -> None:
    path = Path(str(aosr.__file__)).parent.joinpath(*module.split('.')[1:]).with_suffix('.py')
    node = _first_statement(path.read_text(), function)
    if change == 'second':
        changed = ast.parse('before = None').body[0]
    elif change == 'conditional':
        changed = ast.If(test=ast.Constant(True), body=[node], orelse=[])
    else:
        changed = ast.parse('runtime.configure_jax()').body[0]
    with pytest.raises(AssertionError):
        _assert_configuration_first(changed)


@pytest.mark.parametrize('module', ENTRIES)
def test_entry_configures_before_jax(tmp_path: Path, module: str) -> None:
    process = _run(tmp_path, _entry_probe(), module, str(aosr.__file__))
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize('module', ENTRIES)
def test_removing_entry_configuration_is_detected(tmp_path: Path, module: str) -> None:
    root = Path(shutil.copytree(Path(str(aosr.__file__)).parent, tmp_path / 'aosr',
                               ignore=shutil.ignore_patterns('__pycache__')))
    path = root.joinpath(*module.split('.')[1:]).with_suffix('.py')
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    call = _first_statement(original, 'main')
    _assert_configuration_first(call)
    del lines[call.lineno - 1]
    changed = ''.join(lines)
    assert changed != original, '沒有可移除的入口設定呼叫'
    path.write_text(changed)
    process = _run(tmp_path, _entry_probe(), module, str(root / '__init__.py'))
    assert process.returncode != 0
    assert '入口沒有先指定 CPU' in process.stderr or '入口没有先開雙精度' in process.stderr


@pytest.mark.parametrize('enabled', [False, True])
def test_settings_without_loading_jax(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.delitem(sys.modules, 'jax', raising=False)
    env: dict[str, str] = {}
    runtime.configure_jax(platforms='cpu', enable_x64=enabled, env=env)
    assert runtime.jax_settings(env=env) == ('cpu', enabled)
    assert runtime.jax_settings(env={}) == (None, False)
    assert 'jax' not in sys.modules


def test_calculation_configuration_is_explicit_and_cold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, 'jax', raising=False)
    env = {'unrelated': 'keep'}
    runtime.configure_calculation_jax(env=env)
    assert runtime.jax_settings(env=env) == ('cpu', True)
    assert env['unrelated'] == 'keep'
    assert 'jax' not in sys.modules


@pytest.mark.parametrize('change', ['jax', 'jaxlib', 'platform', 'x64'])
def test_changes_that_must_flip_environment_identity(monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    monkeypatch.setattr(runtime, 'jax_settings', lambda: ('cpu', True))
    before = import_closure.environment_digest()
    assert import_closure.environment_digest() == before
    if change in ('jax', 'jaxlib'):
        original = metadata.version
        monkeypatch.setattr(metadata, 'version', lambda name:
                            original(name) + '+changed' if name == change else original(name))
    else:
        changed = ('another-platform', True) if change == 'platform' else ('cpu', False)
        monkeypatch.setattr(runtime, 'jax_settings', lambda: changed)
    assert import_closure.environment_digest() != before


def test_both_jax_distributions_are_measured_without_import() -> None:
    versions = dict(import_closure.physics_dependency_versions())
    assert versions['jax'] == metadata.version('jax')
    assert versions['jaxlib'] == metadata.version('jaxlib')


def test_cold_identity_measurement_does_not_load_jax(tmp_path: Path) -> None:
    source = '''
import sys
from pathlib import Path
import aosr
from aosr import runtime
assert Path(aosr.__file__).resolve() == Path(sys.argv[1]).resolve()
from aosr.reporting.import_closure import environment_digest
from aosr.reporting.physics_identity import physics_identity
from aosr.reporting.modal_diagnosis import modal_identity
from aosr.config.paths import config_path
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
assert 'jax' not in sys.modules
forgotten = environment_digest()
runtime.configure_calculation_jax()
assert environment_digest() != forgotten
physics_identity(capabilities=load_capabilities(config_path('capabilities.toml')),
                 directivity=load_directivity_defaults(config_path('directivity_defaults.toml')))
modal_identity()
assert 'jax' not in sys.modules
'''
    process = _run(tmp_path, source, str(aosr.__file__))
    assert process.returncode == 0, process.stdout + process.stderr


def test_child_environment_uses_injected_source_and_own_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, 'jax', raising=False)
    inherited = {'PATH': '/test/bin', 'PYTHONPATH': str(tmp_path), 'UNRELATED': 'discard'}
    runtime.configure_calculation_jax(env=inherited)
    env = runtime.child_process_env(threads=1, env=inherited)
    assert env['PYTHONPATH'] == str(Path(str(aosr.__file__)).resolve().parents[1])
    assert env['PATH'] == '/test/bin'
    runtime.configure_calculation_jax(env=env)
    assert runtime.jax_settings(env=env) == ('cpu', True)
    assert 'UNRELATED' not in env


def test_search_identity_configures_before_measurement(tmp_path: Path) -> None:
    source = '''
import sys
from pathlib import Path
import aosr
from aosr import runtime
assert Path(aosr.__file__).resolve() == Path(sys.argv[1]).resolve()
from aosr.search.cli import _identity
from aosr.config.paths import config_path
from aosr.reporting.physics_identity import physics_identity
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
found = _identity('dedicated_two_channel_listening_room', config_path('capabilities.toml'))
assert runtime.jax_settings() == ('cpu', True)
assert 'jax' not in sys.modules
assert found.physics_identity == physics_identity(
    capabilities=load_capabilities(config_path('capabilities.toml')),
    directivity=load_directivity_defaults(config_path('directivity_defaults.toml')))
'''
    process = _run(tmp_path, source, str(aosr.__file__))
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize('route', ['job-manager-start', 'subprocess-compute-spawn', 'modal-attach-compute'])
def test_real_children_share_effective_settings_and_identity(tmp_path: Path, route: str) -> None:
    probe = Path(__file__).with_name('_jax_identity_probe.py')
    source = 'import runpy, sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")'
    process = _run(tmp_path, source, str(probe), route, str(tmp_path))
    assert process.returncode == 0, process.stdout + process.stderr + '\n'.join(
        path.read_text() for path in tmp_path.rglob('*.stderr'))


@pytest.mark.parametrize(('flag', 'expected'), [
    (flag, expected) for values, expected in [
        (('y', 'yes', 't', 'true', 'on', '1'), True),
        (('n', 'no', 'f', 'false', 'off', '0'), False),
    ] for value in values for flag in (value, value.upper())
])
def test_cold_x64_matches_jax_bool_env(
    monkeypatch: pytest.MonkeyPatch, flag: str, expected: bool,
) -> None:
    """讀法出處：JAX 0.10.1，jax._src.config 的 config.py::bool_env；lower 後不去空白。"""
    monkeypatch.delitem(sys.modules, 'jax', raising=False)
    assert runtime.jax_settings(env={runtime.ENABLE_X64_VAR: flag}) == (None, expected)
    assert 'jax' not in sys.modules


@pytest.mark.parametrize('flag', ['', ' true', 'true ', ' true ', '\tfalse\n', 'garbage', '2'])
def test_cold_x64_rejects_invalid_jax_bool_env(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    """拒收值出處：JAX 0.10.1，jax._src.config 的 config.py::bool_env。"""
    monkeypatch.delitem(sys.modules, 'jax', raising=False)
    with pytest.raises(ValueError):
        runtime.jax_settings(env={runtime.ENABLE_X64_VAR: flag})
    assert 'jax' not in sys.modules


@pytest.mark.parametrize('enabled', [False, True])
def test_loaded_calculation_configuration_checks_effective_value(tmp_path: Path, enabled: bool) -> None:
    source = '''
import sys
from pathlib import Path
import aosr
from aosr import runtime
assert Path(aosr.__file__).resolve() == Path(sys.argv[1]).resolve()
enabled = sys.argv[2] == 'True'
if enabled:
    runtime.configure_calculation_jax()
else:
    runtime.configure_jax(platforms='cpu', enable_x64=False)
import jax
before = runtime.jax_settings()
assert before == ('cpu', enabled)
if enabled:
    runtime.configure_calculation_jax()
    assert runtime.jax_settings() == before
else:
    try:
        runtime.configure_calculation_jax()
    except RuntimeError as error:
        message = str(error)
        for part in ('JAX 已載入', str(before), str(('cpu', True)),
                     'v2', 'import-time-global-flip-poisons-suite'):
            assert part in message, message
    else:
        raise AssertionError('晚設定不得靜默無效')
    injected = {}
    runtime.configure_calculation_jax(env=injected)
    assert injected[runtime.ENABLE_X64_VAR] == runtime.TRUE_TEXT
    assert runtime.jax_settings() == before
'''
    process = _run(tmp_path, source, str(aosr.__file__), str(enabled))
    assert process.returncode == 0, process.stdout + process.stderr


def test_loaded_settings_ignore_real_environment_and_digest_tracks_jax(tmp_path: Path) -> None:
    source = '''
import os, sys
from pathlib import Path
import aosr
from aosr import runtime
assert Path(aosr.__file__).resolve() == Path(sys.argv[1]).resolve()
from aosr.reporting.import_closure import environment_digest
runtime.configure_calculation_jax()
import jax
before = environment_digest()
runtime.configure_jax(platforms='cpu', enable_x64=False)
assert os.environ[runtime.ENABLE_X64_VAR] == runtime.FALSE_TEXT
assert runtime.jax_settings() == ('cpu', True)
assert environment_digest() == before
jax.config.update('jax_enable_x64', False)
assert runtime.jax_settings() == ('cpu', False)
assert environment_digest() != before
'''
    process = _run(tmp_path, source, str(aosr.__file__))
    assert process.returncode == 0, process.stdout + process.stderr
