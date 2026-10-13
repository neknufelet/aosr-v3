"""以真行程跑命令列身分，經網頁／搜尋的開行程邊界回讀設定。"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import aosr
from aosr import runtime

_CPU_PIN_VARS = ('NPY_DISABLE_CPU_FEATURES', 'OPENBLAS_CORETYPE', runtime.XLA_FLAGS_VAR)


def snapshot() -> dict[str, object]:
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.paths import config_path
    from aosr.reporting.import_closure import environment_digest
    from aosr.reporting.physics_identity import physics_identity
    from aosr.reporting.modal_diagnosis import modal_identity
    return {
        'settings': runtime.jax_settings(), 'environment': environment_digest(),
        'physics': physics_identity(capabilities=load_capabilities(config_path('capabilities.toml')),
                                    directivity=load_directivity_defaults(config_path('directivity_defaults.toml'))),
        'modal': modal_identity(), 'package': str(Path(str(aosr.__file__)).resolve()),
        'pins': {name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ},
    }


def child(output: Path) -> None:
    from aosr.config.paths import config_path
    from aosr.reporting import scheme_cli
    assert 'jax' not in sys.modules
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        exit_code = scheme_cli.main(['identity', '--capabilities', str(config_path('capabilities.toml'))])
        assert exit_code == 0
    cold = snapshot()
    assert 'jax' not in sys.modules
    import jax
    assert getattr(jax.config, 'jax_platforms') == 'cpu' and getattr(jax.config, 'jax_enable_x64') is True
    assert snapshot() == cold
    assert f"物理身分 {cold['physics']}" in printed.getvalue().splitlines()
    output.write_text(json.dumps(cold))


def parent(route: str, directory: Path) -> None:
    from aosr.config.paths import config_path
    runtime.configure_calculation_jax()
    cold = snapshot()
    import jax
    assert getattr(jax.config, 'jax_platforms') == 'cpu' and getattr(jax.config, 'jax_enable_x64') is True
    assert snapshot() == cold
    assert runtime.jax_settings(env={runtime.PLATFORMS_VAR: 'wrong', runtime.ENABLE_X64_VAR: 'false'}) == ('cpu', True)
    output = directory / 'child.json'
    runner = (sys.executable, '-B', str(Path(__file__).resolve()), 'child', str(output))
    pins = cold['pins']
    def pinned_env(*, threads: int | None = None) -> dict[str, str]:
        env = runtime.child_process_env(threads=threads)
        env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
        return env
    if route == 'job-manager-start':
        from aosr.gui.jobs import JobManager
        manager = JobManager(directory, runner, 'test', config_path('capabilities.toml'))
        with patch('aosr.gui.jobs.child_process_env', pinned_env):
            state = manager.start(directory / 'scheme.json')
        process = manager.processes[str(state['run_id'])]
        exit_code = process.wait(timeout=30)
        assert exit_code == 0
        manager.get(str(state['run_id']))
    elif route == 'subprocess-compute-spawn':
        from aosr.search.worker import SubprocessCompute
        compute = SubprocessCompute(capabilities_path=config_path('capabilities.toml'),
                                    engine_commit='test', search_id='test', fem_root=directory)
        with patch('aosr.search.worker.child_process_env', pinned_env):
            process = compute._spawn(runner, directory / 'child.stderr')
        exit_code = process.wait(timeout=30)
        assert exit_code == 0
    else:
        from aosr.search.modal_attach import _compute
        from aosr.search.modal_record import ModalRole, RoleInput
        item = RoleInput(ModalRole(role='baseline'), None, directory / 'scheme.json', directory / 'unused.json')
        with patch('aosr.search.modal_attach.child_process_env', pinned_env):
            exit_code = _compute(item, directory / 'unused.json', cache_dir=directory, lock_fd=None, runner=runner)
            assert exit_code == 0
    found = json.loads(output.read_text())
    assert found == json.loads(json.dumps(cold)), (cold, found)
    assert found['pins'] == pins
    sys.stdout.write(json.dumps(found) + '\n')


if __name__ == '__main__':
    child(Path(sys.argv[2])) if sys.argv[1] == 'child' else parent(sys.argv[1], Path(sys.argv[2]))
