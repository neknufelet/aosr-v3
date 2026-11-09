"""外部審查：聲源身分與零入射角構造都從正式入口守住。"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

from aosr.runtime import child_process_env, configure_jax
from tests.engine._scheme_cache import shared_json

type Measurement = Callable[[str], dict[str, object]]


@pytest.fixture(scope='module')
def measured(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> Measurement:
    def get(mode: str) -> dict[str, object]:
        def produce() -> str:
            directory = tmp_path_factory.mktemp(f'six-external-{mode}')
            output = directory / 'result.json'
            env = child_process_env(threads=1)
            configure_jax(platforms='cpu', enable_x64=True, env=env)
            env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'], str(Path(__file__).resolve().parents[2])))
            done = subprocess.run([sys.executable, '-m', 'tests.engine._six_category_external_probe', mode, str(output)],
                cwd=directory, env=env, capture_output=True, text=True, timeout=900)
            assert done.returncode == 0, done.stdout + done.stderr
            return output.read_text()
        return cast(dict[str, object], json.loads(shared_json(tmp_path_factory, worker_id, f'six-external-{mode}', produce)))
    return get


def test_search_only_speaker_key_order_preserves_all_scores_and_theta_gradient_bitwise(measured: Measurement) -> None:
    row = measured('sources')
    keys = cast(list[list[str]], row['keys'])
    assert keys[0] != keys[1]
    assert row['bitwise'] is True, row
    assert row['metadata'] == [True, True]


def test_both_factories_and_endpoint_reject_source_order_mismatch_and_fingerprint_binds_ids(measured: Measurement) -> None:
    row = measured('rejection')
    errors = cast(list[str | None], row['errors'])
    assert len(errors) == 3 and all(message is not None and '聲源' in message and '順序' in message for message in errors), row
    assert row['fingerprint_changed'] is True


def test_zero_angle_assembly_adjustable_and_kept_both_precisions_and_numpy_product(measured: Measurement) -> None:
    row = measured('empty')
    assert row['zero_angle_count'] == 0 and row['zero_angle_patch'] == 'x0@b'
    rows = cast(list[dict[str, object]], row['rows'])
    assert len(rows) == 4 and all(item['finite'] for item in rows)
    assert row['baseline_finite'] is True
    assert row['empty_dtypes'] == ['complex64', 'complex128']
    expected = cast(dict[str, float], row['numpy'])
    assert abs(expected['actual'] - expected['expected']) <= expected['bound'], expected
