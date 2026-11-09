"""合成數值邊界的真開始／接續控制；植錯只改接續時的新建目標係數。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp

from aosr.gradient import material_gradient as gradient, run
from tests.engine import _material_gradient_cli_probe as replay
from tests.engine._material_gradient_loop_probe import synthetic_header, synthetic_objective
from tests.engine._scheme_cache import bitwise_equal
from tests.engine.test_material_gradient import DESCENT_CASE, _designs, _settings
from tests.engine.test_material_gradient_mount import _project
from tests.engine.test_material_gradient_templates import _context

# 正常控制為 1；植錯時只在本考卷子行程的新建接續目標改成 1 + 1e-12。
RESUME_WEIGHT = 1.0


def main(directory: Path) -> None:
    settings = _settings()
    settings = settings.model_copy(update={'descent': settings.descent.model_copy(update={'max_steps': 3})})
    designs = _designs(DESCENT_CASE, settings=settings)
    header = synthetic_header(designs, settings)
    project = _project()
    resuming = False
    objectives: list[gradient.ChainObjective] = []

    def load(_: run.GradientProject) -> run.GradientProblem:
        original = synthetic_objective(designs, [0])
        factor = RESUME_WEIGHT if resuming else 1.0

        def objective(theta: jnp.ndarray) -> tuple[jnp.ndarray, gradient.ObjectiveAux]:
            value, aux = original(theta)
            return value * factor, aux

        objectives.append(objective)
        context = _context(settings)._replace(objective_for=lambda _: objective)
        return run.GradientProblem(designs, settings, objective, header, {}, context)

    def after_descent() -> bool:
        path = directory / run.LEDGER_FILE
        return path.exists() and replay._last_phase(directory) == 'descent'

    def rows() -> list[dict[str, object]]:
        content = (directory / run.LEDGER_FILE).read_text().splitlines()[1:]
        return [{key: value for key, value in json.loads(line).items() if key != 'seconds'} for line in content]

    error: str | None = None
    with patch.object(run, 'read_project', return_value=project), patch.object(run, 'load_problem', side_effect=load):
        shared = replay._shared_problem_loader()
        with patch.object(run, 'load_problem', side_effect=shared):
            try:
                run.run_search(directory, should_stop=after_descent)
            except gradient.SearchStopped:
                pass
            else:
                raise AssertionError('開始未在下降批後停止')
            expected = rows()
            resuming = True
            try:
                run.resume_search(directory, should_stop=after_descent)
            except gradient.SearchStopped:
                pass
            except ValueError as failure:
                error = str(failure)
    result = {'reproduced': error is None and bitwise_equal(expected, rows()), 'error': error,
              'start_stop': 'descent', 'reconstructed_objectives': len(objectives)}
    (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
