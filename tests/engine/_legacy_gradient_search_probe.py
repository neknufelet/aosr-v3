"""只為既有終點考卷重建 v5/v2/v4-v5 低頻資料；正式搜尋沒有舊目標開關。

演算法沿用現在的下降與候選規則，純量目標刻意指定舊鏈；中間用六類合成外殼寫帳，
最後輸出歷史欄位。資料全在考卷暫存樹，供舊版終點的完整模型控制組讀。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from aosr.gradient import run, material_gradient as g, material_gradient_ledger as ledger
from aosr.config.construction_environment import construction_environment
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.gradient_objective_settings import legacy_gradient_objective_settings
from tests.engine._six_search_helpers import six_objective


def legacy_endpoint_copy(directory: Path, source: Path, name: str = 'legacy-endpoint') -> Path:
    from tests.engine.test_material_gradient_cli import _env, WAIT_S

    target = directory / name / source.name
    shutil.copytree(source, target)
    done = subprocess.run([sys.executable, '-m', 'tests.engine._legacy_gradient_search_probe', str(target)],
        cwd=directory, env=_env(directory), capture_output=True, text=True, timeout=WAIT_S)
    assert done.returncode == 0, done.stdout + done.stderr
    return target


def _downgrade(folder: Path, low_fingerprint: str) -> None:
    for name, version in ((run.LEDGER_FILE, 'aosr.material_gradient_ledger.v5'),
                          (run.TEMPLATES_FILE, 'aosr.material_gradient_templates.v2')):
        path = folder / name
        header, *rows = [json.loads(line) for line in path.read_text().splitlines()]
        header.update(ledger_version=version, objective_fingerprint=low_fingerprint)
        header['score_constants'].pop('category_weights')
        if name == run.TEMPLATES_FILE:
            header['base_ledger_sha256'] = run.file_sha256(folder / run.LEDGER_FILE)
        for row in rows:
            row['components'] = row['components'][:3]
        path.write_text(''.join(json.dumps(item, ensure_ascii=False)+'\n' for item in (header,*rows)))
    result = json.loads((folder / run.RESULT_FILE).read_text())
    result['format_version'] = run.LOW_FREQUENCY_MOUNTED_RESULT_VERSION if result['format_version'] == run.MOUNTED_RESULT_VERSION else run.LOW_FREQUENCY_RESULT_VERSION
    result['components'] = {name: result['components'][name] for name in g.COMPONENT_NAMES[:3]}
    result.update(ledger_sha256=run.file_sha256(folder / run.LEDGER_FILE), template_ledger_sha256=run.file_sha256(folder / run.TEMPLATES_FILE))
    (folder / run.RESULT_FILE).write_text(json.dumps(result))


def main(folder: Path) -> None:
    project = run.read_project(folder)
    scheme, model, _ = run.check_inputs(project)
    problem = run.load_problem(project)
    constants = run.search_objective_constants(project, scheme, model)
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    def factory(designs: tuple[g.SurfaceDesign, ...]) -> g.ChainObjective:
        return six_objective(g.chain_objective(model, designs, constants, air=air, ranges=ranges,
            frequencies=model.frequencies_hz, frequency_batch=project.frequency_batch))
    problem = problem._replace(objective=factory(problem.designs), templates=problem.templates._replace(objective_for=factory))
    for name in (run.LEDGER_FILE, run.TEMPLATES_FILE, run.RESULT_FILE):
        (folder / name).unlink()
    ledger.write_ledger(folder / run.LEDGER_FILE, problem.header, ())
    run._fresh(problem, folder / run.LEDGER_FILE, None)
    run._template_segment(problem, folder, None)
    run.write_result(folder, problem)
    _downgrade(folder, ledger.objective_fingerprint(constants))
    previous = legacy_gradient_objective_settings(load_quality_targets(config_path('quality_targets.toml')), project.purpose)
    run.write_json_atomic(folder / run.PROJECT_FILE, project.model_copy(update={'objective_settings_fingerprint':previous.fingerprint}))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
