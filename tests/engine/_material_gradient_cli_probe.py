"""梯度搜尋命令列考卷的雙精度子行程：只叫產品入口（``aosr.gradient.worker.run_worker``、``aosr.gradient.cli.main`` 的 stop），
寫資料夾與原始紀錄；比對與判決在父行程。

參考與部分重播共用一個行程；另保留迴圈中停止、下降半批截帳各一例，在全新無共用補丁的子行程重建後逐位比。
N 不寫死：先跑參考、記下每一次查停止時帳本最後一列的階段，每一種階段取第一次出現的那一次。
另外把參考帳本截成「前段只寫了一半」的幾份接續，與標頭被改過的一份接續（要拒收）。
空氣（修補二、三）：方案場景的密度與聲速換成另一組、替身換成用同一組空氣建的那份（父行程建好、第二個參數給路徑），
重建的目標在樣板起點的值要逐位等於這裡自己用設定層配方（``construction_environment``）造的那一份，並跟原方案原替身
（空氣等於產品預設）重建的不同。
"""
from __future__ import annotations

import json
import dataclasses
import inspect
import os
from unittest.mock import patch
import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient import cli, run as run_module, material_gradient as gradient_module
from aosr.gradient import material_gradient_loop as loop_module, template_rounds as rounds_module
from aosr.gradient.material_gradient import ChainObjective, Evaluator, ObjectiveAux, SurfaceDesign, baseline_theta, chain_objective, compiled_evaluator, objective_reading, COMPONENT_NAMES
from aosr.gradient.material_gradient_ledger import read_ledger, objective_fingerprint
from aosr.gradient.run import (
    LEDGER_FILE, PROJECT_FILE, STATUS_FILE, STOP_FILE, check_inputs, file_sha256, load_problem, manifest_sha256,
    read_project, stop_requested,
)
from aosr.physics.fem_reduced_receipt import receipt_path
from aosr.physics.fem_reduced_store import load_reduced
from aosr.scoring.gradient_objective_jax import gradient_objective_constants
from aosr.scoring.gradient_objective_settings import gradient_objective_settings
from aosr.scoring.six_category_jax import SixCategoryConstants, build_constants
from aosr.gradient.six_category_chain import six_category_chain_objective
from aosr.reporting.validation import checked_inputs
from aosr.reporting.scheme import Scheme
from aosr.physics.report_io import ReportInput
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.gradient.worker import run_worker
from tests.engine._scheme_cache import bitwise_equal

Record = dict[str, object]


def _folder(directory: Path, name: str, source: Path) -> Path:
    folder = directory / name
    folder.mkdir()
    shutil.copyfile(source / PROJECT_FILE, folder / PROJECT_FILE)
    return folder


def _last_phase(folder: Path) -> str:
    path = folder / LEDGER_FILE
    if not path.exists():
        return "none"
    _, rows = read_ledger(path)
    return rows[-1].phase if rows else "header"


def _status(folder: Path) -> Record:
    return dict(json.loads((folder / STATUS_FILE).read_text(encoding="utf-8")))


def _reference(directory: Path, source: Path) -> tuple[Record, list[str]]:
    """一口氣跑完；每一次查停止都記下那時帳本最後一列的階段（查停止＝下一次求值之前）。"""
    folder = _folder(directory, "reference", source)
    phases: list[str] = []
    check = stop_requested(folder)

    def watch() -> bool:
        phases.append(_last_phase(folder))
        return check()

    code = run_worker(folder, "start", should_stop=watch)
    return {"folder": folder.name, "code": code, "status": _status(folder), "checks": len(phases), "pid": os.getpid()}, phases


def _stopper(folder: Path, at: int, record: Record) -> Callable[[], bool]:
    """第 ``at`` 次查停止之前，用產品的 ``stop`` 命令放記號，再交給產品的判斷（看檔在不在）。"""
    calls = [0]
    check = stop_requested(folder)

    def should_stop() -> bool:
        calls[0] += 1
        if calls[0] == at:
            record['in_rounds'] = any(item.frame.f_globals.get('__name__') == loop_module.__name__
                                      for item in inspect.stack(context=0))
            record["stop_command_code"] = cli.main(["stop", str(folder)])
        return check()

    return should_stop


def _stop_and_resume(directory: Path, source: Path, at: int, phase: str) -> Record:
    folder = _folder(directory, f"stop-{at:05d}", source)
    record: Record = {"folder": folder.name, "at": at, "phase": phase, "pid": os.getpid()}
    record["stopped_code"] = run_worker(folder, "start", should_stop=_stopper(folder, at, record))
    record["stopped_status"] = _status(folder)
    record["marker_after_stop"] = (folder / STOP_FILE).exists()
    stopped = directory / f"{folder.name}.stopped.jsonl"
    if (folder / LEDGER_FILE).exists():
        shutil.copyfile(folder / LEDGER_FILE, stopped)
    record["stopped_ledger"] = stopped.name if stopped.exists() else None
    (folder / STOP_FILE).unlink()  # 命令列 resume 拿到鎖之後做的同一件事
    record["resumed_code"] = run_worker(folder, "resume", should_stop=stop_requested(folder))
    record["resumed_status"] = _status(folder)
    return record


def _cut(directory: Path, source: Path, reference: Path, name: str, keep: int) -> Record:
    """參考帳本留標頭與前 ``keep`` 列（前段只寫了一半），接續。"""
    folder = _folder(directory, name, source)
    lines = (reference / LEDGER_FILE).read_bytes().splitlines(keepends=True)
    (folder / LEDGER_FILE).write_bytes(b"".join(lines[:keep + 1]))
    code = run_worker(folder, "resume", should_stop=stop_requested(folder))
    return {"folder": folder.name, "keep": keep, "code": code, "status": _status(folder), "pid": os.getpid()}


def _other_header(directory: Path, source: Path, reference: Path) -> Record:
    """參考帳本只改標頭的設定指紋（同長度的另一個十六進位字），接續要拒收、帳本一個位元組都不動。"""
    folder = _folder(directory, "other-header", source)
    lines = (reference / LEDGER_FILE).read_bytes().splitlines(keepends=True)
    header = json.loads(lines[0])
    header["settings_fingerprint"] = "0" * len(header["settings_fingerprint"])
    cut = [(json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8"), *lines[1:3]]
    (folder / LEDGER_FILE).write_bytes(b"".join(cut))
    before = (folder / LEDGER_FILE).read_bytes()
    code = run_worker(folder, "resume", should_stop=stop_requested(folder))
    return {"folder": folder.name, "code": code, "status": _status(folder),
            "ledger_unchanged": (folder / LEDGER_FILE).read_bytes() == before}


def _air(directory: Path, source: Path, surrogate: Path) -> Record:
    """方案場景的空氣換成另一組、替身換成同一組空氣建的那份，project.json 指過去（雜湊重量），重建目標；
    值以 hex 交給父行程比。"""
    project = read_project(source)
    built = load_reduced(surrogate)
    density, speed = built.density_kg_m3, built.sound_speed_m_s
    document = json.loads(Path(project.scheme_path).read_text(encoding="utf-8"))
    document["scene"].update({"density_kg_m3": density, "sound_speed_m_s": speed})
    scheme_path = directory / "scheme-other-air.json"
    scheme_path.write_text(json.dumps(document), encoding="utf-8")
    changed = project.model_copy(update={
        "scheme_path": str(scheme_path), "scheme_sha256": file_sha256(scheme_path), "surrogate_path": str(surrogate),
        "surrogate_manifest_sha256": manifest_sha256(surrogate), "receipt_path": str(receipt_path(surrogate)),
        "receipt_sha256": file_sha256(receipt_path(surrogate))})
    problem = load_problem(changed)
    scheme, model, _ = check_inputs(changed)
    score = gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")), changed.purpose)
    constants = gradient_objective_constants(np.asarray(model.frequencies_hz), score, scheme.receiver_set,
                                             len(model.source_names))
    air, ranges = construction_environment(density_kg_m3=density, sound_speed_m_s=speed)
    _, pairs = checked_inputs(scheme, capabilities=load_capabilities(project.capabilities_path),
        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    six = build_constants(scheme.model_copy(update={"purpose": changed.purpose}), pair_inputs={k: v[1] for k, v in pairs.items()})
    expected = six_category_chain_objective(model, problem.designs, constants, six, air=air, ranges=ranges,
        frequencies=model.frequencies_hz)
    theta = jnp.asarray(baseline_theta(problem.designs))

    def value(objective: ChainObjective) -> str:
        return float(objective(theta)[0]).hex()

    return {"rebuilt": value(problem.objective),
            "expected": value(expected), "original": value(load_problem(project).objective)}




def _fingerprints(six: object) -> dict[str, bool]:
    from aosr.scoring.six_category_jax import SixCategoryConstants
    from typing import cast
    constants = cast(SixCategoryConstants, six)
    original = objective_fingerprint(constants)
    result = {}
    for name in constants._fields:
        array = np.asarray(getattr(constants, name))
        changed = array.copy()
        if not changed.size:
            changed = np.zeros((1,), dtype=array.dtype)
        elif array.dtype.kind in 'US':
            changed = np.char.add(array, 'changed')
        elif array.dtype.kind == 'b':
            changed.flat[0] = not changed.flat[0]
        else:
            changed.flat[0] = 0 if changed.flat[0] != 0 else 1
        result[name] = objective_fingerprint(constants._replace(**{name:changed})) != original
    return result


def _reversed_sources(project: object, scheme: object, model: object, settings: object, theta: jnp.ndarray) -> list[float]:
    from typing import cast
    from aosr.physics.fem_reduced import ReducedModel
    original = cast(ReducedModel, model)
    flipped = dataclasses.replace(original, source_names=original.source_names[::-1],
        matrices=dataclasses.replace(original.matrices, sources=original.matrices.sources[::-1]))
    with patch.object(run_module, 'check_inputs', return_value=(scheme,flipped,settings)):
        rebuilt = load_problem(cast(run_module.GradientProject, project))
    (_, aux), _ = compiled_evaluator(rebuilt.objective)(theta)
    return [float(x) for x in objective_reading(aux).components[3:]]

def _control(source: Path, reference: Path) -> Record:
    project = read_project(source)
    scheme, model, settings = check_inputs(project)
    problem = load_problem(project)
    _, pairs = checked_inputs(scheme, capabilities=load_capabilities(project.capabilities_path),
        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    six = build_constants(scheme.model_copy(update={"purpose": project.purpose}), pair_inputs={k:v[1] for k,v in pairs.items()})
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    direct = six_category_chain_objective(model, problem.designs, six.objective(), six, air=air, ranges=ranges, frequencies=model.frequencies_hz)
    old = chain_objective(model, problem.designs, six.objective(), air=air, ranges=ranges, frequencies=model.frequencies_hz)
    _, rows = read_ledger(reference / LEDGER_FILE)
    first = rows[0]
    theta = jnp.asarray(first.theta())
    (_, aux), _ = compiled_evaluator(direct)(theta)
    (_, low), _ = compiled_evaluator(old)(theta)
    actual = [float(x).hex() for x in first.scores()[2]]
    expected = [float(x).hex() for x in objective_reading(aux).components]
    old_hex = [float(x).hex() for x in objective_reading(low).components]
    return dict(names=COMPONENT_NAMES, actual=actual, expected=expected, low_old=old_hex,
                low_bitwise=actual[:3] == old_hex, six_bitwise=actual[3:] == expected[3:],
                fingerprints=_fingerprints(six), reversed_sources=_reversed_sources(project,scheme,model,settings,theta))

def main(directory: Path, air_surrogate: Path, part: str) -> None:
    if part == 'all':
        # 參考與重播共用真正的編譯函式；控制組用直接 import 的 load_problem，仍各自重建。
        evaluators = _ReplayEvaluators()
        with (patch.object(run_module, 'load_problem', side_effect=_shared_problem_loader()),
              patch.object(gradient_module, 'compiled_evaluator', side_effect=evaluators),
              patch.object(run_module, 'compiled_evaluator', side_effect=evaluators),
              patch.object(loop_module, 'compiled_evaluator', side_effect=evaluators),
              patch.object(rounds_module, 'compiled_evaluator', side_effect=evaluators)):
            main(directory, air_surrogate, 'reference')
            combined = json.loads((directory / 'output-reference.json').read_text(encoding='utf-8'))
            for replay in ('stops', 'cuts'):
                combined.update(_replays(directory, directory / 'source', replay))
        (directory / 'output-all.json').write_text(json.dumps(combined, ensure_ascii=False), encoding='utf-8')
        return
    source = directory / "source"
    if part == "reference":
        reference, phases = _reference(directory, source)
        reference_folder = directory / str(reference["folder"])
        _, rows = read_ledger(reference_folder / LEDGER_FILE)
        pre_loop = [index for index, row in enumerate(rows) if row.round is None]
        output: Record = {"reference": reference, "phases": phases,
                          "pre_loop_rows": len(pre_loop),
                          "other_header": _other_header(directory, source, reference_folder),
                          "air": _air(directory, source, air_surrogate),
                          "control": _control(source, reference_folder)}
    elif part.startswith('rebuild-'):
        # 這兩條路在全新子行程直接跑，絕不進 _after_reference 的共用補丁。
        if (run_module.load_problem is not load_problem or
                any(module.compiled_evaluator is not compiled_evaluator
                    for module in (gradient_module, run_module, loop_module, rounds_module))):
            raise RuntimeError('跨重建控制不准套共用載入器或求值器補丁')
        output = _replays(directory, source, part.removeprefix('rebuild-'), isolated=True)
    else:
        output = _after_reference(directory, source, part)
    (directory / f"output-{part}.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(directory / "__pycache__", ignore_errors=True)


def _after_reference(directory: Path, source: Path, part: str) -> Record:
    """同一段的停止／截帳重播共用編譯物件；每次仍重建並核對正式輸入，不快取求值。"""
    with patch.object(run_module, 'load_problem', side_effect=_shared_problem_loader()):
        return _replays(directory, source, part)


def _shared_problem_loader() -> Callable[[run_module.GradientProject], run_module.GradientProblem]:
    """每次真載入，核對輸入及起點值／分項／梯度逐位一致後，才共用目標身分。"""
    original = run_module.load_problem
    constants = _ReplayConstants()
    shared: run_module.GradientProblem | None = None
    objectives: list[tuple[tuple[SurfaceDesign, ...], ChainObjective]] = []

    def load(project: run_module.GradientProject) -> run_module.GradientProblem:
        nonlocal shared
        with patch.object(run_module, 'build_constants', side_effect=constants):
            current = original(project)
        if shared is None:
            objectives.append((current.designs, current.objective))
            shared = current
        if current.header != shared.header or current.designs != shared.designs or current.kept != shared.kept:
            raise ValueError('停止／截帳重播的輸入不同，不准共用編譯物件')
        _require_same_objective(current.objective, shared.objective, current.designs)
        factory = current.templates.objective_for

        def objective_for(designs: tuple[SurfaceDesign, ...]) -> ChainObjective:
            fresh = factory(designs)
            for previous, objective in objectives:
                if previous == designs:
                    _require_same_objective(fresh, objective, designs)
                    return objective
            objectives.append((designs, fresh))
            return fresh

        return current._replace(objective=shared.objective,
                                templates=current.templates._replace(objective_for=objective_for))

    return load


def _require_same_objective(fresh: ChainObjective, shared: ChainObjective,
                            designs: tuple[SurfaceDesign, ...]) -> None:
    """替換前用原求值器各算起點的值、全部分項與梯度；不能讀重播 memo 的舊答案。"""
    theta = jnp.asarray(baseline_theta(designs))
    actual = jax.tree.map(np.asarray, compiled_evaluator(fresh)(theta))
    expected = jax.tree.map(np.asarray, compiled_evaluator(shared)(theta))
    if not bitwise_equal(actual, expected):
        raise ValueError('重建目標在起點的值、分項或梯度逐位不同，不准共用')


class _ReplayConstants:
    """同段重播才共用幾何常數；方案、逐對輸入逐位相同且設定檔未變才准用，不存評分答案。"""

    def __init__(self) -> None:
        self.build = build_constants
        self.inputs: tuple[Scheme, Mapping[tuple[str, str], ReportInput], str] | None = None
        self.value: SixCategoryConstants | None = None

    def __call__(self, scheme: Scheme, *, pair_inputs: Mapping[tuple[str, str], ReportInput],
                 quality_targets_path: Path) -> SixCategoryConstants:
        inputs = (scheme, pair_inputs, file_sha256(quality_targets_path))
        if self.inputs is None:
            self.value = self.build(scheme, pair_inputs=pair_inputs, quality_targets_path=quality_targets_path)
            self.inputs = inputs
        elif not bitwise_equal(inputs, self.inputs):
            raise ValueError('停止／截帳重播的方案、逐對輸入或品質設定不同，不准共用常數')
        if self.value is None:
            raise ValueError('停止／截帳重播的常數尚未建立')
        return self.value


type _Evaluation = tuple[tuple[jax.Array, ObjectiveAux], jax.Array]


class _ReplayEvaluators:
    """本次狀態機考卷共用純數值邊界：同函式與同 θ 位元只算一次，新點真算。

    不存下降決定或帳本；控制組直接匯入的 compiled_evaluator 不經這層，仍獨立求值。
    每個探針行程重建空表，植錯不會讀到上一跑的值。
    """

    def __init__(self) -> None:
        self.functions: dict[ChainObjective, Evaluator] = {}

    def __call__(self, objective: ChainObjective) -> Evaluator:
        if objective not in self.functions:
            actual = compiled_evaluator(objective)
            points: dict[tuple[str, tuple[int, ...], bool, bytes], _Evaluation] = {}

            def evaluate(theta: jax.Array) -> _Evaluation:
                values = np.asarray(theta)
                key = (values.dtype.str, values.shape, bool(theta.weak_type), values.tobytes())
                if key not in points:
                    points[key] = actual(theta)
                return points[key]

            self.functions[objective] = evaluate
        return self.functions[objective]


def _replays(directory: Path, source: Path, part: str, *, isolated: bool = False) -> Record:
    # 父行程已同步拿到 reference 的完成收據才會啟動這一段，不在子行程裡等待檔案。
    output = json.loads((directory / "output-reference.json").read_text(encoding="utf-8"))
    reference, phases = output["reference"], output["phases"]
    reference_folder = directory / str(reference["folder"])
    if part == "stops":
        firsts = {phase: index + 1 for index, phase in reversed(list(enumerate(phases)))}
        chosen = sorted({*firsts.values(), len(phases)})
        # 最後一次回呼在迴圈內；整輪才落帳，phase 仍是 retune，但前段已完整，故走 resume_rounds。
        # 截帳獨立例走 _fresh。兩例各在無補丁新行程守一條產品接續分支。
        independent = len(phases)
        selected = [independent] if isolated else [at for at in chosen if at != independent]
        return {"stops": [{**_stop_and_resume(directory, source, at, phases[at - 1]), 'isolated': isolated}
                           for at in selected]}
    _, rows = read_ledger(reference_folder / LEDGER_FILE)
    pre_loop = [index for index, row in enumerate(rows) if row.round is None]
    snap = next(index for index, row in enumerate(rows) if row.phase == "snap")
    keeps = sorted({max(1, snap // 2), snap, snap + 1, (snap + 1 + len(pre_loop)) // 2})
    # 取下降批中間的合法前綴，明確避開整批邊界，守跨行程的部分下降接續。
    independent = max(1, snap // 2)
    if not 1 <= independent < snap:
        raise ValueError('跨重建截帳控制需要至少兩列下降')
    selected = [independent] if isolated else [keep for keep in keeps if keep != independent]
    cuts = [{**_cut(directory, source, reference_folder, f"cut-{keep:04d}", keep), 'isolated': isolated}
            for keep in selected if keep < len(pre_loop)]
    return {"cuts": cuts}


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
