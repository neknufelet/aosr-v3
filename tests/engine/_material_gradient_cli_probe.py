"""梯度搜尋命令列考卷的雙精度子行程：只叫產品入口（``aosr.gradient.worker.run_worker``、``aosr.gradient.cli.main`` 的 stop），
寫資料夾與原始紀錄；比對與判決在父行程。

一口氣跑完的參考資料夾，與在第 N 次求值之前放停止記號、再接續的資料夾，都在這同一個行程、同一份環境裡跑
（逐位比較要同一條路徑）。N 不寫死：先跑參考、記下每一次查停止時帳本最後一列的階段，每一種階段取第一次出現的那一次。
另外把參考帳本截成「前段只寫了一半」的幾份接續，與標頭被改過的一份接續（要拒收）。
空氣（修補二、三）：方案場景的密度與聲速換成另一組、替身換成用同一組空氣建的那份（父行程建好、第二個參數給路徑），
重建的目標在樣板起點的值要逐位等於這裡自己用設定層配方（``construction_environment``）造的那一份，並跟原方案原替身
（空氣等於產品預設）重建的不同。
"""
from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient import cli
from aosr.gradient.material_gradient import ChainObjective, baseline_theta, chain_objective
from aosr.gradient.material_gradient_ledger import read_ledger
from aosr.gradient.run import (
    LEDGER_FILE, PROJECT_FILE, STATUS_FILE, STOP_FILE, check_inputs, file_sha256, load_problem, manifest_sha256,
    read_project, stop_requested,
)
from aosr.physics.fem_reduced_receipt import receipt_path
from aosr.physics.fem_reduced_store import load_reduced
from aosr.scoring.stage_one_objective_jax import objective_constants
from aosr.scoring.stage_one_score import stage_one_score_settings
from aosr.gradient.worker import run_worker

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
    return {"folder": folder.name, "code": code, "status": _status(folder), "checks": len(phases)}, phases


def _stopper(folder: Path, at: int, record: Record) -> Callable[[], bool]:
    """第 ``at`` 次查停止之前，用產品的 ``stop`` 命令放記號，再交給產品的判斷（看檔在不在）。"""
    calls = [0]
    check = stop_requested(folder)

    def should_stop() -> bool:
        calls[0] += 1
        if calls[0] == at:
            record["stop_command_code"] = cli.main(["stop", str(folder)])
        return check()

    return should_stop


def _stop_and_resume(directory: Path, source: Path, at: int, phase: str) -> Record:
    folder = _folder(directory, f"stop-{at:05d}", source)
    record: Record = {"folder": folder.name, "at": at, "phase": phase}
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
    return {"folder": folder.name, "keep": keep, "code": code, "status": _status(folder)}


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
    score = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), changed.purpose)
    constants = objective_constants(np.asarray(model.frequencies_hz), score, scheme.receiver_set,
                                    len(model.source_names))
    air, ranges = construction_environment(density_kg_m3=density, sound_speed_m_s=speed)
    expected = chain_objective(model, problem.designs, constants, air=air, ranges=ranges,
                               frequencies=model.frequencies_hz)
    theta = jnp.asarray(baseline_theta(problem.designs))

    def value(objective: ChainObjective) -> str:
        return float(objective(theta)[0]).hex()

    return {"rebuilt": value(problem.objective),
            "expected": value(expected), "original": value(load_problem(project).objective)}


def main(directory: Path, air_surrogate: Path) -> None:
    source = directory / "source"
    reference, phases = _reference(directory, source)
    reference_folder = directory / str(reference["folder"])
    firsts = {phase: index + 1 for index, phase in reversed(list(enumerate(phases)))}
    chosen = sorted({*firsts.values(), len(phases)})
    stops = [_stop_and_resume(directory, source, at, phases[at - 1]) for at in chosen]
    _, rows = read_ledger(reference_folder / LEDGER_FILE)
    pre_loop = [index for index, row in enumerate(rows) if row.round is None]
    snap = next(index for index, row in enumerate(rows) if row.phase == "snap")
    keeps = sorted({max(1, snap // 2), snap, snap + 1, (snap + 1 + len(pre_loop)) // 2})
    cuts = [_cut(directory, source, reference_folder, f"cut-{keep:04d}", keep) for keep in keeps if keep < len(pre_loop)]
    output = {"reference": reference, "phases": phases, "stops": stops, "cuts": cuts,
              "pre_loop_rows": len(pre_loop), "other_header": _other_header(directory, source, reference_folder),
              "air": _air(directory, source, air_surrogate)}
    (directory / "output.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(directory / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
