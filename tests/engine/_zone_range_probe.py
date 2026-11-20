"""#1199 雙精度子行程：照 ``project.json`` 重建問題（工人開跑與接續都走的 ``run.load_problem``），記帳本標頭與換樣板那一側
拿到的可調範圍、x0 那一片起跑設計的界；``ledger`` 照重建的標頭寫一本只有標頭的帳；``resume`` 走真的接續入口
（``run.resume_search``），記它拒收的那一句（標頭跟帳本對不上時在跑任何一步之前就拒收）。開跑用
``_material_gradient_carryover_probe same``（真的 ``cli.main(["start", …])``，寫完 ``project.json`` 就停）。

``start``（修補 1，起點投影進範圍）：真的工人開跑入口（``run.run_search``）與接續入口（``run.resume_search``）各走一次，
走到交給連續下降那一步就停（下降換成記下第一個求值 θ 的樁，不求值），記兩次的第一個 θ；再記顯卡把關那一支
（``gpu_preflight._start_theta``）照 ``project.json`` 重建出來的起點，以及命令列沿用抽查（``cli._carryover_spot_check``，
抽查本身換成記下阻抗的樁）拿到的 x0 那一片阻抗與照手算投影後那一組構造（棉 0.1、空腔 0.05）算的阻抗。"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, gpu_preflight, run
from aosr.gradient import material_gradient as search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient.depth_fit import ZoneRange
from aosr.gradient.endpoint import material_impedances
from aosr.materials.construction_models import AirLayer, Assembly
from aosr.physics import fem_reduced_check
from aosr.physics.fem_reduced import FullOperators, impedance_rows
from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances


def _ranges(value: Mapping[str, Mapping[str, ZoneRange]] | None) -> dict[str, dict[str, object]]:
    return {face: {family: given.model_dump(mode="json") for family, given in families.items()}
            for face, families in (value or {}).items()}


def problem(directory: Path) -> dict[str, object]:
    try:
        rebuilt = run.load_problem(run.read_project(directory))
    except ValueError as error:
        return {"error": str(error)}
    x0 = next(design for design in rebuilt.designs if design.surface_id == "x0")
    return {"header": _ranges(rebuilt.header.ranges), "templates": _ranges(rebuilt.templates.ranges),
            "x0": [list(x0.lower), list(x0.upper), list(x0.legal_thicknesses_m)]}


def write_header(directory: Path) -> dict[str, object]:
    rebuilt = run.load_problem(run.read_project(directory))
    ledger.write_ledger(directory / run.LEDGER_FILE, rebuilt.header, ())
    return {"written": True}


def resume(directory: Path) -> dict[str, object]:
    try:
        run.resume_search(directory, should_stop=lambda: True)
    except ValueError as error:
        return {"error": str(error)}
    return {"error": None}


def same_header(directory: Path) -> dict[str, object]:
    rebuilt = run.load_problem(run.read_project(directory))
    try:
        ledger.read_ledger(directory / run.LEDGER_FILE, expected_header=rebuilt.header)
    except ValueError as error:
        return {"error": str(error)}
    return {"error": None}


class _Reached(Exception):
    def __init__(self, theta: search.FloatVector) -> None:
        super().__init__("到了連續下降那一步")
        self.theta = theta


def _first_theta(entry: Callable[[], object]) -> list[str]:
    """真的入口走到 ``projected_descent`` 就停：記下降真正第一個求值的 θ（``descent_start``，同下降本身那一支）。"""
    def reached(objective: object, theta0: search.FloatVector, bounds: tuple[search.FloatVector, search.FloatVector],
                settings: object, **kwargs: object) -> None:
        limits = cast(Sequence[search.DepthLimit], kwargs.get("limits", ()))
        raise _Reached(np.asarray(search.descent_start(theta0, bounds, limits)))

    with patch.object(run, "projected_descent", reached):
        try:
            entry()
        except _Reached as caught:
            return [float(value).hex() for value in caught.theta]
    raise AssertionError("沒走到連續下降")


def _spot_rows(project: run.GradientProject, designs: tuple[search.SurfaceDesign, ...]) -> dict[str, object]:
    scheme, model, _ = run.check_inputs(project)
    seen: list[tuple[FrequencyBoundaryImpedances, tuple[float, ...]]] = []

    def spot(_model: object, _operators: object, impedances: FrequencyBoundaryImpedances, axis: Sequence[float],
             **kwargs: object) -> SimpleNamespace:
        seen.append((impedances, tuple(axis)))
        return SimpleNamespace(passed=True, max_db=0.0)

    with patch.object(fem_reduced_check, "spot_check_reduced", spot):
        cli._carryover_spot_check(model, cast(FullOperators, None), designs, scheme, project.ranges)
    (rows, axis), = seen
    material = next(design.material for design in designs if design.surface_id == "x0")
    assert material is not None
    reference = Assembly(layers=(material.layers[0].model_copy(update={"thickness_m": 0.1}), AirLayer(thickness_m=0.05)),
                         backing="rigid")
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3, sound_speed_m_s=scheme.scene.sound_speed_m_s)
    expected = material_impedances({wall: reference for wall in Wall.all()}, axis, air=air, ranges=ranges)
    actual = np.asarray(impedance_rows([rows], axis)[0][(Wall.X0, "")], dtype=np.complex128)
    wanted = np.asarray(impedance_rows([expected], axis)[0][(Wall.X0, "")], dtype=np.complex128)
    return {"actual": [actual.real.tolist(), actual.imag.tolist()], "reference": [wanted.real.tolist(), wanted.imag.tolist()]}


def start(directory: Path) -> dict[str, object]:
    try:
        fresh = _first_theta(lambda: run.run_search(directory, should_stop=None))
        again = _first_theta(lambda: run.resume_search(directory, should_stop=None))
    except ValueError as error:
        return {"error": str(error)}
    project = run.read_project(directory)
    rebuilt = run.load_problem(project)
    _, rows = ledger.read_ledger(directory / run.LEDGER_FILE, expected_header=rebuilt.header)
    return {"names": list(rebuilt.header.variable_names), "run": fresh, "resume": again, "rows": len(rows),
            "gpu": [value.hex() for value in gpu_preflight._start_theta(rebuilt)],
            "spot": _spot_rows(project, rebuilt.designs)}


if __name__ == "__main__":
    mode, location = sys.argv[1:]
    modes = {"problem": problem, "ledger": write_header, "resume": resume, "same": same_header, "start": start}
    folder = Path(location)
    (folder / "probe.json").write_text(json.dumps(modes[mode](folder), ensure_ascii=False), encoding="utf-8")
    raise SystemExit(0)
