"""命令列分片考卷共用的小房間與慢速物理替身；所有輸出交給 tmp_path。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from aosr.config import frequency_axis
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import three_lane_report
from aosr.reporting import fem_slices, physics_stage, scheme_cli
from tests.engine import _fem_shared_cases as cases
from tests.engine import _scoring_source_model_control as control
from tests.engine.test_scheme_pipeline import _many_fem


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], list[str]]:
    report_axis = cases.FREQUENCIES + tuple(
        value for value in frequency_axis.GEOMETRIC_LANE_FREQUENCIES_HZ
        if value > frequency_axis.FEM_GEOMETRIC_CROSSOVER_CAP_HZ)
    monkeypatch.setattr(frequency_axis, "FEM_LANE_FREQUENCIES_HZ", cases.FREQUENCIES)
    monkeypatch.setattr(frequency_axis, "GEOMETRIC_LANE_FREQUENCIES_HZ", report_axis)
    for module, name, fake in control.STAND_INS:
        monkeypatch.setattr(module, name, fake)
    monkeypatch.setattr(physics_stage, "report_capability", lambda table: three_lane_report._unchecked_capability())
    monkeypatch.setattr(fem_slices, "solve_fem_energies_many", fake_slices)
    paths = []
    for name in ("trial", "other"):
        scheme = cases.small_scheme(name).model_copy(update={"purpose": control.PURPOSE})
        path = tmp_path / name
        path.write_text(scheme.model_dump_json(), encoding="utf-8")
        paths.append(path)
    common = ["--capabilities", str(control.TARGETS.with_stem("capabilities")),
              "--engine-commit", "control"]
    return paths, common


def fake_slices(
    *, room: Room,
    candidates: Mapping[str, tuple[Mapping[str, Point], Mapping[str, Point]]],
    wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
    density_kg_m3: float, sound_speed_m_s: float, frequency_indices: Sequence[int],
) -> dict[str, dict[tuple[str, str], tuple[float, ...]]]:
    """沿用主管線替身，逐索引擷取；分片模型與合併仍走真實程式。"""
    return {name: {pair: tuple(values[index] for index in frequency_indices)
                   for pair, values in _many_fem(
                       room=room, sources=sources, receivers=receivers,
                       wall_impedances=wall_impedances, frequencies_hz=frequencies_hz,
                       density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s).items()}
            for name, (sources, receivers) in candidates.items()}


def slice_args(paths: list[Path], common: list[str], out: Path, index: int, slices: int) -> list[str]:
    return ["fem-slice", *map(str, paths), "--slice", str(index), "--slices", str(slices),
            "--out", str(out), *common]


def parts(tmp_path: Path, paths: list[Path], common: list[str]) -> list[Path]:
    slices = len(cases.FREQUENCIES) // 2
    outputs = [tmp_path / f"part-{index}" for index in range(slices)]
    for index, out in enumerate(outputs):
        code = scheme_cli.main(slice_args(paths, common, out, index, slices))
        assert code == 0
    return outputs


def run_args(path: Path, common: list[str], out: Path, shards: list[Path] | None = None) -> list[str]:
    return ["run", str(path), "--out", str(out), "--run-date", "2026-09-27", *common,
            *([] if shards is None else ["--fem-parts", *map(str, shards)])]


def forbid_fem(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """退回任何有限元素入口都記錄並報錯，讓拒收與注入考卷能確認零次求解。"""
    calls: list[str] = []
    def forbidden(**kwargs: object) -> object:
        calls.append("fem")
        raise AssertionError("帶分片不得求解有限元素")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", forbidden)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", forbidden)
    monkeypatch.setattr(fem_slices, "solve_fem_energies_many", forbidden)
    return calls
