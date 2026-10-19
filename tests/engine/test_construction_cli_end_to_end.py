"""冷正式命令列與真有限元素：同場景換牆材料，三路各自必須改變。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import aosr
import pytest
from governance.exit_codes import note
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.physics.report_io import ReportOutput
from aosr.runtime import ENABLE_X64_VAR, child_process_env
from tests.engine._surface_zone_cases import ASSEMBLY


def documents() -> tuple[dict[str, object], dict[str, object]]:
    fixed: dict[str, object] = {
        "room_m": {"Lx": 1.6, "Ly": 1.2, "Lz": 1.0},
        "source_model": {"kind": "omnidirectional"},
        "source_m": {"x": .3, "y": .3, "z": .4},
        "receiver_m": {"x": 1.2, "y": .7, "z": .6},
        "sound_speed_m_s": 343.0, "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {wall.wall_name(): 1646.4 for wall in Wall.all()},
        "scattering_by_wall": {wall.wall_name(): .2 for wall in Wall.all()},
    }
    construction = dict(fixed)
    construction.pop("impedance_pa_s_per_m_by_wall")
    construction["surface_zones_by_wall"] = {
        wall.wall_name(): [{"area": "whole_wall", "material": ASSEMBLY if wall is Wall.X0 else {
            "kind": "constant_impedance", "impedance_pa_s_m": 1646.4}}] for wall in Wall.all()}
    return fixed, construction


def command(path: Path) -> list[str]:
    return [sys.executable, "-B", "-m", "aosr.physics.three_lane_report_cli", str(path),
            "--format", "json", "--points", "--capabilities", str(config_path("capabilities.toml"))]


def environment(tmp_path: Path) -> dict[str, str]:
    env = child_process_env(threads=1)
    pins = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")
    env.update({name: os.environ[name] for name in pins if name in os.environ})
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(tmp_path), "1"
    assert ENABLE_X64_VAR not in env
    return env


@pytest.mark.parametrize("kind", ("board", "fabric", "micro_perforated", "perforated", "slotted"))
def test_board_cavity_wall_changes_every_lane(tmp_path: Path, kind: str) -> None:
    env = environment(tmp_path)
    loaded = subprocess.run([sys.executable, "-B", "-c", "import aosr, sys; sys.stdout.write(aosr.__file__)"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert Path(loaded.stdout.strip()).resolve() == Path(aosr.__file__).resolve()
    note(f"子行程 aosr.__file__：{loaded.stdout.strip()}")
    from tests.engine.test_construction_facings_models import FACING_CASES
    from tests.engine.test_construction_facings import BACKING
    cases = documents()
    if kind != "board":
        zones = cases[1]["surface_zones_by_wall"]
        assert isinstance(zones, dict)
        layer = FACING_CASES[kind] | (BACKING if kind in ("perforated", "slotted") else {})
        zones["x0"][0]["material"] = {"kind": "assembly", "layers": [layer, {"kind": "air", "thickness_m": .03}], "backing": "rigid"}
    paths = []
    for name, document in zip(("A", "B"), cases, strict=True):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        paths.append(path)
    before = time.perf_counter()
    children: list[subprocess.Popen[str]] = []
    outputs = []
    try:  # 第二支啟動失敗時，已起的那支也要進清理
        for path in paths:
            children.append(subprocess.Popen(command(path), cwd=tmp_path, env=env,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for child in children:
            stdout, stderr = child.communicate(timeout=max(.1, 120 - (time.perf_counter() - before)))
            assert child.returncode == 0, stderr
            outputs.append(ReportOutput.model_validate_json(stdout))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    note(f"A／B 同時起牆鐘：{time.perf_counter() - before:.3f} 秒")
    a, b = outputs
    assert b.capability.status == "experimental"
    assert a.top.room_volume_m3 == b.top.room_volume_m3
    assert a.scene.scene_fingerprint != b.scene.scene_fingerprint
    assert a.points is not None and b.points is not None
    left = {p.frequency_hz: p.fem_energy for p in a.points if p.fem_energy is not None}
    right = {p.frequency_hz: p.fem_energy for p in b.points if p.fem_energy is not None}
    common = left.keys() & right.keys()
    assert common
    assert all(left[f] != right[f] for f in common)
    for old, new in zip(a.bands, b.bands, strict=True):
        assert old.center_frequency_hz == new.center_frequency_hz
        for field in ("reflected_energy", "interference_energy", "late_energy"):
            assert getattr(old, field) != getattr(new, field), (field, old.center_frequency_hz)
    assert a.top.eyring_t60_by_band_s.keys() == b.top.eyring_t60_by_band_s.keys()
    assert all(value != b.top.eyring_t60_by_band_s[f] for f, value in a.top.eyring_t60_by_band_s.items())
    assert a.construction_notes_by_wall is None
    assert b.construction_notes_by_wall is None
    assert "construction_notes_by_wall" not in b.model_dump(mode="json")


@pytest.mark.parametrize("sigma", (60000., 4000.))
def test_porous_extrapolation_survives_report_and_text(tmp_path: Path, sigma: float) -> None:
    _fixed, document = documents()
    zones = document["surface_zones_by_wall"]
    assert isinstance(zones, dict)
    zones["x0"] = [{"area": "whole_wall", "material": {
        "kind": "assembly", "layers": [{"kind": "porous", "thickness_m": .05,
            "flow_resistivity_pa_s_m2": sigma, "material_class": "foam"}], "backing": "rigid"}}]
    path = tmp_path / "C.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    probe = Path(__file__).with_name("_construction_report_probe.py")
    env = environment(tmp_path)
    results = {}
    before = time.perf_counter()
    for format_name in ("json", "text"):
        child = subprocess.run([sys.executable, "-B", str(probe), str(path), format_name],
                               cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
        assert child.returncode == 0, child.stderr
        results[format_name] = child.stdout
    report = ReportOutput.model_validate_json(results["json"])
    assert report.construction_notes_by_wall is not None
    assert set(report.construction_notes_by_wall) == {"x0"}
    notes = report.construction_notes_by_wall["x0"]
    assert {item.layer_index for item in notes} == {1}
    by_code = {item.code: item for item in notes}
    extrapolation = by_code["komatsu_frequency_extrapolation"]
    from aosr.config.construction_environment import construction_environment
    _air, ranges = construction_environment(density_kg_m3=1.2, sound_speed_m_s=343.)
    lower = ranges.komatsu.f_over_sigma_bounds_m3_kg[0] * 60000.
    assert extrapolation.frequencies_hz
    if sigma == 60000.:
        assert all(f < lower for f in extrapolation.frequencies_hz)
    else:
        assert all(f > 5000 / 6000 * 4000 for f in extrapolation.frequencies_hz)
        assert by_code["komatsu_flow_below_samples"].frequencies_hz == ()
        assert "流阻低於原研究樣品範圍" in results["text"]
    assert by_code["komatsu_material_class_outside_fit"].frequencies_hz == ()
    for item in notes:
        assert item.message in results["text"]
    assert "x0 第 1 層" in results["text"]
    assert "外推頻點" in results["text"]
    note(f"C 外推頻點：{extrapolation.frequencies_hz}；C 兩格式牆鐘 {time.perf_counter() - before:.3f} 秒")
