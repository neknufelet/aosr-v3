"""#1084 分區＋實測阻抗表照出正式報表：逐區接縫表的形狀、雙精度子行程裡的同算法與分區對照、冷入口端到端。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from aosr.config.paths import config_path
from aosr.materials.construction_models import MeasuredImpedance
from aosr.physics.report_zone_seams import ZoneSeamTable
from aosr.reporting.result import PairResult, SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from aosr.scoring.contract import EvaluationState
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._zoned_report_cases import small_scheme, split_x0, with_x0
from tests.engine.test_measured_impedance import measured

TABLE = MeasuredImpedance.model_validate(measured())


def _seam_json(frequency_hz: float) -> dict[str, object]:
    values = {"real_pa_s_m": 400.0, "imag_pa_s_m": -200.0, "normal_absorption": 0.5, "random_absorption": 0.5}
    return {"frequency_hz": frequency_hz, "measured": values, "base_assembly": values}


@pytest.mark.parametrize("table,message", [
    ({}, "六面牆名"), ({"x9": {"a": [_seam_json(100.0), _seam_json(1000.0)]}}, "六面牆名"),
    ({"x0": {}}, "不可為空"), ({"x0": {"": [_seam_json(100.0), _seam_json(1000.0)]}}, "不可為空"),
    ({"x0": {"a": [_seam_json(1000.0), _seam_json(100.0)]}}, "低、高"),
    ({"x0": {"a": [_seam_json(100.0), _seam_json(100.0)]}}, "低、高"),
])
def test_zone_seam_table_refuses_bad_shapes_on_read(table: dict[str, object], message: str) -> None:
    """讀回存檔也守形狀：抓空表、錯牆名、空區、空代號、兩端顛倒或同頻。"""
    with pytest.raises(ValidationError, match=message):
        TypeAdapter(ZoneSeamTable).validate_python(table)
    good = {"x0": {"a": [_seam_json(100.0), _seam_json(1000.0)]}}
    assert set(TypeAdapter(ZoneSeamTable).validate_python(good)["x0"]) == {"a"}


def _probe(case: str, tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = forward_cpu_pins(child_process_env(threads=1))
    env["TMPDIR"] = str(tmp_path)
    return subprocess.run([sys.executable, "-B", "-m", "tests.engine._zoned_measured_seams_probe", case, *arguments],
        cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True, text=True, timeout=240)


@pytest.mark.parametrize("case", ("full", "only", "image", "late", "routing", "context"))
def test_zoned_measured_arithmetic_in_precision_process(case: str, tmp_path: Path) -> None:
    """同算法、只報矩形實測區、兩區同表＝整面（鏡像、晚期）、整面與逐區各報一處、錯誤說出哪一區；細節在探針各段的說明。"""
    child = _probe(case, tmp_path)
    assert child.returncode == 0, child.stdout + child.stderr


def test_scheme_cli_zone_with_measured_table_reports_zone_seams(tmp_path: Path) -> None:
    """冷入口真算（雙精度子行程）：x0 的 a 區貼實測表照出六類與逐區接縫；同一面牆兩處都報要在讀回時拒收。

    空氣刻意不用 1.2／343：逐對結果的接縫要用方案自己的空氣（寫死或對調都要紅），答案由探針獨立算整面那一份。
    """
    constant = small_scheme(zoned=True).scene.surface_zones_by_wall
    assert constant is not None
    base = with_x0(split_x0(TABLE, constant["x0"][0].material))
    scheme = Scheme.model_validate(base.model_copy(update={"scene": base.scene.model_copy(
        update={"density_kg_m3": 1.19, "sound_speed_m_s": 344.5})}).model_dump())
    source, out = tmp_path / "scheme.json", tmp_path / "result.json"
    source.write_text(scheme.model_dump_json())
    run = subprocess.run([sys.executable, "-m", "aosr.reporting.scheme_cli", "run", str(source), "--out", str(out),
        "--engine-commit", "control", "--run-date", "2026-11-11", "--capabilities", str(config_path("capabilities.toml"))],
        env=forward_cpu_pins(child_process_env(threads=1)), capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout+run.stderr
    result = SchemeResult.model_validate_json(out.read_bytes())
    answer = _probe("seams_for", tmp_path, str(source), str(tmp_path / "expected.json"))
    assert answer.returncode == 0, answer.stdout + answer.stderr
    expected = json.loads((tmp_path / "expected.json").read_text())
    assert len(result.candidate.evaluations) == 6
    assert all(e.state is EvaluationState.MEASURED for e in result.candidate.evaluations)
    assert len(result.pairs) == 14
    for item in result.pairs:
        assert item.report.measured_seams_by_wall is None
        assert item.measured_seams_by_zone is not None and set(item.measured_seams_by_zone) == {"x0"}
        low, high = item.measured_seams_by_zone["x0"]["a"]
        assert (low.frequency_hz, high.frequency_hz) == (100.0, 1000.0)
        assert (low.measured.real_pa_s_m, low.measured.imag_pa_s_m) == (400.0, -200.0)
        assert (high.measured.real_pa_s_m, high.measured.imag_pa_s_m) == (800.0, 100.0)
        assert [low.model_dump(mode="json"), high.model_dump(mode="json")] == expected
        assert "measured_table_model_fill" in {note.code for note in (item.report.construction_notes_by_wall or {})["x0"]}
    first = result.pairs[0]
    assert first.measured_seams_by_zone is not None
    seams = [seam.model_dump(mode="json") for seam in first.measured_seams_by_zone["x0"]["a"]]
    twice = first.model_dump(mode="json")
    twice["report"]["measured_seams_by_wall"] = {"x0": seams}
    with pytest.raises(ValidationError, match="同時出現在整面與逐區接縫"):
        PairResult.model_validate(twice)
