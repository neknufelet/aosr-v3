"""每區准用清單＋固定材料進搜尋（#1082 段 2-3b）的雙精度考卷：判決在父行程，計算在 ``_material_gradient_allowed_probe``。

- 整段換樣板：x0 被接受換成實測表（整塊、沒有變數）、下一輪從那個狀態再評分（換回家族也評到），每一列只換到清單裡的、
  帳本照設定與清單讀得回、每一個列界接續跟一口氣跑完逐列相同；最後配置照終點匯出的材料就是那份實測表。
  帳本標頭的清單被改掉（實測表不准用了），讀的人拒收。
- 搜尋頻點與終點同一份表：搜尋鏈的常數列跟終點求阻抗逐位相同（含表外補值的頻點）。
- 替身留出驗證：小房間真的完整解；軟實測表與 ≥75ρc 硬實測表的數字照記、都在尺內才算過；截斷到很小秩的替身一定拒收、
  訊息帶差值與兩把尺的名字。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.runtime import child_process_env, configure_jax
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._precision_contracts import contract_value
from tests.engine.test_material_gradient_allowed import CARPET, GAP50, KEPT, MEASURED, _measured

PROBE = Path(__file__).with_name("_material_gradient_allowed_probe.py")
Record = dict[str, object]
RHO_C = 1.2 * 343.0
HARD = "panel_hard"
SOFT = MEASURED
ALLOWED = {"x0": ["glass_wool_48k", "fixed:" + MEASURED], "ceiling": ["glass_wool_48k", "fixed:" + CARPET, "none"]}
# 偏移表：x0 換實測表一定好 1（第 0 輪接受）、天花板不貼一定好 0.6（第 1 輪接受）；其餘只差二次式那幾項。
OFFSETS = {"x0->fixed:" + MEASURED: -1.0, "ceiling->none": -0.6}
START: list[list[str | None]] = [["floor", None], ["ceiling", GAP50], ["x0", GAP50], ["xL", GAP50], ["y0", GAP50],
                                 ["yL", GAP50]]


def _hard() -> Record:
    """≥75ρc 的硬實測表（80ρc 實部、小虛部）：#990 光牆替身沒過契約的那一類。"""
    document = _measured().model_dump(mode="json")
    rows = [{"frequency_hz": frequency, "real_pa_s_m": 80.0 * RHO_C, "imag_pa_s_m": -RHO_C * 100.0 / frequency}
            for frequency in (20.0, 100.0, 400.0, 1000.0)]
    return {**document, "id": HARD, "label": "硬牆實測（自填）", "material": {**document["material"], "points": rows}}


def _problem() -> Record:
    return {"user_fixed": [_measured().model_dump(mode="json"), _hard()], "allowed": ALLOWED, "offsets": OFFSETS,
            "surfaces": START, "kept": KEPT, "measured_id": MEASURED,
            "axis": [16.0, 31.5, 50.0, 75.0, 100.0, 160.0, 250.0, 400.0, 630.0],
            "truncated_rank": 3}


def _run(directory: Path, part: str, problem: Record) -> Record:
    (directory / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    env = forward_cpu_pins(child_process_env(threads=1))
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    process = subprocess.run([sys.executable, str(PROBE), str(directory), part], cwd=directory, env=env,
                             capture_output=True, text=True, timeout=1500)
    assert process.returncode == 0, process.stdout + process.stderr
    record = cast(Record, json.loads((directory / f"output-{part}.json").read_text(encoding="utf-8")))
    assert record["x64"] == "float64"
    return record


@pytest.fixture(scope="module")
def segment_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Record]:
    directory = tmp_path_factory.mktemp("allowed-segment")
    return directory, _run(directory, "segment", _problem())


def _without_seconds(rows: tuple[templates.TemplateRow, ...]) -> list[Record]:
    return [{key: value for key, value in row.model_dump(mode="json").items() if key != "seconds"} for row in rows]


def _start(record: Record) -> templates.TemplateStart:
    start = cast(Record, record["start"])
    surfaces = tuple((str(item[0]), item[1]) for item in cast(list[list[str | None]], start["surfaces"]))
    return templates.TemplateStart(surfaces, tuple(cast(list[str], start["theta_hex"])), float.fromhex(str(start["value"])),
                                   float.fromhex(str(start["screening_total"])),
                                   tuple(float.fromhex(value) for value in cast(list[str], start["components"])))


def test_segment_accepts_a_fixed_material_then_keeps_swapping_inside_the_lists(segment_run: tuple[Path, Record]) -> None:
    """抓固定材料進不了候選、接受後下一輪從它換不走（或換到清單外）、帳本記不下固定材料那一面。"""
    from tests.engine._material_gradient_templates_probe import settings_with
    from aosr.config.material_gradient import load_material_gradient_settings
    from aosr.config.paths import config_path

    directory, record = segment_run
    settings = settings_with(load_material_gradient_settings(config_path("material_gradient.toml")),
                             {"templates": {"max_rounds": 3}})
    header, rows = templates.read_template_ledger(directory / str(record["ledger"]), settings=settings,
                                                  start=_start(record))
    decisions = [row for row in rows if row.decision is not None]
    assert [row.decision.accepted for row in decisions if row.decision is not None][:2] == [
        "x0->fixed:" + MEASURED, "ceiling->none"]
    second_round = {row.candidate for row in rows if row.round == 1 and row.phase == "quick_score"}
    assert "x0->glass_wool_48k" in second_round
    assert {candidate for candidate in second_round if candidate and candidate.startswith("x0->")} == {"x0->glass_wool_48k"}
    assert {candidate for candidate in second_round if candidate and candidate.startswith("ceiling->")} == {
        "ceiling->fixed:" + CARPET, "ceiling->none"}
    for row in rows:
        for face, preset in row.surfaces:
            listed = ALLOWED.get(face)
            if listed is not None:
                assert rounds.target_of(preset, settings) in listed
    final = dict((str(face), preset) for face, preset in cast(list[list[str | None]], record["result_surfaces"]))
    assert final["x0"] == "fixed:" + MEASURED and final["ceiling"] is None
    assert header.allowed_targets == {face: tuple(targets) for face, targets in ALLOWED.items()}


def test_every_row_boundary_resumes_to_the_same_ledger(segment_run: tuple[Path, Record]) -> None:
    directory, record = segment_run
    whole = _without_seconds(templates.read_template_ledger(directory / str(record["ledger"]))[1])
    for name in cast(list[str], record["resumed"]):
        assert _without_seconds(templates.read_template_ledger(directory / name)[1]) == whole, name


def test_endpoint_export_is_the_measured_table_itself(segment_run: tuple[Path, Record]) -> None:
    """抓終點匯出時固定材料那一面變成原牆、剛性或樣板。"""
    record = segment_run[1]
    exported = cast(dict[str, object], record["exported"])
    assert exported["x0"] == _measured().surface_material().model_dump(mode="json")
    assert cast(dict[str, object], record["spec_presets"])["x0"] == "fixed:" + MEASURED


def test_ledger_with_the_fixed_material_struck_from_the_list_is_refused(segment_run: tuple[Path, Record],
                                                                        tmp_path: Path) -> None:
    """抓讀帳本的人不核清單（標頭說 x0 不准用實測表，列裡卻換了也照收）。"""
    from tests.engine._material_gradient_templates_probe import settings_with
    from aosr.config.material_gradient import load_material_gradient_settings
    from aosr.config.paths import config_path

    directory, record = segment_run
    lines = (directory / str(record["ledger"])).read_bytes().splitlines(keepends=True)
    head = json.loads(lines[0])
    head["allowed_targets"]["x0"] = ["glass_wool_48k", "none"]
    head["allowed_targets"]["ceiling"] = ["glass_wool_48k", "none"]
    head["fixed_candidate_digests"] = {}
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_bytes(b"".join([(json.dumps(head, ensure_ascii=False) + "\n").encode("utf-8"), *lines[1:]]))
    settings = settings_with(load_material_gradient_settings(config_path("material_gradient.toml")),
                             {"templates": {"max_rounds": 3}})
    with pytest.raises(ValueError, match="標頭沒記這份固定材料|不在准用清單"):
        templates.read_template_ledger(tampered, settings=settings)


def test_search_rows_and_endpoint_read_the_same_table_bit_for_bit(tmp_path: Path) -> None:
    """抓搜尋頻點與終點各自取表（內插或表外補值兩邊不同）。軸含表外（16、31.5 Hz 與 630 Hz）。"""
    record = _run(tmp_path, "rows", _problem())
    assert record["search"] == record["endpoint"]


def test_holdout_gate_records_passing_tables_and_refuses_a_poor_surrogate(tmp_path: Path) -> None:
    """抓沒做留出驗證就開跑、驗了不照兩把尺判、拒收時不講數字。硬實測表（80ρc）在這間小房的替身上照實記數字。"""
    problem = {**_problem(), "allowed": {"x0": ["none", "fixed:" + SOFT, "fixed:" + HARD],
                                         "ceiling": ["none", "fixed:" + HARD]}}
    record = _run(tmp_path, "gate", problem)
    checks = cast(list[Record], record["checks"])
    frequency, band = contract_value("fem_reduced_holdout_frequency_db"), contract_value("fem_reduced_holdout_band_db")
    assert [check["target"] for check in checks] == ["fixed:" + HARD, "fixed:" + SOFT]
    assert checks[0]["surfaces"] == ["ceiling", "x0"] and checks[1]["surfaces"] == ["x0"]
    for check in checks:
        assert 0.0 <= cast(float, check["frequency_db"]) <= frequency and 0.0 <= cast(float, check["band_db"]) <= band
        assert (check["frequency_ruler"], check["band_ruler"]) == (frequency, band)
        assert cast(int, check["full_solves"]) > 0
    error = cast(str, record["truncated_error"])
    assert error is not None and "沒過留出驗證" in error
    assert "fem_reduced_holdout_frequency_db" in error and "fem_reduced_holdout_band_db" in error and "dB" in error
