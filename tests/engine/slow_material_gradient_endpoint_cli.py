"""本機合併前驗收的搜尋與終點重考卷；考卷本體由原檔原樣搬移。"""
from __future__ import annotations
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from tests.engine._gradient_probe_env import endpoint_probe_env
from tests.engine._scheme_cache import bitwise_equal
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import endpoint_run, run
from aosr.gradient import template_ledger as templates
from aosr.gradient.endpoint_run import EndpointRecord, read_result
from aosr.gradient.report import (
    ENDPOINT_HEADING, FULL_MODEL_LINE, LEGACY_ENDPOINT_NOTE, LEGACY_SCORE_NOTE, PROXY_MISMATCH, STALE_ENDPOINT_LINE,
)
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, save_validation_receipt
from aosr.physics.fem_reduced_store import load_reduced, save_reduced
from aosr.reporting.scheme import load_scheme
from tests.engine._precision_contracts import contract_value
from tests.engine._scheme_cache import shared_json
from tests.engine._legacy_gradient_search_probe import legacy_endpoint_copy
from tests.engine._six_search_cli_case import ENDPOINT_ROOM, model as model, operators as operators
from tests.engine.test_material_gradient import _chain_model, _presets
from tests.engine.test_material_gradient_cli import (
    BAND_NAME, FREQUENCY_NAME, NARROWED_STEPS, PRESETS, WAIT_S, Record, _command, _current_identity,
    _good_report, _json, _kept_materials, _run, _schemes, _settings_text, _sha256, _start_args,
)

from tests.engine.test_material_gradient_endpoint_cli import (
    COMPUTED_FLOATS as COMPUTED_FLOATS,
    COMPUTED_PATHS as COMPUTED_PATHS,
    PROBE as PROBE,
    STALE_CASES as STALE_CASES,
    UNIT_SCALE as UNIT_SCALE,
    _PROXY_FIELDS as _PROXY_FIELDS,
    _batch_probe as _batch_probe,
    _by_wall as _by_wall,
    _compute as _compute,
    _downgraded_ledger as _downgraded_ledger,
    _endpoint_runs as _endpoint_runs,
    _entries as _entries,
    _folder as _folder,
    _inputs as _inputs,
    _kept_resume as _kept_resume,
    _leaves as _leaves,
    _no_axis as _no_axis,
    _other_theta as _other_theta,
    _pre_proxy as _pre_proxy,
    _probe as _probe,
    _probe_work as _probe_work,
    _record as _record,
    _report_lines as _report_lines,
    _search_runs as _search_runs,
    _section as _section,
    _stale as _stale,
    _surrogate as _surrogate,
    _swapped as _swapped,
    _with_field as _with_field,
    _with_theta as _with_theta,
    _within as _within,
    _without as _without,
    minimal_runs as minimal_runs,
    full_runs as runs,
)


def test_endpoint_writes_record_bound_to_result_and_ledger_files(runs: tuple[Path, Record]) -> None:
    """抓 endpoint 沒跑完、endpoint.json 讀不回、沒綁帳本檔或結果檔的摘要（任一變了認不出過期）：回 0、模型讀回、
    ``result_ledger_sha256`` 等於 result.json 記的與帳本檔量的、``result_sha256`` 等於結果檔量的；改善等於基準減終點；
    匯出方案檔的 SHA-256 等於考卷自己量的。"""
    case = _section(runs, "endpoint", "run")
    assert case["code"] == 0, case["stderr"]
    folder = _folder(runs)
    record, result = _record(folder), _json(folder / "result.json")
    assert record.result_ledger_sha256 == result["ledger_sha256"] == _sha256(folder / "ledger.jsonl")
    assert record.result_sha256 == _sha256(folder / "result.json")
    assert record.gradient_id == folder.name
    assert record.improvement.total == record.baseline.full_total - record.report.full_total
    assert record.improvement.components == tuple(
        base - end for base, end in zip(record.baseline.full_components, record.report.full_components, strict=True))
    base_objective, base_proxy = record.baseline.full_objective_total, record.baseline.full_proxy_db
    end_objective, end_proxy = record.report.full_objective_total, record.report.full_proxy_db
    assert base_objective is not None and base_proxy is not None and end_objective is not None and end_proxy is not None
    assert (record.improvement.objective_total, record.improvement.proxy_db) == (
        base_objective - end_objective, base_proxy - end_proxy)
    assert record.exported_scheme_sha256 == _sha256(folder / record.exported_scheme)

def test_endpoint_report_and_baseline_equal_direct_calls(runs: tuple[Path, Record]) -> None:
    """抓終點接錯輸入（θ、設計、評分設定、空氣、軸）、基準用了替身或不是方案原本的六面：探針同一行程裡產品入口
    （``run_endpoint_worker``）寫的 endpoint.json，報告逐位等於直接叫 ``endpoint_check``（同一份 result.json 的 θ、preset
    加考卷自己從舊欄換的原牆）的；基準值、兩分項、大峰逐位等於完整模型解原方案六面交同一把尺、同一支大峰的；
    完整解次數等於軸長。"""
    probe = _section(runs, "legacy_probe")
    assert probe["worker_code"] == 0
    record = EndpointRecord.model_validate(probe["worker_record"])
    probe = _section(runs, "legacy_probe", "endpoint")
    assert json.loads(record.report.to_json()) == probe["report"]
    assert record.baseline.full_total.hex() == probe["baseline_total"]
    assert [value.hex() for value in record.baseline.full_components] == probe["baseline_components"]
    assert [float(cast(float, value)).hex() for value in (
        record.baseline.full_objective_total, record.baseline.full_proxy_db,
        record.baseline.full_interior_max_db)] == probe["baseline_objective"]
    assert record.baseline.big_peak_db_by_speaker == probe["baseline_peaks"]
    assert record.baseline.below_scoring is not None
    assert record.baseline.below_scoring.model_dump(mode="json") == probe["baseline_below_scoring"]
    assert record.baseline.full_solves == len(FEM_LANE_FREQUENCIES_HZ)

def test_cli_endpoint_matches_entry_within_contract(runs: tuple[Path, Record]) -> None:
    """抓命令列終點跟產品入口算的不是同一件事：命令列 endpoint.json 跟探針產品入口那一份，非浮點（身分、摘要、檔名、
    完整解次數、喇叭與座位名、每面規格、丟掉的空腔、θ）逐字相等；終點、基準、改善的浮點每一格
    ``|差| ≤ 尺 × max(|探針那一格|, 1)``（尺的數字借自登記簿 ``stage_one_objective_jax_vs_python``；1 在 dB 欄是 1 dB、
    在無因次合分是 1；僅限本考卷的一次性例外，見模組說明）。同一行程的產品入口對直接呼叫另一題仍逐位比。
    不比秒數與完成時間。"""
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    cli = _leaves("", _without(_json(_folder(runs) / "endpoint.json")))
    entry = _leaves("", _without(cast(Record, _section(runs, "legacy_probe")["worker_record"])))
    assert cli.keys() == entry.keys()
    near = {key for key in cli if key.split("[", 1)[0].split(".")[1:3] in COMPUTED_PATHS}
    differing = {key: (cli[key], entry[key]) for key in cli.keys() - near if cli[key] != entry[key]}
    assert not differing, differing
    outside = {key: (cli[key], entry[key]) for key in near if not _within(cli[key], entry[key], tolerance)}
    assert not outside, outside
    assert near

def test_exported_scheme_keeps_original_walls_and_writes_specs(runs: tuple[Path, Record]) -> None:
    """抓匯出方案的「不貼」面寫成別的材料、有樣板的面不是終點那一套、或方案代號跟原方案一樣（會混淆）：
    ``scheme-endpoint.json`` 用 ``load_scheme`` 讀回；不貼的面材料摘要等於原方案那一面的；有樣板的面等於
    ``materials_from_surfaces``（同一份結果檔的 θ）那一面的摘要；``scheme_id`` 不等於原方案；房間、喇叭、座位不動。"""
    folder, original = _folder(runs), load_scheme(runs[0] / "scheme.json")
    exported = load_scheme(folder / _record(folder).exported_scheme)
    assert exported.scheme_id != original.scheme_id
    assert (exported.scene.room_m, exported.speakers, exported.receiver_set) == (
        original.scene.room_m, original.speakers, original.receiver_set)
    zones = exported.scene.surface_zones_by_wall
    assert zones is not None
    theta = np.asarray([float.fromhex(value) for value in cast(list[str], _json(folder / "result.json")["theta_hex"])])
    settings, kept = load_material_gradient_settings(runs[0] / "settings.toml"), _kept_materials(runs)
    designs = gradient_search.wall_designs_from_presets(PRESETS, settings, _presets(), kept=kept)
    materials = endpoint.materials_from_surfaces(designs, theta, settings)
    for wall, preset in zip(Wall.all(), PRESETS, strict=True):
        name = wall.wall_name()
        wanted = kept[name] if preset is None else materials[wall]
        assert wanted is not None
        assert endpoint.material_digest(zones[name][0].material) == endpoint.material_digest(wanted), name
    assert any(preset is None for preset in PRESETS) and any(preset is not None for preset in PRESETS)

def test_report_after_endpoint_prints_endpoint_section(runs: tuple[Path, Record]) -> None:
    """抓報告不讀 endpoint.json、或印了終點還留著「未做」那行、或沒改檔也印過期：有終點一節（替身、完整模型各自的
    舊合分／新合分／代理，完整模型另有真大峰與上界、「代理 − 內部最大殘差」與「真大峰」兩行、真大峰不大於代理、基準、改善、匯出方案），數字由考卷從
    endpoint.json 自己格式化；舊合分、新合分、代理、真大峰四個字樣都在（#561 壬一：舊分數與新分數並列）。"""
    lines = _report_lines(_section(runs, "endpoint", "report_after"))
    record = _record(_folder(runs))
    assert FULL_MODEL_LINE not in lines and STALE_ENDPOINT_LINE not in lines
    assert any(line.startswith(ENDPOINT_HEADING) for line in lines)
    report, baseline, gain = record.report, record.baseline, record.improvement
    assert report.reduced_objective_total is not None and report.reduced_proxy_db is not None
    assert report.full_objective_total is not None and report.full_proxy_db is not None
    assert report.proxy_gap_bound_db is not None and report.big_peak_db is not None
    assert baseline.full_objective_total is not None and baseline.full_proxy_db is not None
    assert gain.objective_total is not None and gain.proxy_db is not None
    assert (f"  替身：舊合分 {report.reduced_total:.6f}／新合分 {report.reduced_objective_total:.6f}／"
            f"代理 {report.reduced_proxy_db:.3f} dB") in lines
    assert any(line.startswith(
        f"  完整模型：舊合分 {report.full_total:.6f}／新合分 {report.full_objective_total:.6f}／代理 {report.full_proxy_db:.3f} dB"
        f"／真大峰 {report.big_peak_db:.3f} dB／上界 {report.proxy_gap_bound_db:.3f} dB") for line in lines)
    interior = report.full_interior_max_db
    assert interior is not None
    gap = report.full_proxy_db - interior
    assert (f"  終點代理 − 內部最大殘差 ＝ {gap:+.6f} dB（應在 0～上界 {report.proxy_gap_bound_db:.6f} dB 之間）") in lines
    assert any(line.startswith(f"  終點真大峰 {report.big_peak_db:.3f} dB（偵測器；低於內部最大殘差 {interior:.3f} dB")
               for line in lines)
    assert report.big_peak_db <= report.full_proxy_db
    assert any(f"基準（方案原本六面，完整模型）：舊合分 {baseline.full_total:.6f}／新合分 {baseline.full_objective_total:.6f}"
               in line for line in lines)
    assert any(f"改善（基準減終點，完整模型）：舊合分 {gain.total:+.6f}" in line
               and f"新合分 {gain.objective_total:+.6f}、代理 {gain.proxy_db:+.3f} dB" in line for line in lines)
    assert PROXY_MISMATCH not in "\n".join(lines)
    text = "\n".join(lines)
    assert all(word in text for word in ("舊合分", "新合分", "代理", "真大峰"))
    assert any(f"匯出方案：{record.exported_scheme}" in line for line in lines)

def test_endpoint_report_carries_proxy_fields_and_rejects_non_finite(runs: tuple[Path, Record]) -> None:
    """抓終點漏記新合分、代理或上界（#561 壬一：替身與完整模型各帶舊合分、新合分、代理，完整模型另帶真大峰與
    T·ln N）：這一版的 endpoint.json 五格都在、都有限、上界是正的；任一格換成 NaN 或無限大拒收；五格少一格拒收
    （一起有或一起沒有）；這一版的紀錄拿掉五格也拒收（只有上一版准沒有）。"""
    record = _record(_folder(runs))
    assert record.format_version == endpoint_run.ENDPOINT_VERSION
    values = record.report.proxy_fields()
    assert all(value is not None and math.isfinite(value) for value in values)
    assert cast(float, record.report.proxy_gap_bound_db) > 0.0
    document = record.report.model_dump()
    for name in _PROXY_FIELDS:
        for bad in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError):
                endpoint.EndpointReport.model_validate({**document, name: bad})
        with pytest.raises(ValueError):
            endpoint.EndpointReport.model_validate({key: value for key, value in document.items() if key != name})
    stripped = json.loads(record.model_dump_json())
    for name in _PROXY_FIELDS:
        stripped["report"].pop(name)
    with pytest.raises(ValueError):
        EndpointRecord.model_validate(stripped)

def test_pre_proxy_folder_still_reads_in_report_and_resume_is_rejected(runs: tuple[Path, Record]) -> None:
    """抓報告讀不回加代理之前的資料夾（第三張表要跟示範房那一場比），或接續照收舊帳本：示範房形狀的假資料夾
    （帳本 v4、換樣板帳本 v1、結果 v3、終點 v2、project.json 沒有梯度目標指紋）報告回 0、印舊帳本與舊終點的註記、
    終點不判過期、分數只印舊合分與真大峰（沒有「／新合分」可印）；接續回非 0、標準錯誤講「大峰代理」那一句、帳本不動。"""
    case = _section(runs, "pre_proxy")
    report = cast(Record, case["report"])
    assert report["code"] == 0, report["stderr"]
    lines = str(report["stdout"]).splitlines()
    assert any(LEGACY_SCORE_NOTE in line for line in lines)
    assert any(LEGACY_ENDPOINT_NOTE in line for line in lines)
    assert STALE_ENDPOINT_LINE not in lines and FULL_MODEL_LINE not in lines
    record = EndpointRecord.model_validate_json((Path(str(case["folder"])) / "endpoint.json").read_bytes())
    assert record.format_version == endpoint_run.LEGACY_ENDPOINT_VERSION
    assert any(f"舊合分 {record.report.full_total:.6f}" in line and "真大峰" in line for line in lines)
    assert not any("／新合分 " in line for line in lines)
    resume = cast(Record, case["resume"])
    assert resume["code"] != 0 and ledger.PROXY_RERUN_TEXT in str(resume["stderr"]), resume["stderr"]
    assert resume["ledger_unchanged"] is True

def test_endpoint_without_result_fails_and_names_it(runs: tuple[Path, Record]) -> None:
    """抓沒有結果檔也照算：只有 project.json 的資料夾跑 endpoint 回 1、標準錯誤點名 result.json、沒多出任何檔。"""
    case = _section(runs, "endpoint", "no_result")
    assert case["code"] == 1 and "result.json" in str(case["stderr"])
    assert case["entries"] == ["project.json"]

def test_endpoint_rejects_result_with_swapped_theta_then_accepts_restored(runs: tuple[Path, Record]) -> None:
    """抓終點只信結果檔自己說的：結果檔的 θ 換成帳本裡另一個可行點、帳本摘要那一格還等於帳本檔，endpoint 回 1、
    標準錯誤講明結果檔跟帳本對不上並點名 ``theta_hex``、沒多寫任何檔；換回原檔正式跑回 0，報告逐位等於獨立探針。"""
    case, folder = _section(runs, "swapped"), _folder(runs)
    swapped = _section(runs, "swapped", "swapped")
    assert swapped["ledger_sha256"] == _sha256(folder / "ledger.jsonl")
    assert swapped["theta_hex"] != _json(folder / "result.json")["theta_hex"]
    rejected = _section(runs, "swapped", "rejected")
    stderr = str(rejected["stderr"])
    assert rejected["code"] == 1 and "result.json 跟帳本對不上" in stderr and "theta_hex" in stderr, stderr
    assert rejected["entries"] == case["before"]
    restored = _section(runs, "swapped", "restored")
    assert restored["code"] == 0, restored["stderr"]
    record = _record(folder)
    original = EndpointRecord.model_validate(_section(runs, 'legacy_probe')['worker_record'])
    assert json.loads(record.report.to_json()) == json.loads(original.report.to_json())
    assert record.result_sha256 == original.result_sha256

@pytest.mark.parametrize("case", list(STALE_CASES))
def test_report_marks_endpoint_stale_and_names_what_changed(runs: tuple[Path, Record], case: str) -> None:
    """抓報告只比結果檔記的帳本摘要（帳本檔真的變了、結果檔換了 θ、endpoint.json 或 project.json 的搜尋身分換了都照印
    舊終點），或只比各檔對 endpoint.json 的摘要、不比結果↔帳本（摘要欄同步改過之後，結果宣稱的帳本不是現在這本、
    帳本沒收尾照印舊終點）：跑完終點的資料夾抄一份照情境改檔，報告印過期那一行、下一行點名對不上的那幾件，
    不印終點數字。"""
    lines = _report_lines(_section(runs, "stale", case))
    index = lines.index(STALE_ENDPOINT_LINE)
    assert lines[index + 1] == f"  對不上的：{'、'.join(STALE_CASES[case])}"
    assert not any(line.startswith(ENDPOINT_HEADING) for line in lines)

def test_result_from_ledger_equals_written_result(runs: tuple[Path, Record]) -> None:
    """抓寫結果檔與終點核結果檔走的不是同一條重建：``result_from_ledger``（設計清單用考卷自己從方案舊欄換的原牆）
    等於讀回的 result.json，JSON 內容也逐格相等。"""
    folder = _folder(runs)
    project = run.read_project(folder)
    rebuilt = run.result_from_ledger(folder, load_material_gradient_settings(Path(project.settings_path)),
                                     _by_wall(_kept_materials(runs)), legacy=True)
    assert rebuilt == read_result(folder)
    assert json.loads(rebuilt.model_dump_json()) == _json(folder / "result.json")

def test_ledger_header_binds_kept_wall_digests(runs: tuple[Path, Record]) -> None:
    """抓帳本標頭沒綁原牆（方案原牆換了照樣接續舊帳本）：同一份方案與 preset、原牆用考卷自己從舊欄換的那一份，
    ``header_for`` 等於帳本檔裡的標頭（控制組：接續照收）；只把一面原牆的固定阻抗加倍，標頭只差 ``kept_digests``、
    那一面的摘要等於新牆的 ``material_digest``。"""
    folder = _folder(runs)
    project = run.read_project(folder)
    scheme = load_scheme(Path(project.scheme_path))
    settings = load_material_gradient_settings(Path(project.settings_path))
    model = load_reduced(Path(project.surrogate_path))
    constants = run.search_objective_constants(project, scheme, model)
    stored = ledger.read_ledger(folder / "ledger.jsonl", legacy=True)[0]
    kept = _kept_materials(runs)
    same = ledger.header_for(model, run.designs_for(project.presets, settings, kept=_by_wall(kept)), settings,
                             objective_constants=constants)
    assert same == stored and stored.kept_digests
    face = sorted(stored.kept_digests)[0]
    wall = kept[face]
    assert isinstance(wall, ConstantImpedance)
    changed = {**kept, face: ConstantImpedance(impedance_pa_s_m=wall.impedance_pa_s_m * 2.0)}
    other = ledger.header_for(model, run.designs_for(project.presets, settings, kept=_by_wall(changed)), settings,
                              objective_constants=constants)
    differing = {name for name, value in other.model_dump().items() if value != stored.model_dump()[name]}
    assert differing == {"kept_digests"}
    assert other.kept_digests[face] == endpoint.material_digest(changed[face])

def test_resume_rejects_ledger_with_other_kept_wall_digest(runs: tuple[Path, Record]) -> None:
    """抓接續不比原牆摘要：帳本標頭的一面原牆摘要被換掉，命令列 resume 回 1、錯誤點名 ``kept_digests``、帳本一個位元組都不動。"""
    case = _section(runs, "kept_resume")
    assert case["code"] == 1, case
    assert "帳本標頭跟這一次對不上：['kept_digests']" in str(case["stderr"]), case["stderr"]
    assert case["ledger_unchanged"] is True

def test_start_without_low_frequency_axis_fails_before_search(runs: tuple[Path, Record]) -> None:
    """抓低頻軸缺席要等搜尋跑完、到終點才發現：方案場景沒有 ``low_frequency_axis``，start 回 1、標準錯誤點名它、
    根資料夾都沒建。"""
    case = _section(runs, "no_axis")
    assert case["code"] == 1 and "low_frequency_axis" in str(case["stderr"]), case["stderr"]
    assert case["root_exists"] is False

def test_saved_batch_reaches_resumed_objectives(runs: tuple[Path, Record], tmp_path: Path) -> None:
    """雙精度子行程只換求解邊界：接續讀 project、起點與換樣板鏈都必須傳入同一批量。"""
    source = Path(str(runs[1]["new_folder"]))
    project = run.GradientProject.model_validate({**run.read_project(source).model_dump(), "frequency_batch": 7})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    output = _batch_probe(tmp_path, "resume", tmp_path)
    assert output["batches"] == [7, 7]
    assert output["paths"] == ["initial", "candidate"]

def test_saved_batch_reaches_endpoint_chain(runs: tuple[Path, Record], tmp_path: Path) -> None:
    """從存檔跑終點，經 endpoint_check 與編譯目標，直到求解邊界才停；漏任何一層接線都抓。"""
    source = Path(str(runs[1]["folder"]))
    folder = tmp_path / "endpoint-batch"
    shutil.copytree(source, folder, ignore=shutil.ignore_patterns("endpoint.json"))
    project = run.GradientProject.model_validate({**run.read_project(folder).model_dump(), "frequency_batch": 7})
    run.write_json_atomic(folder / run.PROJECT_FILE, project)
    output = _batch_probe(folder, "endpoint", tmp_path)
    assert output["batches"] == [7]
    assert output["paths"] == ["endpoint"]

def test_endpoint_entry_rechecks_carryover_operators(runs: tuple[Path, Record], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """抓終點漏接指紋檢查：從正式入口走到匯出後拒收，不能進完整求解或重寫結果。"""
    folder = tmp_path / "carryover"
    shutil.copytree(_folder(runs), folder, ignore=shutil.ignore_patterns("endpoint.json"))
    project = run.read_project(folder).model_copy(update={"operators_digest": "f" * 64})
    monkeypatch.setattr(endpoint_run, "read_project", lambda folder: project)
    with pytest.raises(ValueError, match="完整模型指紋.*不同"):
        endpoint_run.run_endpoint(folder)
    assert not (folder / endpoint_run.ENDPOINT_FILE).exists()

def test_cli_six_category_bitwise_and_gpu_full_solve(runs: tuple[Path, Record]) -> None:
    """六類結果的 CLI／入口逐位；GPU 完整解替身照收、精修照記，六類仍 CPU 評分。"""
    record = _json(Path(str(runs[1]['new_folder'])) / 'endpoint.json')
    probe = _section(runs, 'probe')
    assert probe['gpu_numpy_calls'] == probe['cpu_numpy_calls'] and cast(int, probe['cpu_numpy_calls']) > 0
    direct = cast(Record, probe['worker_record'])
    assert bitwise_equal(record['six_category'], direct['six_category'])
    gpu = EndpointRecord.model_validate(probe['gpu'])
    assert gpu.device == 'gpu' and gpu.six_category is not None
    assert probe['cpu_six_scoring'] is True
    assert gpu.max_refinement_steps == max(cast(list[int], probe['gpu_steps']))


def test_endpoint_cpu_bytes_unchanged_and_gpu_metadata(runs: tuple[Path, Record], tmp_path: Path) -> None:
    """v4 加不計分低頻註記；CPU 裝置預設不增 JSON 位元組，GPU 兩段真用單精度分解並記修正次數。"""
    record = _record(_folder(runs))
    assert record.format_version == endpoint_run.ENDPOINT_VERSION
    original = record.model_dump(mode="json", exclude={"device", "max_refinement_steps"})
    expected = (json.dumps(original, ensure_ascii=False, allow_nan=False) + "\n").encode()
    for candidate in (record, record.model_copy(update={"device": "cpu"})):
        run.write_json_atomic(tmp_path / "endpoint.json", candidate)
        assert (tmp_path / "endpoint.json").read_bytes() == expected
    probe = _section(runs, "legacy_probe")
    gpu = EndpointRecord.model_validate(probe["gpu"])
    assert probe['gpu_numpy_calls'] == probe['cpu_numpy_calls'] and cast(int, probe['cpu_numpy_calls']) > 0
    assert gpu.device == "gpu" and gpu.max_refinement_steps is not None and gpu.max_refinement_steps > 0
    steps = cast(list[int], probe["gpu_steps"])
    assert len(steps) == gpu.report.full_solves + gpu.baseline.full_solves
    assert gpu.max_refinement_steps == max(steps)
    factors = cast(list[dict[str, object]], probe["gpu_factors"])
    assert len(factors) == 2  # 終點和原牆兩個擁有資源的區段都要走 GPU。
    assert all(factor["freed"] is True for factor in factors)
    assert [factor["factorizations"] for factor in factors] == [gpu.report.full_solves, gpu.baseline.full_solves]
    assert all(cast(int, factor["calls"]) > cast(int, factor["factorizations"]) for factor in factors)
