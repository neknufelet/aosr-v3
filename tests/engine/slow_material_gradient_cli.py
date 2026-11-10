"""本機合併前驗收的搜尋與終點重考卷；考卷本體由原檔原樣搬移。"""
from __future__ import annotations
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, cast
import numpy as np
import pytest
from pydantic import TypeAdapter
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ, LowFrequencyAxis
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Wall
from aosr.gradient import cli
from aosr.gradient import endpoint
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient.report import FULL_MODEL_LINE
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.physics.fem_helmholtz import FrequencyWallImpedances
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_check import ValidationReport, validate_reduced
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, save_validation_receipt
from aosr.physics.fem_reduced_store import manifest_path, save_reduced
from aosr.reporting import physics_identity as identity_module
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_low_frequency_contract import StageOneReasonCode
from aosr.runtime import child_process_env, configure_jax
from tests.engine import _fem_shared_cases as shared_cases
from tests.engine._fem_reduced_cache import shared_holdout_report
from tests.engine._precision_contracts import contract_value
from tests.engine.test_fem_reduced import DENSITY, SOUND_SPEED
from tests.engine._six_search_cli_case import ROOM as CLI_ROOM, model as model, operators as operators
from tests.engine.test_material_gradient import (
    CASES, DESCENT_CASE, PURPOSE, _chain_model, _gradient_settings, _presets, _receivers,
)
from tests.engine.test_material_gradient_loop import _without_seconds
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS

from tests.engine.test_material_gradient_cli import (
    AIR_SURROGATES as AIR_SURROGATES,
    AXIS_SURROGATES as AXIS_SURROGATES,
    BAND_NAME as BAND_NAME,
    FREQUENCY_NAME as FREQUENCY_NAME,
    MOVED_SEAT as MOVED_SEAT,
    MOVED_SPEAKER as MOVED_SPEAKER,
    MOVE_M as MOVE_M,
    NARROWED_STEPS as NARROWED_STEPS,
    OTHER_DENSITY as OTHER_DENSITY,
    OTHER_SPEED as OTHER_SPEED,
    OTHER_STEPS as OTHER_STEPS,
    OUTSIDE_FAMILY as OUTSIDE_FAMILY,
    PRESETS as PRESETS,
    PRESET_TEXT as PRESET_TEXT,
    PROBE as PROBE,
    Record as Record,
    WAIT_S as WAIT_S,
    _FakeProcess as _FakeProcess,
    _chain_operators as _chain_operators,
    _cli_stop as _cli_stop,
    _command as _command,
    _concurrent_start as _concurrent_start,
    _current_identity as _current_identity,
    _cut_copy as _cut_copy,
    _env as _env,
    _first_folder as _first_folder,
    _folder as _folder,
    _good_report as _good_report,
    _holdouts as _holdouts,
    _inputs as _inputs,
    _interrupted as _interrupted,
    _json as _json,
    _kept_materials as _kept_materials,
    _max_steps_flag as _max_steps_flag,
    _moved as _moved,
    _moved_seat as _moved_seat,
    _other_settings as _other_settings,
    _paused as _paused,
    _probe as _probe,
    _probe_folder as _probe_folder,
    _probe_section as _probe_section,
    _project_points_tampered as _project_points_tampered,
    _rejection as _rejection,
    _rejections as _rejections,
    _rows as _rows,
    _run as _run,
    _running_pid as _running_pid,
    _scheme_changed as _scheme_changed,
    _schemes as _schemes,
    _section as _section,
    _settings_path as _settings_path,
    _settings_text as _settings_text,
    _sha256 as _sha256,
    _start_args as _start_args,
    _start_rejection as _start_rejection,
    _surrogate_swapped as _surrogate_swapped,
    _surrogates as _surrogates,
    _without_hash as _without_hash,
    runs as runs,
)


def test_start_writes_pre_loop_in_order_and_closes_loop(runs: tuple[Path, Record]) -> None:
    """抓前段少一段或列序顛倒、迴圈沒收尾、狀態不是 finished、狀態的進度不是從帳本讀的。

    帳本前段（沒有輪號的列）依序是連續下降、取規格、重調三段都在，重調段帶停止原因；最後一列是帶迴圈停止原因的決定列。
    """
    first = _section(runs, "concurrent", "first")
    assert first["code"] == 0, first["stderr"]
    folder = _folder(runs)
    header, rows = ledger.read_ledger(folder / "ledger.jsonl")
    pre_loop = [row.phase for row in rows if row.round is None]
    assert sorted(set(pre_loop), key=pre_loop.index) == ["descent", "snap", "retune"]
    assert pre_loop == sorted(pre_loop, key=["descent", "snap", "retune"].index)
    assert next(row for row in rows if row.phase == "retune" and row.stop_reason is not None) is not None
    last = rows[-1]
    assert last.phase == "round_decision" and last.stop_reason in gradient_search.LOOP_STOP_REASONS
    assert last.decision is not None
    status = _json(folder / "status.json")
    assert status["state"] == "finished" and status["stop_reason"] == last.stop_reason
    assert status["evaluations"] == last.decision.total_evaluations and status["last_value"] == last.value
    assert status["rounds_done"] == sum(1 for row in rows if row.phase == "round_decision")
    assert header.surfaces == tuple(zip(gradient_search.wall_surface_ids(), PRESETS, strict=True))

def test_start_project_records_inputs_and_identities(runs: tuple[Path, Record]) -> None:
    """抓 project.json 記錯檔、雜湊沒量、設定指紋或物理身分抄錯：每一格由考卷自己從檔案重算。"""
    directory, folder = runs[0], _folder(runs)
    project = _json(folder / "project.json")
    scheme, surrogate = directory / "scheme.json", directory / "good" / "rom.npz"
    manifest = _json(manifest_path(surrogate))
    receipt = surrogate.with_name("rom.validation.json")
    assert project["scheme_path"] == str(scheme.resolve())
    assert project["scheme_sha256"] == _sha256(scheme)
    assert project["surrogate_path"] == str(surrogate.resolve())
    assert project["surrogate_manifest_sha256"] == manifest["manifest_sha256"]
    assert project["receipt_path"] == str(receipt.resolve()) and project["receipt_sha256"] == _sha256(receipt)
    assert project["settings_fingerprint"] == load_material_gradient_settings(_settings_path(runs)).fingerprint
    assert project["objective_settings_fingerprint"] == _gradient_settings().fingerprint
    assert project["physics_identity"] == manifest["physics_identity"] == _current_identity()
    assert project["mesh_hash"] == manifest["mesh_hash"]
    assert project["presets"] == list(PRESETS) and project["purpose"] == PURPOSE
    assert project["gradient_id"] == folder.name
    assert "mount" not in project  # #990：不給 --mount 就是剛背，剛背不寫這一格（逐位同加這一格之前）
    scheme_document = Scheme.model_validate_json(scheme.read_text(encoding="utf-8"))
    assert project["source_points"] == {name: [float(value).hex() for value in point.as_tuple()]
                                        for name, point in scheme_document.speakers.items()}
    assert project["receiver_points"] == {point.receiver_id: [float(value).hex() for value in point.position_m]
                                          for point in scheme_document.receiver_set.points}
    receipt_model = _json(receipt)["reduced_model"]
    assert isinstance(receipt_model, dict)
    assert (receipt_model["source_points"], receipt_model["receiver_points"]) == (
        project["source_points"], project["receiver_points"])

def test_result_is_last_decision_and_feasible(runs: tuple[Path, Record]) -> None:
    """抓結果檔取錯列、每面厚度空腔切錯面、取到不可行的點、停止原因或求值次數抄錯、材料摘要算錯面或不貼面記成 None。

    考卷自己從帳本最後一列決定的 θ 照設計清單切回每面，並用設定與 preset 重建設計判可行；材料摘要：有樣板的面是
    ``endpoint.material_digest``（構造預設清單那一份 preset 的構造），不貼的面是原方案那面牆的。
    """
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    by_id, kept = {preset.id: preset for preset in _presets()}, _kept_materials(runs)
    folder = _folder(runs)
    result, rows = _json(folder / "result.json"), _rows(folder)
    last = rows[-1]
    assert last.decision is not None
    designs = gradient_search.wall_designs_from_presets(
        PRESETS, load_material_gradient_settings(_settings_path(runs)), _presets())
    theta = np.asarray([float.fromhex(value) for value in cast(list[str], result["theta_hex"])])
    assert cast(list[str], result["theta_hex"]) == list(last.theta_hex)
    assert gradient_search.is_feasible(theta, designs)
    expected = []
    offset = 0
    for design in designs:
        size = len(design.variables)
        values = [float(value) for value in theta[offset:offset + size]]
        material = (kept[design.surface_id] if design.preset_id is None
                    else adapter.validate_python(by_id[design.preset_id].material))
        expected.append({"surface_id": design.surface_id, "preset_id": design.preset_id,
                         "thickness_m": values[0] if values else None, "cavity_m": values[1] if size > 1 else None,
                         "material_digest": endpoint.material_digest(material)})
        offset += size
    assert result["surfaces"] == expected
    assert (result["value"], result["round_stop_reason"], result["evaluations"]) == (
        last.value, last.stop_reason, last.decision.total_evaluations)
    assert result["components"] == dict(zip(gradient_search.COMPONENT_NAMES, last.components, strict=True))
    assert result["screening_total"] == last.screening_total
    assert result["ledger_sha256"] == _sha256(folder / "ledger.jsonl")

def test_report_prints_segments_rounds_stop_reason_and_final_point(runs: tuple[Path, Record]) -> None:
    """抓報告漏段、漏停止原因、最後可行點切錯面或單位錯（公尺當毫米）、不貼的面印成剛性、endpoint 之前最後一行
    不是「未做」那一行。

    每面那一行由考卷自己從結果檔換算成毫米；不貼的面由考卷從方案舊欄的固定阻抗自己格式化；停止原因與累計求值次數從帳本取。
    """
    report = _section(runs, "report")
    assert report["code"] == 0, report["stderr"]
    lines = str(report["stdout"]).splitlines()
    assert lines[-1] == FULL_MODEL_LINE
    folder = _folder(runs)
    rows, result = _rows(folder), _json(folder / "result.json")
    last = rows[-1]
    assert last.decision is not None
    text = str(report["stdout"])
    for segment in ("連續下降", "取規格", "空腔重調"):
        assert any(line.strip().startswith(f"{segment}：") and "還沒寫進帳本" not in line for line in lines), segment
    assert f"停止原因：{last.stop_reason}" in lines
    assert f"累計求值次數（帳本記的、不含前段）：{last.decision.total_evaluations}" in text
    kept = _kept_materials(runs)
    for spec in cast(list[Record], result["surfaces"]):
        if spec["preset_id"] is None:
            material = kept[str(spec["surface_id"])]
            assert isinstance(material, ConstantImpedance)
            assert f"  {spec['surface_id']}：不貼（保留原牆：固定阻抗 {material.impedance_pa_s_m:.3f} Pa·s/m）" in lines
            continue
        thickness = f"棉板 {float(cast(float, spec['thickness_m'])) * 1000.0:.3f} mm"
        cavity = ("沒有空腔層" if spec["cavity_m"] is None
                  else f"空腔 {float(cast(float, spec['cavity_m'])) * 1000.0:.3f} mm")
        assert f"  {spec['surface_id']}：{spec['preset_id']}，{thickness}，{cavity}" in lines
    for row in rows:
        if row.decision is not None:
            assert any(line.startswith(f"第 {row.round} 輪：交出 六類 J {row.value:.6f}／product {row.components[10]:.6f}") for line in lines)

def test_stop_marker_sampled_per_phase_then_resume_matches_one_shot(runs: tuple[Path, Record]) -> None:
    """抓停止記號沒在求值之間看（跑完才停或根本不停）、停下時寫了半段、接續後跟一口氣跑完的不同。

    不是每一次求值都停一次（那要多跑幾十場），是按階段抽樣：探針先一口氣跑完一份參考、記下每一次查停止時帳本
    最後一列的階段；每一種階段的第一次查、再加整場最後一次查，各開一份用產品的 stop 命令在那一次放記號：
    要回 3、狀態 stopped、記號還在；接續要回 0、帳本（扣秒數）與結果（扣帳本雜湊）逐位等於參考。每一次求值之前
    都有查，由 ``test_stop_checked_before_every_evaluation`` 的次數守。
    """
    reference = cast(Record, _probe_section(runs, "reference"))
    assert reference["code"] == 0
    reference_folder = _probe_folder(runs, reference["folder"])
    expected_rows = _without_seconds(_rows(reference_folder))
    expected_result = _without_hash(_json(reference_folder / "result.json"))
    stops = cast(list[Record], _probe_section(runs, "stops"))
    independent = [stop for stop in stops if stop['isolated']]
    assert len(independent) == 1 and independent[0]['pid'] != reference['pid']
    assert independent[0]['in_rounds'] is True
    stopped = ledger.read_ledger(runs[0] / 'probe' / str(independent[0]['stopped_ledger']))[1]
    assert stopped[-1].phase == 'retune' and stopped[-1].stop_reason is not None  # 前段完整，走 resume_rounds。
    phases = cast(list[str], _probe_section(runs, "phases"))
    assert {stop["phase"] for stop in stops} == set(phases)
    for stop in stops:
        label = f"第 {stop['at']} 次查停止（{stop['phase']}）"
        assert stop["stop_command_code"] == 0, label
        assert stop["stopped_code"] == 3 and stop["marker_after_stop"] is True, label
        assert cast(Record, stop["stopped_status"])["state"] == "stopped", label
        folder = _probe_folder(runs, stop["folder"])
        assert stop["resumed_code"] == 0, label
        assert cast(Record, stop["resumed_status"])["state"] == "finished", label
        assert _without_seconds(_rows(folder)) == expected_rows, label
        assert _without_hash(_json(folder / "result.json")) == expected_result, label

def test_stopped_ledger_is_prefix_without_stop_reason(runs: tuple[Path, Record]) -> None:
    """抓停下時寫了沒算完的半段、或把「人叫停」寫成停止原因、或狀態檔的進度不是從帳本讀的。

    停下那一刻的帳本（探針另存一份）扣秒數是參考帳本的前綴；狀態檔沒有停止原因、累計求值次數等於那份帳本最後一列決定記的。
    """
    reference_folder = _probe_folder(runs, cast(Record, _probe_section(runs, "reference"))["folder"])
    expected = _without_seconds(_rows(reference_folder))
    for stop in cast(list[Record], _probe_section(runs, "stops")):
        status = cast(Record, stop["stopped_status"])
        assert status["stop_reason"] is None, stop["at"]
        if stop["stopped_ledger"] is None:
            assert status["evaluations"] == 0, stop["at"]
            continue
        stopped = ledger.read_ledger(runs[0] / "probe" / str(stop["stopped_ledger"]))[1]
        assert _without_seconds(stopped) == expected[:len(stopped)] and len(stopped) < len(expected), stop["at"]
        assert all(row.phase != "round_decision" or row.stop_reason is None for row in stopped), stop["at"]
        decisions = [row.decision for row in stopped if row.decision is not None]
        assert status["evaluations"] == (decisions[-1].total_evaluations if decisions else 0), stop["at"]

def test_resume_with_half_pre_loop_redoes_pre_loop(runs: tuple[Path, Record]) -> None:
    """抓前段沒寫完卻直接進迴圈（或拒收）、重做前段時沒先截掉舊列（列重複）：帳本只留到下降中途、下降完、
    取規格完、重調中途，接續都要回 0，帳本（扣秒數）逐位等於一口氣跑完的參考。"""
    reference_folder = _probe_folder(runs, cast(Record, _probe_section(runs, "reference"))["folder"])
    expected = _without_seconds(_rows(reference_folder))
    pre_loop = cast(int, _probe_section(runs, "pre_loop_rows"))
    cuts = cast(list[Record], _probe_section(runs, "cuts"))
    independent = [cut for cut in cuts if cut['isolated']]
    reference = cast(Record, _probe_section(runs, 'reference'))
    assert len(independent) == 1 and independent[0]['pid'] != reference['pid']
    assert {cut["keep"] for cut in cuts} and all(cast(int, cut["keep"]) < pre_loop for cut in cuts)
    phases = {expected[cast(int, cut["keep"]) - 1]["phase"] for cut in cuts}
    assert phases == {"descent", "snap", "retune"}
    for cut in cuts:
        assert cut["code"] == 0 and cast(Record, cut["status"])["state"] == "finished", cut["keep"]
        assert _without_seconds(_rows(_probe_folder(runs, cut["folder"]))) == expected, cut["keep"]

def test_resume_rejects_ledger_with_other_header(runs: tuple[Path, Record]) -> None:
    """抓接續不比標頭（設定換了還照舊帳本跑、或把帳本截掉重來）：只改標頭設定指紋的帳本要回 1、狀態 failed、帳本不動。"""
    case = cast(Record, _probe_section(runs, "other_header"))
    status = cast(Record, case["status"])
    assert case["code"] == 1 and status["state"] == "failed"
    assert "settings_fingerprint" in str(status["message"])
    assert case["ledger_unchanged"] is True

def test_cli_stop_command_stops_child_and_resume_clears_marker(runs: tuple[Path, Record]) -> None:
    """抓命令列 stop 要拿鎖（子行程拿著鎖時就停不了）、子行程不看記號、離開碼沒傳出來、resume 不清記號（一接續就又停）。

    接續一份第 0 輪沒寫完的帳本，子行程寫完 running 就暫停，stop 命令放記號、讓它繼續：要回 3、狀態 stopped；
    再接續要回 0、記號被清掉，帳本（扣秒數）逐位等於命令列一口氣跑完的那一份。
    """
    case = _section(runs, "cli_stop")
    assert cast(Record, case["stop"])["code"] == 0
    stopped = cast(Record, case["stopped"])
    assert stopped["code"] == 3, stopped["stderr"]
    assert cast(Record, case["stopped_status"])["state"] == "stopped" and case["marker_after_stop"] is True
    resumed = cast(Record, case["resumed"])
    assert resumed["code"] == 0, resumed["stderr"]
    assert case["marker_after_resume"] is False
    folder = Path(str(case["folder"]))
    assert _json(folder / "status.json")["state"] == "finished"
    assert _without_seconds(_rows(folder)) == _without_seconds(_rows(_folder(runs)))
    assert _without_hash(_json(folder / "result.json")) == _without_hash(_json(_folder(runs) / "result.json"))

def test_rejects_surrogate_without_receipt(runs: tuple[Path, Record]) -> None:
    """抓沒有驗證收據的替身也拿去跑：回 1，標準錯誤是准用閘門的原文（ReducedModelNotValidated、收據），沒建任何資料夾。"""
    case = _rejection(runs, "unvalidated")
    assert case["code"] == 1
    assert "ReducedModelNotValidated" in str(case["stderr"]) and "收據" in str(case["stderr"])
    assert case["root_entries"] == []

def test_rejects_other_physics_identity(runs: tuple[Path, Record]) -> None:
    """抓物理身分不比：start 用物理身分全零（舊程式建的）的替身回 1；resume 時 project.json 記的身分跟現在不同也回 1、
    不起子行程（資料夾裡只有 project.json）。"""
    case = _rejection(runs, "zero_identity")
    assert case["code"] == 1 and "物理身分" in str(case["stderr"]) and case["root_entries"] == []
    resumed = _rejection(runs, "resume_other_identity")
    assert resumed["code"] == 1 and "物理身分" in str(resumed["stderr"])
    assert resumed["entries"] == ["project.json"]

@pytest.mark.parametrize(("case", "label", "name"), (("moved_speaker", "喇叭", MOVED_SPEAKER),
                                                    ("moved_seat", "座位", MOVED_SEAT)))
def test_rejects_scheme_with_moved_speaker_or_seat(runs: tuple[Path, Record], case: str, label: str, name: str) -> None:
    """審查 B1 原樣重現：同一間房、同樣的喇叭與座位名字（網格雜湊不變），只把一支喇叭或一個座位的 x 移 0.1 m，
    拿原來那份驗過的替身 start：回 1，標準錯誤是准用閘門點名那一支或那一座，沒建任何資料夾。"""
    record = _rejection(runs, case)
    assert record["code"] == 1, record["stderr"]
    stderr = str(record["stderr"])
    assert "ReducedModelNotValidated" in stderr and f"{label} {name} 的座標跟驗證收據記的不同" in stderr
    assert record["root_entries"] == []

def test_resume_rejects_changed_scheme_file(runs: tuple[Path, Record]) -> None:
    """抓接續不核方案檔：開始之後方案檔內容換過（同一份方案換縮排，雜湊就不同），resume 回 1、點名 scheme_sha256，
    帳本、狀態檔、結果檔都不動（父行程起子行程之前就拒收）。"""
    record = _rejection(runs, "resume_scheme_changed")
    assert record["code"] == 1 and "scheme_sha256" in str(record["stderr"])
    assert record["unchanged"] is True

@pytest.mark.parametrize(("kind", "label", "name"), (("source_points", "喇叭", MOVED_SPEAKER),
                                                    ("receiver_points", "座位", MOVED_SEAT)))
def test_resume_rejects_tampered_project_points(runs: tuple[Path, Record], kind: str, label: str, name: str) -> None:
    """審查另記：只改 project.json 記的一支喇叭或一個座位的座標（方案檔、替身、收據都不動，方案雜湊照樣對），
    resume 回 1、點名那一支或那一座，帳本與狀態檔原始位元組不變（父行程起子行程之前就拒收）。"""
    record = _rejection(runs, f"resume_project_{kind}")
    assert record["code"] == 1, record["stderr"]
    assert f"{label} {name} 的座標：project.json 記的" in str(record["stderr"])
    assert record["unchanged"] is True

@pytest.mark.parametrize(("case", "surrogate"), (("other_density", "other-density"), ("other_speed", "other-speed")))
def test_start_rejects_surrogate_built_with_other_air(runs: tuple[Path, Record], case: str, surrogate: str) -> None:
    """審查（修補三）：同網格、同座標、物理身分相同、收據合格，只有建模用的密度或聲速跟方案不同的替身拿來 start：
    回 1，標準錯誤說替身建模用的空氣跟方案的不同並列出兩邊的值，根底下沒建資料夾。接續那一側走同一個核對函式。"""
    record = _rejection(runs, case)
    assert record["code"] == 1, record["stderr"]
    density, speed = AIR_SURROGATES[surrogate]
    stderr = str(record["stderr"])
    assert "替身建模用的空氣跟方案的不同" in stderr
    assert f"替身 密度 {density!r}、聲速 {speed!r}；方案 密度 {DENSITY!r}、聲速 {SOUND_SPEED!r}" in stderr
    assert record["root_entries"] == []

@pytest.mark.parametrize("surrogate", tuple(AXIS_SURROGATES))
def test_start_rejects_surrogate_axis_short_of_declared_points(runs: tuple[Path, Record], surrogate: str) -> None:
    """顧問 #971：計分範圍 20–300 Hz，替身只交 80、100、125 Hz（或宣告軸中間少一點），收據照樣合格；start 要回 1、
    標準錯誤點名低頻評估器的理由碼與宣告軸，根底下沒建資料夾（不准拿剩下的頻點代替整段）。"""
    record = _rejection(runs, surrogate)
    assert record["code"] == 1, record["stderr"]
    stderr = str(record["stderr"])
    assert AXIS_SURROGATES[surrogate][1].value in stderr and LowFrequencyAxis.SEARCH.value in stderr
    assert record["root_entries"] == []

def test_resume_rejects_surrogate_axis_short_of_declared_points(runs: tuple[Path, Record]) -> None:
    """接續那一側：跑完的資料夾把 project.json 指到窄軸替身（摘要一起換，前面幾格核對都過），resume 回 1、
    點名理由碼，資料夾裡每一個檔的 SHA-256 前後相同、沒多出檔（父行程起子行程之前就拒收）。"""
    record = _rejection(runs, "resume_narrow_axis")
    assert record["code"] == 1, record["stderr"]
    assert AXIS_SURROGATES["narrow-axis"][1].value in str(record["stderr"])
    assert record["unchanged"] is True

def test_max_steps_flag_is_stored_reported_and_reapplied_on_resume(runs: tuple[Path, Record]) -> None:
    """抓（#1073）：--max-steps 沒算進設定指紋（資料夾名、帳本跟 settings-other.toml 那一份不同）、沒存進 project.json、
    報告沒印、接續重讀設定檔而拒收自己（或接上了但帳本跟一口氣跑完的不同）、把那一格拿掉仍照接或動了資料夾。"""
    record = _section(runs, "max_steps_flag")
    started = cast(Record, record["started"])
    assert started["code"] == 0, started["stderr"]
    steps = OTHER_STEPS["descent.max_steps"]
    first = _folder(runs)
    others = [name for name in cast(list[str], _section(runs, "other_settings")["folders"]) if name != first.name]
    assert record["folders"] == others, (record["folders"], others)
    folder = Path(str(record["folder"]))
    project = _json(folder / "project.json")
    assert project["max_steps"] == steps and project["settings_path"] == str((runs[0] / "settings.toml").resolve())
    assert project["settings_fingerprint"] == load_material_gradient_settings(runs[0] / "settings-other.toml").fingerprint
    assert _without_seconds(_rows(folder)) == _without_seconds(_rows(first.parent / others[0]))
    descent = [row for row in _rows(folder) if row.phase == "descent" and row.round is None]
    assert len(descent) - 1 <= steps
    report = str(cast(Record, record["report"])["stdout"])
    assert f"往下走步數上限：{steps} 步（開跑時指定）" in report.splitlines()
    # 報告其他段（每面總深度上限）也要套回記的步數重讀設定，否則指紋對不上、整段變成「無法確認」
    assert "設定指紋不同" not in report and "：總深度上限 " in report, report
    resumed = cast(Record, record["resumed"])
    assert resumed["code"] == 0, resumed["stderr"]
    cut = Path(str(record["resumed_folder"]))
    assert _json(cut / "status.json")["state"] == "finished"
    assert _without_seconds(_rows(cut)) == _without_seconds(_rows(folder))
    refused = cast(Record, record["refused"])
    assert refused["code"] == 1 and "設定指紋變了" in str(refused["stderr"]), refused
    assert record["refused_unchanged"] is True


def test_same_root_other_settings_gets_own_folder(runs: tuple[Path, Record]) -> None:
    """抓資料夾名沒把設定算進去（換一份設定開始，撞進同一個資料夾被當成「已存在」、或蓋掉前一場）：同一個根、
    同一份方案與替身、另一份設定，start 也要回 0，根底下兩個資料夾名不同，各自的 project.json 記各自的設定指紋。"""
    record = _section(runs, "other_settings")
    assert record["code"] == 0, record["stderr"]
    first = _folder(runs)
    folders = cast(list[str], record["folders"])
    others = [name for name in folders if name != first.name]
    assert first.name in folders and others, folders
    other = first.parent / others[0]
    assert other.name != first.name and other.parent == first.parent == runs[0] / "root-a"
    for folder, settings in ((first, "settings.toml"), (other, "settings-other.toml")):
        assert _json(folder / "project.json")["settings_fingerprint"] == load_material_gradient_settings(
            runs[0] / settings).fingerprint, folder.name
        assert _json(folder / "status.json")["state"] == "finished", folder.name

def test_sigterm_to_cli_terminates_child_and_records_interrupted(runs: tuple[Path, Record]) -> None:
    """抓命令列被終止時留下孤兒、或子行程收不到 SIGTERM（建子行程時擋著的訊號遮罩沒在子行程放開，它要等寬限過了
    被 SIGKILL）：子行程寫完 running 就對命令列送 SIGTERM，命令列回 3、子行程寫 interrupted、整群已不在，
    而且在寬限秒數之內收完。"""
    record = _section(runs, "interrupted")
    assert record["code"] == 3, record["stderr"]
    assert cast(Record, record["status"])["state"] == "interrupted"
    assert record["child_alive"] is False
    assert cast(float, record["seconds"]) < cli.TERMINATE_GRACE_S

def test_rejects_preset_outside_family(runs: tuple[Path, Record]) -> None:
    """抓 preset 不在家族清單也拿去跑：回 1，標準錯誤點名那個 preset，沒建資料夾。"""
    case = _rejection(runs, "outside_family")
    assert case["code"] == 1 and OUTSIDE_FAMILY in str(case["stderr"]) and case["root_entries"] == []

def test_wrong_preset_count_is_argument_error(runs: tuple[Path, Record]) -> None:
    """抓 preset 個數不對當成一般失敗或照跑：參數錯回 2，沒建資料夾。"""
    case = _rejection(runs, "five_presets")
    assert case["code"] == 2 and "--presets" in str(case["stderr"]) and case["root_entries"] == []

def test_busy_folder_rejects_second_start_and_resume(runs: tuple[Path, Record]) -> None:
    """抓資料夾鎖沒拿或沒傳給子行程：子行程跑著（暫停中、鎖在它手上）時，同一組輸入再 start、或 resume 同一個資料夾，
    都要回 1、說還有計算在跑；第一個照樣跑完。"""
    case = _section(runs, "concurrent")
    for name in ("second_start", "busy_resume"):
        record = cast(Record, case[name])
        assert record["code"] == 1 and "還有計算在跑" in str(record["stderr"]), name
    assert cast(Record, case["first"])["code"] == 0

def test_lock_held_by_other_process_rejects_resume_but_not_stop(runs: tuple[Path, Record], tmp_path: Path) -> None:
    """抓 resume 不拿鎖、或 stop 要拿鎖：考卷自己拿著一份抄來的資料夾的鎖，resume 回 1、帳本不動；stop 照樣回 0、建記號。"""
    folder = tmp_path / _folder(runs).name
    shutil.copytree(_folder(runs), folder)
    before = (folder / "ledger.jsonl").read_bytes()
    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resumed = _run(tmp_path, _command("resume", str(folder)))
        stopped = _run(tmp_path, _command("stop", str(folder)))
    finally:
        os.close(descriptor)
    assert resumed["code"] == 1 and "還有計算在跑" in str(resumed["stderr"])
    assert (folder / "ledger.jsonl").read_bytes() == before
    assert stopped["code"] == 0 and (folder / "stop").exists()

def test_rejects_root_under_web_searches(runs: tuple[Path, Record]) -> None:
    """抓梯度資料夾放進網頁讀的 searches/（網頁會把它當擺位搜尋讀、報錯）：回 1、什麼都沒建。"""
    case = _rejection(runs, "under_searches")
    assert case["code"] == 1 and "searches" in str(case["stderr"]) and case["created"] is False

def test_objective_uses_scheme_air_through_config_layer(runs: tuple[Path, Record]) -> None:
    """抓重建目標時自己拿產品預設空氣：方案場景與替身建模的空氣都換成 1.18／340（替身另建、有合格收據）後重建的目標，
    在樣板起點的值逐位等於探針自己用設定層配方（construction_environment）造的那一份，而且跟原方案原替身重建的不同。"""
    air = cast(Record, _probe_section(runs, "air"))
    assert air["rebuilt"] == air["expected"]
    assert air["rebuilt"] != air["original"]

def test_stop_checked_before_every_evaluation(runs: tuple[Path, Record]) -> None:
    """抓有哪一種求值之前沒查停止（下降、取規格那一次、重調、迴圈的梯度與重調）：一口氣跑完那一次查停止的次數，
    要等於帳本記的求值次數（每段下降末列記的這段次數、取規格那一次、迴圈累計），而且下降中、取規格、重調中、
    迴圈都查過（查的那一刻帳本最後一列分別是標頭、下降、取規格、重調；迴圈一整輪算完才寫列，所以迴圈裡查的那一刻
    帳本最後一列是重調或上一輪的決定，迴圈的次數由上面的總數守）。"""
    reference = cast(Record, _probe_section(runs, "reference"))
    rows = _rows(_probe_folder(runs, reference["folder"]))
    segments = sum(row.evaluations or 0 for row in rows if row.phase in ("descent", "retune"))
    snaps = sum(1 for row in rows if row.phase == "snap")
    last = rows[-1].decision
    assert last is not None
    assert reference["checks"] == segments + snaps + last.total_evaluations
    phases = set(cast(list[str], _probe_section(runs, "phases")))
    assert {"header", "descent", "snap", "retune"} <= phases

def test_new_ledger_low_frequency_and_six_categories_are_bitwise_controls(runs: tuple[Path, Record]) -> None:
    control = cast(Record, _probe_section(runs, "control"))
    assert control["low_bitwise"] is True, control
    assert control["six_bitwise"] is True, control
    assert control["actual"] == control["expected"], control

def test_all_six_constants_participate_in_fingerprint(runs: tuple[Path, Record]) -> None:
    from aosr.scoring.six_category_jax import SixCategoryConstants
    control = cast(Record, _probe_section(runs, "control"))
    fingerprints = cast(dict[str, bool], control['fingerprints'])
    assert set(fingerprints) == set(SixCategoryConstants._fields)
    assert all(fingerprints.values()), fingerprints

def test_reversed_surrogate_source_axis_keeps_geometry_aligned(runs: tuple[Path, Record]) -> None:
    from tests.engine.test_six_category_jax import _tolerances
    control = cast(Record, _probe_section(runs, "control"))
    expected = np.asarray([float.fromhex(x) for x in cast(list[str], control['expected'])][3:])
    actual = np.asarray(control['reversed_sources'])
    tolerance = _tolerances(expected[:6])
    assert np.all(np.abs(actual[:6] - expected[:6]) <= tolerance), (actual, expected)
    from tests.engine._six_search_helpers import weights
    assert np.all(np.abs(actual[6:] - expected[6:]) <= np.dot(np.asarray(weights()), tolerance))
