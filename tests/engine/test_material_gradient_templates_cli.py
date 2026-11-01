"""換樣板接進命令列（#561 庚三）：start 跑到換樣板段收尾、庚三之前跑完的資料夾 resume 開換樣板段、endpoint 用最後配置、
報告的換樣板一節、換樣板帳本一動報告就判終點過期。

自帶夾具（不跟 ``test_material_gradient_cli``、``test_material_gradient_endpoint_cli`` 共用 ``runs``，理由同後者：
雲端可能分到不同片）：同一間小房間的驗過替身與同一份方案、縮小步數的設定再打開換樣板段；開始一次、抄一份改成庚三之前的
形狀（沒有 ``templates.jsonl``、``result.json`` 是 v2）接續一次、跑 endpoint 與報告、改檔的情境都先抄一份資料夾再改。
命令列都在子行程（產品入口）；這裡沒有另開探針，比對與判決都在父行程，只讀檔、用產品的讀檔函式與考卷自己的換算。
一場 pytest 只建一次、平行的工人共用（``shared_json``）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import pytest
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.geometry.shoebox import Wall
import numpy as np

from aosr.gradient import endpoint
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import run
from aosr.gradient import template_ledger as templates
from aosr.gradient.report import (
    ENDPOINT_HEADING, NOT_RETUNABLE, QUICK_ONLY, RETUNED, STALE_ENDPOINT_LINE, TEMPLATE_HEADING,
)
from aosr.materials.construction_models import ConstantImpedance
from aosr.reporting.scheme import load_scheme
from tests.engine._scheme_cache import shared_json
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine.test_material_gradient_cli import (
    Record, _command, _json, _kept_materials, _run, _schemes, _settings_text, _sha256, _start_args,
)
from tests.engine.test_material_gradient_endpoint_cli import _surrogate

# 縮小步數（同命令列考卷），換樣板段照正式設定開、只把輪數上限調成 2 輪（考卷要快）。
TEMPLATE_STEPS = {"descent.max_steps": 3, "retune.steps": 2, "templates.max_rounds": 2}
# 庚三之前的 result.json（v2）有、v3 改名或新加的欄。
V3_ONLY = ("round_stop_reason", "template_rounds", "template_evaluations", "template_ledger_sha256")


def _inputs(directory: Path) -> None:
    (directory / "scheme.json").write_text(_schemes()["scheme.json"].model_dump_json(), encoding="utf-8")
    (directory / "settings.toml").write_text(_settings_text(TEMPLATE_STEPS), encoding="utf-8")
    settings = load_material_gradient_settings(directory / "settings.toml")
    assert (settings.descent.max_steps, settings.retune.steps, settings.templates.max_rounds) == (3, 2, 2)


def v2_result(v3: Record) -> Record:
    """考卷自己把結果檔換回庚三之前的 v2 形狀：版本字串、停止原因是庚二的、沒有換樣板那幾欄，
    也沒有壬一加的舊合分與大峰代理。"""
    document = {key: value for key, value in v3.items() if key not in (*V3_ONLY, "screening_total")}
    document["components"] = {key: value for key, value in cast(Record, v3["components"]).items()
                              if key != "big_peak_proxy_db"}
    document["format_version"] = "aosr.material_gradient_result.v2"
    document["stop_reason"] = v3["round_stop_reason"]
    return document


def _old_shape(directory: Path, folder: Path) -> Record:
    """抄一份跑完的資料夾，拿掉 templates.jsonl、result.json 換成 v2（示範房資料夾就是這個形狀），resume。"""
    copy = directory / "old-shape" / folder.name
    shutil.copytree(folder, copy, ignore=shutil.ignore_patterns("endpoint.json", "scheme-endpoint*.json"))
    (copy / "templates.jsonl").unlink()
    v2 = v2_result(_json(copy / "result.json"))
    (copy / "result.json").write_text(json.dumps(v2, ensure_ascii=False), encoding="utf-8")
    report = _run(directory, _command("report", "--folder", str(copy)))
    return {"folder": str(copy), "v2": v2, "report_before": report,
            "resume": _run(directory, _command("resume", str(copy)))}


def _flipped(directory: Path, folder: Path) -> Record:
    """抄一份跑完終點的資料夾，templates.jsonl 第一列快速評分的秒數第一個數字翻一個位元（還是數字、帳本照樣讀得回），印報告。"""
    copy = directory / "flipped" / folder.name
    shutil.copytree(folder, copy)
    content = bytearray((copy / "templates.jsonl").read_bytes())
    start = content.index(b'"phase": "quick_score"')
    at = content.index(b'"seconds": ', start) + len(b'"seconds": ')
    content[at] ^= 1
    (copy / "templates.jsonl").write_bytes(bytes(content))
    return _run(directory, _command("report", "--folder", str(copy)))


def _compute(directory: Path, request: pytest.FixtureRequest) -> str:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    _surrogate(directory, request)
    _inputs(directory)
    root = directory / "root"
    output: Record = {"start": _run(directory, _start_args(directory, root))}
    assert cast(Record, output["start"])["code"] == 0, output["start"]
    (folder,) = (path for path in root.iterdir() if not path.name.startswith("."))
    output["folder"] = str(folder)
    output["old_shape"] = _old_shape(directory, folder)
    output["report_before"] = _run(directory, _command("report", "--folder", str(folder)))
    output["endpoint"] = _run(directory, _command("endpoint", str(folder)))
    output["report_after"] = _run(directory, _command("report", "--folder", str(folder)))
    output["flipped"] = _flipped(directory, folder)
    text = json.dumps(output, ensure_ascii=False)
    (directory / "fixture.json").write_text(text, encoding="utf-8")
    return text


@pytest.fixture(scope="module")
def runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
         ) -> tuple[Path, Record]:
    """一場 pytest 只建一次，平行的工人共用；讀的人只讀。"""
    directory = (tmp_path_factory.mktemp("material-gradient-templates-cli") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "material-gradient-templates-cli-shared")
    text = shared_json(tmp_path_factory, worker_id, "material-gradient-templates-cli",
                       lambda: _compute(directory, request))
    return directory, cast(Record, json.loads(text))


def _section(runs: tuple[Path, Record], *keys: str) -> Record:
    node: object = runs[1]
    for key in keys:
        node = cast(Record, node)[key]
    return cast(Record, node)


def _folder(runs: tuple[Path, Record]) -> Path:
    return Path(str(runs[1]["folder"]))


def _template_rows(folder: Path) -> tuple[templates.TemplateRow, ...]:
    return templates.read_template_ledger(folder / "templates.jsonl")[1]


def _without_seconds(rows: tuple[templates.TemplateRow, ...]) -> list[Record]:
    return [{key: value for key, value in row.model_dump(mode="json").items() if key != "seconds"} for row in rows]


def _without_hashes(result: Record) -> Record:
    return {key: value for key, value in result.items() if key not in ("ledger_sha256", "template_ledger_sha256")}


def _lines(case: Record) -> list[str]:
    assert case["code"] == 0, case["stderr"]
    return str(case["stdout"]).splitlines()


def _accepted(rows: tuple[templates.TemplateRow, ...]) -> list[str]:
    return [str(row.decision.accepted) for row in rows if row.decision is not None and row.decision.accepted]


# ---- start：兩本帳與結果檔 ----

def test_start_closes_template_segment_and_result_v4_matches_ledgers(runs: tuple[Path, Record]) -> None:
    """抓換樣板段沒接上（start 只跑到庚二）、結果檔還是 v2、最後配置不是換樣板帳本最後一列決定的、兩個停止原因混在一起、
    求值次數拿錯本帳、帳本摘要沒綁。"""
    folder = _folder(runs)
    rows = _template_rows(folder)
    result, status = _json(folder / "result.json"), _json(folder / "status.json")
    last = rows[-1]
    assert last.decision is not None and last.stop_reason is not None
    assert result["format_version"] == "aosr.material_gradient_result.v5"  # #990 升 v5（每面安裝方式）
    assert result["stop_reason"] == last.stop_reason
    assert [[item["surface_id"], item["preset_id"]] for item in cast(list[Record], result["surfaces"])] == [
        list(item) for item in last.surfaces]
    assert (result["variable_names"], result["theta_hex"], result["value"]) == (
        list(last.variable_names), list(last.theta_hex), last.value)
    assert (result["template_rounds"], result["template_evaluations"]) == (
        sum(1 for row in rows if row.decision is not None), last.decision.total_evaluations)
    evaluations, _, _, round_stop = run.ledger_progress(ledger.read_ledger(folder / "ledger.jsonl")[1])
    assert round_stop is not None
    assert (result["evaluations"], result["round_stop_reason"]) == (evaluations, round_stop)
    assert result["template_ledger_sha256"] == _sha256(folder / "templates.jsonl")
    assert result["ledger_sha256"] == _sha256(folder / "ledger.jsonl")
    assert (status["state"], status["template_evaluations"], status["template_rounds_done"]) == (
        "finished", last.decision.total_evaluations, result["template_rounds"])
    assert _accepted(rows), "小房間這一份設定要至少換過一面，下面終點那一題才有意義"


def test_v2_result_is_rejected_naming_the_version(runs: tuple[Path, Record]) -> None:
    """抓 v2 結果檔被當 v3 讀（少了換樣板那幾欄還照收）：產品的讀檔拒收、訊息講出 v2 的版本字串。"""
    v2 = cast(Record, _section(runs, "old_shape")["v2"])
    with pytest.raises(ValueError, match="aosr.material_gradient_result.v2"):
        run.parse_result(json.dumps(v2).encode("utf-8"))


def test_resume_of_pre_template_folder_opens_segment_and_overwrites_v2(runs: tuple[Path, Record]) -> None:
    """抓庚三之前跑完的資料夾（庚二收尾、沒有 templates.jsonl、result.json v2）接不上、被當成已跑完、或 v2 沒被覆寫：
    resume 回 0、換樣板帳本扣秒數等於一口氣跑完那一份、result.json 變 v3 且扣兩本帳摘要等於那一份；
    resume 之前報告照印（換樣板一節說還沒有帳本、終點未做）。"""
    case = _section(runs, "old_shape")
    before = _lines(cast(Record, case["report_before"]))
    assert TEMPLATE_HEADING in before and "  換樣板帳本：還沒有" in before
    resumed = cast(Record, case["resume"])
    assert resumed["code"] == 0, resumed["stderr"]
    copy, folder = Path(str(case["folder"])), _folder(runs)
    assert _without_seconds(_template_rows(copy)) == _without_seconds(_template_rows(folder))
    rewritten = _json(copy / "result.json")
    assert rewritten["format_version"] == "aosr.material_gradient_result.v5"  # #990 升 v5
    assert _without_hashes(rewritten) == _without_hashes(_json(folder / "result.json"))
    assert rewritten["template_ledger_sha256"] == _sha256(copy / "templates.jsonl")


# ---- endpoint 用最後配置 ----

def test_endpoint_exports_final_configuration(runs: tuple[Path, Record]) -> None:
    """抓終點仍用 project.presets：換過的那一面在 scheme-endpoint.json 是新家族的構造（材料摘要等於結果檔記的、而且
    不等於原 preset 那一份），不貼的面是方案原本那面的固定阻抗；endpoint.json 綁了換樣板帳本摘要。"""
    case = _section(runs, "endpoint")
    assert case["code"] == 0, case["stderr"]
    folder = _folder(runs)
    record = _json(folder / "endpoint.json")
    result = _json(folder / "result.json")
    assert record["result_template_ledger_sha256"] == result["template_ledger_sha256"] == _sha256(
        folder / "templates.jsonl")
    project, settings = run.read_project(folder), load_material_gradient_settings(runs[0] / "settings.toml")
    exported = load_scheme(folder / str(record["exported_scheme"]))
    zones = exported.scene.surface_zones_by_wall
    assert zones is not None
    kept = _kept_materials(runs)
    specs = cast(list[Record], result["surfaces"])
    designs = run.final_designs([(str(spec["surface_id"]), cast(str | None, spec["preset_id"])) for spec in specs],
                                settings, {wall: kept[wall.wall_name()] for wall in Wall.all()})
    theta = np.asarray([float.fromhex(value) for value in cast(list[str], result["theta_hex"])])
    materials = endpoint.materials_from_surfaces(designs, theta, settings)
    changed = 0
    for wall, spec, original in zip(Wall.all(), specs, project.presets, strict=True):
        name = wall.wall_name()
        digest = endpoint.material_digest(zones[name][0].material)
        wanted = kept[name] if spec["preset_id"] is None else materials[wall]
        assert wanted is not None and digest == endpoint.material_digest(wanted), name
        if spec["preset_id"] is None:
            assert isinstance(wanted, ConstantImpedance), name
        if spec["preset_id"] != original:
            changed += 1
            family = settings.family_of(str(spec["preset_id"])) if spec["preset_id"] is not None else None
            assert spec["preset_id"] is None or (family is not None and spec["preset_id"] == family.template_preset)
    assert changed >= 1


# ---- 報告 ----

def test_report_template_section_labels_each_candidate(runs: tuple[Path, Record]) -> None:
    """抓報告漏了換樣板一節、沒重調的標成重調過（或反過來）、換過去那面沒有空腔變數的沒標「無可重調變數」、
    停止原因不照字面、完整模型確認跑進換樣板一節。每個候選該帶的字由考卷從換樣板帳本自己判。"""
    lines = _lines(_section(runs, "report_before"))
    rows = _template_rows(_folder(runs))
    index = lines.index(TEMPLATE_HEADING)
    section = lines[index + 1:next(at for at, text in enumerate(lines) if text.startswith("最後的可行點"))]
    for row in rows:
        if row.phase != "quick_score":
            continue
        decision = next(item.decision for item in rows if item.round == row.round and item.decision is not None)
        retuned = {outcome.candidate for outcome in decision.outcomes}
        line = next(text for text in section if text.startswith(
            f"    {row.candidate}：快速評分 舊合分 {row.screening_total:.6f}／新合分 {row.value:.6f}"
            f"／代理 {row.proxy_db():.3f} dB，"))
        if row.candidate in retuned:
            assert RETUNED in line and QUICK_ONLY not in line, line
        else:
            assert QUICK_ONLY in line and RETUNED not in line, line
        assert (NOT_RETUNABLE in line) is (row.swapped_retunable is False), line
    assert f"  換樣板停止原因：{rows[-1].stop_reason}" in section
    assert not any("完整模型" in text for text in section)


def test_report_final_point_is_last_template_decision(runs: tuple[Path, Record]) -> None:
    """抓最後可行點還印庚二交出的配置：最後可行點那一節指名換樣板帳本最後一列，換過的面印新的 preset。"""
    lines = _lines(_section(runs, "report_before"))
    rows = _template_rows(_folder(runs))
    assert f"最後的可行點（換樣板帳本第 {rows[-1].step} 列 template_decision）：" in lines
    for surface_id, preset_id in rows[-1].surfaces:
        if preset_id is not None:
            assert any(line.startswith(f"  {surface_id}：{preset_id}，") for line in lines), surface_id


def test_flipped_template_ledger_marks_endpoint_stale(runs: tuple[Path, Record]) -> None:
    """抓報告不綁換樣板帳本：跑完終點之後 templates.jsonl 動一個位元（帳本照樣讀得回），報告印過期、點名換樣板帳本那兩件、
    不印終點數字；沒動之前報告印終點一節。"""
    after = _lines(_section(runs, "report_after"))
    assert STALE_ENDPOINT_LINE not in after and any(line.startswith(ENDPOINT_HEADING) for line in after)
    lines = _lines(_section(runs, "flipped"))
    index = lines.index(STALE_ENDPOINT_LINE)
    assert lines[index + 1] == "  對不上的：換樣板帳本檔跟終點綁的不同、結果宣稱的換樣板帳本不是現在這本"
    assert not any(line.startswith(ENDPOINT_HEADING) for line in lines)
