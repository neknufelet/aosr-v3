"""牆面材料梯度搜尋的終點命令列（#561 辛三）：endpoint、終點報告、匯出方案；審查修補：結果檔要由帳本重建才收、
報告綁現在的帳本檔與結果檔、帳本標頭綁原牆、低頻軸缺席開始前就拒收。

從 ``test_material_gradient_cli`` 搬出來（那份逼近一千行上限），自帶夾具、不跟它共用 ``runs``：兩份檔在雲端可能分到
不同片，共用就是整份命令列夾具在另一片重算。這裡只建驗過的那一份替身（同一間小房間、同一份收據寫法）、同一份方案
與縮小步數的設定、開始一次、跑 endpoint 與報告；過期等情境抄副本，換 θ 拒收則在正式終點前還原。控制組（同一份輸入直接叫
``endpoint_check``、原方案六面完整模型解）在雙精度探針 ``_material_gradient_endpoint_cli_probe``。一場 pytest 只建一次、
平行的工人共用（``shared_json``）。

終點數字分兩層比（照辛一「逐位只比同一條路徑」的做法）：探針在自己的行程裡先叫產品入口
``worker.run_endpoint_worker`` 跑一份終點、再直接叫 ``endpoint_check`` 與完整模型基準，這兩份同一行程逐位比。
探針用 ``endpoint_probe_env``、命令列工作行程用 ``runtime.child_process_env``，兩者都不帶三格處理器釘選
（NPY_DISABLE_CPU_FEATURES、OPENBLAS_CORETYPE、XLA_FLAGS）。跨入口仍沿既有比較契約：非浮點逐字，浮點用
``|差| ≤ 2^-36 × max(|參考值|, 1)``，不把換環境產生的末位差異當成凍結答案。
2^-36 這個數字借自精度契約登記簿的 ``stage_one_objective_jax_vs_python``（JAX 對 Python 評分路徑留給換機器的尺），
但只借它的數值大小、另採混合比較規則（dB 欄以 1 dB 為尺度、無因次合分以 1 為尺度），不是逐字遵守那條契約；
這是 2026-10-30 明確批准、僅限這一張考卷的容差例外，不視為原契約已自動涵蓋，正式歸屬另記候選票 #985。
"""
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
from tests.engine._endpoint_v4_snapshot import SNAPSHOT_JSON
from tests.engine._legacy_gradient_search_probe import legacy_endpoint_copy
from tests.engine._six_search_cli_case import ENDPOINT_ROOM, model as model, operators as operators
from tests.engine.test_material_gradient import _chain_model, _presets
from tests.engine.test_material_gradient_cli import (
    BAND_NAME, FREQUENCY_NAME, NARROWED_STEPS, PRESETS, WAIT_S, Record, _command, _current_identity,
    _good_report, _json, _kept_materials, _run, _schemes, _settings_text, _sha256, _start_args,
)

PROBE = Path(__file__).with_name("_material_gradient_endpoint_cli_probe.py")
# 終點紀錄裡經過替身鏈或完整模型求值的浮點欄（命令列對產品入口用登記簿的尺比）；其餘欄逐字比。
COMPUTED_FLOATS: dict[str, tuple[str, ...]] = {
    "report": ("reduced_total", "reduced_components", "full_total", "full_components", "total_difference",
               "total_difference_bound", "max_frequency_db", "rms_floor_db", "frequency_db_by_seat", "band_db_by_seat",
               "partial_band_db_by_seat", "big_peak_db_by_speaker", "big_peak_db", "reduced_objective_total",
               "reduced_proxy_db", "full_objective_total", "full_proxy_db", "reduced_interior_max_db",
               "full_interior_max_db", "below_scoring"),
    "baseline": ("full_total", "full_components", "big_peak_db_by_speaker", "big_peak_db", "full_objective_total",
                 "full_proxy_db", "full_interior_max_db", "below_scoring"),
    "improvement": ("total", "components", "objective_total", "proxy_db", "below_scoring"),
}
COMPUTED_PATHS = [[part, name] for part, names in COMPUTED_FLOATS.items() for name in names]
# 近零參考值的尺度：dB 欄（分項、大峰、逐頻與帶差）是 1 dB，無因次合分（total、full_total、reduced_total、
# total_difference）是 1；兩種單位同一個數字，不是全部都 1 dB。借 stage_one_objective_jax_vs_python 契約
# truth 那一句「改用 |差|/(1 dB)」的形狀，混合規則見模組說明與候選票 #985。
UNIT_SCALE = 1.0
# 報告過期的六種改法 → 報告「對不上的」那一行要點名的原因（照 stale_reasons 的順序）。
# result_claim、unclosed 兩種連 endpoint.json 的摘要欄一起改成跟改後的檔一致（只剩結果↔帳本對不上）。
STALE_CASES: dict[str, tuple[str, ...]] = {
    "ledger": ("帳本檔跟終點綁的不同", "帳本沒收尾", "結果宣稱的帳本不是現在這本"),
    "result": ("result.json 跟終點讀的不同",),
    "endpoint_id": ("endpoint.json 的 gradient_id 不是這次搜尋的",),
    "project_id": ("endpoint.json 的 gradient_id 不是這次搜尋的", "result.json 的 gradient_id 不是這次搜尋的"),
    "result_claim": ("結果宣稱的帳本不是現在這本",),
    "unclosed": ("帳本沒收尾", "結果宣稱的帳本不是現在這本"),
}


# ---- 夾具：一份替身、開始一次、終點與改檔情境 ----

def _surrogate(directory: Path, request: pytest.FixtureRequest) -> None:
    """驗過的那一份替身：寫法照 ``test_material_gradient_cli._surrogates`` 的 good（現在的物理身分＋留出收據）。"""
    chain = dataclasses.replace(_chain_model(cast(ReducedModel, request.getfixturevalue("model"))),
                                physics_identity=_current_identity())
    rulers = HoldoutRulers(frequency=Ruler(FREQUENCY_NAME, contract_value(FREQUENCY_NAME)),
                           band=Ruler(BAND_NAME, contract_value(BAND_NAME)))
    (directory / "good").mkdir()
    save_reduced(chain, directory / "good" / "rom.npz")
    save_validation_receipt(directory / "good" / "rom.npz", _good_report(request, chain), rulers=rulers, seconds=1.0)


def _inputs(directory: Path) -> None:
    """命令列考卷那一份方案與縮小步數的設定；另一份方案只把場景的低頻軸拿掉。"""
    scheme = _schemes()["scheme.json"]
    scheme = scheme.model_copy(update={"scene": scheme.scene.model_copy(update={"room_m": ENDPOINT_ROOM})})
    (directory / "scheme.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    no_axis = scheme.model_copy(update={"scene": scheme.scene.model_copy(update={"low_frequency_axis": None})})
    (directory / "scheme-no-axis.json").write_text(no_axis.model_dump_json(), encoding="utf-8")
    (directory / "settings.toml").write_text(_settings_text(NARROWED_STEPS), encoding="utf-8")


def _entries(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def _other_theta(folder: Path, settings: Path) -> list[str]:
    """帳本裡第一個跟最後決定不同、而且可行的 θ（換進結果檔冒充結果）。"""
    rows = ledger.read_ledger(folder / "ledger.jsonl", legacy=True)[1]
    designs = gradient_search.wall_designs_from_presets(
        PRESETS, load_material_gradient_settings(settings), _presets())
    return next(list(row.theta_hex) for row in rows if row.theta_hex != rows[-1].theta_hex
                and gradient_search.is_feasible(np.asarray(row.theta()), designs))


def _with_theta(path: Path, theta_hex: list[str]) -> None:
    result = _json(path)
    result["theta_hex"] = theta_hex
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def _with_field(path: Path, key: str, value: str) -> None:
    document = _json(path)
    document[key] = value
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def _endpoint_runs(directory: Path, folder: Path, theta_hex: list[str]) -> Record:
    """同一份 legacy 夾先拒收竄改並還原，再正式跑 endpoint；另守缺結果檔拒收。"""
    swapped = _swapped(directory, folder, theta_hex)
    record: Record = {"run": _run(directory, _command("endpoint", str(folder))), "swapped": swapped}
    swapped['restored'] = record['run']
    record["report_after"] = _run(directory, _command("report", "--folder", str(folder)))
    bare = directory / "no-result" / folder.name
    bare.mkdir(parents=True)
    shutil.copyfile(folder / "project.json", bare / "project.json")
    record["no_result"] = {**_run(directory, _command("endpoint", str(bare))), "entries": _entries(bare)}
    return record


def _swapped(directory: Path, folder: Path, theta_hex: list[str]) -> Record:
    """正式終點前先換 θ 驗拒收、再還原；成功那次由 _endpoint_runs 執行並記入 restored。"""
    original = (folder / "result.json").read_bytes()
    before = _entries(folder)
    _with_theta(folder / "result.json", theta_hex)
    swapped = _json(folder / "result.json")
    rejected = {**_run(directory, _command("endpoint", str(folder))), "entries": _entries(folder)}
    (folder / "result.json").write_bytes(original)
    return {"before": before, "swapped": swapped, "rejected": rejected}


def _stale(directory: Path, folder: Path, case: str, theta_hex: list[str]) -> Record:
    """抄一份跑完終點的資料夾、照 ``case`` 改檔、印報告。result_claim：結果檔的帳本摘要換掉，endpoint.json 的
    ``result_sha256`` 同步成改後結果檔的摘要；unclosed：帳本截掉最後一列，endpoint.json 的 ``result_ledger_sha256``
    同步成截後帳本的摘要。"""
    copy = directory / "stale" / case / folder.name
    shutil.copytree(folder, copy)
    if case in ("ledger", "unclosed"):
        lines = (copy / "ledger.jsonl").read_bytes().splitlines(keepends=True)
        (copy / "ledger.jsonl").write_bytes(b"".join(lines[:-1]))
        if case == "unclosed":
            _with_field(copy / "endpoint.json", "result_ledger_sha256", _sha256(copy / "ledger.jsonl"))
    elif case == "result_claim":
        _with_field(copy / "result.json", "ledger_sha256", "0" * 64)
        _with_field(copy / "endpoint.json", "result_sha256", _sha256(copy / "result.json"))
    elif case == "result":
        _with_theta(copy / "result.json", theta_hex)
    else:
        name = "endpoint.json" if case == "endpoint_id" else "project.json"
        document = _json(copy / name)
        document["gradient_id"] = "f" * 32 if document["gradient_id"] != "f" * 32 else "e" * 32
        (copy / name).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return _run(directory, _command("report", "--folder", str(copy)))


def _kept_resume(directory: Path, folder: Path) -> Record:
    """抄一份跑完的資料夾，帳本標頭的原牆摘要改掉一面（同長度的另一個十六進位字），接續要拒收、帳本不動。"""
    copy = directory / "kept-resume" / folder.name
    shutil.copytree(folder, copy)
    lines = (copy / "ledger.jsonl").read_bytes().splitlines(keepends=True)
    header = json.loads(lines[0])
    face = sorted(header["kept_digests"])[0]
    header["kept_digests"][face] = "0" * len(header["kept_digests"][face])
    (copy / "ledger.jsonl").write_bytes(b"".join([(json.dumps(header, ensure_ascii=False) + "\n").encode(), *lines[1:]]))
    before = (copy / "ledger.jsonl").read_bytes()
    return {**_run(directory, _command("resume", str(copy))), "face": face,
            "ledger_unchanged": (copy / "ledger.jsonl").read_bytes() == before}


_PROXY_FIELDS = ("reduced_objective_total", "reduced_proxy_db", "full_objective_total", "full_proxy_db",
                 "proxy_gap_bound_db", "reduced_interior_max_db", "full_interior_max_db")


def _downgraded_ledger(path: Path, version: str) -> None:
    """帳本換回加代理之前的形狀：標頭版本換成上一版，每列拿掉舊合分、只留兩分項（值不動，只換形狀）。"""
    lines = path.read_bytes().splitlines()
    header = json.loads(lines[0])
    header["ledger_version"] = version
    header.pop("score_constants")
    rows = []
    for line in lines[1:]:
        row = json.loads(line)
        row.pop("screening_total")
        row["components"] = row["components"][:2]
        rows.append(row)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in (header, *rows)), encoding="utf-8")


def _pre_proxy(directory: Path, folder: Path) -> Record:
    """示範房那種形狀（加大峰代理之前跑完的資料夾）：抄一份跑完終點的資料夾，project.json 拿掉梯度目標指紋、
    帳本 v4、換樣板帳本 v1、result.json v3、endpoint.json v2（都拿掉壬一加的欄），各檔互相記的摘要同步成改後的檔，
    印報告、再接續（接續要拒收、帳本不動）。不讀真示範房資料夾。"""
    copy = directory / "pre-proxy" / folder.name
    shutil.copytree(folder, copy)
    project = _json(copy / "project.json")
    project.pop("objective_settings_fingerprint")
    (copy / "project.json").write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
    _downgraded_ledger(copy / "ledger.jsonl", ledger.PRE_PROXY_LEDGER_VERSION)
    _downgraded_ledger(copy / "templates.jsonl", templates.PRE_PROXY_TEMPLATE_LEDGER_VERSION)
    result = _json(copy / "result.json")
    result["format_version"] = run.LEGACY_RESULT_VERSION
    result.pop("screening_total")
    cast(Record, result["components"]).pop("big_peak_proxy_db")
    result["ledger_sha256"] = _sha256(copy / "ledger.jsonl")
    result["template_ledger_sha256"] = _sha256(copy / "templates.jsonl")
    (copy / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    record = _json(copy / "endpoint.json")
    record["format_version"] = endpoint_run.LEGACY_ENDPOINT_VERSION
    record.pop('six_category', None)
    for part in ("report", "baseline", "improvement"):
        cast(Record, record[part]).pop("below_scoring")
    for part, names in (("report", _PROXY_FIELDS), ("baseline", ("full_objective_total", "full_proxy_db",
                                                                  "full_interior_max_db")),
                        ("improvement", ("objective_total", "proxy_db"))):
        for name in names:
            cast(Record, record[part]).pop(name)
    record["result_ledger_sha256"] = result["ledger_sha256"]
    record["result_template_ledger_sha256"] = result["template_ledger_sha256"]
    record["result_sha256"] = _sha256(copy / "result.json")
    (copy / "endpoint.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    before = (copy / "ledger.jsonl").read_bytes()
    output: Record = {"folder": str(copy), "report": _run(directory, _command("report", "--folder", str(copy)))}
    output["resume"] = {**_run(directory, _command("resume", str(copy))),
                        "ledger_unchanged": (copy / "ledger.jsonl").read_bytes() == before}
    return output


def _probe_work(directory: Path, folder: Path) -> Path:
    """每份控制各用自己的資料夾與完整解；只共用探針行程的初始化。"""
    work = directory / "probe"
    work.mkdir(parents=True)
    shutil.copytree(folder, work / folder.name, ignore=shutil.ignore_patterns("endpoint.json", "scheme-endpoint*.json"))
    return work


def _probe(work: Path, name: str) -> Record:
    """一份獨立資料夾，完整跑 worker、直接解與顯卡替身三組控制，不共用答案。"""
    process = subprocess.run([sys.executable, str(PROBE), str(work), name],
                             cwd=work, env=endpoint_probe_env(work),
                             capture_output=True, text=True, timeout=WAIT_S * 2)
    (work / "probe.stdout").write_text(process.stdout)
    (work / "probe.stderr").write_text(process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(Record, json.loads((work / "output.json").read_text(encoding="utf-8")))


def _search_runs(directory: Path, request: pytest.FixtureRequest) -> Record:
    """真搜尋與舊目標重建；下游只計算自己會檢查的終點情境。"""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    _surrogate(directory, request)
    _inputs(directory)
    root = directory / "root"
    output: Record = {"start": _run(directory, _start_args(directory, root))}
    assert cast(Record, output["start"])["code"] == 0, output["start"]
    (folder,) = (path for path in root.iterdir() if not path.name.startswith("."))
    output["new_folder"] = str(folder)
    folder = legacy_endpoint_copy(directory, folder)
    output["folder"] = str(folder)
    return output


def _compute(directory: Path, request: pytest.FixtureRequest) -> str:
    output = _search_runs(directory, request)
    folder = Path(str(output["folder"]))
    current = Path(str(output["new_folder"]))
    # 探針先抄未竄改的輸入；legacy 正式終點在自己的工作內先拒收換 θ、還原，再完整計算。
    work = _probe_work(directory, current)
    old_work = _probe_work(directory / "legacy-probe", folder)
    theta_hex = _other_theta(folder, directory / "settings.toml")
    # 單一終點本機峰值約 4.5 GB；雲端 16 GB／四核還與其他 xdist 工人共用。
    # 四個並行在 run 42578109442 比串行還慢，故最多兩個；換頁原因仍未驗證。
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 2)) as pool:
        # 最長的兩份探針先排，各自資料夾、環境與控制皆獨立。
        futures = {"probe": pool.submit(_probe, work, current.name),
                   "legacy_probe": pool.submit(_probe, old_work, folder.name),
                   "new_endpoint": pool.submit(_run, directory, _command("endpoint", str(current))),
                   "endpoint": pool.submit(_endpoint_runs, directory, folder, theta_hex),
                   "no_axis": pool.submit(_no_axis, directory)}
        # 過期報告與前代理副本必須等舊低頻終點完成，才能抄完整的 endpoint.json。
        output["endpoint"] = futures.pop("endpoint").result()
        output['swapped'] = cast(Record, output['endpoint']).pop('swapped')
        stale = {case: pool.submit(_stale, directory, folder, case, theta_hex) for case in STALE_CASES}
        futures["pre_proxy"] = pool.submit(_pre_proxy, directory, folder)
        # 原牆摘要拒收題要抄新資料夾整份，等其終點寫完，避免抄到原子寫檔的暫存檔。
        output["new_endpoint"] = futures.pop("new_endpoint").result()
        futures["kept_resume"] = pool.submit(_kept_resume, directory, current)
        output.update({name: future.result() for name, future in futures.items()})
        output["stale"] = {case: future.result() for case, future in stale.items()}
    text = json.dumps(output, ensure_ascii=False)
    (directory / "fixture.json").write_text(text, encoding="utf-8")
    return text


def _no_axis(directory: Path) -> Record:
    """只有輸入拒收，獨立於搜尋完成後的所有終點與改檔情境。"""
    root = directory / "root-no-axis"
    return {**_run(directory, _start_args(directory, root, scheme="scheme-no-axis.json")),
            "root_exists": root.exists()}


@pytest.fixture(scope="module")
def full_runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
         ) -> tuple[Path, Record]:
    """一場 pytest 只建一次，平行的工人共用；讀的人只讀。"""
    directory = (tmp_path_factory.mktemp("material-gradient-endpoint-cli") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "material-gradient-endpoint-cli-shared")
    text = shared_json(tmp_path_factory, worker_id, "material-gradient-endpoint-cli",
                       lambda: _compute(directory, request))
    return directory, cast(Record, json.loads(text))


@pytest.fixture(scope="module")
def minimal_runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory,
                 worker_id: str) -> tuple[Path, Record]:
    """換版考卷只需一份真終點；不預算它未讀取的交換 θ、過期報告與數值探針。"""
    directory = (tmp_path_factory.mktemp("endpoint-range") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "endpoint-range-shared")

    def produce() -> str:
        output = _search_runs(directory, request)
        result = _run(directory, _command("endpoint", str(output["folder"])))
        assert result["code"] == 0, result["stderr"]
        return json.dumps(output, ensure_ascii=False)

    text = shared_json(tmp_path_factory, worker_id, "endpoint-range", produce)
    return directory, cast(Record, json.loads(text))


def _section(runs: tuple[Path, Record], *keys: str) -> Record:
    node: object = runs[1]
    for key in keys:
        node = cast(Record, node)[key]
    return cast(Record, node)


def _folder(runs: tuple[Path, Record]) -> Path:
    return Path(str(runs[1]["folder"]))


def _record(folder: Path) -> EndpointRecord:
    return EndpointRecord.model_validate_json((folder / "endpoint.json").read_bytes())


def _by_wall(kept: dict[str, SurfaceMaterial]) -> dict[Wall, SurfaceMaterial]:
    return {wall: kept[wall.wall_name()] for wall in Wall.all()}


def _report_lines(case: Record) -> list[str]:
    assert case["code"] == 0, case["stderr"]
    return str(case["stdout"]).splitlines()


# ---- 終點：寫出的紀錄、對控制組、匯出方案、報告 ----





def _leaves(prefix: str, node: object) -> dict[str, object]:
    """攤平成「路徑 → 葉」；字典照鍵、序列照位置。"""
    if isinstance(node, dict):
        return {key: value for name, child in node.items() for key, value in _leaves(f"{prefix}.{name}", child).items()}
    if isinstance(node, list):
        return {key: value for index, child in enumerate(node)
                for key, value in _leaves(f"{prefix}[{index}]", child).items()}
    return {prefix: node}




def _without(record: Record) -> Record:
    """不比的欄：各段秒數與完成時間。"""
    return {key: value for key, value in record.items() if key not in ("seconds", "finished_at")}


def _within(actual: object, expected: object, tolerance: float) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    assert isinstance(actual, float) and isinstance(expected, float), (actual, expected)
    return abs(actual - expected) <= tolerance * max(abs(expected), UNIT_SCALE)












# ---- 審查修補：結果檔由帳本重建、報告綁現在的檔、帳本標頭綁原牆、低頻軸提前拒收 ----













def _batch_probe(folder: Path, mode: str, directory: Path) -> Record:
    result = subprocess.run([sys.executable, "-m", "tests.engine._frequency_batch_probe", mode, str(folder)],
                            cwd=directory, env=endpoint_probe_env(directory), capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    return cast(Record, json.loads(result.stdout))






def test_endpoint_record_omits_cpu_defaults_and_keeps_gpu_metadata(tmp_path: Path) -> None:
    """抓 CPU 預設多寫 JSON 位元組或 GPU 兩欄漏寫：device 是 cpu（預設或明給）、修正次數是 None 時兩欄都不寫，
    位元組等於扣掉兩欄的傾印；GPU 那份兩欄照寫、讀回相同。紀錄用 v4 快照（快照本身是 GPU 那份，扣掉兩欄就是 CPU
    預設；兩欄各版都有）；v5 真終點同一題在 slow_ 檔。"""
    document = cast(Record, json.loads(SNAPSHOT_JSON)["record"])
    gpu = EndpointRecord.model_validate(document)
    assert gpu.device == "gpu" and gpu.max_refinement_steps is not None
    record = EndpointRecord.model_validate(
        {key: value for key, value in document.items() if key not in ("device", "max_refinement_steps")})
    original = record.model_dump(mode="json", exclude={"device", "max_refinement_steps"})
    expected = (json.dumps(original, ensure_ascii=False, allow_nan=False) + "\n").encode()
    for candidate in (record, record.model_copy(update={"device": "cpu"})):
        run.write_json_atomic(tmp_path / "endpoint.json", candidate)
        assert (tmp_path / "endpoint.json").read_bytes() == expected
    run.write_json_atomic(tmp_path / "endpoint.json", gpu)
    written = json.loads((tmp_path / "endpoint.json").read_bytes())
    assert (written["device"], written["max_refinement_steps"]) == (gpu.device, gpu.max_refinement_steps)
    assert written == document and EndpointRecord.model_validate(written) == gpu
