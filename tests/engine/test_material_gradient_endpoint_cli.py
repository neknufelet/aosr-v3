"""牆面材料梯度搜尋的終點命令列（#561 辛三）：endpoint、終點報告、匯出方案；審查修補：結果檔要由帳本重建才收、
報告綁現在的帳本檔與結果檔、帳本標頭綁原牆、低頻軸缺席開始前就拒收。

從 ``test_material_gradient_cli`` 搬出來（那份逼近一千行上限），自帶夾具、不跟它共用 ``runs``：兩份檔在雲端可能分到
不同片，共用就是整份命令列夾具在另一片重算。這裡只建驗過的那一份替身（同一間小房間、同一份收據寫法）、同一份方案
與縮小步數的設定、開始一次、跑 endpoint 與報告；改檔的情境都先抄一份資料夾再改。控制組（同一份輸入直接叫
``endpoint_check``、原方案六面完整模型解）在雙精度探針 ``_material_gradient_endpoint_cli_probe``。一場 pytest 只建一次、
平行的工人共用（``shared_json``）。

終點數字分兩層比（照辛一「逐位只比同一條路徑」的做法）：探針在自己的行程裡先叫產品入口
``worker.run_endpoint_worker`` 跑一份終點、再直接叫 ``endpoint_check`` 與完整模型基準，這兩份同一行程、同一組處理器
釘選，逐位比。命令列 ``endpoint`` 的子行程走 ``runtime.child_process_env``，刻意只繼承白名單環境、不帶三格處理器釘選
（NPY_DISABLE_CPU_FEATURES、OPENBLAS_CORETYPE、XLA_FLAGS），有 AVX-512 的機器上 JAX 走另一套指令、替身那幾格差在
最後幾位，所以命令列那一份只跟探針的產品入口那一份比：非浮點逐字，浮點用 ``|差| ≤ 2^-36 × max(|參考值|, 1)``。
2^-36 這個數字借自精度契約登記簿的 ``stage_one_objective_jax_vs_python``（JAX 對 Python 評分路徑留給換機器的尺），
但只借它的數值大小、另採混合比較規則（dB 欄以 1 dB 為尺度、無因次合分以 1 為尺度），不是逐字遵守那條契約；
這是 2026-10-30 明確批准、僅限這一張考卷的容差例外，不視為原契約已自動涵蓋，正式歸屬另記候選票 #985。
"""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
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
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_check import validate_reduced
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, save_validation_receipt
from aosr.physics.fem_reduced_store import load_reduced, save_reduced
from aosr.reporting.scheme import load_scheme
from tests.engine._precision_contracts import contract_value
from tests.engine._scheme_cache import shared_json
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine.test_material_gradient import _chain_model, _presets
from tests.engine.test_material_gradient_cli import (
    BAND_NAME, FREQUENCY_NAME, NARROWED_STEPS, PRESETS, WAIT_S, Record, _chain_operators, _command, _current_identity,
    _env, _holdouts, _json, _kept_materials, _run, _schemes, _settings_text, _sha256, _start_args,
)

PROBE = Path(__file__).with_name("_material_gradient_endpoint_cli_probe.py")
# 終點紀錄裡經過替身鏈或完整模型求值的浮點欄（命令列對產品入口用登記簿的尺比）；其餘欄逐字比。
COMPUTED_FLOATS: dict[str, tuple[str, ...]] = {
    "report": ("reduced_total", "reduced_components", "full_total", "full_components", "total_difference",
               "total_difference_bound", "max_frequency_db", "rms_floor_db", "frequency_db_by_seat", "band_db_by_seat",
               "partial_band_db_by_seat", "big_peak_db_by_speaker", "big_peak_db", "reduced_objective_total",
               "reduced_proxy_db", "full_objective_total", "full_proxy_db", "reduced_interior_max_db",
               "full_interior_max_db"),
    "baseline": ("full_total", "full_components", "big_peak_db_by_speaker", "big_peak_db", "full_objective_total",
                 "full_proxy_db", "full_interior_max_db"),
    "improvement": ("total", "components", "objective_total", "proxy_db"),
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
    full = _chain_operators(cast(FullOperators, request.getfixturevalue("operators")), chain)
    rulers = HoldoutRulers(frequency=Ruler(FREQUENCY_NAME, contract_value(FREQUENCY_NAME)),
                           band=Ruler(BAND_NAME, contract_value(BAND_NAME)))
    (directory / "good").mkdir()
    save_reduced(chain, directory / "good" / "rom.npz")
    save_validation_receipt(directory / "good" / "rom.npz", validate_reduced(chain, full, _holdouts()),
                            rulers=rulers, seconds=1.0)


def _inputs(directory: Path) -> None:
    """命令列考卷那一份方案與縮小步數的設定；另一份方案只把場景的低頻軸拿掉。"""
    scheme = _schemes()["scheme.json"]
    (directory / "scheme.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    no_axis = scheme.model_copy(update={"scene": scheme.scene.model_copy(update={"low_frequency_axis": None})})
    (directory / "scheme-no-axis.json").write_text(no_axis.model_dump_json(), encoding="utf-8")
    (directory / "settings.toml").write_text(_settings_text(NARROWED_STEPS), encoding="utf-8")


def _entries(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def _other_theta(folder: Path, settings: Path) -> list[str]:
    """帳本裡第一個跟最後決定不同、而且可行的 θ（換進結果檔冒充結果）。"""
    rows = ledger.read_ledger(folder / "ledger.jsonl")[1]
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


def _endpoint_runs(directory: Path, folder: Path) -> Record:
    """跑完的資料夾跑 endpoint、再印一次報告；另抄一份只有 project.json 的資料夾跑 endpoint（沒有結果檔要拒收）。"""
    record: Record = {"run": _run(directory, _command("endpoint", str(folder)))}
    record["report_after"] = _run(directory, _command("report", "--folder", str(folder)))
    bare = directory / "no-result" / folder.name
    bare.mkdir(parents=True)
    shutil.copyfile(folder / "project.json", bare / "project.json")
    record["no_result"] = {**_run(directory, _command("endpoint", str(bare))), "entries": _entries(bare)}
    return record


def _swapped(directory: Path, pristine: Path, theta_hex: list[str]) -> Record:
    """還沒跑過終點的一份資料夾：結果檔的 θ 換成帳本裡另一個可行點（帳本摘要那一格不動）跑 endpoint；
    再把結果檔換回原樣跑一次。"""
    original = (pristine / "result.json").read_bytes()
    before = _entries(pristine)
    _with_theta(pristine / "result.json", theta_hex)
    swapped = _json(pristine / "result.json")
    rejected = {**_run(directory, _command("endpoint", str(pristine))), "entries": _entries(pristine)}
    (pristine / "result.json").write_bytes(original)
    return {"before": before, "swapped": swapped, "rejected": rejected,
            "restored": _run(directory, _command("endpoint", str(pristine)))}


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
    _downgraded_ledger(copy / "ledger.jsonl", ledger.LEGACY_LEDGER_VERSION)
    _downgraded_ledger(copy / "templates.jsonl", templates.LEGACY_TEMPLATE_LEDGER_VERSION)
    result = _json(copy / "result.json")
    result["format_version"] = run.LEGACY_RESULT_VERSION
    for spec in cast(list[Record], result["surfaces"]):
        spec.pop("mount")  # #990 v5 才有每面安裝方式
    result.pop("screening_total")
    cast(Record, result["components"]).pop("big_peak_proxy_db")
    result["ledger_sha256"] = _sha256(copy / "ledger.jsonl")
    result["template_ledger_sha256"] = _sha256(copy / "templates.jsonl")
    (copy / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    record = _json(copy / "endpoint.json")
    record["format_version"] = endpoint_run.LEGACY_ENDPOINT_VERSION
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


def _probe(directory: Path, folder: Path) -> Record:
    """跑完搜尋的資料夾整份抄給探針（不帶命令列那一次的終點檔，探針自己用產品入口再寫一份）。"""
    work = directory / "probe"
    work.mkdir()
    shutil.copytree(folder, work / folder.name, ignore=shutil.ignore_patterns("endpoint.json", "scheme-endpoint*.json"))
    process = subprocess.run([sys.executable, str(PROBE), str(work), folder.name], cwd=work, env=_env(work),
                             capture_output=True, text=True, timeout=WAIT_S * 2)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(Record, json.loads((work / "output.json").read_text(encoding="utf-8")))


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
    pristine = directory / "pristine" / folder.name
    shutil.copytree(folder, pristine)
    theta_hex = _other_theta(folder, directory / "settings.toml")
    output["endpoint"] = _endpoint_runs(directory, folder)
    output["swapped"] = _swapped(directory, pristine, theta_hex)
    output["stale"] = {case: _stale(directory, folder, case, theta_hex) for case in STALE_CASES}
    output["kept_resume"] = _kept_resume(directory, folder)
    output["pre_proxy"] = _pre_proxy(directory, folder)
    no_axis = directory / "root-no-axis"
    output["no_axis"] = {**_run(directory, _start_args(directory, no_axis, scheme="scheme-no-axis.json")),
                         "root_exists": no_axis.exists()}
    output["probe"] = _probe(directory, folder)
    text = json.dumps(output, ensure_ascii=False)
    (directory / "fixture.json").write_text(text, encoding="utf-8")
    return text


@pytest.fixture(scope="module")
def runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
         ) -> tuple[Path, Record]:
    """一場 pytest 只建一次，平行的工人共用；讀的人只讀。"""
    directory = (tmp_path_factory.mktemp("material-gradient-endpoint-cli") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "material-gradient-endpoint-cli-shared")
    text = shared_json(tmp_path_factory, worker_id, "material-gradient-endpoint-cli",
                       lambda: _compute(directory, request))
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
    probe = _section(runs, "probe")
    assert probe["worker_code"] == 0
    record = EndpointRecord.model_validate(probe["worker_record"])
    probe = _section(runs, "probe", "endpoint")
    assert json.loads(record.report.to_json()) == probe["report"]
    assert record.baseline.full_total.hex() == probe["baseline_total"]
    assert [value.hex() for value in record.baseline.full_components] == probe["baseline_components"]
    assert [float(cast(float, value)).hex() for value in (
        record.baseline.full_objective_total, record.baseline.full_proxy_db,
        record.baseline.full_interior_max_db)] == probe["baseline_objective"]
    assert record.baseline.big_peak_db_by_speaker == probe["baseline_peaks"]
    assert record.baseline.full_solves == len(FEM_LANE_FREQUENCIES_HZ)


def _leaves(prefix: str, node: object) -> dict[str, object]:
    """攤平成「路徑 → 葉」；字典照鍵、序列照位置。"""
    if isinstance(node, dict):
        return {key: value for name, child in node.items() for key, value in _leaves(f"{prefix}.{name}", child).items()}
    if isinstance(node, list):
        return {key: value for index, child in enumerate(node)
                for key, value in _leaves(f"{prefix}[{index}]", child).items()}
    return {prefix: node}


def test_cli_endpoint_matches_entry_within_contract(runs: tuple[Path, Record]) -> None:
    """抓命令列終點跟產品入口算的不是同一件事：命令列 endpoint.json 跟探針產品入口那一份，非浮點（身分、摘要、檔名、
    完整解次數、喇叭與座位名、每面規格、丟掉的空腔、θ）逐字相等；終點、基準、改善的浮點每一格
    ``|差| ≤ 尺 × max(|探針那一格|, 1)``（尺的數字借自登記簿 ``stage_one_objective_jax_vs_python``；1 在 dB 欄是 1 dB、
    在無因次合分是 1；僅限本考卷的一次性例外，見模組說明）。同一行程的產品入口對直接呼叫另一題仍逐位比。
    不比秒數與完成時間。"""
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    cli = _leaves("", _without(_json(_folder(runs) / "endpoint.json")))
    entry = _leaves("", _without(cast(Record, _section(runs, "probe")["worker_record"])))
    assert cli.keys() == entry.keys()
    near = {key for key in cli if key.split("[", 1)[0].split(".")[1:3] in COMPUTED_PATHS}
    differing = {key: (cli[key], entry[key]) for key in cli.keys() - near if cli[key] != entry[key]}
    assert not differing, differing
    outside = {key: (cli[key], entry[key]) for key in near if not _within(cli[key], entry[key], tolerance)}
    assert not outside, outside
    assert near


def _without(record: Record) -> Record:
    """不比的欄：各段秒數與完成時間。"""
    return {key: value for key, value in record.items() if key not in ("seconds", "finished_at")}


def _within(actual: object, expected: object, tolerance: float) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    assert isinstance(actual, float) and isinstance(expected, float), (actual, expected)
    return abs(actual - expected) <= tolerance * max(abs(expected), UNIT_SCALE)


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


# ---- 審查修補：結果檔由帳本重建、報告綁現在的檔、帳本標頭綁原牆、低頻軸提前拒收 ----

def test_endpoint_rejects_result_with_swapped_theta_then_accepts_restored(runs: tuple[Path, Record]) -> None:
    """抓終點只信結果檔自己說的：結果檔的 θ 換成帳本裡另一個可行點、帳本摘要那一格還等於帳本檔，endpoint 回 1、
    標準錯誤講明結果檔跟帳本對不上並點名 ``theta_hex``、沒多寫任何檔；換回原檔再跑回 0，報告逐位等於原資料夾那一次的。"""
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
    record, original = _record(runs[0] / "pristine" / folder.name), _record(folder)
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
                                     _by_wall(_kept_materials(runs)))
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
    stored = ledger.read_ledger(folder / "ledger.jsonl")[0]
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
