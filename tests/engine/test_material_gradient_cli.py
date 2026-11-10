"""牆面材料梯度搜尋命令列（#561 辛一）：開始、接續、停止記號、報告、拒收與資料夾鎖。

替身用 ``test_fem_reduced`` 的小房間（同一個 ``model``／``operators`` 夾具），照 ``test_material_gradient`` 衍生成
一主位加六周圍點、整條正式有限元素軸（``_chain_model``），物理身分換成這份程式現在的物理身分（考卷原本那份是全零，
``start`` 會拿方案匯出的身分去比）；驗證收據照 ``test_fem_reduced_receipt`` 的寫法：留出材料對完整解（座位取值列換成
同樣的七個組合）、兩把尺從精度契約登記簿讀。方案是 ``_fem_shared_cases.small_scheme``（同一間房、同兩支喇叭、同一組
空氣），座位組換成同一組主位加六點——匯出完整模型的網格雜湊就跟替身相同。搜尋設定把連續下降與重調的步數上限調小
（只為了考卷快；其他照正式設定檔），寫在暫存資料夾、用 ``--settings`` 給。

修補一（審查 B1、B2）：替身驗證收據記喇叭與座位座標，方案把一支喇叭或一個座位移 0.1 m（網格雜湊不變）要被
start 拒收；接續前方案檔換過要拒收；同一個根、不同設定落在不同資料夾；建子行程到等待之間的中斷也要收子行程。

命令列一律開真的子行程跑（``python -m aosr.gradient.cli``），雙精度子行程由它自己起。逐位比較只比同一條路徑跑出來的：
命令列跑的跟命令列跑的比；「第 N 次求值時放停止記號」與「前段只寫一半」在探針（``_material_gradient_cli_probe``，
雙精度子行程、只叫產品入口）裡跟同一個探針跑的參考比。一場 pytest 只建一次、平行的工人共用（``shared_json``）。
"""
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

if TYPE_CHECKING:
    from _typeshed import TraceFunction

PROBE = Path(__file__).with_name("_material_gradient_cli_probe.py")
Record = dict[str, object]
PRESETS: tuple[str | None, ...] = CASES[DESCENT_CASE]
PRESET_TEXT = ",".join("none" if preset is None else preset for preset in PRESETS)
OUTSIDE_FAMILY = "fabric_pet_a_gap100"  # construction_presets.toml 有、material_gradient.toml 的家族清單沒有
FREQUENCY_NAME, BAND_NAME = "fem_reduced_holdout_frequency_db", "fem_reduced_holdout_band_db"
# 考卷只為了快把兩個步數上限調小（正式設定檔的 [descent] max_steps 與 [retune] steps），換樣板段的求值上限調成 1（第 0 輪就停、不求值，庚三另有考卷）。
NARROWED_STEPS = {"descent.max_steps": 3, "retune.steps": 2, "templates.max_evaluations": 1, "candidates.max_rounds": 1, "candidates.max_per_round": 1}
# 同一個根、另一份設定（只差下降步數上限）：資料夾名要跟著不同。
OTHER_STEPS = {"descent.max_steps": 2, "retune.steps": 2, "templates.max_evaluations": 1, "candidates.max_rounds": 1, "candidates.max_per_round": 1}
MOVED_SPEAKER, MOVED_SEAT, MOVE_M = "left", "front", 0.1
OTHER_DENSITY, OTHER_SPEED = 1.18, 340.0
# 同網格、同座標、另一組空氣建的替身（各有合格收據）：探針用的那份空氣兩格都換，拒收題各只換一格。
AIR_SURROGATES = {"air-other": (OTHER_DENSITY, OTHER_SPEED), "other-density": (OTHER_DENSITY, SOUND_SPEED),
                  "other-speed": (DENSITY, OTHER_SPEED)}
# 顧問 #971 的隔離反例（只交 80、100、125 Hz）與宣告軸拿掉中間一點：各自在自己的軸上做留出驗證、收據合格，
# start 與 resume 都要拒收，理由碼照低頻評估器的宣告點判法。
AXIS_SURROGATES = {"narrow-axis": ((80.0, 100.0, 125.0), StageOneReasonCode.FREQUENCY_AXIS_MISMATCH),
                   "missing-point": (FEM_LANE_FREQUENCIES_HZ[:47] + FEM_LANE_FREQUENCIES_HZ[48:],
                                     StageOneReasonCode.MISSING_DECLARED_POINT)}
WAIT_S = 600.0


# ---- 夾具：替身、方案、設定、命令列各跑一次 ----

def _holdouts(axis: tuple[float, ...] = FEM_LANE_FREQUENCIES_HZ) -> list[FrequencyWallImpedances]:
    """留出材料：同 ``test_fem_reduced`` 的抽法與種子（99），頻率軸預設是整條正式有限元素軸（衍生替身的軸）。"""
    rng = np.random.default_rng(99)
    frequencies = np.asarray(axis)
    sets: list[FrequencyWallImpedances] = []
    for _ in range(2):
        walls = {}
        for wall in Wall.all():
            resistance = math.exp(rng.uniform(math.log(1.5), math.log(60.0)))
            reactance = rng.uniform(0.0, 5.0)
            walls[wall] = tuple(complex(z) for z in DENSITY * SOUND_SPEED * (resistance - 1j * reactance * 100.0 / frequencies))
        sets.append(walls)
    return sets


def _chain_operators(full: FullOperators, chain: ReducedModel) -> FullOperators:
    """完整模型的座位取值列換成衍生替身同樣的七個組合（``_chain_model`` 的組法），名字照衍生替身。"""
    front, back, side = (full.receivers[name] for name in ("front", "back", "side"))
    rows = [front, back, side, (front + back) / 2, (front + side) / 2, (back + side) / 2, (front + back + side) / 3]
    points = {point.receiver_id: Point(*point.position_m) for point in _receivers().points}
    assert tuple(points) == chain.receiver_names
    return dataclasses.replace(full, receivers=dict(zip(chain.receiver_names, rows, strict=True)),
                               receiver_points=points)


def _current_identity() -> str:
    return identity_module.physics_identity(capabilities=load_capabilities(config_path("capabilities.toml")),
                                            directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))


def _good_report(request: pytest.FixtureRequest, chain: ReducedModel) -> ValidationReport:
    """驗過的那份替身（good）的留出驗證：本檔與終點命令列那份同一份輸入，一場 pytest 只做一次（指紋進共用名）。"""
    return shared_holdout_report(
        cast(pytest.TempPathFactory, request.getfixturevalue("tmp_path_factory")),
        cast(str, request.getfixturevalue("worker_id")), chain,
        _chain_operators(cast(FullOperators, request.getfixturevalue("operators")), chain), _holdouts())


def _surrogates(directory: Path, request: pytest.FixtureRequest, group: str) -> None:
    """替身：驗過的（現在的物理身分＋收據）、沒有收據的、物理身分全零但有收據的，與同網格同座標、另一組空氣建的
    三份（完整模型算子的空氣跟著換、各自做留出驗證寫收據）。"""
    base = _chain_model(cast(ReducedModel, request.getfixturevalue("model")))
    chain = dataclasses.replace(base, physics_identity=_current_identity())
    rulers = HoldoutRulers(frequency=Ruler(FREQUENCY_NAME, contract_value(FREQUENCY_NAME)),
                           band=Ruler(BAND_NAME, contract_value(BAND_NAME)))
    if group == "base":
        report = _good_report(request, chain)
        # 物理身分全零那份要自己驗一次：報告帶驗證當下的計算條件指紋（含物理身分），借 chain 的報告寫不成它的收據（#983）。
        zero_report = validate_reduced(
            base, _chain_operators(cast(FullOperators, request.getfixturevalue("operators")), base), _holdouts())
        for name, surrogate, surrogate_report in (("good", chain, report), ("unvalidated", chain, None),
                                                  ("zero-identity", base, zero_report)):
            (directory / name).mkdir()
            save_reduced(surrogate, directory / name / "rom.npz")
            if surrogate_report is not None:
                save_validation_receipt(directory / name / "rom.npz", surrogate_report, rulers=rulers, seconds=1.0)
    full = _chain_operators(cast(FullOperators, request.getfixturevalue("operators")), chain)
    for name, (density, speed) in AIR_SURROGATES.items():
        if group != name:
            continue
        other = dataclasses.replace(chain, density_kg_m3=density, sound_speed_m_s=speed)
        operators = dataclasses.replace(full, density_kg_m3=density, sound_speed_m_s=speed)
        (directory / name).mkdir()
        save_reduced(other, directory / name / "rom.npz")
        save_validation_receipt(directory / name / "rom.npz", validate_reduced(other, operators, _holdouts()),
                                rulers=rulers, seconds=1.0)
    for name, (axis, _) in AXIS_SURROGATES.items():
        if group != name:
            continue
        other = dataclasses.replace(chain, frequencies_hz=axis)
        (directory / name).mkdir()
        save_reduced(other, directory / name / "rom.npz")
        save_validation_receipt(directory / name / "rom.npz", validate_reduced(other, full, _holdouts(axis)),
                                rulers=rulers, seconds=1.0)


def _moved(point: Point) -> Point:
    return Point(point.x + MOVE_M, point.y, point.z)


def _moved_seat(receivers: ReceiverSet) -> ReceiverSet:
    points = tuple(point.model_copy(update={"position_m": _moved(Point(*point.position_m)).as_tuple()})
                   if point.receiver_id == MOVED_SEAT else point for point in receivers.points)
    return receivers.model_copy(update={"points": points})


def _schemes() -> dict[str, Scheme]:
    """正式方案（場景補搜尋低頻軸：終點量大峰要它），與同一間房只把一支喇叭或一個座位的 x 移 0.1 m 的兩份。"""
    base = shared_cases.small_scheme()
    scene = base.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.SEARCH, "room_m": CLI_ROOM})
    scheme = base.model_copy(update={"receiver_set": _receivers(), "purpose": PURPOSE, "scene": scene})
    speakers = {**scheme.speakers, MOVED_SPEAKER: _moved(scheme.speakers[MOVED_SPEAKER])}
    return {"scheme.json": scheme, "scheme-moved-speaker.json": scheme.model_copy(update={"speakers": speakers}),
            "scheme-moved-seat.json": scheme.model_copy(update={"receiver_set": _moved_seat(scheme.receiver_set)})}


def _settings_text(steps: dict[str, int]) -> str:
    text = config_path("material_gradient.toml").read_text(encoding="utf-8")
    for (section, name), value in ((key.split(".", 1), value) for key, value in steps.items()):  # 鍵寫「表.鍵」
        text = re.sub(rf"(?ms)(^\[{section}\]$.*?^){name} = \d+$", rf"\g<1>{name} = {value}", text, count=1)
    return text


def _inputs(directory: Path) -> None:
    for name, scheme in _schemes().items():
        (directory / name).write_text(scheme.model_dump_json(), encoding="utf-8")
    for name, steps in (("settings.toml", NARROWED_STEPS), ("settings-other.toml", OTHER_STEPS)):
        (directory / name).write_text(_settings_text(steps), encoding="utf-8")
        narrowed = load_material_gradient_settings(directory / name)
        for key, value in steps.items():
            section, field = key.split('.', 1)
            assert getattr(getattr(narrowed, section), field) == value


def _env(directory: Path) -> dict[str, str]:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    return env


def _command(*args: str) -> list[str]:
    return [sys.executable, "-m", "aosr.gradient.cli", *args]


def _start_args(directory: Path, root: Path, surrogate: str = "good", presets: str = PRESET_TEXT,
                scheme: str = "scheme.json", settings: str = "settings.toml") -> list[str]:
    return _command("start", "--scheme", str(directory / scheme), "--surrogate",
                    str(directory / surrogate / "rom.npz"), "--presets", presets, "--purpose", PURPOSE,
                    "--root", str(root), "--settings", str(directory / settings), "--engine-commit", "test")


def _run(directory: Path, argv: list[str]) -> Record:
    done = subprocess.run(argv, cwd=directory, env=_env(directory), capture_output=True, text=True, timeout=WAIT_S)
    return {"code": done.returncode, "stdout": done.stdout, "stderr": done.stderr}


def _running_pid(folder: Path, process: subprocess.Popen[str]) -> int:
    """等子行程把 running 寫進狀態檔，回它的行程編號（子行程是自己那一群的首領）。"""
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"命令列在子行程寫 running 之前就結束了：{process.returncode}")
        try:
            status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = None
        if isinstance(status, dict) and status.get("state") == "running":
            return int(status["pid"])
        time.sleep(0.05)
    raise AssertionError("等不到子行程寫 running")


@contextmanager
def _paused(folder: Path, process: subprocess.Popen[str]) -> Iterator[None]:
    """子行程寫完 running 就整群暫停（它還拿著資料夾鎖），區塊結束一定讓它繼續。"""
    pid = _running_pid(folder, process)
    os.killpg(pid, signal.SIGSTOP)
    try:
        yield
    finally:
        os.killpg(pid, signal.SIGCONT)


def _first_folder(root: Path, process: subprocess.Popen[str]) -> Path:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        found = [path for path in root.iterdir() if not path.name.startswith(".")] if root.exists() else []
        if found:
            return found[0]
        if process.poll() is not None:
            raise AssertionError(f"命令列沒建資料夾就結束了：{process.returncode}")
        time.sleep(0.05)
    raise AssertionError("等不到資料夾")


def _concurrent_start(directory: Path) -> Record:
    """開始一次；它的子行程寫完 running 就暫停，趁鎖還拿著再開始同一組、再接續同一個資料夾，然後讓它跑完。"""
    root = directory / "root-a"
    with subprocess.Popen(_start_args(directory, root), cwd=directory, env=_env(directory), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        folder = _first_folder(root, process)
        with _paused(folder, process):
            second = _run(directory, _start_args(directory, root))
            resume = _run(directory, _command("resume", str(folder)))
        stdout, stderr = process.communicate(timeout=WAIT_S)
    return {"folder": str(folder), "first": {"code": process.returncode, "stdout": stdout, "stderr": stderr},
            "second_start": second, "busy_resume": resume}


def _cut_copy(directory: Path, source: Path, name: str) -> Path:
    """複製開始那份的 project.json 與帳本，帳本截在第 0 輪基準重調那一批後面（第 0 輪沒寫完）。"""
    folder = directory / name / source.name
    folder.mkdir(parents=True)
    shutil.copyfile(source / "project.json", folder / "project.json")
    _, rows = ledger.read_ledger(source / "ledger.jsonl")
    keep = max(index for index, row in enumerate(rows) if row.phase == "baseline_retune") + 1
    lines = (source / "ledger.jsonl").read_bytes().splitlines(keepends=True)
    (folder / "ledger.jsonl").write_bytes(b"".join(lines[:keep + 1]))
    return folder


def _cli_stop(directory: Path, source: Path) -> Record:
    """接續一份沒寫完的；子行程寫完 running 就暫停，用 stop 命令放記號再讓它繼續（要停、回 3），之後再接續跑完。"""
    folder = _cut_copy(directory, source, "root-b")
    with subprocess.Popen(_command("resume", str(folder)), cwd=directory, env=_env(directory), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        with _paused(folder, process):
            stop = _run(directory, _command("stop", str(folder)))
        stdout, stderr = process.communicate(timeout=WAIT_S)
    stopped_status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    marker = (folder / "stop").exists()
    resumed = _run(directory, _command("resume", str(folder)))
    return {"folder": str(folder), "stop": stop, "stopped": {"code": process.returncode, "stderr": stderr},
            "stopped_status": stopped_status, "marker_after_stop": marker, "resumed": resumed,
            "marker_after_resume": (folder / "stop").exists()}


def _start_rejection(directory: Path, name: str) -> Record:
    """拒收：每一個 start 用自己的資料夾根，事後看根底下有沒有被建任何東西。"""
    cases = {"unvalidated": ("unvalidated", PRESET_TEXT, "scheme.json"),
             "zero_identity": ("zero-identity", PRESET_TEXT, "scheme.json"),
             "outside_family": ("good", PRESET_TEXT.replace("glass_wool_24k_50mm", OUTSIDE_FAMILY), "scheme.json"),
             "five_presets": ("good", ",".join(PRESET_TEXT.split(",")[:-1]), "scheme.json"),
             "moved_speaker": ("good", PRESET_TEXT, "scheme-moved-speaker.json"),
             "moved_seat": ("good", PRESET_TEXT, "scheme-moved-seat.json"),
             "other_density": ("other-density", PRESET_TEXT, "scheme.json"),
             "other_speed": ("other-speed", PRESET_TEXT, "scheme.json"),
             **{name: (name, PRESET_TEXT, "scheme.json") for name in AXIS_SURROGATES}}
    if name == "under_searches":
        searches = directory / "data" / "searches"
        return {**_run(directory, _start_args(directory, searches)), "created": searches.exists()}
    surrogate, presets, scheme = cases[name]
    root = directory / f"reject-{name}"
    record = _run(directory, _start_args(directory, root, surrogate, presets, scheme))
    record["root_entries"] = sorted(path.name for path in root.iterdir()) if root.exists() else []
    return record


def _rejections(directory: Path, folder: Path) -> Record:
    """必須在開始那份完成之後，才抄它的資料夾做接續拒收。"""
    records: Record = {}
    tampered = directory / "tampered" / folder.name
    tampered.mkdir(parents=True)
    project = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    project["physics_identity"] = "phys-v1:" + "1" * 64
    (tampered / "project.json").write_text(json.dumps(project), encoding="utf-8")
    records["resume_other_identity"] = {**_run(directory, _command("resume", str(tampered))),
                                        "entries": sorted(path.name for path in tampered.iterdir())}
    records["resume_scheme_changed"] = _scheme_changed(directory, folder)
    for kind, name in (("source_points", MOVED_SPEAKER), ("receiver_points", MOVED_SEAT)):
        records[f"resume_project_{kind}"] = _project_points_tampered(directory, folder, kind, name)
    records["resume_narrow_axis"] = _surrogate_swapped(directory, folder, "narrow-axis")
    return records


def _surrogate_swapped(directory: Path, folder: Path, surrogate: str) -> Record:
    """抄一份跑完的資料夾，project.json 指到另一份替身、連 manifest 與收據摘要一起換成它的（前面幾格核對照樣過），
    再接續：要拒收，資料夾裡每一個檔的位元組都不變、不多出檔。"""
    copied = directory / f"swapped-{surrogate}" / folder.name
    shutil.copytree(folder, copied)
    project = json.loads((copied / "project.json").read_text(encoding="utf-8"))
    npz = (directory / surrogate / "rom.npz").resolve()
    receipt = npz.with_name("rom.validation.json")
    manifest = json.loads(manifest_path(npz).read_text(encoding="utf-8"))
    project.update(surrogate_path=str(npz), receipt_path=str(receipt), receipt_sha256=_sha256(receipt),
                   surrogate_manifest_sha256=manifest["manifest_sha256"])
    (copied / "project.json").write_text(json.dumps(project), encoding="utf-8")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in copied.iterdir()}
    record = _run(directory, _command("resume", str(copied)))
    record["unchanged"] = before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in copied.iterdir()}
    return record


def _project_points_tampered(directory: Path, folder: Path, kind: str, name: str) -> Record:
    """抄一份跑完的資料夾，只改 project.json 記的一支喇叭或一個座位的 x（換成另一個正規 hex），方案檔、替身、
    收據都不動，再接續：要拒收並點名那一支或那一座，帳本與狀態檔一個位元組都不動。"""
    copied = directory / f"tampered-{kind}" / folder.name
    shutil.copytree(folder, copied)
    project = json.loads((copied / "project.json").read_text(encoding="utf-8"))
    x = project[kind][name][0]
    project[kind][name][0] = (float.fromhex(x) + MOVE_M).hex()
    (copied / "project.json").write_text(json.dumps(project), encoding="utf-8")
    before = {item: (copied / item).read_bytes() for item in ("ledger.jsonl", "status.json")}
    record = _run(directory, _command("resume", str(copied)))
    record["unchanged"] = all((copied / item).read_bytes() == data for item, data in before.items())
    return record


def _scheme_changed(directory: Path, folder: Path) -> Record:
    """抄一份跑完的資料夾、方案檔也抄一份給它（project.json 指過去，雜湊仍是開始時量的那個），接著改方案檔的內容
    （同樣的方案、換縮排），再接續：要拒收、帳本與狀態檔都不動。"""
    copied = directory / "scheme-changed" / folder.name
    shutil.copytree(folder, copied)
    scheme = directory / "scheme-changed" / "scheme.json"
    shutil.copyfile(directory / "scheme.json", scheme)
    project = json.loads((copied / "project.json").read_text(encoding="utf-8"))
    project["scheme_path"] = str(scheme)
    (copied / "project.json").write_text(json.dumps(project), encoding="utf-8")
    before = {name: (copied / name).read_bytes() for name in ("ledger.jsonl", "status.json", "result.json")}
    scheme.write_text(json.dumps(json.loads(scheme.read_text(encoding="utf-8")), indent=2), encoding="utf-8")
    record = _run(directory, _command("resume", str(copied)))
    record["unchanged"] = all((copied / name).read_bytes() == data for name, data in before.items())
    return record


def _other_settings(directory: Path) -> Record:
    """同一個根（開始那一份的根）、同一份方案與替身，換一份設定再開始一次：要另開一個資料夾跑完。"""
    root = directory / "root-a"
    record = _run(directory, _start_args(directory, root, settings="settings-other.toml"))
    record["folders"] = sorted(path.name for path in root.iterdir())
    return record


def _max_steps_flag(directory: Path) -> Record:
    """開跑時另外指定步數（#1073）：settings.toml 加 --max-steps 成 settings-other.toml 那一份，在另一個根跑完；
    截掉第 0 輪再接續要跑完（接續照 project.json 記的步數套回去），把那一格拿掉再接續要拒收、資料夾不動。"""
    root = directory / "root-flag"
    started = _run(directory, [*_start_args(directory, root), "--max-steps", str(OTHER_STEPS["descent.max_steps"])])
    folders = sorted(path for path in root.iterdir() if path.is_dir())
    record: Record = {"started": started, "folders": [path.name for path in folders]}
    if started["code"] != 0 or len(folders) != 1:
        return record
    folder = folders[0]
    resumed, dropped = _cut_copy(directory, folder, "root-flag-cut"), _cut_copy(directory, folder, "root-flag-dropped")
    project = json.loads((dropped / "project.json").read_text(encoding="utf-8"))
    project.pop("max_steps", None)
    (dropped / "project.json").write_text(json.dumps(project), encoding="utf-8")
    before = {path.name: _sha256(path) for path in dropped.iterdir()}
    record.update(folder=str(folder), report=_run(directory, _command("report", "--folder", str(folder))),
                  resumed_folder=str(resumed), resumed=_run(directory, _command("resume", str(resumed))),
                  refused=_run(directory, _command("resume", str(dropped))))
    record["refused_unchanged"] = before == {path.name: _sha256(path) for path in dropped.iterdir()}
    return record


def _interrupted(directory: Path) -> Record:
    """開始一次，子行程寫完 running 就對命令列（父行程）送 SIGTERM：子行程要被收掉、寫 interrupted，命令列回 3。"""
    root = directory / "root-term"
    with subprocess.Popen(_start_args(directory, root), cwd=directory, env=_env(directory), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        folder = _first_folder(root, process)
        child = _running_pid(folder, process)
        began = time.monotonic()
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=WAIT_S)
        seconds = time.monotonic() - began
    try:
        os.killpg(child, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    return {"code": process.returncode, "stderr": stderr, "seconds": seconds, "child_alive": alive,
            "status": json.loads((folder / "status.json").read_text(encoding="utf-8"))}


def _probe(directory: Path, folder: Path, part: str) -> Record:
    work = directory / "probe"
    if part in ("reference", "all"):
        (work / "source").mkdir(parents=True)
        shutil.copyfile(folder / "project.json", work / "source" / "project.json")
    process = subprocess.run([sys.executable, str(PROBE), str(work), str(directory / "air-other" / "rom.npz"), part], cwd=work, env=_env(work),
                             capture_output=True, text=True, timeout=WAIT_S * 3)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(Record, json.loads((work / f"output-{part}.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
         ) -> tuple[Path, Record]:
    """各段同步取鎖共用，資料夾依賴先做完才起下一個行程；沒有背景等檔。"""
    from tests.engine._material_gradient_cli_split import split_runs

    return split_runs(request, tmp_path_factory, worker_id)


def _section(runs: tuple[Path, Record], *keys: str) -> Record:
    node: object = runs[1]
    for key in keys:
        node = cast(Record, node)[key]
    return cast(Record, node)


def _folder(runs: tuple[Path, Record]) -> Path:
    return Path(str(_section(runs, "concurrent")["folder"]))


def _rows(folder: Path) -> tuple[ledger.MaterialGradientRow, ...]:
    return ledger.read_ledger(folder / "ledger.jsonl")[1]


def _json(path: Path) -> Record:
    return cast(Record, json.loads(path.read_text(encoding="utf-8")))


def _without_hash(result: Record) -> Record:
    """結果檔扣掉兩本帳的整份雜湊（帳本的秒數每跑不同，雜湊跟著不同）。"""
    return {key: value for key, value in result.items() if key not in ("ledger_sha256", "template_ledger_sha256")}


def _kept_materials(runs: tuple[Path, Record]) -> dict[str, SurfaceMaterial]:
    """控制組：考卷方案舊欄的六面固定阻抗，自己換成 ConstantImpedance（不走產品的 scene_wall_materials）。"""
    fixed = load_scheme(runs[0] / "scheme.json").scene.impedance_pa_s_per_m_by_wall
    assert fixed is not None
    return {name: ConstantImpedance(impedance_pa_s_m=value) for name, value in fixed.items()}


def _settings_path(runs: tuple[Path, Record]) -> Path:
    return runs[0] / "settings.toml"


# ---- 開始：整條鏈、帳本、狀態、結果、報告 ----





def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()






# ---- 停止記號與接續 ----

def _probe_section(runs: tuple[Path, Record], key: str) -> object:
    return _section(runs, "probe")[key]


def _probe_folder(runs: tuple[Path, Record], name: object) -> Path:
    return runs[0] / "probe" / str(name)












# ---- 拒收與資料夾鎖 ----

def _rejection(runs: tuple[Path, Record], name: str) -> Record:
    return _section(runs, "rejections", name)






















class _FakeProcess:
    pid = 2 ** 22 + 7

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_interrupt_between_spawn_and_wait_terminates_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """審查 B2：子行程建好（Popen 回來、還沒進等待）之後的第一個沒擋訊號的行就來中斷，也要收子行程。

    Popen 換成假行程，``sys.settrace`` 在 ``_spawn_and_wait`` 裡、``process`` 已拿到、而且這一刻沒有擋住 SIGINT 的
    第一行注入 KeyboardInterrupt（擋著的時候真的訊號送不進來，所以那幾行不注入）：要叫到收子行程、回 3，
    事後這個行程的訊號遮罩跟叫之前一樣。"""
    fake = _FakeProcess()
    terminated: list[object] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake)
    monkeypatch.setattr(cli, "_terminate", terminated.append)
    target = cli._spawn_and_wait.__code__
    injected: list[int] = []

    def local(frame: FrameType, event: str, arg: object) -> TraceFunction | None:
        unblocked = signal.SIGINT not in signal.pthread_sigmask(signal.SIG_BLOCK, ())
        if event == "line" and not injected and frame.f_locals.get("process") is fake and unblocked:
            injected.append(frame.f_lineno)
            raise KeyboardInterrupt
        return local

    def tracer(frame: FrameType, event: str, arg: object) -> TraceFunction | None:
        return local if frame.f_code is target else None

    before = signal.pthread_sigmask(signal.SIG_BLOCK, ())
    code: int | None = None
    sys.settrace(tracer)
    try:
        code = cli._spawn_and_wait(tmp_path, "start", 0)
    except KeyboardInterrupt:
        pass  # 中斷漏出 _spawn_and_wait＝沒收子行程；留給下面的斷言判紅，不讓 pytest 當成整場被中斷
    finally:
        sys.settrace(None)
    assert injected, "沒有注入到（process 拿到之後沒有下一行）"
    assert terminated == [fake], f"第 {injected} 行中斷，子行程沒被收"
    assert code == cli.EXIT_STOPPED
    assert signal.pthread_sigmask(signal.SIG_BLOCK, ()) == before
