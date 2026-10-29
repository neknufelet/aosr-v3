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
from aosr.physics.fem_reduced_check import validate_reduced
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, save_validation_receipt
from aosr.physics.fem_reduced_store import manifest_path, save_reduced
from aosr.reporting import physics_identity as identity_module
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_low_frequency_contract import StageOneReasonCode
from aosr.runtime import child_process_env, configure_jax
from tests.engine import _fem_shared_cases as shared_cases
from tests.engine._precision_contracts import contract_value
from tests.engine._scheme_cache import shared_json
from tests.engine.test_fem_reduced import DENSITY, SOUND_SPEED, model as model, operators as operators
from tests.engine.test_material_gradient import CASES, DESCENT_CASE, PURPOSE, _chain_model, _presets, _receivers
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
# 考卷只為了快把兩個步數上限調小（正式 30 與 5）；其他設定照正式設定檔。
NARROWED_STEPS = {"max_steps": 3, "steps": 2}
# 同一個根、另一份設定（只差下降步數上限）：資料夾名要跟著不同。
OTHER_STEPS = {"max_steps": 2, "steps": 2}
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


def _surrogates(directory: Path, request: pytest.FixtureRequest) -> None:
    """替身：驗過的（現在的物理身分＋收據）、沒有收據的、物理身分全零但有收據的，與同網格同座標、另一組空氣建的
    三份（完整模型算子的空氣跟著換、各自做留出驗證寫收據）。"""
    base = _chain_model(cast(ReducedModel, request.getfixturevalue("model")))
    chain = dataclasses.replace(base, physics_identity=_current_identity())
    report = validate_reduced(chain, _chain_operators(cast(FullOperators, request.getfixturevalue("operators")), chain),
                              _holdouts())
    rulers = HoldoutRulers(frequency=Ruler(FREQUENCY_NAME, contract_value(FREQUENCY_NAME)),
                           band=Ruler(BAND_NAME, contract_value(BAND_NAME)))
    for name, surrogate, receipt in (("good", chain, True), ("unvalidated", chain, False), ("zero-identity", base, True)):
        (directory / name).mkdir()
        save_reduced(surrogate, directory / name / "rom.npz")
        if receipt:
            save_validation_receipt(directory / name / "rom.npz", report, rulers=rulers, seconds=1.0)
    full = _chain_operators(cast(FullOperators, request.getfixturevalue("operators")), chain)
    for name, (density, speed) in AIR_SURROGATES.items():
        other = dataclasses.replace(chain, density_kg_m3=density, sound_speed_m_s=speed)
        operators = dataclasses.replace(full, density_kg_m3=density, sound_speed_m_s=speed)
        (directory / name).mkdir()
        save_reduced(other, directory / name / "rom.npz")
        save_validation_receipt(directory / name / "rom.npz", validate_reduced(other, operators, _holdouts()),
                                rulers=rulers, seconds=1.0)
    for name, (axis, _) in AXIS_SURROGATES.items():
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
    scene = base.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.SEARCH})
    scheme = base.model_copy(update={"receiver_set": _receivers(), "purpose": PURPOSE, "scene": scene})
    speakers = {**scheme.speakers, MOVED_SPEAKER: _moved(scheme.speakers[MOVED_SPEAKER])}
    return {"scheme.json": scheme, "scheme-moved-speaker.json": scheme.model_copy(update={"speakers": speakers}),
            "scheme-moved-seat.json": scheme.model_copy(update={"receiver_set": _moved_seat(scheme.receiver_set)})}


def _settings_text(steps: dict[str, int]) -> str:
    text = config_path("material_gradient.toml").read_text(encoding="utf-8")
    for key, value in steps.items():
        text = re.sub(rf"(?m)^{key} = \d+$", f"{key} = {value}", text)
    return text


def _inputs(directory: Path) -> None:
    for name, scheme in _schemes().items():
        (directory / name).write_text(scheme.model_dump_json(), encoding="utf-8")
    for name, steps in (("settings.toml", NARROWED_STEPS), ("settings-other.toml", OTHER_STEPS)):
        (directory / name).write_text(_settings_text(steps), encoding="utf-8")
        narrowed = load_material_gradient_settings(directory / name)
        assert {"max_steps": narrowed.descent.max_steps, "steps": narrowed.retune.steps} == steps


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


def _rejections(directory: Path, folder: Path) -> Record:
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
    records: Record = {}
    for name, (surrogate, presets, scheme) in cases.items():
        root = directory / f"reject-{name}"
        record = _run(directory, _start_args(directory, root, surrogate, presets, scheme))
        record["root_entries"] = sorted(path.name for path in root.iterdir()) if root.exists() else []
        records[name] = record
    searches = directory / "data" / "searches"
    records["under_searches"] = {**_run(directory, _start_args(directory, searches)), "created": searches.exists()}
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


def _probe(directory: Path, folder: Path) -> Record:
    work = directory / "probe"
    (work / "source").mkdir(parents=True)
    shutil.copyfile(folder / "project.json", work / "source" / "project.json")
    process = subprocess.run([sys.executable, str(PROBE), str(work), str(directory / "air-other" / "rom.npz")], cwd=work, env=_env(work),
                             capture_output=True, text=True, timeout=WAIT_S * 3)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(Record, json.loads((work / "output.json").read_text(encoding="utf-8")))


def _compute(directory: Path, request: pytest.FixtureRequest) -> str:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    _surrogates(directory, request)
    _inputs(directory)
    output: Record = {"concurrent": _concurrent_start(directory)}
    folder = Path(str(cast(Record, output["concurrent"])["folder"]))
    output["report"] = _run(directory, _command("report", "--folder", str(folder)))
    output["other_settings"] = _other_settings(directory)
    output["interrupted"] = _interrupted(directory)
    output["cli_stop"] = _cli_stop(directory, folder)
    output["rejections"] = _rejections(directory, folder)
    output["probe"] = _probe(directory, folder)
    text = json.dumps(output, ensure_ascii=False)
    (directory / "fixture.json").write_text(text, encoding="utf-8")
    return text


@pytest.fixture(scope="module")
def runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
         ) -> tuple[Path, Record]:
    """一場 pytest 只建一次替身、跑一次命令列與探針，平行的工人共用；讀的人只讀，改檔先抄到自己的 tmp_path。"""
    directory = (tmp_path_factory.mktemp("material-gradient-cli") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "material-gradient-cli-shared")
    text = shared_json(tmp_path_factory, worker_id, "material-gradient-cli", lambda: _compute(directory, request))
    return directory, cast(Record, json.loads(text))


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
    """結果檔扣掉帳本整份雜湊（帳本的秒數每跑不同，雜湊跟著不同）。"""
    return {key: value for key, value in result.items() if key != "ledger_sha256"}


def _kept_materials(runs: tuple[Path, Record]) -> dict[str, SurfaceMaterial]:
    """控制組：考卷方案舊欄的六面固定阻抗，自己換成 ConstantImpedance（不走產品的 scene_wall_materials）。"""
    fixed = load_scheme(runs[0] / "scheme.json").scene.impedance_pa_s_per_m_by_wall
    assert fixed is not None
    return {name: ConstantImpedance(impedance_pa_s_m=value) for name, value in fixed.items()}


def _settings_path(runs: tuple[Path, Record]) -> Path:
    return runs[0] / "settings.toml"


# ---- 開始：整條鏈、帳本、狀態、結果、報告 ----

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
    assert project["physics_identity"] == manifest["physics_identity"] == _current_identity()
    assert project["mesh_hash"] == manifest["mesh_hash"]
    assert project["presets"] == list(PRESETS) and project["purpose"] == PURPOSE
    assert project["gradient_id"] == folder.name
    scheme_document = Scheme.model_validate_json(scheme.read_text(encoding="utf-8"))
    assert project["source_points"] == {name: [float(value).hex() for value in point.as_tuple()]
                                        for name, point in scheme_document.speakers.items()}
    assert project["receiver_points"] == {point.receiver_id: [float(value).hex() for value in point.position_m]
                                          for point in scheme_document.receiver_set.points}
    receipt_model = _json(receipt)["reduced_model"]
    assert isinstance(receipt_model, dict)
    assert (receipt_model["source_points"], receipt_model["receiver_points"]) == (
        project["source_points"], project["receiver_points"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert (result["value"], result["stop_reason"], result["evaluations"]) == (
        last.value, last.stop_reason, last.decision.total_evaluations)
    assert result["components"] == {"own_line_ripple_rms_db": last.components[0],
                                    "seat_robustness_rms_db": last.components[1]}
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
            assert any(line.startswith(f"第 {row.round} 輪：交出的值 {row.value:.6f}") for line in lines)


# ---- 停止記號與接續 ----

def _probe_section(runs: tuple[Path, Record], key: str) -> object:
    return _section(runs, "probe")[key]


def _probe_folder(runs: tuple[Path, Record], name: object) -> Path:
    return runs[0] / "probe" / str(name)


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


# ---- 拒收與資料夾鎖 ----

def _rejection(runs: tuple[Path, Record], name: str) -> Record:
    return _section(runs, "rejections", name)


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
