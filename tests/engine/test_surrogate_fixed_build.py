"""#1082 段 2-8：建替身在裝進格子之前就跑開跑那一道固定材料核對、沒過就補訓練、補不過就不裝；網頁不對開跑會拒的替身說
「可以直接開搜尋」；開跑拒收的句子分得出「訓練過但不夠準」與「沒學過」。

起因（2-6b）：room109demo-part2b 的替身自己的留出驗證過了、裝進格子、預覽說可以開搜尋，開跑前那一道核對跑 8 分鐘後拒收
（實測玻璃棉逐頻 0.388 dB，尺 0.224），而且第一個沒過就停、另一個候選沒核到。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from aosr.gradient import fixed_candidates
from aosr.gradient.surrogate_fixed_checks import (
    NO_RECORD_TEXT,
    OTHER_CONFIGURATION_TEXT,
    FixedCandidateInaccurate,
    read_record,
    write_record,
)
from aosr.gradient.surrogate_coverage import FixedTarget
from aosr.gui.gradient_jobs import GradientJobs, NeedsSurrogate, plan_configuration, plan_fixed
from aosr.gui.gradient_plan import GradientRequest, SearchPlan
from aosr.reporting.scheme import Scheme
from aosr.gui.surrogate_slots import surrogate_path
from aosr.materials.search_library import MaterialsRefused
from aosr.physics.fem_reduced_build import STOP_BUDGET, STOP_ROUND_LIMIT, ReducedModelBuilder, build_reduced_model
from aosr.physics.fem_reduced import impedance_rows, input_hash
from aosr.physics import fem_reduced_check as check
from aosr.runtime import child_process_env, configure_jax
from tests.engine import test_fem_reduced_zones as zones
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._scheme_cache import bitwise_equal
from tests.engine._surrogate_coverage import fixed_checked_on, trained_on
from tests.engine._zone_rom_case import AXIS, SETTINGS

PROBE = Path(__file__).with_name("_surrogate_fixed_extension_probe.py")
Record = dict[str, object]


# ---- 真建：小房間、真的完整解，候選自己的訓練組拿掉 ----

def _probe(tmp_path: Path, mode: str) -> Record:
    env = forward_cpu_pins(child_process_env(threads=1))
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    env["TMPDIR"] = str(tmp_path)
    done = subprocess.run([sys.executable, str(PROBE), str(tmp_path), mode], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr
    return cast(Record, json.loads((tmp_path / "probe.json").read_text(encoding="utf-8")))


def test_build_checks_the_start_configuration_and_never_installs_what_the_start_would_refuse(tmp_path: Path) -> None:
    """抓建替身沒跑開跑那一道核對就裝進格子（2-6b：裝了、預覽說可以開、開跑 8 分鐘後才拒）、補訓練把核對那一組本身
    放進訓練、或補不進點還一直空轉：候選沒過 → 補訓練（跟核對那一組相同的剔除）→ 補的樣本殘差已在停止值內就停、不裝、照實講。"""
    found = _probe(tmp_path, "stall")
    judged = cast(list[dict[str, Record]], found["judged"])
    assert judged and judged[0]["fixed:panel_absorbing"]["passed"] is False
    generated, added = cast(list[int], found["generated"]), cast(int, found["added"])
    assert sum(generated) > added >= 1, (generated, added)  # 有跟核對那一組相同的被剔除，剩下的才進訓練
    assert found["added_equal_to_check"] == [False] * added
    result = cast(Record, found["result"])
    assert found["exit"] == 4 and result["verdict"] == "not_validated" and found["slot"] is False
    message = str(result["message"])
    assert message.startswith("開跑前那一道固定材料核對沒過：") and "panel_absorbing" in message
    assert "再補同類樣本也沒用" in message and "沒有放進格子" in message and "完整解合計 0 次" not in message


def test_build_that_extends_until_the_check_passes_installs_and_the_preview_says_ready(tmp_path: Path) -> None:
    """抓補訓練過了的那條路沒走通：候選先沒過 → 補訓練真的解了候選那幾組 → 重核過了才做留出驗證、裝進格子；
    訓練說明記了補的組、格子留了這一組起跑配置的紀錄（候選過了）、預覽說可以直接開搜尋、裝好的那句講補訓練之後過了。"""
    found = _probe(tmp_path, "pass")
    judged = cast(list[dict[str, Record]], found["judged"])
    assert judged[0]["fixed:panel_absorbing"]["passed"] is False and judged[-1]["fixed:panel_absorbing"]["passed"] is True
    assert all(item["fixed:panel_absorbing"]["passed"] is False for item in judged[:-1])  # 過了就停、不再補
    # 每補一輪就核、過了就停：停止殘差收到 1e-9（池裡殘差到不了），一路補到停止值或池用完才核就會白解
    # （2-6b part2b 的形狀：候選第 4 輪就過了，池裡殘差卡在 0.0015）。這間小房間補一輪就過。
    before, = cast(list[dict[str, int]], found["before"])
    solved = cast(int, cast(Record, found["result"])["full_solves"]) - before["full_solves"]
    assert 0 < solved <= before["batch"] * (len(judged) - 1), (solved, before, len(judged))
    added = cast(int, found["added"])
    assert added >= 1 and found["added_equal_to_check"] == [False] * added
    result = cast(Record, found["result"])
    assert found["exit"] == 0 and found["slot"] is True, result
    assert "開跑前那一道固定材料核對先沒過，補訓練" in str(result["message"]) and "之後過了" in str(result["message"])
    labels = cast(list[str], found["training_labels"])
    assert sum(label.startswith("補訓練第 ") for label in labels) == added, labels
    record = cast(Record, found["record"])
    assert [(row["target"], row["passed"]) for row in cast(list[Record], record["checks"])] == [("fixed:panel_absorbing", True)]
    preview = cast(Record, found["preview"])
    assert preview["found"] is True and preview["message"] is None
    # #1128：每一次核（建模完一次、每補一輪一次）的數字都留在 fixed_check_rounds.jsonl，跟判決逐筆相同；進度有接上。
    rounds = cast(list[Record], found["rounds"])
    assert [(row["check"], row["target"], row["frequency_db"], row["passed"]) for row in rounds] == [
        (index + 1, target, item["frequency_db"], item["passed"])
        for index, check in enumerate(judged) for target, item in sorted(check.items())]
    assert [row["full_solves"] for row in rounds][-1] == cast(Record, found["result"])["full_solves"]
    assert all(item["progress_given"] for check in judged for item in check.values())


def test_build_without_budget_left_refuses_and_says_a_bigger_budget_may_pass(tmp_path: Path) -> None:
    """抓預算用完還照裝、或沒講接下來能怎麼辦：建完預算剛好用完，候選沒過就一組都不補、不裝，講加大完整解上限。"""
    found = _probe(tmp_path, "cap")
    result = cast(Record, found["result"])
    assert found["exit"] == 4 and result["verdict"] == "not_validated" and found["slot"] is False
    assert found["added"] == 0
    message = str(result["message"])
    assert "補訓練到完整解上限" in message and "加大完整解上限（max_full_solves）重建可能會過" in message


# ---- 補訓練：接著同一份快照、只從新的那批抽 ----

def test_extend_continues_the_same_snapshots_on_the_new_sets_only() -> None:
    """抓補訓練從頭重建、從舊的那批抽點、解次不合計算預算、或替身身分沒換：先用前四組建，再交後四組補；
    補的點全在後四組、解次合計、輸入雜湊等於八組一起的、補完對後四組準多了；沒補過的 build_reduced_model 逐位照舊。"""
    operators, materials = zones._zoned_operators(), zones._zoned_materials(1, 8)
    settings = SETTINGS.model_copy(update={"max_full_solves": 400})
    builder = ReducedModelBuilder(operators, AXIS, materials[:4], settings=settings, dense_threads=1)
    base = builder.build()
    plain = build_reduced_model(operators, AXIS, materials[:4], settings=settings, dense_threads=1)
    assert bitwise_equal(base.matrices, plain.matrices) and base.input_hash == plain.input_hash
    assert (base.ledger.rounds, base.ledger.full_solves, base.ledger.stop_reason) == (
        plain.ledger.rounds, plain.ledger.full_solves, plain.ledger.stop_reason)
    assert builder._build is not None
    seen = set(builder._build.seen)
    before = check.validate_reduced(base, operators, materials[4:], dense_threads=1)
    extended = builder.extend(materials[4:])
    new_points = builder._build.seen - seen
    assert new_points and all(material >= 4 for material, _ in new_points)
    assert extended.ledger.full_solves == base.ledger.full_solves + len(new_points) <= settings.max_full_solves
    assert extended.ledger.rounds[:len(base.ledger.rounds)] == base.ledger.rounds
    axis = tuple(float(value) for value in AXIS)
    assert extended.input_hash == input_hash(operators, axis, impedance_rows(materials, axis, patch_keys=operators.patch_keys),
                                             settings) != base.input_hash
    after = check.revalidate_reduced(extended, operators, materials[4:], before.full_pressure,
                                     full_solves=before.full_solves, dense_threads=1)
    assert max(after.frequency_db_by_seat.values()) < max(before.frequency_db_by_seat.values())
    # 一輪一輪補（每輪交回來一次、不給新材料就同一池接著補）跟一次補到底逐位相同；補到一半撞上限那一種也一樣
    # （上限那一輪照舊再量一次殘差才判停）。
    # 上限收到補一輪補不完（這間小房間不設上限一輪補 16 點就把池用完）。
    capped = settings.model_copy(update={"max_full_solves": base.ledger.full_solves + 8})
    for limit in (settings, capped):
        whole = ReducedModelBuilder(operators, AXIS, materials[:4], settings=limit, dense_threads=1)
        whole.build()
        expected = whole.extend(materials[4:])
        stepwise = ReducedModelBuilder(operators, AXIS, materials[:4], settings=limit, dense_threads=1)
        stepwise.build()
        calls, step = 1, stepwise.extend(materials[4:], max_rounds=1)
        while stepwise.stop_reason == STOP_ROUND_LIMIT:
            calls, step = calls + 1, stepwise.extend(max_rounds=1)
        assert (calls > 1) is (limit is settings)  # 沒撞上限時真的分了幾次交回來
        assert bitwise_equal(step.matrices, expected.matrices) and step.input_hash == expected.input_hash
        assert (step.ledger.rounds, step.ledger.full_solves, step.ledger.stop_reason) == (
            expected.ledger.rounds, expected.ledger.full_solves, expected.ledger.stop_reason)
    # 真的撞到上限；停止原因是上限那一輪之後量過殘差才判的（這一例殘差剛好達標），不是沒量就記「預算停止」。
    assert expected.ledger.full_solves == capped.max_full_solves and expected.ledger.stop_reason != STOP_ROUND_LIMIT
    again = check.revalidate_reduced(base, operators, materials[4:], before.full_pressure, full_solves=before.full_solves,
                                     dense_threads=1)
    assert bitwise_equal(again, before)  # 同一份替身重算替身那一側，跟完整驗證逐位相同


def test_extend_with_no_budget_left_solves_nothing_and_says_budget() -> None:
    """抓預算用完還補點：建完剛好用完，補訓練一次都不解、停止原因照實記「預算停止」。"""
    operators, materials = zones._zoned_operators(), zones._zoned_materials(1, 6)
    probe = build_reduced_model(operators, AXIS, materials[:3], settings=SETTINGS, dense_threads=1)
    settings = SETTINGS.model_copy(update={"max_full_solves": probe.ledger.full_solves})
    builder = ReducedModelBuilder(operators, AXIS, materials[:3], settings=settings, dense_threads=1)
    base = builder.build()
    extended = builder.extend(materials[3:])
    assert extended.ledger.full_solves == base.ledger.full_solves and extended.ledger.stop_reason == STOP_BUDGET


# ---- 開跑那一道：全部核完、措辭分訓練過與沒學過 ----

def _fake_gate(monkeypatch: pytest.MonkeyPatch, numbers: dict[str, float]) -> tuple[object, object, tuple[str, ...]]:
    """完整解那一步換成回固定數字的報告（每個候選一個逐頻差，帶能量差 0.001）；回替身、完整模型、兩個候選目標。"""
    from tests.engine.test_material_gradient_allowed import CARPET, MEASURED, _FakeOperators

    calls: list[str] = []
    rows: dict[int, str] = {}

    def impedances(target: str, *args: object, **kwargs: object) -> dict[str, tuple[complex, ...]]:
        row: dict[str, tuple[complex, ...]] = {"x0": (1.0 + 0.0j, 1.0 + 0.0j)}
        rows[id(row)] = target
        return row

    def validate(model: object, operators: object, holdouts: list[dict[str, tuple[complex, ...]]], **kwargs: object) -> object:
        target = rows[id(holdouts[0])]
        calls.append(target)
        value = numbers[target]
        return SimpleNamespace(frequency_db_by_seat={"front": value}, band_db_by_seat={"front": 0.001}, full_solves=3,
                               full_pressure=np.ones((1, 2, 1, 1), dtype=complex),
                               reduced_pressure=np.asarray([[[[1.0]], [[10.0 ** (value / 20.0)]]]], dtype=complex))

    monkeypatch.setattr(check, "validate_reduced", validate)
    monkeypatch.setattr(fixed_candidates, "candidate_designs", lambda designs, target, *args: target)
    monkeypatch.setattr(fixed_candidates, "design_impedances", impedances)
    model = SimpleNamespace(frequencies_hz=(50.0, 293.441), source_names=("s",), receiver_names=("front",))
    operators = _FakeOperators(sources={"s": 0}, receivers={"front": 0})
    monkeypatch.setattr(fixed_candidates, "_calls", calls, raising=False)
    return model, operators, ("fixed:" + CARPET, "fixed:" + MEASURED)


def test_start_checks_every_candidate_and_says_trained_but_not_accurate(monkeypatch: pytest.MonkeyPatch) -> None:
    """抓開跑那一道第一個沒過就停（2-6b：另一個候選從沒核到）、或材料明明在訓練說明裡還說「沒有學過這種材料」：
    兩個都沒過要兩個都點名；訓練過的說在哪個頻率差多少；沒記的才說沒學過。"""
    from tests.engine.test_material_gradient_allowed import _table

    model, operators, (carpet, measured) = _fake_gate(monkeypatch, {"fixed:carpet_on_concrete": 0.3,
                                                                     "fixed:panel_measured": 0.388243})
    allowed = {"x0": ("none", carpet, measured)}
    check_candidates: Callable[..., object] = fixed_candidates.check_fixed_candidates  # 替身與完整模型是樁
    with pytest.raises(FixedCandidateInaccurate) as refused:
        check_candidates(model, operators, (), allowed, _table(), density_kg_m3=1.2, sound_speed_m_s=343.0,
                         trained=(measured,))
    text = str(refused.value)
    assert getattr(fixed_candidates, "_calls") == [carpet, measured]
    assert ("替身對固定材料 panel_measured 沒過留出驗證：替身訓練過這種材料，但開跑這一組配置在 293.441 Hz 不夠準："
            "逐頻最大差 0.388243 dB") in text
    assert "替身對固定材料 carpet_on_concrete 沒過留出驗證：逐頻最大差 0.3 dB" in text and "這份替身沒有學過這種材料" in text
    assert "panel_measured" not in text.split("沒有學過")[0].split("carpet_on_concrete")[-1]
    assert isinstance(refused.value, MaterialsRefused)


# ---- 網頁：預覽與開跑對核對紀錄 ----

def _web(tmp_path: Path) -> tuple[Path, GradientJobs, tuple[Scheme, SearchPlan], tuple[FixedTarget, ...]]:
    """網頁一份：x0 一區准用玻璃棉與一個構造型固定材料，替身樁旁已記訓練過（2-3c）；回資料夾、工作管理器、計畫、候選。"""
    from aosr.gui.material_editor import user_library_path
    from aosr.gui.zone_design import DesignSettings, ZoneDesign, write_design
    from aosr.materials import library
    from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _data, _jobs, _scheme
    from tests.engine.test_material_gradient_allowed import BUILT, _built

    data_dir = _data(tmp_path)
    zones_by_wall = {"x0": {"r0_c0": ZoneDesign(mode="search", allowed=("glass_wool_48k", BUILT)),
                            "r1_c1": ZoneDesign(mode="search")}}
    write_design(data_dir, DesignSettings(scheme_id=NAME, zones=zones_by_wall), _scheme())
    user_library_path(data_dir).write_text(library.dump_user_library((_built(),)), encoding="utf-8")
    jobs = _jobs(data_dir, FakeLauncher())
    _, scheme, design = jobs.inputs(NAME)
    plan, materials, entries = jobs.plan(scheme, design, GradientRequest(max_steps=5))
    trained_on(surrogate_path(data_dir, scheme), materials, plan_fixed(plan, materials, entries))
    return data_dir, jobs, (scheme, plan), plan_fixed(plan, materials, entries)


def test_preview_never_says_ready_for_a_surrogate_not_checked_on_this_start_configuration(tmp_path: Path) -> None:
    """抓預覽對開跑會拒的替身說「可以直接開搜尋」（2-6b）：訓練過但沒核過這一組起跑配置（舊替身沒有紀錄）、核的是別組配置
    （換了裝法）、紀錄是別份替身的，都要說要重建、開跑也不開服務；同一組配置核過才說可以。"""
    from tests.engine.test_gui_gradient_jobs import NAME

    data_dir, jobs, (scheme, plan), fixed = _web(tmp_path)
    surrogate = surrogate_path(data_dir, scheme)
    request = GradientRequest(max_steps=5)

    def preview(given: GradientRequest = request) -> Record:
        return cast(Record, jobs.preview(NAME, given)["surrogate"])

    legacy = preview()
    assert legacy["found"] is False and legacy["missing"] == [] and legacy["message"] == NO_RECORD_TEXT
    with pytest.raises(NeedsSurrogate, match="沒有用開跑那一組起跑配置核過"):
        jobs.start(NAME, request)
    path = fixed_checked_on(surrogate, plan_configuration(plan, scheme), fixed)
    assert preview() == {**legacy, "found": True, "message": None}
    record = read_record(path)
    assert record is not None
    row = record.checks[0]
    for changed in (row.model_copy(update={"passed": False}), row.model_copy(update={"material_sha256": "0" * 64})):
        path.unlink()
        write_record(path, record.model_copy(update={"checks": (changed, *record.checks[1:])}))
        assert preview()["message"] == f"固定材料 {row.target} 沒有在這一組起跑配置核過；要重建（重建時會先核，不夠準就補訓練）"
    path.unlink()
    write_record(path, record)
    assert preview()["found"] is True
    rigid = preview(GradientRequest(max_steps=5, mount="rigid"))
    assert rigid["found"] is False and rigid["message"] == OTHER_CONFIGURATION_TEXT
    manifest = surrogate.with_name(surrogate.stem + ".manifest.json")
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert preview()["message"] == NO_RECORD_TEXT


def test_web_labels_a_start_refused_for_an_inaccurate_candidate_as_needing_a_rebuild(tmp_path: Path) -> None:
    """抓開跑那一道說候選不夠準時網頁只寫「搜尋失敗」、沒給重建的路（2-6b 的工作頁就是這樣）：是替身核不過的一種，
    狀態要是「替身核不過、要重建」，字講候選不夠準。"""
    from aosr.gradient.surrogate_fixed_checks import INACCURATE_STATE_TEXT
    from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _data, _finish, _jobs

    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest())
    line = ("梯度搜尋開始失敗：FixedCandidateInaccurate：替身對固定材料 panel_measured 沒過留出驗證：替身訓練過這種材料，"
            "但開跑這一組配置在 293.441 Hz 不夠準")
    (data_dir / "gradient" / str(view["job_id"]) / "steps" / "0.log").write_text(line + "\n", encoding="utf-8")
    _finish(data_dir, launcher, view, 1)
    view = jobs.get(str(view["job_id"]))
    assert view["state"] == "surrogate_refused" and view["state_text"] == INACCURATE_STATE_TEXT
