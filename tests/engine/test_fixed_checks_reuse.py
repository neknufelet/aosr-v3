"""#1142（老闆 2026-11-13 拍 A）：開跑那一道固定材料核對沿用建替身時核過的數字，只在身分逐欄相同時。

身分：替身（manifest 整份檔案的摘要）、起跑配置（正規寫法的摘要）、頻率軸（逐位）、兩把尺（登記簿現值）、完整模型指紋、
算核對的程式身分、每個候選的兩種材料摘要、准用它的片、核對配置的逐片逐頻阻抗摘要；紀錄要是 v2（舊格式沒有這些欄位）。任何一欄不同、舊格式、
紀錄讀不懂、或替身是接續（carry-over）沿用的，都照舊重核。真建替身、真開跑兩條路逐位相同的那一題在
``test_surrogate_fixed_build.py``（``pass`` 模式）。這裡專挑「不該沿用卻沿用」：每一欄單獨換掉都要重核。
"""
from __future__ import annotations

import io
import json
import math
from collections.abc import Callable
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, fixed_candidates
from aosr.physics import fem_reduced
from aosr.gradient.fixed_candidates import (
    FixedCandidateCheck, candidate_material_digest, candidate_surfaces, holdout_ruler_values,
)
from aosr.gradient.fixed_surfaces import fixed_targets_of
from aosr.gradient.surrogate_coverage import fixed_targets
from aosr.gradient.surrogate_fixed_checks import (
    FIXED_CHECKS_FILE, FIXED_CHECKS_VERSION, FIXED_CHECKS_VERSION_V2, CandidateIdentity, FixedCheckRow, FixedChecksRecord,
    axis_hex, configuration_document, configuration_sha256, impedances_sha256, manifest_digest, read_record, reused_rows,
    unchecked_reason, write_record,
)
from aosr.physics.fem_reduced_store import manifest_path
from tests.engine.test_material_gradient_allowed import CARPET, MEASURED, _table

if TYPE_CHECKING:
    from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
    from aosr.physics.fem_reduced_check import ValidationReport

CARPET_TARGET, MEASURED_TARGET = "fixed:" + CARPET, "fixed:" + MEASURED
ALLOWED: dict[str, tuple[str, ...]] = {"x0": ("none", CARPET_TARGET, MEASURED_TARGET), "y0": (MEASURED_TARGET,)}
AXIS = (50.0, 120.0, 293.441)
CONFIGURATION = configuration_document(presets=("glass_wool_48k_50mm_gap50", None), mount="on_wall",
                                       fixed_surfaces={"floor": "scheme-material"}, max_depths={"x0": 0.3},
                                       allowed=ALLOWED, scene_json='{"scene":1}')
Rows = dict[Wall | tuple[Wall, str], tuple[complex, ...] | None]
OPERATORS = "4" * 64
# 家族設定收窄總深度上限（樣板片的下降起點被投影進去、阻抗跟著變），初值跟著收進新上限內：設定檔照樣讀得進來。
NARROWED_DEPTH = (("max_total_depth_m = 0.300\n", "max_total_depth_m = 0.080\n"),
                  ("initial_cavity_m = 0.050\n", "initial_cavity_m = 0.030\n"))
CODE = "fixcheck-v1:" + "5" * 64


def _digest(rows: Rows) -> str:
    return impedances_sha256(cast("FrequencyBoundaryImpedances", rows))


def _impedances(target: str) -> Rows:
    """每個候選一組假的核對配置（值照目標算，兩個候選不同）；開跑與紀錄兩邊同一份。"""
    scale = 1.0 + len(target)
    return {Wall.X0: tuple(complex(scale * value, -value) for value in AXIS), (Wall.Y0, "z1"): None}


def _candidates() -> tuple[CandidateIdentity, ...]:
    table = _table()
    contents = {item.target: item.coverage().material_sha256 for item in fixed_targets(ALLOWED, table)}
    return tuple(CandidateIdentity(target=target, material_sha256=contents[target],
                                   material_digest=candidate_material_digest(target, table),
                                   surfaces=candidate_surfaces(target, ALLOWED),
                                   impedances_sha256=_digest(_impedances(target)))
                 for target in fixed_targets_of(ALLOWED))


def _record(manifest_sha256: str = "a" * 64) -> FixedChecksRecord:
    """建替身那一側會寫的 v2 紀錄（身分跟 :func:`_candidates` 逐欄相同、每個候選都過了；列照目標排，跟建替身一樣）。"""
    frequency_ruler, band_ruler = holdout_ruler_values()
    rows = tuple(FixedCheckRow(target=item.target, material_sha256=item.material_sha256, frequency_db=0.05 + index / 100,
                               band_db=0.001, frequency_ruler=frequency_ruler, band_ruler=band_ruler, passed=True,
                               surfaces=item.surfaces, material_digest=item.material_digest, full_solves=len(AXIS),
                               impedances_sha256=item.impedances_sha256)
                 for index, item in enumerate(sorted(_candidates(), key=lambda item: item.target)))
    return FixedChecksRecord(schema_version=FIXED_CHECKS_VERSION_V2, configuration_sha256=configuration_sha256(CONFIGURATION),
                             configuration=CONFIGURATION, manifest_sha256=manifest_sha256, frequencies_hz=axis_hex(AXIS),
                             operators_digest=OPERATORS, code_identity=CODE, checks=rows)


def _reuse(record: FixedChecksRecord | None, **changes: object) -> tuple[FixedCheckRow, ...] | str:
    given: dict[str, object] = {"manifest_sha256": "a" * 64, "configuration": CONFIGURATION, "frequencies_hz": AXIS,
                                "rulers": holdout_ruler_values(), "operators_digest": OPERATORS, "code_identity": CODE,
                                "candidates": _candidates(), **changes}
    call: Callable[..., tuple[FixedCheckRow, ...] | str] = reused_rows  # 各格照名字交，型別由產品那一支自己核
    return call(record, **given)


def test_same_identity_reuses_every_row_in_the_start_order() -> None:
    """抓全同卻不沿用、或沿用的列順序照紀錄（照字排）而不是開跑的候選順序（``project.json`` 要照 fixed_targets_of 排）。"""
    found = _reuse(_record())
    assert not isinstance(found, str), found
    assert [row.target for row in found] == list(fixed_targets_of(ALLOWED))
    reversed_order = tuple(reversed(_candidates()))
    found = _reuse(_record(), candidates=reversed_order)
    assert not isinstance(found, str) and [row.target for row in found] == [item.target for item in reversed_order]


def _row_change(target: str, **update: object) -> Callable[[FixedChecksRecord], FixedChecksRecord]:
    def change(record: FixedChecksRecord) -> FixedChecksRecord:
        rows = tuple(row.model_copy(update=update) if row.target == target else row for row in record.checks)
        return record.model_copy(update={"checks": rows})
    return change


def _identity_change(target: str, **update: object) -> tuple[CandidateIdentity, ...]:
    from dataclasses import replace

    change: Callable[..., CandidateIdentity] = replace
    return tuple(change(item, **update) if item.target == target else item for item in _candidates())


RECORD_CHANGES: dict[str, tuple[Callable[[FixedChecksRecord], FixedChecksRecord], str]] = {
    "替身": (lambda record: record.model_copy(update={"manifest_sha256": "b" * 64}), "核的不是這一份替身"),
    "起跑配置": (lambda record: record.model_copy(update={"configuration_sha256": "c" * 64}), "起跑配置跟建替身時"),
    "頻率軸差一個位元": (lambda record: record.model_copy(update={"frequencies_hz": axis_hex(
        (AXIS[0], math.nextafter(AXIS[1], math.inf), AXIS[2]))}), "頻率軸"),
    "頻率軸少一點": (lambda record: record.model_copy(update={"frequencies_hz": axis_hex(AXIS[:2])}), "頻率軸"),
    "完整模型": (lambda record: record.model_copy(update={"operators_digest": "6" * 64}), "完整模型"),
    "算核對的程式": (lambda record: record.model_copy(update={"code_identity": "fixcheck-v1:" + "7" * 64}), "算核對的程式"),
    "逐頻尺": (_row_change(MEASURED_TARGET, frequency_ruler=0.3), "用的尺"),
    "帶能量尺": (_row_change(CARPET_TARGET, band_ruler=0.02), "用的尺"),
    "材料內容摘要": (_row_change(CARPET_TARGET, material_sha256="d" * 64), "材料內容"),
    "材料身分摘要": (_row_change(MEASURED_TARGET, material_digest="e" * 64), "材料內容"),
    "准用的片": (_row_change(MEASURED_TARGET, surfaces=("x0",)), "准用的片"),
    "核對配置阻抗": (_row_change(CARPET_TARGET, impedances_sha256="f" * 64), "核對配置阻抗"),
    "沒過": (_row_change(CARPET_TARGET, passed=False), "核的沒過"),
    "記過了但數字超過尺": (_row_change(CARPET_TARGET, frequency_db=1.0), "核的沒過"),
    "少一個候選": (lambda record: record.model_copy(update={"checks": record.checks[:1]}), "候選跟建替身時核的那幾個不同"),
    "多一個候選": (lambda record: record.model_copy(update={"checks": (*record.checks, record.checks[0].model_copy(
        update={"target": "fixed:other"}))}), "候選跟建替身時核的那幾個不同"),
    "舊格式": (lambda record: FixedChecksRecord(
        schema_version=FIXED_CHECKS_VERSION, configuration_sha256=record.configuration_sha256,
        configuration=record.configuration, manifest_sha256=record.manifest_sha256,
        checks=tuple(row.model_copy(update={"surfaces": None, "material_digest": None, "full_solves": None,
                                            "impedances_sha256": None}) for row in record.checks)), "舊格式"),
}


@pytest.mark.parametrize("name", sorted(RECORD_CHANGES))
def test_any_record_field_that_differs_means_recheck(name: str) -> None:
    """抓紀錄任何一欄跟這一次不同還沿用：替身、起跑配置、頻率軸（差一個位元也算）、兩把尺、兩種材料摘要、准用的片、
    核對配置阻抗、沒過、數字超過尺、候選多一個少一個、舊格式（v1），每一欄單獨換掉都回原因（＝重核）。"""
    change, words = RECORD_CHANGES[name]
    found = _reuse(change(_record()))
    assert isinstance(found, str) and words in found, found


IDENTITY_CHANGES: dict[str, tuple[dict[str, object], str]] = {
    "這一次的替身": ({"manifest_sha256": "b" * 64}, "核的不是這一份替身"),
    "這一次的起跑配置": ({"configuration": {**CONFIGURATION, "mount": "rigid"}}, "起跑配置"),
    "這一次的頻率軸": ({"frequencies_hz": (AXIS[0], AXIS[1], math.nextafter(AXIS[2], 0.0))}, "頻率軸"),
    "這一次的完整模型": ({"operators_digest": "8" * 64}, "完整模型"),
    "這一次的程式或環境": ({"code_identity": "fixcheck-v1:" + "9" * 64}, "算核對的程式"),
    "登記簿的逐頻尺換了": ({"rulers": (holdout_ruler_values()[0] / 2, holdout_ruler_values()[1])}, "用的尺"),
    "登記簿的帶能量尺換了": ({"rulers": (holdout_ruler_values()[0], holdout_ruler_values()[1] / 2)}, "用的尺"),
    "這一次的材料內容": ({"candidates": _identity_change(MEASURED_TARGET, material_sha256="1" * 64)}, "材料內容"),
    "這一次的材料身分": ({"candidates": _identity_change(CARPET_TARGET, material_digest="2" * 64)}, "材料內容"),
    "這一次准用的片": ({"candidates": _identity_change(CARPET_TARGET, surfaces=("x0", "y0"))}, "准用的片"),
    "這一次的核對配置阻抗": ({"candidates": _identity_change(MEASURED_TARGET, impedances_sha256="3" * 64)}, "核對配置阻抗"),
    "這一次少一個候選": ({"candidates": _candidates()[:1]}, "候選"),
}


@pytest.mark.parametrize("name", sorted(IDENTITY_CHANGES))
def test_any_start_identity_that_differs_means_recheck(name: str) -> None:
    """同一份紀錄，這一次那一側任何一欄不同（登記簿的尺改了也算）就重核。"""
    changes, words = IDENTITY_CHANGES[name]
    found = _reuse(_record(), **changes)
    assert isinstance(found, str) and words in found, found


def test_no_record_or_unreadable_record_means_recheck(tmp_path: Path) -> None:
    """抓紀錄不在、壞掉、或版本跟欄位對不上（v2 缺一格、v1 多一格）還當成核過：一律讀成沒有紀錄、重核。"""
    assert _reuse(None) == "替身旁邊沒有建替身時的核對紀錄（或讀不懂）"
    record = _record()
    for name in ("frequencies_hz", "operators_digest", "code_identity"):
        with pytest.raises(ValidationError, match="欄位跟版本不一致"):
            FixedChecksRecord.model_validate({**record.model_dump(mode="json"), name: None})
    with pytest.raises(ValidationError, match="欄位跟版本不一致"):
        FixedChecksRecord.model_validate({**record.model_dump(mode="json"), "schema_version": FIXED_CHECKS_VERSION})
    rows = [row.model_dump(mode="json") for row in record.checks]
    rows[1]["impedances_sha256"] = None
    with pytest.raises(ValidationError, match="欄位跟版本不一致"):
        FixedChecksRecord.model_validate({**record.model_dump(mode="json"), "checks": rows})
    path = tmp_path / FIXED_CHECKS_FILE
    path.write_text(record.model_dump_json().replace('"full_solves":3', '"full_solves":null'), encoding="utf-8")
    assert read_record(path) is None
    path.write_text("{", encoding="utf-8")
    assert read_record(path) is None


def test_record_with_a_candidate_written_twice_is_no_record(tmp_path: Path) -> None:
    """抓同一個候選寫兩列（一列沒過、一列過）照收、比對只看到最後那一列就沿用：不收，讀成沒有紀錄、重核。"""
    record = _record()
    rows = [row.model_dump(mode="json") for row in record.checks]
    doubled = {**record.model_dump(mode="json"), "checks": [{**rows[0], "passed": False, "frequency_db": 1.0}, *rows]}
    with pytest.raises(ValidationError, match="不只一列"):
        FixedChecksRecord.model_validate(doubled)
    path = tmp_path / FIXED_CHECKS_FILE
    path.write_text(json.dumps(doubled), encoding="utf-8")
    assert read_record(path) is None
    assert _reuse(read_record(path)) == "替身旁邊沒有建替身時的核對紀錄（或讀不懂）"


def test_impedance_digest_changes_with_one_bit_and_ignores_key_order() -> None:
    """抓阻抗摘要只看四捨五入的值或被鍵的順序影響：差一個位元（實部或虛部）就不同，鍵換順序相同，剛性（None）跟 0 不同。"""
    rows = _impedances(MEASURED_TARGET)
    digest = _digest(rows)
    assert _digest(dict(reversed(list(rows.items())))) == digest
    values = cast(tuple[complex, ...], rows[Wall.X0])
    nudged = complex(math.nextafter(values[1].real, math.inf), values[1].imag)
    assert _digest({**rows, Wall.X0: (values[0], nudged, values[2])}) != digest
    nudged = complex(values[1].real, math.nextafter(values[1].imag, 0.0))
    assert _digest({**rows, Wall.X0: (values[0], nudged, values[2])}) != digest
    assert _digest({**rows, (Wall.Y0, "z1"): (0j, 0j, 0j)}) != digest
    assert _digest({Wall.X0: values}) != _digest({(Wall.X0, "z1"): values})
    # 單一值（型別准的寫法，整條軸同一個阻抗）：跟一點的逐頻寫法不同，差一個位元也不同。
    scalar = impedances_sha256(cast("FrequencyBoundaryImpedances", {Wall.X0: complex(400.0, -1.0)}))
    assert scalar != _digest({Wall.X0: (complex(400.0, -1.0),)})
    assert scalar != impedances_sha256(cast("FrequencyBoundaryImpedances",
                                            {Wall.X0: complex(math.nextafter(400.0, math.inf), -1.0)}))


# ---- 建替身那一側（surrogate_build._fixed_record）：程式身分建模之前量、寫之前再量 ----

def test_check_code_that_changes_during_the_build_writes_an_old_format_record(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """抓建替身途中程式換過還寫 v2：網頁起的建替身跑主 repo 那份程式，建模途中主線拉了新提交，數字是開頭載入的程式算的，
    寫紀錄那一刻磁碟上卻是新程式；紀錄要記建模開頭量的身分，寫之前再量一次不同就寫成 v1（網頁預覽照讀、開跑重核）。"""
    from aosr.gradient import surrogate_build
    from aosr.gradient.fixed_candidates import CandidateVerdict

    table = _table()
    stage = tmp_path / "stage"
    stage.mkdir()
    npz = stage / "rom.npz"
    manifest_path(npz).write_text('{"manifest": 1}\n', encoding="utf-8")
    start = surrogate_build._Start(presets=("glass_wool_48k_50mm_gap50", None), mount="on_wall",
                                   fixed_surfaces={"floor": "scheme-material"}, max_depths={"x0": 0.3}, allowed=ALLOWED,
                                   table=table, code_identity=CODE)
    fixed = fixed_targets(ALLOWED, table)
    job = SimpleNamespace(start=start, scheme=SimpleNamespace(scene=SimpleNamespace(model_dump_json=lambda: '{"scene":1}')),
                          fixed=fixed, stage=stage)
    report = cast("ValidationReport", SimpleNamespace(full_solves=len(AXIS)))
    verdicts = {item.target: CandidateVerdict(target=item.target, surfaces=item.surfaces, report=report, frequency_db=0.05,
                                              band_db=0.001, worst_hz=math.nan, passed=True, seconds=1.0)
                for item in _candidates()}
    prepared = SimpleNamespace(operators=object(),
                               configurations={target: _impedances(target) for target in fixed_targets_of(ALLOWED)})
    monkeypatch.setattr(fem_reduced, "operators_digest", lambda operators: OPERATORS)
    write: Callable[..., None] = surrogate_build._fixed_record  # 建模的人、驗證報告、完整模型是樁
    path = stage / FIXED_CHECKS_FILE
    found: dict[str, tuple[FixedChecksRecord | None, str]] = {}
    for label, now in (("same", CODE), ("changed", "fixcheck-v1:" + "9" * 64)):
        path.unlink(missing_ok=True)
        monkeypatch.setattr(fixed_candidates, "check_code_identity", lambda now=now: now)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            write(job, npz, verdicts, prepared, AXIS)
        found[label] = (read_record(path), stderr.getvalue())
    same, quiet = found["same"]
    assert same is not None and same.schema_version == FIXED_CHECKS_VERSION_V2 and same.code_identity == CODE
    assert not isinstance(_reuse(same, manifest_sha256=same.manifest_sha256), str) and quiet == ""
    changed, said = found["changed"]
    assert changed is not None and changed.schema_version == FIXED_CHECKS_VERSION and changed.code_identity is None
    assert "建替身途中算核對的程式或環境換過" in said and "照舊重核" in said, said
    assert _reuse(changed, manifest_sha256=changed.manifest_sha256) == "建替身時的核對紀錄是舊格式，沒有沿用要比的欄位"
    keep = {"target", "material_sha256", "frequency_db", "band_db", "frequency_ruler", "band_ruler", "passed"}
    assert [row.model_dump(include=keep) for row in changed.checks] == [row.model_dump(include=keep) for row in same.checks]
    assert unchecked_reason(stage, manifest_path(npz), CONFIGURATION, fixed) is None  # 網頁預覽照舊說核過


# ---- 開跑那一道（命令列 _fixed_candidate_checks）：沿用、或照舊重核 ----

def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, record: FixedChecksRecord | None,
           carryover: bool = False) -> tuple[tuple[FixedCandidateCheck, ...], str, list[object]]:
    """命令列開跑那一道：核對配置照 :func:`_impedances`、重核換成記下呼叫的樁；替身旁放 ``record``。回核對、輸出、重核呼叫。"""
    surrogate = tmp_path / "surrogate.npz"
    manifest_path(surrogate).write_text('{"manifest": 1}\n', encoding="utf-8")
    if record is not None:
        write_record(tmp_path / FIXED_CHECKS_FILE, record.model_copy(update={
            "manifest_sha256": manifest_digest(manifest_path(surrogate))}))
    built: list[object] = []
    rechecked: list[object] = []

    def configurations(designs: object, allowed: dict[str, tuple[str, ...]], *args: object,
                       **kwargs: object) -> dict[str, Rows]:
        found = {target: _impedances(target) for target in fixed_targets_of(allowed)}
        built.append(found)
        return found

    def recheck(*args: object, **kwargs: object) -> tuple[FixedCandidateCheck, ...]:
        rechecked.append(kwargs.get("configurations"))
        return ()

    monkeypatch.setattr(fixed_candidates, "candidate_configurations", configurations)
    monkeypatch.setattr(fixed_candidates, "check_fixed_candidates", recheck)
    monkeypatch.setattr(fixed_candidates, "check_code_identity", lambda: CODE)
    monkeypatch.setattr(fem_reduced, "operators_digest", lambda operators: OPERATORS)
    model = SimpleNamespace(frequencies_hz=AXIS)
    scheme = SimpleNamespace(scene=SimpleNamespace(density_kg_m3=1.2, sound_speed_m_s=343.0))
    stderr = io.StringIO()
    gate: Callable[..., tuple[FixedCandidateCheck, ...]] = cli._fixed_candidate_checks  # 替身、完整模型、方案是樁
    with redirect_stderr(stderr):
        checks = gate(model, object(), (), ALLOWED, _table(), scheme, 1, surrogate, configuration=CONFIGURATION,
                      carryover=carryover)
    assert len(built) == (0 if carryover else 1)  # 核對配置只算一次；沒沿用就把同一份交給重核
    assert carryover or all(given is built[0] for given in rechecked)
    return checks, stderr.getvalue(), rechecked


def test_start_reuses_without_any_full_solve_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓全同還重核（白花 16 分鐘）、沿用了卻沒講、或沿用的筆不照開跑的候選順序、沒標沿用、秒數照抄建替身那一次。"""
    checks, stderr, rechecked = _start(tmp_path, monkeypatch, record=_record())
    assert rechecked == []
    assert [check.target for check in checks] == list(fixed_targets_of(ALLOWED))
    assert all(check.reused and check.seconds == 0.0 and check.full_solves == len(AXIS) for check in checks)
    rows = {row.target: row for row in _record().checks}
    assert [(check.frequency_db, check.band_db, check.material_digest, check.surfaces) for check in checks] == [
        (rows[check.target].frequency_db, rows[check.target].band_db, rows[check.target].material_digest,
         rows[check.target].surfaces) for check in checks]
    assert stderr.count("注意：固定材料 ") == 2 and stderr.count("沿用建替身時核過的數字") == 2, stderr


@pytest.mark.parametrize("case", ["carryover", "v1", "none", "other_configuration"])
def test_start_rechecks_when_not_the_same(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """抓接續（carry-over）那條路、舊格式紀錄、沒有紀錄、起跑配置不同還沿用：照舊重核，輸出講為什麼沒沿用。"""
    record: FixedChecksRecord | None = _record()
    if case == "v1":
        record = RECORD_CHANGES["舊格式"][0](_record())
    elif case == "none":
        record = None
    elif case == "other_configuration":
        record = RECORD_CHANGES["起跑配置"][0](_record())
    checks, stderr, rechecked = _start(tmp_path, monkeypatch, record=record, carryover=case == "carryover")
    assert checks == () and len(rechecked) == 1, stderr
    words = {"carryover": "替身是接續（carry-over）沿用的", "v1": "舊格式", "none": "沒有建替身時的核對紀錄",
             "other_configuration": "起跑配置跟建替身時核的那一組不同"}[case]
    assert words in stderr and "開跑前照舊重核" in stderr and "沿用建替身時核過的數字（" not in stderr, stderr


def test_project_json_keeps_its_bytes_when_not_reused() -> None:
    """抓加了「沿用」那一格之後沒沿用的 ``project.json`` 位元組也跟著變（舊資料夾、舊考卷）：沒沿用不寫那一格。"""
    frequency_ruler, band_ruler = holdout_ruler_values()
    check = FixedCandidateCheck(target=CARPET_TARGET, surfaces=("x0",), frequency_db=0.1, band_db=0.001,
                                frequency_ruler=frequency_ruler, band_ruler=band_ruler, full_solves=3, seconds=1.5,
                                material_digest="a" * 64)
    assert "reused" not in check.model_dump(mode="json") and "reused" not in check.model_dump_json()
    assert check.model_copy(update={"reused": True}).model_dump(mode="json")["reused"] is True


def test_family_setting_that_moves_a_template_start_changes_the_check_impedances_and_means_recheck(tmp_path: Path) -> None:
    """抓家族設定（``--settings``）改了、起跑配置的正規寫法（只記代號）看不出來就沿用：總深度上限收窄，樣板片的下降起點
    被投影到新的上限內，候選核對配置的阻抗就變了；真的起跑設計與核對配置（子行程雙精度，不是樁），阻抗摘要要擋得住、要重核。"""
    import json
    import os
    import subprocess
    import sys
    from dataclasses import replace

    from aosr.runtime import child_process_env, configure_jax
    from tests.engine._cpu_pins import forward_cpu_pins

    env = forward_cpu_pins(child_process_env(threads=1))
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    env["TMPDIR"] = str(tmp_path)
    done = subprocess.run([sys.executable, str(Path(__file__).with_name("_fixed_checks_reuse_settings_probe.py")),
                           str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    found = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert found["changed"] is True and found["built"] != found["now"], found
    expanded = {surface: tuple(targets) for surface, targets in found["expanded"].items()}
    table = _table()
    identity = CandidateIdentity(target=CARPET_TARGET,
                                 material_sha256=fixed_targets(expanded, table)[0].coverage().material_sha256,
                                 material_digest=candidate_material_digest(CARPET_TARGET, table),
                                 surfaces=candidate_surfaces(CARPET_TARGET, expanded), impedances_sha256=found["built"])
    frequency_ruler, band_ruler = holdout_ruler_values()
    record = FixedChecksRecord(
        schema_version=FIXED_CHECKS_VERSION_V2, configuration_sha256=configuration_sha256(CONFIGURATION),
        configuration=CONFIGURATION, manifest_sha256="a" * 64, frequencies_hz=axis_hex(AXIS), operators_digest=OPERATORS,
        code_identity=CODE, checks=(FixedCheckRow(
            target=CARPET_TARGET, material_sha256=identity.material_sha256, frequency_db=0.05, band_db=0.001,
            frequency_ruler=frequency_ruler, band_ruler=band_ruler, passed=True, surfaces=identity.surfaces,
            material_digest=identity.material_digest, full_solves=3, impedances_sha256=found["built"]),))
    assert not isinstance(_reuse(record, candidates=(identity,)), str)  # 同一份設定：沿用
    rechecked = _reuse(record, candidates=(replace(identity, impedances_sha256=found["now"]),))
    assert isinstance(rechecked, str) and "核對配置阻抗" in rechecked


def test_code_identity_follows_the_check_code(tmp_path: Path) -> None:
    """抓算核對的程式（替身求值、比對、完整解）改了還沿用：物理身分不收那幾支，改了就要換程式身分；同一份程式同一個值。"""
    import shutil

    package = Path(fixed_candidates.__file__).resolve().parents[1]
    copy = tmp_path / "aosr"
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    same = fixed_candidates.check_code_identity(package_root=copy)
    assert same == fixed_candidates.check_code_identity(package_root=package)
    target = copy / "physics" / "fem_reduced_check.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n\ndef _changed() -> int:\n    return 1\n", encoding="utf-8")
    assert fixed_candidates.check_code_identity(package_root=copy) != same


def test_code_identity_follows_its_data_files_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓程式身分只收程式、不收閉包經 ``config_path`` 讀的資料檔或執行環境：資料檔差一個位元組、環境摘要換了，身分都要換
    （閉包掃描與程式摘要換成樁，只考這兩段有沒有餵進摘要）。"""
    from aosr.reporting import import_closure

    data = tmp_path / "config" / "data"
    data.mkdir(parents=True)
    (data / "probe.toml").write_bytes(b"a = 1\n")
    environment = ["environment-1"]
    monkeypatch.setattr(import_closure, "import_code_digest", lambda root, module: "code")
    monkeypatch.setattr(import_closure, "scan_import_closure",
                        lambda root, module: (SimpleNamespace(data_files=("probe.toml",)),))
    monkeypatch.setattr(import_closure, "environment_digest", lambda: environment[0])
    first = fixed_candidates.check_code_identity(package_root=tmp_path)
    assert first == fixed_candidates.check_code_identity(package_root=tmp_path)
    (data / "probe.toml").write_bytes(b"a = 2\n")
    second = fixed_candidates.check_code_identity(package_root=tmp_path)
    assert second != first
    environment[0] = "environment-2"
    assert fixed_candidates.check_code_identity(package_root=tmp_path) != second
