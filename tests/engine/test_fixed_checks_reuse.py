"""#1142（老闆 2026-11-13 拍 A）：開跑那一道固定材料核對沿用建替身時核過的數字，只在身分逐欄相同時。

身分：替身（manifest 整份檔案的摘要）、起跑配置（正規寫法的摘要）、頻率軸（逐位）、兩把尺（登記簿現值）、每個候選的
兩種材料摘要、准用它的片、核對配置的逐片逐頻阻抗摘要；紀錄要是 v2（舊格式沒有這些欄位）。任何一欄不同、舊格式、
紀錄讀不懂、或替身是接續（carry-over）沿用的，都照舊重核。真建替身、真開跑兩條路逐位相同的那一題在
``test_surrogate_fixed_build.py``（``pass`` 模式）。這裡專挑「不該沿用卻沿用」：每一欄單獨換掉都要重核。
"""
from __future__ import annotations

import io
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
from aosr.gradient.fixed_candidates import (
    FixedCandidateCheck, candidate_material_digest, candidate_surfaces, holdout_ruler_values,
)
from aosr.gradient.fixed_surfaces import fixed_targets_of
from aosr.gradient.surrogate_coverage import fixed_targets
from aosr.gradient.surrogate_fixed_checks import (
    FIXED_CHECKS_FILE, FIXED_CHECKS_VERSION, FIXED_CHECKS_VERSION_V2, CandidateIdentity, FixedCheckRow, FixedChecksRecord,
    axis_hex, configuration_document, configuration_sha256, impedances_sha256, manifest_digest, read_record, reused_rows,
    write_record,
)
from aosr.physics.fem_reduced_store import manifest_path
from tests.engine.test_material_gradient_allowed import CARPET, MEASURED, _table

if TYPE_CHECKING:
    from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances

CARPET_TARGET, MEASURED_TARGET = "fixed:" + CARPET, "fixed:" + MEASURED
ALLOWED: dict[str, tuple[str, ...]] = {"x0": ("none", CARPET_TARGET, MEASURED_TARGET), "y0": (MEASURED_TARGET,)}
AXIS = (50.0, 120.0, 293.441)
CONFIGURATION = configuration_document(presets=("glass_wool_48k_50mm_gap50", None), mount="on_wall",
                                       fixed_surfaces={"floor": "scheme-material"}, max_depths={"x0": 0.3},
                                       allowed=ALLOWED, scene_json='{"scene":1}')
Rows = dict[Wall | tuple[Wall, str], tuple[complex, ...] | None]


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
                             checks=rows)


def _reuse(record: FixedChecksRecord | None, **changes: object) -> tuple[FixedCheckRow, ...] | str:
    given: dict[str, object] = {"manifest_sha256": "a" * 64, "configuration": CONFIGURATION, "frequencies_hz": AXIS,
                                "rulers": holdout_ruler_values(), "candidates": _candidates(), **changes}
    return reused_rows(record, **given)  # type: ignore[arg-type]


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

    return tuple(replace(item, **update) if item.target == target else item for item in _candidates())  # type: ignore[arg-type]


RECORD_CHANGES: dict[str, tuple[Callable[[FixedChecksRecord], FixedChecksRecord], str]] = {
    "替身": (lambda record: record.model_copy(update={"manifest_sha256": "b" * 64}), "核的不是這一份替身"),
    "起跑配置": (lambda record: record.model_copy(update={"configuration_sha256": "c" * 64}), "起跑配置跟建替身時"),
    "頻率軸差一個位元": (lambda record: record.model_copy(update={"frequencies_hz": axis_hex(
        (AXIS[0], math.nextafter(AXIS[1], math.inf), AXIS[2]))}), "頻率軸"),
    "頻率軸少一點": (lambda record: record.model_copy(update={"frequencies_hz": axis_hex(AXIS[:2])}), "頻率軸"),
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
    with pytest.raises(ValidationError, match="欄位跟版本不一致"):
        FixedChecksRecord.model_validate({**record.model_dump(mode="json"), "frequencies_hz": None})
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
    model = SimpleNamespace(frequencies_hz=AXIS)
    scheme = SimpleNamespace(scene=SimpleNamespace(density_kg_m3=1.2, sound_speed_m_s=343.0))
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        checks = cli._fixed_candidate_checks(
            model, object(), (), ALLOWED, _table(), scheme, 1, surrogate,  # type: ignore[arg-type]
            configuration=CONFIGURATION, carryover=carryover)
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
