"""#1032：第三階段舊資料夾拒收與終點報告版本相容。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from aosr.gradient import endpoint_run
from aosr.gradient.endpoint_run import EndpointRecord
from aosr.gradient.report import ENDPOINT_HEADING
from tests.engine.test_material_gradient_cli import Record, _command, _json, _run
from tests.engine.test_material_gradient_endpoint_cli import (
    _PROXY_FIELDS, _folder, _record, model as model, operators as operators, minimal_runs as runs,
)


@pytest.mark.parametrize("command", ["resume", "endpoint"])
def test_old_scored_range_folder_rejected_with_restart_message(
        runs: tuple[Path, Record], tmp_path: Path, command: str) -> None:
    """舊目標指紋資料夾必須在求解前拒收，說明 #1032 與重新搜尋。"""
    folder = tmp_path / "old-range"
    shutil.copytree(_folder(runs), folder)
    project = _json(folder / "project.json")
    project["objective_settings_fingerprint"] = "e43e0f93a65016ccb1320333c3f80d08d2114d54e4acd7207f7e941f5dfcf971"
    (folder / "project.json").write_text(json.dumps(project), encoding="utf-8")
    if command == "endpoint":
        (folder / "endpoint.json").rename(folder / "endpoint-previous.json")
    before = {name: (folder / name).read_bytes() for name in ("ledger.jsonl", "templates.jsonl", "result.json")}
    result = _run(tmp_path, _command(command, str(folder)))
    assert result["code"] != 0
    message = str(result["stderr"])
    assert "#1032" in message and "第三階段計分範圍" in message and "重新開始搜尋" in message
    assert all((folder / name).read_bytes() == content for name, content in before.items())


def test_endpoint_v4_carries_unscored_notes_and_chinese_report(runs: tuple[Path, Record]) -> None:
    record = _record(_folder(runs))
    assert record.format_version == endpoint_run.ENDPOINT_VERSION
    assert record.report.below_scoring is not None and record.baseline.below_scoring is not None
    assert record.improvement.below_scoring is not None
    assert record.report.below_scoring.range_hz == (20.0, 30.0)
    assert record.baseline.below_scoring.range_hz == (20.0, 30.0)
    from aosr.gradient.report import render_report
    text = render_report(_folder(runs))
    assert "30 Hz 以下（不計分）" in text and "座位平均能量" in text and "原牆" in text
    for name, gain in record.improvement.below_scoring.items():
        base = record.baseline.below_scoring.by_speaker[name].seat_mean_max
        end = record.report.below_scoring.by_speaker[name].seat_mean_max
        assert base is not None and end is not None
        assert gain.seat_mean_max_db == base.db - end.db


@pytest.mark.parametrize("version", ["aosr.material_gradient_endpoint.v3", "aosr.material_gradient_endpoint.v2"])
def test_legacy_endpoint_notes_absent_and_read_requires_legacy(
        runs: tuple[Path, Record], tmp_path: Path, version: str) -> None:
    from aosr.gradient.report import render_report
    folder = tmp_path / "legacy"
    shutil.copytree(_folder(runs), folder)
    document = _json(folder / "endpoint.json")
    document["format_version"] = version
    document.pop('six_category', None)
    for part in ("report", "baseline", "improvement"):
        cast(Record, document[part]).pop("below_scoring", None)
    if version.endswith(".v2"):
        for part, fields in (("report", _PROXY_FIELDS),
                             ("baseline", ("full_objective_total", "full_proxy_db", "full_interior_max_db")),
                             ("improvement", ("objective_total", "proxy_db"))):
            for name in fields:
                cast(Record, document[part]).pop(name)
    (folder / "endpoint.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="版本"):
        endpoint_run.read_endpoint(folder)
    record = endpoint_run.read_endpoint(folder, legacy=True)
    assert record.report.below_scoring is None
    text = render_report(folder)
    assert ENDPOINT_HEADING in text and "Hz 以下（不計分）" not in text


@pytest.mark.parametrize("part", ["report", "baseline", "improvement"])
def test_endpoint_v4_rejects_missing_unscored_annotation(runs: tuple[Path, Record], part: str) -> None:
    document = _json(_folder(runs) / "endpoint.json")
    cast(Record, document[part]).pop("below_scoring", None)
    with pytest.raises(ValueError, match="不計分"):
        EndpointRecord.model_validate(document)
