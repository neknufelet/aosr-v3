"""本機合併前驗收（#1082 段 2-3b）：小房間真跑一場帶每區准用清單（含使用者實測表與內建地毯）的搜尋，從開跑一路到終點。

雲端那兩份（``test_material_gradient_allowed``、``..._runs``）用合成目標與探針考每一個讀的人；這一份開真的子行程：
開跑（命令列自己開雙精度、准用清單核起點、固定材料各做一組替身留出驗證、快照 v2）→ 工作行程重建目標與換樣板
（換樣板只換到清單裡的）→ 寫結果 → 報告 → 終點（完整模型、核清單與固定材料摘要、匯出）。環境照網頁開服務那一條。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from aosr.gradient import run
from aosr.gradient import template_ledger as templates
from aosr.gradient.endpoint_run import read_result
from aosr.materials import library
from aosr.materials import search_library as materials
from tests.engine._six_search_cli_case import model as model, operators as operators
from tests.engine.slow_material_gradient_library_cli import _run
from tests.engine.test_material_gradient_allowed import CARPET, MEASURED, _measured
from tests.engine.test_material_gradient_cli import Record, _command, _start_args
from tests.engine.test_material_gradient_endpoint_cli import _inputs, _surrogate

ALLOWED = ("ceiling=none,fixed:" + MEASURED, "x0=glass_wool_48k,fixed:" + CARPET)


@pytest.fixture(scope="module")
def searched(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Record]:
    directory = tmp_path_factory.mktemp("material-gradient-allowed-cli")
    _surrogate(directory, request)
    _inputs(directory)
    user = directory / "library.json"
    user.write_text(library.dump_user_library((_measured(),)), encoding="utf-8")
    root = directory / "root"
    flags = [item for value in ALLOWED for item in ("--allowed", value)]
    output: Record = {"start": _run(directory, [*_start_args(directory, root), "--materials", str(user), *flags])}
    folders = [path for path in root.iterdir() if not path.name.startswith(".")] if root.exists() else []
    if len(folders) == 1:
        output["report"] = _run(directory, _command("report", "--folder", str(folders[0])))
        output["endpoint"] = _run(directory, _command("endpoint", str(folders[0])))
    return (folders[0] if len(folders) == 1 else directory), output


def test_start_checks_each_fixed_material_and_snapshots_the_user_table(searched: tuple[Path, Record]) -> None:
    folder, output = searched
    start = cast(Record, output["start"])
    assert start["code"] == 0, start
    project = run.read_project(folder)
    assert project.allowed_targets == {"ceiling": ("fixed:" + MEASURED, "none"), "x0": ("glass_wool_48k", "fixed:" + CARPET)}
    assert [check.target for check in project.fixed_candidate_checks] == ["fixed:" + CARPET, "fixed:" + MEASURED]
    assert "替身留出驗證通過" in str(start["stderr"])
    snapshot = folder / materials.SNAPSHOT_FILE
    assert json.loads(snapshot.read_bytes())["format_version"] == materials.SNAPSHOT_VERSION_WITH_FIXED
    header, rows = templates.read_template_ledger(folder / run.TEMPLATES_FILE)
    assert set(header.fixed_candidate_digests) == {"fixed:" + CARPET, "fixed:" + MEASURED}
    for row in rows:
        surfaces = dict(row.surfaces)
        assert surfaces["ceiling"] in (None, "fixed:" + MEASURED)
        assert surfaces["x0"] is not None and surfaces["x0"] != "none"
    assert len(read_result(folder).surfaces) == 6


def test_report_and_endpoint_finish_with_the_lists(searched: tuple[Path, Record]) -> None:
    folder, output = searched
    report, finished = cast(Record, output["report"]), cast(Record, output["endpoint"])
    assert report["code"] == 0, report
    assert finished["code"] == 0, finished
    record = json.loads((folder / "endpoint.json").read_text(encoding="utf-8"))
    assert record["gradient_id"] == folder.name
