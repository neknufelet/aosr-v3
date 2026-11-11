"""本機合併前驗收（#1082 段 2-3a）：小房間真跑一場起點有使用者家族的搜尋，從開跑一路到終點。

雲端那份 ``test_material_gradient_library`` 用替身樁與合成帳本考每一個讀的人；這一份開真的子行程：開跑（命令列自己開
雙精度、物理檢查在開跑行程裡算）→ 工作行程重建目標與換樣板 → 寫結果 → 報告 → 終點（完整模型、核樣板身分、匯出）。
環境照網頁開服務那一條（``child_process_env(threads=1)``，不替它開雙精度），抓「開跑行程沒開雙精度、把精度問題講成材料問題」。
開跑之後改網頁那份材料庫，接續的讀者照讀資料夾裡的快照，終點照樣做得完。
#1082 段 2-3c：替身（考卷合成的那一份）旁邊沒記訓練過這個使用者家族時，真的命令列開跑要拒收、講要重建、不建資料夾；
記了（照產品 ``coverage_for`` 生成）才開得了跑。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

from aosr.config.material_gradient import load_material_gradient_settings
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import run
from aosr.gradient.endpoint_run import read_result
from aosr.materials import library
from aosr.materials import search_library as materials
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._surrogate_coverage import trained_on
from tests.engine._six_search_cli_case import model as model, operators as operators
from tests.engine.test_material_gradient_cli import PRESETS, WAIT_S, Record, _command, _start_args
from tests.engine.test_material_gradient_endpoint_cli import _inputs, _surrogate

USER = "rock_wool_60k"
_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
_POROUS: dict[str, object] = {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 20000.0,
                              "material_class": "fibrous"}


def _entry(flow: float = 20000.0) -> library.LibraryEntry:
    return _ENTRY.validate_python({
        "kind": "optimizable", "id": USER, "label": "60K 岩棉（自填）", "provenance": "model", "source": "使用者自填流阻",
        "porous": {**_POROUS, "flow_resistivity_pa_s_m2": flow},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.04}, "max_total_depth_m": 0.15})


def _web_env(directory: Path) -> dict[str, str]:
    """網頁開服務給命令列的那一份環境；雙精度由命令列入口自己開。"""
    env = forward_cpu_pins(child_process_env(threads=1))
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    return env


def _run(directory: Path, argv: list[str]) -> Record:
    done = subprocess.run(argv, cwd=directory, env=_web_env(directory), capture_output=True, text=True, timeout=WAIT_S)
    return {"code": done.returncode, "stdout": done.stdout, "stderr": done.stderr}


@pytest.fixture(scope="module")
def searched(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Record]:
    directory = tmp_path_factory.mktemp("material-gradient-library-cli")
    _surrogate(directory, request)
    _inputs(directory)
    user = directory / "library.json"
    user.write_text(library.dump_user_library((_entry(),)), encoding="utf-8")
    presets = ",".join("none" if preset is None else USER if index in (0, 2) else preset
                       for index, preset in enumerate(PRESETS))
    root = directory / "root"
    output: Record = {"untrained": _run(directory, [*_start_args(directory, root, presets=presets), "--materials", str(user)])}
    output["untrained_folders"] = sorted(path.name for path in root.iterdir()) if root.exists() else []
    trained_on(directory / "good" / "rom.npz", materials.with_user_materials(
        load_material_gradient_settings(directory / "settings.toml"), run.presets_table(), (_entry(),)))
    output["start"] = _run(directory, [*_start_args(directory, root, presets=presets), "--materials", str(user)])
    folders = [path for path in root.iterdir() if not path.name.startswith(".")] if root.exists() else []
    if len(folders) == 1:
        user.write_text(library.dump_user_library((_entry(30000.0),)), encoding="utf-8")
        output["report"] = _run(directory, _command("report", "--folder", str(folders[0])))
        output["endpoint"] = _run(directory, _command("endpoint", str(folders[0])))
    return (folders[0] if len(folders) == 1 else directory), output


def test_start_refuses_a_surrogate_not_trained_on_the_user_family(searched: tuple[Path, Record]) -> None:
    _, output = searched
    untrained = cast(Record, output["untrained"])
    assert untrained["code"] == 1 and output["untrained_folders"] == []
    assert "SurrogateNotTrainedOn：替身沒用這些材料訓練過，要重建" in str(untrained["stderr"])
    assert "家族 rock_wool_60k 沒訓練過" in str(untrained["stderr"])


def test_start_runs_the_user_family_search_and_snapshots_it(searched: tuple[Path, Record]) -> None:
    folder, output = searched
    start = cast(Record, output["start"])
    assert start["code"] == 0, start
    project = run.read_project(folder)
    snapshot = folder / materials.SNAPSHOT_FILE
    assert project.materials_path == str(snapshot) and project.materials_sha256 == run.file_sha256(snapshot)
    assert snapshot.read_bytes() == materials.snapshot_bytes((cast(library.OptimizableEntry, _entry()),))
    header, _ = ledger.read_ledger(folder / run.LEDGER_FILE)
    assert [preset for _, preset in header.surfaces].count(USER) == 2
    result = read_result(folder)
    assert len(result.surfaces) == 6 and (folder / run.RESULT_FILE).is_file()


def test_report_and_endpoint_read_the_folder_snapshot_after_the_library_changed(searched: tuple[Path, Record]) -> None:
    folder, output = searched
    report, finished = cast(Record, output["report"]), cast(Record, output["endpoint"])
    assert report["code"] == 0, report
    assert finished["code"] == 0, finished
    record = json.loads((folder / "endpoint.json").read_text(encoding="utf-8"))
    assert record["gradient_id"] == folder.name
