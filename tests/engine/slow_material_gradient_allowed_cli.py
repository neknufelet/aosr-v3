"""本機合併前驗收（#1082 段 2-3b）：小房間真跑一場帶每區准用清單（含使用者的構造型固定材料與內建地毯）的搜尋，從開跑一路到終點。

雲端那兩份（``test_material_gradient_allowed``、``..._runs``）用合成目標與探針考每一個讀的人；這一份開真的子行程：
開跑（命令列自己開雙精度、准用清單核起點、固定材料各做一組替身留出驗證、快照 v2）→ 工作行程重建目標與換樣板
（換樣板只換到清單裡的，第 0 輪兩個固定材料候選真的走正式六類評分）→ 寫結果 → 報告 → 終點（完整模型、核清單與固定材料摘要、
匯出）。環境照網頁開服務那一條。另一場（#1082 段 2-3d）x0 准用使用者沒有虛部的實測阻抗表：開跑標「近似」、做關卡，
第 0 輪那個候選走正式六類評分（表內／表外吸音分流），一路跑到終點。

替身樁旁邊沒有訓練說明時，清單有固定材料候選就拒收（#1082 段 2-3c，「替身沒用這些材料訓練過，要重建」）：先跑一次照實拒收，
再照產品那一條（:mod:`aosr.gradient.surrogate_build` 的 ``_open``）記上訓練過這一場的家族與候選，才開正式那一場；實測表那一場也一樣先拒、記上才開（段 2-3d）。

替身要用這份方案自己的算子、整條正式軸真的建一份（不借命令列考卷那份衍生替身：它的座位列是三個真座位的組合、
頻點只建一半，搜尋用得了，拿方案真網格做完整解比一定對不上，關卡照實拒收）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import run
from aosr.gradient import template_ledger as templates
from aosr.gradient.endpoint_run import read_result
from aosr.materials import library
from aosr.materials import search_library as materials
from aosr.physics.fem_helmholtz import FrequencyWallImpedances
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics.fem_reduced_check import validate_reduced
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, save_validation_receipt
from aosr.physics.fem_reduced_store import save_reduced
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.reporting.scheme import load_scheme
from tests.engine._precision_contracts import contract_value
from tests.engine._surrogate_coverage import trained_on
from tests.engine.slow_material_gradient_library_cli import _run
from tests.engine.test_material_gradient_allowed import BUILT, CARPET, MEASURED, _built
from tests.engine.test_material_gradient_cli import NARROWED_STEPS, Record, _command, _settings_text, _start_args
from tests.engine.test_material_gradient_endpoint_cli import _inputs
from tests.engine.test_fem_reduced import DENSITY, SOUND_SPEED, _settings

# 起點（命令列考卷那一組）：地板 48K+空腔、天花板不貼、x0 48K+空腔、xL 48K 貼牆、y0 不貼、yL 24K。其他面只准起點那一個，
# 第 0 輪就只有天花板換構造型固定材料、x0 換內建地毯兩個候選。
ALLOWED = ("floor=glass_wool_48k", "ceiling=none,fixed:" + BUILT, "x0=glass_wool_48k,fixed:" + CARPET, "xL=glass_wool_48k",
           "y0=none", "yL=glass_wool_24k")
# 實測表那一場：只有 x0 換得到那張表，第 0 輪就只有一個候選。表是合成的（不是真實量測）：50 mm 玻璃棉剛背的阻抗實部
# 取整到 0.1，沒有虛部（標「近似」），底構造就是那份玻璃棉。2-3b 那份表的底構造是 5 cm 無損空氣層，表外整段沒有吸音，
# 它單獨當構造六類就不可估，換上去也一樣不可估（搜尋照實記「快速評分不可估」），考不到終點。
REAL_ROWS = ((50.0, 371.9), (100.0, 334.0), (200.0, 308.1), (400.0, 298.5), (800.0, 320.3), (1600.0, 437.9),
             (3150.0, 526.5), (6300.0, 487.4))


def _real_only() -> library.FixedEntry:
    return library.FixedEntry.model_validate({
        "kind": "fixed", "id": MEASURED, "label": "實測吸音板（自填、沒有虛部）", "provenance": "measured", "source": "阻抗管實測（合成考卷用）",
        "material": {"kind": "measured_impedance", "source_and_conditions": "合成考卷用表，非真實量測",
                     "points": [{"frequency_hz": f, "real_pa_s_m": r} for f, r in REAL_ROWS],
                     "base_assembly": {"kind": "assembly", "backing": "rigid", "layers": [
                         {"kind": "porous", "material_class": "fibrous", "model": "komatsu", "thickness_m": 0.05,
                          "flow_resistivity_pa_s_m2": 16000.0}]}}})



MEASURED_ALLOWED = tuple("x0=glass_wool_48k,fixed:" + MEASURED if item.startswith("x0=") else
                         "ceiling=none" if item.startswith("ceiling=") else item for item in ALLOWED)
# 命令列考卷那份縮小步數的設定沿用，只把換樣板打開（那份的求值上限 1 不夠開第 0 輪）。
ROUND_STEPS = {**NARROWED_STEPS, "templates.max_evaluations": 100, "templates.max_rounds": 2, "templates.max_per_round": 1}


def _materials(seed: int, count: int) -> list[FrequencyWallImpedances]:
    """訓練／留出材料：抽法同 ``test_fem_reduced``，頻率軸換成整條正式有限元素軸（替身的軸）。"""
    rng = np.random.default_rng(seed)
    frequencies = np.asarray(FEM_LANE_FREQUENCIES_HZ)
    sets: list[FrequencyWallImpedances] = []
    for _ in range(count):
        sets.append({wall: tuple(complex(z) for z in DENSITY * SOUND_SPEED * (
            math.exp(rng.uniform(math.log(1.5), math.log(60.0))) - 1j * rng.uniform(0.0, 5.0) * 100.0 / frequencies))
            for wall in Wall.all()})
    return sets


def _scheme_surrogate(directory: Path) -> None:
    """方案自己的算子建替身、做留出驗證寫收據，放在暫存資料夾的 good 子資料夾（開跑參數照命令列考卷那一條）。"""
    capabilities = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    operators = export_operators(load_scheme(directory / "scheme.json"), capabilities=capabilities, directivity=directivity)
    model = build_reduced_model(operators, FEM_LANE_FREQUENCIES_HZ, _materials(1, 12), settings=_settings())
    report = validate_reduced(model, operators, _materials(99, 2))
    rulers = HoldoutRulers(frequency=Ruler("fem_reduced_holdout_frequency_db", contract_value("fem_reduced_holdout_frequency_db")),
                           band=Ruler("fem_reduced_holdout_band_db", contract_value("fem_reduced_holdout_band_db")))
    (directory / "good").mkdir()
    save_reduced(model, directory / "good" / "rom.npz")
    save_validation_receipt(directory / "good" / "rom.npz", report, rulers=rulers, seconds=1.0)


def _record_training(directory: Path, user: Path, allowed: tuple[str, ...]) -> None:
    """照建模命令列（``surrogate_build._open``）組這一場的材料與固定材料候選，在 good/ 旁寫訓練說明。"""
    from aosr.config.material_gradient import fixed_target_id, load_material_gradient_settings
    from aosr.gradient.boundary import scheme_patch_keys, surface_ids
    from aosr.gradient.cli import user_entries
    from aosr.gradient.fixed_surfaces import expanded_allowed_targets, fixed_targets_of, parse_allowed_targets
    from aosr.gradient.surrogate_coverage import fixed_targets

    entries = user_entries(user)
    found = materials.with_user_materials(load_material_gradient_settings(directory / "settings-rounds.toml"),
                                          run.presets_table(), entries)
    listed = expanded_allowed_targets(parse_allowed_targets(list(allowed)),
                                      surface_ids(scheme_patch_keys(load_scheme(directory / "scheme.json"))))
    targets = fixed_targets_of(listed)
    candidates = materials.fixed_candidates(targets, found.presets,
                                            materials.user_fixed_entries([fixed_target_id(target) for target in targets], entries))
    trained_on(directory / "good" / "rom.npz", found, fixed_targets(listed, candidates))


@pytest.fixture(scope="module")
def searched(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Record]:
    directory = tmp_path_factory.mktemp("material-gradient-allowed-cli")
    _inputs(directory)
    (directory / "settings-rounds.toml").write_text(_settings_text(ROUND_STEPS), encoding="utf-8")
    _scheme_surrogate(directory)
    user = directory / "library.json"
    user.write_text(library.dump_user_library((_real_only(), _built())), encoding="utf-8")
    measured = [item for value in MEASURED_ALLOWED for item in ("--allowed", value)]
    # 替身涵蓋閘門（#1082 段 2-3c）對實測固定材料：還沒記訓練過它就拒收、不建資料夾；記上之後才開得了跑。
    untrained_measured = directory / "untrained-measured-root"
    output: Record = {"measured_untrained": _run(directory, [*_start_args(directory, untrained_measured, settings="settings-rounds.toml"),
                                                             "--materials", str(user), *measured])}
    output["measured_untrained_folders"] = sorted(path.name for path in untrained_measured.iterdir()) if untrained_measured.exists() else []
    _record_training(directory, user, MEASURED_ALLOWED)
    measured_root = directory / "measured-root"
    output["measured"] = _run(directory, [*_start_args(directory, measured_root, settings="settings-rounds.toml"),
                                          "--materials", str(user), *measured])
    measured_folders = [path for path in measured_root.iterdir() if not path.name.startswith(".")] if measured_root.exists() else []
    output["measured_folders"] = sorted(path.name for path in measured_folders)
    if len(measured_folders) == 1:
        output["measured_endpoint"] = _run(directory, _command("endpoint", str(measured_folders[0])))
        output["measured_folder"] = str(measured_folders[0])
    flags = [item for value in ALLOWED for item in ("--allowed", value)]
    untrained_root = directory / "untrained-root"
    output["untrained"] = _run(directory, [*_start_args(directory, untrained_root, settings="settings-rounds.toml"),
                                           "--materials", str(user), *flags])
    output["untrained_folders"] = sorted(path.name for path in untrained_root.iterdir()) if untrained_root.exists() else []
    _record_training(directory, user, ALLOWED)
    root = directory / "root"
    output["start"] = _run(directory, [*_start_args(directory, root, settings="settings-rounds.toml"), "--materials", str(user),
                                       *flags])
    folders = [path for path in root.iterdir() if not path.name.startswith(".")] if root.exists() else []
    if len(folders) == 1:
        output["report"] = _run(directory, _command("report", "--folder", str(folders[0])))
        output["endpoint"] = _run(directory, _command("endpoint", str(folders[0])))
    return (folders[0] if len(folders) == 1 else directory), output


def test_start_checks_each_fixed_material_and_snapshots_the_user_table(searched: tuple[Path, Record]) -> None:
    folder, output = searched
    untrained = cast(Record, output["untrained"])
    assert untrained["code"] != 0 and output["untrained_folders"] == [], untrained
    assert "替身沒用這些材料訓練過，要重建" in str(untrained["stderr"]) and f"固定材料 fixed:{BUILT} 沒訓練過" in str(untrained["stderr"])
    start = cast(Record, output["start"])
    assert start["code"] == 0, start
    project = run.read_project(folder)
    assert project.allowed_targets["ceiling"] == ("fixed:" + BUILT, "none")
    assert project.allowed_targets["x0"] == ("glass_wool_48k", "fixed:" + CARPET)
    assert [check.target for check in project.fixed_candidate_checks] == ["fixed:" + CARPET, "fixed:" + BUILT]
    assert "替身留出驗證通過" in str(start["stderr"])
    snapshot = folder / materials.SNAPSHOT_FILE
    assert json.loads(snapshot.read_bytes())["format_version"] == materials.SNAPSHOT_VERSION_WITH_FIXED
    header, rows = templates.read_template_ledger(folder / run.TEMPLATES_FILE)
    assert set(header.fixed_candidate_digests) == {"fixed:" + CARPET, "fixed:" + BUILT}
    assert {check.material_digest for check in project.fixed_candidate_checks} == set(header.fixed_candidate_digests.values())
    first = {row.candidate: row for row in rows if row.round == 0 and row.phase == "quick_score"}
    assert set(first) == {"ceiling->fixed:" + BUILT, "x0->fixed:" + CARPET}, sorted(map(str, first))
    assert all(math.isfinite(row.value) for row in first.values())
    for row in rows:
        surfaces = dict(row.surfaces)
        assert surfaces["ceiling"] in (None, "fixed:" + BUILT)
        assert surfaces["x0"] is not None and surfaces["x0"] != "none"
    assert len(read_result(folder).surfaces) == 6


def test_report_and_endpoint_finish_with_the_lists(searched: tuple[Path, Record]) -> None:
    folder, output = searched
    report, finished = cast(Record, output["report"]), cast(Record, output["endpoint"])
    assert report["code"] == 0, report
    assert finished["code"] == 0, finished
    record = json.loads((folder / "endpoint.json").read_text(encoding="utf-8"))
    assert record["gradient_id"] == folder.name
    # 這間小房第 0 輪接受天花板換構造型固定材料（帳本記的決定）；終點匯出的方案天花板就是那份材料本身。
    rows = templates.read_template_ledger(folder / run.TEMPLATES_FILE)[1]
    accepted = [row.decision.accepted for row in rows if row.decision is not None and row.decision.accepted is not None]
    assert accepted[:1] == ["ceiling->fixed:" + BUILT], accepted
    exported = run.scene_wall_materials(load_scheme(folder / "scheme-endpoint.json"),
                                        capabilities=load_capabilities(config_path("capabilities.toml")),
                                        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    assert exported[Wall.CEILING] == _built().surface_material()


def test_a_listed_measured_table_needs_the_surrogate_trained_on_it(searched: tuple[Path, Record]) -> None:
    """抓替身涵蓋閘門（段 2-3c）放過沒訓練過的實測固定材料，或訓練過了還擋（段 2-3d 接上實測表後的組合）：
    同一份清單，替身沒記它就拒收、不建資料夾、點名那張表；記上之後開得了跑（下一題核一路到終點）。"""
    output = searched[1]
    refused = cast(Record, output["measured_untrained"])
    assert refused["code"] != 0 and output["measured_untrained_folders"] == [], refused
    assert "替身沒用這些材料訓練過，要重建" in str(refused["stderr"]), refused
    assert f"固定材料 fixed:{MEASURED} 沒訓練過" in str(refused["stderr"]), refused
    assert cast(Record, output["measured"])["code"] == 0 and len(cast(list[str], output["measured_folders"])) == 1


def test_a_listed_measured_table_is_searched_and_reaches_the_endpoint(searched: tuple[Path, Record]) -> None:
    """抓命令列仍拒收實測阻抗表（2-3b 的暫時拒收）、沒虛部的沒標「近似」、候選沒走正式六類評分（分數不是有限數），
    或終點沒吃到那張表（匯出的 x0 不是那份材料）。"""
    output = searched[1]
    measured = cast(Record, output["measured"])
    assert measured["code"] == 0, measured
    stderr = str(measured["stderr"])
    assert f"固定材料 {MEASURED}" in stderr and "沒有虛部" in stderr and "近似" in stderr, stderr
    assert f"固定材料 fixed:{MEASURED} 替身留出驗證通過" in stderr, stderr
    folder = Path(str(output["measured_folder"]))
    rows = templates.read_template_ledger(folder / run.TEMPLATES_FILE)[1]
    first = [row for row in rows if row.round == 0 and row.phase == "quick_score"]
    assert [row.candidate for row in first] == ["x0->fixed:" + MEASURED], first
    assert all(math.isfinite(row.value) for row in first)
    finished = cast(Record, output["measured_endpoint"])
    assert finished["code"] == 0, finished
    accepted = [row.decision.accepted for row in rows if row.decision is not None and row.decision.accepted is not None]
    exported = run.scene_wall_materials(load_scheme(folder / "scheme-endpoint.json"),
                                        capabilities=load_capabilities(config_path("capabilities.toml")),
                                        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    # 這間小房第 0 輪接受 x0 換實測表（帳本記的決定）；終點匯出的 x0 就是那張表（虛部補 0 的物理格式）。
    assert accepted[:1] == ["x0->fixed:" + MEASURED], accepted
    assert exported[Wall.X0] == _real_only().surface_material()
