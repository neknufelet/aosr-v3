"""本機合併前驗收的搜尋與終點重考卷；考卷本體由原檔原樣搬移。"""
from __future__ import annotations
import dataclasses
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast
import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from tests.engine._six_search_helpers import six_header_for
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.config.material_gradient import MOUNT_ON_WALL, MOUNT_RIGID, MaterialGradientSettings, Mount, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, endpoint, run
from aosr.gradient.endpoint_run import read_result
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.gradient.report import MOUNT_ON_WALL_LINE, MOUNT_RIGID_LINE, MOUNTED_TEXT, material_text
from aosr.materials.construction_models import (
    AirLayer, Assembly, ConstantImpedance, MeasuredImpedance, MeasuredImpedancePoint, MountedOnWall, PorousLayer,
    SurfaceMaterial,
)
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics.fem_reduced_store import save_reduced
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.gradient_objective_jax import gradient_objective_constants
from tests.engine._material_gradient_templates_probe import kept_walls
from tests.engine import _zone_rom_case as zone_rom_case
from tests.engine._scheme_cache import shared_json
from tests.engine._legacy_gradient_search_probe import legacy_endpoint_copy
from tests.engine._material_gradient_legacy import legacy_stopping_settings
from tests.engine._six_search_cli_case import model as model, operators as operators
from tests.engine.test_material_gradient import (
    CASES, DESCENT_CASE, DIFFERENCE_STEP, PURPOSE, _chain_model, _decode, _decode_complex, _gradient_settings,
    _numpy_total, _presets, _probe_point, _receivers, _settings,
)
from tests.engine.test_material_gradient_cli import PRESET_TEXT, Record, _command, _json, _run, _sha256
from tests.engine.test_material_gradient_endpoint_cli import _inputs, _surrogate
from tests.engine.test_material_gradient_templates import KEPT, START, _surfaces
from tests.engine.test_scheme_model import _document
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL, IMPEDANCE_ABS, REFERENCE_REL
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS

from tests.engine.test_material_gradient_mount import (
    CODE_WORDS as CODE_WORDS,
    CONTROL_DIGESTS as CONTROL_DIGESTS,
    CONTROL_GRADIENT_ID as CONTROL_GRADIENT_ID,
    CONTROL_LEDGER_HEADER as CONTROL_LEDGER_HEADER,
    CONTROL_PROJECT as CONTROL_PROJECT,
    CONTROL_RESULT as CONTROL_RESULT,
    CONTROL_RESULT_KEYS as CONTROL_RESULT_KEYS,
    CONTROL_RESULT_VERSION as CONTROL_RESULT_VERSION,
    CONTROL_SURFACE_KEYS as CONTROL_SURFACE_KEYS,
    CONTROL_TEMPLATE_HEADER as CONTROL_TEMPLATE_HEADER,
    PRESETS as PRESETS,
    PROBE as PROBE,
    _cli_compute as _cli_compute,
    _close as _close,
    _context as _context,
    _control_id as _control_id,
    _control_kept as _control_kept,
    _designs as _designs,
    _digest as _digest,
    _header as _header,
    _ledger_line as _ledger_line,
    _older as _older,
    _on_wall as _on_wall,
    _probe_walls as _probe_walls,
    _project as _project,
    _result as _result,
    _rigid_result as _rigid_result,
    _scheme_walls as _scheme_walls,
    _stand_in as _stand_in,
    _template_header as _template_header,
    _template_line as _template_line,
    _whole_model as _whole_model,
    chain_model as chain_model,
    cli_runs as cli_runs,
    probe as probe,
)


def test_chain_impedance_equals_exported_material_impedance(probe: Record) -> None:
    """抓可微鏈與終點走兩種構造（鏈仍用剛背樣板、或終點匯出別的材料）：同一組 θ，鏈的樣板（``design_templates``）換進 θ 的
    垂直入射阻抗，等於終點 ``_design_material`` 那份材料經 ``construction.surface_impedance`` 的阻抗（傳遞矩陣阻抗尺；
    固定阻抗與多層構造原牆逐位）；鏈的值等於終點那幾列直接換進替身求值的值（梯度尺當前向粗檢）；
    不貼的面那一列逐位相等；鏈裡沒有剛性面；而且跟同一組剛背的鏈值不同。"""
    row = _on_wall(probe)
    chain, exported = _decode_complex(row["chain_rows"]), _decode_complex(row["endpoint_rows"])
    assert chain.shape == exported.shape == (len(Wall.all()), len(FEM_LANE_FREQUENCIES_HZ))
    kept_rows = cast(Record, row["kept_rows"])
    for index, (wall, preset) in enumerate(zip(Wall.all(), PRESETS, strict=True)):
        if preset is None:
            np.testing.assert_array_equal(_decode_complex(kept_rows[wall.wall_name()]), exported[index])
            continue
        assert np.all(np.isfinite(chain[index])), wall
        _close(chain[index], exported[index], wall.wall_name())
        if not isinstance(_probe_walls()[wall.wall_name()], MeasuredImpedance):
            # 固定阻抗與多層構造原牆兩條路逐位相同（第一支 #1001 的性質）；實測表的表內內插只到阻抗尺。
            np.testing.assert_array_equal(chain[index], exported[index], err_msg=wall.wall_name())
    mask = cast(list[bool], row["chain_rigid_mask"])
    assert mask == [preset is None for preset in PRESETS]
    rigid = cast(Record, probe[MOUNT_RIGID])
    assert row["value"] != rigid["value"]
    value, control = float.fromhex(str(row["value"])), float.fromhex(str(row["control_value"]))
    assert abs(value - control) <= GRADIENT_REL * abs(control) + GRADIENT_ABS

def test_on_wall_chain_gradient_matches_numpy_difference(chain_model: ReducedModel, probe: Record) -> None:
    """抓貼在原牆上的鏈導數斷線或值接錯構造：照 ``test_material_gradient`` 的寫法，逐參數比 θ_i·∂J/∂θ_i 對中央差分；
    差分與前向值的阻抗由考卷自己重建 ``MountedOnWall``、走既有非可微入口，替身求值與計分走 numpy 那一份。"""
    row = _on_wall(probe)
    designs = _designs(MOUNT_ON_WALL, _probe_walls())
    assert row["names"] == list(gradient_search.design_names(designs)) and row["dtype"] == "float64"
    theta = _probe_point(designs)
    mask = np.zeros(len(Wall.all()), dtype=np.bool_)
    automatic = _decode(row["gradient"])
    assert np.isfinite(automatic).all() and automatic.shape == theta.shape
    center = _numpy_total(chain_model, _decode_complex(row["center"]), mask)
    assert abs(float.fromhex(cast(str, row["value"])) - center) <= GRADIENT_REL * abs(center) + GRADIENT_ABS
    points = cast(dict[str, object], row["points"])
    for index, value in enumerate(theta):
        upper = _numpy_total(chain_model, _decode_complex(points[f"{index}.upper"]), mask)
        lower = _numpy_total(chain_model, _decode_complex(points[f"{index}.lower"]), mask)
        central = (upper - lower) / (2.0 * DIFFERENCE_STEP * value) * value
        scaled = automatic[index] * value
        assert abs(scaled - central) <= GRADIENT_REL * abs(central) + GRADIENT_ABS, (row["names"], index, scaled, central)

def test_three_lane_report_runs_on_the_exported_mounted_scheme(probe: Record) -> None:
    """抓匯出的方案在報表那一頭變回剛背或跑不起來：小方案六面原牆當背後的牆，匯出、存檔讀回，貼樣板的面是貼牆構造、
    原牆等於方案原本那面、等於匯出那一份；三路報表跑得起來：有限元素收到的 x0 阻抗就是非可微入口的值，幾何路與晚期能量有限。"""
    case = cast(Record, _on_wall(probe)["three_lane"])
    zones, walls, exported = (cast(Record, case[key]) for key in ("zones", "walls", "exported"))
    for wall, preset in zip(Wall.all(), PRESETS, strict=True):
        name = wall.wall_name()
        zone = cast(Record, zones[name])
        if preset is None:
            assert zone == walls[name], name
            continue
        assert zone["kind"] == "mounted_on_wall" and zone["wall"] == walls[name] and zone == exported[name], name
    assert case["capability"] == "assembly_by_wall" and case["x0_is_exported"] is True
    for key in ("geometric", "late"):
        values = np.asarray(cast(list[float], case[key]))
        assert values.size and np.all(np.isfinite(values)) and np.all(values >= 0), key

def test_cli_on_wall_search_records_mount_everywhere(cli_runs: tuple[Path, Record]) -> None:
    """抓 ``--mount on_wall`` 沒寫進身分：project.json 記貼在原牆上、資料夾名等於帶安裝方式重算的、不等於同一組輸入的剛背；
    帳本與換樣板帳本標頭記貼在原牆上、帳本標頭六面原牆都綁；result.json 是 v5、每面記貼在原牆上、
    貼樣板的面摘要是貼牆構造的。"""
    directory, output = cli_runs
    folder = Path(str(output["folder"]))
    project = run.read_project(folder)
    assert project.mount == MOUNT_ON_WALL and _json(folder / "project.json")["mount"] == MOUNT_ON_WALL
    assert project.objective_settings_fingerprint is not None

    def folder_name(mount: Mount) -> str:
        return run.gradient_id(
            scheme_sha256=project.scheme_sha256, surrogate_manifest_sha256=project.surrogate_manifest_sha256,
            presets=project.presets, purpose=project.purpose, settings_fingerprint=project.settings_fingerprint,
            physics_identity=project.physics_identity,
            objective_settings_fingerprint=cast(str, project.objective_settings_fingerprint), mount=mount)
    assert folder.name == project.gradient_id == folder_name(MOUNT_ON_WALL)
    assert folder.name != folder_name(MOUNT_RIGID)
    header = ledger.read_ledger(folder / "ledger.jsonl")[0]
    walls = _scheme_walls(directory)
    assert header.mount == MOUNT_ON_WALL
    assert header.kept_digests == {name: endpoint.material_digest(material) for name, material in walls.items()}
    assert templates.read_template_ledger(folder / "templates.jsonl")[0].mount == MOUNT_ON_WALL
    result = run.parse_result((folder / "result.json").read_bytes())
    assert result.format_version == run.MOUNTED_RESULT_VERSION and result.mount == MOUNT_ON_WALL
    settings = load_material_gradient_settings(directory / "settings.toml")
    by_wall: dict[Wall, SurfaceMaterial] = {wall: walls[wall.wall_name()] for wall in Wall.all()}
    designs = run.final_designs([(spec.surface_id, spec.preset_id) for spec in result.surfaces], settings, by_wall,
                                mount=MOUNT_ON_WALL)
    for spec, design in zip(result.surfaces, designs, strict=True):
        assert spec.mount == MOUNT_ON_WALL and spec.material_digest == endpoint.design_digest(design), spec.surface_id
        if design.material is not None:
            assert spec.material_digest == endpoint.material_digest(MountedOnWall(
                layers=design.material.layers, wall=walls[spec.surface_id]))

def test_cli_endpoint_exports_mounted_walls(cli_runs: tuple[Path, Record]) -> None:
    """抓終點匯出悄悄變回剛背：endpoint 回 0；``scheme-endpoint.json`` 讀回，貼樣板的面是貼牆構造、原牆等於方案原本那面、
    等於 ``materials_from_surfaces``（貼在原牆上的設計、結果檔的 θ）那一份；不貼的面是原牆本身。"""
    directory, output = cli_runs
    case = cast(Record, output["endpoint"])
    assert case["code"] == 0, case["stderr"]
    folder = Path(str(output["endpoint_folder"]))
    record = _json(folder / "endpoint.json")
    assert record['format_version'] == 'aosr.material_gradient_endpoint.v5'
    assert record['six_category'] is not None
    zones = load_scheme(folder / str(record["exported_scheme"])).scene.surface_zones_by_wall
    assert zones is not None
    result = run.parse_result((folder / "result.json").read_bytes(), legacy=True)
    assert result.format_version == run.MOUNTED_RESULT_VERSION
    walls = _scheme_walls(directory)
    settings = load_material_gradient_settings(directory / "settings.toml")
    by_wall: dict[Wall, SurfaceMaterial] = {wall: walls[wall.wall_name()] for wall in Wall.all()}
    designs = run.final_designs([(spec.surface_id, spec.preset_id) for spec in result.surfaces], settings, by_wall,
                                mount=MOUNT_ON_WALL)
    theta = np.asarray([float.fromhex(value) for value in result.theta_hex])
    materials = endpoint.materials_from_surfaces(designs, theta, settings)
    for wall, spec in zip(Wall.all(), result.surfaces, strict=True):
        material = zones[wall.wall_name()][0].material
        if spec.preset_id is None:
            assert material == walls[wall.wall_name()], wall
            continue
        assert isinstance(material, MountedOnWall) and material.wall == walls[wall.wall_name()], wall
        assert material == materials[wall], wall

def test_cli_report_says_mounted_on_wall_in_plain_words(cli_runs: tuple[Path, Record]) -> None:
    """抓報告看不出「貼在原牆上」或外露代號：安裝方式那一行是貼在原牆上；最後可行點每個貼樣板的面後面接它背後那面原牆
    （考卷自己從方案舊欄格式化）；整份報告不出現安裝方式與材料種類的代號。"""
    directory, output = cli_runs
    case = cast(Record, output["report"])
    assert case["code"] == 0, case["stderr"]
    text = str(case["stdout"])
    lines = text.splitlines()
    assert MOUNT_ON_WALL_LINE in lines and MOUNT_RIGID_LINE not in lines
    walls = _scheme_walls(directory)
    folder = Path(str(output["folder"]))
    result = run.parse_result((folder / "result.json").read_bytes())
    lined = [spec.surface_id for spec in result.surfaces if spec.preset_id is not None]
    assert lined
    for surface_id in lined:
        tail = f"，{MOUNTED_TEXT}（原牆：{material_text(walls[surface_id])}）"
        assert any(line.startswith(f"  {surface_id}：") and line.endswith(tail) for line in lines), surface_id
    assert not [word for word in CODE_WORDS if word in text]

def test_cli_resume_with_other_mount_is_rejected(cli_runs: tuple[Path, Record]) -> None:
    """抓接續不比安裝方式：貼在原牆上跑完的資料夾，project.json 拿掉安裝方式（讀成剛背）再 resume，回 1、錯誤點名
    ``mount``，帳本、換樣板帳本與結果檔一個位元組都不動。"""
    case = cast(Record, cli_runs[1]["other_mount"])
    assert case["code"] == 1, case
    assert "帳本標頭跟這一次對不上：" in str(case["stderr"]) and "'mount'" in str(case["stderr"]), case["stderr"]
    assert case["unchanged"] is True

def test_cli_rigid_result_is_mainline_v4_and_endpoint_takes_it_as_rigid(cli_runs: tuple[Path, Record]) -> None:
    """控制組（寫入端整條）：同一組輸入不給 ``--mount`` 跑完，result.json 的版本字串、頂層與每面的鍵都等於主線寫入端量的
    （沒有安裝方式那一格、不升版），檔裡沒有 ``mount``；這份 v4 交給終點（``endpoint``）回 0，終點的讀法讀成剛背。
    抓「剛背也寫 v5」與「v4 被當舊版擋在終點外」。"""
    output = cli_runs[1]
    start = cast(Record, output["rigid_start"])
    assert start["code"] == 0, start
    folder = Path(str(output["rigid_legacy_folder"]))
    data = (folder / "result.json").read_bytes()
    document = cast(Record, json.loads(data))
    assert document["format_version"] == CONTROL_RESULT_VERSION and sorted(document) == CONTROL_RESULT_KEYS
    assert all(sorted(spec) == CONTROL_SURFACE_KEYS for spec in cast(list[Record], document["surfaces"]))
    assert b"mount" not in data and "mount" not in _json(folder / "project.json")
    endpoint_case = cast(Record, output["rigid_endpoint"])
    assert endpoint_case["code"] == 0, endpoint_case["stderr"]
    assert read_result(folder).mount == MOUNT_RIGID
