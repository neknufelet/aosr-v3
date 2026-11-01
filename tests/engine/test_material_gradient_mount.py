"""安裝方式「貼在原牆上」接進第三階段梯度搜尋（#990 第二支）：身分、存檔、換樣板、可微鏈、終點匯出、命令列。

``mount``（安裝方式）每次搜尋一個值：``rigid`` 是原本的行為，``on_wall`` 是貼樣板的面新增層背後接那一面方案原本的牆。
控制組：剛背的資料夾名、帳本與換樣板帳本標頭、``project.json``、每面材料摘要，都逐位等於改動前主線（692d1f42）從
寫入端量的值（同一組固定輸入，量法見各常數旁）。可微鏈與差分在雙精度子行程（``_material_gradient_mount_probe``），
替身用 ``test_material_gradient`` 的衍生替身；差分與 numpy 參考照那一份的寫法（``_numpy_total``），尺從傳遞矩陣考卷 import。
命令列整條（start --mount on_wall、endpoint、report、改 project.json 的安裝方式再接續）照 ``test_material_gradient_endpoint_cli``
那一份替身、方案與縮小步數的設定開真的子行程跑。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.config.material_gradient import MOUNT_ON_WALL, MOUNT_RIGID, Mount, load_material_gradient_settings
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, endpoint, run
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
from aosr.physics.fem_reduced_store import save_reduced
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.gradient_objective_jax import gradient_objective_constants
from tests.engine._material_gradient_templates_probe import kept_walls
from tests.engine._scheme_cache import shared_json
from tests.engine.test_fem_reduced import model as model, operators as operators
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


PROBE = Path(__file__).with_name("_material_gradient_mount_probe.py")
PRESETS: tuple[str | None, ...] = CASES[DESCENT_CASE]
# 控制組的固定輸入（改 src 之前在主線 692d1f42 用同一組輸入跑寫入端量的；量的片段只叫產品入口）：
# gradient_id 的七格、六面原牆（固定阻抗 12348＋面序）、替身身分三格的替身、正式設定與正式第一階段設定、兩支喇叭。
CONTROL_GRADIENT_ID = "198075c7e78d6f5eaa13d9a1f0ab5de4"
# 每面 endpoint.design_digest（照 Wall.all()）。
CONTROL_DIGESTS = {
    "floor": "64067cbc21c4ed16397af5af4a3aec65599c74d3f28ecbab0f98d4d11da0266d",
    "ceiling": "7361288944e54ec9783d5faffe5212adb09e30d1c36a91506db69ac78e8491a5",
    "x0": "64067cbc21c4ed16397af5af4a3aec65599c74d3f28ecbab0f98d4d11da0266d",
    "xL": "ed4a66abb87eb2d3d438861eae63460efbee4654b40d3baeeb6af2d46f5f1692",
    "y0": "ff2d4e05a9194967576191b229274cb10a503955fadef73ed58fed7724f78d78",
    "yL": "9aa0be661d4d1836d45bda2695460415acb4847912bb17f78b140d673c4e4fec"}
# 帳本標頭那一列、換樣板帳本標頭那一列（庚二帳本摘要給 "e"×64）、project.json 整檔的 SHA-256。
CONTROL_LEDGER_HEADER = "a417256b0f40af5ab6703348c833bfb468d1e19593a3079061df943a849c2fe5"
CONTROL_TEMPLATE_HEADER = "304b40fe88bb01f381ebce0962f3c0c3cf544151bb63f018769cf0bfe21ece3c"
CONTROL_PROJECT = "d51fbff4611f4915537fc6df82b5de4c1822dcb1b3c334c6245c9eb4f5979485"
CODE_WORDS = ("on_wall", "rigid", "mounted_on_wall", "constant_impedance", "assembly")


# ---- 固定輸入 ----

def _control_kept() -> dict[str, SurfaceMaterial]:
    return {wall.wall_name(): ConstantImpedance(impedance_pa_s_m=12348.0 + index) for index, wall in enumerate(Wall.all())}


def _designs(mount: Mount, kept: dict[str, SurfaceMaterial] | None = None) -> tuple[gradient_search.SurfaceDesign, ...]:
    return gradient_search.wall_designs_from_presets(PRESETS, _settings(), _presets(), kept=kept or _control_kept(),
                                                     mount=mount)


def _control_id(mount: Mount | None = None) -> str:
    """控制組那七格輸入的資料夾名；``mount`` 不給就是呼叫端不傳這一格（預設）。"""
    if mount is None:
        return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=PRESETS,
                               purpose=PURPOSE, settings_fingerprint="c" * 64, physics_identity="physics-x",
                               objective_settings_fingerprint="d" * 64)
    return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=PRESETS,
                           purpose=PURPOSE, settings_fingerprint="c" * 64, physics_identity="physics-x",
                           objective_settings_fingerprint="d" * 64, mount=mount)


def _stand_in() -> ReducedModel:
    """標頭只讀替身身分三格，用一份只有這三格的替身（不建替身）。"""
    return cast(ReducedModel, SimpleNamespace(input_hash="input-x", mesh_hash="mesh-x", physics_identity="physics-x"))


def _header(designs: tuple[gradient_search.SurfaceDesign, ...]) -> ledger.MaterialGradientHeader:
    constants = gradient_objective_constants(np.asarray(FEM_LANE_FREQUENCIES_HZ), _gradient_settings(), _receivers(), 2)
    return ledger.header_for(_stand_in(), designs, _settings(), objective_constants=constants)


def _template_header(header: ledger.MaterialGradientHeader, kept: dict[str, SurfaceMaterial]) -> templates.TemplateHeader:
    return run.template_header_for(header, _settings(), {wall: kept[wall.wall_name()] for wall in Wall.all()},
                                   base_ledger_sha256="e" * 64)


def _project(mount: Mount = MOUNT_RIGID) -> run.GradientProject:
    return run.GradientProject(
        format_version=run.PROJECT_VERSION, gradient_id=CONTROL_GRADIENT_ID, scheme_path="/s", scheme_sha256="a" * 64,
        surrogate_path="/m", surrogate_manifest_sha256="b" * 64, receipt_path="/r", receipt_sha256="f" * 64,
        presets=PRESETS, purpose=PURPOSE, settings_path="/t", settings_fingerprint="c" * 64, capabilities_path="/c",
        physics_identity="physics-x", mesh_hash="mesh-x", source_points={}, receiver_points={}, engine_commit=None,
        started_at="2026-11-01T00:00:00+00:00", objective_settings_fingerprint="d" * 64, mount=mount)


def _ledger_line(path: Path, header: ledger.MaterialGradientHeader) -> bytes:
    """寫入端：``write_ledger`` 獨佔建檔、只有標頭，回第一列的位元組。"""
    ledger.write_ledger(path, header, ())
    return path.read_bytes().splitlines(keepends=True)[0]


def _template_line(path: Path, header: templates.TemplateHeader) -> bytes:
    templates.create_template_ledger(path, header)
    return path.read_bytes().splitlines(keepends=True)[0]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- 控制組：剛背逐位不變 ----

def test_rigid_identities_equal_mainline_bit_for_bit(tmp_path: Path) -> None:
    """抓剛背的身分被這一刀改到（資料夾名永遠加 mount、標頭或 project.json 永遠寫安裝方式、摘要換算法）：同一組固定輸入，
    資料夾名、每面材料摘要、帳本標頭列、換樣板帳本標頭列、project.json 整檔，都逐位等於改動前主線從寫入端量的值；
    不給安裝方式（預設）與明給剛背一樣。"""
    assert _control_id() == _control_id(MOUNT_RIGID) == CONTROL_GRADIENT_ID
    designs = _designs(MOUNT_RIGID)
    assert designs == gradient_search.wall_designs_from_presets(PRESETS, _settings(), _presets(), kept=_control_kept())
    assert {design.surface_id: endpoint.design_digest(design) for design in designs} == CONTROL_DIGESTS
    header = _header(designs)
    line = _ledger_line(tmp_path / "ledger.jsonl", header)
    assert _digest(line) == CONTROL_LEDGER_HEADER and b"mount" not in line
    template_line = _template_line(tmp_path / "templates.jsonl", _template_header(header, _control_kept()))
    assert _digest(template_line) == CONTROL_TEMPLATE_HEADER and b"mount" not in template_line
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, _project())
    assert _digest((tmp_path / run.PROJECT_FILE).read_bytes()) == CONTROL_PROJECT


# ---- 身分：貼在原牆上是另一個身分，六面原牆都綁 ----

def test_on_wall_gets_its_own_folder_digests_and_headers(tmp_path: Path) -> None:
    """抓新方式的分數沿用舊的：貼在原牆上的資料夾名不同於剛背；貼樣板的面材料摘要是貼牆構造（層＋原牆）的、不等於剛背；
    不貼的面摘要不變；帳本標頭記安裝方式，``kept_digests`` 六面都有、貼樣板的面是它背後那面原牆的摘要；換樣板帳本標頭抄安裝方式。"""
    assert _control_id(MOUNT_ON_WALL) != CONTROL_GRADIENT_ID
    kept = _control_kept()
    designs = _designs(MOUNT_ON_WALL)
    for design, preset in zip(designs, PRESETS, strict=True):
        digest = endpoint.design_digest(design)
        if preset is None:
            assert design.backing_material is None and digest == CONTROL_DIGESTS[design.surface_id]
            continue
        assert design.material is not None and design.backing_material == kept[design.surface_id]
        assert digest == endpoint.material_digest(MountedOnWall(layers=design.material.layers,
                                                                wall=design.backing_material))
        assert digest != CONTROL_DIGESTS[design.surface_id]
    header = _header(designs)
    assert header.mount == MOUNT_ON_WALL
    assert header.kept_digests == {name: endpoint.material_digest(material) for name, material in kept.items()}
    assert set(header.kept_digests) == {wall.wall_name() for wall in Wall.all()}
    line = _ledger_line(tmp_path / "ledger.jsonl", header)
    assert b'"mount":"on_wall"' in line.replace(b" ", b"")
    template = _template_header(header, kept)
    assert template.mount == MOUNT_ON_WALL
    assert b'"mount":"on_wall"' in _template_line(tmp_path / "templates.jsonl", template).replace(b" ", b"")


def test_on_wall_header_binds_the_wall_behind_each_template() -> None:
    """抓 ``kept_digests`` 漏了貼樣板面的原牆：只把一面貼樣板的面（地板）的原牆加倍，貼在原牆上的標頭只差地板那一格摘要
    （接續會拒收）；剛背的標頭不看那一面（跟改動前一樣）。"""
    floor = Wall.all()[0].wall_name()
    assert PRESETS[0] is not None
    changed = {**_control_kept(), floor: ConstantImpedance(impedance_pa_s_m=2 * 12348.0)}
    same, other = _header(_designs(MOUNT_ON_WALL)), _header(_designs(MOUNT_ON_WALL, changed))
    differing = {name for name in ledger.MaterialGradientHeader.model_fields if getattr(same, name) != getattr(other, name)}
    assert differing == {"kept_digests"}
    assert {name for name in same.kept_digests if same.kept_digests[name] != other.kept_digests[name]} == {floor}
    assert _header(_designs(MOUNT_RIGID)) == _header(_designs(MOUNT_RIGID, changed))


# ---- 設計的收拒 ----

def test_surface_design_takes_the_wall_only_when_mounted_with_a_template() -> None:
    """抓原牆在不該收的地方被收下、或貼在原牆上卻沒有原牆：剛背帶原牆、不貼的面帶原牆、貼在原牆上沒有原牆、
    原牆缺那一面、原牆本身已是貼牆構造、不認識的安裝方式，一律拒收。"""
    rigid = _designs(MOUNT_RIGID)
    lined, plain = rigid[0], rigid[1]
    assert lined.material is not None and plain.material is None and plain.kept_material is not None
    wall = ConstantImpedance(impedance_pa_s_m=12348.0)
    with pytest.raises(ValueError, match="只有貼在原牆上"):
        gradient_search.SurfaceDesign(**{**vars(lined), "backing_material": wall})
    with pytest.raises(ValueError, match="只有貼在原牆上"):
        gradient_search.SurfaceDesign(**{**vars(plain), "mount": MOUNT_ON_WALL, "backing_material": wall})
    with pytest.raises(ValueError, match="要給那一面原本的牆"):
        gradient_search.SurfaceDesign(**{**vars(lined), "mount": MOUNT_ON_WALL})
    with pytest.raises(ValueError, match="安裝方式只收"):
        gradient_search.SurfaceDesign(**{**vars(lined), "mount": "glued"})
    missing = {name: material for name, material in _control_kept().items() if name != lined.surface_id}
    with pytest.raises(ValueError, match="方案原本的牆的材料清單沒有這一面"):
        _designs(MOUNT_ON_WALL, missing)
    with pytest.raises(ValueError, match="方案原本的牆的材料清單沒有這一面"):
        gradient_search.wall_designs_from_presets(PRESETS, _settings(), _presets(), mount=MOUNT_ON_WALL)
    assert lined.material is not None
    nested = {**_control_kept(), lined.surface_id: MountedOnWall(layers=lined.material.layers, wall=wall)}
    with pytest.raises(ValueError, match="已經是貼在原牆上的構造"):
        _designs(MOUNT_ON_WALL, nested)
    with pytest.raises(ValueError, match="一次搜尋只有一種安裝方式"):
        gradient_search.designs_mount((*_designs(MOUNT_ON_WALL)[:3], *rigid[3:]))


# ---- 讀回：缺安裝方式讀成剛背；安裝方式不同拒收接續 ----

def test_missing_mount_reads_as_rigid_and_resumes(tmp_path: Path) -> None:
    """抓舊資料夾讀不進或被當成別的安裝方式：沒有 ``mount`` 那一格的 project.json（改動前寫的，位元組同控制組）讀成剛背；
    沒有那一格的帳本與換樣板帳本標頭（改動前寫的那一列）讀回時等於這一次重建的剛背標頭（照收接續）；
    貼在原牆上的 project.json 寫出再讀回仍是貼在原牆上。"""
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, _project())
    document = _json(tmp_path / run.PROJECT_FILE)
    assert "mount" not in document and run.read_project(tmp_path).mount == MOUNT_RIGID
    header = _header(_designs(MOUNT_RIGID))
    line = _ledger_line(tmp_path / "ledger.jsonl", header)
    assert "mount" not in json.loads(line)
    assert ledger.read_ledger(tmp_path / "ledger.jsonl", expected_header=header)[0] == header
    template = _template_header(header, _control_kept())
    _template_line(tmp_path / "templates.jsonl", template)
    assert templates.read_template_ledger(tmp_path / "templates.jsonl", expected_header=template)[0].mount == MOUNT_RIGID
    on_wall = tmp_path / "on-wall"
    on_wall.mkdir()
    run.write_json_atomic(on_wall / run.PROJECT_FILE, _project(MOUNT_ON_WALL))
    assert _json(on_wall / run.PROJECT_FILE)["mount"] == MOUNT_ON_WALL
    assert run.read_project(on_wall).mount == MOUNT_ON_WALL


@pytest.mark.parametrize(("written", "expected"), ((MOUNT_ON_WALL, MOUNT_RIGID), (MOUNT_RIGID, MOUNT_ON_WALL)))
def test_ledgers_of_other_mount_are_rejected(tmp_path: Path, written: Mount, expected: Mount) -> None:
    """抓接續不比安裝方式：帳本與換樣板帳本以一種安裝方式寫，拿另一種重建的標頭讀，拒收、訊息點名 ``mount``。"""
    kept = _control_kept()
    header, other = _header(_designs(written)), _header(_designs(expected))
    _ledger_line(tmp_path / "ledger.jsonl", header)
    with pytest.raises(ValueError, match="帳本標頭跟這一次對不上：.*'mount'"):
        ledger.read_ledger(tmp_path / "ledger.jsonl", expected_header=other)
    _template_line(tmp_path / "templates.jsonl", _template_header(header, kept))
    with pytest.raises(ValueError, match="換樣板帳本標頭跟這一次對不上：.*'mount'"):
        templates.read_template_ledger(tmp_path / "templates.jsonl", expected_header=_template_header(other, kept))


# ---- result.json v5 ----

def _result(mount: Mount) -> run.GradientResult:
    specs = tuple(run.SurfaceSpec(surface_id=wall.wall_name(), preset_id=preset, thickness_m=None if preset is None else 0.05,
                                  cavity_m=None, material_digest=CONTROL_DIGESTS[wall.wall_name()], mount=mount)
                  for wall, preset in zip(Wall.all(), PRESETS, strict=True))
    return run.GradientResult(
        format_version=run.RESULT_VERSION, gradient_id=CONTROL_GRADIENT_ID, surfaces=specs, variable_names=("floor.layer1.thickness_m",),
        theta_hex=((0.05).hex(),), value=1.5, screening_total=1.25, components=run.Components(
            own_line_ripple_rms_db=1.0, seat_robustness_rms_db=2.0, big_peak_proxy_db=3.0),
        round_stop_reason="沒有候選", stop_reason="沒有可換的樣板", evaluations=3, rounds=1, template_rounds=1,
        template_evaluations=0, ledger_sha256="1" * 64, template_ledger_sha256="2" * 64)


def _older(document: Record, version: str, *, proxy: bool) -> bytes:
    """考卷自己把 v5 換回舊形狀：每面拿掉安裝方式；v3 另拿掉舊合分與代理。"""
    older = json.loads(json.dumps(document))
    older["format_version"] = version
    for spec in cast(list[Record], older["surfaces"]):
        spec.pop("mount")
    if not proxy:
        older.pop("screening_total")
        cast(Record, older["components"]).pop("big_peak_proxy_db")
    return json.dumps(older).encode("utf-8")


@pytest.mark.parametrize("mount", (MOUNT_RIGID, MOUNT_ON_WALL))
def test_result_v5_reads_back_and_older_versions_are_legacy(mount: Mount) -> None:
    """抓 v5 讀不回、舊版被當新版收（終點與接續吃到沒有安裝方式的結果）或報告讀不回舊版：v5 每面帶安裝方式、讀回整個相等；
    v4 與 v3 交給終點的讀法拒收、訊息講版本，報告的讀法（``legacy``）讀得回、版本標成舊版、每面沒有安裝方式；
    v5 少了安裝方式、v4 多了安裝方式都拒收。"""
    result = _result(mount)
    document = result.model_dump(mode="json")
    assert all(spec["mount"] == mount for spec in cast(list[Record], document["surfaces"]))
    assert run.parse_result(json.dumps(document).encode("utf-8")) == result
    for version, proxy in ((run.PRE_MOUNT_RESULT_VERSION, True), (run.LEGACY_RESULT_VERSION, False)):
        content = _older(document, version, proxy=proxy)
        with pytest.raises(ValueError, match=version.replace(".", r"\.")):
            run.parse_result(content)
        legacy = run.parse_result(content, legacy=True)
        assert legacy.format_version == version and all(spec.mount is None for spec in legacy.surfaces)
        assert legacy.surfaces == tuple(spec.model_copy(update={"mount": None}) for spec in result.surfaces)
    unmounted = json.loads(_older(document, run.RESULT_VERSION, proxy=True))
    with pytest.raises(ValidationError, match="每面要帶安裝方式"):
        run.parse_result(json.dumps(unmounted).encode("utf-8"))
    with pytest.raises(ValidationError, match="每面要帶安裝方式"):
        run.parse_result(json.dumps({**document, "format_version": run.PRE_MOUNT_RESULT_VERSION}).encode("utf-8"),
                         legacy=True)


# ---- 換樣板：沿用這一次搜尋的安裝方式 ----

def _context(mount: Mount) -> rounds.TemplateContext:
    def never(_: tuple[gradient_search.SurfaceDesign, ...]) -> gradient_search.ChainObjective:
        raise AssertionError("這一題不該建目標")
    return rounds.TemplateContext(_settings(), _presets(), kept_walls(KEPT), never, mount)


@pytest.mark.parametrize("mount", (MOUNT_RIGID, MOUNT_ON_WALL))
def test_swap_keeps_the_mount_and_the_wall_behind(mount: Mount) -> None:
    """抓換樣板時安裝方式掉回剛背、或換過去的樣板背後接錯牆：從起點配置換一面到別的家族、把不貼的面換成家族、
    把貼的面換成不貼，換完每面安裝方式都等於這一次搜尋的；貼在原牆上時換過去的樣板背後是那一面原牆，剛背沒有原牆。"""
    context = _context(mount)
    designs = rounds.designs_for_surfaces(_surfaces(START), context)
    theta = gradient_search.snap_to_specs(gradient_search.baseline_theta(designs), designs).theta
    kept = kept_walls(KEPT)
    for surface_id, target in (("x0", "glass_wool_24k"), ("floor", "glass_wool_48k"), ("ceiling", "none")):
        swapped, _ = rounds.swap(designs, np.asarray(theta), surface_id, target, context)
        assert {design.mount for design in swapped} == {mount}, surface_id
        design = next(item for item in swapped if item.surface_id == surface_id)
        if target == "none":
            assert design.material is None and design.backing_material is None and design.kept_material == kept[surface_id]
            continue
        assert design.material is not None and design.preset_id == rounds.preset_for_target(target, _settings())
        assert design.backing_material == (kept[surface_id] if mount == MOUNT_ON_WALL else None), surface_id


# ---- 終點匯出：貼樣板的面寫成貼牆構造，原牆是方案那一面 ----

def test_exported_materials_are_mounted_on_the_scheme_wall() -> None:
    """抓匯出時悄悄變回剛背：貼在原牆上的設計匯出、寫進方案、存成 JSON 讀回，貼樣板的面是貼牆構造、原牆等於那一面原本的牆、
    層的厚度與空腔照 θ；不貼的面是原牆本身。剛背同一組照舊是剛背多層構造。"""
    kept = _control_kept()
    designs = _designs(MOUNT_ON_WALL)
    theta = _probe_point(designs)
    materials = endpoint.materials_from_surfaces(designs, theta, _settings())
    scheme = Scheme.model_validate(_document())
    document = scheme.model_dump(mode="json")
    document["scene"] = endpoint.scene_with_materials(scheme.scene, materials).model_dump(mode="json")
    zones = Scheme.model_validate(json.loads(json.dumps(document))).scene.surface_zones_by_wall
    assert zones is not None
    rigid = endpoint.materials_from_surfaces(_designs(MOUNT_RIGID), theta, _settings())
    for wall, preset in zip(Wall.all(), PRESETS, strict=True):
        name, material = wall.wall_name(), zones[wall.wall_name()][0].material
        if preset is None:
            assert material == kept[name], name
            continue
        assert isinstance(material, MountedOnWall) and material.wall == kept[name], name
        reference = rigid[wall]
        assert isinstance(reference, Assembly) and reference.backing == "rigid"
        assert material.layers == reference.layers, name


# ---- 命令列參數 ----

def test_cli_mount_defaults_to_rigid_and_takes_on_wall() -> None:
    """抓 ``--mount`` 沒接上或預設不是剛背：不給是剛背、給 on_wall 收下、其他值是參數錯（離開碼 2）。"""
    base = ["start", "--scheme", "s.json", "--surrogate", "m.npz", "--presets", PRESET_TEXT, "--purpose", PURPOSE,
            "--root", "root"]
    assert cli._parser().parse_args(base).mount == MOUNT_RIGID
    assert cli._parser().parse_args([*base, "--mount", MOUNT_ON_WALL]).mount == MOUNT_ON_WALL
    with pytest.raises(SystemExit) as raised:
        cli._parser().parse_args([*base, "--mount", "glued"])
    assert raised.value.code == 2


# ---- 可微鏈：貼在原牆上的阻抗與導數（雙精度子行程） ----

def _probe_walls() -> dict[str, SurfaceMaterial]:
    """三種原牆都上場：貼樣板的地板固定阻抗、x0 多層構造、xL 實測表（50–500 Hz，表外底構造補）、yL 另一個固定阻抗；
    不貼的兩面固定阻抗。"""
    base = Assembly(layers=(PorousLayer(thickness_m=.1, flow_resistivity_pa_s_m2=10000., material_class="fibrous"),),
                    backing="rigid")
    measured = MeasuredImpedance(points=(
        MeasuredImpedancePoint(frequency_hz=50., real_pa_s_m=3000., imag_pa_s_m=-2000.),
        MeasuredImpedancePoint(frequency_hz=200., real_pa_s_m=2500., imag_pa_s_m=-500.),
        MeasuredImpedancePoint(frequency_hz=500., real_pa_s_m=2000., imag_pa_s_m=-100.)),
        base_assembly=base, source_and_conditions="考卷假表：阻抗管、垂直入射、樣品 100 mm")
    assembly = Assembly(layers=(PorousLayer(thickness_m=.03, flow_resistivity_pa_s_m2=15000., material_class="fibrous"),
                                AirLayer(thickness_m=.2)), backing="rigid")
    walls: list[SurfaceMaterial] = [ConstantImpedance(impedance_pa_s_m=12348.), ConstantImpedance(impedance_pa_s_m=15000.),
                                    assembly, measured, ConstantImpedance(impedance_pa_s_m=9000.),
                                    ConstantImpedance(impedance_pa_s_m=20000.)]
    return {wall.wall_name(): material for wall, material in zip(Wall.all(), walls, strict=True)}


@pytest.fixture(scope="module")
def chain_model(model: ReducedModel) -> ReducedModel:
    return _chain_model(model)


@pytest.fixture(scope="module")
def probe(chain_model: ReducedModel, tmp_path_factory: pytest.TempPathFactory) -> Record:
    directory = tmp_path_factory.mktemp("material-gradient-mount")
    save_reduced(chain_model, directory / "model.npz")
    designs = _designs(MOUNT_ON_WALL, _probe_walls())
    problem = {"presets": list(PRESETS), "theta": [value.hex() for value in _probe_point(designs)],
               "step": DIFFERENCE_STEP, "purpose": PURPOSE, "receivers": _receivers().model_dump(mode="json"),
               "kept": {name: material.model_dump(mode="json") for name, material in _probe_walls().items()},
               "scheme": _document()}
    (directory / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    process = subprocess.run([sys.executable, str(PROBE), str(directory)], cwd=directory, env=env,
                             capture_output=True, text=True, timeout=900)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(Record, json.loads((directory / "output.json").read_text(encoding="utf-8")))


def _on_wall(probe: Record) -> Record:
    return cast(Record, probe[MOUNT_ON_WALL])


def _close(ours: NDArray[np.complex128], reference: NDArray[np.complex128], label: str) -> None:
    """傳遞矩陣移植考卷那把阻抗尺（實虛部各自相對加絕對），同 ``test_mounted_on_wall._close``。"""
    for part in (np.real, np.imag):
        np.testing.assert_allclose(part(ours), part(reference), rtol=REFERENCE_REL, atol=IMPEDANCE_ABS, err_msg=label)


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


# ---- 命令列整條：start --mount on_wall、endpoint、report、改安裝方式再接續 ----

def _cli_compute(directory: Path, request: pytest.FixtureRequest) -> str:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    _surrogate(directory, request)
    _inputs(directory)
    root = directory / "root"
    start = _command("start", "--scheme", str(directory / "scheme.json"), "--surrogate",
                     str(directory / "good" / "rom.npz"), "--presets", PRESET_TEXT, "--purpose", PURPOSE,
                     "--root", str(root), "--settings", str(directory / "settings.toml"), "--engine-commit", "test",
                     "--mount", MOUNT_ON_WALL)
    output: Record = {"start": _run(directory, start)}
    assert cast(Record, output["start"])["code"] == 0, output["start"]
    (folder,) = (path for path in root.iterdir() if not path.name.startswith("."))
    output["folder"] = str(folder)
    output["endpoint"] = _run(directory, _command("endpoint", str(folder)))
    output["report"] = _run(directory, _command("report", "--folder", str(folder)))
    copy = directory / "other-mount" / folder.name
    shutil.copytree(folder, copy)
    project = _json(copy / "project.json")
    project.pop("mount")
    (copy / "project.json").write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
    kept = ("ledger.jsonl", "templates.jsonl", "result.json")
    before = {name: _sha256(copy / name) for name in kept}
    output["other_mount"] = {**_run(directory, _command("resume", str(copy))),
                             "unchanged": before == {name: _sha256(copy / name) for name in kept}}
    return json.dumps(output, ensure_ascii=False)


@pytest.fixture(scope="module")
def cli_runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
             ) -> tuple[Path, Record]:
    """一場 pytest 只跑一次，平行的工人共用；讀的人只讀。"""
    directory = (tmp_path_factory.mktemp("material-gradient-mount-cli") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "material-gradient-mount-cli-shared")
    text = shared_json(tmp_path_factory, worker_id, "material-gradient-mount-cli",
                       lambda: _cli_compute(directory, request))
    return directory, cast(Record, json.loads(text))


def _scheme_walls(directory: Path) -> dict[str, ConstantImpedance]:
    """考卷方案舊欄的六面固定阻抗，自己換成 ConstantImpedance（不走產品的 scene_wall_materials）。"""
    fixed = load_scheme(directory / "scheme.json").scene.impedance_pa_s_per_m_by_wall
    assert fixed is not None
    return {name: ConstantImpedance(impedance_pa_s_m=value) for name, value in fixed.items()}


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
    assert result.format_version == run.RESULT_VERSION
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
    folder = Path(str(output["folder"]))
    record = _json(folder / "endpoint.json")
    zones = load_scheme(folder / str(record["exported_scheme"])).scene.surface_zones_by_wall
    assert zones is not None
    result = run.parse_result((folder / "result.json").read_bytes())
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


def test_report_material_text_of_walls_has_no_codes() -> None:
    """控制組：報告印原牆那一句用的 ``material_text``，三種原牆都不出現代號（貼牆那一行靠它）。"""
    for material in _probe_walls().values():
        text = material_text(material)
        assert text and not [word for word in CODE_WORDS if word in text], text

