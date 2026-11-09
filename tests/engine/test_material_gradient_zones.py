"""#561 第六步：每區原材料、面片序、固定與身分；整面控制組取自 53eccd56。"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from aosr.config.material_gradient import MOUNT_ON_WALL, MOUNT_RIGID
from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint, run
from aosr.gradient.fixed_surfaces import parse_fixed_surfaces
from aosr.materials.construction_models import MountedOnWall
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.scheme import Scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._surface_zone_cases import rectangle_document
from tests.engine._material_gradient_legacy import legacy_stopping_settings
from tests.engine.test_material_gradient import _settings
from tests.engine.test_material_gradient_mount import (
    _control_id, _control_kept, _designs, _digest, _project, _template_header,
)


def _scheme() -> Scheme:
    scheme = Scheme.model_validate(rectangle_document())
    zones = scheme.scene.surface_zones_by_wall
    assert zones is not None
    # 同牆各區刻意不同，抓把「不貼」區偷換成整面材料。
    changed = {name: tuple(zone.model_copy(update={"material": zone.material.model_copy(
        update={"impedance_pa_s_m": 2100.0 + 1100.0 * index})}) for index, zone in enumerate(rows))
        for name, rows in zones.items()}
    return Scheme.model_validate({**scheme.model_dump(), "scene": {**scheme.scene.model_dump(),
                                 "surface_zones_by_wall": changed}})


def test_whole_control_at_starting_commit(tmp_path: Path) -> None:
    """控制輸入與寫入端在 53eccd56 重量，與安裝方式考卷逐位相同。"""
    # 主線 53eccd56e431872407aac9160924b8dc8c48851e：施工前由各寫入端重測。
    assert _control_id() == "198075c7e78d6f5eaa13d9a1f0ab5de4"
    designs = _designs(MOUNT_RIGID)
    assert {d.surface_id: endpoint.design_digest(d) for d in designs} == {
        "floor": "64067cbc21c4ed16397af5af4a3aec65599c74d3f28ecbab0f98d4d11da0266d",
        "ceiling": "7361288944e54ec9783d5faffe5212adb09e30d1c36a91506db69ac78e8491a5",
        "x0": "64067cbc21c4ed16397af5af4a3aec65599c74d3f28ecbab0f98d4d11da0266d",
        "xL": "ed4a66abb87eb2d3d438861eae63460efbee4654b40d3baeeb6af2d46f5f1692",
        "y0": "ff2d4e05a9194967576191b229274cb10a503955fadef73ed58fed7724f78d78",
        "yL": "9aa0be661d4d1836d45bda2695460415acb4847912bb17f78b140d673c4e4fec"}
    from aosr.gradient import material_gradient_ledger as ledger
    from aosr.scoring.gradient_objective_jax import gradient_objective_constants
    from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
    from tests.engine.test_material_gradient import _gradient_settings, _receivers
    from tests.engine.test_material_gradient_mount import _stand_in
    constants = gradient_objective_constants(np.asarray(FEM_LANE_FREQUENCIES_HZ), _gradient_settings(), _receivers(), 2)
    settings = legacy_stopping_settings(_settings())
    header = ledger.header_for(_stand_in(), designs, settings, objective_constants=constants)
    # 歷史標頭只讀；新版搜尋的標頭不能冒充舊版位元組。
    line = (json.dumps(header.model_dump(mode='json'),ensure_ascii=False)+'\n').encode()
    (tmp_path / 'legacy-ledger.jsonl').write_bytes(line)
    assert ledger.read_ledger(tmp_path / 'legacy-ledger.jsonl',legacy=True)[0] == header
    # #561 更新家族設定指紋；#1032 更新計分範圍的目標常數指紋；這兩格是純標頭序列化，沒有重錄求解答案。
    assert _digest(line) == "1e8c6edf1569caa3e3c4e8e8bb310d0d59c7f194216689f35a0c9b6160d0c4df"
    assert _digest((json.dumps(_template_header(header, _control_kept(), settings).model_dump(mode="json"),ensure_ascii=False)+"\n").encode()) == "fcaed69bbd2a1076cd417de2e96a9cee779d644b1f101a94f7516b39e065afe5"
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, _project())
    assert _digest((tmp_path / run.PROJECT_FILE).read_bytes()) == "d51fbff4611f4915537fc6df82b5de4c1822dcb1b3c334c6245c9eb4f5979485"


def test_each_zone_keeps_its_own_material_and_backing() -> None:
    scheme = _scheme()
    kept = run.scene_wall_materials(scheme, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    designs = run.designs_for((None,) * 21, _settings(), kept=kept)
    assert tuple(d.surface_id for d in designs[:5]) == (
        "floor", "ceiling@r0_c0", "ceiling@r0_c1", "ceiling@r1_c0", "ceiling@r1_c1")
    zones = scheme.scene.surface_zones_by_wall
    assert zones is not None
    for design in designs:
        wall, _, zone_id = design.surface_id.partition("@")
        original = next(z for z in zones[wall] if not isinstance(z, RectangleZone) or z.id == zone_id)
        assert design.kept_material == original.material
    mounted = run.designs_for((None, "glass_wool_48k_50mm", *([None] * 19)), _settings(), kept=kept, mount=MOUNT_ON_WALL)
    materials = endpoint.materials_from_surfaces(mounted, np.asarray([0.05]), _settings())
    material = materials[(Wall.CEILING, "r0_c0")]
    assert isinstance(material, MountedOnWall) and material.wall == zones["ceiling"][0].material


def test_fixed_wall_expands_and_fixed_patch_is_local() -> None:
    kept = run.scene_wall_materials(_scheme(), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    for target, count in (("ceiling", 4), ("ceiling@r0_c1", 1)):
        fixed = parse_fixed_surfaces([f"{target}=carpet_on_concrete"])
        designs = run.designs_for((None,) * 21, _settings(), kept=kept, fixed_surfaces=fixed)
        actual = [d.surface_id for d in designs if d.fixed_material_id is not None]
        assert len(actual) == count and (target in actual if count == 1 else all(n.startswith("ceiling@") for n in actual))
    with pytest.raises(ValueError, match="同一面"):
        parse_fixed_surfaces(["ceiling=carpet_on_concrete", "ceiling@r0_c1=carpet_on_concrete"])


def test_zone_identity_and_export_bind_ids_and_order(tmp_path: Path) -> None:
    from aosr.gradient.boundary import scheme_patch_keys
    from tests.engine.test_material_gradient_mount import _control_id

    scheme = _scheme()
    keys = scheme_patch_keys(scheme)
    def identity(patches: tuple[tuple[Wall, str], ...]) -> str:
        return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64,
            presets=(None,) * 21, purpose="music", settings_fingerprint="c" * 64,
            physics_identity="p", objective_settings_fingerprint="d" * 64, patch_keys=patches)
    first = identity(keys)
    assert first == identity(scheme_patch_keys(_scheme()))
    renamed = tuple((w, "renamed" if i == 1 else z) for i, (w, z) in enumerate(keys))
    reordered = (keys[0], keys[2], keys[1], *keys[3:])
    assert first != identity(renamed)
    assert first != identity(reordered)
    assert _control_id() == "198075c7e78d6f5eaa13d9a1f0ab5de4"
    kept = run.scene_wall_materials(scheme, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    designs = run.designs_for((None,) * 21, _settings(), kept=kept)
    materials = endpoint.materials_from_surfaces(designs, np.asarray([]), _settings())
    scene = endpoint.scene_with_materials(scheme.scene, materials)
    exported = Scheme.model_validate({**scheme.model_dump(), "scene": scene.model_dump()})
    assert exported.scene.surface_zones_by_wall == scheme.scene.surface_zones_by_wall


@pytest.mark.parametrize("surface_id", ("x0", "x0@right"))
@pytest.mark.parametrize("valid", (False, True), ids=("missing-dot", "qualified"))
def test_template_row_requires_dot_after_surface_id(surface_id: str, valid: bool) -> None:
    """抓帳本把沒有「.變數」的整面或分區名字算成合法變數。"""
    from aosr.gradient.template_ledger import TemplateRow

    document = {"step": 0, "phase": "baseline_retune", "round": 0, "candidate": None,
        "surfaces": ((surface_id, "glass_wool_48k_50mm"),),
        "variable_names": (f"{surface_id}.layer1.thickness_m" if valid else surface_id,),
        "theta_hex": (0.05.hex(),), "value": 1.0, "screening_total": 1.0, "components": (1.0, 0.0, 0.0),
        "gradient_norm": 0.0, "step_size": 0.0, "seconds": 0.0, "stop_reason": None,
        "final_trial_step": None, "evaluations": None, "decision": None, "swapped_retunable": None}
    if valid:
        row = TemplateRow.model_validate(document)
        assert row.variable_names == document["variable_names"]
    else:
        with pytest.raises(ValueError, match="變數個數 0 不合"):
            TemplateRow.model_validate(document)


def _zone_probe(directory: Path, part: str) -> None:
    """各段獨立雙精度子行程，沿用三格處理器釘選與原先的 55 秒上限。"""
    import os
    import subprocess
    import sys

    from aosr.runtime import child_process_env, configure_jax
    from tests.engine.test_fem_complex import _CPU_PIN_VARS

    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    probe = Path(__file__).with_name("_material_gradient_zones_probe.py")
    result = subprocess.run([sys.executable, str(probe), str(directory), part], env=env, cwd=directory,
                             capture_output=True, text=True, timeout=55)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (directory / f"{part}.json").exists()


@pytest.fixture(scope="module")
def zone_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """只建一次小房間替身；存檔由各子行程讀回，不共用編譯或求值結果。"""
    directory = tmp_path_factory.mktemp("gradient-zones")
    _zone_probe(directory, "build")
    return directory


@pytest.fixture(scope="module")
def zone_search(zone_model: Path) -> Path:
    """短搜尋及樣板輪只跑一次，終點從這一跑的帳本讀取最後配置。"""
    _zone_probe(zone_model, "search")
    return zone_model


def test_partitioned_chain_and_central_difference(zone_model: Path) -> None:
    """抓鏈未按片求阻抗、梯度斷線與面片清單不等仍通過。"""
    _zone_probe(zone_model, "chain")


def test_partitioned_search_and_templates(zone_search: Path) -> None:
    """抓帳本缺片、目標不等於獨立替身參考、固定片進樣板輪。"""
    assert (zone_search / "search.json").exists()


def test_partitioned_full_endpoint_export_and_report(zone_search: Path) -> None:
    """抓完整終點與逐片材料不符、匯出遺失座標、報告缺分區或固定標記。"""
    _zone_probe(zone_search, "endpoint")


def test_fixed_patches_validate_coverage_and_leave_one_searchable() -> None:
    from aosr.gradient import cli

    kept = run.scene_wall_materials(_scheme(), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    designs = run.designs_for((None,) * 21, _settings(), kept=kept)
    assert cli._presets(",".join(["none"] * 21)) == (None,) * 21
    fixed = {d.surface_id: "carpet_on_concrete" for d in designs}
    with pytest.raises(ValueError, match="全部面片固定"):
        run.designs_for((None,) * 21, _settings(), kept=kept, fixed_surfaces=fixed)
    with pytest.raises(ValueError, match="不在搜尋面片清單"):
        run.designs_for((None,) * 21, _settings(), kept=kept, fixed_surfaces={"ceiling@absent": "carpet_on_concrete"})
