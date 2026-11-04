"""#561 第六步：每區原材料、面片序、固定與身分；整面控制組取自 53eccd56。"""
from __future__ import annotations

from pathlib import Path

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
from tests.engine.test_material_gradient import _settings
from tests.engine.test_material_gradient_mount import (
    _control_id, _control_kept, _designs, _digest, _header, _ledger_line, _project, _template_header, _template_line,
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
    header = _header(designs)
    assert _digest(_ledger_line(tmp_path / "ledger.jsonl", header)) == "a417256b0f40af5ab6703348c833bfb468d1e19593a3079061df943a849c2fe5"
    assert _digest(_template_line(tmp_path / "templates.jsonl", _template_header(header, _control_kept()))) == "304b40fe88bb01f381ebce0962f3c0c3cf544151bb63f018769cf0bfe21ece3c"
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


def test_partitioned_chain_search_templates_and_full_endpoint(tmp_path: Path) -> None:
    """抓鏈未按片求阻抗、梯度斷線、帳本缺片、固定片進樣板輪、匯出遺失區座標。"""
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
    result = subprocess.run([sys.executable, str(probe), str(tmp_path)], env=env, cwd=tmp_path,
                             capture_output=True, text=True, timeout=55)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "probe.json").exists()


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
