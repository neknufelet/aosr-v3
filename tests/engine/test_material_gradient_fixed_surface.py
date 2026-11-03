"""#990 固定面：合成整場、換樣板與接續、身分、命令列拒收、主線控制組。

移除候選過濾會破壞逐列固定材料；移除身分鍵會撞搜尋；忽略 project 設定會破壞重建與終點核對。
控制組在改 src 前於 814d471a121b60d99ad93c04612636bb21d86522 擷取；扣除秒數與由秒數衍生的帳本摘要，
其餘列與結果逐位比，另沿用 mount 考卷的實際寫入端位元組控制組。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from aosr.runtime import child_process_env, configure_jax
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, run
from aosr.gradient import material_gradient as search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.gradient.endpoint import design_digest, materials_from_surfaces
from aosr.gradient.endpoint_run import checked_result
from aosr.gradient.fixed_surfaces import fixed_materials, parse_fixed_surfaces
from aosr.gradient.report import _kept_texts
from aosr.physics.report_surfaces import WholeWallZone
from tests.engine._material_gradient_templates_probe import SYNTHETIC_SCORES, synthetic_factory
from tests.engine.test_material_gradient import _presets, _settings
from tests.engine.test_material_gradient_mount import (
    _control_kept, _header, _project, test_rigid_identities_equal_mainline_bit_for_bit as _mount_control,
)

FIXED = {"floor": "glass_wool_24k_50mm"}
PRESETS = ("glass_wool_48k_50mm_gap50",) * 6
# 改動前主線（814d471a）跑的合成整場：三份產物的 SHA-256。擷取紀錄留在本票工作區的證據資料夾，不進版控。
CONTROL: dict[str, str] = {'ledger.jsonl': '87362b54c36120c7bb78f6f4a180501635be884203e0c177c299e8cdd9f72b11', 'templates.jsonl': '65c2511a3dd47432f9bb78c6de22414af0bd8af0311a89c52d068e652b035a15', 'result.json': '3a7dcefbc6ee19ba59c301e5c50773772b4b0b48acd6461161420c3620cb7de3'}


def _problem(fixed: dict[str, str] | None = None) -> run.GradientProblem:
    settings = _settings()
    kept = _control_kept()
    designs = run.designs_for(PRESETS, settings, kept={wall: kept[wall.wall_name()] for wall in Wall.all()}, fixed_surfaces=fixed)
    factory = synthetic_factory(settings, {}, [0])
    context = rounds.TemplateContext(settings, _presets(), kept, factory, fixed_surfaces=fixed)
    header = _header(designs).model_copy(update={"score_constants": SYNTHETIC_SCORES})
    return run.GradientProblem(designs, settings, factory(designs), header,
                               {wall: kept[wall.wall_name()] for wall in Wall.all()}, context)


def _snapshot(folder: Path) -> dict[str, str]:
    """秒數是量測值；結果中的帳本摘要也受秒數影響，其餘全部進逐位控制。"""
    result = {}
    for name in (run.LEDGER_FILE, run.TEMPLATES_FILE, run.RESULT_FILE):
        lines = [json.loads(line) for line in (folder / name).read_text().splitlines()]
        for item in lines:
            item.pop("seconds", None)
            for key in ("base_ledger_sha256", "ledger_sha256", "template_ledger_sha256"):
                if key in item:
                    item[key] = "measured-seconds-dependent"
        result[name] = hashlib.sha256(json.dumps(lines, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return result


def _run(folder: Path, monkeypatch: pytest.MonkeyPatch, fixed: dict[str, str] | None = None) -> run.GradientProblem:
    folder.mkdir()
    project = _project() if fixed is None else _project().model_copy(update={"fixed_surfaces": fixed})
    run.write_json_atomic(folder / run.PROJECT_FILE, project)
    problem = _problem(fixed)
    monkeypatch.setattr(run, "load_problem", lambda project: problem)
    run.run_search(folder, should_stop=None)
    return problem


def test_no_fixed_matches_mainline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓預設路徑的任何帳本數值、配置、欄位或結果改變；只剔除實測秒數。"""
    _invoke(tmp_path, "control")
    assert json.loads((tmp_path / "snapshot.json").read_text()) == CONTROL
    _mount_control(tmp_path)


def _check_fixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓固定面被帶變數、換樣板、讀回還原成原牆，並經真正的終點結果核對與材料匯出。"""
    folder = tmp_path / "fixed"
    problem = _run(folder, monkeypatch, FIXED)
    # 預期材料由既有資料與方案解析器取得，不呼叫本次新增的材料解析函式。
    raw = next(p.material for p in _presets() if p.id == "glass_wool_24k_50mm")
    material = WholeWallZone.model_validate({"area": "whole_wall", "material": raw}).material
    floor = problem.designs[0]
    assert floor.material is None and floor.kept_material == material and not floor.variables
    header, rows = ledger.read_ledger(folder / run.LEDGER_FILE)
    assert header.fixed_surfaces == FIXED
    assert all(not name.startswith("floor.") for name in header.variable_names)
    assert all(run.surface_specs(header, row.theta())[0].preset_id is None for row in rows)
    th, tr = templates.read_template_ledger(folder / run.TEMPLATES_FILE, settings=problem.settings)
    assert th.fixed_surfaces == FIXED
    assert any(row.phase == "quick_score" for row in tr)
    for row in tr:
        assert row.surfaces[0] == ("floor", None)
        assert row.candidate is None or not row.candidate.startswith("floor->")
        designs = rounds.designs_for_surfaces(row.surfaces, problem.templates)
        assert designs[0].kept_material == material and not designs[0].variables
        assert design_digest(designs[0]) == design_digest(floor)
    before = run.parse_result((folder / run.RESULT_FILE).read_bytes())
    assert before.surfaces[0].fixed_material_id == FIXED["floor"]
    assert run.resume_search(folder, should_stop=None) == before
    assert checked_result(folder, before, header, problem.settings, problem.kept) == before
    designs = run.final_designs([(s.surface_id, s.preset_id) for s in before.surfaces], problem.settings, problem.kept,
                                fixed_surfaces=FIXED)
    assert materials_from_surfaces(designs, np.asarray([float.fromhex(v) for v in before.theta_hex]), problem.settings)[Wall.all()[0]] == material
    project = run.read_project(folder)
    assert project.fixed_surfaces == FIXED
    assert "固定" in _kept_texts(project)["floor"] and FIXED["floor"] in _kept_texts(project)["floor"]
    changed = _problem({"floor": "glass_wool_96k_50mm"})
    monkeypatch.setattr(run, "load_problem", lambda project: changed)
    with pytest.raises(ValueError, match="標頭.*對不上"):
        run.resume_search(folder, should_stop=None)


def _identity(fixed: dict[str, str]) -> str:
    return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=PRESETS,
                           purpose="music", settings_fingerprint="c" * 64, physics_identity="p",
                           objective_settings_fingerprint="d" * 64, fixed_surfaces=fixed)


def test_fixed_identity_binds_face_material_and_sorted_mapping() -> None:
    """抓固定設定沒進身分，或輸入順序造成不同身分。"""
    assert len({_identity({}), _identity(FIXED), _identity({"floor": "glass_wool_96k_50mm"}),
                _identity({"ceiling": FIXED["floor"]})}) == 4
    assert _identity({**FIXED, "x0": "glass_wool_96k_50mm"}) == _identity({"x0": "glass_wool_96k_50mm", **FIXED})


def test_carpet_fixed_floor_uses_catalog_material_and_distinct_identity() -> None:
    """抓地毯 id 被拒收、固定材料被換掉，或與玻璃棉撞搜尋身分。"""
    fixed = parse_fixed_surfaces(["floor=carpet_on_concrete"])
    assert fixed == {"floor": "carpet_on_concrete"}
    raw = next(p.material for p in _presets() if p.id == "carpet_on_concrete")
    expected = WholeWallZone.model_validate({"area": "whole_wall", "material": raw}).material
    assert fixed_materials(fixed) == {"floor": expected}
    assert _identity(fixed) != _identity(FIXED)


def test_carpet_random_absorption_stays_below_padded_carpet(tmp_path: Path) -> None:
    """抓直接鋪地毯被改成比加墊更吸音；冷子行程開雙精度，不跑搜尋。"""
    env = child_process_env(threads=1)
    env["TMPDIR"] = str(tmp_path)
    child = subprocess.run([sys.executable, "-B", "-m", "tests.engine._carpet_absorption_probe"],
        cwd=Path(__file__).resolve().parents[2], env=env, text=True, capture_output=True, timeout=120)
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize(("values", "message"), [
    (["roof=glass_wool_24k_50mm"], "roof"),
    (["floor=missing"], "missing"),
    (["floor=glass_wool_24k_50mm"] * 2, "floor"),
    ([f"{wall.wall_name()}=glass_wool_24k_50mm" for wall in Wall.all()], "六面"),
    (["floor"], "面=材料"), (["floor="], "材料"),
])
def test_cli_rejects_invalid_fixed_before_calculation(values: list[str], message: str, tmp_path: Path,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    """抓不合法選項跑到方案載入或建搜尋目錄；故意給不存在的方案與替身。"""
    args = ["start", "--scheme", str(tmp_path / "absent"), "--surrogate", "absent", "--presets", ",".join(PRESETS),
            "--purpose", "music", "--root", str(tmp_path / "out")]
    for value in values:
        args.extend(("--fixed-surface", value))
    try:
        code = cli.main(args)
    except SystemExit as exited:
        assert isinstance(exited.code, int)
        code = exited.code
    assert code == cli.EXIT_FAILED
    error = capsys.readouterr().err
    assert message in error and "FileNotFoundError" not in error
    assert not (tmp_path / "out").exists()


def _invoke(directory: Path, part: str) -> None:
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    from tests.engine.test_wall_surfaces import _CPU_PIN_VARS
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).resolve().parents[2]), "src"))
    probe = Path(__file__).with_name("_material_gradient_fixed_probe.py")
    process = subprocess.run([sys.executable, str(probe), str(directory), part], env=env,
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr


def test_fixed_material_survives_every_row_resume_and_endpoint(tmp_path: Path) -> None:
    """合成整場在雙精度子行程，超過 60 秒直接失敗。"""
    _invoke(tmp_path, "fixed")
