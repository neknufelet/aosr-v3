"""綁定表（#1206 第 1 支）雙精度子行程：自編八片小場景、合成目標，整條產品路徑（前段、候選確認、換樣板、結果檔）。

場景全部自編：地板、天花、x0、xL 四面整面，y0 與 yL 各切 a、b 兩片，共八片；交給搜尋的只有 y0@a、y0@b、yL@a 三片，
其餘五片固定用原牆。原牆是自編的固定阻抗。設定是這支檔自己寫的一份（:data:`SETTINGS`），不讀產品設定檔，
所以釘死的位元組不跟著產品設定檔的調整變。

三段，各寫一個資料夾（``project.json``、``ledger.jsonl``、``templates.jsonl``、``result.json``）：

- ``plain``：不給綁定表、目標恆為零。零目標的每一格都是精確的浮點數，位元組跟處理器無關，考卷把四個檔的 SHA-256
  釘成常數（基準是加綁定表之前的主線 7d73a450）。這一段不碰任何綁定相關的名字，主線上原樣跑得動。
- ``bowl``：不給綁定表、左右不對稱的二次式。位元組跟處理器有關，不釘；施工時在同一台機器拿主線與改後各跑一次比。
- ``tied``：y0@a 與 yL@a 成對，同一個不對稱二次式；另外截帳本接續、把一格 θ 改一個最小浮點單位再接續。

牆鐘秒數一律換成 0（把四個模組的 ``time`` 換成假時鐘），帳本與結果檔才逐位可重現。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tests.engine._material_gradient_templates_probe import SYNTHETIC_SCORES
from tests.engine._six_search_helpers import six_objective

from aosr.config.material_gradient import SCHEME_MATERIAL, MaterialGradientSettings
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import material_gradient_loop as loop
from aosr.gradient import run
from aosr.gradient import template_rounds as rounds
from aosr.gradient.boundary import BoundaryKey, named_materials, partition_identity, patch_key
from aosr.gradient.fixed_surfaces import expanded_fixed_surfaces
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.physics.fem_reduced_boundary import PatchKey
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays


Record = dict[str, object]
Ties = tuple[tuple[str, str], ...]
GAP50 = "glass_wool_48k_50mm_gap50"
NAMES = ("floor", "ceiling", "x0", "xL", "y0@a", "y0@b", "yL@a", "yL@b")
PRESETS: tuple[str | None, ...] = (None, None, None, None, GAP50, GAP50, GAP50, None)
FIXED = {"floor": SCHEME_MATERIAL, "ceiling": SCHEME_MATERIAL, "x0": SCHEME_MATERIAL, "xL": SCHEME_MATERIAL,
         "yL@b": SCHEME_MATERIAL}
TIES: Ties = (("y0@a", "yL@a"),)
# 自編設定：三個家族用產品構造預設的樣板代號，其餘數字是這份考卷自己的（輪數與步數刻意小、改善門檻 0.01）。
SETTINGS: Record = {
    "descent": {"max_steps": 2, "initial_step": 0.01, "backtrack_ratio": 0.5, "armijo_constant": 0.0001,
                "gradient_norm_stop": 0.01, "min_step": 1e-06},
    "retune": {"steps": 3},
    "candidates": {"max_per_round": 3, "improvement_threshold": 0.01, "max_rounds": 4, "max_evaluations": 2000},
    "templates": {"max_per_round": 2, "max_rounds": 3, "max_evaluations": 1500, "max_seconds": 7200},
    "export": {"min_cavity_m": 0.001},
    "family": [
        {"id": "glass_wool_48k", "presets": ["glass_wool_48k_50mm", GAP50], "thicknesses_m": [0.05, 0.1, 0.2],
         "cavity_range_m": [0.0, 0.25], "max_total_depth_m": 0.3, "template_preset": GAP50,
         "initial_thickness_m": 0.05, "initial_cavity_m": 0.05},
        {"id": "glass_wool_24k", "presets": ["glass_wool_24k_50mm"], "thicknesses_m": [0.05, 0.1, 0.2],
         "cavity_range_m": [0.0, 0.25], "max_total_depth_m": 0.3, "template_preset": "glass_wool_24k_50mm",
         "initial_thickness_m": 0.05, "initial_cavity_m": None},
        {"id": "glass_wool_96k", "presets": ["glass_wool_96k_50mm"], "thicknesses_m": [0.05, 0.1, 0.2],
         "cavity_range_m": [0.0, 0.25], "max_total_depth_m": 0.3, "template_preset": "glass_wool_96k_50mm",
         "initial_thickness_m": 0.05, "initial_cavity_m": None}]}
# 不對稱二次式的偏移：成對兩片都換成 96K 時各扣這麼多，讓「一對一起換」真的被接受一次。
OFFSETS = {"y0@a->glass_wool_96k": -0.5, "yL@a->glass_wool_96k": -0.5}
FILES = (run.PROJECT_FILE, run.LEDGER_FILE, run.TEMPLATES_FILE, run.RESULT_FILE)


class _Clock:
    """假時鐘：永遠是 0，帳本每一列的秒數就都是 0。"""

    @staticmethod
    def perf_counter() -> float:
        return 0.0


def settings() -> MaterialGradientSettings:
    return MaterialGradientSettings.model_validate(SETTINGS)


def kept_patches() -> dict[PatchKey, SurfaceMaterial]:
    """八片自編原牆：固定阻抗 20000＋1000×片序（Pa·s/m）。"""
    return {patch_key(name): ConstantImpedance(impedance_pa_s_m=20000.0 + 1000.0 * index)
            for index, name in enumerate(NAMES)}


def designs() -> tuple[gradient_search.SurfaceDesign, ...]:
    return run.designs_for(PRESETS, settings(), kept=kept_patches(), fixed_surfaces=FIXED, table=run.presets_table())


def _factory(kind: str) -> rounds.ObjectiveFactory:
    """``zero``：恆為零（乘零，梯度也恆為零）。``bowl``：Σ w·(θ − 目標)² ＋ 偏移，棉板權重是片序＋1、目標 0.2，
    空腔權重 10、目標 0.03＋0.01×片序——成對兩片的權重與目標都不同，左右不對稱。下降只走兩步、到不了 0.2，
    取規格落在 0.1，所以候選確認會接受「往厚」。同一個配置回同一份函式。"""
    made: dict[rounds.Surfaces, gradient_search.ChainObjective] = {}
    setting = settings()

    def factory(candidate: tuple[gradient_search.SurfaceDesign, ...]) -> gradient_search.ChainObjective:
        surfaces = rounds.surfaces_of(candidate)
        if surfaces in made:
            return made[surfaces]
        weights, targets, offset = [], [], 0.0
        for index, design in enumerate(candidate):
            offset += OFFSETS.get(rounds.candidate_id(design.surface_id, rounds.target_of(design.preset_id, setting)), 0.0)
            for name in design.variables:
                cavity = name == design.cavity_variable
                weights.append(10.0 if cavity else float(index + 1))
                targets.append(0.03 + 0.01 * index if cavity else 0.2)
        weight, target = np.asarray(weights), np.asarray(targets)

        def objective(theta: jax.Array) -> tuple[jax.Array, GradientObjectiveArrays]:
            shift = theta - jnp.asarray(target, dtype=theta.dtype)
            value = jnp.sum(jnp.asarray(weight, dtype=theta.dtype) * shift * shift) + offset
            if kind == "zero":
                value = value * 0.0
            return value, GradientObjectiveArrays(value[None], value[None], value, value * 0.5, value * 0.25,
                                                  value * 0.125, value, value * 0.125)

        made[surfaces] = six_objective(objective)
        return made[surfaces]

    return factory


def identity(ties: Ties = ()) -> str:
    """純字串輸入的資料夾名；空表也明寫 ``ties=()``（開跑的命令列就是這樣叫），要跟沒有這個引數同一個名。"""
    return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=PRESETS, purpose="music",
                           settings_fingerprint=settings().fingerprint, physics_identity="synthetic",
                           objective_settings_fingerprint="d" * 64, fixed_surfaces=FIXED,
                           patch_keys=tuple(patch_key(name) for name in NAMES), ties=ties)


def project(ties: Ties = ()) -> run.GradientProject:
    return run.GradientProject(
        format_version=run.PROJECT_VERSION, gradient_id=identity(ties), scheme_path="/s", scheme_sha256="a" * 64,
        surrogate_path="/m", surrogate_manifest_sha256="b" * 64, receipt_path="/r", receipt_sha256="f" * 64,
        presets=PRESETS, purpose="music", settings_path="/t", settings_fingerprint=settings().fingerprint,
        capabilities_path="/c", physics_identity="synthetic", mesh_hash="synthetic", source_points={},
        receiver_points={}, engine_commit=None, started_at="2026-11-17T00:00:00+00:00",
        objective_settings_fingerprint="d" * 64, fixed_surfaces=FIXED, ties=ties)


def header(ties: Ties = ()) -> ledger.MaterialGradientHeader:
    """合成目標沒有替身：身分三格與評分指紋填固定字樣，其他照設計清單；分數常數配合成目標的分項。"""
    made = designs()
    return ledger.MaterialGradientHeader(
        ledger_version=ledger.LEDGER_VERSION, input_hash="synthetic", mesh_hash="synthetic",
        physics_identity="synthetic", surfaces=tuple((design.surface_id, design.preset_id) for design in made),
        variable_names=gradient_search.design_names(made), settings_fingerprint=settings().fingerprint,
        objective_fingerprint=hashlib.sha256(b"synthetic").hexdigest(),
        kept_digests=ledger._kept_digests(made), score_constants=SYNTHETIC_SCORES,
        fixed_surfaces={design.surface_id: design.fixed_material_id for design in made
                        if design.fixed_material_id is not None},
        boundary_patches=partition_identity(tuple(patch_key(name) for name in NAMES)), ties=ties)


def problem(kind: str, ties: Ties = ()) -> run.GradientProblem:
    made, setting, kept, factory = designs(), settings(), kept_patches(), _factory(kind)
    context = rounds.TemplateContext(setting, run.presets_table(), named_materials(kept), factory,
                                     fixed_surfaces=expanded_fixed_surfaces(FIXED, NAMES), max_depths={}, ties=ties)
    wide: dict[BoundaryKey, SurfaceMaterial] = {key: material for key, material in kept.items()}
    return run.GradientProblem(made, setting, factory(made), header(ties), wide, context)


def _search(folder: Path, case: run.GradientProblem, ties: Ties) -> None:
    """整條產品路徑：寫 ``project.json``、帳本標頭，前段與候選確認、換樣板、結果檔。"""
    folder.mkdir()
    run.write_json_atomic(folder / run.PROJECT_FILE, project(ties))
    ledger.write_ledger(folder / run.LEDGER_FILE, case.header, ())
    run._fresh(case, folder / run.LEDGER_FILE, None)
    run._template_segment(case, folder, None)
    run.write_result(folder, case)


def _cut(source: Path, target: Path, keep: Callable[[list[Record]], int], file: str) -> None:
    """另存一份資料夾，``file`` 那本帳留標頭與前 ``keep(列)`` 列；結果檔拿掉（接續會重寫）。"""
    shutil.copytree(source, target)
    lines = (target / file).read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines[1:]]
    (target / file).write_bytes(b"".join(lines[:keep(rows) + 1]))
    (target / run.RESULT_FILE).unlink()
    if file == run.LEDGER_FILE:
        (target / run.TEMPLATES_FILE).unlink()


def _resume(folder: Path, case: run.GradientProblem) -> None:
    run._resume(case, folder / run.LEDGER_FILE, None)
    run._template_segment(case, folder, None)
    run.write_result(folder, case)


def _after_first_decision(phase: str) -> Callable[[list[Record]], int]:
    """留到第一列決定之後再多一列（半輪）；沒有第二輪就留到那一列決定。"""
    def keep(rows: list[Record]) -> int:
        first = next(index for index, row in enumerate(rows) if row["phase"] == phase)
        return min(first + 2, len(rows) - 1)
    return keep


def _resumed(directory: Path, name: str, case: run.GradientProblem) -> Record:
    """兩本帳各截一次接續，回每一份跟一口氣跑完的那一份是不是四個檔逐位相同。"""
    found: Record = {}
    for label, file, phase in (("rounds", run.LEDGER_FILE, "round_decision"),
                               ("templates", run.TEMPLATES_FILE, "template_decision")):
        target = directory / f"{name}-resume-{label}"
        _cut(directory / name, target, _after_first_decision(phase), file)
        _resume(target, case)
        found[label] = all((target / item).read_bytes() == (directory / name / item).read_bytes() for item in FILES)
    return found


def _broken(directory: Path, case: run.GradientProblem) -> Record:
    """帳本只留前段，把前段最後一列裡 yL@a 空腔那一格改一個最小浮點單位，再接續：產品程式要拋錯、不往下算。"""
    target = directory / "tied-broken"
    _cut(directory / "tied", target, lambda rows: sum(1 for row in rows if row["round"] is None), run.LEDGER_FILE)
    path = target / run.LEDGER_FILE
    lines = path.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[-1])
    cell = case.header.variable_names.index("yL@a.layer2.thickness_m")
    row["theta_hex"][cell] = float(np.nextafter(float.fromhex(row["theta_hex"][cell]), 1.0)).hex()
    lines[-1] = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(b"".join(lines))
    before = path.read_bytes()
    try:
        _resume(target, case)
    except Exception as error:
        return {"error": type(error).__name__, "message": str(error), "is_value_error": isinstance(error, ValueError),
                "ledger_unchanged": path.read_bytes() == before, "result_written": (target / run.RESULT_FILE).exists()}
    return {"error": None}


def main(argv: list[str]) -> int:
    directory, parts = Path(argv[0]), argv[1:]
    for module in (gradient_search, loop, rounds, run):
        setattr(module, "time", _Clock)
    output: Record = {}
    for part in parts:
        ties = TIES if part == "tied" else ()
        case = problem("zero" if part == "plain" else "bowl", ties)
        _search(directory / part, case, ties)
        output[part] = {"gradient_id": identity(ties), "resumed": _resumed(directory, part, case),
                        "digests": {item: hashlib.sha256((directory / part / item).read_bytes()).hexdigest()
                                    for item in FILES}}
        if part == "tied":
            output["broken"] = _broken(directory, case)
    (directory / "output.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
