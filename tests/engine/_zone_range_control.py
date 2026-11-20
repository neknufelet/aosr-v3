"""#1199 控制組：沒寫範圍時，計畫回覆、起跑設計、帳本身分的位元組。

只用改動前主線（46b1ea3d）就有的介面，主線與分支跑同一支：尺是在主線跑這一支量的，釘在 ``test_zone_range.py``。
房間與設計設定全是考卷自己編的（``_zone_symmetry_room`` 與 ``test_gui_gradient_jobs`` 的假房間）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient import cli, run
from aosr.gradient.boundary import scheme_patch_keys
from aosr.gradient.material_gradient import SurfaceDesign
from aosr.gradient.start_config import scene_wall_materials
from aosr.gui.gradient_plan import GradientRequest, SearchPlan, configuration_arguments, plan_search, start_arguments
from aosr.gui.zone_design import DesignSettings
from aosr.physics.fem_reduced import ReducedModel
from aosr.reporting.scheme import Scheme
from aosr.scoring.gradient_objective_jax import gradient_objective_constants
from tests.engine._six_search_helpers import six_header_for
from tests.engine._zone_symmetry_room import searching, symmetry_document
from tests.engine.test_material_gradient import _gradient_settings, _receivers
from tests.engine.test_material_gradient_mount import _project

SETTINGS = load_material_gradient_settings(config_path("material_gradient.toml"))
CUTS = {"x0": [("b", (0.4, 1.5), (0.3, 2.4)), ("a", (2.5, 3.6), (0.3, 2.4))], "y0": [("s", (1.0, 3.0), (0.5, 2.2))],
        "yL": [("s", (1.0, 3.0), (0.5, 2.2))], "ceiling": [("c", (2.0, 3.5), (1.2, 2.8))]}
ZONES = {"x0": ["b", "a"], "y0": ["s"], "yL": ["s"], "ceiling": ["c"]}
WOOL = "glass_wool_48k"


def control_design() -> dict[str, object]:
    """左右對稱兩對（前牆 b、a；左右牆 s）、准用清單、每一區深度上限、天花一區只准 48K 與不貼。數字是編的。"""
    document = searching(ZONES)
    zones = cast(dict[str, dict[str, dict[str, object]]], document["zones"])
    for wall, zone in (("x0", "b"), ("x0", "a")):
        zones[wall][zone].update({"allowed": [WOOL, "glass_wool_24k", "none"], "max_depth_m": 0.25})
    for wall in ("y0", "yL"):
        zones[wall]["s"].update({"max_depth_m": 0.2})
    zones["ceiling"]["c"].update({"allowed": [WOOL, "none"]})
    return document


def control_scheme() -> Scheme:
    return Scheme.model_validate(symmetry_document(CUTS))


def control_plan(design: dict[str, object] | None = None) -> SearchPlan:
    return plan_search(control_scheme(), DesignSettings.model_validate(design or control_design()), GradientRequest(max_steps=5),
                       SETTINGS)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan_bytes(plan: SearchPlan) -> bytes:
    """計畫存檔、開跑參數、建替身核固定材料那一份參數，接成一份。"""
    return json.dumps([plan.model_dump_json(), start_arguments(plan), configuration_arguments(plan)],
                      ensure_ascii=False).encode()


def start_designs(plan: SearchPlan) -> tuple[tuple[SurfaceDesign, ...], dict[str, tuple[str, ...]]]:
    """命令列開跑那一組起跑配置（``cli.start_designs``，網頁核左右對稱也叫它）。"""
    return cli.start_designs(control_scheme(), SETTINGS, run.presets_table(), plan.presets, mount=plan.request.mount,
                             fixed=dict(plan.fixed_surfaces), caps=dict(plan.max_depths), allowed=dict(plan.allowed),
                             capabilities=load_capabilities(config_path("capabilities.toml")),
                             directivity=load_directivity_defaults(config_path("directivity_defaults.toml")), ties=plan.ties)


def designs_bytes(plan: SearchPlan) -> bytes:
    designs, expanded = start_designs(plan)
    return repr((designs, expanded)).encode()


def ledger_bytes(plan: SearchPlan) -> bytes:
    """帳本兩份標頭（庚二、換樣板）、資料夾名與 ``project.json``，接成一份。模型只帶標頭讀的中繼資料。"""
    scheme = control_scheme()
    keys = scheme_patch_keys(scheme)
    model = cast(ReducedModel, SimpleNamespace(input_hash="input-x", mesh_hash="mesh-x", physics_identity="physics-x",
                                              patch_keys=keys, matrices=SimpleNamespace(boundary=(None,) * len(keys))))
    designs, expanded = start_designs(plan)
    constants = gradient_objective_constants(np.asarray(FEM_LANE_FREQUENCIES_HZ), _gradient_settings(), _receivers(), 2)
    header = six_header_for(model, designs, SETTINGS, objective_constants=constants)
    header = header.model_copy(update={"ties": plan.ties})
    kept = scene_wall_materials(scheme, capabilities=load_capabilities(config_path("capabilities.toml")),
                                directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    template = run.template_header_for(header, SETTINGS, kept, base_ledger_sha256="e" * 64,
                                       allowed=expanded, table=run.presets_table())
    identity = run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=plan.presets,
                               purpose="music", settings_fingerprint="c" * 64, physics_identity="phys",
                               objective_settings_fingerprint="d" * 64, max_depths=plan.max_depths, patch_keys=keys,
                               allowed_targets=plan.allowed, ties=plan.ties)
    project = run.GradientProject.model_validate({**_project().model_dump(), "presets": plan.presets,
                                                  "max_depths": plan.max_depths, "allowed_targets": plan.allowed,
                                                  "ties": plan.ties, "gradient_id": identity})
    return json.dumps([header.model_dump_json(), template.model_dump_json(), identity, project.model_dump_json()],
                      ensure_ascii=False).encode()


def preview_bytes(tmp_path: Path) -> bytes:
    """網頁第三階段頁的預覽回覆（``GradientJobs.preview``）：准用清單、深度上限各一區；暫存路徑換成固定字。"""
    from aosr.gui.zone_design import write_design
    from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _data, _jobs, _scheme

    data_dir = _data(tmp_path)
    document = {"schema_version": "aosr.zone_design.v1", "scheme_id": NAME, "zones": {
        "x0": {"r0_c0": {"mode": "search", "allowed": [WOOL, "glass_wool_24k", "none"], "max_depth_m": 0.25},
               "r1_c1": {"mode": "search"}},
        "ceiling": {"r0_c1": {"mode": "keep_original"}}}}
    write_design(data_dir, DesignSettings.model_validate(document), _scheme())
    reply = _jobs(data_dir, FakeLauncher()).preview(NAME, GradientRequest(max_steps=5))
    return json.dumps(reply, ensure_ascii=False, sort_keys=True).replace(str(tmp_path), "<tmp>").encode()


def limits_bytes() -> bytes:
    """分區頁的限制檢查回覆（``zone_limits``，判准用材料組放不放得下用內建材料組）。"""
    from aosr.gui.gradient_plan import builtin_depth_targets
    from aosr.gui.zone_limits import zone_limits

    reply = zone_limits(control_scheme().model_dump(mode="json"), control_design(),
                        depth_targets=lambda: builtin_depth_targets(SETTINGS))
    return json.dumps(reply, ensure_ascii=False, sort_keys=True).encode()


def control_hashes(tmp_path: Path) -> dict[str, str]:
    plan = control_plan()
    return {"plan": sha(plan_bytes(plan)), "designs": sha(designs_bytes(plan)), "ledger": sha(ledger_bytes(plan)),
            "preview": sha(preview_bytes(tmp_path)), "limits": sha(limits_bytes())}

