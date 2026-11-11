"""#1092 分區結果頁考卷共用：小房 x 起點牆切兩區，a 區在原牆上貼一層棉、b 區不動；原牆基準由產品推（不手寫）。

設計那一份真算：雙精度子行程跑正式入口，同一次 pytest 只算一次（約 46 秒，#1092 本機量）。
原牆基準不在這裡算：由網頁上按鈕、正式網頁行程自己開正式計算（考卷看的就是那一條路）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.materials.construction_models import ConstantImpedance, MountedOnWall
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._scheme_cache import shared_json
from tests.engine._zoned_report_cases import small_scheme, split_x0, with_x0

RUN_DATE = "2026-11-11"


def original_x0() -> ConstantImpedance:
    zones = small_scheme(zoned=True).scene.surface_zones_by_wall
    assert zones is not None
    material = zones["x0"][0].material
    assert isinstance(material, ConstantImpedance)
    return material


def mounted(wall: ConstantImpedance, thickness_m: float = 0.05) -> MountedOnWall:
    return MountedOnWall.model_validate({"kind": "mounted_on_wall", "layers": [{
        "kind": "porous", "thickness_m": thickness_m, "flow_resistivity_pa_s_m2": 19100.0,
        "model": "komatsu", "material_class": "fibrous"}], "wall": wall.model_dump(mode="json")})


def design_scheme() -> Scheme:
    wall = original_x0()
    base = with_x0(split_x0(mounted(wall), wall))
    return Scheme.model_validate(base.model_dump(mode="json") | {"scheme_id": "zone-design"})


def real_design(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> str:
    """設計那一份結果的 JSON 原文（真算）。"""
    def produce() -> str:
        folder = tmp_path_factory.mktemp("zone-result-design")
        source, out = folder / "design.scheme.json", folder / "design.result.json"
        source.write_text(design_scheme().model_dump_json(), encoding="utf-8")
        child = subprocess.run([sys.executable, "-m", "aosr.reporting.scheme_cli", "run", str(source), "--out", str(out),
                                "--engine-commit", "control", "--run-date", RUN_DATE,
                                "--capabilities", str(config_path("capabilities.toml"))],
                               env=forward_cpu_pins(child_process_env(threads=1)), capture_output=True, text=True,
                               timeout=600)
        assert child.returncode == 0, child.stderr[-2000:]
        text = out.read_text(encoding="utf-8")
        assert SchemeResult.model_validate_json(text).scheme == design_scheme()
        return text

    return shared_json(tmp_path_factory, worker_id, "zone-result-design", produce)
