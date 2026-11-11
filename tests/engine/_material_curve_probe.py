"""材料曲線考卷的冷子行程（#1082 段 2-2b）：正式開機設定（中央處理器＋雙精度）下打曲線介面，同一行程另直接呼叫物理當對照。

pytest 的行程刻意不開雙精度（引擎考卷的 conftest 有寫為什麼），材料走 JAX 要在冷子行程驗（#309）。
輸入（標準輸入，JSON）：``{案例名: {"requests": [{"method", "path", "content", "headers"?}], "physics": [牆面材料], "axis": [Hz]}}``；
``content`` 是原樣送出的字串（壞 JSON 也照送）。輸出（標準輸出，JSON）：每個案例每一步的狀態碼、回應內文、
那一步之後使用者材料庫檔的 SHA-256（檔不在就是 null），以及每一份牆面材料直接呼叫物理在 ``axis`` 上的吸音。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import cast

from aosr.runtime import configure_calculation_jax

# 要先開雙精度才能載入算數函式庫，所以這一行在其餘載入之前。
configure_calculation_jax()

import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter
from starlette.testclient import TestClient

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.material_editor import user_library_path
from aosr.materials.construction import random_incidence_absorption
from aosr.materials.construction_models import SurfaceMaterial

_MATERIAL: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> None:
    data_dir = Path(sys.argv[1])
    cases = cast(dict[str, dict[str, object]], json.load(sys.stdin))
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    library = user_library_path(data_dir)
    out: dict[str, object] = {}
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir)), base_url="http://localhost") as client:
        for name, case in cases.items():
            steps = []
            for request in cast(list[dict[str, object]], case["requests"]):
                headers = {"Content-Type": "application/json", **cast(dict[str, str], request.get("headers", {}))}
                reply = client.request(str(request["method"]), str(request["path"]), content=str(request["content"]).encode(), headers=headers)
                steps.append({"status": reply.status_code, "body": reply.json(), "library": _digest(library)})
            axis = jnp.asarray(cast(list[float], case["axis"]))
            expected = [[float(value) for value in np.asarray(random_incidence_absorption(
                _MATERIAL.validate_python(material), axis, air=air, ranges=ranges)[0])]
                for material in cast(list[object], case["physics"])]
            out[name] = {"steps": steps, "expected": expected}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
