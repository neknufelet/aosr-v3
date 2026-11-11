"""材料庫考卷的隔離子行程：從標準輸入讀一份物理用牆面材料與頻率，雙精度算垂直入射阻抗，寫實部、虛部與註記代號。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_models import SurfaceMaterial


def main() -> None:
    request = json.loads(sys.stdin.read())
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    material = adapter.validate_python(request["material"])
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    frequencies = jnp.asarray(request["frequencies_hz"], dtype=jnp.float64)
    impedance, notes = surface_impedance(material, frequencies, air=air, ranges=ranges)
    values = np.asarray(impedance)
    Path(sys.argv[1]).write_text(json.dumps({"real": values.real.tolist(), "imag": values.imag.tolist(),
                                             "notes": [note.code for note in notes]}), encoding="utf-8")


if __name__ == "__main__":
    main()
