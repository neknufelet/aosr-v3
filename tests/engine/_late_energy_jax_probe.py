"""晚期混響 JAX 版的子行程探針：父考卷開好雙精度才叫它，結果寫在 tmp_path 給父考卷讀。

只把題目交給產品公開入口，不在探針裡重寫公式；比對與判決全在父考卷。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.geometry.shoebox import Room
from aosr.physics import late_energy_jax


class Case(TypedDict):
    room_m: list[str]
    n_per_wall: int
    alpha: list[list[str]]


# 布林旗標另外交回；四個浮點量的編碼與 dtype 名單不收它。
_FLAG = "dissipation_resolved"


def _hex(values: jax.Array) -> list[str]:
    return [float(value).hex() for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _arrays(result: late_energy_jax.LateEnergyArrays) -> dict[str, list[str]]:
    return {name: _hex(value) for name, value in result._asdict().items() if name != _FLAG}


def _dtypes(result: late_energy_jax.LateEnergyArrays) -> list[str]:
    return sorted({str(value.dtype) for name, value in result._asdict().items() if name != _FLAG})


def _flags(result: late_energy_jax.LateEnergyArrays) -> dict[str, object]:
    flag = result.dissipation_resolved
    return {"dtype": str(flag.dtype), "values": [bool(value) for value in np.asarray(flag).reshape(-1)]}


def _run_case(case: Case) -> dict[str, object]:
    lx, ly, lz = (float.fromhex(value) for value in case["room_m"])
    geometry = late_energy_jax.late_energy_geometry(Room(lx, ly, lz), case["n_per_wall"])
    alpha = jnp.asarray([[float.fromhex(value) for value in row] for row in case["alpha"]], dtype=jnp.float64)
    single = alpha.astype(jnp.float32)
    compiled = jax.jit(late_energy_jax.late_energy_arrays)

    def total(values: jax.Array) -> jax.Array:
        return jnp.sum(late_energy_jax.late_energy_arrays(geometry, values).late_reverberant_energy)

    eager = late_energy_jax.late_energy_arrays(geometry, alpha)
    first = compiled(geometry, alpha)
    second = compiled(geometry, alpha)
    single_result = compiled(geometry, single)
    gradient = jax.grad(total)(alpha)
    single_gradient = jax.jit(jax.grad(total))(single)
    return {
        "eager": _arrays(eager),
        "jit": _arrays(first),
        "jit_again": _arrays(second),
        "single": _arrays(single_result),
        "gradient": _hex(gradient),
        "single_gradient": _hex(single_gradient),
        "double_dtypes": _dtypes(eager) + _dtypes(first) + [str(gradient.dtype)],
        "single_dtypes": _dtypes(single_result) + [str(single_gradient.dtype)],
        "resolved": {"eager": _flags(eager), "jit": _flags(first), "single": _flags(single_result)},
    }


def main(problem_path: Path, output_path: Path) -> None:
    problem = cast(dict[str, Case], json.loads(problem_path.read_text(encoding="utf-8")))
    output = {"cases": {name: _run_case(case) for name, case in problem.items()}}
    output_path.write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
