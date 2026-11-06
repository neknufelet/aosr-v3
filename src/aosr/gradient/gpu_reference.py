"""開跑精度閘門的 CPU 雙精度參考子行程；只求給定 θ 的目標值，不跑搜尋。"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from pydantic import TypeAdapter

from aosr import runtime
from aosr.gradient.run import GradientProject, load_problem


def main() -> None:
    """標準輸入兩列：project JSON、θ 的 hex 字串列；標準輸出一個 hex 目標值。"""
    if runtime.jax_settings() != ("cpu", True):
        raise RuntimeError("精度參考子行程必須是 CPU 雙精度")
    project = GradientProject.model_validate_json(sys.stdin.readline())
    theta = TypeAdapter(tuple[str, ...]).validate_json(sys.stdin.read())
    problem = load_problem(project.model_copy(update={"device": "cpu", "precision_check": None}))
    value, _ = jax.jit(problem.objective)(jnp.asarray([float.fromhex(x) for x in theta], dtype=jnp.float64))
    sys.stdout.write(float(value).hex() + "\n")


if __name__ == "__main__":
    main()
