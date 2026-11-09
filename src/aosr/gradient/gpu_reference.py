"""開跑精度閘門的 CPU 雙精度參考子行程；只求給定 θ 的目標值，不跑搜尋。"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from pydantic import TypeAdapter

from aosr import runtime
from aosr.gradient.run import GradientProject, load_problem
from aosr.gradient.gpu import PrecisionReading
from aosr.gradient.six_category_chain import SixCategoryChainAux


def main() -> None:
    """輸入 project JSON、θ 的 hex 列；第三列選 six_category 回三組原值 JSON，否則沿用 hex 目標值。"""
    if runtime.jax_settings() != ("cpu", True):
        raise RuntimeError("精度參考子行程必須是 CPU 雙精度")
    project = GradientProject.model_validate_json(sys.stdin.readline())
    theta = TypeAdapter(tuple[str, ...]).validate_json(sys.stdin.readline())
    six_category = sys.stdin.read().strip() == 'six_category'
    problem = load_problem(project.model_copy(update={"device": "cpu", "precision_check": None}))
    value, aux = jax.jit(problem.objective)(jnp.asarray([float.fromhex(x) for x in theta], dtype=jnp.float64))
    if six_category:
        if not isinstance(aux, SixCategoryChainAux):
            raise ValueError('CPU 參考必須收到六類 aux')
        reading = PrecisionReading.model_validate(dict(objective_total=float(value), product_total=float(aux.product_total),
            categories=tuple(float(x) for x in aux.categories[:6])))
        sys.stdout.write(reading.model_dump_json()+'\n')
    else:
        sys.stdout.write(float(value).hex() + "\n")


if __name__ == "__main__":
    main()
