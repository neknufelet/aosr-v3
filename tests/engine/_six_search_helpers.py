"""低頻演算法控制題的六類帳本外殼；不冒充物理六類答案。

舊控制題維持原來的純量與梯度，只把低頻三項保留在新列，另外五類是合成值。
物理六類接線由 test_six_category_search 與 test_six_category_chain 的獨立答案守。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from aosr.config.material_gradient import MaterialGradientSettings
from aosr.gradient.material_gradient import SurfaceDesign
from aosr.gradient.material_gradient_ledger import MaterialGradientHeader
from aosr.physics.fem_reduced import ReducedModel

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient.six_category_chain import SixCategoryChainAux
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays, GradientObjectiveConstants
from aosr.scoring.ranking import read_ranking_rules
from aosr.scoring.six_category_jax import SixCategoryArrays, SixCategoryConstants
from tests.engine._six_category_jax_cases import NAMES


@cache
def weights() -> tuple[float, ...]:
    values = read_ranking_rules(load_quality_targets(config_path('quality_targets.toml')),
                                'dedicated_two_channel_listening_room').category_weights
    return tuple(values[name] for name in NAMES)


def ledger_constants(low: GradientObjectiveConstants) -> SixCategoryConstants:
    values = {name: np.asarray(0.) for name in SixCategoryConstants._fields}
    values.update({'objective_'+name: np.asarray(value) for name, value in zip(low._fields, low, strict=True)})
    values['category_weights'] = np.asarray(weights())
    return SixCategoryConstants(**values)


def six_aux(value: jax.Array, low: GradientObjectiveArrays) -> SixCategoryChainAux:
    zero, valid = jnp.zeros_like(value), jnp.asarray(True)
    first = value - weights()[-1] * low.objective_total
    product = first + weights()[-1] * low.screening_total
    categories = SixCategoryArrays(first, zero, zero, zero, zero, low.screening_total,
        valid, valid, valid, valid, valid, valid, low.objective_total, product, value,
        jnp.asarray(1), jnp.asarray(1), zero, jnp.asarray(False))
    return SixCategoryChainAux(categories, low, value, product)


def six_objective(objective: Callable[[jax.Array], tuple[jax.Array, GradientObjectiveArrays]]
                  ) -> Callable[[jax.Array], tuple[jax.Array, SixCategoryChainAux]]:
    def wrapped(theta: jax.Array) -> tuple[jax.Array, SixCategoryChainAux]:
        value, low = objective(theta)
        return value, six_aux(value, low)
    return wrapped


def six_header_for(model: ReducedModel, designs: Sequence[SurfaceDesign], settings: MaterialGradientSettings, *,
                   objective_constants: GradientObjectiveConstants) -> MaterialGradientHeader:
    from aosr.gradient.material_gradient_ledger import header_for

    return header_for(model, designs, settings, objective_constants=ledger_constants(objective_constants))


def expand_components(value: float, screening: float, components: tuple[float, ...]) -> tuple[float, ...]:
    low = components
    # 合成演算法題的傳入 value 已是舊低頻含代理目標；此處只把原來答案裝進六類外殼。
    first = value - weights()[-1] * value
    product = first + weights()[-1] * screening
    return (*low, first, 0., 0., 0., 0., screening, value, product)
