"""六類單精度包裝：材料／替身單精度，評分雙精度；大常數作動態入參。"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from aosr.gradient.gpu import single_precision_constants
from aosr.physics.fem_reduced_jax import ReducedConstants
from aosr.scoring.six_category_jax import SixCategoryConstants

if TYPE_CHECKING:
    from aosr.gradient.six_category_chain import SixCategoryChainAux

type ScoreArrays = tuple[jax.Array, ...]
type Kernel = Callable[[jax.Array, ReducedConstants, SixCategoryConstants], tuple[jax.Array, SixCategoryChainAux]]
type Evaluation = tuple[tuple[jax.Array, SixCategoryChainAux], jax.Array]


def score_array_names(constants: SixCategoryConstants) -> tuple[str, ...]:
    """所有非純量浮點／複數欄走動態入參；索引、偏移、字串、bool 與純量留 CPU 閉包。

    不設大小門檻，連小向量也採同一規則，避免房間變大後悄悄嵌入編譯圖。
    結構欄供 Python 切片、長度、索引迴圈與 float／bool 轉換；數值陣列始終雙精度。
    """
    return tuple(name for name, value in zip(constants._fields, constants, strict=True)
                 if value.ndim > 0 and value.dtype.kind in 'fc')


def double_precision_score_arrays(constants: SixCategoryConstants) -> ScoreArrays:
    """只搬裝置，不降精度；一份供起點與所有候選共用。"""
    return tuple(cast(jax.Array, jax.device_put(getattr(constants, name))) for name in score_array_names(constants))


def double_precision_leaf(value: jax.Array) -> jax.Array:
    """只升浮點／複數葉子；旗標與計數保持原型別。"""
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        return value.astype(jnp.complex128)
    if jnp.issubdtype(value.dtype, jnp.floating):
        return value.astype(jnp.float64)
    return value


class SixCategorySinglePrecisionObjective:
    """θ 先轉 float32；kernel（數值鏈）負責在評分前把五個數值輸入升 complex128。

    第二個動態入參是單精度替身；第三個是雙精度六類陣列。結構欄保留在 constants 的閉包副本。
    aosr_compiled_evaluator 供既有搜尋編譯快取直接取用，避免外層 jit 又把裝置常數關回閉包。
    """

    def __init__(self, kernel: Kernel, constants: ReducedConstants, score_constants: SixCategoryConstants, *,
                 device_constants: ReducedConstants | None = None, score_arrays: ScoreArrays | None = None) -> None:
        self.constants = single_precision_constants(constants) if device_constants is None else device_constants
        self.dynamic_names = score_array_names(score_constants)
        self.score_arrays = double_precision_score_arrays(score_constants) if score_arrays is None else score_arrays

        def single(theta: jax.Array, reduced: ReducedConstants, arrays: ScoreArrays) -> tuple[jax.Array, SixCategoryChainAux]:
            # NamedTuple 的建構型別是 numpy；同形狀的 JAX 追蹤陣列在既有核心直接由 jnp.asarray 消費。
            bind = cast(Callable[..., SixCategoryConstants], score_constants._replace)
            scoring = bind(**dict(zip(self.dynamic_names, arrays, strict=True)))
            with jax.default_matmul_precision('highest'):
                value, aux = kernel(theta.astype(jnp.float32), reduced, scoring)
                return value.astype(jnp.float64), jax.tree.map(double_precision_leaf, aux)

        self.forward = jax.jit(single)
        self.compiled = jax.jit(jax.value_and_grad(single, has_aux=True))
        self.aosr_compiled_evaluator = self.evaluate

    def __call__(self, theta: jax.Array) -> tuple[jax.Array, SixCategoryChainAux]:
        return cast('tuple[jax.Array, SixCategoryChainAux]', self.forward(theta, self.constants, self.score_arrays))

    def evaluate(self, theta: jax.Array) -> Evaluation:
        """搜尋共用的值／aux／梯度入口；外部 θ 與梯度維持雙精度。"""
        return cast(Evaluation, self.compiled(theta, self.constants, self.score_arrays))
