"""搜尋單精度鏈：裝置常數明確入參、區域最高乘法精度、登記簿精度閘門。"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict

from aosr.config.precision_contracts import default_precision_contracts_path, load_precision_contracts
from aosr.physics.fem_reduced_jax import ReducedConstants
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays

Device = Literal["cpu", "gpu"]
CONTRACT: Final = "rom_search_gpu_single_vs_cpu_double"
SIX_CATEGORY_CONTRACT: Final = "six_category_search_gpu_single_vs_cpu_double"
Kernel = Callable[[jax.Array, ReducedConstants], tuple[jax.Array, GradientObjectiveArrays]]
Evaluation = tuple[tuple[jax.Array, GradientObjectiveArrays], jax.Array]


def single_precision_constants(constants: ReducedConstants) -> ReducedConstants:
    """一次轉換並放上裝置；呼叫端持有這一包，供整次搜尋的目標共用。"""
    def put(value: np.ndarray | jax.Array) -> jax.Array:
        dtype = jnp.complex64 if np.issubdtype(value.dtype, np.complexfloating) else (
            jnp.bool_ if np.issubdtype(value.dtype, np.bool_) else jnp.float32)
        return cast(jax.Array, jax.device_put(jnp.asarray(value, dtype=dtype)))
    return cast(ReducedConstants, jax.tree.map(put, constants))


class SinglePrecisionObjective:
    """只有鏈用單精度；外部 θ、回傳值與梯度為雙精度。常數不關入 jit 的閉包。

    compiled（編譯函式）的第二個入參是整包裝置陣列，無須依賴 JAX 的常數提升開關。
    有範圍的乘法設定只在追蹤這條鏈時生效，涵蓋身分閉包內的 TMM 與求解反傳。
    """

    def __init__(self, kernel: Kernel, constants: ReducedConstants, *,
                 device_constants: ReducedConstants | None = None) -> None:
        self.constants = single_precision_constants(constants) if device_constants is None else device_constants

        def single(theta: jax.Array, arrays: ReducedConstants) -> tuple[jax.Array, GradientObjectiveArrays]:
            with jax.default_matmul_precision("highest"):
                value, aux = kernel(theta.astype(jnp.float32), arrays)
                return value.astype(jnp.float64), jax.tree.map(lambda x: x.astype(jnp.float64), aux)

        self.forward = jax.jit(single)
        self.compiled = jax.jit(jax.value_and_grad(single, has_aux=True))

    def __call__(self, theta: jax.Array) -> tuple[jax.Array, GradientObjectiveArrays]:
        return cast(tuple[jax.Array, GradientObjectiveArrays], self.forward(theta, self.constants))

    def evaluate(self, theta: jax.Array) -> Evaluation:
        """下降迴圈用的值＋梯度；大常數在最外層編譯邊界作為動態入參。"""
        return cast(Evaluation, self.compiled(theta, self.constants))


class LegacyPrecisionComparison(BaseModel):
    """同一個起點的 CPU 雙精度與顯卡單精度目標值；連同 θ 保存於 project.json。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    contract: Literal["rom_search_gpu_single_vs_cpu_double"] = CONTRACT
    cpu_double: float
    gpu_single: float
    absolute_difference_db: float
    theta_hex: tuple[str, ...]


def precision_comparison(*, cpu_double: float, gpu_single: float, theta: Sequence[float]) -> LegacyPrecisionComparison:
    """保留兩邊原始值及絕對差；非有限結果直接拒收。"""
    return LegacyPrecisionComparison(cpu_double=cpu_double, gpu_single=gpu_single,
        absolute_difference_db=abs(gpu_single - cpu_double), theta_hex=tuple(float(x).hex() for x in theta))


def require_precision(comparison: LegacyPrecisionComparison | PrecisionComparison) -> None:
    """唯一的尺從登記簿讀取，超過就拒跑；也核對記錄的差沒有被改寫。"""
    if isinstance(comparison, PrecisionComparison):
        _require_six_category_precision(comparison)
        return
    limit = load_precision_contracts(default_precision_contracts_path())[CONTRACT].value
    difference = abs(comparison.gpu_single - comparison.cpu_double)
    if difference != comparison.absolute_difference_db or difference > limit:
        raise ValueError(f"顯卡單精度 objective_total 絕對差 {difference!r} dB 超過或不符登記尺 {limit!r} dB："
                         f"CPU 雙精度={comparison.cpu_double!r}，GPU 單精度={comparison.gpu_single!r}；拒絕搜尋")


type SixCosts = tuple[float, float, float, float, float, float]


class PrecisionReading(BaseModel):
    """同次求值的三組原值；供獨立 CPU 參考行程交換。"""

    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    objective_total: float
    product_total: float
    categories: SixCosts


class PrecisionComparison(BaseModel):
    """六類同 θ 的無因次比較；只有 objective_total（搜尋 J）決定閘門。"""

    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    contract: Literal['six_category_search_gpu_single_vs_cpu_double'] = SIX_CATEGORY_CONTRACT
    cpu_double: float
    gpu_single: float
    absolute_difference: float
    theta_hex: tuple[str, ...]
    cpu_product: float
    gpu_product: float
    product_absolute_difference: float
    cpu_categories: SixCosts
    gpu_categories: SixCosts
    category_absolute_differences: SixCosts


def six_category_precision_comparison(*, cpu_double: float, gpu_single: float, theta: Sequence[float],
                                     cpu_product: float, gpu_product: float,
                                     cpu_categories: Sequence[float], gpu_categories: Sequence[float]) -> PrecisionComparison:
    """同次求值的 J、產品總分與六類原值及差；不替診斷量新訂門檻。"""
    return PrecisionComparison.model_validate(dict(cpu_double=cpu_double, gpu_single=gpu_single,
        absolute_difference=abs(gpu_single-cpu_double), theta_hex=tuple(float(x).hex() for x in theta),
        cpu_product=cpu_product, gpu_product=gpu_product, product_absolute_difference=abs(gpu_product-cpu_product),
        cpu_categories=tuple(cpu_categories), gpu_categories=tuple(gpu_categories),
        category_absolute_differences=tuple(abs(a-b) for a,b in zip(cpu_categories,gpu_categories,strict=True))))


def _require_six_category_precision(comparison: PrecisionComparison) -> None:
    """逐欄核記錄沒被改寫，再只以 J 比新尺；舊 dB 契約從不參與這條路。"""
    limit = load_precision_contracts(default_precision_contracts_path())[SIX_CATEGORY_CONTRACT].value
    difference = abs(comparison.gpu_single-comparison.cpu_double)
    categories = tuple(abs(a-b) for a,b in zip(comparison.cpu_categories,comparison.gpu_categories,strict=True))
    if (difference != comparison.absolute_difference or difference > limit
            or abs(comparison.gpu_product-comparison.cpu_product) != comparison.product_absolute_difference
            or categories != comparison.category_absolute_differences):
        raise ValueError(f'顯卡單精度 objective_total 絕對差 {difference!r}（absolute，無因次）超過或不符登記尺 {limit!r}；'
                         f'CPU 雙精度={comparison.cpu_double!r}，GPU 單精度={comparison.gpu_single!r}；拒絕搜尋')
