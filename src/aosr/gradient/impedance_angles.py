"""共用的空入射角入口；沒有反射路徑的面片不呼叫材料求值器。"""
from collections.abc import Callable

import jax
import jax.numpy as jnp


def impedance_at_angles_or_empty(axis: jax.Array, angles: jax.Array | None,
                                 calculate: Callable[[], jax.Array]) -> jax.Array:
    """空角表回 (F, 0) 複數列，精度跟頻率軸走；非空與法向入口原樣求值。"""
    if angles is not None and angles.size == 0:
        return jnp.empty((axis.size, 0), dtype=jnp.result_type(axis.dtype, jnp.complex64))
    return calculate()
