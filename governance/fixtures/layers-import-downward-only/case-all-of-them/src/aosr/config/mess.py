"""下層引上層 ＋ 碰 JAX 全域設定。"""
import jax

from aosr.physics import solver

jax.config.update("jax_enable_x64", True)

VALUE = solver
