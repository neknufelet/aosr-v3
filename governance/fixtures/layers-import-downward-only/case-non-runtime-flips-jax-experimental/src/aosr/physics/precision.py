"""違規：第三層翻 JAX 的全域精度開關。"""
import jax

jax.experimental.enable_x64()
