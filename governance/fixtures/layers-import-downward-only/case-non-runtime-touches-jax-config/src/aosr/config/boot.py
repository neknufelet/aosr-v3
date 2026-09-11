"""違規：第二層碰 JAX 的全域設定，而且名字被改過。"""
from jax import config as jc

jc.update("jax_enable_x64", True)
