"""三種動態取模組 ＋ 另一個 JAX 全域開關入口。"""
import importlib
import sys

import jax

jax.experimental.enable_x64()

ONE = importlib.import_module("aosr.physics.solver")
TWO = __import__("aosr.physics.solver")
THREE = sys.modules["aosr.physics"]
