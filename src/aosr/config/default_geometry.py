"""Default shoebox seed geometry — SSOT for scripts' fallback/seed positions.

NOT user-runtime config (CHORAS supplies real positions via JSON); these are the
fallback used when no positions are provided and the seed for dev figure scripts.
Plain nested lists/tuples (caller wraps in jnp.array) so this module stays
jax-free and import-cheap.
"""

DEFAULT_DIMS_M: tuple[float, float, float] = (5.5, 4.2, 2.8)
DEFAULT_SOURCE_XYZ: list[list[float]] = [[1.2, 2.8, 1.4]]   # one-side speaker, K=1
DEFAULT_RECEIVER_XYZ: list[list[float]] = [
    [3.2, 2.1, 1.2],
    [3.5, 1.7, 1.2],
    [3.5, 2.5, 1.2],
]
DEFAULT_EAR_HEIGHT_M: float = 1.2
