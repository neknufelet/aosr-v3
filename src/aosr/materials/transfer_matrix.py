"""JAX 版多孔材與空氣層的垂直入射傳遞矩陣（#560 第一步）。

改寫來源：https://github.com/rinaldipp/tmm
commit b2e9a90f55fcc985b887a28a094a121fcd0c99b3，TMM.equivalent_fluid_model、
porous_layer、air_layer、_compute_global_matrix、reflection_and_absorption_coefficient。

時間慣例沿用原套件 e^{jωt}，前進波 e^{-jkx}、矩陣非對角項 +j sin(kt)。
層按「入射面 → 剛性背牆」排列並從左到右連乘。面積 s0=1，剛性背牆 Zs=A/C，
交出比阻抗 Pa·s/m；厚度 m、流阻 Pa·s/m²、頻率 Hz。矩陣形狀 (..., 2, 2)。
不共軛、不裁切吸收係數、不接引擎計算。MaterialResponse.from_impedance 可裝結果，
時間慣例轉接屬後續 #309。空氣條件由呼叫端注入 PhysicsConstants，不在材料層讀設定檔。

連續參數用 JAX 陣列運算；模型名、纖維種類與層數為 Python 靜態選項。
呼叫端負責提供有限正頻率、正流阻、非負厚度、正孔隙率與正曲折度；
mechel_1976 的孔隙率還須不大於 1。超出經驗適用範圍仍沿用原公式，並非準確性宣稱。
C=60 等號用經驗支；分界本身不光滑，grad 回選中支的導數。兩支在合法域皆有限，
因此 jnp.where 未選支不會以 NaN 污染梯度。wilson_db 的 0.4 是固定經驗係數
（原式 gamma=1.4 的 gamma-1），不是環境比熱比。

MIT License

Copyright (c) 2021 Rinaldi Polese Petrolli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from aosr.config.physics_constants import PhysicsConstants


_COEFFICIENTS: dict[str, tuple[float, ...]] = {
    "db": (0.0978, 0.7, 0.189, 0.595, 0.0571, 0.754, 0.087, 0.732),
    "miki": (0.122, 0.618, 0.18, 0.618, 0.079, 0.632, 0.12, 0.632),
    "qunli": (0.188, 0.554, 0.163, 0.592, 0.209, 0.548, 0.105, 0.607),
    "mechel_gf_lowX": (0.135, 0.646, 0.396, 0.458, 0.0688, 0.707, 0.196, 0.549),
    "mechel_gf_highX": (0.102, 0.705, 0.179, 0.674, 0.0235, 0.887, 0.0875, 0.77),
    "mechel_rf_lowX": (0.136, 0.641, 0.322, 0.502, 0.081, 0.699, 0.191, 0.556),
    "mechel_rf_highX": (0.103, 0.716, 0.179, 0.663, 0.0563, 0.725, 0.127, 0.655),
    "mac": (0.0982, 0.685, 0.288, 0.526, 0.0729, 0.66228, 0.187, 0.5379),
}
_GRUNDMANN: dict[int, tuple[tuple[complex, ...], tuple[complex, ...]]] = {
    1: (
        (-0.00355757 - 0.0000164897j, 0.421329 + 0.342011j, -0.507733 + 0.086655j,
         -0.142339 + 1.25986j, 1.29048 - 0.0820811j, -0.771857 - 0.668050j),
        (0.0026786 + 0.00385761j, 0.135298 - 0.394160j, 0.946702 + 1.47653j,
         -1.45202 - 4.56233j, 4.03171 + 7.56031j, -2.86993 - 4.90437j),
    ),
    2: (
        (-0.00451836 + 0.000541333j, 0.421987 + 0.376270j, -0.383809 - 0.353780j,
         -0.610867 + 2.59922j, 1.13341 - 1.74819j, 0j),
        (-0.00171387 + 0.00119489j, 0.283876 - 0.292168j, -0.463860 + 0.188081j,
         3.12736 + 0.941600j, -2.10920 - 1.32398j, 0j),
    ),
}


def _heat_ratio(air: PhysicsConstants) -> float:
    if air.specific_heat_ratio is None:
        raise ValueError("specific_heat_ratio is required by this model")
    return air.specific_heat_ratio


def _mechel_1976(c: jax.Array, k0: jax.Array, air: PhysicsConstants,
                 porosity: jax.Array) -> tuple[jax.Array, jax.Array]:
    """C>60 用低頻延伸；複數平方根的未選支在正 C、正孔隙率下亦有限。"""
    empirical_gamma = 0.189 * c**0.6185 + 1j * (1 + 0.0978 * c**0.6929)
    empirical_z = 1 + 0.04885 * c**0.754 - 1j * 0.087 * c**0.7307
    low_gamma = jnp.sqrt(-1.4664 + 1j * (_heat_ratio(air) * porosity / (2 * jnp.pi)) * c)
    low_z = (c / (2 * jnp.pi) + 1j * (4 / (3 * porosity))) / low_gamma
    gamma = jnp.where(c > 60.0, low_gamma, empirical_gamma)
    z = jnp.where(c > 60.0, low_z, empirical_z)
    return -1j * k0 * gamma, air.rho_c * z


def _wilson_2015(frequency: jax.Array, sigma: jax.Array, k0: jax.Array,
                 air: PhysicsConstants, porosity: jax.Array,
                 tortuosity: jax.Array) -> tuple[jax.Array, jax.Array]:
    if air.prandtl_number is None:
        raise ValueError("prandtl_number is required by wilson_2015")
    q = jnp.sqrt(tortuosity)
    tau0 = 2 * air.air_density_kg_m3 * tortuosity / (porosity * sigma)
    omega_tau0 = 2 * jnp.pi * frequency * tau0
    relax_vor = 1 / jnp.sqrt(1 + 1j * omega_tau0)
    relax_ent = 1 / jnp.sqrt(1 + 1j * air.prandtl_number * omega_tau0)
    density_term = -2j / omega_tau0 + 1 + relax_vor
    compressibility_term = 1 + (_heat_ratio(air) - 1) * relax_ent
    zc = air.rho_c * (q / porosity) * jnp.sqrt(density_term / compressibility_term)
    kc = k0 * q * jnp.sqrt(density_term * compressibility_term)
    return kc, zc


def _grundmann(x: jax.Array, k0: jax.Array, air: PhysicsConstants,
               fibre_type: int) -> tuple[jax.Array, jax.Array]:
    if fibre_type not in _GRUNDMANN:
        raise ValueError("fibre_type must be 1 (rock wool) or 2 (glass fibre)")
    betak, betaz = _GRUNDMANN[fibre_type]
    def polynomial(beta: tuple[complex, ...]) -> jax.Array:
        return (beta[0] * x**-1 + beta[1] * x**-0.5 + beta[2] + beta[3] * x**0.5
                + beta[4] * x + beta[5] * x**1.5)
    return -1j * k0 * polynomial(betak), air.rho_c * polynomial(betaz)


def equivalent_fluid_model(
    frequencies_hz: jax.Array, flow_resistivity_pa_s_m2: jax.Array, *,
    air: PhysicsConstants, model: str = "mac", fibre_type: int = 1,
    porosity: jax.Array | float = 0.95, tortuosity: jax.Array | float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """回傳複數波數 kc [rad/m]、特性比阻抗 zc [Pa·s/m]；模型選項為靜態。"""
    frequency = jnp.asarray(frequencies_hz)
    sigma = jnp.asarray(flow_resistivity_pa_s_m2)
    h, tort = jnp.asarray(porosity), jnp.asarray(tortuosity)
    k0 = 2 * jnp.pi * frequency / air.sound_speed_m_s
    x = air.air_density_kg_m3 * frequency / sigma
    if model in _COEFFICIENTS:
        c = _COEFFICIENTS[model]
        kc = k0 * (1 + c[0] * x**-c[1] - 1j * c[2] * x**-c[3])
        zc = air.rho_c * (1 + c[4] * x**-c[5] - 1j * c[6] * x**-c[7])
        return kc, zc
    if model == "komatsu":
        y = 2 - jnp.log10(frequency / sigma)
        return (k0 * (1 + 0.0004 * y**6.2 - 1j * 0.0069 * y**4.1),
                air.rho_c * (1 + 0.00027 * y**6.2 - 1j * 0.0047 * y**4.1))
    if model == "mechel_1976":
        return _mechel_1976(sigma / (air.air_density_kg_m3 * frequency), k0, air, h)
    if model == "wilson_db":
        entropy = 1 + 0.4 / jnp.sqrt(1 + 1j * 19 * x)
        viscosity = 1 - 1 / jnp.sqrt(1 + 1j * 13 * x)
        return k0 * jnp.sqrt(entropy / viscosity), (air.rho_c / h) / jnp.sqrt(entropy * viscosity)
    if model == "wilson_2015":
        return _wilson_2015(frequency, sigma, k0, air, h, tort)
    if model == "mechel_grundmann":
        return _grundmann(x, k0, air, fibre_type)
    raise ValueError(f"unknown equivalent-fluid model: {model}")


def _fluid_layer(kc: jax.Array, zc: jax.Array, thickness_m: jax.Array) -> jax.Array:
    phase = kc * jnp.asarray(thickness_m)
    cosine, sine = jnp.cos(phase), jnp.sin(phase)
    zc = jnp.broadcast_to(zc, phase.shape)
    first = jnp.stack((cosine, 1j * zc * sine), axis=-1)
    second = jnp.stack((1j * sine / zc, cosine), axis=-1)
    return jnp.stack((first, second), axis=-2)


def porous_layer(
    frequencies_hz: jax.Array, thickness_m: jax.Array, flow_resistivity_pa_s_m2: jax.Array, *,
    air: PhysicsConstants, model: str = "mac", fibre_type: int = 1,
    porosity: jax.Array | float = 0.95, tortuosity: jax.Array | float = 1.0,
) -> jax.Array:
    """單一多孔層矩陣；所有連續參數用 SI，沒有 mm 或 kPa 換算。"""
    kc, zc = equivalent_fluid_model(frequencies_hz, flow_resistivity_pa_s_m2,
                                   air=air, model=model, fibre_type=fibre_type,
                                   porosity=porosity, tortuosity=tortuosity)
    return _fluid_layer(kc, zc, thickness_m)


def air_layer(frequencies_hz: jax.Array, thickness_m: jax.Array, *, air: PhysicsConstants) -> jax.Array:
    """無耗空氣層矩陣；ρc 從注入的空氣密度與聲速推導。"""
    k0 = 2 * jnp.pi * jnp.asarray(frequencies_hz) / air.sound_speed_m_s
    return _fluid_layer(k0, jnp.asarray(air.rho_c), thickness_m)


def global_matrix(layers: Sequence[jax.Array]) -> jax.Array:
    """入射面到背牆的矩陣依序從左到右相乘；層數為靜態。"""
    if not layers:
        raise ValueError("at least one layer is required")
    total = layers[0]
    for layer in layers[1:]:
        total = total @ layer
    return total


def rigid_backing_impedance(matrix: jax.Array) -> jax.Array:
    """剛性背牆的比表面阻抗 A/C [Pa·s/m]；共振極點不另加正則化。"""
    return matrix[..., 0, 0] / matrix[..., 1, 0]


def reflection_and_absorption_coefficient(zs: jax.Array, *, air: PhysicsConstants) -> tuple[jax.Array, jax.Array]:
    """垂直入射 r=(Zs−ρc)/(Zs+ρc)，α=1−|r|²，沿用原套件不裁切。"""
    reflection = (zs - air.rho_c) / (zs + air.rho_c)
    return reflection, 1 - jnp.abs(reflection)**2


def compute(layers: Sequence[jax.Array], *, air: PhysicsConstants) -> tuple[jax.Array, jax.Array, jax.Array]:
    """計算剛性背牆層疊，依序回傳表面比阻抗、反射係數、吸音係數。"""
    zs = rigid_backing_impedance(global_matrix(layers))
    reflection, absorption = reflection_and_absorption_coefficient(zs, air=air)
    return zs, reflection, absorption
