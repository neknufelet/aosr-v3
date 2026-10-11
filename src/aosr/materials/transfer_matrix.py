"""JAX 版垂直入射傳遞矩陣（#560 第一步與第二步）。

改寫來源：https://github.com/rinaldipp/tmm
commit b2e9a90f55fcc985b887a28a094a121fcd0c99b3，TMM.equivalent_fluid_model、
porous_layer、air_layer、_compute_global_matrix、reflection_and_absorption_coefficient。
第二步沿用 constant_z、membrane_layer、porous_facing_layer、perforated_panel_layer、
slotted_panel_layer、viscothermal_circular 與 _bessel_ingard_circular_sheet_impedance。

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

第二步連續參數亦為 JAX 陣列；method、end_correction、free 與 None 的有無為靜態。
呼叫端供正孔徑/孔距/縫寬/厚度、0<開孔率<=1、正面密度、非負面流阻，
Rebillard 還須 0<孔隙率<=1 與正曲折度。constant_z 可為零或複數。
原碼 constant_z 是直接終端，這裡用秩一終端矩陣表示，可接在層疊最末端，
不能當壓力跳躍薄片；它前面的層仍按同一順序連乘。eq_fluid 原碼忽略 rho，
故這一支亦忽略 surface_mass_kg_m2；其他三支保留原式的板質量並聯。

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
from typing import cast

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


def _sheet_layer(impedance: jax.Array) -> jax.Array:
    """壓力跳躍薄片 [[1,Z],[0,1]]；面積一，使用比阻抗。"""
    ones, zeros = jnp.ones_like(impedance), jnp.zeros_like(impedance)
    return jnp.stack((jnp.stack((ones, impedance), axis=-1),
                      jnp.stack((zeros, ones), axis=-1)), axis=-2)


def constant_z(frequencies_hz: jax.Array, impedance_pa_s_m: jax.Array) -> jax.Array:
    """定值比表面阻抗 [Pa·s/m] 的秩一終端 [[Z,0],[1,0]]，不是薄片。

    原套件直接指定 Zs，並清除原有層。此矩陣接在層疊最末端即可提供同一終端，
    單獨送 compute 亦回傳指定 Zs；零阻抗也有限。頻率只決定廣播形狀。
    """
    z = jnp.asarray(impedance_pa_s_m) + jnp.zeros_like(jnp.asarray(frequencies_hz))
    zeros = jnp.zeros_like(z)
    return jnp.stack((jnp.stack((z, zeros), axis=-1),
                      jnp.stack((jnp.ones_like(z), zeros), axis=-1)), axis=-2)


def membrane_layer(frequencies_hz: jax.Array, surface_mass_kg_m2: jax.Array) -> jax.Array:
    """柔膜層，面密度 [kg/m²]；Z=jωm，不含彎曲剛度或支撐損失。"""
    return _sheet_layer(1j * 2 * jnp.pi * jnp.asarray(frequencies_hz) * jnp.asarray(surface_mass_kg_m2))


@jax.jit
def _compiled_circular_bessel_terms(z: jax.Array) -> tuple[jax.Array, jax.Array]:
    """直接求 q=J1(z)/(z J0(z)) 和 D=1−2q，避免大數與小引數消去。

    反向連分式 q=1/(2−z²/(4−z²/(6−…)))，固定 256 項，從尾端零開始。
    兩條下半平面射線上的分母有限，不計算 J0 或 J1。D=−tail/(2−tail)
    在 z→0 保留 O(z²)，不以 1−2q 相減丟精度；z=0 時 q=1/2、D=0。
    β=1e-3…1024 的兩條射線對 SciPy 縮放比值實測約 2e-15 相對差；
    192 項於 β=1024 尚有約 1.3e-11 截斷差，選 256 項壓至雙精度舍入。
    此精度證據限該範圍，不宣稱任意複數或無限大引數。固定迴圈支援 jit/vmap/grad。
    """
    argument = jnp.asarray(z)
    square = argument * argument
    def backward(index: int, tail: jax.Array) -> jax.Array:
        order = 256 - index
        return square / (2 * order - tail)
    tail = jax.lax.fori_loop(0, 255, backward, jnp.zeros_like(argument))
    denominator = 2 - tail
    return 1 / denominator, -tail / denominator


def _circular_bessel_terms(z: jax.Array) -> tuple[jax.Array, jax.Array]:
    """保留編譯函式的回傳型別；JAX 的 jit 包裝器型別資訊未保留原簽章。"""
    return cast(tuple[jax.Array, jax.Array], _compiled_circular_bessel_terms(z))


def _bessel_j1_over_z_j0(z: jax.Array) -> jax.Array:
    """私有複數比值；原式需要 J1/J0 時乘回 z，不另求兩個貝索值。"""
    return _circular_bessel_terms(z)[0]


def _circular_open_area(diameter: jax.Array, spacing: jax.Array | None,
                        open_area: jax.Array | None) -> jax.Array:
    if open_area is not None:
        return jnp.asarray(open_area)
    if spacing is None:
        raise ValueError("open_area or hole_spacing_m is required")
    return jnp.pi / (2 * jnp.asarray(spacing) / diameter)**2


def _facing_parameters(thickness: jax.Array, mass: jax.Array | None,
                       resistance: jax.Array | None, *, air: PhysicsConstants, method: str,
                       density: jax.Array | None, diameter: jax.Array | None,
                       spacing: jax.Array | None, open_area: jax.Array | None,
                       free: bool) -> tuple[jax.Array, jax.Array]:
    if method == 'rebillard':
        if mass is None and density is None:
            mass = jnp.asarray(0.25)
        if resistance is None and diameter is None:
            resistance = jnp.asarray(200.0)
    if mass is None and density is not None:
        mass = jnp.asarray(density) * thickness
    if resistance is None:
        if diameter is None:
            raise ValueError("resistance_pa_s_m or hole_diameter_m and open_area/pitch is required")
        diameter = jnp.asarray(diameter)
        phi = _circular_open_area(diameter, spacing, open_area)
        resistance = 32 * air.air_viscosity_pa_s * thickness / (phi * diameter**2)
    if mass is None:
        if method != 'pieren' or free:
            raise ValueError("surface_mass_kg_m2 or density_kg_m3 is required")
        mass = jnp.asarray(0.0)  # 固定面層不使用質量。
    return jnp.asarray(mass), jnp.asarray(resistance)


def porous_facing_layer(
    frequencies_hz: jax.Array, thickness_m: jax.Array, *, air: PhysicsConstants,
    surface_mass_kg_m2: jax.Array | None = None, resistance_pa_s_m: jax.Array | None = None,
    porosity: jax.Array | float = 0.7, tortuosity: jax.Array | float = 1.0,
    method: str = 'rebillard', free: bool = True, density_kg_m3: jax.Array | None = None,
    hole_diameter_m: jax.Array | None = None, hole_spacing_m: jax.Array | None = None,
    open_area: jax.Array | None = None,
) -> jax.Array:
    """薄多孔面層；厚度/等效孔徑/孔距 [m]，面密度 [kg/m²]，面流阻 [Pa·s/m]。

    Rebillard 耦合式或 Pieren 的自由並聯/固定面流阻；保留 aliases 與原碼名義
    面密度 0.25、面流阻 200。未提供質量時可由密度×厚度推導；未提供面流阻
    可由圓孔 32µt/(φd²) 推導。面流阻不是體積流阻率 [Pa·s/m²]。
    """
    method = method.lower()
    if method in ('pieren_free', 'pieren_fixed'):
        free = method == 'pieren_free'
        method = 'pieren'
    if method not in ('rebillard', 'pieren'):
        raise ValueError("unknown porous-facing method")
    t = jnp.asarray(thickness_m)
    mass, resistance = _facing_parameters(t, surface_mass_kg_m2, resistance_pa_s_m, air=air,
        method=method, density=density_kg_m3, diameter=hole_diameter_m, spacing=hole_spacing_m,
        open_area=open_area, free=free)
    omega = 2 * jnp.pi * jnp.asarray(frequencies_hz)
    if method == 'pieren':
        jw_mass = 1j * omega * mass
        z = jw_mass * resistance / (jw_mass + resistance) if free else resistance + jnp.zeros_like(omega, dtype=complex)
    else:
        h, tau = jnp.asarray(porosity), jnp.asarray(tortuosity)
        rho2 = h * air.air_density_kg_m3
        coupling = 1j * omega * (tau - 1) * rho2 * t + resistance * h**2
        c = 1j * omega * rho2 * t + coupling
        b = 1j * omega * mass + coupling
        denominator = h * (coupling * (1 - h) + b * h) + (1 - h) * (coupling * h + c * (1 - h))
        z = (c * b - coupling**2) / denominator
    return _sheet_layer(z)


def _corrected_hole_length(t: jax.Array, d: jax.Array, phi: jax.Array,
                           end_correction: str | None) -> jax.Array:
    option = 'jb' if end_correction is None else end_correction
    if option == 'nesterov':
        delta = 0.85 * (1 - 1.47 * phi**0.5 + 0.47 * phi**1.5)
    elif option in ('jb', 'jaouen_becot'):
        delta = 0.85 * (1 - 1.13 * phi**0.5 - 0.09 * phi + 0.27 * phi**1.5)
    elif option == 'beranek':
        return t + 0.85 * d
    else:
        raise ValueError("unknown circular-hole end correction")
    return 2 * delta * d / 2 + t


def _parallel_plate_mass(z: jax.Array, omega: jax.Array, surface_mass: jax.Array | None,
                         solid_fraction: jax.Array | float) -> jax.Array:
    if surface_mass is None:
        return z
    mass_z = 1j * omega * jnp.asarray(surface_mass) * solid_fraction
    return z * mass_z / (z + mass_z)


def _bessel_ingard_circular_sheet_impedance(t: jax.Array, d: jax.Array, phi: jax.Array,
                                           omega: jax.Array, air: PhysicsConstants) -> jax.Array:
    radius = d / 2
    kd = radius * jnp.sqrt(air.air_density_kg_m3 * omega / air.air_viscosity_pa_s)
    aperture_factor = _circular_bessel_terms(kd * jnp.sqrt(jnp.asarray(-1j)))[1]
    z_aperture = 1j * omega * air.air_density_kg_m3 * t / (phi * aperture_factor)
    hole_area = jnp.pi * radius**2
    xi = jnp.sqrt(phi)
    end_length = 0.96 * jnp.sqrt(hole_area) * (1 - 1.24 * xi + 0.27 * xi**2)
    z_end = 1j * omega * air.air_density_kg_m3 * end_length / phi
    z_end_resistance = jnp.sqrt(2.0) * kd * air.air_viscosity_pa_s / (2 * radius * phi)
    return z_aperture + z_end + z_end_resistance


def perforated_panel_layer(
    frequencies_hz: jax.Array, thickness_m: jax.Array, hole_diameter_m: jax.Array, *,
    air: PhysicsConstants, hole_spacing_m: jax.Array | None = None,
    open_area: jax.Array | None = None, surface_mass_kg_m2: jax.Array | None = None,
    end_correction: str | None = None, method: str = 'barrier',
) -> jax.Array:
    """圓孔板四算法；厚度/孔徑/方形孔距 [m]，完整表面面密度 [kg/m²]。

    barrier、barrier_mpp 為集中薄片，並聯質量按 (1−φ) 加權；bessel_ingard
    為貝索短管薄片，並聯質量用完整表面。eq_fluid 是有限黏熱孔道，原碼
    忽略板密度，故此支忽略面密度。末端修正只用於 barrier/eq_fluid；其他
    兩支依原碼忽略該選項。所有 φ 為比例而非百分比，預設由方形孔距推導。
    """
    if method not in ('barrier', 'barrier_mpp', 'bessel_ingard', 'eq_fluid'):
        raise ValueError("unknown perforated-panel method")
    t, d = jnp.asarray(thickness_m), jnp.asarray(hole_diameter_m)
    phi = _circular_open_area(d, hole_spacing_m, open_area)
    omega = 2 * jnp.pi * jnp.asarray(frequencies_hz)
    viscosity, density = air.air_viscosity_pa_s, air.air_density_kg_m3
    if method in ('barrier', 'eq_fluid'):
        length = _corrected_hole_length(t, d, phi, end_correction)
        if method == 'eq_fluid':
            kc, zc = viscothermal_circular(frequencies_hz, d, air=air)
            return _fluid_layer(kc, zc / phi, length)
        resistance = jnp.sqrt(8 * viscosity * density * omega) * (1 + t / d) / phi
        z = 1j * length * omega * density / phi + resistance
    elif method == 'barrier_mpp':
        cis = d * jnp.sqrt(omega * density / (4 * viscosity))
        kr = jnp.sqrt(1 + cis**2 / 32) + jnp.sqrt(2.0) / 32 * cis * d / t
        resistance = 32 * viscosity / phi * t / d**2 * kr
        km = 1 + 1 / jnp.sqrt(1 + cis**2 / 2) + 0.85 * d / t
        z = resistance + 1j * omega * density * t / phi * km
    else:
        z = _bessel_ingard_circular_sheet_impedance(t, d, phi, omega, air)
    solid_fraction = 1.0 if method == 'bessel_ingard' else 1 - phi
    return _sheet_layer(_parallel_plate_mass(z, omega, surface_mass_kg_m2, solid_fraction))


def slotted_panel_layer(
    frequencies_hz: jax.Array, thickness_m: jax.Array, slot_width_m: jax.Array, *,
    air: PhysicsConstants, slot_spacing_m: jax.Array | None = None,
    open_area: jax.Array | None = None, surface_mass_kg_m2: jax.Array | None = None,
    method: str = 'barrier',
) -> jax.Array:
    """無限長週期長縫板；原碼僅 barrier。厚度/縫寬/縫距 [m]、面密度 [kg/m²]。

    φ=w/s 或直接提供。末端修正 t−2w/π·log(sin(πφ/2))，集中黏性面流阻
    [Pa·s/m] 加聲學質量；板質量按 (1−φ) 並聯，質量輸入是完整表面。
    """
    if method != 'barrier':
        raise ValueError("slotted-panel method must be barrier")
    t, w = jnp.asarray(thickness_m), jnp.asarray(slot_width_m)
    if open_area is None:
        if slot_spacing_m is None:
            raise ValueError("open_area or slot_spacing_m is required")
        phi = w / jnp.asarray(slot_spacing_m)
    else:
        phi = jnp.asarray(open_area)
    omega = 2 * jnp.pi * jnp.asarray(frequencies_hz)
    length = t + 2 * w * (-1 / jnp.pi) * jnp.log(jnp.sin(0.5 * jnp.pi * phi))
    resistance = .5 * jnp.sqrt(2 * air.air_viscosity_pa_s * air.air_density_kg_m3 * omega) * (4 + 2 * t / w)
    z = (resistance + 1j * omega * air.air_density_kg_m3 * length) / phi
    return _sheet_layer(_parallel_plate_mass(z, omega, surface_mass_kg_m2, 1 - phi))


def viscothermal_circular(
    frequencies_hz: jax.Array, hole_diameter_m: jax.Array, *, air: PhysicsConstants,
    open_area: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """圓孔內黏熱傳播 kc [rad/m]、zc [Pa·s/m]；孔徑 [m]，不是板層。

    密度用 √(−j)β 射線，體積彈性模數用 √(−jPr)β 射線。保留原碼選填
    open_area 的相容縮放：只令 zc/φ，通常應由穿孔板層套用。需要注入
    比熱比、普朗特數與大氣壓，不在材料層讀設定或默填物性。
    """
    if air.prandtl_number is None or air.atmospheric_pressure_pa is None:
        raise ValueError("prandtl_number and atmospheric_pressure_pa are required")
    omega = 2 * jnp.pi * jnp.asarray(frequencies_hz)
    beta = jnp.asarray(hole_diameter_m) / 2 * jnp.sqrt(omega * air.air_density_kg_m3 / air.air_viscosity_pa_s)
    viscous = beta * jnp.sqrt(jnp.asarray(-1j))
    thermal = beta * jnp.sqrt(jnp.asarray(-1j * air.prandtl_number))
    effective_density = air.air_density_kg_m3 / _circular_bessel_terms(viscous)[1]
    gamma = _heat_ratio(air)
    effective_bulk = air.atmospheric_pressure_pa * gamma / (1 + (gamma - 1) * 2 * _bessel_j1_over_z_j0(thermal))
    kc = omega * jnp.sqrt(effective_density / effective_bulk)
    zc = jnp.sqrt(effective_density * effective_bulk)
    if open_area is not None:
        zc = zc / jnp.asarray(open_area)
    return kc, zc
