"""JAX 版傳遞矩陣：垂直、斜入射與無規入射（#560 三步）。

改寫來源：https://github.com/rinaldipp/tmm
commit b2e9a90f55fcc985b887a28a094a121fcd0c99b3，TMM.equivalent_fluid_model、
porous_layer、air_layer、_compute_global_matrix、reflection_and_absorption_coefficient。
第二步沿用 constant_z、membrane_layer、porous_facing_layer、perforated_panel_layer、
slotted_panel_layer、viscothermal_circular 與 _bessel_ingard_circular_sheet_impedance。

時間慣例沿用原套件 e^{jωt}，前進波 e^{-jkx}、矩陣非對角項 +j sin(kt)。
層按「入射面 → 背後負載」排列並從左到右連乘。面積 s0=1，剛性背牆 Zs=A/C，
compute 未給負載時沿用剛背；有限負載 ZL 用 Zs=(A·ZL+B)/(C·ZL+D)。
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
constant_z 沿用原碼的整面表面阻抗直接指定，回傳頻率形狀的阻抗，不是矩陣層；
它不接任何既有層。背後負載只走 compute 的 load_impedance_pa_s_m 選項。
eq_fluid 原碼忽略 rho，
故這一支亦忽略 surface_mass_kg_m2；其他三支保留原式的板質量並聯。

沒搬的介面預設：原套件定值阻抗 85 Pa·s/m、膜面密度 8.05 kg/m²（1 mm ×
8050 kg/m³）、板厚/孔徑/孔距 19/8/16 mm（長縫板用縫寬/縫距）。產品要求明給
阻抗、膜面密度、板厚、孔徑或縫寬；孔距或縫距與開孔率須至少給一個。
產品用 SI；面層 Rebillard 的名義估值仍沿用原碼。入射面積固定一；第三步搬斜入射、場入射導納積分與Paris吸音。

未給 incidence_angles_deg 時走前兩步原路徑；給純量或一維角度 [度] 則加一軸：
頻率/參數廣播形狀 F 的矩陣為 F+(Nθ,2,2)，阻抗/反射/吸音為 F+(Nθ,)。
角度是最後一個物理軸，層疊必須使用同一角度格；積分後回 F。薄片與
eq_fluid孔道只廣播角度，照原碼不修正其層阻抗；多孔用kt=k0 sinθ、
kz=√(kc²−kt²)、Zc,n=Zc kc/kz；空氣直接用k0 cosθ，避免近90°平方相消。
角度逐值有限且0≤θ<90；具體值即時拒收，不轉動態角度成float。jit內只用jax.debug.callback盡量拒收：
JAX說明這種回呼在高階轉換下可能被丟掉、重複或重排，不是保證；保證要在jit外的最佳化入口先驗範圍（#561）。
這個回呼要有CPU裝置：顯卡上JAX_PLATFORMS只給cuda時，整支jit呼叫會失敗，要寫成cuda,cpu（2026-10-11 RTX 3060 實測）。
無規範圍為靜態[start,stop,step]，預設field [0,78,1]、paris [0,90,1]，
沿用原碼round與linspace包含端點；stop接近90時用nextafter(90,0)。
近90°不截斷cos、不加epsilon；負載可很大但合法域不除零，完整Paris端點用雙精度。
field對導納乘sinθ作SciPy 1.18.1 Simpson平均，偶數點用Cartwright尾段；
Paris對α sinθ cosθ作梯形平均，只回吸音、不定義複數阻抗。角度範圍是靜態；角度陣列與物理參數可微分。兩者需要至少兩個
嚴格遞增角度點，拒收零寬積分；場入射對阻抗參考平面敏感，頂層空氣會改其答案。
原碼在無規入射且頂層為空氣時發執行期警告；積分函式不知層種類，不重現該警告。
多孔層照原碼不設kz=0防護：個別參數在近掠角可退化成kc=kt，屬繼承限制，合法角度不保證有限。
Simpson本身保留原碼零分母回零，先替換除數才where，未選支梯度亦有限。
背牆明給backing=rigid/air/radiation，與有限load互斥；s0固定1 m²，
radiation_area_m2就是原srad [m²]，預設1。air負載ρc/cosθ/srad；
radiation用圓口半徑√(srad/π)的低ka近似，ρc[0.25(ka)²+j0.6133ka]/srad，
不另加cosθ。矩陣以固定單位入射面積對應體積速度；srad非1是原碼的末端面積換算，
不是改入射面積。輻射模型超出低ka範圍仍照原式算，不宣稱高頻物理精度。

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

simpson、_simpson_segments 與 _safe_divide 的零分母規則改寫自 SciPy 1.18.1
scipy.integrate._quadrature 模組的 simpson、_basic_simpson（BSD-3-Clause）：

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
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
    incidence_angles_deg: jax.Array | float | None = None,
) -> jax.Array:
    """單一多孔層矩陣；所有連續參數用 SI，沒有 mm 或 kPa 換算。"""
    kc, zc = equivalent_fluid_model(frequencies_hz, flow_resistivity_pa_s_m2,
                                   air=air, model=model, fibre_type=fibre_type,
                                   porosity=porosity, tortuosity=tortuosity)
    if incidence_angles_deg is not None:
        return _oblique_porous_layer(kc, zc, frequencies_hz, thickness_m, incidence_angles_deg, air)
    return _fluid_layer(kc, zc, thickness_m)


def air_layer(frequencies_hz: jax.Array, thickness_m: jax.Array, *, air: PhysicsConstants,
              incidence_angles_deg: jax.Array | float | None = None) -> jax.Array:
    """無耗空氣層矩陣；ρc 從注入的空氣密度與聲速推導。"""
    k0 = 2 * jnp.pi * jnp.asarray(frequencies_hz) / air.sound_speed_m_s
    if incidence_angles_deg is not None:
        cosine = jnp.cos(_angles_radians(incidence_angles_deg))
        return _fluid_layer(k0[..., None] * cosine, air.rho_c / cosine, jnp.asarray(thickness_m)[..., None])
    return _fluid_layer(k0, jnp.asarray(air.rho_c), thickness_m)


def global_matrix(layers: Sequence[jax.Array]) -> jax.Array:
    """入射面到負載的 (...,2,2) 矩陣依序相乘；拒收阻抗陣列等非矩陣輸入。"""
    if not layers:
        raise ValueError("at least one layer is required")
    for layer in layers:
        if layer.ndim < 2 or layer.shape[-2:] != (2, 2):
            raise ValueError("each layer must be a (..., 2, 2) transfer matrix; constant_z is a surface impedance")
    total = layers[0]
    for layer in layers[1:]:
        total = total @ layer
    return total


def rigid_backing_impedance(matrix: jax.Array) -> jax.Array:
    """剛性背牆的比表面阻抗 A/C [Pa·s/m]；共振極點不另加正則化。"""
    return matrix[..., 0, 0] / matrix[..., 1, 0]


def reflection_and_absorption_coefficient(zs: jax.Array, *, air: PhysicsConstants,
    incidence_angles_deg: jax.Array | float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """r=(Zs cosθ−ρc)/(Zs cosθ+ρc)、α=1−|r|²；不給角度走原式，不裁切。"""
    if incidence_angles_deg is not None:
        zs = zs * jnp.cos(_angles_radians(incidence_angles_deg))
    reflection = (zs - air.rho_c) / (zs + air.rho_c)
    return reflection, 1 - jnp.abs(reflection)**2


def compute(
    layers: Sequence[jax.Array], *, air: PhysicsConstants,
    load_impedance_pa_s_m: jax.Array | float | complex | None = None,
    incidence_angles_deg: jax.Array | float | None = None, backing: str | None = None,
    frequencies_hz: jax.Array | None = None, radiation_area_m2: jax.Array | float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """層疊回傳表面比阻抗、反射係數、吸音係數；負載 [Pa·s/m] 可廣播。

    None 是剛性背牆 A/C，保留第一步運算；有限負載 Z 用 (A·Z+B)/(C·Z+D)。
    垂直入射空氣背牆可給 air.rho_c；斜入射可給負載形狀 F+(Nθ,)。
    backing 明給 rigid/air/radiation；輻射需頻率與srad面積，詳模組說明。
    constant_z 是整面阻抗，不送入 layers。
    有限負載時 α=1−|r|² 也算進穿透到背後的能量：純空氣接空氣負載得 α=1，不是空氣層吃掉了能量。
    """
    if incidence_angles_deg is not None:
        theta = _angles_radians(incidence_angles_deg)
        if any(layer.ndim < 3 or layer.shape[-3] != theta.size for layer in layers):
            raise ValueError("every layer must carry the matching incidence angle axis")
    matrix = global_matrix(layers)
    if backing is not None:
        if load_impedance_pa_s_m is not None:
            raise ValueError("backing and load_impedance_pa_s_m are mutually exclusive")
        load_impedance_pa_s_m = _backing_load(backing, frequencies_hz, incidence_angles_deg, air, radiation_area_m2)
    if load_impedance_pa_s_m is None:
        zs = rigid_backing_impedance(matrix)
    else:
        load = jnp.asarray(load_impedance_pa_s_m)
        zs = (matrix[..., 0, 0] * load + matrix[..., 0, 1]) / (matrix[..., 1, 0] * load + matrix[..., 1, 1])
    reflection, absorption = reflection_and_absorption_coefficient(zs, air=air, incidence_angles_deg=incidence_angles_deg)
    return zs, reflection, absorption


def _sheet_layer(impedance: jax.Array) -> jax.Array:
    """壓力跳躍薄片 [[1,Z],[0,1]]；面積一，使用比阻抗。"""
    ones, zeros = jnp.ones_like(impedance), jnp.zeros_like(impedance)
    return jnp.stack((jnp.stack((ones, impedance), axis=-1),
                      jnp.stack((zeros, ones), axis=-1)), axis=-2)


def constant_z(frequencies_hz: jax.Array, impedance_pa_s_m: jax.Array, *,
               incidence_angles_deg: jax.Array | float | None = None) -> jax.Array:
    """直接回傳整面的定值比表面阻抗 [Pa·s/m]，按頻率形狀廣播。

    原套件此呼叫清除既有層；本函式不接層、只給 Zs。零或複數都可。
    阻抗須為純量，頻率須為純量或一維；拒收會被誤認成矩陣的二維輸入。
    不是傳遞矩陣層，不送入 compute/global_matrix；背後負載用 compute 的選項。
    """
    frequency, impedance = jnp.asarray(frequencies_hz), jnp.asarray(impedance_pa_s_m)
    if frequency.ndim > 1 or impedance.ndim != 0:
        raise ValueError("constant_z requires a scalar impedance and scalar or 1D frequencies")
    return _broadcast_angles(impedance + jnp.zeros_like(frequency), incidence_angles_deg)


def membrane_layer(frequencies_hz: jax.Array, surface_mass_kg_m2: jax.Array, *,
                   incidence_angles_deg: jax.Array | float | None = None) -> jax.Array:
    """柔膜層，面密度 [kg/m²]；Z=jωm，不含彎曲剛度或支撐損失。"""
    z = 1j * 2 * jnp.pi * jnp.asarray(frequencies_hz) * jnp.asarray(surface_mass_kg_m2)
    return _sheet_layer(_broadcast_angles(z, incidence_angles_deg))


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
    incidence_angles_deg: jax.Array | float | None = None,
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
    return _sheet_layer(_broadcast_angles(z, incidence_angles_deg))


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
    incidence_angles_deg: jax.Array | float | None = None,
) -> jax.Array:
    """圓孔板四算法；厚度/孔徑/方形孔距 [m]，完整表面面密度 [kg/m²]。

    barrier、barrier_mpp 為集中薄片，並聯質量按 (1−φ) 加權；bessel_ingard
    為貝索短管薄片，並聯質量用完整表面。eq_fluid 是有限黏熱孔道，原碼
    忽略板密度，故此支忽略面密度。末端修正只用於 barrier/eq_fluid；其他
    兩支依原碼忽略該選項。所有 φ 為比例而非百分比，預設由方形孔距推導。
    bessel_ingard/eq_fluid 的貝索比值精度只驗到 β≤1024，β=(d/2)√(ωρ/µ)。
    超出時仍固定算 256 項、不拒收，精度會變差；其他兩算法不使用該比值。
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
            if incidence_angles_deg is not None:
                return _fluid_layer(_broadcast_angles(kc, incidence_angles_deg),
                    _broadcast_angles(zc / phi, incidence_angles_deg), length[..., None])
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
    return _sheet_layer(_broadcast_angles(_parallel_plate_mass(z, omega, surface_mass_kg_m2, solid_fraction), incidence_angles_deg))


def slotted_panel_layer(
    frequencies_hz: jax.Array, thickness_m: jax.Array, slot_width_m: jax.Array, *,
    air: PhysicsConstants, slot_spacing_m: jax.Array | None = None,
    open_area: jax.Array | None = None, surface_mass_kg_m2: jax.Array | None = None,
    method: str = 'barrier',
    incidence_angles_deg: jax.Array | float | None = None,
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
    return _sheet_layer(_broadcast_angles(_parallel_plate_mass(z, omega, surface_mass_kg_m2, 1 - phi), incidence_angles_deg))


def viscothermal_circular(
    frequencies_hz: jax.Array, hole_diameter_m: jax.Array, *, air: PhysicsConstants,
    open_area: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """圓孔內黏熱傳播 kc [rad/m]、zc [Pa·s/m]；孔徑 [m]，不是板層。

    密度用 √(−j)β 射線，體積彈性模數用 √(−jPr)β 射線。保留原碼選填
    open_area 的相容縮放：只令 zc/φ，通常應由穿孔板層套用。需要注入
    比熱比、普朗特數與大氣壓，不在材料層讀設定或默填物性。
    貝索比值精度只驗到 β≤1024，β=(d/2)√(ωρ/µ)；超出仍固定算 256 項，
    不拒收也不切換算法，精度會變差。
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


def _require(valid: jax.Array, message: str) -> None:
    """具體值即時拒收；jit/vmap內的動態值只靠debug回呼盡量拒收（JAX不保證執行）；回呼不參與數值或導數。"""
    def check(flag: object) -> None:
        if not bool(flag):
            raise ValueError(message)
    if isinstance(valid, jax.core.Tracer):
        jax.debug.callback(check, valid)
    else:
        check(valid)


def _angles_radians(incidence_angles_deg: jax.Array | float) -> jax.Array:
    angles = jnp.atleast_1d(jnp.asarray(incidence_angles_deg))
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("incidence_angles_deg must be a nonempty scalar or 1D angle grid")
    _require(jnp.all(jnp.isfinite(angles) & (angles >= 0) & (angles < 90)),
             "incidence angles must be finite and 0 <= theta < 90 degrees")
    return jnp.deg2rad(angles)


def _broadcast_angles(values: jax.Array, incidence_angles_deg: jax.Array | float | None) -> jax.Array:
    if incidence_angles_deg is None:
        return values
    theta = _angles_radians(incidence_angles_deg)
    return jnp.broadcast_to(values[..., None], values.shape + (theta.size,))


def _oblique_porous_layer(kc: jax.Array, zc: jax.Array, frequencies_hz: jax.Array,
    thickness_m: jax.Array, incidence_angles_deg: jax.Array | float, air: PhysicsConstants,
) -> jax.Array:
    theta = _angles_radians(incidence_angles_deg)
    k0 = 2 * jnp.pi * jnp.asarray(frequencies_hz) / air.sound_speed_m_s
    kt = k0[..., None] * jnp.sin(theta)
    kc, zc = kc[..., None], zc[..., None]
    kz = jnp.sqrt(kc**2 - kt**2)
    zc_normal = zc * kc / kz
    return _fluid_layer(kz, zc_normal, jnp.asarray(thickness_m)[..., None])


def diffuse_incidence_angles_deg(angle_range: Sequence[float] | None = None, *, method: str = 'field') -> jax.Array:
    """原碼的靜態範圍生成；step是目標間距，round後用含端點linspace。"""
    if method not in ('field', 'paris'):
        raise ValueError("diffuse method must be field or paris")
    try:
        span = [float(v) for v in angle_range] if angle_range is not None else [0., 90. if method == 'paris' else 78., 1.]
    except (TypeError, ValueError):
        raise ValueError("diffuse incidence range must contain finite numeric values") from None
    if len(span) != 3 or not all(math.isfinite(v) for v in span):
        raise ValueError("diffuse incidence range must be finite [start, stop, step]")
    start, stop, step = span
    if step <= 0 or stop <= start or start < 0 or stop > 90:
        raise ValueError("diffuse incidence range requires 0 <= start < stop <= 90 and step > 0")
    grazing = abs(stop - 90.) <= 1e-8 + 1e-5 * 90.
    if grazing:
        stop = math.nextafter(90., 0.)
    intervals = round((stop - start) / step)
    values = [start + i * ((stop - start) / intervals) for i in range(intervals)] + [stop] if intervals else [start]
    result = jnp.asarray(values)
    if grazing and intervals:
        result = result.at[-1].set(jnp.nextafter(jnp.asarray(90., dtype=result.dtype), jnp.asarray(0., dtype=result.dtype)))
    return result


def unflanged_pipe_radiation_load(frequencies_hz: jax.Array, *, air: PhysicsConstants,
    radiation_area_m2: jax.Array | float = 1.0,
) -> jax.Array:
    """原碼低ka圓管口負載；srad [m²]，s0固定1 m²下交Pa·s/m。"""
    area = jnp.asarray(radiation_area_m2)
    _require(jnp.all(jnp.isfinite(area) & (area > 0)), "radiation_area_m2 must be finite and positive")
    ka = 2 * jnp.pi * jnp.asarray(frequencies_hz) / air.sound_speed_m_s * jnp.sqrt(area / jnp.pi)
    return air.rho_c * (0.25 * ka**2 + 1j * 0.6133 * ka) / area


def _backing_load(backing: str, frequencies_hz: jax.Array | None,
    incidence_angles_deg: jax.Array | float | None, air: PhysicsConstants, area: jax.Array | float,
) -> jax.Array | None:
    if backing == 'rigid':
        return None
    if backing == 'radiation':
        if frequencies_hz is None:
            raise ValueError("radiation backing requires frequencies_hz")
        return _broadcast_angles(unflanged_pipe_radiation_load(frequencies_hz, air=air, radiation_area_m2=area), incidence_angles_deg)
    if backing != 'air':
        raise ValueError("backing must be rigid, air or radiation")
    srad = jnp.asarray(area)
    _require(jnp.all(jnp.isfinite(srad) & (srad > 0)), "radiation_area_m2 must be finite and positive")
    cosine = jnp.cos(_angles_radians(incidence_angles_deg)) if incidence_angles_deg is not None else jnp.asarray(1.)
    return air.rho_c / cosine / srad


def _safe_divide(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """SciPy apply_where的零分母規則；先使未選支分母有限再計算，保grad有限。"""
    valid = denominator != 0
    quotient = numerator / jnp.where(valid, denominator, 1)
    return jnp.where(valid, quotient, 0)


def _simpson_segments(y: jax.Array, x: jax.Array, stop: int) -> jax.Array:
    h = jnp.diff(x)
    h0, h1 = h[:stop:2], h[1:stop + 1:2]
    hsum, hprod = h0 + h1, h0 * h1
    ratio = _safe_divide(h0, h1)
    terms = hsum / 6 * (y[..., :stop:2] * (2 - _safe_divide(jnp.ones_like(ratio), ratio))
        + y[..., 1:stop + 1:2] * (hsum * _safe_divide(hsum, hprod))
        + y[..., 2:stop + 2:2] * (2 - ratio))
    return jnp.sum(terms, axis=-1)


def simpson(y: jax.Array, x: jax.Array) -> jax.Array:
    """SciPy 1.18.1的給定x積分；最後一軸、x為一維，支持複數與不等間距。

    單點回零、兩點梯形；偶數點尾段是Cartwright修正，不丟末區間。
    零間距照SciPy回零分母支；此支有限，不宣稱重複點處物理導數有意義。
    """
    values, samples = jnp.asarray(y), jnp.asarray(x)
    if samples.ndim != 1 or values.ndim == 0 or samples.size == 0 or values.shape[-1] != samples.size:
        raise ValueError("simpson requires nonempty 1D x matching the last axis of y")
    n = samples.size
    if n == 1:
        return jnp.zeros_like(values[..., 0])
    if n == 2:
        return .5 * (samples[-1] - samples[-2]) * (values[..., -1] + values[..., -2])
    if n % 2:
        return _simpson_segments(values, samples, n - 2)
    result = _simpson_segments(values, samples, n - 3)
    h0, h1 = samples[-2] - samples[-3], samples[-1] - samples[-2]
    alpha = _safe_divide(2 * h1**2 + 3 * h0 * h1, 6 * (h1 + h0))
    beta = _safe_divide(h1**2 + 3 * h0 * h1, 6 * h0)
    eta = _safe_divide(h1**3, 6 * h0 * (h0 + h1))
    return result + alpha * values[..., -1] + beta * values[..., -2] - eta * values[..., -3]


def trapezoidal_integral(y: jax.Array, x: jax.Array) -> jax.Array:
    """NumPy梯形法，沿最後一軸；x一維，支持對每個取樣點微分。"""
    values, samples = jnp.asarray(y), jnp.asarray(x)
    if samples.ndim != 1 or values.ndim == 0 or values.shape[-1] != samples.size:
        raise ValueError("trapezoidal_integral requires 1D x matching the last axis of y")
    return jnp.sum(jnp.diff(samples) * (values[..., 1:] + values[..., :-1]) / 2, axis=-1)


def _integration_angles(z_angle: jax.Array, incidence_angles_deg: jax.Array) -> jax.Array:
    theta = _angles_radians(incidence_angles_deg)
    if theta.size < 2 or z_angle.ndim == 0 or z_angle.shape[-1] != theta.size:
        raise ValueError("angular integration requires at least two angles matching the last impedance axis")
    _require(jnp.all(jnp.diff(theta) > 0), "integration angles must be strictly increasing")
    return theta


def field_impedance(z_angle: jax.Array, incidence_angles_deg: jax.Array) -> jax.Array:
    """導納乘sinθ用Simpson平均後取倒數；Pa·s/m，不是Paris吸音。"""
    z = jnp.asarray(z_angle)
    theta = _integration_angles(z, incidence_angles_deg)
    weights = jnp.sin(theta)
    admittance = simpson((1 / z) * weights, theta) / simpson(weights, theta)
    return 1 / admittance


def diffuse_absorption_coefficient(z_angle: jax.Array, incidence_angles_deg: jax.Array, *, air: PhysicsConstants) -> jax.Array:
    """Paris無規吸音：梯形平均α sinθ cosθ，只回實數吸音。"""
    z = jnp.asarray(z_angle)
    theta = _integration_angles(z, incidence_angles_deg)
    weights = jnp.sin(theta) * jnp.cos(theta)
    _, alpha = reflection_and_absorption_coefficient(z, air=air, incidence_angles_deg=incidence_angles_deg)
    return trapezoidal_integral(alpha * weights, theta) / trapezoidal_integral(weights, theta)
