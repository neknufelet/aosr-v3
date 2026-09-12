"""材料這一塊的核心資料契約：頻率軸 :class:`FrequencyAxis` 與材料回應 :class:`MaterialResponse`。

**這兩個型別為什麼住第 3 層。** 它們是 JAX 的 pytree（一種「葉子是陣列、其餘是靜態中介
資料」的容器，``jax.tree_util`` 拆得開、``jax.jit`` 穿得過），所以碰得到 JAX；決策紙
``engine-first-block-config-shape`` 拍過：不碰 JAX 的共用型別沉到第 2 層，碰 JAX 的留第 3 層，
第 2 層一支都不准載入 JAX。

**這一支不做驗證，除了散射那一格。** 上一代把吸收係數上下限、``unresolved`` 的閘、
locally-reacting 的前提放在另一支模組裡，刻意**不放**在容器上，好讓 pytree 保持「純資料、
組得回去」。這裡照做：``Z_surface`` 的形狀與有限性在這裡**不驗**（不要為了讓某一題好看
就加一道新的禁止——那是改契約）。散射係數是例外，理由寫在 :meth:`MaterialResponse.__post_init__`。

**數值與行為跟上一代一致，結構隨便改。** 標準答案由 ``blueprint/materials_cut1_answers.json``
在唯讀的 donor 樹上跑出來，考卷逐格比（值、dtype、shape、複數的實部虛部、pytree 的葉子與
靜態中介資料、公開簽章的必填性與預設值、以及每一種錯誤的型別與訊息）。
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

# ── 邊界模型 ───────────────────────────────────────────────────────────────

BoundaryModel = Literal[
    "local_impedance",
    "angle_dependent",
    "extended_reacting",
    "unresolved",
]
"""一個表面的聲學回應可以被求解器怎麼用。

* ``local_impedance``   直接當成 FEM 的阻抗邊界。
* ``angle_dependent``   轉接層要入射角才算得出來。
* ``extended_reacting`` **不准**被當成局部反應的表面來用。
* ``unresolved``        資料不足，禁止進入求解器（不准無聲降級）。
"""

BOUNDARY_MODELS: frozenset[str] = frozenset(get_args(BoundaryModel))

ResolutionMode = Literal["linear_hz", "third_octave", "custom"]
"""頻率軸的解析度慣例（每倍頻幾點、或者是一組自訂的頻率）。"""

RESOLUTION_MODES: frozenset[str] = frozenset(get_args(ResolutionMode))

# 平面／樣板材料的散射係數種子。值的出處是上一代那份散射設定檔（平面的實務最小值）；
# 那份檔在上一代的 config 目錄底下，不在這棵樹裡，所以這裡不寫它的路徑（寫了就是一個
# 解析不到的死引用）。明白寫著「沒有散射頻譜」的舊材料走的是彙總邊界上的房間平均，不是這一格。
MATERIAL_SCATTERING_DEFAULT_S: float = 0.1

# 「這一格是靜態中介資料，不是葉子」的標記。`flax.struct` 讀的就是這一格中介資料；
# 它自己那支 `struct.field()` 沒有型別標註（mypy 嚴格模式判它 no-untyped-call），
# 所以這裡直接用標準庫的 `dataclasses.field(metadata=…)`——同一份中介資料、有標註。
STATIC: dict[str, bool] = {"pytree_node": False}


# ── 頻率軸 ─────────────────────────────────────────────────────────────────


@struct.dataclass
class FrequencyAxis:
    """一個頻譜量的頻率支撐。

    ``freqs_hz`` 是**唯一的葉子**（一串實數，單位 Hz）；``resolution`` 與 ``convention``
    是靜態中介資料（不會被微分、``tree_map`` 也不碰它們）。

    軸是明寫的，不是算出來的：對照測試才比得到「同一條軸上的兩個值」，而不是兩條對不齊的
    軸上的兩串數字。

    **這裡不排序、也不強迫轉成浮點**：``from_hz`` 把輸入原樣交給 ``jnp.asarray``，
    所以整數進來就是整數的 dtype（受全域 x64 開關影響）、遞減的輸入就維持遞減。要排序與
    驗證的是上層那個 API schema，不是這個容器。
    """

    freqs_hz: jax.Array
    resolution: str = dataclasses.field(default="custom", metadata=STATIC)
    convention: str = dataclasses.field(default="hz", metadata=STATIC)

    @property
    def n_freq(self) -> int:
        """這條軸上有幾個頻率點。"""
        return int(self.freqs_hz.shape[0])

    @property
    def f_min(self) -> float:
        """**第一個**頻率點（不是最小值——這條軸沒有被排序過）。"""
        return float(self.freqs_hz[0])

    @property
    def f_max(self) -> float:
        """**最後一個**頻率點（同上，不是最大值）。"""
        return float(self.freqs_hz[-1])

    @classmethod
    def from_hz(
        cls,
        freqs_hz: object,
        *,
        resolution: ResolutionMode = "custom",
    ) -> FrequencyAxis:
        """從任何像陣列的東西（單位 Hz）造一條軸。"""
        arr = jnp.asarray(freqs_hz)
        return cls(freqs_hz=arr, resolution=resolution, convention="hz")


# ── 材料回應 ───────────────────────────────────────────────────────────────


@struct.dataclass
class MaterialResponse:
    """隨頻率變化的表面阻抗與散射係數，加上求解器要用的中介資料。

    葉子：``Z_surface``、``scattering_coeff``（可以沒有）、以及 ``freq_axis`` 自己那一片。
    靜態中介資料：``is_locally_reacting``、``boundary_model``、``material_id``。

    這個物件刻意很小。某一個求解器要的東西（反射係數、吸收係數）由轉接層當場算，
    不存在這裡——存了就會有兩份可能對不上的真相。
    """

    Z_surface: jax.Array
    freq_axis: FrequencyAxis
    scattering_coeff: jax.Array | None = dataclasses.field(default=None)
    is_locally_reacting: bool = dataclasses.field(default=True, metadata=STATIC)
    boundary_model: str = dataclasses.field(default="unresolved", metadata=STATIC)
    material_id: str = dataclasses.field(default="", metadata=STATIC)

    def __post_init__(self) -> None:
        """散射係數那一格的守門（**這個容器唯一驗的東西**）。

        兩件事分開處理，理由不一樣：

        * **形狀**永遠驗。它跟頻率軸的長度對不上，後面每一步都會拿到錯的東西。
        * **範圍與有限性**只在**不是 tracer 的時候**驗。tracer 是 JAX 在編譯期用來代替真值
          的替身；要它交出具體的數字就等於強迫當場算一次，本來合法的 pytree 變換會因此爆掉。
          所以編譯期只留形狀那一道，eager（真的有值的時候）兩道都留。

        順手把散射係數轉成 ``float32`` 存回去——這是上一代的行為，dtype 也是契約。
        """
        scattering = self.scattering_coeff
        if scattering is None:
            return
        spectrum = jnp.asarray(scattering, dtype=jnp.float32)
        expected = (self.freq_axis.n_freq,)
        if spectrum.shape != expected:
            raise ValueError(
                f"scattering_coeff must have shape {expected}, got {spectrum.shape}"
            )
        if not isinstance(spectrum, jax.core.Tracer):
            concrete = np.asarray(jax.device_get(spectrum))
            out_of_range = np.any(concrete < 0.0) or np.any(concrete > 1.0)
            if not np.all(np.isfinite(concrete)) or out_of_range:
                raise ValueError("scattering_coeff must be in [0, 1] and finite")
        object.__setattr__(self, "scattering_coeff", spectrum)

    @property
    def n_freq(self) -> int:
        """這份回應有幾個頻率點（跟它的軸同一個數）。"""
        return self.freq_axis.n_freq

    @classmethod
    def from_impedance(
        cls,
        Z_surface: object,  # noqa: N803  # expires=2026-12-08 reason=參數名是公開契約的一部分（呼叫端用關鍵字傳它），改成小寫就是改 API
        freq_axis: FrequencyAxis,
        *,
        material_id: str,
        boundary_model: BoundaryModel = "local_impedance",
        is_locally_reacting: bool = True,
        scattering_coeff: jax.Array | float | None = MATERIAL_SCATTERING_DEFAULT_S,
    ) -> MaterialResponse:
        """從一串複數阻抗與一條明寫的軸造一份回應。

        **不驗 ``Z_surface``**（形狀、有限性都不驗）：驗證住在別的地方，這裡只負責裝。
        散射係數給純量就展開成整條軸那麼長，給 ``None`` 就真的是沒有。
        """
        if scattering_coeff is None:
            scattering_spectrum = None
        else:
            scattering_input = jnp.asarray(scattering_coeff, dtype=jnp.float32)
            scattering_spectrum = (
                jnp.full((freq_axis.n_freq,), scattering_input, dtype=jnp.float32)
                if scattering_input.ndim == 0
                else scattering_input
            )
        return cls(
            Z_surface=jnp.asarray(Z_surface),
            freq_axis=freq_axis,
            scattering_coeff=scattering_spectrum,
            is_locally_reacting=is_locally_reacting,
            boundary_model=boundary_model,
            material_id=material_id,
        )

    @classmethod
    def from_alpha(
        cls,
        alpha_bands: Sequence[float],
        band_freqs: Sequence[float],
        freq_axis: FrequencyAxis,
        rho_c: float,
        *,
        material_id: str,
        boundary_model: BoundaryModel = "local_impedance",
        scattering_coeff: jax.Array | float | None = MATERIAL_SCATTERING_DEFAULT_S,
    ) -> MaterialResponse:
        """從一張吸收係數表造一份回應（直接寫死吸收頻譜的那種材料）。

        材料的本質量是表面阻抗，所以 α 要被抬成一個**實數、零相位**的阻抗，用的是垂直
        入射的被動反解 ``α = 1 − |r|²``。三步：

        1. **先把 α 夾進 [0, 1]**。資料表上的隨機入射值可能是 1.05～1.20，夾掉而**不是**
           丟錯——丟錯會讓一份合法的材料在下游才殺掉整跑。
        2. **在 ``log10(Hz)`` 上插值**，把 K 個頻帶攤到軸上的每一點。``np.interp`` 是
           **平坦延伸**（不會震盪）：軸上超出頻帶範圍的點取最靠近那一格的值。
        3. 抬成實數阻抗，交給 :meth:`from_impedance`。

        :param alpha_bands: 每個頻帶中心的吸收係數。
        :param band_freqs: 頻帶中心頻率（Hz），嚴格遞增。
        :param freq_axis: 目標軸。
        :param rho_c: 空氣的特性阻抗 ρ₀·c（Pa·s/m）。**由呼叫端注入**，不在這裡算。
        :param material_id: 給註冊表用的出身 id。
        :returns: ``Z_surface`` 是實數（零相位）的一份回應。
        """
        alpha = np.asarray(alpha_bands, dtype=float)
        freqs = np.asarray(band_freqs, dtype=float)
        _check_alpha_table(alpha, freqs)
        alpha = np.clip(alpha, 0.0, 1.0)
        target_f = np.asarray(freq_axis.freqs_hz, dtype=float)
        alpha_full = np.interp(np.log10(target_f), np.log10(freqs), alpha)
        # 被動的垂直入射反解：α = 1 − |r|²。兩處 1e-12 是夾子不是修飾——沒有它們，
        # α = 0 的那一格會算出除以零。不准換成別種物理近似。
        s = np.sqrt(np.clip(1.0 - alpha_full, 1e-12, 1.0))
        z_real = rho_c * (1.0 + s) / np.maximum(1.0 - s, 1e-12)
        return cls.from_impedance(
            jnp.asarray(z_real) + 0.0j,
            freq_axis,
            material_id=material_id,
            boundary_model=boundary_model,
            is_locally_reacting=True,
            scattering_coeff=scattering_coeff,
        )


def _check_alpha_table(alpha: np.ndarray, freqs: np.ndarray) -> None:
    """吸收係數表的四道守門（順序也是契約：哪一種錯先被抓到是行為）。"""
    if alpha.ndim != 1 or freqs.ndim != 1:
        raise ValueError("alpha_bands and band_freqs must be 1-D")
    if alpha.shape != freqs.shape:
        raise ValueError(
            f"alpha_bands length {alpha.shape[0]} != band_freqs length {freqs.shape[0]}"
        )
    if alpha.shape[0] < 1:
        raise ValueError("alpha_bands must have at least one band")
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(freqs)):
        raise ValueError("alpha_bands / band_freqs have non-finite entries")
    if freqs.shape[0] > 1 and not np.all(np.diff(freqs) > 0.0):
        raise ValueError("band_freqs must be strictly ascending")
