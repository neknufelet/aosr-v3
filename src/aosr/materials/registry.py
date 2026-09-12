"""材料註冊表：材料 id（字串）對到一份 :class:`~aosr.materials.response.MaterialResponse`。

今天這張表裝的是已經建好的回應，外加兩個**解析式**的佔位材料（剛性牆、頻率平坦的阻抗）
——它們不需要 TMM，拿來當初始值、參考案例與測試道具剛好。真正由 TMM 算出來的材料是後面的事。

材料資料庫永遠只是「初始值／限制／產品對應」的來源；真正的最佳化變數是 TMM 那幾個連續參數。
"""
from __future__ import annotations

import jax.numpy as jnp

from aosr.materials.response import BoundaryModel, FrequencyAxis, MaterialResponse
from aosr.materials.source import SOURCE_KINDS, MaterialEntry, SourceKind


class MaterialRegistry:
    """一張放在記憶體裡的 ``material_id → MaterialEntry``。"""

    def __init__(self) -> None:
        """開一張空表。"""
        self._materials: dict[str, MaterialEntry] = {}

    def register(
        self,
        mat: MaterialResponse,
        *,
        source_kind: SourceKind | None = None,
        source_ref: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """把一份回應登記進來（出身沒給的時候照下面的規則推）。

        **覆寫時出身的繼承規則是行為，不是細節**，兩個方向刻意不一樣：

        * **沒給 ``source_kind``**（隱式覆寫）：把既有那一筆的 kind **連同它的 ref** 一起
          留著。不然重登記一次就把 ``measured_alpha``／``computed_tmm`` 悄悄洗成
          ``analytic``，而且沒有人看得出來。全新的材料沒給 kind 才預設 ``analytic``。
        * **明著給了 ``source_kind``**：**不**繼承舊的 ``source_ref``。ref 描述的是舊出身，
          換了 kind 還留著它就是一個過期的指標（就算新舊 kind 剛好一樣也一樣不繼承）。
        """
        if not mat.material_id:
            raise ValueError("cannot register a material with an empty material_id")
        existing = self._materials.get(mat.material_id)
        if source_kind is None:
            if existing is not None:
                source_kind = existing.source_kind
                if source_ref is None:
                    source_ref = existing.source_ref
            else:
                source_kind = "analytic"
        if source_kind not in SOURCE_KINDS:
            raise ValueError(
                f"unknown source_kind {source_kind!r}; expected one of {sorted(SOURCE_KINDS)}"
            )
        self.register_entry(
            MaterialEntry(response=mat, source_kind=source_kind, source_ref=source_ref),
            overwrite=overwrite,
        )

    def register_entry(self, entry: MaterialEntry, *, overwrite: bool = False) -> None:
        """把一筆現成的（回應＋出身）登記進來。

        兩道守門跟 :meth:`register` 是同一組，**刻意各寫一次**：這一支是公開入口，
        呼叫端可以完全不經過 :meth:`register` 就進來。
        """
        if not entry.response.material_id:
            raise ValueError("cannot register a material with an empty material_id")
        if entry.source_kind not in SOURCE_KINDS:
            raise ValueError(
                f"unknown source_kind {entry.source_kind!r}; expected one of {sorted(SOURCE_KINDS)}"
            )
        material_id = entry.response.material_id
        if material_id in self._materials and not overwrite:
            raise KeyError(f"material {material_id!r} already registered")
        self._materials[material_id] = entry

    def get(self, material_id: str) -> MaterialResponse:
        """那個 id 的回應本身（**同一個物件**，不是一份拷貝）。"""
        return self.entry(material_id).response

    def entry(self, material_id: str) -> MaterialEntry:
        """那個 id 的整筆（回應＋出身）。找不到就丟 ``KeyError``，訊息帶已知的 id。"""
        try:
            return self._materials[material_id]
        except KeyError:
            raise KeyError(
                f"unknown material {material_id!r}; known: {sorted(self._materials)}"
            ) from None

    def source_kind(self, material_id: str) -> SourceKind:
        """那個 id 的出身種類。"""
        return self.entry(material_id).source_kind

    def ids(self) -> list[str]:
        """表上所有的 id，**排序好的 list**（容器種類也是契約）。"""
        return sorted(self._materials)

    def __contains__(self, material_id: object) -> bool:
        """``"m1" in registry``。"""
        return material_id in self._materials

    def __len__(self) -> int:
        """``len(registry)``——表上有幾筆。"""
        return len(self._materials)


# ── 解析式的佔位材料（不需要 TMM）──────────────────────────────────────────


def rigid_wall(
    freq_axis: FrequencyAxis,
    *,
    material_id: str = "rigid_wall",
    z_magnitude: float = 1e8,
) -> MaterialResponse:
    """幾乎全反射：很大的實數阻抗，於是 ``α ≈ 0``。"""
    impedance = jnp.full((freq_axis.n_freq,), complex(z_magnitude, 0.0))
    return MaterialResponse.from_impedance(
        impedance,
        freq_axis,
        material_id=material_id,
        boundary_model="local_impedance",
        is_locally_reacting=True,
    )


def constant_impedance(
    freq_axis: FrequencyAxis,
    z_value: complex,
    *,
    material_id: str,
    boundary_model: BoundaryModel = "local_impedance",
    is_locally_reacting: bool = True,
) -> MaterialResponse:
    """頻率平坦的表面阻抗（材料本身的量，不含 ρc）。"""
    impedance = jnp.full((freq_axis.n_freq,), complex(z_value))
    return MaterialResponse.from_impedance(
        impedance,
        freq_axis,
        material_id=material_id,
        boundary_model=boundary_model,
        is_locally_reacting=is_locally_reacting,
    )
