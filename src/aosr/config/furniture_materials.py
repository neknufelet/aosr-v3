"""#559 家具材質登記簿：預設共用，複核上下界依家具種類記錄。

未知頻帶用明確清單表示；上下界可含未知頻帶的假設值，不能冒充預設資料。
此層不能匯入 geometry（幾何層），家具代號由考卷對第二支 FurnitureKind 核對。
"""
from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Self, TypeAlias, get_args

from pydantic import Field, model_validator

from aosr.config._furniture_records import EstimatedProvenance, FrozenRecord, PositiveNumber


FurnitureKindCode: TypeAlias = Literal["sofa", "chair", "coffee_table", "desk", "ceiling_cloud"]
FurnitureMaterialName: TypeAlias = Literal["fabric", "leather", "wood", "glass", "absorptive_cloud"]
# config 在 geometry 下方，這裡只存種類代號；方案與材質登記簿共用這一份准用表。
FURNITURE_MATERIAL_KINDS: Final[Mapping[FurnitureMaterialName, frozenset[FurnitureKindCode]]] = MappingProxyType({
    "fabric": frozenset(("sofa", "chair")), "leather": frozenset(("sofa", "chair")),
    "wood": frozenset(("coffee_table", "desk", "ceiling_cloud")),
    "glass": frozenset(("coffee_table", "desk")), "absorptive_cloud": frozenset(("ceiling_cloud",)),
})


def _increasing(frequencies: tuple[float, ...]) -> None:
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("頻帶必須嚴格遞增")


class AbsorptionBands(FrozenRecord):
    """明給軸的有限正吸音率；大於 1 的原始值保留，換算層才逐點夾值。"""

    band_center_hz: tuple[PositiveNumber, ...] = Field(min_length=1)
    absorption: tuple[PositiveNumber, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_axis(self) -> Self:
        _increasing(self.band_center_hz)
        if len(self.band_center_hz) != len(self.absorption):
            raise ValueError("頻帶與吸音率長度必須一致")
        return self

    def by_frequency(self) -> dict[float, float]:
        """供驗證比對的臨時映射，不把可變字典存進凍結資料。"""
        return dict(zip(self.band_center_hz, self.absorption, strict=True))


class KindBounds(FrozenRecord):
    """某家具種類的假設上下界，包含預設未知但複核可明給的頻帶。"""

    kind: FurnitureKindCode
    lower: AbsorptionBands
    upper: AbsorptionBands

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.lower.band_center_hz != self.upper.band_center_hz:
            raise ValueError("下界與上界的頻帶清單必須相同，否則延伸段會讓下界高過上界")
        if any(low > high for low, high in zip(self.lower.absorption, self.upper.absorption, strict=True)):
            raise ValueError("下界不得大於上界")
        return self


class FurnitureMaterial(FrozenRecord):
    """一種材質的全部頻帶、已知預設、明列未知、逐種類界線及出處。"""

    applicable_kinds: tuple[FurnitureKindCode, ...] = Field(min_length=1)
    band_center_hz: tuple[PositiveNumber, ...] = Field(min_length=1)
    default: AbsorptionBands
    unknown_bands_hz: tuple[PositiveNumber, ...]
    bounds: tuple[KindBounds, ...] = Field(min_length=1)
    label: str = Field(min_length=1)
    provenance: EstimatedProvenance

    @model_validator(mode="after")
    def complete_bands_and_bounds(self) -> Self:
        _increasing(self.band_center_hz)
        _increasing(self.unknown_bands_hz)
        known, unknown = set(self.default.band_center_hz), set(self.unknown_bands_hz)
        if known & unknown or known | unknown != set(self.band_center_hz):
            raise ValueError("預設頻帶與未知頻帶必須不重疊，合起來等於全部頻帶")
        if any(min(known) < band < max(known) for band in unknown):
            raise ValueError("未知頻帶只能在預設頻帶的兩端，中間缺值不准用內插冒充")
        kinds, bound_kinds = self.applicable_kinds, tuple(item.kind for item in self.bounds)
        if len(set(kinds)) != len(kinds) or len(set(bound_kinds)) != len(bound_kinds):
            raise ValueError("家具種類或界線種類不可重複")
        if set(kinds) != set(bound_kinds):
            raise ValueError("適用種類必須逐種提供上下界")
        for bounds in self.bounds:
            self._check_bounds(bounds)
        return self

    def _check_bounds(self, bounds: KindBounds) -> None:
        """上下界頻帶相同已由 KindBounds 保證，這裡只看一邊。"""
        lower, upper = bounds.lower.by_frequency(), bounds.upper.by_frequency()
        all_bands, known = set(self.band_center_hz), self.default.by_frequency()
        if not lower.keys() <= all_bands:
            raise ValueError("上下界只能使用全部頻帶清單裡的頻帶")
        if not known.keys() <= lower.keys():
            raise ValueError("上下界必須涵蓋所有有預設值的頻帶")
        for band, value in known.items():
            if not lower[band] <= value <= upper[band]:
                raise ValueError("每個種類須符合下界 ≤ 預設 ≤ 上界")

    def for_kind(self, kind: str) -> KindBounds:
        """依明給種類取界線；不從另一種家具偷偷借上下界。"""
        for bounds in self.bounds:
            if bounds.kind == kind:
                return bounds
        raise ValueError(f"此材質不適用家具種類 {kind!r}")


class FurnitureMaterials(FrozenRecord):
    """五類材質；每種家具至少有一個適用選項。"""

    fabric: FurnitureMaterial
    leather: FurnitureMaterial
    wood: FurnitureMaterial
    glass: FurnitureMaterial
    absorptive_cloud: FurnitureMaterial

    def materials(self) -> tuple[tuple[FurnitureMaterialName, FurnitureMaterial], ...]:
        """回傳有名字的不可變材質清單，供呼叫端列選項。"""
        return (("fabric", self.fabric), ("leather", self.leather), ("wood", self.wood),
                ("glass", self.glass), ("absorptive_cloud", self.absorptive_cloud))

    @model_validator(mode="after")
    def furniture_has_choices(self) -> Self:
        available = {kind for _, material in self.materials() for kind in material.applicable_kinds}
        if set(get_args(FurnitureKindCode)) - available:
            raise ValueError("某種家具沒有材質可選")
        for name, material in self.materials():
            if not set(material.applicable_kinds) <= FURNITURE_MATERIAL_KINDS[name]:
                raise ValueError(f"{name} 的適用家具種類錯誤")
        return self


def load_furniture_materials(path: str | Path) -> FurnitureMaterials:
    """只讀呼叫端明給的 TOML 路徑，不搜尋或內建預設路徑。"""
    with Path(path).open("rb") as file:
        return FurnitureMaterials.model_validate(tomllib.load(file))
