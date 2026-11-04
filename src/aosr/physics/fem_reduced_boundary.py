"""替身的有序面片鍵（#561）：牆序固定、同牆區序來自方案，與網格首現順序無關。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

from aosr.geometry.shoebox import Wall
from aosr.physics.fem_helmholtz import FrequencyWallImpedances, Impedance, checked_impedance_rows

PatchKey: TypeAlias = tuple[Wall, str]
PatchKeys: TypeAlias = tuple[PatchKey, ...]
FrequencyPatchImpedances: TypeAlias = Mapping[PatchKey, Impedance | None | Sequence[Impedance | None]]
FrequencyBoundaryImpedances: TypeAlias = FrequencyWallImpedances | FrequencyPatchImpedances
BoundaryImpedances: TypeAlias = Mapping[Wall, Impedance | None] | Mapping[PatchKey, Impedance | None]
ImpedanceRows: TypeAlias = Mapping[PatchKey, tuple[Impedance | None, ...]]


def boundary_patches(zones: Mapping[Wall, Sequence[str]] | None = None) -> PatchKeys:
    """整面鍵為 (Wall, "")；分區鍵為 (Wall, 區代號)，依呼叫端（方案）列出的區序。"""
    zones = {} if zones is None else zones
    if set(zones) - set(Wall.all()):
        raise ValueError("面片含未知牆面")
    for names in zones.values():
        if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("分區面片代號必須非空且同面唯一")
    return tuple((wall, name) for wall in Wall.all() for name in zones.get(wall, ("",)))


def checked_patch_keys(keys: PatchKeys) -> PatchKeys:
    """六面皆覆蓋、不混整面與分區、無重複，並且牆序正確；區序保留。"""
    zones: dict[Wall, list[str]] = {}
    for wall, name in keys:
        if name:
            zones.setdefault(wall, []).append(name)
    if keys != boundary_patches(zones):
        raise ValueError("面片清單必須照牆序完整覆蓋，整面與分區不得混用")
    return keys


def patch_values(values: BoundaryImpedances, keys: PatchKeys) -> dict[PatchKey, Impedance | None]:
    """舊六面輸入只適用於六個整面；有面片鍵時必須恰好給齊。"""
    if keys == boundary_patches() and set(values) == set(Wall.all()):
        whole = cast(Mapping[Wall, Impedance | None], values)
        return {(wall, ""): whole[wall] for wall in Wall.all()}
    if set(values) != set(keys):
        raise ValueError("阻抗必須恰好覆蓋面片清單")
    result = cast(Mapping[PatchKey, Impedance | None], values)
    for key in keys:
        if key[1] and result[key] is None:
            raise ValueError("分區面片一定要有材料阻抗，不能給 None")
    return {key: result[key] for key in keys}


def checked_patch_rows(values: FrequencyBoundaryImpedances, axis: tuple[float, ...], keys: PatchKeys) -> ImpedanceRows:
    """借用產品逐頻阻抗閘門，保留原數值型別、拒收分區剛性與不完整材料。"""
    if keys == boundary_patches() and set(values) == set(Wall.all()):
        whole = checked_impedance_rows(cast(FrequencyWallImpedances, values), axis)
        return {(wall, ""): whole[wall] for wall in Wall.all()}
    if set(values) != set(keys):
        raise ValueError("逐頻阻抗必須恰好覆蓋面片清單")
    patches = cast(FrequencyPatchImpedances, values)
    result = {}
    for key in keys:
        wall, zone = key
        # 單片借六面閘門驗一列；其餘整面剛性不參與算術。
        row = checked_impedance_rows({w: patches[key] if w == wall else None for w in Wall.all()}, axis)[wall]
        if zone and row[0] is None:
            raise ValueError("分區面片一定要有材料阻抗，不能給 None")
        result[key] = row
    return result
