"""搜尋面片共用座標：@ 不在區代號字集，也不占變數的點或命令列的等號。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from aosr.geometry.shoebox import Wall
from aosr.physics.fem_reduced_boundary import PatchKey, PatchKeys, boundary_patches, checked_patch_keys
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.scheme import Scheme

BoundaryKey: TypeAlias = Wall | PatchKey


def surface_id(key: BoundaryKey) -> str:
    wall, zone = key if isinstance(key, tuple) else (key, "")
    return wall.wall_name() + ("@" + zone if zone else "")


def patch_key(name: str) -> PatchKey:
    wall, separator, zone = name.partition("@")
    found = next((w for w in Wall.all() if w.wall_name() == wall), None)
    if found is None or (separator and (not zone or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in zone))):
        raise ValueError(f"面片代號不認識：{name!r}")
    return found, zone


def surface_ids(keys: PatchKeys) -> tuple[str, ...]:
    return tuple(surface_id(key) for key in keys)


def scheme_patch_keys(scheme: Scheme) -> PatchKeys:
    zones = scheme.scene.surface_zones_by_wall or {}
    return boundary_patches({wall: tuple(z.id for z in zones.get(wall.wall_name(), ()) if isinstance(z, RectangleZone))
                             for wall in Wall.all() if any(isinstance(z, RectangleZone) for z in zones.get(wall.wall_name(), ()))})


def design_patch_keys(names: Sequence[str]) -> PatchKeys:
    return checked_patch_keys(tuple(patch_key(name) for name in names))


def named_materials[K: BoundaryKey, V](materials: Mapping[K, V]) -> dict[str, V]:
    return {surface_id(key): value for key, value in materials.items()}


def material_keys[K: BoundaryKey, V](materials: Mapping[K, V]) -> PatchKeys:
    names = named_materials(materials)
    if all(isinstance(key, Wall) for key in materials):
        return boundary_patches()
    # 方案建立的字典已照 boundary_patches 排好；不要排序抹掉區序。
    return design_patch_keys(tuple(names))


def partition_identity(keys: PatchKeys) -> tuple[tuple[str, str], ...]:
    return tuple((w.wall_name(), z) for w, z in keys) if keys != boundary_patches() else ()
