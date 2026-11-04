"""鏡像法的六面反射來源：構造在路徑角直接求值，固定牆沿用純量捷徑。

同一均勻牆每次反彈的攤開角相同，仍用 R ** count（同面反彈次方）。
構造反射由逐位純量考卷守，不屬於 reflection_product_ulp（舊固定資料比較契約）。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from aosr.geometry.shoebox import Room, Wall, wall_count_signature
from aosr.materials.construction_models import ConstantImpedance, ConstructionNote, SurfaceMaterial
from aosr.physics import amplitude as amp
from aosr.physics.report_surfaces import (
    RectangleZone, RectangleZonesUnsupported, SurfaceZone, rectangle_zone_at,
    validate_surface_partition, whole_wall_materials,
)
from aosr.physics.wall_surfaces import wall_normal_impedances, wall_path_impedances

if TYPE_CHECKING:
    from aosr.physics.room_paths import RoomPath

ComplexRow = NDArray[np.complex128]


def reflection_coefficient_row(z: ComplexRow, cos_theta: float, rho_c: float) -> ComplexRow:
    """整排 CPython（Python 複數算術）的 Smith 除法；不改運算順序與零號。"""
    real = z.real * cos_theta - z.imag * 0.0
    imag = z.real * 0.0 + z.imag * cos_theta
    ar, ai, br, bi = real - rho_c, imag - 0.0, real + rho_c, imag + 0.0
    out = np.empty(z.shape, dtype=np.complex128)
    dominant = np.abs(br) >= np.abs(bi)
    for mask, real_dominant in ((dominant, True), (~dominant, False)):
        a, b, c, d = ar[mask], ai[mask], br[mask], bi[mask]
        if real_dominant:
            ratio = d / c
            denominator = c + d * ratio
            out.real[mask] = (a + b * ratio) / denominator
            out.imag[mask] = (b - a * ratio) / denominator
        else:
            ratio = c / d
            denominator = c * ratio + d
            out.real[mask] = (a * ratio + b) / denominator
            out.imag[mask] = (b * ratio - a) / denominator
    return out


@dataclass(frozen=True)
class WallReflectionSources:
    """同一條頻率軸的固定材料、構造及場景空氣；按聲源接收對批次求路徑角。"""

    materials_by_wall: Mapping[Wall, SurfaceMaterial]
    fixed_materials: amp.Materials
    density_kg_m3: float
    sound_speed_m_s: float
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None
    rectangle_zones_by_wall: Mapping[Wall, tuple[RectangleZone, ...]] | None = None
    room: Room | None = None

    @property
    def has_rectangle_zones(self) -> bool:
        return bool(self.rectangle_zones_by_wall)

    @property
    def has_construction(self) -> bool:
        return (any(not isinstance(m, ConstantImpedance) for m in self.materials_by_wall.values()) or
                any(not isinstance(z.material, ConstantImpedance)
                    for zones in (self.rectangle_zones_by_wall or {}).values() for z in zones))

    def normal_rows(self) -> dict[str, tuple[complex, ...]]:
        """散射權重只取 Z(f,0°)，固定值保留 complex(float) 的舊順序。"""
        if self.rectangle_zones_by_wall:
            raise RectangleZonesUnsupported(next(iter(self.rectangle_zones_by_wall)).wall_name())
        axis = self.fixed_materials.frequencies_hz
        rows = wall_normal_impedances(self.materials_by_wall, axis,
            density_kg_m3=self.density_kg_m3, sound_speed_m_s=self.sound_speed_m_s,
            notes_by_wall=self.notes_by_wall)
        return {w.wall_name(): ((complex(z),) * len(axis) if isinstance(z, float) else z)
                for w, z in rows.items() if z is not None}

    def impedances_for_paths(
        self, paths: Sequence[RoomPath], receiver: tuple[float, float, float],
    ) -> dict[str, dict[float, ComplexRow]]:
        """每面只收真的打到它的 float（浮點數）cos，逐位去重、直接算角，不內插。"""
        if self.rectangle_zones_by_wall:
            raise RectangleZonesUnsupported(next(iter(self.rectangle_zones_by_wall)).wall_name())
        result: dict[str, dict[float, ComplexRow]] = {}
        for wall in amp.CANONICAL_WALLS:
            material = self.materials_by_wall.get(Wall.from_name(wall))
            if material is None or isinstance(material, ConstantImpedance):
                continue
            cosines = dict.fromkeys(amp.incidence_cos(amp._wall_axis(wall), p.dist_m, receiver, p.image)
                                    for p in paths if wall_count_signature(p.identity)[wall])
            if cosines:
                result[wall] = self._evaluate_wall(wall, material, tuple(cosines))
        return result

    def _evaluate_wall(
        self, wall: str, material: SurfaceMaterial, cosines: tuple[float, ...],
    ) -> dict[float, ComplexRow]:
        """交給 wall_path_impedances：延後匯入核心、補到八欄桶減少重編譯、丟掉補欄，都在那一支做。"""
        return wall_path_impedances(material, self.fixed_materials.frequencies_hz, cosines,
            wall_name=wall, density_kg_m3=self.density_kg_m3, sound_speed_m_s=self.sound_speed_m_s,
            notes_by_wall=self.notes_by_wall)

    def apply(
        self, paths: Sequence[RoomPath], receiver: tuple[float, float, float], sound_speed: float,
    ) -> list[RoomPath]:
        """幾何先全部列完，再給每路原有壓力算式；未打構造牆仍走舊反射乘積。"""
        if self.has_rectangle_zones:
            return self._apply_zones(paths, receiver, sound_speed)
        rows = self.impedances_for_paths(paths, receiver)
        out = []
        for path in paths:
            counts = wall_count_signature(path.identity)
            product = (self._product(path, receiver, counts, rows) if any(counts[w] for w in rows) else
                       amp.reflection_product(self.fixed_materials, path.dist_m, receiver, path.image, counts))
            pressure = amp.path_pressure(self.fixed_materials, path.dist_m, sound_speed, product)
            out.append(replace(path, reflection_product=product, path_pressure=pressure))
        return out

    def _zone_counts(self, path: RoomPath) -> dict[tuple[str, str], int]:
        """逐點逐欠牆查唯一分區；同牆同區合計次數後才求 R 次方。"""
        counts: dict[tuple[str, str], int] = {}
        for bounce in path.bounces:
            for name in bounce.walls:
                zones = (self.rectangle_zones_by_wall or {}).get(Wall.from_name(name))
                key = (name, rectangle_zone_at(name, bounce.point, zones).id if zones else "")
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _apply_zones(
        self, paths: Sequence[RoomPath], receiver: tuple[float, float, float], sound_speed: float,
    ) -> list[RoomPath]:
        """僅矩形分區入口使用；每個構造分區只批次求實際命中角，未命中分區不求值。"""
        counts = [self._zone_counts(path) for path in paths]
        whole_rows = replace(self, rectangle_zones_by_wall=None).impedances_for_paths(paths, receiver)
        zone_rows: dict[tuple[str, str], dict[float, ComplexRow]] = {}
        for wall, zones in (self.rectangle_zones_by_wall or {}).items():
            name = wall.wall_name()
            for zone in zones:
                if isinstance(zone.material, ConstantImpedance):
                    continue
                cosines = tuple(dict.fromkeys(amp.incidence_cos(wall.axis(), path.dist_m, receiver, path.image)
                    for path, hits in zip(paths, counts, strict=True) if hits.get((name, zone.id))))
                if cosines:
                    zone_rows[name, zone.id] = self._evaluate_wall(name, zone.material, cosines)
        out = []
        for path, hits in zip(paths, counts, strict=True):
            if any(zone_id for _wall, zone_id in hits):
                product = self._zoned_product(path, receiver, hits, whole_rows, zone_rows)
            else:
                whole_counts = wall_count_signature(path.identity)
                product = (self._product(path, receiver, whole_counts, whole_rows)
                    if any(whole_counts[w] for w in whole_rows) else amp.reflection_product(
                        self.fixed_materials, path.dist_m, receiver, path.image, whole_counts))
            pressure = amp.path_pressure(self.fixed_materials, path.dist_m, sound_speed, product)
            out.append(replace(path, reflection_product=product, path_pressure=pressure,
                rectangle_zone_walls=tuple(w.wall_name() for w in (self.rectangle_zones_by_wall or {}))))
        return out

    def _zoned_product(
        self, path: RoomPath, receiver: tuple[float, float, float], counts: dict[tuple[str, str], int],
        whole_rows: dict[str, dict[float, ComplexRow]],
        zone_rows: dict[tuple[str, str], dict[float, ComplexRow]],
    ) -> tuple[complex, ...]:
        """規範牆序、輸入分區序；同一面每區沿用攤開角，R 只算一次再取次方。"""
        product = np.ones(len(self.fixed_materials.frequencies_hz), dtype=np.complex128)
        for wall in Wall.all():
            name = wall.wall_name()
            zones = (self.rectangle_zones_by_wall or {}).get(wall)
            entries = (tuple((z.id, z.material) for z in zones) if zones else
                       (("", self.materials_by_wall[wall]),))
            for zone_id, material in entries:
                count = counts.get((name, zone_id), 0)
                if not count:
                    continue
                cos = amp.incidence_cos(wall.axis(), path.dist_m, receiver, path.image)
                if isinstance(material, ConstantImpedance):
                    r = np.full(product.shape, amp.reflection_coefficient(
                        complex(material.impedance_pa_s_m), cos, self.fixed_materials.rho_c), dtype=np.complex128)
                else:
                    row = zone_rows[name, zone_id][cos] if zones else whole_rows[name][cos]
                    r = reflection_coefficient_row(row, cos, self.fixed_materials.rho_c)
                if not np.isfinite(r).all():
                    raise ValueError(f"{name} 分區 {zone_id} 的反射係數不是有限數")
                product = _multiply(product, r if count == 1 else _power(r, count))
        return tuple(product.tolist())

    def _product(
        self, path: RoomPath, receiver: tuple[float, float, float], counts: dict[str, int],
        rows: dict[str, dict[float, ComplexRow]],
    ) -> tuple[complex, ...]:
        """按原牆序整排 R 次方及連乘；混合路的固定牆 R 仍以純量算一次再展開。"""
        product = np.ones(len(self.fixed_materials.frequencies_hz), dtype=np.complex128)
        for wall in amp.CANONICAL_WALLS:
            count = counts[wall]
            if not count:
                continue
            cos = amp.incidence_cos(amp._wall_axis(wall), path.dist_m, receiver, path.image)
            if wall in rows:
                r = reflection_coefficient_row(rows[wall][cos], cos, self.fixed_materials.rho_c)
            else:
                z = self.fixed_materials.constant_impedances[wall]
                assert z is not None
                r = np.full(product.shape, amp.reflection_coefficient(z, cos, self.fixed_materials.rho_c), dtype=np.complex128)
            if not np.isfinite(r).all():
                bad = int(np.flatnonzero(~np.isfinite(r))[0])
                raise ValueError(f"{wall} 在 {self.fixed_materials.frequencies_hz[bad]} Hz、cos={cos!r} "
                                 "的反射係數不是有限數（阻抗大到溢位），不讓它流進能量結果")
            product = _multiply(product, r if count == 1 else _power(r, count))
        return tuple(product.tolist())


def _multiply(left: ComplexRow, right: ComplexRow) -> ComplexRow:
    """整排 CPython 複數乘法：四個乘積照原順序，不交給 numpy 的複數乘法。"""
    out = np.empty(left.shape, dtype=np.complex128)
    out.real = left.real * right.real - left.imag * right.imag
    out.imag = left.real * right.imag + left.imag * right.real
    return out


def _power(r: ComplexRow, count: int) -> ComplexRow:
    """整排照 CPython 的整數次方：從 1+0j 起、逐位元平方倍增。

    np.power 的二次方直接平方、不乘 1+0j，正零與負零會跟純量 ``r ** count`` 不同。
    """
    result = np.ones(r.shape, dtype=np.complex128)
    base, mask = r, 1
    while count >= mask:
        if count & mask:
            result = _multiply(result, base)
        mask <<= 1
        base = _multiply(base, base)
    return result


def wall_reflection_sources(
    materials_by_wall: Mapping[Wall, SurfaceMaterial], frequencies_hz: tuple[float, ...],
    *, density_kg_m3: float, sound_speed_m_s: float,
    rho_c_pa_s_per_m: float | None = None,
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None,
    rectangle_zones_by_wall: Mapping[Wall, tuple[RectangleZone, ...]] | None = None,
    room: Room | None = None,
) -> WallReflectionSources:
    """六面 SurfaceMaterial（表面材料）、場景空氣及一條軸建出唯一反射來源。"""
    if set(materials_by_wall) | set(rectangle_zones_by_wall or {}) != set(Wall.all()):
        raise ValueError("materials_by_wall 必須恰好包含六面牆")
    if rectangle_zones_by_wall:
        if room is None:
            raise ValueError("反射分區必須提供房間尺寸")
        if set(materials_by_wall) & set(rectangle_zones_by_wall):
            raise ValueError("同一面牆的整面材料與矩形分區不可同時提供")
        if any(not zones for zones in rectangle_zones_by_wall.values()):
            raise ValueError("矩形分區清單不可為空")
        validate_surface_partition(room, {w.wall_name(): zones for w, zones in rectangle_zones_by_wall.items()})
    if not frequencies_hz or any(not math.isfinite(f) or f <= 0 for f in frequencies_hz):
        raise ValueError("頻率軸必須非空且全為有限正數")
    if any(not math.isfinite(v) or v <= 0 for v in (density_kg_m3, sound_speed_m_s)):
        raise ValueError("場景密度與聲速必須是有限正數")
    fixed = {wall.wall_name(): (complex(material.impedance_pa_s_m),) * len(frequencies_hz)
             for wall, material in materials_by_wall.items() if isinstance(material, ConstantImpedance)}
    # 舊入口只給ρc時，不用 (ρc/c)*c 重建它；除乘來回可能損失一位。
    rho_c = density_kg_m3 * sound_speed_m_s if rho_c_pa_s_per_m is None else rho_c_pa_s_per_m
    if not math.isfinite(rho_c) or rho_c <= 0:
        raise ValueError("rho_c 必須是有限正數")
    return WallReflectionSources(dict(materials_by_wall), amp.Materials(rho_c,
                                 frequencies_hz, fixed), density_kg_m3, sound_speed_m_s, notes_by_wall,
                                 rectangle_zones_by_wall, room)


def zoned_reflection_sources(
    room: Room, surface_zones_by_wall: Mapping[str, Sequence[SurfaceZone]], frequencies_hz: tuple[float, ...],
    *, density_kg_m3: float, sound_speed_m_s: float, rho_c_pa_s_per_m: float | None = None,
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None,
) -> WallReflectionSources:
    """鏡像法的分區物理入口；整面一區直接交回原入口，不經逐區乘法。

    先核六面、精確覆蓋與不重疊；矩形牆不放進整面材料字典，避免下游誤取第一區。
    """
    if set(surface_zones_by_wall) != set(Wall.wall_names()):
        raise ValueError("surface_zones_by_wall 必須恰好包含六面牆")
    normalized = {name: tuple(zones) for name, zones in surface_zones_by_wall.items()}
    validate_surface_partition(room, normalized)
    whole: dict[Wall, SurfaceMaterial] = {}
    rectangles: dict[Wall, tuple[RectangleZone, ...]] = {}
    for name, zones in normalized.items():
        rectangular = tuple(zone for zone in zones if isinstance(zone, RectangleZone))
        if rectangular and len(rectangular) == len(zones):
            rectangles[Wall.from_name(name)] = rectangular
        else:
            whole[Wall.from_name(name)] = whole_wall_materials({name: zones})[name]
    return wall_reflection_sources(whole, frequencies_hz, density_kg_m3=density_kg_m3,
        sound_speed_m_s=sound_speed_m_s, rho_c_pa_s_per_m=rho_c_pa_s_per_m, notes_by_wall=notes_by_wall,
        rectangle_zones_by_wall=rectangles or None, room=room if rectangles else None)


def lane_reflection_sources(
    materials_by_wall: Mapping[Wall, SurfaceMaterial],
    frequencies_hz: tuple[float, ...], *, rho_c_pa_s_per_m: float, sound_speed_m_s: float,
    density_kg_m3: float | None,
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None,
    surface_zones_by_wall: Mapping[str, Sequence[SurfaceZone]] | None = None,
    room: Room | None = None,
) -> WallReflectionSources:
    """報表讀者的共用入口：六面材料必填，構造一定要帶場景密度。

    只給 ρc（特性阻抗）的舊入口照用那個 ρc，密度只拿來過驗證，固定牆用不到；
    構造的 Z 要用場景密度，從 ρc 除回來可能差最後一位，所以不准反推。
    """
    given_materials = (tuple(z.material for zones in surface_zones_by_wall.values() for z in zones)
                       if surface_zones_by_wall is not None else tuple(materials_by_wall.values()))
    if density_kg_m3 is None and any(not isinstance(m, ConstantImpedance) for m in given_materials):
        raise ValueError("構造覆寫必須帶場景密度，不從 rho_c 反推")
    density = rho_c_pa_s_per_m / sound_speed_m_s if density_kg_m3 is None else density_kg_m3
    if surface_zones_by_wall is not None:
        if materials_by_wall:
            raise ValueError("整面材料與表面分區不可同時提供")
        if room is None:
            raise ValueError("反射分區必須提供房間尺寸")
        return zoned_reflection_sources(room, surface_zones_by_wall, frequencies_hz, density_kg_m3=density,
            sound_speed_m_s=sound_speed_m_s, rho_c_pa_s_per_m=rho_c_pa_s_per_m, notes_by_wall=notes_by_wall)
    return wall_reflection_sources(materials_by_wall, frequencies_hz, density_kg_m3=density,
        sound_speed_m_s=sound_speed_m_s, rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        notes_by_wall=notes_by_wall)
