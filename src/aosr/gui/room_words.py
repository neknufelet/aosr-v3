"""一間房給人看的一句（#1134）：尺寸、哪幾面貼了吸音材料。首頁與結果頁共用，只排字、不改存檔。

「貼了吸音材料」：貼在原牆上的構造、實測表、含板與空腔以外的層的多層構造（吸音棉、布、微穿孔板、開縫板…）；
固定阻抗與只有板＋空腔的多層構造（例如 109 的天花薄板＋空腔）當原本的面。
"""
from __future__ import annotations

from typing import Final

from aosr.gui.wall_words import wall_words
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, MeasuredImpedance, MountedOnWall, SurfaceMaterial
from aosr.physics.report_surfaces import RectangleZone
from aosr.physics.zone_fill import wall_size_m
from aosr.reporting.scheme import Scheme

FOUR_WALLS: Final = ("x0", "xL", "y0", "yL")


def treated(material: SurfaceMaterial) -> bool:
    """這一塊是不是「貼了吸音材料」：貼在原牆上、實測表、含板與空腔以外的層的多層構造。"""
    if isinstance(material, MountedOnWall | MeasuredImpedance):
        return True
    return isinstance(material, Assembly) and any(not isinstance(layer, BoardLayer | AirLayer)
                                                  for layer in material.layers)


def _metres(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def size_words(scheme: Scheme) -> str:
    room = scheme.scene.room_m
    return f"長 {_metres(room.Lx)}、寬 {_metres(room.Ly)}、高 {_metres(room.Lz)} 公尺"


def treated_zones(scheme: Scheme) -> dict[str, list[tuple[tuple[float, float], SurfaceMaterial]]]:
    """每一面有貼東西的區（沿牆的 u 範圍與材料）；整面的 u 是整面寬。"""
    room = scheme.scene.room_m
    found: dict[str, list[tuple[tuple[float, float], SurfaceMaterial]]] = {}
    for wall, zones in (scheme.scene.surface_zones_by_wall or {}).items():
        width = wall_size_m(room, wall)[0]
        for zone in zones:
            if treated(zone.material):
                span = zone.u_m if isinstance(zone, RectangleZone) else (0.0, width)
                found.setdefault(wall, []).append((span, zone.material))
    return found


def treated_words(scheme: Scheme) -> str:
    """一句話：哪幾面有貼東西（前牆 2 塊、後牆整面…）；都沒有就說原樣。"""
    names = wall_words(scheme)
    zones = treated_zones(scheme)
    every = scheme.scene.surface_zones_by_wall or {}
    parts = []
    for wall in (*FOUR_WALLS, "ceiling", "floor"):
        items = zones.get(wall, [])
        if items:
            whole = len(items) == 1 and len(every.get(wall, ())) == 1
            parts.append(f"{names[wall]}{'整面' if whole else f' {len(items)} 塊'}")
    return "貼了吸音材料：" + "、".join(parts) if parts else "牆面都是原樣，沒貼吸音材料"
