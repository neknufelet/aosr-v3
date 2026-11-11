"""網頁去哪裡找一份切法的替身（surrogate，縮減模型；#1093，1-6c 建好的替身放進同一格）。

一份切法一格：``data_dir/surrogates/<格子代號>/surrogate.npz``，旁邊照縮減模型存檔的規矩放 manifest 與留出驗證收據。
格子代號是方案裡「匯出完整模型、核替身會用到的那幾格」的雜湊（照 ``export_operators`` 與搜尋命令列的准用閘門讀）：
房間尺寸、每面牆的分區矩形（代號、座標、順序都算：閘門照名字與順序比面片）、喇叭與座位座標（座位照座位組的順序）、
空氣（密度、聲速）、低頻軸（替身的頻率軸要算得了它）。材料、用途、散射、反射階數都不算：同一份切法換材料共用同一份替身。

物理身分刻意不算進來：程式改了物理身分，舊替身由命令列的准用閘門判（核得過就沿用、不行就拒收並講原因），網頁不自己猜。
這裡只回答「去哪裡找」；准不准用一律交給命令列的閘門（網格雜湊、物理身分、座標、空氣、頻率軸逐項核），不拿別份切法的替身頂替。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from aosr.geometry.shoebox import Wall
from aosr.physics.fem_reduced_receipt import receipt_path
from aosr.physics.fem_reduced_store import manifest_path
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.scheme import Scheme

SLOTS_DIR: Final = "surrogates"
SURROGATE_FILE: Final = "surrogate.npz"
SLOT_VERSION: Final = "aosr.gui.surrogate_slot.v1"


def slot_document(scheme: Scheme) -> dict[str, object]:
    """算格子代號的那一份內容（照牆序；整面一區的牆不列矩形）。"""
    scene = scheme.scene
    zones = scene.surface_zones_by_wall or {}
    partitions = {wall.wall_name(): [[zone.id, list(zone.u_m), list(zone.v_m)]
                                     for zone in zones.get(wall.wall_name(), ()) if isinstance(zone, RectangleZone)]
                  for wall in Wall.all()}
    return {
        "version": SLOT_VERSION,
        "room_m": [scene.room_m.length(axis) for axis in range(3)],
        "partitions": {wall: rectangles for wall, rectangles in partitions.items() if rectangles},
        "speakers": {name: list(point.as_tuple()) for name, point in sorted(scheme.speakers.items())},
        "receivers": [[point.receiver_id, list(point.position_m)] for point in scheme.receiver_set.points],
        "density_kg_m3": scene.density_kg_m3, "sound_speed_m_s": scene.sound_speed_m_s,
        "low_frequency_axis": None if scene.low_frequency_axis is None else scene.low_frequency_axis.value,
    }


def slot_key(scheme: Scheme) -> str:
    text = json.dumps(slot_document(scheme), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def slot_dir(data_dir: Path, scheme: Scheme) -> Path:
    return data_dir / SLOTS_DIR / slot_key(scheme)


def surrogate_path(data_dir: Path, scheme: Scheme) -> Path:
    """這份切法的替身該放的位置（不管在不在）。"""
    return slot_dir(data_dir, scheme) / SURROGATE_FILE


def missing_parts(path: Path) -> list[str]:
    """替身三件（模型、manifest、留出驗證收據）缺哪幾件；空的就是三件都在。"""
    parts = {"替身模型": path, "替身說明檔（manifest）": manifest_path(path), "留出驗證收據": receipt_path(path)}
    return [name for name, part in parts.items() if not part.is_file()]


def found_surrogate(data_dir: Path, scheme: Scheme) -> Path | None:
    """三件都在就回模型路徑，缺任何一件就當沒有（准不准用另由命令列閘門判）。"""
    path = surrogate_path(data_dir, scheme)
    return None if missing_parts(path) else path
