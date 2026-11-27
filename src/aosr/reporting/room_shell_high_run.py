"""網頁「自訂房間」那一頁的高音背景計算：一份 Rhino 檔加每個圖層的吸音，算反射路徑、回聲圖、衰減與逐頻能量。

跟低頻那一支（:mod:`aosr.reporting.room_shell_run`）吃同一份快照（模型檔路徑、每個圖層是硬牆還是幾倍 ρc
的實數阻抗），輸出一份 JSON 給網頁畫圖。晚期分格與形狀因子只看房間，算一次；路徑表每支喇叭各算一次。

- 喇叭當全向聲源（自訂房間還沒有喇叭指向性可選）；散射每個圖層都用材料預設值。
- 「硬牆」那一檔用 :data:`~aosr.config.shell_lane.SHELL_HARD_WALL_IMPEDANCE_RATIO` 倍 ρc。
- 被擋掉的路徑：同一張路徑表把遮擋拿掉再算一次，多出來的那幾條就是被天雲、板子擋掉的。
- 嵌在牆上的喇叭（貼著外殼那一面）沿那一面的法線往房內挪 :data:`SPEAKER_WALL_NUDGE_M` 才算得了鏡像。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray

from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ, GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.config.product_air import product_air_environment
from aosr.config.room_shell_import import load_room_shell_import_settings
from aosr.config.shell_lane import SHELL_GEOMETRY_TOL_M, SHELL_HARD_WALL_IMPEDANCE_RATIO, SHELL_PATCH_EDGE_M
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.rhino_room import read_rhino_room
from aosr.geometry.room_shell import MAIN_SEAT, RoomShell, check_room_shell, signed_volume
from aosr.geometry.shell_faces import ShellFaces, make_occluders, shell_faces
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.shell_geometric_lane import ShellAcoustics, scattered_path_pressures, solve_shell_geometric_lane
from aosr.physics.shell_late_energy import ShellLateGeometry, shell_late_geometry, solve_shell_late_energy_by_order
from aosr.physics.shell_paths import ShellPathTable, shell_path_table
from aosr.physics.wall_surfaces import impedance_absorption_rows
from aosr.reporting.room_shell_picture import layer_color

# 回聲圖每條反射的大小取這一段頻率的平均（人耳最敏感、反射最分得出來的那一段）。
ECHO_BAND_HZ = (500.0, 2000.0)
# 衰減圖：逐階算到降 60 dB（每一階換成「平均自由路徑 ÷ 聲速」的時間）。要算幾階先用平均吸音率估
# （降 60 dB 要 ln(10⁶)/(−ln(1−ᾱ)) 階），多算一倍留給吸音不平均的房間，最多 DECAY_MAX_ORDERS 階。
DECAY_FLOOR_DB = -60.0
DECAY_MAX_ORDERS = 3000
DECAY_MAX_POINTS = 240
# 吸音只看實數阻抗、跟頻率無關，晚期衰減每個頻點都一樣，取一個頻點算就好。
DECAY_FREQUENCY_HZ = 1000.0
# 嵌牆喇叭往房內挪的距離（公尺）：5 毫米，一階反射的路程只多一公分，回聲圖上看不出來。
SPEAKER_WALL_NUDGE_M = 0.005
# 喇叭離聽位比這個近就不算（回聲圖以直達聲當 0 dB，距離是 0 就沒有基準）。
MIN_SPEAKER_SEAT_M = 0.1
OMNI = SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL)
FloatArray = NDArray[np.float64]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.reporting.room_shell_high_run")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    return parser


def high_frequencies() -> tuple[float, ...]:
    """高音那一路的頻率軸：正式細軸從低頻有限元素交棒那一點以上。"""
    return tuple(f for f in GEOMETRIC_LANE_FREQUENCIES_HZ if f >= FEM_GEOMETRIC_CROSSOVER_CAP_HZ)


def _impedances(ratios: Mapping[str, object], layers: tuple[str, ...], rho_c: float) -> dict[str, complex]:
    out: dict[str, complex] = {}
    for name in layers:
        if name not in ratios:
            raise ValueError(f"圖層「{name}」沒有選吸音")
        ratio = ratios[name]
        if ratio is not None and not isinstance(ratio, (int, float)):
            raise ValueError(f"圖層「{name}」的吸音不是數字")
        out[name] = complex((SHELL_HARD_WALL_IMPEDANCE_RATIO if ratio is None else float(ratio)) * rho_c)
    return out


def _room_volume(room: RoomShell) -> float:
    """空氣體積：外殼圍的減掉每個造型自己的體積。"""
    return abs(signed_volume(room.shell)) - sum(abs(signed_volume(solid.triangles)) for solid in room.solids)


def _polyline(table: ShellPathTable, index: int) -> list[list[float]]:
    points = [list(table.source)]
    points += [[round(float(v), 4) for v in table.bounce_points[index, b]]
               for b in range(table.max_order) if table.bounce_layer[index, b] >= 0]
    return [*points, list(table.receiver)]


def _via(table: ShellPathTable, index: int) -> list[str]:
    return [table.layers[int(table.bounce_layer[index, b])] for b in range(table.max_order)
            if table.bounce_layer[index, b] >= 0]


def _blocked(faces: ShellFaces, table: ShellPathTable) -> list[dict[str, object]]:
    """拿掉遮擋重算一次，多出來（反彈面序列不在原表裡）的就是被擋掉的路徑。"""
    free = shell_path_table(dataclasses.replace(faces, occluders=make_occluders(())), table.source, table.receiver,
                            max_order=table.max_order, tol_m=SHELL_GEOMETRY_TOL_M)
    kept = {tuple(int(v) for v in row) for row in table.bounce_plane}
    return [{"points": _polyline(free, i), "order": int(free.order[i]), "via": _via(free, i)}
            for i in range(free.path_count) if tuple(int(v) for v in free.bounce_plane[i]) not in kept]


def _off_the_wall(point: tuple[float, float, float], faces: ShellFaces) -> tuple[float, float, float]:
    """貼著（或差一點點穿出）外殼某一面的點，沿那一面的法線（朝空氣）挪到離面 :data:`SPEAKER_WALL_NUDGE_M`。"""
    moved = np.asarray(point, dtype=np.float64)
    planes = np.unique(faces.plane_of[:faces.shell_triangles])
    for plane in planes[faces.supporting[planes]]:
        height = float(moved @ faces.plane_normals[plane] - faces.plane_offsets[plane])
        if -SPEAKER_WALL_NUDGE_M < height < SPEAKER_WALL_NUDGE_M:
            moved = moved + (SPEAKER_WALL_NUDGE_M - height) * faces.plane_normals[plane]
    return (float(moved[0]), float(moved[1]), float(moved[2]))


def _db(values: FloatArray | tuple[float, ...]) -> list[float]:
    return [round(float(10.0 * np.log10(max(float(v), 1e-30))), 2) for v in values]


def _speaker(faces: ShellFaces, late: ShellLateGeometry, source: tuple[float, float, float],
             seat: tuple[float, float, float], impedance: dict[str, complex], rho_c: float, c: float,
             frequencies: tuple[float, ...]) -> dict[str, object]:
    table = shell_path_table(faces, source, seat, max_order=REFLECTION_ORDER_K, tol_m=SHELL_GEOMETRY_TOL_M)
    prepared = ShellAcoustics(faces, table, late)
    lane = solve_shell_geometric_lane(prepared=prepared, source_model=OMNI, sound_speed_m_s=c, rho_c_pa_s_per_m=rho_c,
                                      frequencies_hz=frequencies, impedance_by_layer=impedance)
    pressure = scattered_path_pressures(prepared=prepared, sound_speed_m_s=c, rho_c_pa_s_per_m=rho_c,
                                        frequencies_hz=frequencies, impedance_by_layer=impedance)
    band = (np.asarray(frequencies) >= ECHO_BAND_HZ[0]) & (np.asarray(frequencies) <= ECHO_BAND_HZ[1])
    level = np.mean(np.abs(pressure[:, band]) ** 2, axis=1)
    relative = _db(level / level[0])
    paths = [{"points": _polyline(table, i), "order": int(table.order[i]), "via": _via(table, i),
              "t_ms": round(float(table.dist_m[i] / c * 1000.0), 3), "db": relative[i]}
             for i in range(table.path_count)]
    early = np.asarray(lane.direct_energy) + np.asarray(lane.reflected_energy) + np.asarray(lane.interference_energy)
    return {"paths": paths, "blocked": _blocked(faces, table),
            "energy_db": {"direct": _db(lane.direct_energy), "early": _db(early), "late": _db(lane.late_energy),
                          "total": _db(lane.geometric_energy)}}


def _decay_orders(absorption: FloatArray, layer_areas: FloatArray) -> int:
    alpha = float(layer_areas @ absorption / layer_areas.sum())
    if alpha <= 0.0:
        return DECAY_MAX_ORDERS
    needed = 2.0 * np.log(10.0 ** (-DECAY_FLOOR_DB / 10.0)) / -np.log1p(-min(alpha, 1.0 - 1e-12))
    return int(min(DECAY_MAX_ORDERS, max(12, np.ceil(needed))))


def _decay(late: ShellLateGeometry, impedance: dict[str, complex], rho_c: float, c: float,
           volume: float) -> dict[str, object]:
    """晚期能量逐階到降 60 dB，每一階換成時間：第 k 階 ≈ k × 平均自由路徑 ÷ 聲速；另回 T30 推的殘響時間。"""
    rows = impedance_absorption_rows({name: (impedance[name],) for name in late.layers}, rho_c)
    absorption = np.asarray([rows[name][0] for name in late.layers], dtype=np.float64)
    orders = _decay_orders(absorption, late.layer_areas())
    band = solve_shell_late_energy_by_order(late, absorption[:, None], (DECAY_FREQUENCY_HZ,), max_order=orders).bands[0]
    energy = np.asarray(band.energy_by_order, dtype=np.float64)
    level = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-30))
    step_ms = 4.0 * volume / float(late.areas.sum()) / c * 1000.0
    times = (np.arange(len(level)) + 1) * step_ms
    end = int(np.argmax(level <= DECAY_FLOOR_DB)) + 1 if bool((level <= DECAY_FLOOR_DB).any()) else len(level)
    keep = np.unique(np.linspace(0, end - 1, min(end, DECAY_MAX_POINTS)).round().astype(np.int64))
    fit = (level <= -5.0) & (level >= -35.0)
    t60 = None
    if int(fit.sum()) >= 2:
        slope = float(np.polyfit(times[fit] / 1000.0, level[fit], 1)[0])
        t60 = round(-60.0 / slope, 2) if slope < 0 else None
    return {"points": [{"t_ms": round(float(times[k]), 2), "db": round(float(level[k]), 2)} for k in keep],
            "t60_s": t60}


def run(snapshot: Mapping[str, object], *, engine_commit: str,
        patch_edge_m: float = SHELL_PATCH_EDGE_M) -> dict[str, object]:
    """``patch_edge_m`` 只給考卷切粗一點省時間，網頁不傳。"""
    started = time.monotonic()
    air, _ = product_air_environment()
    settings = load_room_shell_import_settings(config_path("room_shell_import.toml"))
    room = read_rhino_room(Path(str(snapshot["model_path"])), settings.max_file_bytes)
    report = check_room_shell(room, settings)
    if not report.ok:
        raise ValueError("模型沒過匯入檢查：" + "；".join(p.text for p in report.problems if p.blocking))
    ratios = snapshot["impedance_ratio_by_layer"]
    if not isinstance(ratios, dict):
        raise ValueError("快照缺每個圖層的吸音")
    rho_c, c = air.air_density_kg_m3 * air.sound_speed_m_s, air.sound_speed_m_s
    faces = shell_faces(room, settings.coplanar_angle_deg, settings.coplanar_offset_m, SHELL_GEOMETRY_TOL_M)
    impedance = _impedances(ratios, faces.layers, rho_c)
    seat = next(item for item in room.seats if item.kind == MAIN_SEAT)
    positions = [_off_the_wall(speaker.position, faces) for speaker in room.speakers]
    for number, position in enumerate(positions, start=1):
        if float(np.linalg.norm(np.subtract(position, seat.position))) < MIN_SPEAKER_SEAT_M:
            raise ValueError(f"喇叭 {number} 跟聽位幾乎在同一點（不到 {MIN_SPEAKER_SEAT_M * 100:.0f} 公分），請把其中一個移開")
    # 先擋掉算不了的，再做吃時間的分格與形狀因子。
    late = shell_late_geometry(faces, patch_edge_m=patch_edge_m)
    frequencies = high_frequencies()
    speakers = [{"label": f"喇叭 {number}（{speaker.kind}）", "position": list(position),
                 **_speaker(faces, late, position, seat.position, impedance, rho_c, c, frequencies)}
                for number, (speaker, position) in enumerate(zip(room.speakers, positions, strict=True), start=1)]
    order = sorted(faces.layers)
    triangles = {name: np.round(faces.corners[faces.layer_of == faces.layers.index(name)], 4).tolist()
                 for name in order}
    return {"engine_commit": engine_commit, "frequencies_hz": [round(f, 3) for f in frequencies],
            "seat": list(seat.position), "speakers": speakers,
            "decay": _decay(late, impedance, rho_c, c, _room_volume(room)),
            "layers": [{"name": name, "color": layer_color(i)} for i, name in enumerate(order)],
            "triangles": triangles, "solid_layers": sorted({t.layer for solid in room.solids for t in solid.triangles}),
            "hidden_area_m2": round(late.hidden_area_m2, 2),
            "seconds": round(time.monotonic() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot: object = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError("快照不是物件")
    result = run(snapshot, engine_commit=args.engine_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=args.out.parent, delete=False, suffix=".tmp") as handle:
        json.dump(result, handle, ensure_ascii=False)
    Path(handle.name).replace(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
