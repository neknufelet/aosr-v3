"""網頁「自訂房間」那一頁的背景計算（票 #1315）：一份 Rhino 檔加每個圖層的吸音，算模態與主聽位聲壓。

網頁伺服器不切網格、不解方程（gmsh 只能在主執行緒初始化，解一次要幾分鐘），全部交給這個子行程。
輸入是工作管理員封存的一份小 JSON（模型檔路徑、每個圖層選的那一筆材料庫材料，null＝硬牆）；
每一筆照材料庫同一套算法換成這條頻率軸上的垂直入射阻抗（構造逐頻算、不內插）；
輸出一份 JSON：頻率軸、每支喇叭到主聽位的聲壓級、前幾個硬牆模態、網格的剖視圖。

頻率軸就是長方體那一路正式的有限元素軸（20 Hz 起、每八度 24 點、到 300 Hz），網格也照它的
最高頻切。``--max-frequency-hz`` 只給考卷用來切小一點的網格，網頁不傳。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray

from aosr import runtime
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ, FEM_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.config.product_air import product_air_environment
from aosr.config.room_shell_import import load_room_shell_import_settings
from aosr.geometry.rhino_room import read_rhino_room
from aosr.geometry.room_shell import MAIN_SEAT, check_room_shell
from aosr.geometry.room_shell_mesh import ShellMesh, generate_room_shell_mesh
from aosr.materials.library import LibraryEntry, preview_material
from aosr.physics.fem_shell import assemble_shell_operators, rigid_mode_frequencies, solve_shell_responses
from aosr.physics.wall_surfaces import normal_impedance
from aosr.reporting.room_shell_picture import cutaway_svg
from pydantic import TypeAdapter

MODE_COUNT = 16
ELEMENTS_PER_WAVELENGTH = 6
MESH_RANDOM_SEED = 1
_ENTRY: TypeAdapter[LibraryEntry] = TypeAdapter(LibraryEntry)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.reporting.room_shell_run")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--max-frequency-hz", type=float, default=FEM_GEOMETRIC_CROSSOVER_CAP_HZ)
    return parser


def _air_side(mesh: ShellMesh) -> NDArray[np.float64]:
    """每個邊界三角形朝空氣（朝自己那顆四面體）的法線。"""
    opposite: dict[tuple[int, ...], int] = {}
    for tet in mesh.tetrahedra:
        for k in range(4):
            opposite[tuple(sorted(int(v) for j, v in enumerate(tet) if j != k))] = int(tet[k])
    corners = mesh.nodes[mesh.boundary_triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inner = np.array([mesh.nodes[opposite[tuple(sorted(int(v) for v in row))]] for row in mesh.boundary_triangles])
    flip = np.einsum("ij,ij->i", normals, inner - corners[:, 0]) < 0
    normals[flip] *= -1.0
    return normals


def run(snapshot: Mapping[str, object], *, max_frequency_hz: float, engine_commit: str) -> dict[str, object]:
    started = time.monotonic()
    air, _ = product_air_environment()  # 自訂房間沒有場景，用產品標準空氣（跟材料頁同一支）
    settings = load_room_shell_import_settings(config_path("room_shell_import.toml"))
    room = read_rhino_room(Path(str(snapshot["model_path"])), settings.max_file_bytes)
    report = check_room_shell(room, settings)
    if not report.ok:
        raise ValueError("模型沒過匯入檢查：" + "；".join(p.text for p in report.problems if p.blocking))
    chosen = snapshot["material_by_layer"]
    if not isinstance(chosen, dict):
        raise ValueError("快照缺每個圖層的材料")
    mesh = generate_room_shell_mesh(room, weld_m=settings.weld_m, max_frequency_hz=max_frequency_hz,
                                    elements_per_wavelength=ELEMENTS_PER_WAVELENGTH,
                                    sound_speed_m_s=air.sound_speed_m_s, random_seed=MESH_RANDOM_SEED)
    operators = assemble_shell_operators(mesh)
    modes = rigid_mode_frequencies(operators, count=MODE_COUNT, sound_speed_m_s=air.sound_speed_m_s)
    frequencies = tuple(f for f in FEM_LANE_FREQUENCIES_HZ if f <= max_frequency_hz)
    impedances: dict[str, float | None | tuple[complex, ...]] = {}
    for name in mesh.layer_names:
        document = chosen.get(name)
        if document is None:
            impedances[name] = None
            continue
        material = preview_material(_ENTRY.validate_python(document))
        try:
            impedances[name], _ = normal_impedance(material, frequencies, density_kg_m3=air.air_density_kg_m3,
                                                   sound_speed_m_s=air.sound_speed_m_s)
        except ValueError as exc:
            raise ValueError(f"圖層 {name}：{exc}") from exc
    seat = next(item for item in room.seats if item.kind == MAIN_SEAT)
    curves = []
    for number, speaker in enumerate(room.speakers, start=1):
        pressure = solve_shell_responses(operators, layer_impedances=impedances, source=speaker.position,
                                         receivers=[seat.position], frequencies_hz=frequencies,
                                         density_kg_m3=air.air_density_kg_m3,
                                         sound_speed_m_s=air.sound_speed_m_s)[:, 0]
        level = 20.0 * np.log10(np.abs(pressure) / air.ref_pressure_pa)
        curves.append({"label": f"喇叭 {number}（{speaker.kind}）", "spl_db": [round(float(v), 2) for v in level]})
    # 圖層顏色照名字排序的位置配，跟網頁上傳時那張模型圖、圖層清單同一套顏色。
    order = sorted(mesh.layer_names)
    colors = np.array([order.index(name) for name in mesh.layer_names], dtype=np.int64)[mesh.boundary_layers]
    return {"engine_commit": engine_commit, "frequencies_hz": [round(f, 3) for f in frequencies],
            "modes_hz": [round(float(f), 2) for f in modes[1:]], "curves": curves,
            "mesh": {"nodes": len(mesh.nodes), "tetrahedra": len(mesh.tetrahedra),
                     "max_frequency_hz": max_frequency_hz, "volume_m3": round(mesh.volume_m3(), 3)},
            "mesh_svg": cutaway_svg(mesh.nodes[mesh.boundary_triangles], _air_side(mesh), [int(v) for v in colors]),
            "layer_names": order, "seconds": round(time.monotonic() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    runtime.configure_calculation_jax()  # 構造材料的阻抗要雙精度；子行程是正式入口，開機就固定
    args = _parser().parse_args(argv)
    snapshot: object = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError("快照不是物件")
    result = run(snapshot, max_frequency_hz=args.max_frequency_hz, engine_commit=args.engine_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=args.out.parent, delete=False, suffix=".tmp") as handle:
        json.dump(result, handle, ensure_ascii=False)
    Path(handle.name).replace(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
