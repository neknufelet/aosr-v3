"""鏡像路徑的 JSON 與人看表輸出；只組資料與字串，不執行命令列。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from aosr.physics import totals
from aosr.physics.compare import _complex_hex_dec, _dec_hex

if TYPE_CHECKING:
    from aosr.geometry.shoebox import Bounce
    from aosr.physics.receivers import Receiver, ReceiverResult
    from aosr.physics.room_paths import RoomPath


def path_to_dict(path: RoomPath) -> dict[str, object]:
    """一條路徑攤成答案檔 ``paths`` 那一筆的形狀（含 dec 與 hex；反彈是額外欄位）。

    有材料時多 ``reflection_product`` 與 ``path_pressure``（各六個頻帶的複數格，跟答案檔同形）；
    沒有材料時那兩格是空清單，不寫（保持第三段 ``--json`` 輸出逐位不變）。
    """
    body: dict[str, object] = {
        "index": path.index,
        "order": path.order,
        "identity": list(path.identity),
        "image_xyz": {
            "x": _dec_hex(path.image[0]),
            "y": _dec_hex(path.image[1]),
            "z": _dec_hex(path.image[2]),
        },
        "dist_m": _dec_hex(path.dist_m),
        "delay_s": _dec_hex(path.delay_s),
        "bounces": [
            {
                "wall": "+".join(bounce.walls),
                # 這一格永遠是 true：展開器現在直接拒絕落在牆外的點，所以「有沒有在牆內」
                # 不再是量出來的狀態。保留它是因為單點 JSON 的形狀被「逐位等於主線」那題
                # 凍結著（tests/engine/test_multi_receiver.py），拿掉會讓既有輸出少一格。
                "point": {
                    "x": _dec_hex(bounce.point[0]),
                    "y": _dec_hex(bounce.point[1]),
                    "z": _dec_hex(bounce.point[2]),
                },
                "t": _dec_hex(bounce.t),
                "in_wall": True,
            }
            for bounce in path.bounces
        ],
    }
    if path.reflection_product:
        body["reflection_product"] = [
            _complex_hex_dec(z) for z in path.reflection_product
        ]
        body["path_pressure"] = [_complex_hex_dec(z) for z in path.path_pressure]
    if path.has_patch_materials:
        body["bounce_cells"] = [
            {"wall": wall, "row": row, "col": col}
            for wall, row, col in path.bounce_cells
        ]
    return body


def paths_to_payload(paths: list[RoomPath]) -> dict[str, object]:
    """全部路徑整包攤成 ``{"paths": [...]}``。"""
    return {"paths": [path_to_dict(p) for p in paths]}


def _json_payload(paths: list[RoomPath], has_materials: bool) -> dict[str, object]:
    """命令列 JSON：沒有材料維持舊形，有材料才加答案檔同形的 ``totals``。"""
    payload = paths_to_payload(paths)
    if has_materials:
        payload["totals"] = totals.totals_to_payload(totals.totals_from_paths(paths))
    return payload


def _multi_json_payload(
    receivers: tuple[Receiver, ...], results: dict[str, ReceiverResult]
) -> dict[str, object]:
    """多點 JSON：receivers 是依輸入排序的 id→record，並另列逐點 totals。"""
    rows: dict[str, dict[str, object]] = {}
    totals_by_receiver: dict[str, object] = {}
    for receiver in receivers:
        result = results[receiver.id]
        row: dict[str, object] = {
            "xyz": dict(zip("xyz", receiver.point.as_tuple(), strict=True)),
            "paths": [path_to_dict(path) for path in result.paths],
            "totals": None,
        }
        if result.totals is not None:
            row["totals"] = totals.totals_to_payload(result.totals)
        rows[receiver.id] = row
        totals_by_receiver[receiver.id] = row["totals"]
    return {"receivers": rows, "totals_by_receiver": totals_by_receiver}


def _bounce_cell(bounce: Bounce) -> tuple[str, str, str]:
    """一個反彈格的顯示：牆名、反彈點 ``(x,y,z)`` 與 ``t``。"""
    point = ", ".join(repr(v) for v in bounce.point)
    # in_wall 這一格永遠 True（展開器直接拒絕落在牆外的點），但單點表格的形狀被
    # 「逐位等於主線」那題凍結著，所以照原樣印。
    return "+".join(bounce.walls), f"({point})", f"t={bounce.t!r}, in_wall=True"


def _totals_table_lines(
    paths: list[RoomPath], frequencies: tuple[float, ...]
) -> list[str]:
    """有振幅的人看表尾：每頻帶的總壓力、相位、直達／反射能量與能量比。"""
    computed = totals.totals_from_paths(paths)
    lines = ["總量"]
    for idx, frequency in enumerate(frequencies):
        pressure = computed.pressure[idx]
        phase = math.degrees(math.atan2(pressure.imag, pressure.real))
        direct = computed.direct_energy[idx]
        reflected = computed.reflected_energy[idx]
        ratio_db = (
            "—"
            if direct == 0.0 or reflected == 0.0
            else f"{10.0 * math.log10(reflected / direct):.3f}"
        )
        lines.append(
            f"  f={frequency:g} Hz  |P|={abs(pressure):.3f}  相位={phase:.3f}°  "
            f"直達能量={direct:.3f}  反射能量={reflected:.3f}  反射/直達 dB={ratio_db}"
        )
    return lines


def _human_table(paths: list[RoomPath], frequencies: tuple[float, ...] = ()) -> str:
    """一張人看的表：每條一行，逐次反彈印成「x0→floor→yL」並列各反彈點。

    牆那一欄從接收點那一側往回列（第一面是最後碰到的牆），不是時間順序（票 #475）；
    欄名寫「walls(接收點往回)」，完整說法在 :data:`aosr.geometry.shoebox.WALL_SEQUENCE_ORDER`。

    有材料時振幅**另起一區塊**：每個頻帶自己一行（``f=125 Hz  |p|=0.3110  phase=-61.87°``），
    數字取 6 位有效（``:6g``），不要六個頻帶擠成一行 350 字元。
    """
    has_amplitude = any(p.reflection_product for p in paths)
    lines = [
        f"{'index':>5} {'order':>5}  {'walls(接收點往回)':<28} "
        f"{'img':<46} {'dist_m':>20} {'delay_s':>20}"
    ]
    for p in paths:
        wall_seq = (
            "direct"
            if not p.bounces
            else "→".join(wall for bounce in p.bounces for wall in bounce.walls)
        )
        image = " ".join(f"{v!r}" for v in p.image)
        lines.append(
            f"{p.index:>5} {p.order:>5}  {wall_seq:<28} "
            f"{image:<46} {p.dist_m!r:>20} {p.delay_s!r:>20}"
        )
        cell_cursor = 0
        for bounce in p.bounces:
            wall, point, detail = _bounce_cell(bounce)
            cell = ""
            if p.has_patch_materials:
                touched_cells = p.bounce_cells[
                    cell_cursor : cell_cursor + len(bounce.walls)
                ]
                if len(touched_cells) == 1:
                    _cell_wall, row, col = touched_cells[0]
                    cell = f" 格=({row}, {col})"
                else:
                    cell = " 格=" + "+".join(
                        f"{cell_wall}({row}, {col})"
                        for cell_wall, row, col in touched_cells
                    )
                cell_cursor += len(bounce.walls)
            lines.append(f"{'':>10}    ↳ {wall}: {point} {detail}{cell}")
        if has_amplitude and p.reflection_product and p.path_pressure:
            lines.append(f"{'':>10}    振幅")
            for idx in range(len(p.path_pressure)):
                freq = frequencies[idx] if idx < len(frequencies) else float("nan")
                mag = abs(p.path_pressure[idx])
                phase = math.degrees(
                    math.atan2(p.path_pressure[idx].imag, p.path_pressure[idx].real)
                )
                lines.append(
                    f"{'':>10}      f={freq:g} Hz  |p|={mag:.6g}  phase={phase:.6g}°"
                )
    if has_amplitude and frequencies:
        lines.extend(_totals_table_lines(paths, frequencies))
    return "\n".join(lines) + "\n"


def _multi_human_table(
    receivers: tuple[Receiver, ...],
    results: dict[str, ReceiverResult],
    frequencies: tuple[float, ...],
) -> str:
    """多點人看表：每個接收點一節，路徑與總量不跨點混加。"""
    sections: list[str] = []
    for receiver in receivers:
        x, y, z = receiver.point.as_tuple()
        heading = f"接收點 {receiver.id}（x={x!r}, y={y!r}, z={z!r}）\n"
        sections.append(heading + _human_table(results[receiver.id].paths, frequencies))
    return "\n".join(sections)
