"""鞋盒房間的鏡像聲源路徑：讀輸入、算直達＋一階到三階反射路徑、列印。

**這一支做三件事。** ``load_room_input`` 讀一份 JSON 輸入檔（房間、聲源、接收點、聲速、
最大反射階數），``image_source_paths`` 用鏡像法（image source method）算直達加 ``max_order``
以內全部反射路徑（一階是六面牆各一次、二階/三階是 identity 枚舉出來的全部組合），``main``
是命令列入口。答案檔的讀取與逐條比對搬到 :mod:`aosr.physics.compare`（票 #206 第 2a 段），
這一支 import 進來用（``compare_paths`` 等），命令列行為逐字不變。

**怎麼跑**::

    uv run python -m aosr.physics.room_paths <input.json>

**數值契約（逐位一致）。** 鏡像座標由 :func:`aosr.geometry.shoebox.image_from_identity`
每軸一字算（``2*n*L + s*src``，不逐牆鏡射）；距離一律 :func:`aosr.geometry.shoebox.distance`
（``math.sqrt(dx*dx + dy*dy + dz*dz)``），到達時間＝距離／聲速。這些都是 ``float`` 的純
標準庫算術，``float.hex()`` 必須跟 ``blueprint/reference_room_answers*.json`` 逐字相等
（決策紙 precision-contract）。

**振幅（有 ``materials`` 才算，公式見 :mod:`aosr.physics.amplitude`）。** 輸入檔多一節
``materials``（``rho_c``、``frequencies_hz``、六面牆各一份阻抗清單）時，每條路徑補算反射
乘積與路徑壓力（雙精度複數），並跟振幅答案檔依契約比（決策紙
``docs/decisions/precision-contract-amplitude-phase-scaled.md``：反射乘積絕對差 ≤ 2^-21、
壓力相對差 ≤ 2^-21·(ωτ+1) + 2^-21/|refl|，相對差以複數差的模除以參考值的模計）。沒有
``materials`` 就照舊只算幾何，輸出跟第三段逐位不變。

**輸出走 ``print``。** 這一支的命令列標準輸出就是它的產品（人看的表、``--json`` 機器格式、
``--compare`` 逐條判決），規矩卡 ``style-guard`` 的輸出層白名單已經把這個檔列進去。

**身份（identity）的定義。** 六元組 ``(nx, sx, ny, sy, nz, sz)``，跟答案檔一致：每軸一組
（反射次數、正負號）。直達 path 是 ``(0,1,0,1,0,1)``；階數由 identity 獨立算
（``order_of``）。路徑順序照答案檔的 ``sorted((order, identity))``（直達是唯一的 order 0，
自然排最前）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aosr.geometry.shoebox import (
    Bounce,
    Point,
    Room,
    distance,
    enumerate_identities,
    expand_bounces,
    image_from_identity,
    order_of,
)
from aosr.physics.amplitude import (
    CANONICAL_WALLS,
    Materials,
    path_amplitude,
)
from aosr.physics import totals
from aosr.physics.compare import (
    AnswerFile,
    PathComparison as PathComparison,
    _complex_hex_dec as _complex_hex_dec,
    _dec_hex as _dec_hex,
    _group_total_diffs,
    _load_answer as _load_answer,
    _load_answer_file,
    _mapping as _mapping,
    compare_paths as compare_paths,
    wall_name_seq_from_identity as wall_name_seq_from_identity,
)

# max_order 的合法範圍（含）：1 到 3。0 與負數不是一段路徑都沒有就是非法；4 以上是上一代
# 的上限、不是「還沒寫」——所以超出這個範圍一律 ValueError，不是 NotImplementedError。
SUPPORTED_MIN_ORDER: Final[int] = 1
SUPPORTED_MAX_ORDER: Final[int] = 3


@dataclass(frozen=True)
class RoomInput:
    """一份輸入檔解析後的結果：房、聲源、接收點、聲速、最大反射階數、可選的材料。

    ``materials`` 是 ``None`` 的時候照舊只算幾何（第三段的行為不變）；有值的時候每條路徑
    多算反射乘積與路徑壓力（票 #194 第二段）。
    """

    room: Room
    source: Point
    receiver: Point
    sound_speed: float
    max_order: int
    materials: Materials | None = None


@dataclass(frozen=True)
class RoomPath:
    """一條路徑（直達或一階以上反射）。

    ``identity`` 是答案檔那個六元組；``image`` 是（鏡像）聲源座標；``dist_m`` 是鏡像到
    接收點的直線距離；``delay_s`` 是距離／聲速；``bounces`` 是逐次反彈的展開（牆名、
    反彈點、線段參數 ``t``、在不在牆內），直達路徑是空 tuple。``reflection_product`` 與
    ``path_pressure`` 各有六個頻帶複數（跟答案檔同形），沒有材料時是空 tuple。
    """

    index: int
    order: int
    identity: tuple[int, int, int, int, int, int]
    image: tuple[float, float, float]
    dist_m: float
    delay_s: float
    bounces: tuple[Bounce, ...]
    reflection_product: tuple[complex, ...] = ()
    path_pressure: tuple[complex, ...] = ()


def _one_path(
    room: Room,
    source: Point,
    receiver: Point,
    c: float,
    identity: tuple[int, int, int, int, int, int],
    materials: Materials | None,
) -> RoomPath:
    """由一個 identity 六元組算出一條路徑（直達與任何階都適用）。

    ``materials`` 非 ``None`` 時多算反射乘積與路徑壓力（六個頻帶複數）；``None`` 時那兩格
    是空 tuple（照舊只算幾何）。
    """
    image = image_from_identity(room, identity, source)
    dist = distance(image, receiver)
    bounces = expand_bounces(room, identity, source, receiver)
    reflection_product: tuple[complex, ...] = ()
    path_pressure: tuple[complex, ...] = ()
    if materials is not None:
        reflection_product, path_pressure = path_amplitude(
            materials,
            identity,
            dist,
            c,
            receiver.as_tuple(),
            image.as_tuple(),
        )
    return RoomPath(
        index=-1,
        order=order_of(identity),
        identity=identity,
        image=image.as_tuple(),
        dist_m=dist,
        delay_s=dist / c,
        bounces=bounces,
        reflection_product=reflection_product,
        path_pressure=path_pressure,
    )


def image_source_paths(
    room: Room,
    source: Point,
    receiver: Point,
    c: float,
    max_order: int = 1,
    materials: Materials | None = None,
) -> list[RoomPath]:
    """算直達＋ ``max_order`` 以內全部反射路徑，順序 ``sorted((order, identity))``。

    ``max_order`` 合法範圍 1～3（含）；0、負數、4 以上丟 ``ValueError``：4 以上是上一代的
    上限、不是「還沒寫」所以不是 ``NotImplementedError``，0 與負數根本不成一段合約。
    identity 由 :func:`enumerate_identities` 照 donor 的去重規則枚舉（跟答案檔的 identity
    集合同一套），每一條用 :func:`image_from_identity` 直接算鏡像（不逐牆鏡射）、用
    :func:`expand_bounces` 展開逐次反彈（反彈點 ``in_wall`` 照實量，``False`` 不丟路徑）。
    ``materials`` 非 ``None`` 時每條路徑補上反射乘積與路徑壓力（見 :mod:`aosr.physics.amplitude`）。
    """
    if not SUPPORTED_MIN_ORDER <= max_order <= SUPPORTED_MAX_ORDER:
        raise ValueError(
            f"max_order={max_order} 不在合法範圍 [{SUPPORTED_MIN_ORDER}, "
            f"{SUPPORTED_MAX_ORDER}]：4 以上是上一代的上限、不是還沒寫，0 與負數不成合約"
        )

    paths = [
        _one_path(room, source, receiver, c, identity, materials)
        for identity in enumerate_identities(max_order)
    ]

    ordered = sorted(paths, key=lambda p: (p.order, p.identity))
    numbered = [
        RoomPath(
            index=i,
            order=p.order,
            identity=p.identity,
            image=p.image,
            dist_m=p.dist_m,
            delay_s=p.delay_s,
            bounces=p.bounces,
            reflection_product=p.reflection_product,
            path_pressure=p.path_pressure,
        )
        for i, p in enumerate(ordered)
    ]
    return numbered


# ── 載入輸入檔 ──────────────────────────────────────────────────────────────

# 輸入檔頂層欄位（跟答案檔 ``parameters`` 的幾何/聲學欄位同一套，不含 units/convention/
# grid_shapes 這種 donor 自己的中介資料）。``materials`` 是選填：有就多算振幅、沒有就照舊
# 只算幾何（第三段的行為不變）。
_INPUT_KEYS: Final[tuple[str, ...]] = (
    "room",
    "source_xyz_m",
    "receiver_xyz_m",
    "sound_speed_m_s",
    "max_order",
)
_OPTIONAL_KEYS: Final[tuple[str, ...]] = ("materials",)
_ROOM_KEYS: Final[tuple[str, ...]] = ("Lx_m", "Ly_m", "Lz_m")
_XYZ_KEYS: Final[tuple[str, ...]] = ("x", "y", "z")
_MATERIAL_KEYS: Final[tuple[str, ...]] = (
    "rho_c",
    "frequencies_hz",
    "floor",
    "ceiling",
    "x0",
    "xL",
    "y0",
    "yL",
)


def _number(node: object, where: str) -> float:
    """把一個節點收窄成 float。不是數（或布林）就丟 ValueError 指名在哪一格。"""
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise ValueError(f"{where} 不是一個數：{node!r}")
    return float(node)


def _positive(node: object, where: str) -> float:
    """收窄成嚴格大於 0 的 float。0 或負數就丟 ValueError 指名在哪一格。"""
    value = _number(node, where)
    if value <= 0.0:
        raise ValueError(f"{where} 必須大於 0（是 {value!r}），房間三邊與聲速不為 0")
    return value


def _positive_finite(node: object, where: str, tail: str) -> float:
    """收窄成嚴格大於 0 且有限的 float。0、負數、inf、nan 都 ValueError，``tail`` 補一句
    講這是哪個量（rho_c 用它把錯誤訊息講對，不再借房間三邊的尾巴）。"""
    value = _number(node, where)
    if not math.isfinite(value):
        raise ValueError(f"{where} 不是有限的數（是 {value!r}）")
    if value <= 0.0:
        raise ValueError(f"{where} 必須大於 0（是 {value!r}），{tail}")
    return value


def _integer(node: object, where: str) -> int:
    """把一個節點收窄成 int。不是整數（或布林）就丟 ValueError 指名在哪一格。"""
    if isinstance(node, bool) or not isinstance(node, int):
        raise ValueError(f"{where} 不是一個整數：{node!r}")
    return node


def _xyz(mapping: dict[str, object], where: str) -> Point:
    """``source_xyz_m``／``receiver_xyz_m`` 那一格：x/y/z 三個非負 float 組成的點。

    缺 x/y/z 任何一軸的訊息講「缺欄位」，不是「不是一個數」。
    """
    for key in _XYZ_KEYS:
        if key not in mapping:
            raise ValueError(f"{where} 缺欄位：{key}")
    return Point(
        x=_number(mapping["x"], f"{where}.x"),
        y=_number(mapping["y"], f"{where}.y"),
        z=_number(mapping["z"], f"{where}.z"),
    )


def _reject_extra(mapping: dict[str, object], allowed: tuple[str, ...], where: str) -> None:
    """一份一層表裡出現白名單以外的鍵就丟 ValueError，列出多的鍵名。"""
    extra = sorted(set(mapping) - set(allowed))
    if extra:
        raise ValueError(f"{where} 有多餘欄位：{', '.join(extra)}")


def _complex_cell(node: object, where: str) -> complex:
    """把一個阻抗格（``{"real":…, "imag":…}``）收窄成複數。缺格／不是數／實部 ≤ 0／
    實虛非有限（inf、nan）都 ValueError。"""
    cell = _mapping(node, where)
    for key in ("real", "imag"):
        if key not in cell:
            raise ValueError(f"{where} 缺欄位：{key}")
    real = _number(cell["real"], f"{where}.real")
    imag = _number(cell["imag"], f"{where}.imag")
    if not math.isfinite(real) or not math.isfinite(imag):
        raise ValueError(f"{where} 實虛部不是有限的數（是 {real!r} 與 {imag!r}）")
    if real <= 0.0:
        raise ValueError(f"{where}.real 必須嚴格大於 0（是 {real!r}），阻抗實部不為 0")
    return complex(real, imag)


def _wall_impedance_row(node: object, wall: str, where: str, n_freq: int) -> tuple[complex, ...]:
    """一面牆的阻抗清單：必須有 ``n_freq`` 個複數格，頻帶數對不上就 ValueError。"""
    if not isinstance(node, list):
        raise ValueError(f"{where} 不是一串東西：{node!r}")
    if len(node) != n_freq:
        raise ValueError(f"{where} 有 {len(node)} 格，不是 {n_freq} 格（頻帶數對不上）")
    return tuple(_complex_cell(entry, f"{where}[{idx}]") for idx, entry in enumerate(node))


def _load_materials(node: object) -> Materials:
    """把輸入檔的 ``materials`` 節收窄成 :class:`Materials`。

    欄位：``rho_c``（Pa·s/m，嚴格大於 0 且有限）、``frequencies_hz``（非空、每個嚴格大於 0
    且有限）、每面牆（floor/ceiling/x0/xL/y0/yL）一個阻抗清單（每個頻帶一個複數
    ``{real, imag}``）。缺牆、頻帶數對不上、阻抗實部 ≤ 0 或實虛非有限、頻率非正或非有限、
    ``frequencies_hz`` 空，都 ValueError 指明哪一格。
    """
    table = _mapping(node, "materials")
    for key in _MATERIAL_KEYS:
        if key not in table:
            raise ValueError(f"materials 缺欄位：{key}")
    _reject_extra(table, _MATERIAL_KEYS, "materials")

    rho_c = _positive_finite(table["rho_c"], "materials.rho_c", "rho_c 不為 0")
    freqs_raw = table["frequencies_hz"]
    if not isinstance(freqs_raw, list):
        raise ValueError(f"materials.frequencies_hz 不是一串東西：{freqs_raw!r}")
    frequencies = tuple(_number(v, f"materials.frequencies_hz[{i}]") for i, v in enumerate(freqs_raw))
    if len(frequencies) == 0:
        raise ValueError("materials.frequencies_hz 是空的：至少要一個頻帶")
    for f in frequencies:
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError(f"materials.frequencies_hz 裡有非正或非有限的頻率（是 {frequencies!r}）")
    n_freq = len(frequencies)

    walls: dict[str, tuple[complex, ...]] = {}
    for wall in CANONICAL_WALLS:
        walls[wall] = _wall_impedance_row(table[wall], wall, f"materials.{wall}", n_freq)
    return Materials(rho_c=rho_c, frequencies_hz=frequencies, walls=walls)


def load_room_input(path: Path) -> RoomInput:
    """讀一份 JSON 輸入檔，回 :class:`RoomInput`。

    欄位：``room.Lx_m/Ly_m/Lz_m``（三邊嚴格大於 0）、``source_xyz_m.x/y/z``、
    ``receiver_xyz_m.x/y/z``、``sound_speed_m_s``（嚴格大於 0）、``max_order``（整數），
    選填 ``materials``（見 :func:`_load_materials`）。缺欄位的訊息講「缺欄位」，多餘欄位的
    訊息列出多的欄位名（輸入檔寫錯字才抓得到），0 或負的房間三邊／聲速講哪一格。``path`` 必填。
    """
    with path.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _mapping(data, "輸入檔")

    for key in _INPUT_KEYS:
        if key not in root:
            raise ValueError(f"輸入檔缺欄位：{key}")
    _reject_extra(root, _INPUT_KEYS + _OPTIONAL_KEYS, "輸入檔")

    room = _mapping(root["room"], "room")
    for key in _ROOM_KEYS:
        if key not in room:
            raise ValueError(f"輸入檔缺欄位：room.{key}")
    _reject_extra(room, _ROOM_KEYS, "room")

    source_map = _mapping(root["source_xyz_m"], "source_xyz_m")
    receiver_map = _mapping(root["receiver_xyz_m"], "receiver_xyz_m")
    source = _xyz(source_map, "source_xyz_m")
    receiver = _xyz(receiver_map, "receiver_xyz_m")
    _reject_extra(source_map, _XYZ_KEYS, "source_xyz_m")
    _reject_extra(receiver_map, _XYZ_KEYS, "receiver_xyz_m")

    materials = _load_materials(root["materials"]) if "materials" in root else None

    return RoomInput(
        room=Room(
            Lx=_positive(room["Lx_m"], "room.Lx_m"),
            Ly=_positive(room["Ly_m"], "room.Ly_m"),
            Lz=_positive(room["Lz_m"], "room.Lz_m"),
        ),
        source=source,
        receiver=receiver,
        sound_speed=_positive(root["sound_speed_m_s"], "sound_speed_m_s"),
        max_order=_integer(root["max_order"], "max_order"),
        materials=materials,
    )


# ── JSON 序列化（跟答案檔 paths 同形）────────────────────────────────────────


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
                "wall": bounce.wall,
                "point": {
                    "x": _dec_hex(bounce.point[0]),
                    "y": _dec_hex(bounce.point[1]),
                    "z": _dec_hex(bounce.point[2]),
                },
                "t": _dec_hex(bounce.t),
                "in_wall": bounce.in_wall,
            }
            for bounce in path.bounces
        ],
    }
    if path.reflection_product:
        body["reflection_product"] = [_complex_hex_dec(z) for z in path.reflection_product]
        body["path_pressure"] = [_complex_hex_dec(z) for z in path.path_pressure]
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


# ── 命令列 ───────────────────────────────────────────────────────────────────


def _bounce_cell(bounce: Bounce) -> tuple[str, str, str]:
    """一個反彈格的顯示：牆名、反彈點 ``(x,y,z)``、``t`` 與在不在牆上。"""
    point = ", ".join(repr(v) for v in bounce.point)
    flag = "True" if bounce.in_wall else "False"
    return bounce.wall, f"({point})", f"t={bounce.t!r}, in_wall={flag}"


def _totals_table_lines(paths: list[RoomPath], frequencies: tuple[float, ...]) -> list[str]:
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

    有材料時振幅**另起一區塊**：每個頻帶自己一行（``f=125 Hz  |p|=0.3110  phase=-61.87°``），
    數字取 6 位有效（``:6g``），不要六個頻帶擠成一行 350 字元。
    """
    has_amplitude = any(p.reflection_product for p in paths)
    lines = [
        f"{'index':>5} {'order':>5}  {'walls(時序)':<28} "
        f"{'img':<46} {'dist_m':>20} {'delay_s':>20}"
    ]
    for p in paths:
        wall_seq = "direct" if not p.bounces else "→".join(b.wall for b in p.bounces)
        image = " ".join(f"{v!r}" for v in p.image)
        lines.append(
            f"{p.index:>5} {p.order:>5}  {wall_seq:<28} "
            f"{image:<46} {p.dist_m!r:>20} {p.delay_s!r:>20}"
        )
        for bounce in p.bounces:
            wall, point, detail = _bounce_cell(bounce)
            lines.append(f"{'':>10}    ↳ {wall}: {point} {detail}")
        if has_amplitude and p.reflection_product and p.path_pressure:
            lines.append(f"{'':>10}    振幅")
            for idx in range(len(p.path_pressure)):
                freq = frequencies[idx] if idx < len(frequencies) else float("nan")
                mag = abs(p.path_pressure[idx])
                phase = math.degrees(math.atan2(p.path_pressure[idx].imag, p.path_pressure[idx].real))
                lines.append(
                    f"{'':>10}      f={freq:g} Hz  |p|={mag:.6g}  phase={phase:.6g}°"
                )
    if has_amplitude and frequencies:
        lines.extend(_totals_table_lines(paths, frequencies))
    return "\n".join(lines) + "\n"


def _print_path_rows(results: list[PathComparison]) -> None:
    """印逐條路徑判決；總判決由 :func:`_path_verdict` 組字。"""
    for result in results:
        if result.diffs:
            print(
                f"不同  index {result.index}（牆次簽名 {result.wall_seq}）  "
                + "；".join(result.diffs)
            )
            continue
        suffix = ""
        if result.max_refl_frac > 0.0 or result.max_pp_frac > 0.0:
            suffix = (
                f"（反射用到界線 {result.max_refl_frac * 100:.3f}%、"
                f"壓力 {result.max_pp_frac * 100:.3f}%）"
            )
        print(f"相同  index {result.index}（牆次簽名 {result.wall_seq}）{suffix}")


def _path_verdict(results: list[PathComparison], answer_has_amp: bool) -> str:
    """組路徑總判決，維持第 2a 段既有字句。"""
    total = len(results)
    different = sum(1 for r in results if r.diffs)
    max_refl_frac = max((r.max_refl_frac for r in results), default=0.0)
    max_pp_frac = max((r.max_pp_frac for r in results), default=0.0)
    worst = max(
        (r.worst for r in results if r.worst is not None),
        key=lambda w: w[4],
        default=None,
    )
    if different == 0:
        verdict = f"{total} 條全部相同"
        if not answer_has_amp:
            verdict += "；答案檔無振幅，未比"
        elif max_refl_frac > 0.0 or max_pp_frac > 0.0:
            verdict += (
                f"，振幅全部在契約內（反射最大用到 {max_refl_frac * 100:.3f}%、"
                f"壓力 {max_pp_frac * 100:.3f}%）"
            )
        return verdict
    verdict = f"{total} 條裡有 {different} 條不同"
    if worst is not None:
        kind, f_idx, diff, tol, frac = worst
        verdict += f"；超出最多：{kind}[{f_idx}] 差 {diff!r} = 界線 {tol!r} 的 {frac:.3f} 倍"
    return verdict


def _print_total_rows(
    comparison: totals.TotalsComparison, frequencies: tuple[float, ...]
) -> None:
    """從整份總量裁判的差異清單印每頻帶判決，不重跑裁判。"""
    grouped = _group_total_diffs(comparison.diffs, len(frequencies))
    for frequency, band_diffs in zip(frequencies, grouped, strict=True):
        if band_diffs:
            print(f"總量  f={frequency:g} Hz 超界：" + "；".join(band_diffs))
            continue
        print(f"總量  f={frequency:g} Hz 在契約內")


def _totals_verdict(comparison: totals.TotalsComparison) -> str:
    """組總量的最後一句：三個最大百分比，或超界格數與最壞一格。"""
    if not comparison.diffs:
        return (
            "總量全部在契約內（總壓力最大用到 "
            f"{comparison.max_pressure_frac * 100:.3f}%、"
            f"直達 {comparison.max_direct_frac * 100:.3f}%、"
            f"反射 {comparison.max_reflected_frac * 100:.3f}%）"
        )
    verdict = f"總量有 {len(comparison.diffs)} 格超界"
    if comparison.worst is not None:
        kind, f_idx, diff, bound, frac = comparison.worst
        verdict += (
            f"，超出最多：{kind}[{f_idx}] 差 {diff!r} = "
            f"界線 {bound!r} 的 {frac:.3f} 倍"
        )
    return verdict


def _print_compare(
    results: list[PathComparison],
    answer_has_amp: bool,
    total_note: str | None,
    total_comparison: totals.TotalsComparison | None = None,
    frequencies: tuple[float, ...] = (),
) -> None:
    """先印逐條路徑，再印每頻帶總量，最後把兩個總判決接成一行。"""
    _print_path_rows(results)
    path_verdict = _path_verdict(results, answer_has_amp)
    if total_comparison is None:
        print(f"{path_verdict}；{total_note}")
        return
    _print_total_rows(total_comparison, frequencies)
    print(f"{path_verdict}；{_totals_verdict(total_comparison)}")


def _total_comparison_state(
    paths: list[RoomPath],
    answer: AnswerFile,
    frequencies: tuple[float, ...] | None,
) -> tuple[
    str | None,
    totals.TotalsComparison | None,
]:
    """決定總量是未比或實比；形狀錯誤原樣交給命令列轉離開碼 2。"""
    if not answer.has_totals:
        return "答案檔無總量，未比", None
    if frequencies is None or not all(path.path_pressure for path in paths):
        return "我方沒有振幅，總量未比", None
    answer_totals = _mapping(answer.totals, "totals")
    comparison = totals.compare_totals(paths, answer_totals, frequencies)
    return None, comparison


def _compare_command(
    answer_path: Path,
    paths: list[RoomPath],
    frequencies: tuple[float, ...] | None,
) -> int:
    """執行 ``--compare``：路徑與總量判決、報表及三種離開碼。"""
    if not answer_path.exists():
        print(f"讀不到答案檔：{answer_path}")
        return 2
    try:
        answer = _load_answer_file(answer_path)
    except (ValueError, OSError) as exc:
        print(f"答案檔讀不進或形狀不對：{exc}")
        return 2

    results = compare_paths(paths, answer.paths, frequencies)
    answer_has_amp = any(
        "reflection_product" in entry or "path_pressure" in entry for entry in answer.paths
    )
    should_compare_totals = (
        answer.has_totals
        and frequencies is not None
        and all(path.path_pressure for path in paths)
    )
    if should_compare_totals:
        try:
            totals.totals_from_paths(paths)
        except ValueError as exc:
            print(f"我方總量算不出來：{exc}")
            return 2
    try:
        total_note, total_comparison = _total_comparison_state(paths, answer, frequencies)
    except ValueError as exc:
        print(f"答案檔 totals 形狀不對：{exc}")
        return 2
    _print_compare(
        results,
        answer_has_amp,
        total_note,
        total_comparison,
        frequencies or (),
    )
    path_failed = any(result.diffs for result in results)
    totals_failed = total_comparison is not None and bool(total_comparison.diffs)
    return 1 if path_failed or totals_failed else 0


def main(argv: list[str]) -> int:
    """命令列入口：算路徑，印人看的表；``--json`` 印機器格式；``--compare`` 逐條比。

    回傳離開碼：``--compare`` 全同回 0、有不同回 1、讀不到檔回 2；程式自己炸掉（未預期
    例外）也要回 2，不准回 1。
    """
    parser = argparse.ArgumentParser(description="鞋盒房間的鏡像聲源路徑（直達＋一階到三階反射）")
    parser.add_argument("input", type=Path, help="輸入 JSON 檔（room/source/receiver/sound_speed/max_order）")
    parser.add_argument("--json", action="store_true", help="印答案檔 paths 同形的機器格式")
    parser.add_argument("--compare", type=Path, help="跟這份答案檔逐條比 hex")
    args = parser.parse_args(argv)

    try:
        if not args.input.exists():
            print(f"讀不到輸入檔：{args.input}")
            return 2

        try:
            inputs = load_room_input(args.input)
        except (ValueError, OSError) as exc:
            print(f"輸入檔讀不進或欄位不對：{exc}")
            return 2

        try:
            paths = image_source_paths(
                inputs.room,
                inputs.source,
                inputs.receiver,
                inputs.sound_speed,
                inputs.max_order,
                inputs.materials,
            )
        except ValueError as exc:
            # 兩種 ValueError 都會走到這：max_order 不合法，或反彈展開撞到退化組態
            # （反彈點打在牆的邊上）。兩者都印那一句原本的訊息、回 2，不吞掉。
            print(exc)
            return 2

        if args.compare is not None:
            frequencies = inputs.materials.frequencies_hz if inputs.materials is not None else None
            return _compare_command(args.compare, paths, frequencies)

        if args.json:
            payload = _json_payload(paths, inputs.materials is not None)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        freqs = inputs.materials.frequencies_hz if inputs.materials is not None else ()
        print(_human_table(paths, freqs), end="")
        return 0
    except Exception as exc:
        print(f"未預期錯誤：{exc!r}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
