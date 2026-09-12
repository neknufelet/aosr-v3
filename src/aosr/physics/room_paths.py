"""鞋盒房間的鏡像聲源路徑：讀輸入、算直達＋一階到三階反射路徑、列印與比對。

**這一支做三件事。** ``load_room_input`` 讀一份 JSON 輸入檔（房間、聲源、接收點、聲速、
最大反射階數），``image_source_paths`` 用鏡像法（image source method）算直達加 ``max_order``
以內全部反射路徑（一階是六面牆各一次、二階/三階是 identity 枚舉出來的全部組合），``main``
是命令列入口。

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
    pressure_tolerance,
    reflection_tolerance,
)

# max_order 的合法範圍（含）：1 到 3。0 與負數不是一段路徑都沒有就是非法；4 以上是上一代
# 的上限、不是「還沒寫」——所以超出這個範圍一律 ValueError，不是 NotImplementedError。
SUPPORTED_MIN_ORDER: Final[int] = 1
SUPPORTED_MAX_ORDER: Final[int] = 3

# 直達路徑的 identity：全 0、全 +1。
_DIRECT_IDENTITY: Final[tuple[int, int, int, int, int, int]] = (0, 1, 0, 1, 0, 1)


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


@dataclass(frozen=True)
class PathComparison:
    """``--compare`` 對一條路徑的判決：index、牆名序列、差異清單（空＝完全相同）。

    ``max_refl_frac`` 是反射乘積「用到界線幾成」的最大值、``max_pp_frac`` 是壓力「用到界線
    幾成」的最大值（都是差／界線的比值：≤1 在契約內，>1 就超界；沒比振幅時兩者都是 0）。
    ``worst`` 記「超界最多」的那一格（``(量名, 頻帶, 差, 界線, 倍數)``），只在有超界時非空——
    判決行有超界時要印出它，而不是只印總數。判決行從這個結構印，不用字串 startswith 去撈。
    """

    index: int
    wall_seq: str
    diffs: list[str]
    max_refl_frac: float = 0.0
    max_pp_frac: float = 0.0
    worst: tuple[str, int, float, float, float] | None = None


def wall_name_seq_from_identity(identity: tuple[int, int, int, int, int, int]) -> str:
    """identity 六元組 → 牆名序列（canonical 順序展開，從 identity 純推、不靠幾何）。

    六元組 ``(nx, sx, ny, sy, nz, sz)`` 每軸一組（反射次數、正負號）。每軸兩面牆的反射
    次數拆成「zero 面幾次、L 面幾次」（``sign=+1`` 兩面各 ``|n|``、``sign=-1`` 依 ``n``
    的正負把奇數次分給兩面）；牆名照 canonical 順序 floor/ceiling/x0/xL/y0/yL 展開、用
    ``→`` 接起來。直達（全 0 全 +1）回 ``"direct"``。這是**牆次的簽名**（哪面牆幾次），
    不是逐次反彈的時間順序——時間順序是幾何量，答案檔沒有存，比對時不看。
    """
    if identity == _DIRECT_IDENTITY:
        return "direct"
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])

    def _zero_l(nx: int, sign: int) -> tuple[int, int]:
        """一軸的 (zero 面次數, L 面次數)。"""
        if sign == 1:
            return abs(nx), abs(nx)
        m = 2 * nx - 1
        order = abs(m)
        if m > 0:
            return order // 2, (order + 1) // 2
        return (order + 1) // 2, order // 2

    z_lo, z_hi = _zero_l(n[2], s[2])
    x_lo, x_hi = _zero_l(n[0], s[0])
    y_lo, y_hi = _zero_l(n[1], s[1])
    parts: list[str] = []
    parts.extend(["floor"] * z_lo)
    parts.extend(["ceiling"] * z_hi)
    parts.extend(["x0"] * x_lo)
    parts.extend(["xL"] * x_hi)
    parts.extend(["y0"] * y_lo)
    parts.extend(["yL"] * y_hi)
    return "→".join(parts)


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


def _mapping(node: object, where: str) -> dict[str, object]:
    """把一個節點收窄成一層表。不是表就丟 ValueError（指名在哪一格）。"""
    if not isinstance(node, dict):
        raise ValueError(f"{where} 不是一層表：{node!r}")
    return {str(key): value for key, value in node.items()}


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


# ── JSON 序列化與比對（跟答案檔 paths 同形）──────────────────────────────────


def _dec_hex(value: float) -> dict[str, str]:
    """一個浮點同時存十進位與十六進位兩格（跟答案檔的 ``dec``／``hex`` 同形）。"""
    return {"dec": repr(value), "hex": value.hex()}


def _complex_hex_dec(value: complex) -> dict[str, object]:
    """一個複數存實／虛／abs 三格，各含 ``dec`` 與 ``hex``（跟答案檔同形）。"""
    return {
        "real": _dec_hex(value.real),
        "imag": _dec_hex(value.imag),
        "abs": _dec_hex(abs(value)),
    }


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


def _hex_cell(node: object, where: str) -> str:
    """答案檔裡 ``dist_m``／``delay_s``／``image_xyz.<axis>`` 的 hex 字串。"""
    cell = _mapping(node, where)
    hexed = cell.get("hex")
    if not isinstance(hexed, str):
        raise ValueError(f"{where} 沒有 hex 那一格")
    return hexed


def _answer_hexes(
    path: dict[str, object],
) -> tuple[tuple[str, str, str] | None, str, str]:
    """一條答案檔 path → ((img_x_hex, img_y_hex, img_z_hex)|None, dist_hex, delay_hex)。

    ``image_xyz`` 只在幾何答案檔（``reference_room_answers*.json``）有；振幅答案檔
    （``reference_amplitude_*.json``）沒有那一格，回 ``None``（比對時跳過鏡像）。
    """
    image = path.get("image_xyz")
    if image is None:
        image_hexes: tuple[str, str, str] | None = None
    else:
        table = _mapping(image, "paths.image_xyz")
        image_hexes = (
            _hex_cell(table.get("x"), "image_xyz.x"),
            _hex_cell(table.get("y"), "image_xyz.y"),
            _hex_cell(table.get("z"), "image_xyz.z"),
        )
    return (
        image_hexes,
        _hex_cell(path.get("dist_m"), "dist_m"),
        _hex_cell(path.get("delay_s"), "delay_s"),
    )


def _dec_cell(node: object, where: str) -> float:
    """答案檔一個 ``{dec, hex}`` 格的 ``dec``，還原成 float（字串或數都收）。"""
    cell = _mapping(node, where)
    dec = cell.get("dec")
    if dec is None:
        raise ValueError(f"{where}.dec 是 null")
    if isinstance(dec, str):
        return float(dec)
    if isinstance(dec, (int, float)) and not isinstance(dec, bool):
        return float(dec)
    raise ValueError(f"{where}.dec 不是數：{dec!r}")


def _answer_complex_cell(node: object, where: str) -> complex:
    """答案檔一個複數格（``{real:{dec,…}, imag:{dec,…}, …}``），還原成複數。"""
    cell = _mapping(node, where)
    return complex(_dec_cell(cell.get("real"), f"{where}.real"), _dec_cell(cell.get("imag"), f"{where}.imag"))


def _answer_complex_list(node: object, where: str) -> list[complex]:
    """答案檔一條路徑的 ``reflection_product``／``path_pressure``（六頻帶複數清單）。"""
    if not isinstance(node, list):
        raise ValueError(f"{where} 不是一串東西：{node!r}")
    return [_answer_complex_cell(entry, f"{where}[{idx}]") for idx, entry in enumerate(node)]


def _compare_amplitude(
    ours: RoomPath,
    their_refl: list[complex],
    their_pp: list[complex],
    frequencies: tuple[float, ...],
) -> tuple[list[str], float, float, tuple[str, int, float, float, float] | None]:
    """比一條路徑的反射乘積與路徑壓力依契約逐格，回（差異清單、反射用到幾成、壓力用到
    幾成、超界最多的那一格）。

    反射乘積每分量（實、虛、abs）絕對差 ≤ :func:`aosr.physics.amplitude.reflection_tolerance`；
    路徑壓力相對差用**複數差的模**：``|Δp| ≤ pressure_tolerance·|p|``（``Δp`` 是複數差，
    不是逐分量各比）。``max_refl_frac``／``max_pp_frac`` 是「差／界線」的最大值（≤1 在契約
    內，>1 就超界）；``worst`` 記超界最多那格 ``(量名, 頻帶, 差, 界線, 倍數)``。
    """
    tau = ours.delay_s
    diffs: list[str] = []
    max_refl_frac = 0.0
    max_pp_frac = 0.0
    worst: tuple[str, int, float, float, float] | None = None

    def _frac(diff: float, tol: float) -> float:
        if tol == 0.0:
            return float("inf") if diff != 0.0 else 0.0
        return diff / tol

    def _note_worst(kind: str, f_idx: int, diff: float, tol: float) -> None:
        nonlocal worst
        frac = _frac(diff, tol)
        if worst is None or frac > worst[4]:
            worst = (kind, f_idx, diff, tol, frac)

    # 反射乘積：每分量絕對差 ≤ 2^-21（實、虛、abs 各比一次）。
    for f_idx, freq in enumerate(frequencies):
        if f_idx >= len(ours.reflection_product) or f_idx >= len(their_refl):
            break
        refl_answer = their_refl[f_idx]
        refl_mine = ours.reflection_product[f_idx]
        for comp_name, a_val, b_val in (
            ("real", refl_answer.real, refl_mine.real),
            ("imag", refl_answer.imag, refl_mine.imag),
            ("abs", abs(refl_answer), abs(refl_mine)),
        ):
            diff = abs(a_val - b_val)
            tol = reflection_tolerance(freq, tau, abs(refl_answer))
            if diff > tol:
                diffs.append(
                    f"reflection_product[{f_idx}].{comp_name} 超界：差 {diff!r} > 界線 {tol!r}"
                )
                _note_worst("reflection_product", f_idx, diff, tol)
            max_refl_frac = max(max_refl_frac, _frac(diff, tol))

    # 路徑壓力：複數差的模 ≤ tol·|p|（不是逐分量各比）。
    for f_idx, freq in enumerate(frequencies):
        if f_idx >= len(ours.path_pressure) or f_idx >= len(their_pp):
            break
        pp_answer = their_pp[f_idx]
        pp_mine = ours.path_pressure[f_idx]
        abs_refl = abs(their_refl[f_idx])
        tol_rel = pressure_tolerance(freq, tau, abs_refl)
        abs_tol = tol_rel * abs(pp_answer)
        diff = abs(pp_mine - pp_answer)
        if diff > abs_tol:
            diffs.append(
                f"path_pressure[{f_idx}] 超界：複數差模 {diff!r} > 界線 {abs_tol!r}"
            )
            _note_worst("path_pressure", f_idx, diff, abs_tol)
        max_pp_frac = max(max_pp_frac, _frac(diff, abs_tol))

    return diffs, max_refl_frac, max_pp_frac, worst


def _answer_identity(entry: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    """答案檔一條路徑的 identity 六元組。"""
    raw = entry.get("identity")
    if not isinstance(raw, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in raw):
        raise ValueError(f"答案檔 identity 不是整數串：{raw!r}")
    if len(raw) != 6:
        raise ValueError(f"答案檔 identity 不是六元組：{raw!r}")
    return (raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])


def _answer_index(entry: dict[str, object]) -> int:
    """答案檔一條路徑的 index。缺了或不是整數就丟 ValueError。"""
    raw = entry.get("index")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"答案檔 index 不是整數：{raw!r}")
    return raw


def _answer_order(entry: dict[str, object]) -> int:
    """答案檔一條路徑的 order。缺了或不是整數就丟 ValueError。"""
    raw = entry.get("order")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"答案檔 order 不是整數：{raw!r}")
    return raw


def _amplitude_comparison(
    ours: RoomPath,
    theirs: dict[str, object],
    frequencies: tuple[float, ...] | None,
    one: list[str],
) -> tuple[float, float, tuple[str, int, float, float, float] | None]:
    """比一條路徑的振幅（反射乘積與壓力）依契約，把差異寫進 ``one``、回（反射幾成、壓力
    幾成、超界最多那格）。

    條件反過來看「答案檔那一側有沒有振幅欄」：答案檔有而我方沒有（沒 materials／頻帶空）→
    報「我方沒有振幅可比」；答案檔沒振幅欄 → 不進這支（幾何那側已比完）；兩邊都有才逐格比。
    """
    answer_has_amp = "reflection_product" in theirs or "path_pressure" in theirs
    if not answer_has_amp:
        return 0.0, 0.0, None
    if not ours.reflection_product:
        one.append("我方沒有振幅可比")
        return 0.0, 0.0, None
    if "reflection_product" not in theirs or "path_pressure" not in theirs:
        one.append("答案檔沒有 reflection_product／path_pressure 振幅欄")
        return 0.0, 0.0, None
    if frequencies is None:
        one.append("我方沒有振幅可比")
        return 0.0, 0.0, None
    try:
        their_refl = _answer_complex_list(theirs["reflection_product"], "reflection_product")
        their_pp = _answer_complex_list(theirs["path_pressure"], "path_pressure")
    except ValueError as exc:
        one.append(f"答案檔振幅欄讀不進：{exc}")
        return 0.0, 0.0, None
    if len(their_refl) != len(frequencies) or len(their_pp) != len(frequencies):
        one.append(
            f"答案檔振幅頻帶數 {len(their_refl)}/{len(their_pp)} "
            f"不是 {len(frequencies)}"
        )
        return 0.0, 0.0, None
    amp_diffs, max_refl_frac, max_pp_frac, worst = _compare_amplitude(
        ours, their_refl, their_pp, frequencies
    )
    one.extend(amp_diffs)
    return max_refl_frac, max_pp_frac, worst


def _compare_one_path(
    ours: RoomPath,
    theirs: dict[str, object],
    position: int,
    frequencies: tuple[float, ...] | None,
) -> PathComparison:
    """比 v3 一條路徑跟答案檔一條：index、order、identity、牆名序列、hex、反彈、振幅。"""
    theirs_identity = _answer_identity(theirs)
    our_wall = wall_name_seq_from_identity(ours.identity)
    their_wall = wall_name_seq_from_identity(theirs_identity)

    one: list[str] = []
    if ours.index != position:
        one.append(f"index 是 {ours.index}，不等於位置 {position}")
    if ours.index != _answer_index(theirs):
        one.append(f"index 不同：v3={ours.index}，答案={_answer_index(theirs)}")
    if ours.order != _answer_order(theirs):
        one.append(f"order 不同：v3={ours.order}，答案={_answer_order(theirs)}")
    if ours.identity != theirs_identity:
        one.append(f"identity 不同：v3={list(ours.identity)!r}，答案={list(theirs_identity)!r}")
    if our_wall != their_wall:
        one.append(f"牆名序列不同：v3={our_wall!r}，答案={their_wall!r}")
    if len(ours.bounces) != ours.order:
        one.append(f"反彈數 {len(ours.bounces)} 不等於 order {ours.order}")

    our_image = tuple(v.hex() for v in ours.image)
    their_image, their_dist, their_delay = _answer_hexes(theirs)
    if their_image is not None:
        for axis_index, axis_label in enumerate(("x", "y", "z")):
            if our_image[axis_index] != their_image[axis_index]:
                one.append(
                    f"鏡像 {axis_label} hex 不同：v3={our_image[axis_index]!r}，"
                    f"答案={their_image[axis_index]!r}"
                )

    if ours.dist_m.hex() != their_dist:
        one.append(f"dist_m.hex 不同：v3={ours.dist_m.hex()!r}，答案={their_dist!r}")
    if ours.delay_s.hex() != their_delay:
        one.append(f"delay_s.hex 不同：v3={ours.delay_s.hex()!r}，答案={their_delay!r}")
    for bounce in ours.bounces:
        if bounce.in_wall is False:
            one.append(f"反射點（{bounce.wall}）不在牆上")

    max_refl_frac, max_pp_frac, worst = _amplitude_comparison(
        ours, theirs, frequencies, one
    )
    return PathComparison(
        index=ours.index,
        wall_seq=our_wall,
        diffs=one,
        max_refl_frac=max_refl_frac,
        max_pp_frac=max_pp_frac,
        worst=worst,
    )


def compare_paths(
    paths: list[RoomPath],
    answer_paths: list[dict[str, object]],
    frequencies: tuple[float, ...] | None = None,
) -> list[PathComparison]:
    """逐條比 v3 算出來的路徑跟答案檔，回結構化結果（每一條一筆：index、牆名序列、差異清單）。

    每一條比：index 等於位置、order、identity、牆名序列（答案檔沒有牆名欄，兩邊 identity 各
    推一次 :func:`wall_name_seq_from_identity` 再比）、``image_xyz`` 三個 hex、``dist_m``
    hex、``delay_s`` hex，以及反彈數等於 order、每個反彈點的 ``in_wall`` 為 True；答案檔有
    振幅欄時依契約比反射乘積與壓力（我方沒振幅就報「我方沒有振幅可比」）。兩邊條數不同要
    逐條報「少了哪一條（牆名序列）」，不靠 zip 截斷。
    """
    diffs: list[PathComparison] = []
    wall_of = wall_name_seq_from_identity
    common = min(len(paths), len(answer_paths))

    for position in range(common):
        theirs = _mapping(answer_paths[position], "answer_paths 的一筆")
        diffs.append(_compare_one_path(paths[position], theirs, position, frequencies))

    # 條數不同：逐條報少了哪一條（牆名序列），不用 zip 截斷。
    if len(paths) > common:
        for ours in paths[common:]:
            diffs.append(
                PathComparison(
                    index=ours.index,
                    wall_seq=wall_of(ours.identity),
                    diffs=[f"答案檔少了這一條（{wall_of(ours.identity)}）"],
                )
            )
    elif len(answer_paths) > common:
        for position in range(common, len(answer_paths)):
            theirs = _mapping(answer_paths[position], "answer_paths 的一筆")
            name = wall_of(_answer_identity(theirs))
            diffs.append(
                PathComparison(
                    index=position,
                    wall_seq=name,
                    diffs=[f"v3 少了這一條（{name}）"],
                )
            )

    return diffs


# ── 命令列 ───────────────────────────────────────────────────────────────────


def _bounce_cell(bounce: Bounce) -> tuple[str, str, str]:
    """一個反彈格的顯示：牆名、反彈點 ``(x,y,z)``、``t`` 與在不在牆上。"""
    point = ", ".join(repr(v) for v in bounce.point)
    flag = "True" if bounce.in_wall else "False"
    return bounce.wall, f"({point})", f"t={bounce.t!r}, in_wall={flag}"


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
    return "\n".join(lines) + "\n"


def _load_answer(path: Path) -> list[dict[str, object]]:
    """讀一份答案檔，回它的 ``paths``（一層表清單）。"""
    with path.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _mapping(data, "答案檔")
    raw_paths = root.get("paths")
    if not isinstance(raw_paths, list):
        raise ValueError("答案檔沒有 paths 那一欄，或它不是一串東西")
    return [_mapping(entry, "答案檔 paths 的一筆") for entry in raw_paths]


def _print_compare(results: list[PathComparison], answer_has_amp: bool) -> None:
    """把逐條判決印到標準輸出，最後印一行總判決。

    每一行由那一條的**所有**差異格決定，並印出是哪一格不同；有振幅時每條另印自己的
    「用到界線幾成」（反射、壓力兩個數）。牆名寫成「牆次簽名 ``x0→xL``」——那是 identity
    推出來的「哪面牆幾次」多集（跟人看的表裡逐次反彈的時間順序 ``walls(時序)`` 是兩種東西）。
    總判決分反射、壓力兩個數；有超界時印出超出最多那一格與倍數。答案檔沒振幅欄時加註
    「答案檔無振幅，未比」。
    """
    total = len(results)
    different = sum(1 for r in results if r.diffs)
    max_refl_frac = max((r.max_refl_frac for r in results), default=0.0)
    max_pp_frac = max((r.max_pp_frac for r in results), default=0.0)
    worst = max(
        (r.worst for r in results if r.worst is not None),
        key=lambda w: w[4],
        default=None,
    )
    for r in results:
        if r.diffs:
            print(f"不同  index {r.index}（牆次簽名 {r.wall_seq}）  " + "；".join(r.diffs))
        else:
            suffix = ""
            if r.max_refl_frac > 0.0 or r.max_pp_frac > 0.0:
                suffix = (
                    f"（反射用到界線 {r.max_refl_frac * 100:.3f}%、"
                    f"壓力 {r.max_pp_frac * 100:.3f}%）"
                )
            print(f"相同  index {r.index}（牆次簽名 {r.wall_seq}）{suffix}")
    if different == 0:
        verdict = f"{total} 條全部相同"
        if not answer_has_amp:
            verdict += "；答案檔無振幅，未比"
        elif max_refl_frac > 0.0 or max_pp_frac > 0.0:
            verdict += (
                f"，振幅全部在契約內（反射最大用到 {max_refl_frac * 100:.3f}%、"
                f"壓力 {max_pp_frac * 100:.3f}%）"
            )
        print(verdict)
    else:
        verdict = f"{total} 條裡有 {different} 條不同"
        if worst is not None:
            kind, f_idx, diff, tol, frac = worst
            verdict += f"；超出最多：{kind}[{f_idx}] 差 {diff!r} = 界線 {tol!r} 的 {frac:.3f} 倍"
        print(verdict)


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
            if not args.compare.exists():
                print(f"讀不到答案檔：{args.compare}")
                return 2
            try:
                answers = _load_answer(args.compare)
            except (ValueError, OSError) as exc:
                print(f"答案檔讀不進或形狀不對：{exc}")
                return 2

            frequencies = inputs.materials.frequencies_hz if inputs.materials is not None else None
            results = compare_paths(paths, answers, frequencies)
            answer_has_amp = any(
                "reflection_product" in entry or "path_pressure" in entry for entry in answers
            )
            _print_compare(results, answer_has_amp)
            return 1 if any(r.diffs for r in results) else 0

        if args.json:
            print(json.dumps(paths_to_payload(paths), ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        freqs = inputs.materials.frequencies_hz if inputs.materials is not None else ()
        print(_human_table(paths, freqs), end="")
        return 0
    except Exception as exc:
        print(f"未預期錯誤：{exc!r}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
