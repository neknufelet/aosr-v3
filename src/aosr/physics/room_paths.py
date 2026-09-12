"""鞋盒房間的鏡像聲源路徑：讀輸入、算七條路徑（直達＋六面牆各一次反射）、列印與比對。

**這一支做三件事。** ``load_room_input`` 讀一份 JSON 輸入檔（房間、聲源、接收點、聲速、
最大反射階數），``image_source_paths`` 用鏡像法（image source method）算直達加六面牆各一次
反射共七條路徑，``main`` 是命令列入口。

**怎麼跑**（``src/`` 還沒裝進 uv 的環境——pyproject 的 ``package=false``，另有票處理）::

    PYTHONPATH=src uv run python -m aosr.physics.room_paths <input.json>

**數值契約（逐位一致）。** 距離一律 :func:`aosr.geometry.shoebox.distance`
（``math.sqrt(dx*dx + dy*dy + dz*dz)``），到達時間＝距離／聲速，鏡像＝``2*plane - coord``
（plane 是 0 或那一軸的房長）。這些都是 ``float`` 的純標準庫算術，``float.hex()`` 必須跟
``blueprint/reference_room_answers.json`` 逐字相等（決策紙 precision-contract）。

**輸出走 ``print``。** 這一支的命令列標準輸出就是它的產品（人看的表、``--json`` 機器格式、
``--compare`` 逐條判決），規矩卡 ``style-guard`` 的輸出層白名單已經把這個檔列進去。

**身份（identity）的定義。** 六元組 ``(nx, sx, ny, sy, nz, sz)``，跟答案檔一致：每軸一組
（反射次數、正負號）。直達 path 是 ``(0,1,0,1,0,1)``；一次反射是恰好一軸 ``s=-1``，
``n=0`` 是那一軸的 zero 面（x0/y0/floor）、``n=1`` 是 L 面（xL/yL/ceiling）。階數由
identity 獨立算（直達 0、一次反射 1）。路徑順序照答案檔的 ``sorted((order, identity))``。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aosr.geometry.shoebox import (
    Point,
    Room,
    Wall,
    distance,
    in_wall,
    mirror_point,
    reflect_point,
)

# 一次反射（max_order=1）唯一的合法階數。超過這一階還沒寫，見 image_source_paths。
SUPPORTED_MAX_ORDER: Final[int] = 1

# 直達路徑的 identity：全 0、全 +1。
_DIRECT_IDENTITY: Final[tuple[int, int, int, int, int, int]] = (0, 1, 0, 1, 0, 1)


@dataclass(frozen=True)
class RoomInput:
    """一份輸入檔解析後的結果：房、聲源、接收點、聲速、最大反射階數。"""

    room: Room
    source: Point
    receiver: Point
    sound_speed: float
    max_order: int


@dataclass(frozen=True)
class RoomPath:
    """一條路徑（直達或一次反射）。

    ``identity`` 是答案檔那個六元組；``image`` 是（鏡像）聲源座標；``dist_m`` 是鏡像到
    接收點的直線距離；``delay_s`` 是距離／聲速；``refl_pt``／``t`` 是一次反射的交點與
    線段參數（直達路徑兩者皆 ``None``）；``in_wall`` 是反射點是否落在牆面矩形內且
    ``t`` 在 (0,1)（直達路徑是 ``None``）。
    """

    index: int
    order: int
    identity: tuple[int, int, int, int, int, int]
    wall: str
    image: tuple[float, float, float]
    dist_m: float
    delay_s: float
    refl_pt: tuple[float, float, float] | None
    t: float | None
    in_wall: bool | None


@dataclass(frozen=True)
class PathComparison:
    """``--compare`` 對一條路徑的判決：index、牆名、差異清單（空＝完全相同）。

    判決行從這個結構印，不用字串 startswith 去撈「哪一格不同」。
    """

    index: int
    wall: str
    diffs: list[str]


def _identity_for_wall(wall: Wall) -> tuple[int, int, int, int, int, int]:
    """一面牆對應的一次反射 identity：該軸 ``s=-1``、``n`` 依 zero/L 取 0/1，其餘軸 0、+1。"""
    axis = wall.axis()
    n = [0, 0, 0]
    s = [1, 1, 1]
    if wall.kind() != "zero":
        n[axis] = 1
    s[axis] = -1
    return (n[0], s[0], n[1], s[1], n[2], s[2])


def wall_name_from_identity(identity: tuple[int, int, int, int, int, int]) -> str | None:
    """identity 六元組 → 牆名（或 ``"direct"``）。看不懂（非 order 0/1）回 ``None``。

    六元組 ``(nx, sx, ny, sy, nz, sz)``：每軸一組（反射次數、正負號）。order 0＝全 0、
    全 +1 → ``"direct"``；order 1＝恰好一軸 ``s=-1``，``n=0`` 是該軸 zero 面
    （x0/y0/floor）、``n=1`` 是 L 面（xL/yL/ceiling）。
    """
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])
    if all(x == 0 for x in n) and all(x == 1 for x in s):
        return "direct"
    neg = [i for i, sign in enumerate(s) if sign == -1]
    if len(neg) != 1:
        return None
    axis = neg[0]
    kind = "zero" if n[axis] == 0 else "L"
    if kind == "zero":
        return ("x0", "y0", "floor")[axis]
    return ("xL", "yL", "ceiling")[axis]


def _direct_path(source: Point, receiver: Point, c: float) -> RoomPath:
    """直達路徑（order 0）：沒有反射點、``in_wall=None``。"""
    source_xyz = source.as_tuple()
    direct_dist = distance(source, receiver)
    return RoomPath(
        index=-1,
        order=0,
        identity=_DIRECT_IDENTITY,
        wall="direct",
        image=source_xyz,
        dist_m=direct_dist,
        delay_s=direct_dist / c,
        refl_pt=None,
        t=None,
        in_wall=None,
    )


def _one_wall_path(room: Room, wall: Wall, source: Point, receiver: Point, c: float) -> RoomPath:
    """對一面牆的一次反射路徑（order 1），``in_wall`` 由 in_wall() 現算。"""
    image = mirror_point(room, wall, source)
    image_xyz = image.as_tuple()
    image_point = Point(image_xyz[0], image_xyz[1], image_xyz[2])
    dist = distance(image_point, receiver)
    refl = reflect_point(room, wall, image_point, receiver)
    return RoomPath(
        index=-1,
        order=1,
        identity=_identity_for_wall(wall),
        wall=wall.wall_name(),
        image=image_xyz,
        dist_m=dist,
        delay_s=dist / c,
        refl_pt=refl.point.as_tuple() if refl.point is not None else None,
        t=refl.t,
        in_wall=in_wall(room, wall, refl),
    )


def image_source_paths(
    room: Room,
    source: Point,
    receiver: Point,
    c: float,
    max_order: int = 1,
) -> list[RoomPath]:
    """算直達＋六面牆各一次反射共七條路徑，順序 ``sorted((order, identity))``。

    ``max_order`` 不是 1 就丟 ``NotImplementedError``（明列尚未支援的階數）。每一條一次
    反射路徑都對它的牆跑一次 ``in_wall``（``t`` 在 (0,1) 且反射點落在牆面矩形內），結果
    記進 ``RoomPath.in_wall``；不在牆上的路徑不丟掉，只把它標成 ``False``。
    """
    if max_order != SUPPORTED_MAX_ORDER:
        raise NotImplementedError(
            f"尚未支援 max_order={max_order}：目前只寫了 direct（0）加六面牆一次反射（1），"
            f"max_order={max_order} 的多次反射還沒有實現"
        )

    paths = [_direct_path(source, receiver, c)]
    paths.extend(_one_wall_path(room, wall, source, receiver, c) for wall in Wall.all())

    ordered = sorted(paths, key=lambda p: (p.order, p.identity))
    numbered = [
        RoomPath(
            index=i,
            order=p.order,
            identity=p.identity,
            wall=p.wall,
            image=p.image,
            dist_m=p.dist_m,
            delay_s=p.delay_s,
            refl_pt=p.refl_pt,
            t=p.t,
            in_wall=p.in_wall,
        )
        for i, p in enumerate(ordered)
    ]
    return numbered


# ── 載入輸入檔 ──────────────────────────────────────────────────────────────

# 輸入檔頂層欄位（跟答案檔 ``parameters`` 的幾何/聲學欄位同一套，不含 units/convention/
# grid_shapes 這種 donor 自己的中介資料）。
_INPUT_KEYS: Final[tuple[str, ...]] = (
    "room",
    "source_xyz_m",
    "receiver_xyz_m",
    "sound_speed_m_s",
    "max_order",
)
_ROOM_KEYS: Final[tuple[str, ...]] = ("Lx_m", "Ly_m", "Lz_m")
_XYZ_KEYS: Final[tuple[str, ...]] = ("x", "y", "z")


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


def load_room_input(path: Path) -> RoomInput:
    """讀一份 JSON 輸入檔，回 :class:`RoomInput`。

    欄位：``room.Lx_m/Ly_m/Lz_m``（三邊嚴格大於 0）、``source_xyz_m.x/y/z``、
    ``receiver_xyz_m.x/y/z``、``sound_speed_m_s``（嚴格大於 0）、``max_order``（整數）。
    缺欄位的訊息講「缺欄位」，多餘欄位的訊息列出多的欄位名（輸入檔寫錯字才抓得到），
    0 或負的房間三邊／聲速講哪一格。``path`` 必填。
    """
    with path.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _mapping(data, "輸入檔")

    for key in _INPUT_KEYS:
        if key not in root:
            raise ValueError(f"輸入檔缺欄位：{key}")
    _reject_extra(root, _INPUT_KEYS, "輸入檔")

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
    )


# ── JSON 序列化與比對（跟答案檔 paths 同形）──────────────────────────────────


def _dec_hex(value: float) -> dict[str, str]:
    """一個浮點同時存十進位與十六進位兩格（跟答案檔的 ``dec``／``hex`` 同形）。"""
    return {"dec": repr(value), "hex": value.hex()}


def path_to_dict(path: RoomPath) -> dict[str, object]:
    """一條路徑攤成答案檔 ``paths`` 那一筆的形狀（含 dec 與 hex）。"""
    return {
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
    }


def paths_to_payload(paths: list[RoomPath]) -> dict[str, object]:
    """七條路徑整包攤成 ``{"paths": [...]}``。"""
    return {"paths": [path_to_dict(p) for p in paths]}


def _hex_cell(node: object, where: str) -> str:
    """答案檔裡 ``dist_m``／``delay_s``／``image_xyz.<axis>`` 的 hex 字串。"""
    cell = _mapping(node, where)
    hexed = cell.get("hex")
    if not isinstance(hexed, str):
        raise ValueError(f"{where} 沒有 hex 那一格")
    return hexed


def _answer_hexes(path: dict[str, object]) -> tuple[tuple[str, str, str], str, str]:
    """一條答案檔 path → ((img_x_hex, img_y_hex, img_z_hex), dist_hex, delay_hex)。"""
    image = _mapping(path.get("image_xyz"), "paths.image_xyz")
    return (
        (
            _hex_cell(image.get("x"), "image_xyz.x"),
            _hex_cell(image.get("y"), "image_xyz.y"),
            _hex_cell(image.get("z"), "image_xyz.z"),
        ),
        _hex_cell(path.get("dist_m"), "dist_m"),
        _hex_cell(path.get("delay_s"), "delay_s"),
    )


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


def compare_paths(paths: list[RoomPath], answer_paths: list[dict[str, object]]) -> list[PathComparison]:
    """逐條比 v3 算出來的路徑跟答案檔，回結構化結果（每一條一筆：index、牆名、差異清單）。

    每一條比：index 等於位置、order、identity、牆名（答案檔沒有牆名欄，用兩邊 identity 各推
    一次牆名再比）、``image_xyz`` 三個 hex、``dist_m`` hex、``delay_s`` hex，以及反射點
    ``in_wall`` 是否為 True。兩邊條數不同要逐條報「少了哪一條（牆名）」，不靠 zip 截斷。
    """
    diffs: list[PathComparison] = []
    wall_of = wall_name_from_identity
    common = min(len(paths), len(answer_paths))

    for position in range(common):
        ours = paths[position]
        theirs = _mapping(answer_paths[position], "answer_paths 的一筆")
        theirs_identity = _answer_identity(theirs)
        our_wall = wall_of(ours.identity)
        their_wall = wall_of(theirs_identity)

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
            one.append(f"牆名不同：v3={our_wall!r}，答案={their_wall!r}")

        our_image = tuple(v.hex() for v in ours.image)
        their_image, their_dist, their_delay = _answer_hexes(theirs)
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
        if ours.in_wall is False:
            one.append("反射點不在牆上")

        diffs.append(PathComparison(index=ours.index, wall=ours.wall, diffs=one))

    # 條數不同：逐條報少了哪一條（牆名），不用 zip 截斷。
    if len(paths) > common:
        for ours in paths[common:]:
            diffs.append(
                PathComparison(
                    index=ours.index,
                    wall=ours.wall,
                    diffs=[f"答案檔少了這一條（{ours.wall}）"],
                )
            )
    elif len(answer_paths) > common:
        for position in range(common, len(answer_paths)):
            theirs = _mapping(answer_paths[position], "answer_paths 的一筆")
            name = wall_of(_answer_identity(theirs))
            label = name if name is not None else f"identity {_answer_identity(theirs)!r}"
            diffs.append(
                PathComparison(
                    index=position,
                    wall=label,
                    diffs=[f"v3 少了這一條（{label}）"],
                )
            )

    return diffs


# ── 命令列 ───────────────────────────────────────────────────────────────────


def _human_table(paths: list[RoomPath]) -> str:
    """一張人看的表：index、wall、order、鏡像三格、反射點三格、t、在不在牆上、dist、delay。"""
    lines = [
        f"{'index':>5} {'wall':<8} {'order':>5}  "
        f"{'img_x':>20} {'img_y':>20} {'img_z':>20}  "
        f"{'refl_x':>20} {'refl_y':>20} {'refl_z':>20}  {'t':>11} {'in_wall':>7}  "
        f"{'dist_m':>20} {'delay_s':>20}"
    ]
    for p in paths:
        rx, ry, rz = (("—", "—", "—") if p.refl_pt is None else tuple(f"{v!r}" for v in p.refl_pt))
        t = "—" if p.t is None else f"{p.t!r}"
        inw = "—" if p.in_wall is None else ("True" if p.in_wall else "False")
        lines.append(
            f"{p.index:>5} {p.wall:<8} {p.order:>5}  "
            f"{p.image[0]!r:>20} {p.image[1]!r:>20} {p.image[2]!r:>20}  "
            f"{rx:>20} {ry:>20} {rz:>20}  {t:>11} {inw:>7}  "
            f"{p.dist_m!r:>20} {p.delay_s!r:>20}"
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


def _print_compare(results: list[PathComparison]) -> None:
    """把逐條判決印到標準輸出：每一行由那一條的**所有**差異格決定，並印出是哪一格不同。"""
    for r in results:
        if r.diffs:
            print(f"不同  index {r.index}（{r.wall}）  " + "；".join(r.diffs))
        else:
            print(f"相同  index {r.index}（{r.wall}）")


def main(argv: list[str]) -> int:
    """命令列入口：算七條路徑，印人看的表；``--json`` 印機器格式；``--compare`` 逐條比。

    回傳離開碼：``--compare`` 全同回 0、有不同回 1、讀不到檔回 2；程式自己炸掉（未預期
    例外）也要回 2，不准回 1。
    """
    parser = argparse.ArgumentParser(description="鞋盒房間的鏡像聲源路徑（直達＋六面牆一次反射）")
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
                inputs.room, inputs.source, inputs.receiver, inputs.sound_speed, inputs.max_order
            )
        except NotImplementedError as exc:
            print(f"這一階還沒寫：{exc}")
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

            results = compare_paths(paths, answers)
            _print_compare(results)
            return 1 if any(r.diffs for r in results) else 0

        if args.json:
            print(json.dumps(paths_to_payload(paths), ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        print(_human_table(paths), end="")
        return 0
    except Exception as exc:
        print(f"未預期錯誤：{exc!r}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
