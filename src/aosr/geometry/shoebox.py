"""鞋盒（shoebox）房間的純幾何零件：房、點、六面牆、鏡射、反射點。

**這一支只 import 標準庫**（``math``、``dataclasses``、``enum``、``typing``），不碰 numpy、不碰 JAX
（規矩卡 ``layers-import-downward-only`` 只讓最底層 ``runtime`` 動全域設定，而這一題的
數值契約只要 ``float`` 的 ``math.sqrt`` 就夠用）。它住在 ``geometry`` 那一層，別的層
可以來拿它，它自己不去拿任何上層。

**數值契約。** 這一支的每一下算術都要跟 ``blueprint/reference_room_answers.json`` 的
``float.hex()`` 逐字相等（決策紙 precision-contract）：距離用
``math.sqrt(dx*dx + dy*dy + dz*dz)`` 照這個順序；鏡射用 ``2*plane - coord``（plane 是
0 或那一軸的房間長）；到達時間由 ``physics`` 那一層拿距離除以聲速，這一支不碰聲速。

**牆的規範順序。** 六面牆照 v2 的 canonical 順序：floor（z=0）、ceiling（z=Lz）、x0、xL、
y0、yL。``Wall`` 枚舉照這個順序宣告，``Wall.all()`` 回同一順序。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

# 牆序列的方向只在這裡寫一次（票 #475）：:func:`expand_bounces` 從接收點往鏡像點找交點，
# 所以反彈與由它排出的牆序列是從接收點那一側往回列。報表路徑表、反射評估、反射左右差的
# 欄位說明都拿這一句，不各寫一份。
WALL_SEQUENCE_ORDER: Final[str] = (
    "從接收點那一側往回列：第一面是最後碰到的牆，最後一面是聲源發出後最先碰到的牆；"
    "同一個反彈點同時碰到的幾面（交線、角落）相鄰並列、照牆的固定排列，彼此沒有先後"
)


@dataclass(frozen=True)
class Point:
    """三維空間裡的一個點（frozen，可當字典鍵）。"""

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        """把這個點攤成 (x, y, z) 三元組，方便跟答案檔/獨立幾何互餵。"""
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Room:
    """鞋盒房間：三條邊的長度（m）。聲速不屬於房，由 ``physics`` 層另外管。"""

    Lx: float
    Ly: float
    Lz: float

    def length(self, axis: int) -> float:
        """軸 index（0=x、1=y、2=z）那一軸的房間長度。"""
        return (self.Lx, self.Ly, self.Lz)[axis]


class Wall(Enum):
    """鞋盒房間的六面牆。值 ``(axis, kind)``：哪一軸、那一軸的 0 面或 L 面。

    宣告順序就是 v2 的 canonical 順序：floor、ceiling、x0、xL、y0、yL。
    """

    FLOOR = (2, "zero")
    CEILING = (2, "L")
    X0 = (0, "zero")
    XL = (0, "L")
    Y0 = (1, "zero")
    YL = (1, "L")

    def axis(self) -> int:
        """這面牆垂直於哪一軸（0=x、1=y、2=z）。"""
        return self.value[0]

    def kind(self) -> str:
        """這面牆是那一軸的 0 面（"zero"）還是 L 面（"L"）。"""
        return self.value[1]

    def plane(self, room: Room) -> float:
        """這面牆住的那個座標平面：0 面是 0.0，L 面是那一軸的房長。"""
        if self.kind() == "zero":
            return 0.0
        return room.length(self.axis())

    def wall_name(self) -> str:
        """牆名（跟答案檔/獨立幾何的牆名一致）：floor/ceiling/x0/xL/y0/yL。"""
        return {
            Wall.FLOOR: "floor",
            Wall.CEILING: "ceiling",
            Wall.X0: "x0",
            Wall.XL: "xL",
            Wall.Y0: "y0",
            Wall.YL: "yL",
        }[self]

    @classmethod
    def all(cls) -> tuple["Wall", ...]:
        """六面牆照 canonical 順序。"""
        return (cls.FLOOR, cls.CEILING, cls.X0, cls.XL, cls.Y0, cls.YL)

    @classmethod
    def wall_names(cls) -> tuple[str, ...]:
        """六面牆名照 canonical 順序（跟 ``CANONICAL_WALL_ORDER`` 一致）。"""
        return tuple(wall.wall_name() for wall in cls.all())

    @classmethod
    def from_name(cls, name: str) -> "Wall":
        """牆名 → 那面牆。不認識就當場炸（只認六面牆）。"""
        for wall in cls.all():
            if wall.wall_name() == name:
                return wall
        raise ValueError(f"未知牆名：{name!r}")


@dataclass(frozen=True)
class ReflectionPoint:
    """一次反射的交點：線段參數 ``t``。

    ``t`` 的定義是 ``point = image + t * (target - image)``；線段跟牆面平行（分母為 0）
    的時候沒有有限且確定的交點，``point`` 是 ``None``、``t`` 是 ``None``。
    直達路徑（order 0）沒有反射點，整個用 ``None`` 表示。
    """

    point: Point | None
    t: float | None


def mirror_point(room: Room, wall: Wall, point: Point) -> Point:
    """把一個點對 ``wall`` 鏡射到牆的對面，回鏡像座標。

    數值契約：``image = 2*plane - coord``，plane 是 0 或那一軸的房長（照這個順序寫，
    才能跟 ``float.hex()`` 逐字一致）。
    """
    plane = wall.plane(room)
    axis = wall.axis()
    coord = (point.x, point.y, point.z)[axis]
    mirrored = 2.0 * plane - coord
    values = [point.x, point.y, point.z]
    values[axis] = mirrored
    return Point(values[0], values[1], values[2])


def distance(a: Point, b: Point) -> float:
    """兩點歐氏距離，``math.sqrt(dx*dx + dy*dy + dz*dz)`` 照這個順序算（數值契約）。"""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def reflect_point(
    room: Room,
    wall: Wall,
    image: Point,
    target: Point,
) -> ReflectionPoint:
    """鏡像→目標的線段跟 ``wall`` 平面的交點，以及線段參數 ``t``。

    ``t`` 定義是 ``point = image + t * (target - image)``。線段跟牆面平行（分母為 0）
    時沒有有限且確定的交點，回 ``ReflectionPoint(None, None)``。
    """
    axis = wall.axis()
    plane = wall.plane(room)
    image_axis = (image.x, image.y, image.z)[axis]
    target_axis = (target.x, target.y, target.z)[axis]
    denom = target_axis - image_axis
    if denom == 0.0:
        return ReflectionPoint(point=None, t=None)
    t = (plane - image_axis) / denom
    point = [
        (image.x, image.y, image.z)[k] + t * ((target.x, target.y, target.z)[k] - (image.x, image.y, image.z)[k])
        for k in range(3)
    ]
    point[axis] = plane  # 直接釘在平面上，消掉浮點尾差
    refl = Point(point[0], point[1], point[2])
    return ReflectionPoint(point=refl, t=t)


def in_wall(room: Room, wall: Wall, refl: ReflectionPoint) -> bool:
    """反射點落在牆面矩形**內部**、且 ``t`` 在開區間 (0,1)。精確比對，不留容差。

    這支保留嚴格「單面牆內部」判斷；座標剛好等於 0 或房長時回 ``False``。需要接納交線或
    角點的逐次展開由 :func:`expand_bounces` 同時收集所有相交牆面，不靠這支把邊界算進單面。
    """
    if refl.point is None or refl.t is None:
        return False
    if not (0.0 < refl.t < 1.0):
        return False
    axis = wall.axis()
    values = (refl.point.x, refl.point.y, refl.point.z)
    for k in range(3):
        if k == axis:
            continue
        value = values[k]
        if not (0.0 < value < room.length(k)):
            return False
    return True


# ── 二階以上（票 #187）：identity、鏡像座標、逐次反彈 ─────────────────────────
#
# 這段跟 ``blueprint/reference_room_geometry.py`` 的 ``image_from_identity``／
# ``order_of``／``enumerate_identities``／``expand_bounces`` 是**同一條規則**，考卷會比
# 兩邊的 identity 集合與反彈結果；``src/`` 不准 import ``blueprint``（規矩卡
# layers-import-downward-only），所以這裡自己實作一份，一個字都不照抄那份檔，
# 只照同一條數值契約：鏡像 ``2*n*L + s*src``（每軸）、距離 ``math.sqrt(dx*dx+dy*dy+dz*dz)``。

Identity = tuple[int, int, int, int, int, int]


def _identity_terms(identity: Identity) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """identity 六元組 → 三軸各自的 ``(n, sign)``（x、y、z 的順序）。"""
    return (
        (identity[0], identity[1]),
        (identity[2], identity[3]),
        (identity[4], identity[5]),
    )


def axis_order(n: int, sign: int) -> int:
    """一軸的反彈次數：``sign=+1`` 是 ``2*|n|``，``sign=-1`` 是 ``|2n-1|``。

    跟 ``blueprint/reference_room_geometry.py`` 的 ``_axis_order`` 同一條：``sign=+1`` 時
    來回各一次（兩面各 ``|n|`` 次），``sign=-1`` 時奇數次（net 一次翻面）。
    """
    if sign == 1:
        return 2 * abs(n)
    return abs(2 * n - 1)


def order_of(identity: Identity) -> int:
    """從 identity 六元組獨立算出的階數（三軸反彈次數相加），直達 0。"""
    return sum(axis_order(n, sign) for n, sign in _identity_terms(identity))


def image_from_identity(room: Room, identity: Identity, source: Point) -> Point:
    """由 identity 六元組直接算鏡像座標，任何階適用，**不逐牆鏡射**。

    公式每軸獨立：``image = 2*n*L + s*src``（L 是那一軸房長、src 是聲源那一軸）。
    ``2.0*n*L + s*src`` 照這個順序寫，才跟答案檔的 hex 逐字一致（直達＝全 0 全 +1＝
    聲源本尊）。刻意不用一次一次的 :func:`mirror_point` 鏈：鏡回再鏡出去在浮點上可能跟
    直接乘不是同一位元。
    """
    terms = _identity_terms(identity)
    return Point(
        x=2.0 * terms[0][0] * room.length(0) + terms[0][1] * source.x,
        y=2.0 * terms[1][0] * room.length(1) + terms[1][1] * source.y,
        z=2.0 * terms[2][0] * room.length(2) + terms[2][1] * source.z,
    )


def _axis_terms(max_order: int) -> tuple[tuple[int, int], ...]:
    """一軸的 ``(n, sign)`` 字典，照 donor 的去重規則。

    逐 ``n`` 從 ``-max_order`` 到 ``max_order``、逐 ``sign`` 跑 ``(1, -1)``，只收
    ``axis_order(n, sign) <= max_order`` 的那幾組（超過的組合總階數必然超標，先剪掉）。
    """
    terms: list[tuple[int, int]] = []
    for n in range(-max_order, max_order + 1):
        for sign in (1, -1):
            if axis_order(n, sign) <= max_order:
                terms.append((n, sign))
    return tuple(terms)


def wall_count_signature(identity: Identity) -> dict[str, int]:
    """identity 六元組 → 每面牆反彈幾次的簽名（多重集，**不看時間順序**）。

    純從 identity 的「每軸 ``(n, sign)``」算（跟 ``blueprint/reference_room_geometry.py`` 與
    ``blueprint/reference_amplitude_check.py`` 的 ``wall_count_signature`` 同一條拆法）：
    ``sign=+1`` 時該軸兩面牆各 ``|n|`` 次；``sign=-1`` 時該軸總共 ``|2n-1|`` 次、依 ``n`` 正負
    拆給 zero 面（floor/x0/y0）與 L 面（ceiling/xL/yL）。牆名照 canonical 順序。這份簽名是
    :mod:`aosr.physics.amplitude` 算反射乘積時「牆打幾次乘幾次」的輸入。
    """
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])

    def _lo_hi(nx: int, sign: int) -> tuple[int, int]:
        """一軸的 (zero 面次數, L 面次數)，照 donor ``_axis_wall_counts``。"""
        if sign == 1:
            return abs(nx), abs(nx)
        m = 2 * nx - 1
        order = abs(m)
        if m > 0:
            return order // 2, (order + 1) // 2
        return (order + 1) // 2, order // 2

    z_lo, z_hi = _lo_hi(n[2], s[2])
    x_lo, x_hi = _lo_hi(n[0], s[0])
    y_lo, y_hi = _lo_hi(n[1], s[1])
    return {
        "floor": z_lo,
        "ceiling": z_hi,
        "x0": x_lo,
        "xL": x_hi,
        "y0": y_lo,
        "yL": y_hi,
    }


def enumerate_identities(max_order: int) -> tuple[Identity, ...]:
    """枚舉 ``max_order`` 以內全部 identity 六元組，照 donor 的去重規則，回**排好序**的一串。

    三軸各自走 :func:`_axis_terms`（單軸先剪到 ``<= max_order``），三軸反彈次數加總（＝
    :func:`order_of`）超過 ``max_order`` 就丟。回傳照 ``(order, identity)`` 排序——跟答案檔
    ``paths`` 的 ``index`` 順序同一條，所以路徑可以照這個順序給 index。
    """
    ids: list[Identity] = []
    for nx, sx in _axis_terms(max_order):
        for ny, sy in _axis_terms(max_order):
            for nz, sz in _axis_terms(max_order):
                identity: Identity = (nx, sx, ny, sy, nz, sz)
                if order_of(identity) <= max_order:
                    ids.append(identity)
    return tuple(sorted(ids, key=lambda ident: (order_of(ident), ident)))


@dataclass(frozen=True)
class Bounce:
    """一個反彈點的展開結果：牆名們、反彈點與線段參數 ``t``。

    ``t`` 定義是 ``point = current + t * (image - current)``（current 是「目前鏡像」往
    「下一階鏡像」連線的起點）。一般點的 ``walls`` 只有一面；落在邊線或角點時，同一點
    只帶「同時命中且 identity 尚欠」的牆面，代表它一次消化同樣數量的反射。房外反彈點
    會在 :func:`expand_bounces` 建構本物件前直接拒絕，因此不保留永遠只會是 ``True`` 的欄位。
    ``wall`` 保留既有單牆讀法，回第一面。
    """

    walls: tuple[str, ...]
    point: tuple[float, float, float]
    t: float

    @property
    def wall(self) -> str:
        """既有單牆介面的相容讀法；交線點要讀 :attr:`walls` 才完整。"""
        return self.walls[0]


def wall_grid_cell(
    room: Room,
    wall: str,
    point: tuple[float, float, float],
    rows: int,
    cols: int,
) -> tuple[int, int]:
    """反彈點在牆面均勻格網裡的 ``(row, col)``，cells 依 row-major 攤平。

    牆面 ``u`` 軸是 column、``v`` 軸是 row：floor/ceiling 用 (x,y)，x0/xL
    用 (y,z)，y0/yL 用 (x,z)。沿用上一代的 ``int(frac * count)`` 再 clamp；因此
    點恰在內部格線時落到上方／右方那格，點在牆的最外緣則夾在最後一格。
    只支援均勻格；上一代的非均勻 ``wall_edges`` 不在這一段。
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"牆 {wall!r} 的 grid 必須是正整數，是 {rows}×{cols}")
    wall_axes = {
        "floor": (0, 1),
        "ceiling": (0, 1),
        "x0": (1, 2),
        "xL": (1, 2),
        "y0": (0, 2),
        "yL": (0, 2),
    }
    try:
        u_axis, v_axis = wall_axes[wall]
    except KeyError as exc:
        raise ValueError(f"未知牆名：{wall!r}") from exc
    col = min(max(int(point[u_axis] / room.length(u_axis) * cols), 0), cols - 1)
    row = min(max(int(point[v_axis] / room.length(v_axis) * rows), 0), rows - 1)
    return row, col


def _pinned_point(
    current: Point,
    image: Point,
    t: float,
    walls: tuple[Wall, ...],
    room: Room,
) -> Point:
    """線段 ``current + t*(image-current)`` 的交點，把所有命中牆軸直接釘回平面值。

    跟一階 :func:`reflect_point` 同一招：``point[axis] = plane``，消掉浮點尾差——下一階反彈
    從這個釘好的點出發，不從尾差飄掉的座標出發。
    """
    values = [
        current.x + t * (image.x - current.x),
        current.y + t * (image.y - current.y),
        current.z + t * (image.z - current.z),
    ]
    for wall in walls:
        values[wall.axis()] = wall.plane(room)
    return Point(values[0], values[1], values[2])


def _crossing_candidates(
    room: Room,
    current: Point,
    image: Point,
    remaining: dict[str, int],
) -> list[tuple[float, Wall]]:
    """找這一步 identity 尚欠、且線段會在開區間穿過的牆面。"""
    candidates: list[tuple[float, Wall]] = []
    for wall in Wall.all():
        if remaining[wall.wall_name()] == 0:
            continue
        axis = wall.axis()
        denom = image.as_tuple()[axis] - current.as_tuple()[axis]
        if denom == 0.0:
            continue
        t = (wall.plane(room) - current.as_tuple()[axis]) / denom
        if 0.0 < t < 1.0:
            candidates.append((t, wall))
    return sorted(candidates, key=lambda pair: pair[0])


def _first_crossing_walls(
    room: Room,
    current: Point,
    image: Point,
    candidates: list[tuple[float, Wall]],
) -> tuple[float, tuple[Wall, ...], Point]:
    """把最先命中的同點牆面收成一組，並把交點釘回所有牆平面。"""
    t, wall = candidates[0]
    first_point = _pinned_point(current, image, t, (wall,), room)
    first_values = first_point.as_tuple()
    touching = tuple(
        candidate_wall
        for candidate_t, candidate_wall in candidates
        if candidate_t == t
        or first_values[candidate_wall.axis()] == candidate_wall.plane(room)
    )
    return t, touching, _pinned_point(current, image, t, touching, room)


def _peel_image_layer(
    room: Room,
    image: Point,
    touching: tuple[Wall, ...],
    remaining: dict[str, int],
) -> Point:
    """把鏡像沿同點命中的每面牆各剝一層，並同步扣掉 identity 的牆面帳。"""
    for wall in touching:
        image = mirror_point(room, wall, image)
        remaining[wall.wall_name()] -= 1
    return image


def expand_bounces(
    room: Room,
    identity: Identity,
    source: Point,
    receiver: Point,
) -> tuple[Bounce, ...]:
    """把 identity 展開成反彈點；各點的牆面數加總為 ``order_of(identity)``。

    **做法**（鏡像聲源「攤開」直線）：從接收點往鏡像點連線，依序跟「線段參數 ``t`` 最小且
    ``t`` 在 (0,1)」的那面牆求交——那是離接收點最遠那一階的反彈點。接著把「目前的鏡像」
    再對這面牆鏡回房內（變成少一階的鏡像），從交點繼續往新鏡像連線、找下一面牆，重複
    到消化的牆面數等於 ``order_of(identity)``，剝到最後鏡像就是聲源本尊。所以回傳的反彈
    是從接收點往回排：第一個是聲音最後碰到的牆（見 :data:`WALL_SEQUENCE_ORDER`）。

    找不到 ``t`` 在 (0,1) 的待消化牆面時仍丟 ``ValueError``：線段真的沒有穿牆，算不出來。
    兩面或三面「同時命中且 identity 尚欠」的牆，其 ``t`` 相同或第一面交點精確落在另一面
    邊界時，把它們收成同一個 :class:`Bounce`，各鏡回一次、各扣一次 identity 的牆面帳。
    交點落在所選牆面範圍外時用另一種錯誤明說牆名與座標。直達回空 tuple。
    """
    total = order_of(identity)
    image = image_from_identity(room, identity, source)
    current = receiver
    bounces: list[Bounce] = []
    remaining = wall_count_signature(identity)
    consumed = 0

    def _raise_no_crossing() -> None:
        raise ValueError(
            "線段沒穿過任何一面牆，反彈路徑算不出來"
            f"（identity={identity!r}，已消化反射數={consumed}）"
        )

    def _raise_outside_wall(wall: Wall, point: Point) -> None:
        raise ValueError(
            "反彈點落在那面牆的範圍外，反彈路徑算不出來"
            f"（identity={identity!r}，牆={wall.wall_name()}，反彈點={point.as_tuple()!r}）"
        )

    while consumed < total:
        candidates = _crossing_candidates(room, current, image, remaining)
        if not candidates:
            _raise_no_crossing()
        t, touching, point = _first_crossing_walls(
            room, current, image, candidates
        )
        if any(
            coordinate < 0.0 or coordinate > room.length(axis)
            for axis, coordinate in enumerate(point.as_tuple())
        ):
            _raise_outside_wall(touching[0], point)
        if consumed + len(touching) > total:
            raise ValueError(
                "這個反彈點會讓反射階數超過 identity 的階數"
                f"（identity={identity!r}，identity 階數={total}，"
                f"已消化反射數={consumed}，本點牆面="
                f"{tuple(item.wall_name() for item in touching)!r}）"
            )
        bounces.append(
            Bounce(
                walls=tuple(touching_wall.wall_name() for touching_wall in touching),
                point=point.as_tuple(),
                t=t,
            )
        )
        image = _peel_image_layer(room, image, touching, remaining)
        consumed += len(touching)
        current = point
    return tuple(bounces)
