"""鞋盒（shoebox）房間的純幾何零件：房、點、六面牆、鏡射、反射點。

**這一支只 import 標準庫**（``math``、``dataclasses``、``enum``），不碰 numpy、不碰 JAX
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

    兩條非牆軸的座標要落在開區間 ``(0, L)``：座標剛好等於 0 或那一軸的房長，代表反射點打
    在牆的邊線上（退化組態），不算真的落在那面牆的內部，回 ``False`` 交給呼叫端（
    :func:`expand_bounces`）去判成退化組態、丟 ``ValueError``。
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
    """一次反彈的展開結果：牆名、反彈點、線段參數 ``t``、反彈點是否落在牆矩形內。

    ``t`` 定義是 ``point = current + t * (image - current)``（current 是「目前鏡像」往
    「下一階鏡像」連線的起點）。``in_wall`` 照實量（``t`` 在 (0,1) 且兩條非牆軸的座標落在
    [0, L] 內）；撞在邊線外就照實記 ``False``，不丟、不調。
    """

    wall: str
    point: tuple[float, float, float]
    t: float
    in_wall: bool


def _pinned_point(
    current: Point,
    image: Point,
    t: float,
    wall: Wall,
    room: Room,
) -> Point:
    """線段 ``current + t*(image-current)`` 的交點，把牆軸座標直接釘回平面值（0 或 L）。

    跟一階 :func:`reflect_point` 同一招：``point[axis] = plane``，消掉浮點尾差——下一階反彈
    從這個釘好的點出發，不從尾差飄掉的座標出發。
    """
    axis = wall.axis()
    plane = wall.plane(room)
    values = [
        current.x + t * (image.x - current.x),
        current.y + t * (image.y - current.y),
        current.z + t * (image.z - current.z),
    ]
    values[axis] = plane
    return Point(values[0], values[1], values[2])


def expand_bounces(
    room: Room,
    identity: Identity,
    source: Point,
    receiver: Point,
) -> tuple[Bounce, ...]:
    """把一個 identity 的逐次反彈展開成牆名與反彈點，重複 ``order_of(identity)`` 次。

    **做法**（鏡像聲源「攤開」直線）：從接收點往鏡像點連線，依序跟「線段參數 ``t`` 最小且
    ``t`` 在 (0,1)」的那面牆求交——那是離接收點最遠那一階的反彈點。接著把「目前的鏡像」
    再對這面牆鏡回房內（變成少一階的鏡像），從交點繼續往新鏡像連線、找下一面牆，重複
    ``order_of(identity)`` 次，剝到最後鏡像就是聲源本尊。

    **退化組態不靜靜少算。** 三種退化一律丟 ``ValueError``（不是 ``break``，也不「記個
    ``in_wall=False`` 就算了」）：① 找不到 ``t`` 在 (0,1) 的牆——線段沒穿過任何一面牆；②
    兩面牆的 ``t`` 相等——反彈點打在兩面牆交界的邊或角上；③ 反彈點的非牆軸座標剛好等於
    0 或房長——:func:`in_wall` 用開區間 ``(0, L)`` 現算回 ``False``。後兩者正是「不打在牆
    內、打在牆邊」的同一種退化，訊息一致，並附 identity 與那一點。直達 identity 回空 tuple。
    """
    total = order_of(identity)
    image = image_from_identity(room, identity, source)
    current = receiver
    bounces: list[Bounce] = []

    def _raise_degenerate(point: Point | None) -> None:
        raise ValueError(
            "反彈點打在牆的邊上，是退化組態，上一代也明說不支援"
            f"（identity={identity!r}，反彈點={point.as_tuple() if point is not None else None!r}）"
        )

    for _ in range(total):
        candidates: list[tuple[float, Wall]] = []
        for wall in Wall.all():
            axis = wall.axis()
            denom = image.as_tuple()[axis] - current.as_tuple()[axis]
            if denom == 0.0:
                continue
            t = (wall.plane(room) - current.as_tuple()[axis]) / denom
            if 0.0 < t < 1.0:
                candidates.append((t, wall))
        if not candidates:
            _raise_degenerate(None)
        candidates.sort(key=lambda pair: pair[0])
        t, wall = candidates[0]
        tied = [w for (tt, w) in candidates if tt == t]
        if len(tied) > 1:
            _raise_degenerate(_pinned_point(current, image, t, wall, room))
        point = _pinned_point(current, image, t, wall, room)
        refl = ReflectionPoint(point=point, t=t)
        if not in_wall(room, wall, refl):
            _raise_degenerate(point)
        bounces.append(
            Bounce(
                wall=wall.wall_name(),
                point=point.as_tuple(),
                t=t,
                in_wall=True,
            )
        )
        image = mirror_point(room, wall, image)
        current = point
    return tuple(bounces)
