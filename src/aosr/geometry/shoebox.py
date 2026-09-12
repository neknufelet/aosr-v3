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
    """反射點落在牆面矩形內、且 ``t`` 在開區間 (0,1)。精確比對，不留容差。"""
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
        if not (0.0 <= value <= room.length(k)):
            return False
    return True
