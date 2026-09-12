"""獨立幾何：只 import 標準庫的 shoebox（鞋盒狀房間）鏡像聲源重算。

**為什麼要有這一支。** 票 #175 的考卷（``tests/test_reference_room_answers.py``）要用一支
「跟產生器完全無關」的程式，把 ``blueprint/reference_room_answers.json`` 裡每一條路徑的
距離、延遲、鏡像座標、反射點重新算一遍，跟答案檔逐格字串相等比對。這一支就是那支獨立
程式：它**只 import 標準庫**（``math``、``dataclasses``），不讀檔、不印東西、純函式，
所以「答案檔的值」跟「驗它的算式」不共用任何一支會算錯就一起錯的程式（找碴那一層要獨立，
見 CLAUDE.md 七條習慣第 4 條）。

**不寫死任何房間尺寸或聲速。** 房間幾何（三軸長度、聲速）全部由呼叫端透過
:class:`Room` 餵進來，這一支一個數字都不自己定——同一組尺寸只住在產生器檔頭那一份
凍結常數，這裡要是也寫一份，會長成「改了產生器忘了改獨立幾何」的第二份副本。

**只算第一階反射。** 答案檔的合約是「直達 ＋ 六面牆各一次反射」共七條路徑（``max_order=1``）。
鏡像法：直達路的端點就是聲源本尊；對某面牆反射一次的路徑，把聲源對那面牆鏡射成鏡像、
量鏡像到接收點的直線距離與時間——那是同一個數值，但算式跟凍結的 donor 完全無關。

**牆名與階數怎麼推。** identity 是六元組 ``(nx, sx, ny, sy, nz, sz)``（每軸一組「反射次數、
正負號」）。order 0（全 0、全 +1）是直達；order 1 是恰好一軸 ``s=-1``（對該軸的 0 面牆
或 L 面牆）。階數由 :func:`order_of` 從 identity 獨立算：每軸 ``sign=+1`` 是 ``2*|n|``、
``sign=-1`` 是 ``|2n-1|``，三軸相加。0 面牆名 x0/y0/floor、L 面牆名 xL/yL/ceiling。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# 幾何處理的軸數：x/y/z 三軸。identity 每軸兩格（次數、正負號），
# 所以 identity 的長度 = 2 * 軸數（考卷裡 length 的期望值拿這個算，不寫死數字）。
NUM_AXES: Final[int] = 3


@dataclass(frozen=True)
class Room:
    """鞋盒房間與傳播介質的幾何參數（由呼叫端餵，這一支不自己定）。"""

    lx: float
    """x 軸房間長度（m）。"""
    ly: float
    """y 軸房間長度（m）。"""
    lz: float
    """z 軸房間長度（m）。"""
    c: float
    """聲速（m/s）。"""

    def length(self, axis: int) -> float:
        """軸 index（0=x、1=y、2=z）那一軸的房間長度。"""
        return (self.lx, self.ly, self.lz)[axis]


# 六面牆的規範順序，跟 v2 的 CANONICAL_WALL_ORDER 一致：
# floor（z=0）、ceiling（z=Lz）、x0、xL、y0、yL。`kind` 是 zero（該軸座標 = 0 的面）
# 或 L（該軸座標 = 房間長的面）。
_CANONICAL_WALLS: tuple[tuple[str, tuple[int, str]], ...] = (
    ("floor", (2, "zero")),
    ("ceiling", (2, "L")),
    ("x0", (0, "zero")),
    ("xL", (0, "L")),
    ("y0", (1, "zero")),
    ("yL", (1, "L")),
)


@dataclass(frozen=True)
class Reflection:
    """一階反射路徑的完整重算結果（直達路徑的 ``refl_pt``／``t`` 是 None）。"""

    wall: str
    image: tuple[float, float, float]
    dist_m: float
    delay_s: float
    refl_pt: tuple[float, float, float] | None
    t: float | None
    in_wall: bool | None


def image_of(room: Room, wall: str, src: tuple[float, float, float]) -> tuple[float, float, float]:
    """把聲源鏡射到 ``wall`` 對面，回傳鏡像座標（六面牆名之一）。"""
    axis, _kind = _wall_axis_kind(wall)
    value = wall_plane_value(room, wall)
    out = list(src)
    out[axis] = 2.0 * value - src[axis]
    return (out[0], out[1], out[2])


def wall_plane_value(room: Room, wall: str) -> float:
    """這面牆住在它那個軸的哪個座標平面上（zero 面是 0、L 面是那一軸的長）。"""
    axis, kind = _wall_axis_kind(wall)
    if kind == "zero":
        return 0.0
    return room.length(axis)


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """兩點的歐氏距離（標準庫 ``math.sqrt``，不碰 numpy）。"""
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def reflect_point(
    room: Room,
    wall: str,
    image: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[tuple[float, float, float] | None, float | None]:
    """鏡像→目標的線段跟 ``wall`` 平面的交點，以及線段參數 ``t``。

    ``t`` 的定義是 ``point = image + t * (target - image)``。線段跟牆面平行
    （分母為 0）時沒有有限且確定的交點，回傳 ``(None, None)``。
    """
    axis, _kind = _wall_axis_kind(wall)
    plane = wall_plane_value(room, wall)
    denom = target[axis] - image[axis]
    if denom == 0.0:
        return None, None
    t = (plane - image[axis]) / denom
    point = [
        image[k] + t * (target[k] - image[k]) for k in range(3)
    ]
    point[axis] = plane  # 直接釘在平面上，消掉浮點尾差
    return ((point[0], point[1], point[2]), t)


def edge_margin(room: Room, refl_pt: tuple[float, float, float], wall: str) -> float:
    """反射點到牆面四邊最近的垂直距離（兩條非牆軸的 0／L 四個方向取最小）。"""
    axis, _kind = _wall_axis_kind(wall)
    other = [k for k in range(3) if k != axis]
    margins: list[float] = []
    for k in other:
        length = room.length(k)
        value = refl_pt[k]
        margins.append(value)          # 到 0 邊
        margins.append(length - value)  # 到 L 邊
    return min(margins)


def in_wall(
    room: Room,
    wall: str,
    refl_pt: tuple[float, float, float] | None,
    t: float | None,
) -> bool:
    """反射點落在牆面矩形內、且 ``t`` 在開區間 (0,1)。精確比對，不留容差。

    反射點恰好碰在鏡像或接收點上（``t`` 在端點）不算真的過牆；兩條非牆軸
    的座標要落在閉區間 [0, L] 內。
    """
    if refl_pt is None or t is None:
        return False
    if not (0.0 < t < 1.0):
        return False
    axis, _kind = _wall_axis_kind(wall)
    for k in range(3):
        if k == axis:
            continue
        value = refl_pt[k]
        if not (0.0 <= value <= room.length(k)):
            return False
    return True


def decode_identity(identity: tuple[int, int, int, int, int, int]) -> str | None:
    """identity 六元組 → 牆名（或 ``"direct"``）。看不懂（非 order 0/1）回 ``None``。

    六元組 ``(nx, sx, ny, sy, nz, sz)``：每軸一組 (反射次數, 正負號)。
    order 0＝全 0、全 +1 → ``"direct"``；order 1＝恰好一軸 ``s=-1``，``n=0`` 是該軸
    zero 面（x0/y0/floor）、``n=1`` 是 L 面（xL/yL/ceiling）。
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


def _axis_order(n: int, sign: int) -> int:
    """一軸的反射次數：``sign=+1`` 是 ``2*|n|``，``sign=-1`` 是 ``|2n-1|``。

    這個定義跟 donor 的 ``_axis_wall_counts`` 推出的每個軸的總反射次數一致：
    ``sign=+1`` 時來回各一次（兩面各算 ``|n|``），``sign=-1`` 時起奇數次反射。
    """
    if sign == 1:
        return 2 * abs(n)
    return abs(2 * n - 1)


def order_of(identity: tuple[int, int, int, int, int, int]) -> int:
    """從 identity 六元組獨立算出的階數（反射總次數），直達 0、一次反射 1。

    六元組 ``(nx, sx, ny, sy, nz, sz)``：每軸一組 (反射次數, 正負號)。三軸的
    反射次數相加就是這一條路徑的階數。
    """
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])
    return _axis_order(n[0], s[0]) + _axis_order(n[1], s[1]) + _axis_order(n[2], s[2])


def direct_reflection(
    room: Room, src: tuple[float, float, float], recv: tuple[float, float, float]
) -> Reflection:
    """直達路徑（order 0）：距離＝聲源到接收點直線，沒有反射點。"""
    d = dist(src, recv)
    return Reflection(
        wall="direct",
        image=src,
        dist_m=d,
        delay_s=d / room.c,
        refl_pt=None,
        t=None,
        in_wall=None,
    )


def wall_reflection(
    room: Room,
    wall: str,
    src: tuple[float, float, float],
    recv: tuple[float, float, float],
) -> Reflection:
    """對 ``wall`` 的一次反射路徑（order 1）。"""
    image = image_of(room, wall, src)
    d = dist(image, recv)
    refl_pt, t = reflect_point(room, wall, image, recv)
    return Reflection(
        wall=wall,
        image=image,
        dist_m=d,
        delay_s=d / room.c,
        refl_pt=refl_pt,
        t=t,
        in_wall=in_wall(room, wall, refl_pt, t),
    )


def canonical_walls() -> tuple[str, ...]:
    """規範順序的六面牆名（跟 v2 的 ``CANONICAL_WALL_ORDER`` 一致）。"""
    return tuple(name for name, _ in _CANONICAL_WALLS)


def _wall_axis_kind(wall: str) -> tuple[int, str]:
    """牆名 → (軸 index, kind)。未知牆名當場炸（這一支只認六面牆）。"""
    for name, (axis, kind) in _CANONICAL_WALLS:
        if name == wall:
            return (axis, kind)
    raise ValueError(f"未知牆名：{wall!r}")
