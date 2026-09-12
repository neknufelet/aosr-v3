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

**只 import 標準庫、不寫死尺寸。** 鏡像法：直達路的端點就是聲源本尊；對某面牆反射一次的
路徑，把聲源對那面牆鏡射成鏡像、量鏡像到接收點的直線距離與時間——那是同一個數值，
但算式跟凍結的 donor 完全無關。二階以上一路通案：鏡像座標、階數、反彈展開各有一支
純函式，任一階都適用。

**牆名與階數怎麼推。** identity 是六元組 ``(nx, sx, ny, sy, nz, sz)``（每軸一組「反射次數、
正負號」）。order 0（全 0、全 +1）是直達；order 1 是恰好一軸 ``s=-1``（對該軸的 0 面牆
或 L 面牆）。階數由 :func:`order_of` 從 identity 獨立算：每軸 ``sign=+1`` 是 ``2*|n|``、
``sign=-1`` 是 ``|2n-1|``，三軸相加。0 面牆名 x0/y0/floor、L 面牆名 xL/yL/ceiling。

**不只算第一階。** 票 #187 把合約擴到二階、三階：任一 identity（任意階）的鏡像座標直接由
:func:`image_from_identity` 用 ``2*n*L + s*src`` 每軸一字算，不逐牆鏡射；identity 集合由
:func:`enumerate_identities` 照 donor 的去重規則獨立枚舉；逐次反彈由 :func:`expand_bounces`
展開（從接收點往鏡像點連線、依序跟最近那面牆求交、鏡回房內，重複階數次）。
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


def image_from_identity(
    room: Room,
    identity: tuple[int, int, int, int, int, int],
    src: tuple[float, float, float],
) -> tuple[float, float, float]:
    """從 identity 六元組直接算鏡像座標，任何階都適用，不逐牆鏡射。

    identity ``(nx, sx, ny, sy, nz, sz)`` 每軸一組 (反射次數 ``n``、正負號 ``s``)，鏡像公式
    每軸獨立：``image = 2*n*L + s*src``（L 是那一軸的房間長、src 是聲源那一軸的座標）。
    這跟 donor 的 ``_enumerate_image_paths`` 逐字同一條：``2.0*x_term[0]*dims[axis] +
    x_term[1]*src[axis]``。直達（全 0、全 +1）回聲源本尊。
    """
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])
    return (
        2.0 * n[0] * room.length(0) + s[0] * src[0],
        2.0 * n[1] * room.length(1) + s[1] * src[1],
        2.0 * n[2] * room.length(2) + s[2] * src[2],
    )


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
    """反射點落在牆面矩形**內部**、且 ``t`` 在開區間 (0,1)。精確比對，不留容差。

    反射點恰好碰在鏡像或接收點上（``t`` 在端點）不算真的過牆；兩條非牆軸
    的座標要落在開區間 ``(0, L)`` 內——座標剛好等於 0 或房長代表打在牆的邊線上（退化
    組態），回 ``False`` 交給呼叫端判成退化、丟 ``ValueError``。
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
        if not (0.0 < value < room.length(k)):
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


def _axis_terms(max_order: int) -> tuple[tuple[int, int], ...]:
    """一軸的 (n, sign) 字典（照 donor ``_axis_terms``／``_axis_wall_counts`` 的去重規則）。

    逐 ``n`` 從 ``-max_order`` 到 ``max_order``、逐 ``sign`` 跑 ``(1, -1)``，只收「這一軸
    自己的反彈次數 ``_axis_order(n, sign) <= max_order``」的那幾組——超過的組合就算另外兩軸
    一次都不彈，總階數也必然超標，所以直接剪掉。這是去重規則的第一道：單軸先剪。
    """
    terms: list[tuple[int, int]] = []
    for n in range(-max_order, max_order + 1):
        for sign in (1, -1):
            if _axis_order(n, sign) <= max_order:
                terms.append((n, sign))
    return tuple(terms)


def enumerate_identities(max_order: int) -> frozenset[tuple[int, int, int, int, int, int]]:
    """獨立枚舉 ``max_order`` 以內的全部 identity 六元組（照 donor 的去重規則，不是自己另寫）。

    **去重規則**（照抄 donor ``_enumerate_image_paths`` 的結尾，其單軸字典又住在
    ``_axis_terms`` 與 ``_axis_wall_counts``）：

    * 三軸各自的 ``(n, sign)`` 走 :func:`_axis_terms`（單軸反彈次數先剪到 ``<= max_order``）。
    * 三軸反彈次數加總（＝ :func:`order_of`）就是這一條的階數；``order > max_order`` 就丟。
    * 一個 identity 六元組由三軸的 ``(n, sign)`` 唯一定死（單軸那一對就是 identity 那一格），
      所以枚舉是單射、沒有兩條會撞出同一個 identity——donor 那份 ``seen`` 字典的檢查只是
      對「同一個 identity 卻算出兩種牆次數簽名」的防禦性斷言，不是真的在除重；這裡連簽名
      都不算，只回集合。

    回傳的是 identity 的**集合**（階數、身分都唯一，集合就是這份合約要的「哪些 identity」）。
    """
    ids: set[tuple[int, int, int, int, int, int]] = set()
    for nx, sx in _axis_terms(max_order):
        for ny, sy in _axis_terms(max_order):
            for nz, sz in _axis_terms(max_order):
                if order_of((nx, sx, ny, sy, nz, sz)) <= max_order:
                    ids.add((nx, sx, ny, sy, nz, sz))
    return frozenset(ids)


def wall_count_signature(
    identity: tuple[int, int, int, int, int, int],
) -> dict[str, int]:
    """identity → 每面牆反彈幾次的簽名（多重集，**不看時間順序**）。

    這是跟 :func:`expand_bounces` **獨立**的第二個判準：反彈展開是幾何量（逐牆求交、把鏡像
    一面一面剝回去），而這一份簽名純從 identity 六元組的「每軸 ``(n, sign)``」算出來，一次
    求交、一次鏡射都不做——所以它抓得到「展開算錯／少算一次」卻抓不到「簽名也跟著一起錯」
    的那種傷，反之亦然。兩者互補，不是同一份算式的兩個影本。

    每軸的拆法照上一代 donor 的 ``_axis_wall_counts``（藍圖 docstring 已寫的規則）：
    ``sign=+1`` 時該軸兩面牆各 ``|n|`` 次；``sign=-1`` 時總共 ``|2n-1|`` 次、依 ``n`` 正負
    拆給 zero 面（floor/x0/y0）與 L 面（ceiling/xL/yL）。牆名鍵照 canonical 順序
    floor/ceiling/x0/xL/y0/yL。
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


@dataclass(frozen=True)
class Bounce:
    """一階反彈的展開結果：牆名、反彈點、線段參數 ``t``、反彈點是否落在牆矩形內。"""

    wall: str
    point: tuple[float, float, float]
    t: float
    in_wall: bool


def expand_bounces(
    room: Room,
    identity: tuple[int, int, int, int, int, int],
    src: tuple[float, float, float],
    recv: tuple[float, float, float],
) -> list[Bounce]:
    """把一個 identity 的逐次反彈展開成牆名與反彈點，重複階數次，任何階都適用。

    **做法**（鏡像聲源「攤開」直線的方法）：從接收點往鏡像點連線，依序跟「最近」的那面牆
    （線段參數 ``t`` 最小、且 ``t`` 在 (0,1) 的那面）求交；這個交點就是離接收點最遠那一階
    的反彈點。接著把「目前的鏡像」再對這面牆鏡回房內（變成少一階的鏡像），從這個交點
    繼續往新的鏡像連線、找下一面牆，重複 ``order_of(identity)`` 次——每過一面牆就剝掉一階，
    剝到最後鏡像就是聲源本尊。交點的牆軸座標直接釘回平面值（0 或 L，跟一階
    :func:`reflect_point` 同一招），下一階從釘好的點出發。

    **退化組態不靜靜少算（丟 ``ValueError``）。** 三種退化一律丟，不 ``break``、不「記個
    ``in_wall=False`` 就算了」：① 找不到 ``t`` 在 (0,1) 的牆；② 兩面牆的 ``t`` 相等（打在
    邊或角上）；③ 反彈點的非牆軸座標剛好等於 0 或房長（:func:`in_wall` 開區間 ``(0, L)``
    現算回 ``False``）。訊息一致，附 identity 與那一點。
    """
    total = order_of(identity)
    image = image_from_identity(room, identity, src)
    cur = recv
    bounces: list[Bounce] = []

    def _degenerate(point: tuple[float, float, float] | None) -> None:
        raise ValueError(
            "反彈點打在牆的邊上，是退化組態，上一代也明說不支援"
            f"（identity={identity!r}，反彈點={point!r}）"
        )

    def _pin(wall: str, t: float) -> tuple[float, float, float]:
        """線段交點的牆軸座標釘回平面值（0 或 L），回三軸 tuple（釘好、型別固定）。"""
        axis, _kind = _wall_axis_kind(wall)
        plane = wall_plane_value(room, wall)
        px = cur[0] + t * (image[0] - cur[0])
        py = cur[1] + t * (image[1] - cur[1])
        pz = cur[2] + t * (image[2] - cur[2])
        values = [px, py, pz]
        values[axis] = plane
        return (values[0], values[1], values[2])

    for _ in range(total):
        candidates: list[tuple[float, str]] = []
        for wall in canonical_walls():
            axis, _kind = _wall_axis_kind(wall)
            denom = image[axis] - cur[axis]
            if denom == 0.0:
                continue
            t = (wall_plane_value(room, wall) - cur[axis]) / denom
            if 0.0 < t < 1.0:
                candidates.append((t, wall))
        if not candidates:
            _degenerate(None)
        candidates.sort(key=lambda pair: pair[0])
        t, wall = candidates[0]
        tied = [w for (tt, w) in candidates if tt == t]
        if len(tied) > 1:
            # 兩面牆同一 t：反彈點是那兩面牆交界上的點（邊或角），退化。
            _degenerate(_pin(wall, t))
        point = _pin(wall, t)
        if not in_wall(room, wall, point, t):
            _degenerate(point)
        bounces.append(
            Bounce(
                wall=wall,
                point=point,
                t=t,
                in_wall=True,
            )
        )
        image = image_of(room, wall, image)
        cur = point
    return bounces


def _wall_axis_kind(wall: str) -> tuple[int, str]:
    """牆名 → (軸 index, kind)。未知牆名當場炸（這一支只認六面牆）。"""
    for name, (axis, kind) in _CANONICAL_WALLS:
        if name == wall:
            return (axis, kind)
    raise ValueError(f"未知牆名：{wall!r}")
