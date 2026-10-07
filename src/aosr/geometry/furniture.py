"""第一版家具的純幾何：正放盒子、有限矩形鏡射與線段遮擋。

座標單位為公尺；桌子只代表桌板，不填實板下空間。這支不讀檔、不載入物理、
方案或搜尋。呼叫端從精度契約登記簿讀 furniture_geometry_contact，經
contact_margin_m 換成絕對界線；所有接觸判斷必給 margin_m，沒有備份容差。

點的深內部是每軸離兩面皆超過界線；線段穿過這個開區間才遮擋。端點在
深內部一律是輸入錯（零長度也一樣）；其餘零長度線段不擋。面上端點只免除
接觸本身，朝外離開不擋、穿過實體仍擋。重疊看各軸共同區間的深度是否超過
一份界線；不是把兩顆盒子各縮一份而容許兩倍穿入。

遮擋一次只看一顆盒子：兩顆平貼的盒子之間，沿共用面走的線段兩顆都判不擋，
會從接縫漏過去。所以方案驗證用 boxes_too_close 不准兩件家具貼在一起。
"""
from __future__ import annotations

import math
from dataclasses import InitVar, dataclass
from enum import StrEnum
from numbers import Real
from typing import TypeAlias


Vec3: TypeAlias = tuple[float, float, float]


class FurnitureKind(StrEnum):
    """家具種類；沙發／座椅貼地，其餘只表示懸空的板。"""

    SOFA = "sofa"
    CHAIR = "chair"
    COFFEE_TABLE = "coffee_table"
    DESK = "desk"
    CEILING_CLOUD = "ceiling_cloud"

    @property
    def grounded(self) -> bool:
        """底面是否須在地板接觸帶內。"""
        return self in (FurnitureKind.SOFA, FurnitureKind.CHAIR)


class FaceDirection(StrEnum):
    """房間座標方向；旋轉後仍按房間軸命名，不按家具本地軸。"""

    TOP = "top"
    BOTTOM = "bottom"
    X_PLUS = "+x"
    X_MINUS = "-x"
    Y_PLUS = "+y"
    Y_MINUS = "-y"

    @property
    def axis(self) -> int:
        """法向軸：x、y、z 分別為 0、1、2。"""
        return {self.TOP: 2, self.BOTTOM: 2, self.X_PLUS: 0,
                self.X_MINUS: 0, self.Y_PLUS: 1, self.Y_MINUS: 1}[self]

    @property
    def sign(self) -> float:
        """朝外方向的正負號。"""
        return 1.0 if self in (self.TOP, self.X_PLUS, self.Y_PLUS) else -1.0

    @property
    def edge_axes(self) -> tuple[int, int]:
        """兩條邊按房間軸序排列；全長與此順序一致。"""
        return {0: (1, 2), 1: (0, 2), 2: (0, 1)}[self.axis]


def _number(value: float, label: str) -> float:
    """拒收文字、布林與非有限實數，不把布林洗成角度或尺寸。"""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label}必須是有限實數")
    return float(value)


def _point(value: Vec3, label: str) -> Vec3:
    """三軸座標正規化成不可變的有限浮點三元組。"""
    if len(value) != 3:
        raise ValueError(f"{label}必須有三個座標")
    return (_number(value[0], label), _number(value[1], label), _number(value[2], label))


def _positive(value: float, label: str) -> float:
    value = _number(value, label)
    if value <= 0.0:
        raise ValueError(f"{label}必須是有限正數")
    return value


def _margin(value: float) -> float:
    value = _number(value, "接觸界線")
    if value < 0.0:
        raise ValueError("接觸界線必須是有限非負數")
    return value


def _room_lengths(room_size_m: Vec3) -> Vec3:
    lengths = _point(room_size_m, "房間邊長")
    return (_positive(lengths[0], "房間邊長"), _positive(lengths[1], "房間邊長"),
            _positive(lengths[2], "房間邊長"))


def _unit(axis: int, sign: float = 1.0) -> Vec3:
    return (sign if axis == 0 else 0.0, sign if axis == 1 else 0.0, sign if axis == 2 else 0.0)


@dataclass(frozen=True)
class FurnitureFace:
    """有限矩形面；法向與兩條單位邊由房間方向推得，全長不含接觸界線。"""

    direction: FaceDirection
    center_m: Vec3
    edge_lengths_m: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", FaceDirection(self.direction))
        object.__setattr__(self, "center_m", _point(self.center_m, "面中心"))
        if len(self.edge_lengths_m) != 2:
            raise ValueError("矩形面須有兩條邊全長")
        object.__setattr__(self, "edge_lengths_m", (
            _positive(self.edge_lengths_m[0], "面邊長"), _positive(self.edge_lengths_m[1], "面邊長")))

    @property
    def normal(self) -> Vec3:
        """朝外法向（房間座標的單位向量）。"""
        return _unit(self.direction.axis, self.direction.sign)

    @property
    def edge_units(self) -> tuple[Vec3, Vec3]:
        """兩條邊的單位向量；按房間軸序取正向，不宣稱為有向表面框架。"""
        first, second = self.direction.edge_axes
        return _unit(first), _unit(second)


@dataclass(frozen=True, kw_only=True)
class FurnitureBox:
    """底面中心＋本地寬深高的不可變盒子；margin_m 只用於建構時驗底面。"""

    kind: FurnitureKind | str
    width_m: float
    depth_m: float
    height_m: float
    bottom_center_m: Vec3
    margin_m: InitVar[float]
    yaw_deg: float = 0.0

    def __post_init__(self, margin_m: float) -> None:
        margin = _margin(margin_m)
        try:
            kind = FurnitureKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"未知家具種類：{self.kind!r}") from exc
        message = "第一版只收正放的盒子：繞垂直軸角度必須是數字 0／90／180／270 度"
        yaw = _number(self.yaw_deg, message)
        if yaw not in (0.0, 90.0, 180.0, 270.0):
            raise ValueError(message)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "yaw_deg", yaw)
        for name in ("width_m", "depth_m", "height_m"):
            object.__setattr__(self, name, _positive(getattr(self, name), "家具尺寸"))
        bottom = _point(self.bottom_center_m, "底面中心")
        object.__setattr__(self, "bottom_center_m", bottom)
        if kind.grounded and abs(bottom[2]) > margin:
            raise ValueError("貼地家具的底面必須在地板接觸界線內")
        if not kind.grounded and bottom[2] <= margin:
            raise ValueError("懸空家具的底面必須高於地板並超過接觸界線")
        for low, high in zip(self.minimum_m, self.maximum_m, strict=True):
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("家具尺寸在此座標無法表示有限的非退化盒子")

    @property
    def extents_m(self) -> Vec3:
        """沿房間 x、y、z 的盒子全長（不是房間尺寸）；直角轉向只交換寬深。"""
        if self.yaw_deg in (90.0, 270.0):
            return self.depth_m, self.width_m, self.height_m
        return self.width_m, self.depth_m, self.height_m

    @property
    def minimum_m(self) -> Vec3:
        """房間座標最小角；z 由底面中心直接給出。"""
        x, y, z = self.bottom_center_m
        width, depth, _ = self.extents_m
        return x - width / 2.0, y - depth / 2.0, z

    @property
    def maximum_m(self) -> Vec3:
        """房間座標最大角；桌板下方不算盒內。"""
        x, y, z = self.bottom_center_m
        width, depth, height = self.extents_m
        return x + width / 2.0, y + depth / 2.0, z + height

    @property
    def faces(self) -> tuple[FurnitureFace, ...]:
        """六面按上、下、+x、−x、+y、−y 的固定順序回傳。"""
        return tuple(self._face(direction) for direction in FaceDirection)

    def _face(self, direction: FaceDirection) -> FurnitureFace:
        low, high = self.minimum_m, self.maximum_m
        center = [low[axis] + (high[axis] - low[axis]) / 2.0 for axis in range(3)]
        center[direction.axis] = high[direction.axis] if direction.sign > 0.0 else low[direction.axis]
        first, second = direction.edge_axes
        lengths = self.extents_m
        return FurnitureFace(direction, (center[0], center[1], center[2]), (lengths[first], lengths[second]))

    @property
    def exposed_faces(self) -> tuple[FurnitureFace, ...]:
        """貼地種類不露底面；懸空種類六面全部外露（決策紙第 16 條）。"""
        grounded = FurnitureKind(self.kind).grounded
        return tuple(face for face in self.faces if not grounded or face.direction != FaceDirection.BOTTOM)


def contact_margin_m(room_size_m: Vec3, *, contact_rel: float) -> float:
    """呼叫端讀登記簿的相對值，乘房間最長邊換成公尺；不讀檔、不內建尺。"""
    return _margin(max(_room_lengths(room_size_m)) * _margin(contact_rel))


def mirror_point(point: Vec3, face: FurnitureFace, *, margin_m: float) -> Vec3:
    """對面的無限平面精確鏡射；不夾在有限面內，不因接觸帶改點的位置。"""
    _margin(margin_m)
    xyz = list(_point(point, "鏡射點"))
    axis = face.direction.axis
    xyz[axis] = _number(2.0 * face.center_m[axis] - xyz[axis], "鏡像座標")
    return xyz[0], xyz[1], xyz[2]


def single_bounce_point(source: Vec3, receiver: Vec3, face: FurnitureFace, *, margin_m: float) -> Vec3 | None:
    """只找一跳反射點；兩點都須深在外側，有限面含邊及邊外一份接觸帶。

    回 None 代表没有該反射。有效交點保留真實位置，不把邊外接觸帶的點夾回面內；
    沒有在這裡檢查其他家具遮擋，後續呼叫端須對實際折線每一段呼叫遮擋零件。
    """
    margin = _margin(margin_m)
    source, receiver = _point(source, "聲源"), _point(receiver, "接收點")
    axis, sign = face.direction.axis, face.direction.sign
    plane = face.center_m[axis]
    if sign * (source[axis] - plane) <= margin or sign * (receiver[axis] - plane) <= margin:
        return None
    image = mirror_point(source, face, margin_m=margin)
    denominator = _number(receiver[axis] - image[axis], "反射線段方向")
    if denominator == 0.0:
        return None
    fraction = (plane - image[axis]) / denominator
    if not 0.0 < fraction < 1.0:
        return None
    hit = [(1.0 - fraction) * image[i] + fraction * receiver[i] for i in range(3)]
    hit[axis] = plane
    for edge_axis, length in zip(face.direction.edge_axes, face.edge_lengths_m, strict=True):
        if abs(hit[edge_axis] - face.center_m[edge_axis]) > length / 2.0 + margin:
            return None
    return _point((hit[0], hit[1], hit[2]), "反射點")


def _deep_inside(point: Vec3, box: FurnitureBox, margin: float) -> bool:
    return all(low + margin < coordinate < high - margin
               for coordinate, low, high in zip(point, box.minimum_m, box.maximum_m, strict=True))


def point_inside_box(point: Vec3, box: FurnitureBox, *, margin_m: float) -> bool:
    """每軸離兩面均嚴格超過界線才在內部；面、邊、角與接觸帶不算內部。"""
    return _deep_inside(_point(point, "待查點"), box, _margin(margin_m))


def segment_blocked_by_box(start: Vec3, end: Vec3, box: FurnitureBox, *, margin_m: float) -> bool:
    """分軸夾開區間；擦面、單點相交、零長度不擋，深內部端點先拒收。

    面上端點不造成遮擋；整段有任何非空區間穿過深內部仍擋，不跳過整顆反射家具。
    """
    margin = _margin(margin_m)
    start, end = _point(start, "線段端點"), _point(end, "線段端點")
    if _deep_inside(start, box, margin) or _deep_inside(end, box, margin):
        raise ValueError("線段端點在家具盒子內部超過接觸界線")
    if start == end:
        return False
    entering, leaving = 0.0, 1.0
    for first, last, low, high in zip(start, end, box.minimum_m, box.maximum_m, strict=True):
        low, high = low + margin, high - margin
        if low >= high:
            return False
        delta = _number(last - first, "線段方向")
        if delta == 0.0:
            if first <= low or first >= high:
                return False
            continue
        first_t, last_t = sorted(((low - first) / delta, (high - first) / delta))
        entering, leaving = max(entering, first_t), min(leaving, last_t)
        if entering >= leaving:
            return False
    return entering < leaving


def box_within_room(box: FurnitureBox, room_size_m: Vec3, *, margin_m: float) -> bool:
    """盒子各軸不得超出房間，牆外一份界線僅吸收浮點尾差。"""
    margin, lengths = _margin(margin_m), _room_lengths(room_size_m)
    return all(low >= -margin and high <= length + margin
               for low, high, length in zip(box.minimum_m, box.maximum_m, lengths, strict=True))


def boxes_too_close(first: FurnitureBox, second: FurnitureBox, *, margin_m: float) -> bool:
    """任一軸的間隙都不超過一份界線就算太近：貼面、貼邊、貼角與重疊都算，免得接縫漏線。"""
    margin = _margin(margin_m)
    return all(max(low_a, low_b) - min(high_a, high_b) <= margin for low_a, high_a, low_b, high_b
               in zip(first.minimum_m, first.maximum_m, second.minimum_m, second.maximum_m, strict=True))


def boxes_overlap(first: FurnitureBox, second: FurnitureBox, *, margin_m: float) -> bool:
    """三軸交集深度都超過一份界線才重疊；共面、共邊、共角不算。"""
    margin = _margin(margin_m)
    return all(min(high_a, high_b) - max(low_a, low_b) > margin for low_a, high_a, low_b, high_b
               in zip(first.minimum_m, first.maximum_m, second.minimum_m, second.maximum_m, strict=True))
