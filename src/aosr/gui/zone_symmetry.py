"""左右對稱的配對（#1206 第 2 支）：由程式最佳化的區，哪兩區對著對稱軸互為鏡像。純函式：不開行程、不寫檔。

只有這一份算法：分區頁從伺服器拿它的結果（``/api/zone-limits`` 順便回），開跑計畫（``gradient_plan.plan_search``）
用它填綁定表。規則（老闆 11-17 選「左右對稱，預設打開」、11-18 對稱軸選「喇叭中點到主位」）：

- **開關**住在設計設定 ``left_right_symmetry``：只存「關」（``off``），缺這一格就是開（:func:`symmetry_on`）。
- **只管由程式最佳化的區**；指定貼一種、保留原樣的區不配、不列。
- **對稱軸**＝兩支喇叭的水平中點到主位的連線（``reporting.scheme`` 的 ``project_facing``／``project_midpoint``）。
  定不出前牆、或主位橫向離兩支喇叭中點超過容差：整張不配，照實寫原因。
- **鏡像**每一邊都在容差內才算：前後牆、地板、天花在同一面裡對著軸鏡像；左右兩面牆同位置對同位置，
  而且只有軸在房間正中（差在容差內）時才互為鏡像。自己跨在軸上、左右置中的區不用配。
- **設定也要一樣**：位置對上的每一對交給呼叫端給的檢查（開跑計畫給的是引擎的開跑前檢查
  ``aosr.gradient.ties.check_ties_against_start``），它拒收的那一對算沒配成、原因照它回的那一句。這裡不另寫一份規則。
- **沒配成**：由程式最佳化的區，沒進一對、也不是自己置中的，都算沒配成、都列原因。兩區的：位置對上但設定不同、
  鏡像位置疊到但差超過容差（這一種把兩種切法寫出來，數字是離哪面牆幾公尺）、軸不在正中的左右牆同位置兩區。
  只有一區的（半對）：鏡像位置那一區不是由程式最佳化（指定貼一種或保留原樣，``other`` 記是哪一區，它自己不掛記號）、
  鏡像位置沒有一樣大小的區、鏡像位置超出牆面、跨在軸上但沒有置中、軸不在正中的左右牆上落單的區。
  每一筆帶一句 ``summary``（「左右對稱」那一塊直接寫出來，老闆 11-18 點頭的樣稿那一列就寫原因）；差幾公分、
  兩種切法在配對細節。切法只列做得到的：切過去會超出牆面、或會切在另一區自己身上的不列。

一對裡在前的是面序在前的那一片（命令列面片的順序），跟綁定表那一層的代表片一致；沒有主次，兩片共用同一組數字。
給人看的句子裡用 ``{first}``／``{second}`` 代表那兩區，畫面換成它自己的號碼。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from aosr.gradient.boundary import scheme_patch_keys, surface_ids
from aosr.gui.wall_words import LEFT_OF, OPPOSITE, wall_words
from aosr.gui.zone_design import WHOLE_WALL_KEY, DesignSettings, effective_modes
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.scheme import Scheme, project_facing, project_midpoint

# 配對規則本身（#1206 規格定的「差在 1 公分內才算一對」），不是計算之間的精度契約；同 zone_edit 的 MIN_SIDE_M 那一類。
PAIR_WITHIN_M: Final = 0.01
SWITCHED_OFF: Final = "off"
NOT_BETWEEN: Final = "主位不在兩支喇叭中間，定不出對稱軸"
NO_FRONT: Final = "從兩支喇叭的中點和主位看不出哪一面是前牆，定不出對稱軸"
SIDES_NOT_MIRRORED: Final = "對稱軸不在房間正中，左右兩面牆不互為鏡像"
SettingsCheck = Callable[[str, str], str | None]  # （面序在前的片, 在後的片）→ 可以配回 None，不行回一句白話原因
Span = tuple[float, float]
_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_FLAT: Final = ("floor", "ceiling")
_SLACK: Final = 1e-9  # 浮點尾數：2c − u 這種算出來的邊不會剛好是整數公分


class Axis(BaseModel):
    """對稱軸：沿哪一個座標（``along``）、橫向落在幾公尺（``across_m``）、在不在房間正中；前牆與左牆是哪一面；
    前後牆的代號（軸在這兩面上是一條直線）與地板圖上那一條線的兩端（房間座標 x、y）。"""

    model_config = _FROZEN
    along: Literal["x", "y"]
    across_m: float
    centred: bool
    front: str
    left: str
    end_walls: tuple[str, str]
    line_m: tuple[tuple[float, float], tuple[float, float]]
    words: str  # 給人看的一句：對稱軸跟著什麼走、現在離左牆幾公尺、在不在房間正中


class Pair(BaseModel):
    model_config = _FROZEN
    first: str
    second: str


class Figure(BaseModel):
    """配對細節那張小圖：``wall`` 這一面上 ``second`` 那一區的框，與 ``first`` 對過來的框（都是這一面的 u、v 公尺）。"""

    model_config = _FROZEN
    wall: str
    second_u_m: Span
    second_v_m: Span
    mirrored_u_m: Span
    mirrored_v_m: Span


class Missed(BaseModel):
    """沒配成一對的區（``second`` 是 None 時只有 ``first`` 一區）：``position`` 兩區位置差超過容差（有兩種切法與小圖）、
    ``settings`` 兩區位置對上但設定檢查拒收、``side_walls`` 軸不在正中的左右牆（同位置兩區，或落單一區）、
    ``not_search`` 鏡像位置那一區（``other``）不是由程式最佳化、``no_zone`` 鏡像位置沒有一樣大小的區、
    ``outside`` 鏡像位置超出牆面、``straddle`` 跨在軸上但沒有置中（鏡像位置跟它自己疊在一起）。
    ``summary`` 是「左右對稱」那一塊直接寫出來的一句（為什麼沒配成、要怎麼辦），``reason``／``detail``／``cuts`` 在配對細節。
    ``cuts`` 只列做得到的切法：切過去會超出牆面、或會切在另一區自己身上的不列（一種都沒有就是空的）。"""

    model_config = _FROZEN
    first: str
    second: str | None = None
    other: str | None = None
    kind: Literal["position", "settings", "side_walls", "not_search", "no_zone", "outside", "straddle"]
    summary: str
    reason: str
    detail: str
    cuts: tuple[str, ...] = ()
    figure: Figure | None = None


class Symmetry(BaseModel):
    """配對結果。``searched``：由程式最佳化的片（面序）；``centred``：自己跨在軸上的片；``alone``：各自調的區數
    （沒進任何一對的，含置中的）；``reason``：整張不配的原因（定不出軸）。關掉時 ``enabled`` 是 False、其餘照空的。"""

    model_config = _FROZEN
    enabled: bool
    reason: str | None = None
    axis: Axis | None = None
    searched: tuple[str, ...] = ()
    pairs: tuple[Pair, ...] = ()
    missed: tuple[Missed, ...] = ()
    centred: tuple[str, ...] = ()
    alone: int = 0


@dataclass(frozen=True)
class _Zone:
    surface: str
    wall: str
    u: Span
    v: Span


def symmetry_on(design: DesignSettings | None) -> bool:
    """左右對稱開著嗎：設計設定只存「關」，缺這一格（舊檔、沒有設計設定）一律是開。"""
    return design is None or design.left_right_symmetry != SWITCHED_OFF


def _near(left: float, right: float) -> bool:
    return abs(left - right) <= PAIR_WITHIN_M + _SLACK


def _same(left: Span, right: Span) -> bool:
    return _near(left[0], right[0]) and _near(left[1], right[1])


def _axis(scheme: Scheme) -> Axis | str:
    try:
        facing, middle = project_facing(scheme), project_midpoint(scheme)
    except ValueError:
        return NO_FRONT
    along = 0 if facing[0] else 1
    name: Literal["x", "y"] = "x" if along == 0 else "y"
    across = middle[1 - along]
    if not _near(scheme.receiver_set.primary.position_m[1 - along], across):
        return NOT_BETWEEN
    room = scheme.scene.room_m
    lengths = (room.Lx, room.Ly)
    front = ("xL" if facing[0] > 0 else "x0") if along == 0 else ("yL" if facing[1] > 0 else "y0")
    ends = ((0.0, across), (lengths[0], across)) if along == 0 else ((across, 0.0), (across, lengths[1]))
    left, centred = LEFT_OF[front], _near(across, lengths[1 - along] / 2)
    from_left = across if left.endswith("0") else lengths[1 - along] - across
    words = (f"對稱軸跟著兩支喇叭的中點和主位走，現在離{wall_words(scheme)[left]} {from_left:.2f} 公尺、"
             f"{'在' if centred else '不在'}房間正中；主位不在兩支喇叭中間時，就不配對。")
    return Axis(along=name, across_m=across, centred=centred, front=front, left=left,
                end_walls=("x0", "xL") if along == 0 else ("y0", "yL"), line_m=ends, words=words)


def _size(scheme: Scheme, wall: str) -> tuple[float, float]:
    room = scheme.scene.room_m
    return (room.Lx, room.Ly) if wall in _FLAT else (room.Ly if wall[0] == "x" else room.Lx, room.Lz)


def _zones(scheme: Scheme, design: DesignSettings | None) -> tuple[list[_Zone], list[_Zone]]:
    """（由程式最佳化的區, 其餘的區），各照命令列面片的順序；整面一區的牆照整面的範圍。"""
    modes = effective_modes(design, scheme)
    zones_by_wall = scheme.scene.surface_zones_by_wall or {}
    searched: list[_Zone] = []
    others: list[_Zone] = []
    for surface in surface_ids(scheme_patch_keys(scheme)):
        wall, separator, key = surface.partition("@")
        width, height = _size(scheme, wall)
        u, v = (0.0, width), (0.0, height)
        for zone in zones_by_wall.get(wall, ()):
            if separator and isinstance(zone, RectangleZone) and zone.id == key:
                u, v = zone.u_m, zone.v_m
        (searched if modes[wall][key if separator else WHOLE_WALL_KEY] == "search" else others).append(_Zone(surface, wall, u, v))
    return searched, others


class _Frame:
    """一份方案的鏡像與說法：哪幾面是左右牆、每一面的 u／v 哪一個是橫向，以及「離哪面牆幾公尺」怎麼寫。"""

    def __init__(self, scheme: Scheme, axis: Axis) -> None:
        self.axis = axis
        self.names = wall_words(scheme)
        room = scheme.scene.room_m
        self.across_length, self.along_length = (room.Ly, room.Lx) if axis.along == "x" else (room.Lx, room.Ly)
        self.sizes = {wall: _size(scheme, wall) for wall in (*_FLAT, "x0", "xL", "y0", "yL")}

    def side(self, wall: str) -> bool:
        return wall not in _FLAT and wall not in self.axis.end_walls

    def _across_is_u(self, wall: str) -> bool:
        return wall in self.axis.end_walls or (wall in _FLAT and self.axis.along == "y")

    def _flipped(self, span: Span) -> Span:
        return 2 * self.axis.across_m - span[1], 2 * self.axis.across_m - span[0]

    def mirrored(self, zone: _Zone) -> tuple[str, Span, Span]:
        """這一區對到軸的另一邊：左右牆是對面那一面的同位置，其餘是同一面裡橫向翻過去。"""
        if self.side(zone.wall):
            return OPPOSITE[zone.wall], zone.u, zone.v
        if self._across_is_u(zone.wall):
            return zone.wall, self._flipped(zone.u), zone.v
        return zone.wall, zone.u, self._flipped(zone.v)

    def _from(self, wall: str, span: Span, length: float) -> str:
        low, high = span if wall.endswith("0") else (length - span[1], length - span[0])
        return f"離{self.names[wall]} {low:.2f}～{high:.2f} 公尺"

    def _across(self, span: Span) -> str:
        return self._from(self.axis.left, span, self.across_length)

    def _along(self, span: Span) -> str:
        return self._from(self.axis.front, span, self.along_length)

    def words(self, wall: str, u: Span, v: Span, which: tuple[bool, bool]) -> str:
        """一個框寫成「離左牆 a～b 公尺、離地 c～d 公尺」；``which`` 挑要寫 u、v 哪幾邊。"""
        if wall in _FLAT:
            both = (self._across(u), self._along(v)) if self._across_is_u(wall) else (self._along(u), self._across(v))
        else:
            both = (self._along(u) if self.side(wall) else self._across(u), f"離地 {v[0]:.2f}～{v[1]:.2f} 公尺")
        return "、".join(text for text, wanted in zip(both, which, strict=True) if wanted)


    def inside(self, wall: str, u: Span, v: Span) -> bool:
        """這個框整個在這一面牆裡（放過浮點尾數）。"""
        width, height = self.sizes[wall]
        return u[0] >= -_SLACK and u[1] <= width + _SLACK and v[0] >= -_SLACK and v[1] <= height + _SLACK

    def _edge_names(self, wall: str, across: bool, upright: bool, span: Span) -> tuple[str, str]:
        """一段範圍的兩個邊叫什麼（先小的那一邊）：橫向的講靠中間還是靠哪面牆、縱深的講靠前牆還是後牆、高度講下緣上緣。"""
        if upright:
            return "下緣", "上緣"
        if not across:
            front, back = f"靠{self.names[self.axis.front]}那一邊", f"靠{self.names[OPPOSITE[self.axis.front]]}那一邊"
            return (front, back) if self.axis.front.endswith("0") else (back, front)
        low_wall, high_wall = (self.axis.left, OPPOSITE[self.axis.left]) if self.axis.left.endswith("0") else (OPPOSITE[self.axis.left], self.axis.left)
        low, high = f"靠{self.names[low_wall]}那一邊", f"靠{self.names[high_wall]}那一邊"
        if span[1] <= self.axis.across_m + PAIR_WITHIN_M:
            return low, "靠中間那一邊"
        if span[0] >= self.axis.across_m - PAIR_WITHIN_M:
            return "靠中間那一邊", high
        return low, high

    def gaps(self, wall: str, u: Span, v: Span, zone: _Zone) -> str:
        """框（u, v）跟 ``zone`` 疊著比，哪幾個邊差多少：「靠中間那一邊差 20 公分、靠右牆那一邊差 5 公分」。"""
        flat, u_across = wall in _FLAT, self._across_is_u(wall)
        found = []
        for mine, theirs, across, upright in ((u, zone.u, u_across, False), (v, zone.v, flat and not u_across, not flat)):
            for name, here, there in zip(self._edge_names(wall, across, upright, theirs), mine, theirs, strict=True):
                if not _near(here, there):
                    found.append(f"{name}差 {_centimetres(abs(here - there))} 公分")
        return "、".join(found)


def _centimetres(metres: float) -> str:
    return f"{metres * 100:.1f}".rstrip("0").rstrip(".")


def _overlap(u: Span, v: Span, zone: _Zone) -> float:
    wide = min(u[1], zone.u[1]) - max(u[0], zone.u[0])
    tall = min(v[1], zone.v[1]) - max(v[0], zone.v[0])
    return wide * tall if wide > _SLACK and tall > _SLACK else 0.0


_GO_CUT: Final = "要配對，回「房間方案輸入」切到左右對稱的位置"


def _position_miss(frame: _Frame, first: _Zone, second: _Zone) -> Missed:
    wall, u, v = frame.mirrored(first)
    back_wall, back_u, back_v = frame.mirrored(second)
    which = (not _same(u, second.u), not _same(v, second.v))
    limit = f"{PAIR_WITHIN_M * 100:g} 公分"
    detail = (f"{{first}} 對到對稱軸另一邊是{frame.words(wall, u, v, which)}；{{second}} 是"
              f"{frame.words(wall, second.u, second.v, which)}。把 {{first}} 翻過去跟 {{second}} 疊著比："
              f"{frame.gaps(wall, u, v, second)}。每一邊都要在 {limit}以內才算一對。")
    # 做得到的切法才列：切過去要整個在牆裡，而且不能切在留著不動的那一區自己身上。
    cuts = tuple(text for text, (to_wall, to_u, to_v), stays in (
        (f"把 {{second}} 切到{frame.words(wall, u, v, which)}，就和 {{first}} 配成一對", (wall, u, v), first),
        (f"把 {{first}} 切到{frame.words(back_wall, back_u, back_v, which)}，就和 {{second}} 配成一對",
         (back_wall, back_u, back_v), second))
        if frame.inside(to_wall, to_u, to_v) and not (to_wall == stays.wall and _overlap(to_u, to_v, stays) > 0.0))
    sized = all(_near(mine[1] - mine[0], theirs[1] - theirs[0]) for mine, theirs in ((u, second.u), (v, second.v)))
    what = "位置沒對齊" if sized else "大小不同"
    if cuts:
        summary = f"{what}，各自調；{_GO_CUT}"
    else:
        outside = not frame.inside(wall, u, v) or not frame.inside(back_wall, back_u, back_v)
        stuck = "鏡像過去會超出牆面" if outside else "鏡像過去會疊在它自己身上"
        summary = f"{stuck}，這兩區沒辦法配成一對，各自調"
        detail += f"{stuck}，這兩區沒辦法配成一對。"
    return Missed(first=first.surface, second=second.surface, kind="position", summary=summary,
                  reason="兩區大小或位置不一樣，對不成鏡像", detail=detail, cuts=cuts,
                  figure=Figure(wall=wall, second_u_m=second.u, second_v_m=second.v, mirrored_u_m=u, mirrored_v_m=v))


def _lone_miss(frame: _Frame, zone: _Zone, others: list[_Zone]) -> Missed:
    """半對：這一區由程式最佳化，鏡像位置沒有由程式最佳化的對象。"""
    if frame.side(zone.wall) and not frame.axis.centred:
        return Missed(first=zone.surface, kind="side_walls", summary=_SIDES_SUMMARY, reason=SIDES_NOT_MIRRORED,
                      detail=f"{{first}} 在左右牆上，但{SIDES_NOT_MIRRORED}，所以沒有對象可以配。")
    wall, u, v = frame.mirrored(zone)
    other = next((item for item in others if item.wall == wall and _same(u, item.u) and _same(v, item.v)), None)
    if other is not None:
        return Missed(first=zone.surface, other=other.surface, kind="not_search",
                      summary="對面那一區不是由程式最佳化，各自調；兩區都改成由程式最佳化才配對",
                      reason="鏡像位置那一區不是由程式最佳化",
                      detail="{first} 對到對稱軸另一邊是 {other}，但 {other} 是指定貼一種或保留原樣，不是由程式最佳化。"
                             "把 {other} 也改成由程式最佳化，兩區就配成一對。")
    if not frame.inside(wall, u, v):
        return Missed(first=zone.surface, kind="outside", summary="鏡像過去會超出牆面，沒辦法配成一對，各自調",
                      reason="鏡像過去會超出牆面",
                      detail="{first} 對到對稱軸另一邊會超出牆面，這一區沒辦法配成一對。")
    if wall == zone.wall and _overlap(u, v, zone) > 0.0:
        return Missed(first=zone.surface, kind="straddle", summary="跨在對稱軸上但左右沒有置中，各自調",
                      reason="跨在對稱軸上，但左右沒有置中",
                      detail="{first} 跨在對稱軸上，但左右沒有置中；鏡像位置跟它自己疊在一起，所以沒有對象可以配。")
    where = frame.words(wall, u, v, (True, True))
    return Missed(first=zone.surface, kind="no_zone",
                  summary="對面沒有一樣大小的區，各自調；要配對，回「房間方案輸入」在對稱的位置切出一區",
                  reason="鏡像位置沒有一樣大小的區",
                  detail=f"{{first}} 對到對稱軸另一邊是{_other_wall(frame, wall, zone)}{where}，那裡沒有剛好這樣大小的一區。"
                         "在那裡切出一區、也改成由程式最佳化，兩區就配成一對。")


_SIDES_SUMMARY: Final = "對稱軸不在房間正中，左右牆不配對，各自調"


def _other_wall(frame: _Frame, wall: str, zone: _Zone) -> str:
    """鏡像落在另一面牆時先講是哪一面（左右牆互為鏡像）。"""
    return "" if wall == zone.wall else f"{frame.names[wall]}的"


def pair_zones(scheme: Scheme, design: DesignSettings | None, check: SettingsCheck) -> Symmetry:
    """配對。``check`` 只在兩區都由程式最佳化、位置對上的那幾對上叫（關掉、定不出軸、沒有這種對象時一次都不叫）。"""
    zones, others = _zones(scheme, design)
    searched = tuple(zone.surface for zone in zones)
    if not symmetry_on(design):
        return Symmetry(enabled=False, searched=searched, alone=len(zones))
    axis = _axis(scheme)
    if isinstance(axis, str):
        return Symmetry(enabled=True, reason=axis, searched=searched, alone=len(zones))
    frame = _Frame(scheme, axis)
    pairs: list[Pair] = []
    missed: list[Missed] = []
    centred: list[str] = []
    used: set[str] = set()
    for zone in zones:
        if zone.surface in used:
            continue
        wall, u, v = frame.mirrored(zone)
        if wall == zone.wall and _same(u, zone.u) and _same(v, zone.v):
            centred.append(zone.surface)
            used.add(zone.surface)
            continue
        other = next((item for item in zones if item.surface not in used and item is not zone and item.wall == wall
                      and _same(u, item.u) and _same(v, item.v)), None)
        if other is None:
            continue
        used.update((zone.surface, other.surface))
        if frame.side(zone.wall) and not axis.centred:
            missed.append(Missed(first=zone.surface, second=other.surface, kind="side_walls", summary=_SIDES_SUMMARY,
                                 reason=SIDES_NOT_MIRRORED,
                                 detail=f"{{first}} 和 {{second}} 在兩面牆的同一個位置，但{SIDES_NOT_MIRRORED}，所以不配成一對。"))
            continue
        refused = check(zone.surface, other.surface)
        if refused is None:
            pairs.append(Pair(first=zone.surface, second=other.surface))
        else:
            missed.append(Missed(first=zone.surface, second=other.surface, kind="settings",
                                 summary=f"{refused}，各自調；兩區設成一樣就配成一對", reason=refused,
                                 detail=f"{{first}} 和 {{second}} 的位置對上了，但{refused}。兩區設成一樣就會配成一對。"))
    for zone in zones:
        if zone.surface in used or (frame.side(zone.wall) and not axis.centred):
            continue
        wall, u, v = frame.mirrored(zone)
        near = [(area, item) for item in zones if item.surface not in used and item is not zone and item.wall == wall
                and (area := _overlap(u, v, item)) > 0.0]
        if near:
            other = max(near, key=lambda found: found[0])[1]
            used.update((zone.surface, other.surface))
            missed.append(_position_miss(frame, zone, other))
    missed.extend(_lone_miss(frame, zone, others) for zone in zones if zone.surface not in used)
    return Symmetry(enabled=True, axis=axis, searched=searched, pairs=tuple(pairs), missed=tuple(missed),
                    centred=tuple(centred), alone=len(zones) - 2 * len(pairs))
