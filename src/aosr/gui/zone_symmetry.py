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
  檢查分得出兩種沒過：這一對兩區設定不一樣（回一句原因），與整份設定現在開不了搜尋、核不了（回 :class:`Unchecked`，
  那一列不叫人把兩區「設成一樣」——兩區本來可能就一樣）。
- **沒配成**：由程式最佳化的區，沒進一對、也不是自己置中的，都算沒配成、都列原因。兩區的：位置對上但設定不同、
  鏡像位置疊到但差超過容差（這一種把兩種切法寫出來，數字是離哪面牆幾公尺）、軸不在正中的左右牆同位置兩區。
  只有一區的（半對）：鏡像位置那一區不是由程式最佳化（指定貼一種或保留原樣，``other`` 記是哪一區，它自己不掛記號）、
  鏡像位置沒有一樣大小的區、鏡像位置超出牆面、跨在軸上但沒有置中、軸不在正中的左右牆上落單的區。
  每一筆帶一句 ``summary``（「左右對稱」那一塊直接寫出來，老闆 11-18 點頭的樣稿那一列就寫原因）；差幾公分、
  兩種切法在配對細節。切法只列「照做之後兩區真的配成一對」的（:func:`_cut_blocked`）：照「房間方案輸入」真的會做的那樣
  算一遍——只把要搬的那一區換到鏡像位置（照句子印的數字、過網頁的吸邊）、交給切牆工具 ``zone_edit.cut_wall`` 切、切完兩塊都還在原位、設定跟著走。
  兩種都切不出來時那一列寫「只搬其中一區切不出鏡像的位置」，細節兩個方向各寫一句為什麼。

一對裡在前的是面序在前的那一片（命令列面片的順序），跟綁定表那一層的代表片一致；沒有主次，兩片共用同一組數字。
給人看的句子裡用 ``{first}``／``{second}`` 代表那兩區，畫面換成它自己的號碼。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, NamedTuple, cast

from pydantic import BaseModel, ConfigDict

from aosr.gradient.boundary import scheme_patch_keys, surface_ids
from aosr.gui.wall_words import LEFT_OF, OPPOSITE, wall_words
from aosr.gui.zone_design import WHOLE_WALL_KEY, DesignSettings, effective_modes
from aosr.gui.zone_edit import CutBox, WallName, ZoneCutError, cut_wall, snap_edges
from aosr.physics.report_surfaces import RectangleZone
from aosr.physics.zone_fill import FILL_PREFIX
from aosr.reporting.scheme import Scheme, project_facing, project_midpoint

# 配對規則本身（#1206 規格定的「差在 1 公分內才算一對」），不是計算之間的精度契約；同 zone_edit 的 MIN_SIDE_M 那一類。
PAIR_WITHIN_M: Final = 0.01
SWITCHED_OFF: Final = "off"
NOT_BETWEEN: Final = "主位不在兩支喇叭中間，定不出對稱軸"
NO_FRONT: Final = "從兩支喇叭的中點和主位看不出哪一面是前牆，定不出對稱軸"
SIDES_NOT_MIRRORED: Final = "對稱軸不在房間正中，左右兩面牆不互為鏡像"


class Unchecked(NamedTuple):
    """設定檢查做不成（不是這一對兩區設定不一樣，是整份設定現在開不了搜尋或開跑前檢查沒過）：``reason`` 一句、
    ``because`` 是細節那一句「…位置對上了，但」後面接的話。"""

    reason: str
    because: str


# （面序在前的片, 在後的片）→ 可以配回 None；這一對兩區設定不一樣回一句白話原因；整份設定核不了回 :class:`Unchecked`。
SettingsCheck = Callable[[str, str], "str | Unchecked | None"]
_PRINTED: Final = 2  # 句子裡的公尺印到小數幾位（``_Frame._from`` 的 ``.2f``）
_UNCHECKED_SUMMARY: Final = "位置對上了；這份設定現在開不了搜尋，等開得了再看兩區的設定一不一樣"
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
    ``cuts`` 只列照做之後兩區真的配成一對的切法（:func:`_cut_blocked` 照切牆工具算過一遍；一種都沒有就是空的，
    這時 ``detail`` 兩個方向各寫一句為什麼切不出來）。"""

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
        self._scheme = scheme
        self._document: dict[str, object] | None = None

    def document(self) -> dict[str, object]:
        """方案寫成切牆工具吃的那種字典（要試切法時才轉，一份方案只轉一次）。"""
        if self._document is None:
            self._document = self._scheme.model_dump(mode="json")
        return self._document

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


def _cut_blocked(frame: _Frame, moved: _Zone, to: tuple[str, Span, Span], stays: _Zone, stays_word: str) -> str | None:
    """「把 ``moved`` 切到 ``to``、``stays`` 留著不動」照做之後兩區會不會真的配成一對：會回 None，不會回一句為什麼。

    照「房間方案輸入」真的會做的那樣算：網頁送給切牆工具的是這一面畫的框（矩形區扣掉自動補滿的，``elevation.js`` 的
    ``drawnBoxes``），只把要搬的那一區換到鏡像位置——換成人照句子打的數字、再過網頁那一步吸邊，吸完離鏡像位置
    超過容差就切不到；切交給 ``zone_edit.cut_wall`` 判（最小邊、不重疊都是它的規則，
    這裡不另寫）；切完照位置找回兩塊（自動補的代號每切一次就重編，不照代號找），而且每一塊沿用的設定
    （``inherited``，``zone_modes.js`` 的 ``remapZoneDesign`` 照它搬）要是原來那一區的。``stays_word`` 是句子裡怎麼叫留著的那一區。"""
    wall, u, v = to
    if not frame.inside(wall, u, v):
        return "會超出牆面"
    if wall == stays.wall and _overlap(u, v, stays) > 0.0:
        return f"會疊在 {stays_word} 上"
    # 整面一區的那一面還沒有框：照做就是在鏡像位置畫第一個框（代號網頁會另取，這裡隨便給一個），設定沿用整面那一區的。
    key = moved.surface.partition("@")[2] or WHOLE_WALL_KEY
    if key.startswith(FILL_PREFIX):
        return "它是自動補滿的那一塊，要先在立面圖畫一個框，才搬得動"
    scene = cast(dict[str, object], frame.document()["scene"])
    entries = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])[wall]
    drawn = [entry for entry in entries if entry["area"] == "rectangle" and not cast(str, entry["id"]).startswith(FILL_PREFIX)]
    others = [CutBox(id=cast(str, entry["id"]), u_m=cast(Span, tuple(cast(list[float], entry["u_m"]))),
                     v_m=cast(Span, tuple(cast(list[float], entry["v_m"])))) for entry in drawn if entry["id"] != key]
    # 人打的是句子印出來的數字（公分），網頁送出之前還會吸邊（``zone_edit.snap_edges``）：照那樣算出真的會送的框。
    typed = CutBox(id=key, u_m=(round(u[0], _PRINTED), round(u[1], _PRINTED)), v_m=(round(v[0], _PRINTED), round(v[1], _PRINTED)))
    sent = snap_edges(typed, others, *frame.sizes[wall])
    if not (_same(sent.u_m, u) and _same(sent.v_m, v)):
        return "照打數字，框邊會被吸到牆邊或旁邊那一區的邊上，切不到這個位置"
    try:
        cut = cut_wall(frame.document(), cast(WallName, wall), [*others, sent])
    except ZoneCutError:
        # 過了吸邊，網頁畫得出來的擺法只剩「疊在別的框上」會被拒收；不是網頁畫的方案（有不到最小邊的框或縫）另寫一句。
        if any(_overlap(sent.u_m, sent.v_m, _Zone(other.id, wall, other.u_m, other.v_m)) > 0.0 for other in others):
            return "會疊在這一面別的區上"
        return "切牆工具不收這個框"
    inherited = cast(dict[str, str], cut["inherited"])

    def kept(at_u: Span, at_v: Span, source: str) -> bool:
        return any(_same(at_u, cast(Span, tuple(cast(list[float], zone["u_m"])))) and _same(at_v, cast(Span, tuple(cast(list[float], zone["v_m"]))))
                   and inherited[cast(str, zone["id"])] == source for zone in cast(list[dict[str, object]], cut["zones"]))

    if not kept(u, v, key) or (stays.wall == wall and not kept(stays.u, stays.v, stays.surface.partition("@")[2])):
        return f"補滿之後 {stays_word} 那一塊的形狀會跟著變"
    return None


def _position_miss(frame: _Frame, first: _Zone, second: _Zone) -> Missed:
    wall, u, v = frame.mirrored(first)
    back_wall, back_u, back_v = frame.mirrored(second)
    which = (not _same(u, second.u), not _same(v, second.v))
    limit = f"{PAIR_WITHIN_M * 100:g} 公分"
    detail = (f"{{first}} 對到對稱軸另一邊是{frame.words(wall, u, v, which)}；{{second}} 是"
              f"{frame.words(wall, second.u, second.v, which)}。把 {{first}} 翻過去跟 {{second}} 疊著比："
              f"{frame.gaps(wall, u, v, second)}。每一邊都要在 {limit}以內才算一對。")
    # 兩個方向各試一次：只搬一區、另一區留著。照做之後真的配成一對的才列。
    tried = ((f"把 {{second}} 切到{frame.words(wall, u, v, which)}", "{first}",
              _cut_blocked(frame, second, (wall, u, v), first, "{first}")),
             (f"把 {{first}} 切到{frame.words(back_wall, back_u, back_v, which)}", "{second}",
              _cut_blocked(frame, first, (back_wall, back_u, back_v), second, "{second}")))
    cuts = tuple(f"{move}，就和 {partner} 配成一對" for move, partner, blocked in tried if blocked is None)
    if cuts:
        sized = all(_near(mine[1] - mine[0], theirs[1] - theirs[0]) for mine, theirs in ((u, second.u), (v, second.v)))
        summary = f"{'位置沒對齊' if sized else '大小不同'}，各自調；{_GO_CUT}"
    else:
        summary = "只搬其中一區切不出鏡像的位置，各自調"
        detail += "只搬其中一區切不出鏡像的位置：" + "；".join(f"{move}，{blocked}" for move, _, blocked in tried) + "。"
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
                      summary="對稱軸另一邊那一區不是由程式最佳化，各自調；兩區都改成由程式最佳化才配對",
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
                  summary="對稱軸另一邊沒有一樣大小的區，各自調；要配對，回「房間方案輸入」在對稱的位置切出一區",
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
        elif isinstance(refused, Unchecked):
            missed.append(Missed(first=zone.surface, second=other.surface, kind="settings", summary=_UNCHECKED_SUMMARY,
                                 reason=refused.reason, detail=f"{{first}} 和 {{second}} 的位置對上了，但{refused.because}"))
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
