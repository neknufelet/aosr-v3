"""計算前的家具預篩；判定共用家具配置入口，搜尋層只另量公尺違反量。"""
from dataclasses import dataclass
from itertools import combinations
import math

from aosr.geometry.furniture import FurnitureBox, FurnitureKind, Vec3, contact_margin_m
from aosr.geometry.shoebox import Point
from aosr.reporting import furniture_layout
from aosr.reporting.scheme import ListenerPlacement, Scheme, absolute_furniture
from aosr.reporting.validation import SchemeProblem
from aosr.search import constraints
from aosr.search.constraints import Reason, Violation
from aosr.search.layout_settings import Box, Cabinet, LayoutSettings


def segment_box_length_m(start: Vec3, end: Vec3, box: FurnitureBox) -> float:
    """逐軸切線段（slab 法），量穿過原盒子的長度；接觸帶只用於上游判定。

    t 限在 [0,1]，三軸進入值取最大、離開值取最小，再乘原線段長度。
    平行軸在面上或面外、單點交集與零長度都回零；不縮盒來少算實際長度。
    """
    entering, leaving = 0.0, 1.0
    for first, last, low, high in zip(start, end, box.minimum_m, box.maximum_m, strict=True):
        delta = last - first
        if delta == 0.0:
            if first <= low or first >= high:
                return 0.0
            continue
        first_t, last_t = sorted(((low - first) / delta, (high - first) / delta))
        entering, leaving = max(entering, first_t), min(leaving, last_t)
        if entering >= leaving:
            return 0.0
    return (leaving - entering) * math.dist(start, end)


def _outside_m(box: FurnitureBox, room_size: Vec3) -> float:
    """每軸近牆與遠牆的超出距離，取最大；各軸用自己的房間邊長。"""
    return max(0.0, *(max(-low, high - length) for low, high, length
                      in zip(box.minimum_m, box.maximum_m, room_size, strict=True)))


def _separation_m(first: FurnitureBox, second: FurnitureBox) -> float:
    """各軸分離＝較大下緣減較小上緣，取最大；正值是間隙、負值是重疊。

    三軸均重疊時，最大分離的負值等於最小交集深度的負值（包含一盒包住另一盒）。
    缺口＝接觸界線減最大分離：分開時補不足間隙，重疊時加上最小交集深度。
    這是盒間接觸規格的違反量，並非把整件家具移出的最短位移。
    """
    return max(max(low_a, low_b) - min(high_a, high_b) for low_a, high_a, low_b, high_b
               in zip(first.minimum_m, first.maximum_m, second.minimum_m, second.maximum_m, strict=True))


def _point_depth_m(point: Vec3, box: FurnitureBox) -> float:
    """到六面距離的最小值；盒外或接觸為非正值。"""
    return min(min(coordinate - low, high - coordinate)
               for coordinate, low, high in zip(point, box.minimum_m, box.maximum_m, strict=True))


def _placement_amount_m(scheme: Scheme, margin: float) -> float:
    """配置入口丟錯後重建原盒子，只量越界、間隙與點內部三種已定義缺口。

    面向、底面或其他結構錯誤仍直接丟錯；不用錯誤文字辨因，也不補假量。
    """
    items = absolute_furniture(scheme)
    if items is None:
        raise ValueError("家具配置不合法卻沒有家具可量")
    boxes = tuple(FurnitureBox(kind=item.kind, width_m=item.width_m, depth_m=item.depth_m,
                               height_m=item.height_m, bottom_center_m=item.bottom_center_m,
                               yaw_deg=item.yaw_deg, margin_m=margin) for item in items)
    room = scheme.scene.room_m
    room_size = (room.Lx, room.Ly, room.Lz)
    amounts = [amount for box in boxes if (amount := _outside_m(box, room_size)) > margin]
    for first, second in combinations(boxes, 2):
        separation = _separation_m(first, second)
        if separation <= margin:
            amounts.append(max(margin, margin - separation))
    points = tuple(point.as_tuple() for point in scheme.speakers.values()) + tuple(
        receiver.position_m for receiver in scheme.receiver_set.points)
    amounts.extend(depth for box in boxes for point in points if (depth := _point_depth_m(point, box)) > margin)
    if not amounts:
        raise ValueError("家具配置不合法，但無法量出越界、間隙不足或點在盒內的違反量")
    return max(margin, *amounts)


def check(scheme: Scheme, *, contact_rel: float) -> tuple[Violation, ...]:
    """空 tuple＝家具擺放與全部直達合法；擺放錯先於遮擋，只回一碼及最大量。"""
    if scheme.furniture is None:
        return ()
    try:
        boxes = furniture_layout.furniture_boxes(scheme, contact_rel=contact_rel)
    except ValueError:
        room = scheme.scene.room_m
        margin = contact_margin_m((room.Lx, room.Ly, room.Lz), contact_rel=contact_rel)
        return (Violation(Reason.FURNITURE_PLACEMENT_INVALID, _placement_amount_m(scheme, margin)),)
    blocked = furniture_layout.direct_blockers(scheme, boxes)
    by_id = {item.furniture_id: item.box for item in boxes.furniture}
    receivers = {receiver.receiver_id: receiver.position_m for receiver in scheme.receiver_set.points}
    amounts = tuple(segment_box_length_m(scheme.speakers[speaker].as_tuple(), receivers[receiver], by_id[item])
                    for (speaker, receiver), ids in blocked.items() for item in ids)
    if not amounts:
        return ()
    amount = max(amounts)
    if not math.isfinite(amount) or amount <= 0.0:
        raise ValueError("直達判定被擋，但穿盒長度不是有限正公尺")
    return (Violation(Reason.DIRECT_PATH_BLOCKED, max(boxes.margin_m, amount)),)


def _furniture_prism(box: FurnitureBox) -> constraints._Prism:
    x, y, z = box.minimum_m
    xx, yy, zz = box.maximum_m
    return constraints._Prism(((x, y), (xx, y), (xx, yy), (x, yy)), z, zz)


def _contact_penetration(first: constraints._Prism, second: constraints._Prism, margin: float) -> float:
    """高度只作門檻；水平退出距離仍是原幾何量，不扣接觸界線。"""
    if min(first.high_z, second.high_z) - max(first.low_z, second.low_z) <= margin:
        return 0.0
    return constraints._penetration(first, second)


def _placement_hits(scheme: Scheme, cabinet_size: Cabinet, keep_out: tuple[Box, ...], *,
                    contact_rel: float) -> tuple[tuple[Reason, str, str | None, float], ...]:
    """同一份具名家具盒子；每支、每件的真實量供候選彙總與建檔定位（喇叭代號；禁區那類是家具自己，記 None）。"""
    boxes = furniture_layout.furniture_boxes(scheme, contact_rel=contact_rel)
    setup = scheme.speaker_setup
    table = next((item for item in boxes.furniture if item.box.kind in (
        FurnitureKind.DESK, FurnitureKind.COFFEE_TABLE)), None) if setup and setup.mount == "desk" else None
    primary = Point(*scheme.receiver_set.primary.position_m)
    cabinets = tuple((speaker_id, constraints._cabinet(point, primary, cabinet_size))
                     for speaker_id, point in scheme.speakers.items())
    if table is not None:
        # 桌面承托是結構條件；不從喇叭 z 減中心回推，也不再比兩份浮點高度。
        top = table.box.maximum_m[2]
        cabinets = tuple((speaker_id, constraints._Prism(item.polygon, top, top + cabinet_size.height_m))
                         for speaker_id, item in cabinets)
    moving = {item.furniture_id for item in scheme.furniture or () if isinstance(item.placement, ListenerPlacement)}
    hits: list[tuple[Reason, str, str | None, float]] = []
    for item in boxes.furniture:
        prism = _furniture_prism(item.box)
        for speaker_id, cabinet in cabinets:
            if item is not table:
                hits.append((Reason.CABINET_IN_FURNITURE, item.furniture_id, speaker_id,
                             _contact_penetration(cabinet, prism, boxes.margin_m)))
            if setup and setup.mount == "stand":
                column = constraints._Prism(cabinet.polygon, 0.0, cabinet.low_z)
                hits.append((Reason.STAND_SPACE_OCCUPIED, item.furniture_id, speaker_id,
                             _contact_penetration(column, prism, boxes.margin_m)))
            if item is table:
                x, y, _ = item.box.minimum_m
                xx, yy, _ = item.box.maximum_m
                amount = max(max(x - px, px - xx, y - py, py - yy, 0.0) for px, py in cabinet.polygon)
                hits.append((Reason.CABINET_OFF_TABLE, item.furniture_id, speaker_id, amount))
        if item.furniture_id in moving:
            hits.extend((Reason.FURNITURE_IN_KEEP_OUT, item.furniture_id, None,
                         _contact_penetration(prism, constraints._box_prism(region), boxes.margin_m))
                        for region in keep_out)
    return tuple(hit for hit in hits if hit[3] > boxes.margin_m)


def check_candidate(scheme: Scheme, settings: LayoutSettings, *, contact_rel: float) -> tuple[Violation, ...]:
    """只給搜尋候選：現有預篩之後補四條施工限制；原方案保留 check 的 B4 語意。"""
    if scheme.furniture is None:
        return ()
    amounts: dict[Reason, float] = {}
    for reason, _, _, amount in _placement_hits(scheme, settings.cabinet, settings.keep_out, contact_rel=contact_rel):
        amounts[reason] = max(amounts.get(reason, 0.0), amount)
    return tuple(Violation(reason, amounts[reason]) for reason in sorted(amounts))


PLACEMENT_NAMES: dict[Reason, str] = {Reason.CABINET_IN_FURNITURE: "箱體穿入家具", Reason.CABINET_OFF_TABLE: "箱體超出桌面",
                                      Reason.STAND_SPACE_OCCUPIED: "腳架下方有家具"}


@dataclass(frozen=True)
class PlacementHit:
    """方案自己的一條擺放問題：哪一條、哪件家具、哪支喇叭、差多少（公尺）。代號原樣，不拼進路徑再切。"""

    reason: Reason
    furniture_id: str
    speaker_id: str
    amount_m: float


def scheme_placement_hits(scheme: Scheme, *, contact_rel: float) -> tuple[PlacementHit, ...]:
    """只用方案自己的喇叭設定與家具，不需要搜尋設定；箱體一律用方案的，禁區是提案要求、不套原方案。

    建搜尋前核原方案與輸入頁存檔前提醒（#753）共用這一份。
    """
    setup = scheme.speaker_setup
    if setup is None:
        return ()
    cabinet = Cabinet.model_validate(setup.cabinet.model_dump())
    # 每支喇叭各一條、點名是哪一支（客戶現況准左右不對稱，量可能不同）。
    return tuple(PlacementHit(reason, identifier, speaker, amount)
                 for reason, identifier, speaker, amount in _placement_hits(scheme, cabinet, (), contact_rel=contact_rel)
                 if speaker is not None)


def original_placement_problems(scheme: Scheme, *, contact_rel: float) -> tuple[SchemeProblem, ...]:
    """建搜尋前只核現況物理可擺性；禁區是提案要求，不套原方案。"""
    return tuple(SchemeProblem(f"speakers.{hit.speaker_id}.furniture.{hit.furniture_id}.{hit.reason.value}",
        f"原方案喇叭 {hit.speaker_id} {PLACEMENT_NAMES[hit.reason]}：家具 {hit.furniture_id}，{hit.amount_m!r} m")
        for hit in scheme_placement_hits(scheme, contact_rel=contact_rel))
