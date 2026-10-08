"""座位鎖定的建檔前核對；只在搜尋層拒收整場不可能成立的輸入。"""
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.geometry.furniture import FurnitureKind, contact_margin_m
from aosr.geometry.shoebox import Point, Wall
from aosr.reporting.furniture_layout import furniture_boxes
from aosr.reporting.scheme import ListenerPlacement, Scheme, project_facing
from aosr.reporting.validation import SchemeProblem, SchemeValidationError
from aosr.search import constraints, furniture_prefilter
from aosr.search.layout import derived_listening_distance
from aosr.search.layout_settings import LayoutSettings


def _geometric_problems(project: Scheme, settings: LayoutSettings, margin: float) -> tuple[SchemeProblem, ...]:
    wall = Wall.from_name(settings.front_wall)
    axis = wall.axis()
    p = project.receiver_set.primary.position_m
    facing = project_facing(project)
    expected = {(-1.0, 0.0): "x0", (1.0, 0.0): "xL", (0.0, -1.0): "y0", (0.0, 1.0): "yL"}[facing]
    problems = []
    if settings.front_wall != expected:
        problems.append(SchemeProblem("settings.layout.front_wall",
            f"座位鎖定時前牆要是原方案面向的 {expected}；設定寫 {settings.front_wall}"))
    half = project.scene.room_m.length(1 - axis) / 2.0
    across = p[1 - axis]
    offset = settings.axis_offset_m
    if abs(half + offset - across) > margin:
        label = "y" if axis == 0 else "x"
        problems.append(SchemeProblem("settings.layout.axis_offset_m",
            f"座位鎖定時中軸要通過主位：主位{label} {across!r} m，推出偏移 {across - half!r} m；設定寫 {offset!r} m"))
    if settings.ear_height_m != p[2]:
        problems.append(SchemeProblem("settings.layout.ear_height_m",
            f"座位鎖定時耳高要等於主位高度 {p[2]!r} m；設定寫 {settings.ear_height_m!r} m"))
    front = settings.front_distance_m
    low = derived_listening_distance(project, settings, front.high)
    high = derived_listening_distance(project, settings, front.low)
    if settings.listening_distance_m is not None:
        problems.append(SchemeProblem("settings.layout.listening_distance_m",
            f"座位鎖定時不准寫聆聽距離範圍；聆聽距離由座位推出，範圍 {low!r}～{high!r} m"))
    if low <= 0.0:
        distance = p[axis] if wall.kind() == "zero" else wall.plane(project.scene.room_m) - p[axis]
        problems.append(SchemeProblem("settings.layout.front_distance_m",
            f"座位鎖定時離前牆上限 {front.high!r} m 要小於主位到前牆的距離 {distance!r} m（聆聽距離由座位推出，要永遠為正）"))
    return tuple(problems)


def _keep_out_problems(project: Scheme, settings: LayoutSettings, *, contact_rel: float) -> tuple[SchemeProblem, ...]:
    problems = []
    for receiver in project.receiver_set.points:
        point = Point(*receiver.position_m)
        for region in settings.keep_out:
            amount = constraints._seat_penetration(point, region)
            if amount > 0.0:
                problems.append(SchemeProblem(f"receiver_set.{receiver.receiver_id}",
                    f"座位鎖定時座位 {receiver.receiver_id} 在門或走道禁區內：{amount!r} m"))
    if project.furniture is not None:
        boxes = furniture_boxes(project, contact_rel=contact_rel)
        moving = {item.furniture_id for item in project.furniture if isinstance(item.placement, ListenerPlacement)}
        for item in boxes.furniture:
            if item.furniture_id not in moving:
                continue
            prism = furniture_prefilter._furniture_prism(item.box)
            for region in settings.keep_out:
                amount = furniture_prefilter._contact_penetration(prism, constraints._box_prism(region), boxes.margin_m)
                if amount > boxes.margin_m:
                    problems.append(SchemeProblem(f"furniture.{item.furniture_id}",
                        f"座位鎖定時家具 {item.furniture_id} 在門或走道禁區內：{amount!r} m"))
    return tuple(problems)


def _desk_possible(project: Scheme, settings: LayoutSettings, *, contact_rel: float) -> bool:
    """只核聲學中心的桌頂矩形交集；箱體足跡與其他限制仍逐候選判。"""
    if project.speaker_setup is None or project.speaker_setup.mount != "desk":
        return True
    boxes = furniture_boxes(project, contact_rel=contact_rel)
    table = next(item.box for item in boxes.furniture if item.box.kind in (FurnitureKind.DESK, FurnitureKind.COFFEE_TABLE))
    wall = Wall.from_name(settings.front_wall)
    axis = wall.axis()
    inward = 1.0 if wall.kind() == "zero" else -1.0
    front = settings.front_distance_m
    along = sorted(wall.plane(project.scene.room_m) + inward * value for value in (front.low, front.high))
    if max(along[0], table.minimum_m[axis]) > min(along[1], table.maximum_m[axis]):
        return False
    across = project.receiver_set.primary.position_m[1 - axis]
    maximum = min(settings.spacing_m.high, 2.0 * (across - table.minimum_m[1 - axis]),
                  2.0 * (table.maximum_m[1 - axis] - across))
    return settings.spacing_m.low <= maximum


def check_seat_lock(project: Scheme, settings: LayoutSettings) -> None:
    """建目錄之前才呼叫；open 不重驗新限制。"""
    if not settings.seat_locked:
        return
    contact_rel = furniture_contact_rel(default_precision_contracts_path())
    room = project.scene.room_m
    margin = contact_margin_m((room.Lx, room.Ly, room.Lz), contact_rel=contact_rel)
    try:
        problems = (*_geometric_problems(project, settings, margin),
                    *_keep_out_problems(project, settings, contact_rel=contact_rel))
        if not _desk_possible(project, settings, contact_rel=contact_rel):
            problems = (*problems, SchemeProblem("settings.layout",
                "座位鎖定時離前牆與間距範圍內，沒有任何一組能讓兩支喇叭的聲學中心落在桌面頂矩形上方"))
    except ValueError as error:
        raise SchemeValidationError((SchemeProblem("settings.layout.seat_locked", str(error)),)) from error
    if problems:
        raise SchemeValidationError(problems)
