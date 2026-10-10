"""座位鎖定、喇叭放桌面與型號適用聆聽距離的建檔前核對；只在搜尋層拒收整場不可能成立的輸入。"""
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.geometry.furniture import FurnitureKind, contact_margin_m
from aosr.geometry.shoebox import Point, Wall, distance
from aosr.reporting.furniture_layout import furniture_boxes
from aosr.reporting.scheme import ListenerPlacement, Scheme, project_facing
from aosr.reporting.validation import SchemeProblem, SchemeValidationError
from aosr.search import constraints, furniture_prefilter, layout
from aosr.search.layout import LayoutParams, derived_listening_distance
from aosr.search.layout_settings import Cabinet, LayoutSettings


def front_wall_of(project: Scheme) -> str:
    """原方案面向那面牆的名字；鎖定時前牆一定要是它。"""
    facing = project_facing(project)
    return {(-1.0, 0.0): "x0", (1.0, 0.0): "xL", (0.0, -1.0): "y0", (0.0, 1.0): "yL"}[facing]


def _geometric_problems(project: Scheme, settings: LayoutSettings, margin: float) -> tuple[SchemeProblem, ...]:
    p = project.receiver_set.primary.position_m
    expected = front_wall_of(project)
    wall_ok = settings.front_wall == expected
    problems = []
    if not wall_ok:
        # 中軸、離前牆上限、推出範圍都要用前牆算；牆錯時那幾條不報，免得印出用錯牆算的值把人帶去改錯欄位。
        problems.append(SchemeProblem("settings.layout.front_wall",
            f"座位鎖定時前牆要是原方案面向的 {expected}；設定寫 {settings.front_wall}"))
    wall = Wall.from_name(settings.front_wall)
    axis = wall.axis()
    half = project.scene.room_m.length(1 - axis) / 2.0
    across = p[1 - axis]
    offset = settings.axis_offset_m
    if wall_ok and abs(half + offset - across) > margin:
        label = "y" if axis == 0 else "x"
        problems.append(SchemeProblem("settings.layout.axis_offset_m",
            f"座位鎖定時中軸要通過主位：主位{label} {across!r} m，推出偏移 {across - half!r} m；設定寫 {offset!r} m"))
    # 耳高與「不准寫聆聽距離範圍」跟前牆無關，牆錯也照報。
    if settings.ear_height_m != p[2]:
        problems.append(SchemeProblem("settings.layout.ear_height_m",
            f"座位鎖定時耳高要等於主位高度 {p[2]!r} m；設定寫 {settings.ear_height_m!r} m"))
    front = settings.front_distance_m
    low = derived_listening_distance(project, settings, front.high)
    high = derived_listening_distance(project, settings, front.low)
    if settings.listening_distance_m is not None:
        derived = f"，範圍 {low!r}～{high!r} m" if wall_ok else ""
        problems.append(SchemeProblem("settings.layout.listening_distance_m",
            f"座位鎖定時不准寫聆聽距離範圍；聆聽距離由座位推出{derived}"))
    if wall_ok and low <= 0.0:
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


def _always_out(cabinet: Cabinet) -> tuple[float, float, float]:
    """箱體朝主位時一定伸出聲學中心的量：往牆那側、往聽者那側、左右外側（兩支都朝內，箱背的角一定往外）。"""
    behind = cabinet.depth_m - cabinet.acoustic_center_behind_front_m
    toward_wall = min(behind, cabinet.width_m / 2.0)
    return toward_wall, min(cabinet.acoustic_center_behind_front_m, cabinet.width_m / 2.0), toward_wall


def _desk_possible(project: Scheme, settings: LayoutSettings, *, contact_rel: float) -> bool:
    """必要條件：箱體朝主位時一定伸出聲學中心的那段先從桌面頂扣掉，再看聲學中心有沒有落點。

    喇叭朝主位，箱背一定朝牆：往牆那側至少伸出 min(箱深−聲學中心離前面板, 箱寬/2)，
    往聽者那側至少 min(聲學中心離前面板, 箱寬/2)；座位鎖定時喇叭中點橫向對齊主位，兩支都朝內，
    箱背的角一定往外，左右外側也至少 min(箱深−聲學中心離前面板, 箱寬/2)。只扣一定伸出的量，不會誤擋可行的設定；
    伸出桌緣不超過接觸界線不算超出，跟候選預篩「箱體超出桌面」同一把界線。
    """
    if project.speaker_setup is None or project.speaker_setup.mount != "desk":
        return True
    boxes = furniture_boxes(project, contact_rel=contact_rel)
    table = next(item.box for item in boxes.furniture if item.box.kind in (FurnitureKind.DESK, FurnitureKind.COFFEE_TABLE))
    wall = Wall.from_name(settings.front_wall)
    axis = wall.axis()
    inward = 1.0 if wall.kind() == "zero" else -1.0
    toward_wall, toward_listener, sideways = _always_out(settings.cabinet)
    low_margin, high_margin = (toward_wall, toward_listener) if inward > 0.0 else (toward_listener, toward_wall)
    contact = boxes.margin_m
    front = settings.front_distance_m
    along = sorted(wall.plane(project.scene.room_m) + inward * value for value in (front.low, front.high))
    if max(along[0], table.minimum_m[axis] + low_margin - contact) > min(along[1], table.maximum_m[axis] - high_margin + contact):
        return False
    across = project.receiver_set.primary.position_m[1 - axis]
    maximum = min(settings.spacing_m.high, 2.0 * (across - table.minimum_m[1 - axis] - sideways + contact),
                  2.0 * (table.maximum_m[1 - axis] - sideways - across + contact))
    return settings.spacing_m.low <= maximum


DESK_REACH_NOTE = "（只扣了箱體朝主位時一定伸出聲學中心的部分；範圍內也不一定都放得上）"


def _desk_reach_problems(project: Scheme, settings: LayoutSettings, *, contact_rel: float) -> tuple[SchemeProblem, ...]:
    """沒鎖定時的必要條件：承托的桌子跟著主位走，桌面相對主位固定；聲學中心在主位正前方聆聽距離、左右各半個間距。

    一定伸出的量與接觸界線照 _desk_possible，聆聽距離與間距各自判，只擋一定全滅的範圍；
    訊息寫的是必要範圍，範圍內也不一定都放得上。書桌與茶几一定跟著主位（方案驗證拒收釘在房間裡的），所以桌面相對主位的位置跟離前牆無關。
    """
    boxes = furniture_boxes(project, contact_rel=contact_rel)
    table = next(item for item in boxes.furniture if item.box.kind in (FurnitureKind.DESK, FurnitureKind.COFFEE_TABLE))
    fx, fy = project_facing(project)
    px, py, _ = project.receiver_set.primary.position_m
    corners = tuple((x, y) for x in (table.box.minimum_m[0], table.box.maximum_m[0])
                    for y in (table.box.minimum_m[1], table.box.maximum_m[1]))
    # 主位座標系：前方 f、左方 (−f_y, f_x)；候選換前牆時桌子跟著轉，這兩個量不變。
    ahead = tuple((x - px) * fx + (y - py) * fy for x, y in corners)
    left = tuple((y - py) * fx - (x - px) * fy for x, y in corners)
    toward_wall, toward_listener, sideways = _always_out(settings.cabinet)
    near, far = min(ahead) + toward_listener, max(ahead) - toward_wall
    widest = 2.0 * min(max(left) - sideways, -min(left) - sideways)
    contact = boxes.margin_m  # 每支箱體的角可以伸出桌緣到這把界線；間距兩支各算一次。
    listening = settings.listening_distance_m
    assert listening is not None  # 設定驗證已保證未鎖定時必填。
    problems = []
    if max(listening.low, near - contact) > min(listening.high, far + contact):
        reach = (f"聲學中心至少要落在主位前方 {near!r}～{far!r} m 之內" if near <= far
                 else "桌面前後深度扣掉之後放不下聲學中心")
        problems.append(SchemeProblem("settings.layout.listening_distance_m",
            f"喇叭放桌面時聆聽距離範圍 {listening.low!r}～{listening.high!r} m 內，箱體都放不上桌面頂："
            f"桌子跟著主位走，{reach}" + DESK_REACH_NOTE))
    if settings.spacing_m.low > widest + 2.0 * contact:
        problems.append(SchemeProblem("settings.layout.spacing_m",
            f"喇叭放桌面時間距下限 {settings.spacing_m.low!r} m 超過桌面最多放得下的 {widest!r} m" + DESK_REACH_NOTE))
    return tuple(problems)


def check_desk_reach(project: Scheme, settings: LayoutSettings) -> None:
    """建目錄之前才呼叫；座位鎖定時桌面由 check_seat_lock 判，沒放桌面的不讀登記簿。"""
    if settings.seat_locked or project.speaker_setup is None or project.speaker_setup.mount != "desk":
        return
    try:
        problems = _desk_reach_problems(project, settings,
                                        contact_rel=furniture_contact_rel(default_precision_contracts_path()))
    except ValueError as error:
        raise SchemeValidationError((SchemeProblem("settings.layout", str(error)),)) from error
    if problems:
        raise SchemeValidationError(problems)


def _reach(project: Scheme, settings: LayoutSettings, params: LayoutParams) -> tuple[float, float]:
    """照候選同一條算式擺一次，回兩支喇叭聲學中心到主位三維距離的最小與最大。"""
    placement = layout.place(project, settings, params)
    values = [distance(center, placement.primary) for center in (placement.left, placement.right)]
    return min(values), max(values)


def check_listening_reach(project: Scheme, settings: LayoutSettings) -> None:
    """型號適用聆聽距離（喇叭聲學中心到主位的三維距離）跟搜尋範圍碰不到，就在建目錄之前拒收（#765）。

    三維距離對聆聽距離、間距都只會變大（座位鎖定時聆聽距離隨離前牆變小），所以範圍兩端各擺一次：
    最遠那端仍不到下限、或最近那端已超過上限，整場一定全滅。比的是兩端實際擺出來的值，剛好等於界線也放行。
    """
    limits = settings.listening_range_m
    if limits is None:
        return
    front, spacing = settings.front_distance_m, settings.spacing_m
    if settings.seat_locked:
        far = LayoutParams(front.low, spacing.high, derived_listening_distance(project, settings, front.low))
        near = LayoutParams(front.high, spacing.low, derived_listening_distance(project, settings, front.high))
    else:
        listening = settings.listening_distance_m
        assert listening is not None  # 設定驗證已保證未鎖定時必填。
        far = LayoutParams(front.low, spacing.high, listening.high)
        near = LayoutParams(front.low, spacing.low, listening.low)
    nearest, farthest = _reach(project, settings, near)[0], _reach(project, settings, far)[1]
    if farthest < limits.low or nearest > limits.high:
        raise SchemeValidationError((SchemeProblem("settings.layout.listening_range_m",
            f"型號適用聆聽距離 {limits.low!r}～{limits.high!r} m（喇叭聲學中心到主位的三維距離）跟搜尋範圍碰不到："
            f"範圍兩端擺出來的三維距離只有 {nearest!r}～{farthest!r} m"),))


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
        # 桌面判準也要用前牆算；牆錯時只報前牆那條。
        if settings.front_wall == front_wall_of(project) and not _desk_possible(project, settings, contact_rel=contact_rel):
            problems = (*problems, SchemeProblem("settings.layout",
                "座位鎖定時離前牆與間距範圍內，沒有任何一組能讓兩支喇叭的箱體放上桌面頂（箱體朝主位時一定伸出聲學中心的部分已扣掉）"))
    except ValueError as error:
        raise SchemeValidationError((SchemeProblem("settings.layout.seat_locked", str(error)),)) from error
    if problems:
        raise SchemeValidationError(problems)
