"""輸入頁唯一的家具預設與快捷資料；只組方案，物理換算和驗證仍呼叫原零件。"""
from __future__ import annotations

from copy import deepcopy
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.furniture_materials import FURNITURE_MATERIAL_KINDS
from aosr.config.paths import config_path
from aosr.config.representative_speakers import load_representative_speakers
from aosr.gui.problem_text import plain_problems
from aosr.reporting.display import FURNITURE_FIELDS, FURNITURE_KINDS, FURNITURE_MATERIALS
from aosr.reporting.scheme import FurnitureSpec, Scheme, absolute_furniture, project_facing, project_midpoint
from aosr.reporting.validation import validate_scheme


# 尺寸與來源只住這裡；下列位置已是施工單指定的相對座標，不在瀏覽器推算。
FURNITURE_DEFAULTS: dict[str, dict[str, object]] = {
    "sofa": {"item": {"furniture_id": "sofa", "kind": "sofa", "material": "fabric",
                       "width_m": 1.80, "depth_m": 0.75, "height_m": 0.65,
                       "placement": {"forward_m": 0.075, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0}},
             "description": "尺寸：老闆 2026-10-10，沿用決策紙第 21 條；材質、貼地與位置：主對話估，耳朵在背面前 0.30 公尺。"},
    "coffee_table": {"item": {"furniture_id": "table", "kind": "coffee_table", "material": "wood",
                               "width_m": 1.20, "depth_m": 0.60, "height_m": 0.03,
                               "placement": {"forward_m": 1.15, "left_m": 0.0, "bottom_height_m": 0.47, "yaw_deg": 0}},
                     "description": "尺寸與頂面高 0.50 公尺：老闆 2026-10-10；板厚、木質與位置：主對話估，近邊離耳朵 0.85 公尺。"},
    "desk": {"item": {"furniture_id": "table", "kind": "desk", "material": "wood",
                       "width_m": 1.40, "depth_m": 0.75, "height_m": 0.03,
                       "placement": {"forward_m": 0.825, "left_m": 0.0, "bottom_height_m": 0.72, "yaw_deg": 0}},
             "description": "尺寸與頂面高 0.75 公尺：老闆 2026-10-10；板厚、木質與位置：主對話估，近邊離耳朵 0.45 公尺。"},
    "ceiling_cloud": {"item": {"furniture_id": "cloud", "kind": "ceiling_cloud", "material": "absorptive_cloud",
                                "width_m": 1.35, "depth_m": 1.80, "height_m": 0.05},
                      "description": "尺寸與底面在天花板下 0.15 公尺：老闆 2026-10-10；材質、中心與角度規則：主對話估，中心在兩喇叭水平中點與主位的中間，深邊順面向軸；只在勾選時算一次。"},
}
CLOUD_CEILING_GAP_M = 0.15
SHORTCUTS: dict[str, dict[str, object]] = {
    "working": {"label": "工作", "ear_height_m": 1.20, "sofa": False, "table": "desk"},
    "listening": {"label": "聆聽", "ear_height_m": 1.05, "sofa": True, "table": "coffee_table"},
}
# 角色、可編輯的種類與相對角；座椅及非零相對角無法由此表單完整表達，保留唯讀。
ROLES: dict[str, list[str]] = {"sofa": ["sofa"], "table": ["coffee_table", "desk"], "cloud": ["ceiling_cloud"]}
RELATIVE_YAW = 0
MOUNTS = {"bookshelf": ["stand", "desk"], "floorstanding": ["floor"]}
INPUT_SHAPE_ERROR = "方案格式不完整，請先填好房間、喇叭與座位"


class InputShapeError(ValueError):
    """表單使用者資料不完整或選項不合法；程式自身的錯誤不冒充這一種。"""


class InputEdit(BaseModel):
    """輸入頁一次填值請求；不在方案裡新增欄位。"""

    model_config = ConfigDict(extra="forbid")
    scheme: dict[str, object]
    action: Literal["furniture", "remove_furniture", "shortcut", "setup", "speaker_kind", "representative", "mount"]
    value: str | bool


def input_defaults() -> dict[str, object]:
    """唯讀端點：數字、准用材質、角色與擺法由伺服器給，中文名稱走名稱表。"""
    speakers = load_representative_speakers(config_path("representative_speakers.toml"))
    return {"furniture": deepcopy(FURNITURE_DEFAULTS), "shortcuts": deepcopy(SHORTCUTS),
            "roles": ROLES, "relative_yaw": RELATIVE_YAW, "mounts": MOUNTS,
            "materials": {kind: [material for material, kinds in FURNITURE_MATERIAL_KINDS.items() if kind in kinds]
                          for kind in FURNITURE_DEFAULTS},
            "representative_speakers": {kind: {"cabinet": getattr(speakers, kind).cabinet_fields_m()}
                                        for kind in MOUNTS}}


def editable_role(item: dict[str, object]) -> str | None:
    """只接住本頁能完整表達的固定角色；原件角度無法編輯時也不得抹成零。"""
    role = str(item.get("furniture_id"))
    if item.get("kind") not in ROLES.get(role, []):
        return None
    placement = cast(dict[str, object], item.get("placement", {}))
    return role if role == "cloud" or placement.get("yaw_deg") == RELATIVE_YAW else None


def _items(document: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], document.get("furniture") or [])


def _input_mapping(value: object, fields: tuple[str, ...]) -> dict[str, object]:
    if not isinstance(value, dict) or any(field not in value for field in fields):
        raise InputShapeError(INPUT_SHAPE_ERROR)
    return cast(dict[str, object], value)


def _check_furniture_shape(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise InputShapeError(INPUT_SHAPE_ERROR)
    for item in value:
        entry = _input_mapping(item, ("furniture_id", "kind", "material", "width_m", "depth_m", "height_m", "placement"))
        if any(not isinstance(entry[key], str) for key in ("furniture_id", "kind", "material")):
            raise InputShapeError(INPUT_SHAPE_ERROR)
        placement = _input_mapping(entry["placement"], ("yaw_deg",))
        if entry["kind"] == "ceiling_cloud":
            center = placement.get("bottom_center_m")
            if not isinstance(center, list | tuple) or len(center) != 3:
                raise InputShapeError(INPUT_SHAPE_ERROR)
        else:
            _input_mapping(placement, ("forward_m", "left_m", "bottom_height_m"))
        # 唯讀原件沒有半填格子；先核完整形狀，避免顯示字串時才噴型別或格式錯。
        if editable_role(entry) is None:
            FurnitureSpec.model_validate(entry)


def check_input_shape(document: dict[str, object]) -> None:
    """只核表單要讀寫的容器與欄位；半填的數字仍交正式驗證列原因。"""
    scene = _input_mapping(document.get("scene"), ("room_m", "impedance_pa_s_per_m_by_wall"))
    _input_mapping(scene["room_m"], ("Lx", "Ly", "Lz"))
    _input_mapping(scene["impedance_pa_s_per_m_by_wall"], ())
    if scene.get("scattering_by_wall") is not None:
        _input_mapping(scene["scattering_by_wall"], ())
    speakers = _input_mapping(document.get("speakers"), ())
    for point in speakers.values():
        _input_mapping(point, ("x", "y", "z"))
    receivers = _input_mapping(document.get("receiver_set"), ("points",))
    points = receivers["points"]
    if not isinstance(points, list):
        raise InputShapeError(INPUT_SHAPE_ERROR)
    has_primary = False
    for point in points:
        entry = _input_mapping(point, ("receiver_id", "role", "position_m"))
        has_primary = has_primary or entry["role"] == "primary"
        position = entry["position_m"]
        if not isinstance(position, list | tuple) or len(position) != 3:
            raise InputShapeError(INPUT_SHAPE_ERROR)
    if not has_primary:
        raise InputShapeError(INPUT_SHAPE_ERROR)
    _check_furniture_shape(document.get("furniture"))
    setup = document.get("speaker_setup")
    if setup is not None:
        entry = _input_mapping(setup, ("kind", "mount", "representative", "cabinet"))
        if not isinstance(entry["kind"], str) or entry["kind"] not in MOUNTS:
            raise InputShapeError(INPUT_SHAPE_ERROR)
        _input_mapping(entry["cabinet"], ())


def _has_table(document: dict[str, object]) -> bool:
    return any(item["kind"] in ROLES["table"] for item in _items(document))


def mount_options(document: dict[str, object], kind: str) -> list[str]:
    """全部茶几、書桌都能承托；多張原件桌面仍交原方案驗證拒收。"""
    has_table = _has_table(document)
    return [mount for mount in MOUNTS[kind] if mount != "desk" or has_table]


def _geometry_scheme(document: dict[str, object]) -> Scheme:
    # 箱體與家具格可能正在填（空白或高度不符），天雲定位只需既有房間、喇叭與座位。
    return Scheme.model_validate(document | {"furniture": None, "speaker_setup": None})


def default_furniture(kind: str, document: dict[str, object]) -> dict[str, object]:
    """天雲只在勾選當下按現況定位，往後原樣存房間座標。"""
    item = deepcopy(cast(dict[str, object], FURNITURE_DEFAULTS[kind]["item"]))
    if kind == "ceiling_cloud":
        scheme = _geometry_scheme(document)
        mx, my = project_midpoint(scheme)
        px, py, _ = scheme.receiver_set.primary.position_m
        try:
            fx, _ = project_facing(scheme)
        except ValueError as exc:
            raise InputShapeError(str(exc)) from exc
        item["placement"] = {"bottom_center_m": [(mx + px) / 2, (my + py) / 2,
                                                  scheme.scene.room_m.Lz - CLOUD_CEILING_GAP_M],
                             "yaw_deg": 90 if fx else 0}
    return item


def _replace_role(document: dict[str, object], role: str, item: dict[str, object] | None) -> None:
    original = _items(document)
    if any(old.get("furniture_id") == role and editable_role(old) != role for old in original):
        raise InputShapeError("同代號的原件這一頁不能改，請保留原方案")
    replaced = False
    result: list[dict[str, object]] = []
    for old in original:
        if editable_role(old) != role:
            result.append(old)
        elif item is not None:
            result.append(item)
            replaced = True
    if item is not None and not replaced:
        result.append(item)
    document["furniture"] = result or None


def _shortcut(document: dict[str, object], name: str) -> None:
    preset = SHORTCUTS[name]
    receivers = cast(dict[str, list[dict[str, object]]], document["receiver_set"])["points"]
    if not receivers or any(not isinstance(cast(list[object], point["position_m"])[2], int | float)
                            for point in receivers):
        raise InputShapeError("請先填好主位與周圍點的高度，再按快捷")
    primary = next(point for point in receivers if point["role"] == "primary")
    delta = cast(float, preset["ear_height_m"]) - cast(list[float], primary["position_m"])[2]
    for role, kind in (("sofa", "sofa" if preset["sofa"] else None), ("table", str(preset["table"]))):
        # 快捷只填能改的角色；同代號唯讀原件保持原樣，其他格子照樣填。
        if not any(item["furniture_id"] == role and editable_role(item) is None for item in _items(document)):
            _replace_role(document, role, default_furniture(kind, document) if kind else None)
    for point in receivers:
        position = cast(list[float], point["position_m"])
        point["position_m"] = [position[0], position[1], position[2] + delta]


def _speaker_edit(document: dict[str, object], action: str, value: str | bool) -> None:
    if action == "setup":
        if not value:
            document.pop("speaker_setup", None)
            return
        data = cast(dict[str, dict[str, object]], input_defaults()["representative_speakers"])
        document["speaker_setup"] = {"kind": "bookshelf", "mount": "stand", "representative": True,
                                     "cabinet": data["bookshelf"]["cabinet"]}
        return
    setup = cast(dict[str, object], document.get("speaker_setup"))
    if not setup:
        raise InputShapeError("先勾選喇叭設定")
    if action == "speaker_kind":
        if value not in MOUNTS:
            raise InputShapeError("喇叭類型只收書架或落地")
        setup["kind"] = value
        setup["mount"] = mount_options(document, str(value))[0]
    elif action == "representative":
        if not isinstance(value, bool):
            raise InputShapeError("代表模型要填真假值")
        setup["representative"] = value
    elif action == "mount":
        if value not in mount_options(document, str(setup["kind"])):
            raise InputShapeError("這種擺法不能選；放桌面要先選前方桌面，書架放腳架或桌面、落地放地面")
        setup["mount"] = value
    if action in ("speaker_kind", "representative") and setup["representative"]:
        models = cast(dict[str, dict[str, object]], input_defaults()["representative_speakers"])
        setup["cabinet"] = models[str(setup["kind"])]["cabinet"]


def edit_input(request: InputEdit) -> dict[str, object]:
    """只回一份填好的方案；不存檔、不改喇叭 z、不改不能編輯的原件。"""
    document = deepcopy(request.scheme)
    check_input_shape(document)
    had_table = _has_table(document)
    action, value = request.action, request.value
    if action == "furniture":
        if value not in FURNITURE_DEFAULTS:
            raise InputShapeError("這一頁只提供沙發、茶几、書桌與天雲")
        item = default_furniture(str(value), document)
        _replace_role(document, str(item["furniture_id"]), item)
    elif action == "remove_furniture":
        if value not in ROLES:
            raise InputShapeError("這一頁沒有這個家具角色")
        _replace_role(document, str(value), None)
    elif action == "shortcut":
        if value not in SHORTCUTS:
            raise InputShapeError("快捷只收工作或聆聽")
        _shortcut(document, str(value))
    else:
        _speaker_edit(document, action, value)
    setup = cast(dict[str, object] | None, document.get("speaker_setup"))
    if setup and setup["mount"] == "desk" and had_table and not _has_table(document):
        setup["mount"] = mount_options(document, str(setup["kind"]))[0]
    return document


def _readonly_text(item: dict[str, object]) -> str:
    name = FURNITURE_KINDS.get(str(item["kind"]), "家具")
    material = FURNITURE_MATERIALS.get(str(item["material"]), "未識別材質")
    dimensions = "、".join(f"{FURNITURE_FIELDS[key][0]} {float(cast(float, item[key])):.12g} 公尺"
                          for key in ("width_m", "depth_m", "height_m"))
    placement = cast(dict[str, object], item["placement"])
    if "bottom_center_m" in placement:
        coords = "、".join(f"{axis} {n:.12g}" for axis, n in zip("xyz", cast(list[float], placement["bottom_center_m"]), strict=True))
        position = f"房間底面中心 {coords} 公尺"
    else:
        position = "、".join(f"{FURNITURE_FIELDS[key][0]} {cast(float, placement[key]):.12g} 公尺"
                            for key in ("forward_m", "left_m", "bottom_height_m"))
    return f"{name}（{item['furniture_id']}）：{material}；{dimensions}；{position}；角度 {placement['yaw_deg']} 度。這一頁不能改，存檔原樣保留。"


def input_field_values(document: dict[str, object]) -> dict[str, dict[str, object]]:
    """數字格的顯示字串由伺服器格式化；另外帶原值，未手改時存檔不因顯示捨入失真。"""
    values: dict[str, object] = {}
    scene = cast(dict[str, object], document.get("scene", {}))
    for name, prefix in (("room_m", "room"), ("impedance_pa_s_per_m_by_wall", "wall"), ("scattering_by_wall", "scatter")):
        for key, value in cast(dict[str, object], scene.get(name) or {}).items():
            values[f"{prefix}-{key}"] = value
    for identifier, point in cast(dict[str, dict[str, object]], document.get("speakers", {})).items():
        for axis, value in point.items():
            values[f"speaker-{identifier}-{axis}"] = value
    receivers = cast(dict[str, object], document.get("receiver_set", {}))
    for point in cast(list[dict[str, object]], receivers.get("points", [])):
        for axis, value in zip("xyz", cast(list[object], point["position_m"]), strict=True):
            values[f"receiver-{point['receiver_id']}-{axis}"] = value
    for item in _items(document):
        role = editable_role(item)
        if role is None:
            continue
        for key in ("width_m", "depth_m", "height_m"):
            values[f"furniture-{role}-{key}"] = item[key]
        placement = cast(dict[str, object], item["placement"])
        if role == "cloud":
            for axis, value in zip("xyz", cast(list[object], placement["bottom_center_m"]), strict=True):
                values[f"furniture-{role}-{axis}"] = value
        else:
            for key in ("forward_m", "left_m", "bottom_height_m"):
                values[f"furniture-{role}-{key}"] = placement[key]
    setup = cast(dict[str, object] | None, document.get("speaker_setup"))
    if setup:
        for key, value in cast(dict[str, object], setup["cabinet"]).items():
            values[f"cabinet-{key}"] = value
    return {key: {"value": value, "text": f"{value:.12g}" if isinstance(value, int | float) else ""}
            for key, value in values.items()}


def input_preview(document: dict[str, object], capabilities: CapabilityTable,
                  directivity: DirectivityDefaults) -> dict[str, object]:
    """房間座標只呼叫 absolute_furniture；高度訊息直接用原驗證回的中文句。"""
    check_input_shape(document)
    items = _items(document)
    furniture_items = [{"furniture_id": item["furniture_id"], "role": editable_role(item),
                        "editable": editable_role(item) is not None} for item in items]
    controls = {role: not any(item["furniture_id"] == role and editable_role(item) is None for item in items)
                for role in ROLES}
    readonly = [{"item": item, "text": _readonly_text(item)} for item in items if editable_role(item) is None]
    setup = cast(dict[str, object] | None, document.get("speaker_setup"))
    problems = validate_scheme(document, capabilities=capabilities, directivity=directivity)
    heights: dict[str, str] = {}
    for problem in problems:
        if "跟擺法推出值不同" in problem.message:
            # 逐支保留完整驗證句（不用合併後的路徑猜喇叭代號，代號可含點）。
            for identifier in cast(dict[str, object], document["speakers"]):
                if problem.path == f"speakers.{identifier}.z":
                    heights[identifier] = str(plain_problems((problem,), document)[0]["text"])
    coordinates: dict[str, str] = {}
    hint = ""
    try:
        scheme = Scheme.model_validate(document)
        for item in absolute_furniture(scheme) or ():
            x, y, z = item.bottom_center_m
            coordinates[item.furniture_id] = f"房間底面中心：x {x:.12g}、y {y:.12g}、z {z:.12g} 公尺"
        if setup and setup["mount"] == "stand" and len({point.z for point in scheme.speakers.values()}) > 1:
            hint = "搜尋只用一個高度，兩支要一樣"
    except ValueError:
        # 半填表單由正式檢查列原因，座標清掉，避免舊座標配新格子。
        pass
    return {"readonly": readonly, "coordinates": coordinates, "height_problems": heights,
            "furniture_items": furniture_items, "furniture_controls": controls,
            "field_values": input_field_values(document),
            "stand_hint": hint, "mounts": mount_options(document, str(setup["kind"])) if setup else []}
