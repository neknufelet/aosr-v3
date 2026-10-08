"""遞迴檢查 Scheme 每個欄位；連動欄位逐名說明，不靠空清單保險冒充逐項比較。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import TypeAlias, cast, get_args, get_origin

from pydantic import BaseModel

from aosr.gui.compare_view import scheme_differences
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._furniture_scheme_results import scheme_pair as scheme_pair
from tests.engine._speaker_setup_cases import setup


Json: TypeAlias = None | bool | str | int | float | list["Json"] | dict[str, "Json"]


EXEMPTIONS = {
    "schema_version": "Literal 固定版號，方案驗證器拒收其他值。",
    "scheme_id": "施工單明定排除方案代號；它不是方案設定。",
    "channel_group.channels[].role": "恰好兩個聲道且比較對引用角色，須連動修改。",
    "channel_group.channels[].speaker_id": "恰好兩支喇叭且不能未使用，須連動修改。",
    "channel_group.comparisons[].left_role": "比較對須引用存在的兩個聲道角色，須連動修改。",
    "channel_group.comparisons[].right_role": "比較對須引用存在的兩個聲道角色，須連動修改。",
    "receiver_set.points[].role": "唯一主位與至少一個周圍點不能單獨換角色，須連動修改。",
}


def _leaves(annotation: object, prefix: str = "") -> set[str]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return set().union(*(_leaves(field.annotation, f"{prefix}.{name}" if prefix else name)
                              for name, field in annotation.model_fields.items()))
    if isinstance(annotation, type) and is_dataclass(annotation):
        return {f"{prefix}.{field.name}" for field in fields(annotation)}
    args = get_args(annotation)
    child_prefix = prefix + ("[]" if get_origin(annotation) in (list, tuple, dict) else "")
    nested = set().union(*(_leaves(arg, child_prefix)
                           for arg in args)) if args else set()
    descendants = nested - {prefix, child_prefix}
    return descendants or {prefix}


def _child(value: Json, key: str | int) -> Json:
    if isinstance(value, dict) and isinstance(key, str):
        return value[key]
    if isinstance(value, list) and isinstance(key, int):
        return value[key]
    raise TypeError("考卷的欄位路徑與資料形狀不符")


def _change(document: dict[str, Json], path: tuple[str | int, ...], value: Json) -> Scheme:
    changed = deepcopy(document)
    cursor: Json = changed
    for key in path[:-1]:
        cursor = _child(cursor, key)
    key = path[-1]
    if isinstance(cursor, dict) and isinstance(key, str):
        cursor[key] = value
    elif isinstance(cursor, list) and isinstance(key, int):
        cursor[key] = value
    else:
        raise TypeError("考卷的欄位路徑與資料形狀不符")
    return Scheme.model_validate(changed)


def test_every_scheme_model_field_is_compared_or_has_named_reason(scheme_pair: tuple[SchemeResult, ...]) -> None:
    scheme = scheme_pair[1].scheme
    document = cast(dict[str, Json], scheme.model_dump(mode="json"))
    seen = set()
    cases: dict[str, tuple[tuple[str | int, ...], Json]] = {
        "purpose": (("purpose",), "other_purpose"),
        "source_model": (("source_model",), "omnidirectional"),
        "scene.sound_speed_m_s": (("scene", "sound_speed_m_s"), 344.0),
        "scene.density_kg_m3": (("scene", "density_kg_m3"), 1.3),
        "scene.reflection_order_k": (("scene", "reflection_order_k"), 4),
        "scene.low_frequency_axis": (("scene", "low_frequency_axis"), "verification_linear_1hz"),
        "scene.impedance_pa_s_per_m_by_wall": (("scene", "impedance_pa_s_per_m_by_wall", "x0"), 5000.0),
        "scene.scattering_by_wall": (("scene", "scattering_by_wall"), {wall: 0.2 for wall in ("x0", "xL", "y0", "yL", "floor", "ceiling")}),
        "channel_group.feature_match_tolerance_hz": (("channel_group", "feature_match_tolerance_hz"), 2.0),
        "receiver_set.points[].receiver_id": (("receiver_set", "points", 1, "receiver_id"), "side-new"),
        "receiver_set.points[].position_m": (("receiver_set", "points", 1, "position_m"), [3.12, 4.0, 1.2]),
        "receiver_set.points[].importance": (("receiver_set", "points", 1, "importance"), 2.0),
        "receiver_set.points[].direction_relative_to_primary": (("receiver_set", "points", 1, "direction_relative_to_primary"), "right"),
        "furniture[].furniture_id": (("furniture", 0, "furniture_id"), "seat-new"),
        "furniture[].kind": (("furniture", 0, "kind"), "chair"),
        "furniture[].material": (("furniture", 0, "material"), "leather"),
    }
    for name, (path, value) in cases.items():
        changes = scheme_differences(scheme, _change(document, path, value))
        assert changes and all(row.path != "other_settings" for row in changes), name
        seen.add(name)
    seen.update(_numeric_fields(scheme, document))
    seen.update(_cloud_fields(scheme))
    seen.update(_speaker_fields(scheme))
    assert _leaves(Scheme) == seen | EXEMPTIONS.keys()
    assert all(EXEMPTIONS.values())


def _speaker_fields(scheme: Scheme) -> set[str]:
    document = cast(dict[str, Json], scheme.model_dump(mode="json") | {"speaker_setup": setup()})
    original = Scheme.model_validate(document)
    cases: dict[str, Json] = {"kind": "floorstanding", "mount": "desk", "representative": False,
                              "cabinet.width_m": 0.22, "cabinet.depth_m": 0.29, "cabinet.height_m": 0.36,
                              "cabinet.acoustic_center_behind_front_m": 0.01,
                              "cabinet.acoustic_center_above_bottom_m": 0.21}
    for name, value in cases.items():
        if name in ("kind", "mount"):
            # 類型與擺法須連動；桌面還要一張桌子。逐名證明各欄位有自己的差異列。
            changed = cast(dict[str, Json], scheme.model_dump(mode="json") |
                           {"speaker_setup": setup("floorstanding", "floor")})
            if name == "mount":
                changed = cast(dict[str, Json], scheme.model_dump(mode="json") |
                               {"speaker_setup": setup("bookshelf", "desk")})
                changed["furniture"] = [cast(Json, {"furniture_id": "table", "kind": "desk", "material": "wood",
                    "width_m": 1.0, "depth_m": 0.5, "height_m": 0.03,
                    "placement": {"forward_m": 1.0, "left_m": 0.0, "bottom_height_m": 0.7, "yaw_deg": 0}})]
            other = Scheme.model_validate(changed)
        else:
            other = _change(document, ("speaker_setup", *name.split(".")), value)
        rows = {row.path: row for row in scheme_differences(original, other)}
        assert f"speaker_setup.{name}" in rows
        assert "other_settings" not in rows
    return {f"speaker_setup.{name}" for name in cases}


def _numeric_fields(scheme: Scheme, document: dict[str, Json]) -> set[str]:
    paths: list[tuple[str, tuple[str | int, ...]]] = [(f"scene.room_m.{axis}", ("scene", "room_m", axis)) for axis in ("Lx", "Ly", "Lz")]
    paths += [(f"speakers[].{axis}", ("speakers", "left", axis)) for axis in ("x", "y", "z")]
    paths += [(f"furniture[].{field}", ("furniture", 0, field)) for field in ("width_m", "depth_m", "height_m")]
    paths += [(f"furniture[].placement.{field}", ("furniture", 0, "placement", field))
              for field in ("forward_m", "left_m", "bottom_height_m", "yaw_deg")]
    for name, path in paths:
        value: Json = document
        for key in path:
            value = _child(value, key)
        assert isinstance(value, float | int)
        changed = _change(document, path, value + (90 if path[-1] == "yaw_deg" else 0.01))
        changes = scheme_differences(scheme, changed)
        assert changes and all(row.path != "other_settings" for row in changes), name
    return {name for name, _ in paths}


def _cloud_fields(scheme: Scheme) -> set[str]:
    document = cast(dict[str, Json], scheme.model_dump(mode="json"))
    item = _child(_child(document, "furniture"), 0)
    assert isinstance(item, dict)
    item.update(kind="ceiling_cloud", material="wood", placement={"bottom_center_m": [2.0, 3.0, 2.0], "yaw_deg": 0})
    cloud = Scheme.model_validate(document)
    cases: tuple[tuple[str, Json], ...] = (("bottom_center_m", [2.1, 3.0, 2.0]), ("yaw_deg", 90))
    for field, value in cases:
        changes = scheme_differences(cloud, _change(document, ("furniture", 0, "placement", field), value))
        assert {row.path for row in changes} == {f"furniture.seat.placement.{field}"}
    return {"furniture[].placement.bottom_center_m", "furniture[].placement.yaw_deg"}
