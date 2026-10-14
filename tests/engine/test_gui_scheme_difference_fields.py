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
    "scene.surface_zones_by_wall[][].area": "Literal 只收 whole_wall，其他分區都由三入口拒收考卷擋下。",
    "scene.surface_zones_by_wall[][].material.fill_rule": "Literal 只收第一版補值規則；實測格式考卷拒收其他值，預設值隨方案保存。",
    "scene.surface_zones_by_wall[][].material.base_assembly.kind": "底構造只收 Assembly；實測格式考卷拒收固定阻抗與巢狀實測表。",
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
    seen.update(_surface_fields(scheme))
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


def _surface_fields(scheme: Scheme) -> set[str]:
    from tests.engine._surface_zone_cases import NEW, OLD, zone_scene

    document = cast(dict[str, Json], scheme.model_dump(mode="json"))
    scene = cast(dict[str, Json], document["scene"])
    scene.pop(OLD)
    scene[NEW] = cast(Json, zone_scene(mixed=True)[NEW])
    original = Scheme.model_validate(document)
    cases: dict[str, tuple[tuple[str | int, ...], Json]] = {
        "backing": (("floor", 0, "material", "backing"), "air"),
        "layers[].thickness_m": (("floor", 0, "material", "layers", 1, "thickness_m"), 0.2),
        "layers[].surface_mass_kg_m2": (("floor", 0, "material", "layers", 0, "surface_mass_kg_m2"), 0.3),
        "layers[].resistance_pa_s_m": (("floor", 0, "material", "layers", 0, "resistance_pa_s_m"), 90.0),
        "impedance_pa_s_m": (("ceiling", 0, "material", "impedance_pa_s_m"), 2000.0),
        "kind": (("floor", 0, "material"), {"kind": "constant_impedance", "impedance_pa_s_m": 2000.0}),
        "layers[].kind": (("floor", 0, "material", "layers", 0), {"kind": "air", "thickness_m": 0.1}),
    }
    for name, (path, value) in cases.items():
        changes = scheme_differences(original, _change(document, ("scene", NEW, *path), value))
        assert changes and all(row.path != "other_settings" for row in changes), name
    porous: Json = {"kind": "porous", "thickness_m": 0.1, "flow_resistivity_pa_s_m2": 10000.0,
                    "model": "wilson_2015", "material_class": "fibrous", "porosity": 0.98, "tortuosity": 1.1}
    porous_scheme = _change(document, ("scene", NEW, "floor", 0, "material", "layers", 0), porous)
    porous_document = cast(dict[str, Json], porous_scheme.model_dump(mode="json"))
    parameters: dict[str, Json] = {"flow_resistivity_pa_s_m2": 12000.0, "material_class": "foam",
                                  "porosity": 0.97, "tortuosity": 1.2}
    for name, value in parameters.items():
        changed = _change(porous_document, ("scene", NEW, "floor", 0, "material", "layers", 0, name), value)
        assert {row.path for row in scheme_differences(porous_scheme, changed)} == {f"scene.{NEW}.floor"}
    miki: Json = {"kind": "porous", "thickness_m": 0.1, "flow_resistivity_pa_s_m2": 10000.0,
                  "model": "miki", "material_class": "fibrous"}
    changed = _change(porous_document, ("scene", NEW, "floor", 0, "material", "layers", 0), miki)
    assert {row.path for row in scheme_differences(porous_scheme, changed)} == {f"scene.{NEW}.floor"}
    names = set(cases) | {f"layers[].{name}" for name in parameters} | {"layers[].model"}
    return {f"scene.{NEW}[][].material.{name}" for name in names} | _measured_surface_fields(document)


def _measured_surface_fields(document: dict[str, Json]) -> set[str]:
    from tests.engine.test_measured_impedance import measured
    from tests.engine._surface_zone_cases import NEW
    measured_material = measured()
    base = cast(dict[str, object], measured_material['base_assembly'])
    layers = cast(list[dict[str, object]], base['layers'])
    layers.append({'kind': 'porous', 'thickness_m': .05, 'flow_resistivity_pa_s_m2': 19100.,
        'material_class': 'fibrous', 'model': 'wilson_2015', 'porosity': .98, 'tortuosity': 1.1})
    prefix: tuple[str | int, ...] = ('scene', NEW, 'floor', 0, 'material')
    original = _change(document, prefix, cast(Json, measured_material))
    data = cast(dict[str, Json], original.model_dump(mode='json'))
    cases: dict[str, tuple[tuple[str | int, ...], Json]] = {
        'source_and_conditions': (('source_and_conditions',), '另一支量測管'),
        'points[].frequency_hz': (('points', 0, 'frequency_hz'), 110.),
        'points[].real_pa_s_m': (('points', 0, 'real_pa_s_m'), 410.),
        'points[].imag_pa_s_m': (('points', 0, 'imag_pa_s_m'), -210.),
        'base_assembly.backing': (('base_assembly', 'backing'), 'air'),
        'base_assembly.layers[].kind': (('base_assembly', 'layers', 0), {'kind': 'air', 'thickness_m': .02}),
        'base_assembly.layers[].thickness_m': (('base_assembly', 'layers', 1, 'thickness_m'), .04),
        'base_assembly.layers[].surface_mass_kg_m2': (('base_assembly', 'layers', 0, 'surface_mass_kg_m2'), .2),
        'base_assembly.layers[].resistance_pa_s_m': (('base_assembly', 'layers', 0, 'resistance_pa_s_m'), 90.),
        'base_assembly.layers[].flow_resistivity_pa_s_m2': (('base_assembly', 'layers', 2, 'flow_resistivity_pa_s_m2'), 20000.),
        'base_assembly.layers[].material_class': (('base_assembly', 'layers', 2, 'material_class'), 'foam'),
        'base_assembly.layers[].porosity': (('base_assembly', 'layers', 2, 'porosity'), .97),
        'base_assembly.layers[].tortuosity': (('base_assembly', 'layers', 2, 'tortuosity'), 1.2),
        'base_assembly.layers[].model': (('base_assembly', 'layers', 2), {'kind': 'porous', 'model': 'miki',
            'thickness_m': .05, 'flow_resistivity_pa_s_m2': 19100., 'material_class': 'fibrous'}),
    }
    for name, (path, value) in cases.items():
        rows = scheme_differences(original, _change(data, (*prefix, *path), value))
        assert {row.path for row in rows} == {f'scene.{NEW}.floor'}, name
    return {f'scene.{NEW}[][].material.{name}' for name in cases}
