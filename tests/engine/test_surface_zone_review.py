"""之五丙第一批：抓回誤登、錯中文、對齊、綱要、物件旁門與牆序的退步。"""
import json
import re
from pathlib import Path
from typing import Self, cast

import pytest
from pydantic import ValidationError, model_validator

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.gui.compare_view import scheme_differences
from aosr.gui.problem_text import field_name, material_summary
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, Layer, PorousLayer
from aosr.physics import report_io
from aosr.physics.report_surfaces import WholeWallZone
from aosr.physics.report_validation import ReportInputError
from aosr.reporting.scheme import Scene, Scheme
from aosr.reporting.validation import SchemeValidationError, validated_scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair, reference_document
from tests.engine._surface_zone_cases import NEW, OLD, mixed_scheme, zone_scene, zone_document
from tests.engine._zone_review_cases import (
    MATERIAL_CASES, ORIGINAL_CASES, porous, reject_at, report_document, reverse_walls, review_scene)
from tests.engine._zone_schema_reader import Schema, accepts


def test_fem_registered_materials_are_accepted_by_its_actual_routes() -> None:
    table = load_capabilities(config_path("capabilities.toml"))
    entry = next(entry for entry in table.entry if entry.name == "fem_rigid")
    actual_routes = {"rigid_walls", "real_frequency_independent_impedance"}
    assert {row.materials for row in entry.capability if row.status != "unsupported"} <= actual_routes
    assert "assembly_by_wall" not in {row.materials for row in entry.capability}
    report = next(entry for entry in table.entry if entry.name == "three_lane_report")
    assert "assembly_by_wall" in {row.materials for row in report.capability}


def test_material_descriptions_belong_to_the_displayed_model() -> None:
    fixed = ConstantImpedance(impedance_pa_s_m=1646.4)
    layer = PorousLayer.model_validate(porous())
    assembly = Assembly(layers=(layer,), backing="rigid")
    assert "固定阻抗" in material_summary(fixed)
    assert "constant_impedance" not in material_summary(fixed)
    assert "薄板層" not in material_summary(fixed)
    label, _, unit = _description(PorousLayer, "thickness_m").partition("，")
    assert label in material_summary(assembly) and unit in material_summary(assembly)
    assert "空腔厚度" not in material_summary(assembly)


def _description(model: type[ConstantImpedance] | type[PorousLayer], name: str) -> str:
    return str(model.model_fields[name].description).split("（", 1)[-1].removesuffix("）")


@pytest.mark.parametrize("tag,model,name", [("constant_impedance", ConstantImpedance, "kind"),
                                         ("porous", PorousLayer, "thickness_m")])
def test_error_leaf_uses_its_material_or_layer_description(
    tag: str, model: type[ConstantImpedance] | type[PorousLayer], name: str,
) -> None:
    middle = "assembly.layers.0." if tag == "porous" else ""
    label = field_name(f"scene.{NEW}.floor.0.material.{middle}{tag}.{name}", {})
    description = _description(model, name)
    expected = description.split("，", 1)[0] if name == "kind" else description
    assert label is not None and expected in label
    assert tag not in label
    assert ("空腔厚度" if tag == "porous" else "層種類代號") not in label


def test_old_and_mixed_comparison_changes_only_the_floor() -> None:
    rows = scheme_differences(Scheme.model_validate(reference_document()), mixed_scheme())
    assert {row.path for row in rows} == {f"scene.{NEW}.floor"}
    floor = rows[0]
    assert "固定阻抗" in floor.a_text and "構造" in floor.b_text
    assert "未設定" not in floor.a_text + floor.b_text


def test_fixed_comparison_keeps_the_executed_cb3b295d_control() -> None:
    changed = reference_document()
    scene = cast(dict[str, object], changed["scene"])
    cast(dict[str, float], scene[OLD])["floor"] = 2000.
    scene["reflection_order_k"] = 2
    rows = scheme_differences(Scheme.model_validate(reference_document()), Scheme.model_validate(changed))
    # 來源：cb3b295d 的 compare_view 程式在改動前實跑；沒有用當次函式建立答案。
    assert [(row.path, row.label, row.a_text, row.b_text) for row in rows] == [
        (f"scene.{OLD}.floor", "地板阻抗", "1646 帕·秒／公尺", "2000 帕·秒／公尺"),
        ("scene.reflection_order_k", "反射階數", "未設定", "2 階")]


@pytest.mark.parametrize("case", (*ORIGINAL_CASES, *MATERIAL_CASES))
@pytest.mark.parametrize("entry", ("scheme", "report", "search"))
def test_material_errors_are_located_and_chinese(tmp_path: Path, case: str, entry: str) -> None:
    scene, path, hint = review_scene(case)
    with pytest.raises((SchemeValidationError, ValidationError, ReportInputError)) as caught:
        reject_at(entry, scene, tmp_path)
    text = str(caught.value)
    assert path in text and hint in text, text
    messages = ([str(error["msg"]) for error in caught.value.errors()]
                if isinstance(caught.value, ValidationError) else [text])
    for message in messages:
        assert re.search(r"[A-Za-z]+(?:\s+[A-Za-z]+){2,}", message) is None, message
        assert "Field required" not in message and "Input should" not in message
    assert not (tmp_path / "search").exists()


def test_invalid_layer_has_only_its_real_error() -> None:
    scene, path, _ = review_scene("layer")
    with pytest.raises(SchemeValidationError) as caught:
        validated_scheme(reference_document() | {"scene": scene})
    assert {problem.path for problem in caught.value.problems} == {f"scene.{path}"}


@pytest.mark.parametrize("case", ("empty-zones", "two", "null-old", "missing-material-kind",
    "missing-layer-kind", "wilson-missing", "komatsu-porosity", "mechel-large", "miki-tortuosity",
    "mechel-missing", "wilson-null", "wilson-valid", "mechel-valid", "miki-valid",
    "komatsu-null", "mechel-tortuosity", "wilson-large", "fixed-valid"))
def test_schema_and_model_agree_on_review_table(tmp_path: Path, case: str) -> None:
    scene = _schema_scene(case)
    root = cast(Schema, report_io.ReportInput.model_json_schema())
    walls = cast(Schema, root["properties"])
    contract = {key: root[key] for key in ("oneOf",)} | {
        "type": "object", "properties": {OLD: walls[OLD], NEW: walls[NEW]}}
    schema_accepts = accepts(contract, scene, root)
    try:
        report_io.load_input_document(report_document(scene), CAPABILITIES, DIRECTIVITY)
    except ReportInputError:
        model_accepts = False
    else:
        model_accepts = True
    expected = case in {"wilson-valid", "wilson-large", "mechel-valid", "miki-valid", "fixed-valid"}
    assert model_accepts is expected, case
    assert schema_accepts is expected, case
    report_io.regenerate_schema_files(tmp_path)
    saved = json.loads((tmp_path / "three_lane_report_input.schema.json").read_text())
    assert saved["properties"] == root["properties"] and saved["$defs"] == root["$defs"]


def _schema_scene(case: str) -> dict[str, object]:
    if case in MATERIAL_CASES or case in ORIGINAL_CASES:
        return review_scene(case)[0]
    scene = zone_scene(mixed=True)
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    if case == "empty-zones":
        zones["floor"] = []
    elif case == "fixed-valid":
        return zone_scene()
    else:
        layer = porous("wilson_2015" if case.startswith("wilson") else
                       "mechel_1976" if case.startswith("mechel") else "miki" if case.startswith("miki") else "komatsu")
        if case in {"wilson-valid", "wilson-null", "wilson-large"}:
            layer |= {"porosity": None if case == "wilson-null" else 1.2 if case == "wilson-large" else .98,
                      "tortuosity": 1.1}
        elif case.startswith("mechel"):
            layer["porosity"] = .98
            if case == "mechel-tortuosity":
                layer["tortuosity"] = 1.1
        elif case == "komatsu-null":
            layer["porosity"] = None
        cast(dict[str, object], zones["floor"][0]["material"])["layers"] = [layer]
    return scene


@pytest.mark.parametrize("forgery", ("copy-area", "construct-area", "construct-measured", "constant-measured"))
@pytest.mark.parametrize("entry", ("scene", "report-model", "report", "scheme-model", "scheme"))
def test_python_objects_cannot_bypass_nested_validation(forgery: str, entry: str) -> None:
    scene = zone_scene(mixed=forgery != "constant-measured")
    zones = cast(dict[str, object], scene[NEW])
    original = WholeWallZone.model_validate(cast(list[object], zones["floor"])[0])
    if forgery in ("copy-area", "construct-area"):
        forged = (original.model_copy(update={"area": "rectangle"}) if forgery == "copy-area" else
                  WholeWallZone.model_construct(area="rectangle", material=original.material))
        hint, leaf = "整面一區", "area"
    else:
        material = ({"kind": "measured_impedance"} if forgery == "construct-measured" else
                    ConstantImpedance(impedance_pa_s_m=1646.4).model_copy(update={"kind": "measured_impedance"}))
        forged = WholeWallZone.model_construct(area="whole_wall", material=material)
        hint, leaf = "必填", "material.measured_impedance.points"
    zones["floor"] = (forged,)
    with pytest.raises((ValidationError, ReportInputError, SchemeValidationError)) as caught:
        _object_entry(entry, scene)
    assert f"{NEW}.floor.0.{leaf}" in str(caught.value) and hint in str(caught.value)


def test_revalidation_keeps_strict_types_and_omitted_fields() -> None:
    """重驗不准把省略改成明給，也不准把資料類別與 tuple 換型別讓嚴格模式拒收合法物件。"""
    scheme = Scheme.model_validate(reference_document())
    again = Scheme.model_validate(scheme, strict=True)
    assert again.model_fields_set == scheme.model_fields_set
    assert again.scene.model_fields_set == scheme.scene.model_fields_set
    assert again.model_dump(exclude_unset=True) == scheme.model_dump(exclude_unset=True)


def test_revalidation_keeps_explicit_null_of_forged_layer() -> None:
    """偽造的 Komatsu 層明給孔隙率空值：重驗要照字典輸入一樣拒收，不能把明給的空值洗成省略。"""
    scene = zone_scene(mixed=True)
    zone = cast(list[dict[str, object]], cast(dict[str, object], scene[NEW])["floor"])[0]
    layers = cast(list[object], cast(dict[str, object], zone["material"])["layers"])
    layers[0] = PorousLayer.model_construct(None, **(porous() | {"porosity": None}))
    with pytest.raises(ValidationError, match="不讀"):
        Scene.model_validate(scene)


def test_revalidation_rejects_forged_null_over_unset_default() -> None:
    """偽造層沒明給模型、現值卻是空值：重驗要照現值拒收，不能悄悄換回預設的 Komatsu。"""
    scene = zone_scene(mixed=True)
    zone = cast(list[dict[str, object]], cast(dict[str, object], scene[NEW])["floor"])[0]
    layers = cast(list[object], cast(dict[str, object], zone["material"])["layers"])
    layers[0] = PorousLayer.model_construct(set(porous()) - {"model"}, **(porous() | {"model": None}))
    with pytest.raises(ValidationError, match="合法選項"):
        Scene.model_validate(scene)


@pytest.mark.parametrize("material", ("constant", "assembly", "air", "porous", "board"))
@pytest.mark.parametrize("entry", ("scene", "scheme", "report"))
def test_objects_built_with_default_kind_are_accepted(material: str, entry: str) -> None:
    """直接建構的材料或層省略 kind（種類）用預設值，跟明給 kind 的字典是同一份材料，三入口都要收、結果一樣。"""
    layers: dict[str, Layer] = {
        "air": AirLayer(thickness_m=.1), "board": BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),
        "porous": PorousLayer(thickness_m=.1, flow_resistivity_pa_s_m2=10000., material_class="fibrous")}
    built = (ConstantImpedance(impedance_pa_s_m=1646.4) if material == "constant" else
             Assembly(layers=(AirLayer(kind="air", thickness_m=.1),), backing="rigid") if material == "assembly" else
             Assembly(kind="assembly", layers=(layers[material],), backing="rigid"))
    saved = []
    for value in (built, built.model_dump(mode="json")):
        scene = zone_scene(mixed=True)
        cast(list[dict[str, object]], cast(dict[str, object], scene[NEW])["floor"])[0]["material"] = value
        saved.append(Scene.model_validate(scene).model_dump_json() if entry == "scene" else
                     Scheme.model_validate(reference_document() | {"scene": scene}).model_dump_json()
                     if entry == "scheme" else
                     report_io.load_input_document(report_document(scene), CAPABILITIES, DIRECTIVITY).model_dump_json())
    assert saved[0] == saved[1]


def test_values_filled_by_validators_survive_scheme_revalidation() -> None:
    """驗證器補出、沒明給的值也是場景的現值：交給方案重驗不准掉回預設，存檔要跟同值字典一樣。"""
    class FilledScene(Scene):
        @model_validator(mode="after")
        def filled(self) -> Self:
            if self.reflection_order_k is None:
                object.__setattr__(self, "reflection_order_k", 2)
            return self

    source = FilledScene.model_validate(reference_document()["scene"])
    assert "reflection_order_k" not in source.model_fields_set
    expected = Scheme.model_validate(reference_document() | {"scene": source.model_dump(mode="json")})
    assert Scheme.model_validate(reference_document() | {"scene": source}).model_dump_json() == (
        expected.model_dump_json())


def test_schema_reader_honours_ref_siblings_and_json_booleans() -> None:
    """判讀器自己的牙：$ref 旁的限制要生效、真假值不是 0／1；否則綱要錯了一致性考卷照樣綠。"""
    root: Schema = {"$defs": {"Any": {}}}
    assert accepts({"$ref": "#/$defs/Any"}, {}, root)
    assert not accepts({"$ref": "#/$defs/Any", "not": {}}, {}, root)
    assert not accepts({"const": 1}, True, root) and not accepts({"enum": [0]}, False, root)
    assert accepts({"const": 1}, 1.0, root) and accepts({"enum": [True]}, True, root)


def _object_entry(entry: str, scene: dict[str, object]) -> None:
    if entry == "scene":
        Scene.model_validate(Scene.model_construct(_fields_set=set(scene), **scene))
    elif entry in {"report", "report-model"}:
        inputs = report_io.load_input_document(pair(mixed_scheme()), CAPABILITIES, DIRECTIVITY)
        forged = inputs.model_copy(update={NEW: scene[NEW]})
        if entry == "report":
            report_io.load_input_document(forged, CAPABILITIES, DIRECTIVITY)
        else:
            report_io.ReportInput.model_validate(forged, context={"table": CAPABILITIES, "directivity": DIRECTIVITY})
    else:
        project = Scheme.model_validate(reference_document()).model_copy(update={
            "scene": Scene.model_construct(_fields_set=set(scene), **scene)})
        if entry == "scheme":
            validated_scheme(project)
        else:
            Scheme.model_validate(project)


def test_reversed_fixed_walls_serialize_to_identical_bytes() -> None:
    old = reverse_walls(cast(dict[str, object], reference_document()["scene"]), OLD)
    new = reverse_walls(zone_scene(), NEW)
    assert Scene.model_validate(new).model_dump_json() == Scene.model_validate(old).model_dump_json()
    assert report_io.load_input_document(report_document(new), CAPABILITIES, DIRECTIVITY).model_dump_json() == (
        report_io.load_input_document(report_document(old), CAPABILITIES, DIRECTIVITY).model_dump_json())


def test_mixed_round_trip_keeps_input_wall_order() -> None:
    scene = reverse_walls(zone_scene(mixed=True), NEW)
    original = Scene.model_validate(scene)
    restored = Scene.model_validate_json(original.model_dump_json())
    assert restored.surface_zones_by_wall is not None
    assert tuple(restored.surface_zones_by_wall) == tuple(cast(dict[str, object], scene[NEW]))


def test_revalidating_fixed_objects_keeps_saved_bytes_and_scene_identity() -> None:
    original = Scheme.model_validate(reference_document())
    assert validated_scheme(original).model_dump_json() == original.model_dump_json()
    inputs = report_io.load_input_document(pair(original), CAPABILITIES, DIRECTIVITY)
    revalidated = report_io.load_input_document(inputs, CAPABILITIES, DIRECTIVITY)
    assert revalidated.model_dump_json() == inputs.model_dump_json()
    assert report_io.scene_fingerprint(revalidated) == report_io.scene_fingerprint(inputs)


def test_plain_material_summary_reads_the_own_model_description(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.gui.problem_text import fixed_material_summary
    from aosr.materials.construction_models import AirLayer, BoardLayer

    monkeypatch.setattr(ConstantImpedance.model_fields["kind"], "description",
                        "kind（表面種類代號，constant_impedance 表示考卷固定實數阻抗材料）")
    monkeypatch.setattr(ConstantImpedance.model_fields["impedance_pa_s_m"], "description",
                        "impedance_pa_s_m（考卷整面正有限實數阻抗，Pa·s/m）")
    monkeypatch.setattr(PorousLayer.model_fields["thickness_m"], "description", "thickness_m（考卷多孔厚度，m）")
    monkeypatch.setattr(AirLayer.model_fields["thickness_m"], "description", "thickness_m（考卷空腔厚度，m）")
    fixed = ConstantImpedance(impedance_pa_s_m=1646.4)
    assembly = Assembly(layers=(PorousLayer.model_validate(porous()),
        BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.), AirLayer(thickness_m=.43)), backing="rigid")
    summary = material_summary(assembly)
    # 描述原字照上畫面：網頁不另改材料的中文（改字等於在 src 另寫一份對照表）
    assert material_summary(fixed).startswith("考卷固定實數阻抗材料；考卷整面正有限實數阻抗 1646.4 Pa·s/m")
    assert fixed_material_summary(1646.4) == material_summary(fixed)
    assert "考卷多孔厚度" in summary and "考卷空腔厚度" in summary
    assert "0.43 m" in summary and "0.15 kg/m²" in summary and "80 Pa·s/m" in summary
    for code in ("constant_impedance", "assembly", "porous", "air", "board", "rigid", "fibrous", "komatsu", "None"):
        assert code not in summary and code not in material_summary(fixed)


@pytest.mark.parametrize("change", ("model", "small-number"))
def test_plain_summary_does_not_hide_material_parameter_changes(change: str) -> None:
    from copy import deepcopy
    from typing import cast

    first = zone_document(mixed=True)
    scene = cast(dict[str, object], first["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    original_layer = porous() | {"model": "komatsu"}
    zones["floor"][0]["material"] = {"kind": "assembly", "layers": [original_layer], "backing": "rigid"}
    second = deepcopy(first)
    changed_zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], second["scene"])[NEW])
    layer = cast(list[dict[str, object]], cast(dict[str, object], changed_zones["floor"][0]["material"])["layers"])[0]
    if change == "model":
        layer["model"] = "miki"
    else:
        layer["thickness_m"] = float(cast(float, layer["thickness_m"])) + .000000001
    rows = scheme_differences(Scheme.model_validate(first), Scheme.model_validate(second))
    assert {row.path for row in rows} == {f"scene.{NEW}.floor"}
    assert all(row.a_text != row.b_text for row in rows)
    if change == "model":
        assert "小松" in rows[0].a_text and "三木" in rows[0].b_text
