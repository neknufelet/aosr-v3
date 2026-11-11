"""材料庫（#1082 段 2-1）考卷：格式與檢查、內建材料跟今天的產品設定一樣、實測表沒有虛部時的轉換。

內建可最佳化材料要能逐欄重建出今天第三階段梯度設定讀出來的家族（搜尋之後改從材料庫拿材料時，舊資料夾才不會變）；
內建固定材料要原樣帶著構造預設的材料。沒有虛部的實測表在子行程裡真的走一次物理的垂直入射阻抗。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.materials import library
from aosr.materials.construction_models import Assembly, MeasuredImpedance, MeasuredImpedancePoint, SurfaceMaterial
from aosr.runtime import child_process_env, configure_jax
from tests.engine._cpu_pins import forward_cpu_pins

_ROWS = ((100.0, 900.0, -1500.0), (200.0, 700.0, -800.0), (400.0, 500.0, -300.0))
_BASE: dict[str, object] = {"kind": "assembly", "backing": "rigid", "layers": [
    {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 19100.0, "material_class": "fibrous"}]}
_CONDITIONS = "阻抗管內徑 40 mm、垂直入射、樣品 50 mm"


def _table(*, imag: bool = True) -> dict[str, object]:
    points = [{"frequency_hz": f, "real_pa_s_m": r, **({"imag_pa_s_m": i} if imag else {})} for f, r, i in _ROWS]
    return {"kind": "measured_impedance", "points": points, "base_assembly": _BASE, "source_and_conditions": _CONDITIONS}


def _optimizable(**changes: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "kind": "optimizable", "id": "rock_wool_60k", "label": "60K 岩棉（自填）", "provenance": "model", "source": "使用者自填流阻",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 20000.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.075, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": None},
        "max_total_depth_m": 0.15, "suggested_total_depth_m": 0.1}
    entry.update(changes)
    return entry


def _fixed(**changes: object) -> dict[str, object]:
    entry: dict[str, object] = {"kind": "fixed", "id": "panel_measured", "label": "實測吸音板（只有實部）", "provenance": "measured",
                                "source": _CONDITIONS, "material": _table(imag=False)}
    entry.update(changes)
    return entry


_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)


def test_builtins_are_every_preset_as_fixed_and_every_family_as_optimizable() -> None:
    """內建材料：構造預設每筆一個固定材料、梯度設定每個家族一個可最佳化材料，代號一一對上。"""
    entries = library.builtin_entries()
    presets = load_construction_presets(config_path("construction_presets.toml"))
    families = load_material_gradient_settings(config_path("material_gradient.toml")).family
    assert [e.id for e in entries if e.kind == "fixed"] == [p.id for p in presets]
    assert [e.id for e in entries if e.kind == "optimizable"] == [f.id for f in families]
    assert all(e.provenance == "literature" for e in entries)
    assert len({e.id for e in entries}) == len({e.label for e in entries}) == len(entries)


def test_builtin_families_rebuild_todays_gradient_settings() -> None:
    """內建可最佳化材料逐欄重建今天的家族設定；初值排出來的構造就是那個家族的換樣板構造預設。"""
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    presets = {p.id: p for p in load_construction_presets(config_path("construction_presets.toml"))}
    optimizable = [e for e in library.builtin_entries() if isinstance(e, library.OptimizableEntry)]
    assert optimizable
    for entry in optimizable:
        family = settings.family_by_id(entry.id)
        assert library.family_settings(entry) == family
        assert entry.template_preset is not None
        assert library.template_assembly(entry) == Assembly.model_validate(presets[entry.template_preset].material)


def test_builtin_fixed_entries_carry_the_preset_material_unchanged() -> None:
    """內建固定材料的物理材料＝構造預設那份材料原樣驗證出來的。"""
    presets = {p.id: p for p in load_construction_presets(config_path("construction_presets.toml"))}
    fixed = [e for e in library.builtin_entries() if isinstance(e, library.FixedEntry)]
    for entry in fixed:
        assert entry.surface_material() == TypeAdapter(SurfaceMaterial).validate_python(presets[entry.id].material)
        assert (entry.label, entry.source, entry.links, entry.notes) == (
            presets[entry.id].label, presets[entry.id].source, presets[entry.id].links, presets[entry.id].notes)


@pytest.mark.parametrize(("entry", "basis"), [
    (_optimizable(), "simulated"),
    (_fixed(material=_BASE), "simulated"),
    (_fixed(material={"kind": "constant_impedance", "impedance_pa_s_m": 123480.0}), "approximate"),
    (_fixed(material=_table(imag=True)), "measured"),
    (_fixed(), "approximate"),
])
def test_label_follows_how_the_impedance_is_obtained(entry: dict[str, object], basis: str) -> None:
    """標語照阻抗怎麼得到推：層構造＝模擬、有虛部的實測表＝實測、沒虛部的實測表與固定實數阻抗＝近似。"""
    parsed = _ENTRY.validate_python(entry)
    assert library.impedance_basis(parsed) == basis
    assert library.basis_label(parsed) == library.BASIS_LABELS[basis]


def test_table_with_imaginary_parts_converts_to_the_same_physics_table() -> None:
    """有虛部的表轉成物理用的實測阻抗表，跟直接寫成物理表的那一份相等。"""
    entry = _ENTRY.validate_python(_fixed(material=_table(imag=True)))
    assert isinstance(entry, library.FixedEntry)
    direct = MeasuredImpedance(points=tuple(MeasuredImpedancePoint(frequency_hz=f, real_pa_s_m=r, imag_pa_s_m=i) for f, r, i in _ROWS),
                               base_assembly=Assembly.model_validate(_BASE), source_and_conditions=_CONDITIONS)
    assert entry.surface_material() == direct


def _impedance(tmp_path: Path, material: object, frequencies: list[float]) -> dict[str, list[object]]:
    env = forward_cpu_pins(child_process_env(threads=1))
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    env["TMPDIR"] = str(tmp_path)
    output = tmp_path / "result.json"
    probe = Path(__file__).with_name("_material_library_probe.py")
    process = subprocess.run([sys.executable, str(probe), str(output)], cwd=tmp_path, env=env, capture_output=True, text=True,
                             input=json.dumps({"material": material, "frequencies_hz": frequencies}), timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, list[object]], json.loads(output.read_text(encoding="utf-8")))


def test_table_without_imaginary_parts_evaluates_as_pure_real_inside_the_table(tmp_path: Path) -> None:
    """沒有虛部的表：轉換後虛部全是 0、走物理的垂直入射阻抗算得出來；表內頻點實部等於表、虛部是 0，表外照底構造補值。"""
    entry = _ENTRY.validate_python(_fixed())
    assert isinstance(entry, library.FixedEntry)
    physics = entry.surface_material()
    assert isinstance(physics, MeasuredImpedance)
    assert [row.imag_pa_s_m for row in physics.points] == [0.0, 0.0, 0.0]
    inside = _impedance(tmp_path, physics.model_dump(mode="json"), [f for f, _r, _i in _ROWS])
    assert inside["real"] == pytest.approx([r for _f, r, _i in _ROWS], rel=1e-12)
    assert inside["imag"] == [0.0, 0.0, 0.0]
    assert inside["notes"] == []
    outside = _impedance(tmp_path, physics.model_dump(mode="json"), [1000.0])
    assert outside["notes"] == ["measured_table_model_fill"]


def test_a_table_missing_only_some_imaginary_parts_is_refused() -> None:
    """虛部要整張都有或整張都沒有；只缺一列的表不收。"""
    table = _table(imag=True)
    points = cast(list[dict[str, object]], table["points"])
    del points[1]["imag_pa_s_m"]
    with pytest.raises(ValueError, match="整張"):
        _ENTRY.validate_python(_fixed(material=table))


def test_a_table_the_physics_contract_refuses_is_refused_here_too() -> None:
    """跟物理用的實測阻抗表同一份契約：頻率沒有嚴格遞增，物理那邊不收，這裡也不收。"""
    table = _table(imag=False)
    points = cast(list[dict[str, object]], table["points"])
    points[1], points[2] = points[2], points[1]
    with pytest.raises(ValueError, match="嚴格遞增"):
        _ENTRY.validate_python(_fixed(material=table))


def _thickness(**changes: object) -> dict[str, object]:
    spec: dict[str, object] = {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.075, 0.1], "initial_m": 0.05}
    spec.update(changes)
    return spec


@pytest.mark.parametrize(("changes", "message"), [
    ({"porous_thickness": _thickness(initial_m=0.075)}, "要等於多孔厚度初值"),
    ({"porous_thickness": _thickness(range_m=[0.05, 0.2])}, "範圍就是最薄到最厚"),
    ({"porous_thickness": _thickness(choices_m=None)}, "要給產品厚度"),
    ({"porous_thickness": _thickness(choices_m=[0.1, 0.05], range_m=[0.1, 0.05])}, "下限"),
    ({"porous_thickness": _thickness(adjustable=False, choices_m=None)}, "上下限要相等"),
    ({"porous_thickness": _thickness(choices_m=[0.05, 0.05, 0.1])}, "不重複"),
    ({"porous": {"kind": "porous", "thickness_m": 0.06, "flow_resistivity_pa_s_m2": 20000.0, "material_class": "fibrous"},
      "porous_thickness": _thickness(initial_m=0.06)}, "不是產品厚度"),
    ({"cavity": {"adjustable": True, "range_m": [-0.01, 0.1], "initial_m": None}}, "下限"),
    ({"cavity": {"adjustable": True, "range_m": [0.02, 0.1], "initial_m": None}}, "一開始沒有這一層"),
    ({"cavity": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1], "initial_m": 0.05}}, "不給產品厚度"),
    ({"cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.2}}, "不在範圍"),
    ({"porous_thickness": _thickness(adjustable=False, range_m=[0.05, 0.05], choices_m=None),
      "cavity": {"adjustable": False, "range_m": [0.05, 0.05], "initial_m": 0.05}, "suggested_total_depth_m": None}, "至少要有一格能調"),
    ({"max_total_depth_m": 0.04}, "最薄排法"),
    ({"cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.1}, "max_total_depth_m": 0.12}, "初值總深"),
    ({"suggested_total_depth_m": 0.2}, "建議總深"),
    ({"suggested_total_depth_m": 0.04}, "建議總深"),
    ({"max_total_depth_m": 0.08, "suggested_total_depth_m": 0.09}, "建議總深"),
    ({"preset_members": ["a", "a"]}, "重複"),
    ({"template_preset": "glass_wool_48k_50mm"}, "不在 preset_members"),
    ({"id": "none"}, "保留代號"),
    ({"links": ["http://example.org"]}, "https"),
    ({"notes": [""]}, "註記"),
])
def test_inconsistent_optimizable_entries_are_refused(changes: dict[str, object], message: str) -> None:
    """可最佳化材料的檢查：厚度與初值、產品厚度、能不能調、總深與建議總深、舊家族對應、保留代號、出處網址。"""
    with pytest.raises(ValueError, match=message):
        _ENTRY.validate_python(_optimizable(**changes))


def test_a_consistent_optimizable_entry_reports_its_depth_bounds() -> None:
    """好的可最佳化材料收得下；層能排的總深＝多孔最薄＋空腔下限～多孔最厚＋空腔上限。"""
    entry = _ENTRY.validate_python(_optimizable())
    assert isinstance(entry, library.OptimizableEntry)
    assert entry.depth_bounds() == (0.05, 0.2)
    with pytest.raises(ValueError, match="現在還收不到"):
        library.family_settings(entry)


def test_the_cavity_lower_bound_counts_toward_the_shallowest_depth() -> None:
    """空腔下限不是 0 時，最薄的總深＝多孔最薄＋空腔下限。"""
    entry = _ENTRY.validate_python(_optimizable(cavity={"adjustable": True, "range_m": [0.02, 0.1], "initial_m": 0.02}))
    assert isinstance(entry, library.OptimizableEntry)
    assert entry.depth_bounds() == (0.05 + 0.02, 0.1 + 0.1)


def test_a_product_thickness_must_be_positive() -> None:
    """產品厚度要是正數（單獨一格厚度也查，不靠可最佳化材料那一層補擋）。"""
    with pytest.raises(ValueError, match="正數"):
        library.ThicknessSpec(adjustable=True, range_m=(0.0, 0.05), choices_m=(0.0, 0.05), initial_m=0.05)


def test_a_zero_initial_cavity_means_no_air_layer() -> None:
    """空腔初值 0 跟 null 一樣：排出來的構造沒有空氣層（物理的空氣層厚度一定要正）。"""
    entry = _ENTRY.validate_python(_optimizable(cavity={"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.0}))
    assert isinstance(entry, library.OptimizableEntry)
    assert library.template_assembly(entry).layers == (entry.porous,)


def test_an_entry_without_cavity_maps_to_a_family_with_no_cavity() -> None:
    """沒有空腔這一格的可最佳化材料轉成家族設定：空腔範圍 0～0、沒有空腔初值；成員、產品厚度、初始厚度原樣帶。"""
    entry = _ENTRY.validate_python(_optimizable(
        cavity=None, porous_thickness=_thickness(initial_m=0.075),
        porous={"kind": "porous", "thickness_m": 0.075, "flow_resistivity_pa_s_m2": 20000.0, "material_class": "fibrous"},
        suggested_total_depth_m=None, preset_members=["rock_wool_60k_50mm", "rock_wool_60k_75mm"], template_preset="rock_wool_60k_75mm"))
    assert isinstance(entry, library.OptimizableEntry)
    family = library.family_settings(entry)
    assert (family.cavity_range_m, family.initial_cavity_m) == ((0.0, 0.0), None)
    assert (family.presets, family.thicknesses_m, family.initial_thickness_m) == (
        ("rock_wool_60k_50mm", "rock_wool_60k_75mm"), (0.05, 0.075, 0.1), 0.075)


def test_user_library_round_trips_and_a_missing_file_is_empty(tmp_path: Path) -> None:
    """使用者材料庫寫成 JSON 再讀回來一模一樣；檔案不在就是空的；內建在前、使用者在後。"""
    path = tmp_path / "materials.json"
    assert library.load_user_library(path) == ()
    entries = (_ENTRY.validate_python(_optimizable()), _ENTRY.validate_python(_fixed()))
    text = library.dump_user_library(entries)
    path.write_text(text, encoding="utf-8")
    assert library.load_user_library(path) == entries
    assert library.dump_user_library(library.load_user_library(path)) == text
    assert library.full_library(path) == (*library.builtin_entries(), *entries)
    assert library.full_library(None) == library.builtin_entries()


@pytest.mark.parametrize("changes", [{"id": "glass_wool_48k"}, {"label": "48K 玻璃棉 50 mm 貼牆"}])
def test_user_materials_cannot_shadow_builtins(tmp_path: Path, changes: dict[str, object]) -> None:
    """內建唯讀：使用者材料的代號或顯示名跟內建撞到，寫跟讀都拒收。"""
    entry = _ENTRY.validate_python(_fixed(**changes))
    with pytest.raises(ValueError, match="不可重複"):
        library.dump_user_library((entry,))
    path = tmp_path / "materials.json"
    path.write_text(json.dumps({"schema_version": library.SCHEMA_VERSION, "materials": [entry.model_dump(mode="json")]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="不可重複"):
        library.load_user_library(path)


def test_two_user_materials_with_the_same_id_are_refused() -> None:
    """使用者材料彼此代號也不准重複。"""
    one = _ENTRY.validate_python(_fixed())
    two = _ENTRY.validate_python(_fixed(label="另一塊"))
    with pytest.raises(ValueError, match="不可重複"):
        library.dump_user_library((one, two))
