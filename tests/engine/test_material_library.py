"""材料庫（#1082 段 2-1）考卷：格式與檢查、內建材料跟今天的產品設定一樣、實測表沒有虛部時的轉換。

內建可最佳化材料要能逐欄重建出今天第三階段梯度設定讀出來的家族（搜尋之後改從材料庫拿材料時，舊資料夾才不會變）；
內建固定材料要原樣帶著構造預設的材料。沒有虛部的實測表在子行程裡真的走一次物理的垂直入射阻抗。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

from aosr.config.catalog_surfaces import load_catalog_surfaces
from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.materials import library
from aosr.materials.catalog_absorption import random_incidence_absorption
from aosr.materials.construction_models import Assembly, MeasuredImpedance, MeasuredImpedancePoint, SurfaceMaterial
from aosr.runtime import child_process_env, configure_jax
from tests.engine._cpu_pins import forward_cpu_pins

_ROWS = ((100.0, 900.0, -1500.0), (200.0, 700.0, -800.0), (400.0, 500.0, -300.0))
_BASE: dict[str, object] = {"kind": "assembly", "backing": "rigid", "layers": [
    {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 19100.0, "material_class": "fibrous"}]}
_CONDITIONS = "阻抗管內徑 40 mm、垂直入射、樣品 50 mm"


def _table(*, imag: bool = True) -> dict[str, object]:
    points = [{"frequency_hz": f, "real_pa_s_m": r, **({"imag_pa_s_m": i} if imag else {})} for f, r, i in _ROWS]
    return {"kind": "measured_impedance", "points": points, "base_assembly": copy.deepcopy(_BASE), "source_and_conditions": _CONDITIONS}


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
    """內建材料：構造預設每筆一個固定材料、文獻硬面每筆一個固定材料、梯度設定每個家族一個可最佳化材料，代號一一對上。"""
    entries = library.builtin_entries()
    presets = load_construction_presets(config_path("construction_presets.toml"))
    families = load_material_gradient_settings(config_path("material_gradient.toml")).family
    catalog = load_catalog_surfaces(config_path("catalog_surfaces.toml"))[1]
    assert [e.id for e in entries if e.kind == "fixed"] == [p.id for p in presets] + [c.id for c in catalog]
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
    fixed = [e for e in library.builtin_entries() if isinstance(e, library.FixedEntry) and e.id in presets]
    assert len(fixed) == len(presets)
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


def _swap_two_rows(table: dict[str, object]) -> None:
    points = cast(list[dict[str, object]], table["points"])
    points[1], points[2] = points[2], points[1]


def _drop_base_kind(table: dict[str, object]) -> None:
    del cast(dict[str, object], table["base_assembly"])["kind"]


@pytest.mark.parametrize(("break_it", "message"), [(_swap_two_rows, "嚴格遞增"), (_drop_base_kind, "kind")])
def test_a_table_the_physics_contract_refuses_is_refused_here_too(break_it: Callable[[dict[str, object]], None],
                                                                   message: str) -> None:
    """跟物理用的實測阻抗表同一份契約：物理那邊不收的原始表（頻率沒嚴格遞增、底構造沒寫 kind），這裡也不收。"""
    physics_table, table = _table(), _table(imag=False)
    break_it(physics_table)
    break_it(table)
    with pytest.raises(ValueError, match=message):
        MeasuredImpedance.model_validate(physics_table)
    with pytest.raises(ValueError, match=message):
        _ENTRY.validate_python(_fixed(material=table))


def test_a_serialized_physics_table_loads_and_converts_back_unchanged() -> None:
    """物理用的實測阻抗表寫成 JSON（含補值規則版本、出處頭尾空白）拿來當固定材料，轉回物理一模一樣。"""
    physics = MeasuredImpedance.model_validate({**_table(), "source_and_conditions": f"  {_CONDITIONS}\n"})
    entry = _ENTRY.validate_python(_fixed(material=physics.model_dump(mode="json")))
    assert isinstance(entry, library.FixedEntry)
    assert entry.surface_material() == physics


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


def test_fixed_surface_distinguishes_standalone_material_from_a_family() -> None:
    entry = library.FixedEntry.model_validate(_fixed())
    assert library.fixed_surface(entry) == entry.surface_material()
    assert library.fixed_surface(_ENTRY.validate_python(_optimizable())) is None


# 手冊原表（pyroomacoustics 材料資料庫，與 PTB 同源），125 Hz 起逐八度；另抄一份釘在這裡，設定檔抄錯一格就紅。
HANDBOOK_ALPHA = {
    "door_wood_solid": (0.14, 0.10, 0.06, 0.08, 0.10, 0.10, 0.10),
    "glass_single_3mm": (0.08, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02),
    "glass_double_gap30": (0.15, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02),
    "wood_floor_on_joists": (0.18, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07),
    "wood_floor_on_sleepers": (0.09, 0.06, 0.05, 0.05, 0.05, 0.04),
    "gypsum_double_13mm_stud": (0.15, 0.10, 0.06, 0.04, 0.04, 0.05, 0.05),
    "plasterboard_ceiling": (0.20, 0.15, 0.10, 0.08, 0.04, 0.02),
    "brick_rendered": (0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04),
    "brick_painted": (0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02),
    "concrete_smooth": (0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05),
    "concrete_floor": (0.01, 0.03, 0.05, 0.02, 0.02, 0.02, 0.02),
    "hard_surface_average": (0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05),
}


def test_catalog_surfaces_reproduce_their_handbook_absorption_at_each_band() -> None:
    """文獻硬面（門、窗、木地板、石膏板、磚、混凝土）：換成阻抗表後，在每個頻帶中心用 Paris 公式正算回去，
    等於手冊那一格；頻帶外平坦延伸，最低頻帶以下各點都等於最低頻帶那一格。"""
    table, surfaces = load_catalog_surfaces(config_path("catalog_surfaces.toml"))
    entries = {e.id: e for e in library.catalog_entries()}
    rho_c = load_physics_constants(config_path("physics_constants.toml")).rho_c
    assert list(entries) == [s.id for s in surfaces] == list(HANDBOOK_ALPHA)
    for surface in surfaces:
        entry = entries[surface.id]
        assert (entry.label, entry.provenance, entry.source) == (surface.label, "literature", surface.source)
        assert entry.imaginary_missing and library.basis_label(entry) == library.BASIS_LABELS["approximate"]
        material = entry.surface_material()
        assert isinstance(material, MeasuredImpedance)
        rows = {round(point.frequency_hz, 3): point.real_pa_s_m for point in material.points}
        assert material.points[0].frequency_hz <= table.lowest_hz * 2 ** (1 / table.points_per_octave)
        assert material.points[-1].frequency_hz >= table.highest_hz / 2 ** (1 / table.points_per_octave)
        alphas = HANDBOOK_ALPHA[surface.id]
        assert surface.absorption == alphas, surface.id
        assert surface.band_center_hz == tuple(125.0 * 2.0**k for k in range(len(alphas))), surface.id
        assert "Hz 以下" in material.source_and_conditions and "平延伸" in material.source_and_conditions
        for band, alpha in zip(surface.band_center_hz, alphas, strict=True):
            back = random_incidence_absorption(rows[band] / rho_c)
            assert back == pytest.approx(alpha, abs=1e-4), (surface.id, band)
        assert material.points[0].real_pa_s_m == rows[surface.band_center_hz[0]]


def test_catalog_surface_above_the_paris_peak_is_refused_not_clamped() -> None:
    """手冊值超過 Paris 頂點會被夾掉、不再是那一格；整筆拒收，不默默換值。"""
    table, surfaces = load_catalog_surfaces(config_path("catalog_surfaces.toml"))
    loud = surfaces[0].model_copy(update={"absorption": (0.99, *surfaces[0].absorption[1:])})
    with pytest.raises(ValueError, match="clamped"):
        library._catalog_entry(loud, library._catalog_axis(table), load_physics_constants(config_path("physics_constants.toml")).rho_c)
