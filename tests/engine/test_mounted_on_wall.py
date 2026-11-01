"""貼在原牆上的構造（#990）：新增層背後接原牆的表面阻抗，不是剛背。

算數全在隔離的雙精度子行程（同 test_construction_diff 的開法），父考卷只讀它寫在 tmp 的結果。
性質：新增層厚度歸零就是原牆、原牆阻抗越大越像剛背（差距照原牆倍數縮小）、可微與非可微兩條路相同（實測表原牆的表內內插照阻抗尺，其餘逐位）、換軸換空氣照新條件現算、
梯度對中央差分（尺從傳遞矩陣移植考卷 import，不另寫數字）、格式收拒、方案讀回、三路報表接得上；
控制組：舊多層構造（剛背）的三個入口逐位等於改動前主線釘住的值。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from aosr.materials.construction_models import Assembly, ConstantImpedance, ConstructionNote, MountedOnWall, SurfaceMaterial
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.runtime import child_process_env, configure_jax
from tests.engine._surface_zone_cases import ASSEMBLY, NEW, zone_document
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL, IMPEDANCE_ABS, REFERENCE_REL
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS


PROBE = Path(__file__).with_name("_mounted_on_wall_probe.py")
WALL_KINDS = ("constant", "measured", "assembly")
# 改動前主線（38cea88f）在同一個子行程開法下量的值：兩筆預設剛背、頻率 31.5／125／1000 Hz、角度 0／60 度。
# 從寫入端（construction 三個入口）取，十六進位逐位；垂直與角度是（實部, 虛部），角度格照頻率主序。
CONTROL: dict[str, dict[str, list[object]]] = {
    "glass_wool_48k_50mm_gap50": {
        "normal": [["0x1.66a7760585f2ap+8", "-0x1.66ac8f719e3e0p+12"], ["0x1.8af5f3018b75ap+8", "-0x1.508f4a35912dfp+10"],
                   ["0x1.6b3830028c201p+9", "-0x1.ed2dc1908f61ep+6"]],
        "angles": [["0x1.66a7760585f2ap+8", "-0x1.66ac8f719e3e0p+12"], ["0x1.b14482f54e54dp+8", "-0x1.099122778e725p+13"],
                   ["0x1.8af5f3018b75ap+8", "-0x1.508f4a35912e0p+10"], ["0x1.9a39b1150dadbp+8", "-0x1.0a442838bbabdp+11"],
                   ["0x1.6b3830028c1ffp+9", "-0x1.ed2dc1908f627p+6"], ["0x1.ff1203b3b4d9bp+8", "-0x1.e9cf004218fbcp+6"]],
        "diffuse": ["0x1.4e7ff3ad830b6p-6", "0x1.e4a20d373dc4ep-3", "0x1.d64a6275de950p-1"],
    },
    "perforated_0p8mm_d10_gw24k_50mm": {
        "normal": [["0x1.e6fef9443d17dp+8", "-0x1.432cf9989cb9cp+13"], ["0x1.294a666f15d34p+8", "-0x1.53426ecfc3919p+11"],
                   ["0x1.81f16a5c9ee75p+7", "-0x1.152655ff5304ap+7"]],
        "angles": [["0x1.e6fef9443d17dp+8", "-0x1.432cf9989cb9cp+13"], ["0x1.704682747b41ep+9", "-0x1.4679b85702732p+13"],
                   ["0x1.294a666f15d46p+8", "-0x1.53426ecfc391ap+11"], ["0x1.1216a55fc33e0p+9", "-0x1.5c415f8998debp+11"],
                   ["0x1.81f16a5c9ee75p+7", "-0x1.152655ff5304ap+7"], ["0x1.88deaa7b880f5p+8", "-0x1.b5c78dceb9eacp+7"]],
        "diffuse": ["0x1.3fb813f9bcd1fp-6", "0x1.2e2116ebb7634p-3", "0x1.a24459b5b1f9bp-1"],
    },
}


def _run(mode: str, directory: Path) -> object:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(root)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    output = directory / f"{mode}.json"
    process = subprocess.run([sys.executable, str(PROBE), mode, str(output)], cwd=directory, env=env,
                             capture_output=True, text=True, timeout=600)
    assert process.returncode == 0, process.stdout + process.stderr
    result = cast(dict[str, object], json.loads(output.read_text()))
    assert result["dtype"] == "float64"
    return result["rows"]


def _decode(bits: object) -> np.ndarray:
    row = cast(dict[str, list[str]], bits)
    real = np.asarray([float.fromhex(value) for value in row["real"]])
    imag = np.asarray([float.fromhex(value) for value in row["imag"]])
    return real + 1j * imag


def _same(ours: object, existing: object, label: str) -> None:
    a, b = cast(dict[str, object], ours), cast(dict[str, object], existing)
    assert a["dtype"] == b["dtype"] == "complex128", label
    assert a["shape"] == b["shape"], label
    np.testing.assert_array_equal(_decode(a), _decode(b), err_msg=label)


def _close(ours: object, existing: object, label: str) -> None:
    """實測表與構造原牆照傳遞矩陣移植考卷那把阻抗尺（實虛部各自相對加絕對）。"""
    a, b = cast(dict[str, object], ours), cast(dict[str, object], existing)
    assert a["shape"] == b["shape"], label
    for part in (np.real, np.imag):
        np.testing.assert_allclose(part(_decode(a)), part(_decode(b)), rtol=REFERENCE_REL, atol=IMPEDANCE_ABS, err_msg=label)


@pytest.fixture(scope="module")
def paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], _run("paths", tmp_path_factory.mktemp("mounted-paths")))


def test_old_rigid_assembly_is_unchanged_bit_for_bit(tmp_path: Path) -> None:
    """控制組：舊多層構造剛背的垂直、角度與無規吸音，逐位等於改動前主線釘住的值。"""
    rows = cast(dict[str, dict[str, object]], _run("control", tmp_path))
    for name, pinned in CONTROL.items():
        row = rows[name]
        for key in ("normal", "angles"):
            measured = _decode(row[key])
            expected = np.asarray([float.fromhex(re) + 1j * float.fromhex(im)
                                   for re, im in cast(list[list[str]], pinned[key])])
            np.testing.assert_array_equal(measured, expected, err_msg=f"{name}:{key}")
        assert row["diffuse"] == pinned["diffuse"], name


def test_removing_new_layers_gives_the_wall(tmp_path: Path) -> None:
    """新增層厚度全設 0：固定阻抗原牆逐位等於原牆，實測表與構造原牆照阻抗尺；垂直與 Paris 角度格都比。"""
    rows = cast(dict[str, dict[str, object]], _run("zero", tmp_path))
    assert set(rows) == set(WALL_KINDS)
    for name, row in rows.items():
        assert row["zeroed"] == ["layer1.thickness_m", "layer2.thickness_m"], name
        check = _same if name == "constant" else _close
        check(row["normal"], row["normal_wall"], f"{name}:normal")
        check(row["angles"], row["angles_wall"], f"{name}:angles")


def test_stiff_wall_converges_to_rigid_backing(tmp_path: Path) -> None:
    """原牆取三個相隔同一倍數的有限大阻抗：跟剛背的差距每一步縮小的倍數取對數，四捨五入到整數等於原牆倍數的對數。

    不用無窮大（(A·Z+B)/(C·Z+D) 在 Z=∞ 會出非數值），也不自創門檻：收斂形狀是 1/Z，量的是縮小的數量級。
    """
    rows = cast(dict[str, list[float]], _run("rigid", tmp_path))
    loads = rows["loads"]
    for key in ("normal", "angles"):
        gaps = rows[key]
        assert all(gap > 0 for gap in gaps), (key, gaps)
        for index in range(len(loads) - 1):
            assert round(math.log10(gaps[index] / gaps[index + 1])) == round(math.log10(loads[index + 1] / loads[index])), (key, gaps)


def test_differentiable_path_equals_existing_entry(paths: dict[str, dict[str, object]]) -> None:
    """基準參數上可微入口等於非可微入口：三種原牆、垂直與 Paris 角度格；而且真的不是剛背。
    固定阻抗與多層構造原牆逐位；實測表原牆的表內值是 JAX 現算的內插（對數函式跟 numpy 差最後一兩位），照阻抗尺。"""
    assert set(paths) == set(WALL_KINDS)
    for name, row in paths.items():
        check = _close if name == "measured" else _same
        check(row["normal"], row["normal_existing"], f"{name}:normal")
        check(row["angles"], row["angles_existing"], f"{name}:angles")
        assert not np.array_equal(_decode(row["normal"]), _decode(row["rigid_normal"])), name
        assert row["variables"] == ["layer1.thickness_m", "layer1.flow_resistivity_pa_s_m2", "layer2.thickness_m"], name


def test_random_incidence_and_wall_notes(paths: dict[str, dict[str, object]]) -> None:
    """無規吸音比值（不是只看有限、0 到 1）：三種原牆都等於同一份角度阻抗另用 numpy 積的 Paris；
    原牆是多層構造時另等於「新增層＋原牆各層」直接串成的剛背構造。實測表原牆的補值註記帶「原牆」前綴，樣板帶同一份註記。"""
    for name, row in paths.items():
        diffuse = np.asarray(cast(list[float], row["diffuse"]))
        assert diffuse.shape == (len(cast(dict[str, list[str]], row["normal"])["real"]),), name
        np.testing.assert_allclose(diffuse, cast(list[float], row["diffuse_reference"]), rtol=REFERENCE_REL, atol=0, err_msg=name)
        assert (row["diffuse_concatenated"] is None) is (name != "assembly"), name
        assert row["notes"] == row["template_notes"], name
    np.testing.assert_allclose(cast(list[float], paths["assembly"]["diffuse"]),
                               cast(list[float], paths["assembly"]["diffuse_concatenated"]), rtol=REFERENCE_REL, atol=0)
    measured = cast(list[str], paths["measured"]["notes"])
    assert any(message.startswith("原牆模型補值") for message in measured), measured
    assert all(not message.startswith("原牆") for message in cast(list[str], paths["constant"]["notes"]))


def test_gradients_match_central_difference(tmp_path: Path) -> None:
    """三種原牆、每個變數、垂直與 60 度、80 與 1000 Hz：jit(grad) 對中央差分；每個變數至少一個探點兩邊都非零。"""
    rows = cast(list[dict[str, object]], _run("gradients", tmp_path))
    assert {cast(str, row["wall"]) for row in rows} == set(WALL_KINDS)
    nonzero: dict[str, set[str]] = {name: set() for name in WALL_KINDS}
    variables: dict[str, set[str]] = {name: set() for name in WALL_KINDS}
    for row in rows:
        automatic, central = float(cast(float, row["automatic_scaled"])), float(cast(float, row["central_scaled"]))
        assert np.isfinite(automatic) and np.isfinite(central), row
        np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS, err_msg=str(row))
        variables[cast(str, row["wall"])].add(cast(str, row["variable"]))
        if abs(automatic) > GRADIENT_ABS and abs(central) > GRADIENT_ABS:
            nonzero[cast(str, row["wall"])].add(cast(str, row["variable"]))
    for name in WALL_KINDS:
        assert variables[name] == {"layer1.thickness_m", "layer1.flow_resistivity_pa_s_m2", "layer2.thickness_m"}, name
        assert nonzero[name] == variables[name], (name, sorted(variables[name] - nonzero[name]))


def test_template_is_not_rigid_and_wall_errors_are_prefixed(tmp_path: Path) -> None:
    """樣板交給替身時不是剛性面；原牆的錯帶「原牆」前綴。"""
    rows = cast(dict[str, object], _run("rejections", tmp_path))
    assert cast(str, rows["bad_wall"]).startswith("原牆") and "超出手動試算範圍" in cast(str, rows["bad_wall"])
    assert rows["rigid_mask"] == [False, True, True, True, True, True]


@pytest.fixture(scope="module")
def live_wall(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], _run("live_wall", tmp_path_factory.mktemp("mounted-live-wall")))


def test_wall_load_is_computed_live_for_axis_and_air(live_wall: dict[str, dict[str, object]]) -> None:
    """原牆負載求值當下現算，三種原牆都一樣：
    換空氣後垂直、0 度、非可微入口三者一致（審查：垂直用舊快取、0 度用新空氣）；
    jit 裡換一條同長的軸（含實測表的表列頻點、表內外分界不同），垂直與 Paris 角度格都等於非可微入口在新軸上的值
    （審查：實測表原牆靜默用舊表內值）。實測表的表內內插是 JAX 現算，跟 numpy 那一份差在最後一兩位，所以照阻抗尺。"""
    for name in WALL_KINDS:
        row = live_wall[name]
        check = _close if name == "measured" else _same
        check(row["air_normal"], row["air_existing"], f"{name}:air_normal")
        _close(row["air_angle_zero"], row["air_existing"], f"{name}:air_angle_zero")
        _close(row["jit_other_axis"], row["other_axis_existing"], f"{name}:jit_other_axis")
        _close(row["jit_other_axis_angles"], row["other_axis_angles_existing"], f"{name}:jit_other_axis_angles")


def test_review_synthetic_measured_wall_follows_the_new_axis(live_wall: dict[str, dict[str, object]]) -> None:
    """審查的合成表（100／1000 Hz 兩列）：100 Hz 建樣板，jit 與具體值在 1000 Hz 求，都回原牆在 1000 Hz 的表值 2000（命中表列逐位）。"""
    row = live_wall["review_synthetic"]
    np.testing.assert_array_equal(_decode(row["wall_at_other_axis"]), [2000.0 + 0j])
    for key in ("jit_other_axis", "eager_other_axis"):
        _close(row[key], row["wall_at_other_axis"], key)


# 單精度的尺：四個 float32 機器精度（相對）。值本身經過單精度的對數與內插，雙精度那一份當真值。
SINGLE_REL = 4 * float(np.finfo(np.float32).eps)


def test_single_precision_close_knots_stay_finite_and_match_existing_entry(tmp_path: Path) -> None:
    """單精度下很近的表列（審查：1000 與 1000.0001220703125 Hz 取對數後相等、除以零出 NaN；另三案：兩列在單精度下
    捨入成同一個值、求值點落在兩列重合之間、表中間兩列重合、表的下端存不下；#1000 一族：審查原例 999.999995 與 1000.00002 Hz
    求 1000 Hz 不在中點（舊寫法固定取一半回 1500，既有入口 1200），以及兩列重合或差一兩個 ulp 時求兩列之間、命中、恰在外側）：
    直接與 jit、垂直與角度，值有限，而且新增層厚度 0 時等於既有入口（雙精度）同一頻點的原牆值、1 cm 時等於既有入口貼上新增層的值
    （都照單精度的尺）；對新增層厚度的梯度在厚度 0 與 1 cm 都有限。"""
    rows = cast(dict[str, dict[str, object]], _run("close_knots", tmp_path))
    assert set(rows) == {"review", "collide", "straddle", "middle", "edge", "asymmetric", "mirror", "one_ulp", "hit_one_ulp", "two_ulp"}
    for name, row in rows.items():
        results = cast(list[dict[str, object]], row["results"])
        assert len(results) == 8, name
        for result in results:
            bits = cast(dict[str, object], result["value"])
            assert bits["dtype"] == "complex64", (name, result)
            value = _decode(bits)
            assert np.all(np.isfinite(value)) and math.isfinite(cast(float, result["gradient"])), (name, result)
            # middle（#1000 改）：求值點的真值 1000 Hz 比 1000.00001 Hz 那一列小，答案是既有入口的 1999.99，不是那一列的表值 2000
            # （舊考卷釘的，相對差約 5e-6，超過單精度的尺）；挑錯段（照捨入後的表列找）會回 3000。
            key = ("existing" if result["thickness"] == 0.0 else "mounted") + ("_angles" if result["angles"] else "")
            np.testing.assert_allclose(value, _decode(row[key]).reshape(value.shape), rtol=SINGLE_REL, atol=0, err_msg=f"{name}:{key}:{result}")
    np.testing.assert_allclose(_decode(rows["review"]["existing"]), [1500.0], rtol=SINGLE_REL)
    np.testing.assert_allclose(_decode(rows["straddle"]["existing"]), [1500.0], rtol=SINGLE_REL)
    np.testing.assert_allclose(_decode(rows["asymmetric"]["existing"]), [1200.0], rtol=SINGLE_REL)
    assert abs(_decode(rows["edge"]["existing"])[0] - 1000.0) > 100.0  # 50 Hz 照真值在表外：既有入口走底構造，不是表值


# 改動前主線（fb6f92d0）在同一個子行程開法下量的值：雙精度一般表（probe 的 CONTROL_TABLE）表內七點（命中三列與各段之間）、新增層厚度 0，
# 從寫入端（construction_diff.surface_impedance_from_parameters）取，十六進位逐位；直接與 jit 在這七點逐位相同，兩種模式都對這一份。
# 表外兩點（50、1500 Hz）走底構造，直接與 jit 在虛部末幾位本來就不同（換 CPU 末位也可能變），不逐位釘，對既有入口照阻抗尺。
TABLE_CONTROL_INSIDE: dict[str, list[str]] = {
    "real": ["0x1.f400000000000p+9", "0x1.38790720d4c7fp+10", "0x1.76fdb10eadacbp+10", "0x1.7700000000000p+10",
             "0x1.9eaa9d08202afp+10", "0x1.e079329f74919p+10", "0x1.f400000000000p+10"],
    "imag": ["0x0.0p+0", "-0x1.8fd360d21e991p+5", "-0x1.8ff89cfbc55bdp+6", "-0x1.9000000000000p+6",
             "-0x1.077761a699de5p+7", "-0x1.70c1ea98ba828p+7", "-0x1.9000000000000p+7"],
}


def test_double_precision_table_inside_pinned_and_outside_matches_existing_entry(tmp_path: Path) -> None:
    """控制組（#1000）：雙精度一般表的原牆值，直接與 jit 兩種模式：表內七點（三個命中表列、各段之間）逐位等於改動前主線釘住的值；
    表外兩端（走底構造）對既有入口（雙精度）照傳遞矩陣移植考卷那把阻抗尺（同 _close）。"""
    rows = cast(dict[str, dict[str, object]], _run("table_control", tmp_path))
    outside = np.asarray(rows["outside"])
    assert outside.tolist() == [True, False, False, False, False, False, False, False, True]
    existing = _decode(rows["existing"])[outside]
    for mode in ("eager", "jit"):
        bits = rows[mode]
        assert bits["dtype"] == "complex128", mode
        value = _decode(bits)
        np.testing.assert_array_equal(value[~outside], _decode(TABLE_CONTROL_INSIDE), err_msg=f"{mode}:inside")
        for part in (np.real, np.imag):
            np.testing.assert_allclose(part(value[outside]), part(existing), rtol=REFERENCE_REL, atol=IMPEDANCE_ABS, err_msg=f"{mode}:outside")


def _note_cases(tmp_path: Path) -> dict[str, tuple[SurfaceMaterial, tuple[ConstructionNote, ...]]]:
    rows = cast(dict[str, dict[str, object]], _run("notes", tmp_path))
    return {name: (TypeAdapter(SurfaceMaterial).validate_python(row["material"]),
                   tuple(ConstructionNote.model_validate(note) for note in cast(list[object], row["notes"])))
            for name, row in rows.items()}


def test_wall_notes_have_their_own_layer_numbers_and_show_as_wall(tmp_path: Path) -> None:
    """原牆註記的層編號接著新增層往牆體側數：新增第 1 層與原牆第 1 層同號註記不再撞合併鍵與輸出驗證鍵，
    報表與網頁寫成「原牆第 1 層」；實測表原牆的補值註記兩個顯示端都照實測牆寫（不出「第 None 層」、不丟補值原因）。"""
    from types import SimpleNamespace

    from aosr.geometry.shoebox import Wall
    from aosr.gui.result_view import _construction_notes
    from aosr.physics.report_io import ReportInput, ReportOutput
    from aosr.reporting.result import SchemeResult
    from aosr.physics.report_measured_text import construction_note_lines
    from aosr.physics.wall_surfaces import merge_construction_notes

    cases = _note_cases(tmp_path)
    assert [(note.layer_index, note.code) for note in cases["fill"][1]] == [
        (None, "measured_table_model_fill"), (2, "komatsu_frequency_extrapolation")]
    collide = cases["collide"][1]
    assert [(note.layer_index, note.code, note.material_class) for note in collide] == [
        (1, "komatsu_frequency_extrapolation", "foam"), (1, "komatsu_material_class_outside_fit", "foam"),
        (2, "komatsu_frequency_extrapolation", "fibrous")]
    merged: dict[Wall, tuple[ConstructionNote, ...]] = {}
    merge_construction_notes(merged, Wall.X0, collide)
    assert merged[Wall.X0] == collide
    # 輸出驗證器那一支（同層同代號不可重複）直接拿來跑；改動前這三筆會被判重複。
    canonical = ReportOutput.__pydantic_decorators__.model_validators["_construction_notes_are_canonical"].func
    canonical(SimpleNamespace(construction_notes_by_wall={"x0": collide}))
    for name, expected_cli, expected_web in (
            ("collide", ["x0 第 1 層：Komatsu", "x0 第 1 層：泡棉", "x0 原牆第 1 層：Komatsu"],
             ["x 起點牆第 1 層：10 Hz 超出", "x 起點牆第 1 層：泡棉", "x 起點牆原牆第 1 層：10 Hz 超出"]),
            ("fill", ["x0：原牆模型補值：實測表涵蓋 50–500 Hz，表外以底構造計算；模型補值頻點 10 Hz", "x0 原牆底構造第 1 層：Komatsu"],
             ["x 起點牆：原牆模型補值：實測表涵蓋 50–500 Hz，表外以底構造計算；模型補值頻點 10 Hz",
              "x 起點牆原牆底構造第 1 層：10 Hz 超出"])):
        material, notes = cases[name]
        zones = {wall.wall_name(): (SimpleNamespace(material=material if wall is Wall.X0 else None),) for wall in Wall.all()}
        cli = construction_note_lines(cast(ReportInput, SimpleNamespace(surface_zones_by_wall=zones)), {"x0": notes})
        report = SimpleNamespace(construction_notes_by_wall={"x0": notes})
        web = _construction_notes(cast(SchemeResult, SimpleNamespace(
            pairs=(SimpleNamespace(report=report),), scheme=SimpleNamespace(scene=SimpleNamespace(surface_zones_by_wall=zones)))))
        assert len(cli) == len(expected_cli) + 1 and len(web) == len(expected_web), (cli, web)
        for line, start in (*zip(cli[1:], expected_cli), *zip(web, expected_web)):
            assert line.startswith(start) and "None" not in line, (line, start)


def test_three_lane_report_runs_with_a_mounted_wall(tmp_path: Path) -> None:
    """三路報表 x0 換成貼在原牆上的構造：有限元素收到的阻抗就是非可微入口的值，幾何路與晚期都有限且跟原牆不同。"""
    rows = cast(dict[str, dict[str, object]], _run("three_lane", tmp_path))
    fixed, mounted = rows["fixed"], rows["mounted"]
    assert fixed["capability"] == "real_frequency_independent_impedance"
    assert mounted["capability"] == "assembly_by_wall"
    assert mounted["x0_is_mounted"] is True
    for key in ("geometric", "late"):
        values = np.asarray(cast(list[float], mounted[key]))
        assert np.all(np.isfinite(values)) and np.all(values >= 0), key
        assert not np.array_equal(values, np.asarray(cast(list[float], fixed[key]))), key


def _mounted_document() -> dict[str, object]:
    return {"kind": "mounted_on_wall", "layers": cast(list[object], ASSEMBLY["layers"]),
            "wall": {"kind": "constant_impedance", "impedance_pa_s_m": 12348.0}}


@pytest.mark.parametrize("change", ("no_layers", "nested_wall", "missing_wall", "backing_field", "wall_kind_missing"))
def test_format_rejects_bad_mounted_documents(change: str) -> None:
    """零層、原牆又是貼牆構造（不遞迴）、沒有原牆、多給背後負載欄、原牆沒給種類代號，一律拒收。"""
    document = _mounted_document()
    if change == "no_layers":
        document["layers"] = []
    elif change == "nested_wall":
        document["wall"] = _mounted_document()
    elif change == "missing_wall":
        document.pop("wall")
    elif change == "backing_field":
        document["backing"] = "rigid"
    else:
        document["wall"] = {"impedance_pa_s_m": 12348.0}
    with pytest.raises(ValidationError):
        TypeAdapter(SurfaceMaterial).validate_python(document)


@pytest.mark.parametrize("wall", (
    {"kind": "constant_impedance", "impedance_pa_s_m": 12348.0},
    ASSEMBLY,
    {"kind": "measured_impedance", "points": [
        {"frequency_hz": 50.0, "real_pa_s_m": 3000.0, "imag_pa_s_m": -2000.0},
        {"frequency_hz": 500.0, "real_pa_s_m": 2000.0, "imag_pa_s_m": -100.0}],
     "base_assembly": ASSEMBLY, "source_and_conditions": "考卷假表：阻抗管、垂直入射"}))
def test_format_accepts_three_wall_kinds_and_scheme_round_trips(tmp_path: Path, wall: dict[str, object]) -> None:
    """三種原牆都收；方案存檔讀回逐欄相等，多層構造原樣、背後負載欄沒有被塞進新種類。"""
    document = zone_document()
    zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])
    zones["floor"][0]["material"] = _mounted_document() | {"wall": wall}
    scheme = Scheme.model_validate(document)
    material = cast(dict[str, list[object]], scheme.scene.surface_zones_by_wall)["floor"][0]
    assert isinstance(getattr(material, "material"), MountedOnWall)
    path = tmp_path / "scheme.json"
    path.write_text(scheme.model_dump_json(), encoding="utf-8")
    loaded = load_scheme(path)
    assert loaded == scheme
    dumped = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    floor = cast(dict[str, list[dict[str, dict[str, object]]]], cast(dict[str, object], dumped["scene"])[NEW])["floor"][0]["material"]
    assert set(floor) == {"kind", "layers", "wall"}
    assert TypeAdapter(SurfaceMaterial).validate_python(floor["wall"]) == TypeAdapter(SurfaceMaterial).validate_python(wall)


def test_assembly_model_is_unchanged() -> None:
    """多層構造的欄位、背後負載的選項一格不變；新種類沒有背後負載欄，原牆不收自己。"""
    assert list(Assembly.model_fields) == ["kind", "layers", "backing"]
    assert str(Assembly.model_fields["backing"].annotation) == "typing.Literal['rigid', 'air']"
    assert list(MountedOnWall.model_fields) == ["kind", "layers", "wall"]
    assert isinstance(TypeAdapter(SurfaceMaterial).validate_python(_mounted_document()), MountedOnWall)
    assert TypeAdapter(SurfaceMaterial).validate_python(_mounted_document()["wall"]) == ConstantImpedance(impedance_pa_s_m=12348.0)


def test_plain_summary_and_report_codes_for_mounted_wall() -> None:
    """網頁白話摘要不外露代號、原牆也寫出來；能力分類照原牆走；小松提醒看新增層與原牆。"""
    from aosr.geometry.shoebox import Wall
    from aosr.gui.problem_text import PATH_WORDS, material_form_spec, material_summary
    from aosr.physics.report_research_text import komatsu_research_notice
    from aosr.physics.report_surfaces import WholeWallZone, report_materials_code

    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    porous = {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 19100.0, "material_class": "fibrous"}
    mounted = adapter.validate_python(_mounted_document() | {"layers": [porous, {"kind": "air", "thickness_m": 0.05}]})
    summary = material_summary(mounted)
    label = cast(str, material_form_spec()["mounted_on_wall"]["label"])
    assert summary.startswith(label) and "原牆：" in summary and "12348" in summary
    for code in ("mounted_on_wall", "constant_impedance", "assembly", "porous", "air", "rigid", "fibrous", "komatsu", "None"):
        assert code not in summary, code
    assert PATH_WORDS["mounted_on_wall"] == label
    fixed = {wall: ConstantImpedance(impedance_pa_s_m=12348.0) for wall in Wall.all()}
    measured_wall = adapter.validate_python({"kind": "measured_impedance", "points": [
        {"frequency_hz": 50.0, "real_pa_s_m": 3000.0, "imag_pa_s_m": -2000.0},
        {"frequency_hz": 500.0, "real_pa_s_m": 2000.0, "imag_pa_s_m": -100.0}],
        "base_assembly": ASSEMBLY, "source_and_conditions": "考卷假表"})
    assert isinstance(mounted, MountedOnWall) and not isinstance(measured_wall, MountedOnWall | Assembly | ConstantImpedance)
    assert report_materials_code(fixed | {Wall.X0: mounted}) == "assembly_by_wall"
    assert report_materials_code(fixed | {Wall.X0: mounted.model_copy(update={"wall": measured_wall})}) == "measured_impedance_by_wall"
    board = adapter.validate_python(ASSEMBLY)
    assert isinstance(board, Assembly)
    komatsu_wall = adapter.validate_python({"kind": "assembly", "layers": [porous], "backing": "rigid"})
    cases = ((mounted, True), (mounted.model_copy(update={"layers": board.layers}), False),
             (mounted.model_copy(update={"layers": board.layers, "wall": komatsu_wall}), True))
    for material, expected in cases:
        zones: dict[str, tuple[WholeWallZone, ...]] = {wall.wall_name(): (WholeWallZone(area="whole_wall", material=item),)
                                                       for wall, item in (fixed | {Wall.X0: material}).items()}
        assert (komatsu_research_notice(zones) is not None) is expected, material


@pytest.mark.parametrize("wall", (
    {"kind": "constant_impedance", "impedance_pa_s_m": 12348.0},
    {"kind": "assembly", "layers": [{"kind": "board", "surface_mass_kg_m2": 8.0, "resistance_pa_s_m": 0.0},
                {"kind": "air", "thickness_m": 0.2}],
     "backing": "air"},
    {"kind": "measured_impedance", "points": [
        {"frequency_hz": 50.0, "real_pa_s_m": 3000.0, "imag_pa_s_m": -2000.0},
        {"frequency_hz": 500.0, "real_pa_s_m": 2000.0, "imag_pa_s_m": -100.0}],
     "base_assembly": ASSEMBLY, "source_and_conditions": "考卷假表"}))
def test_gradient_report_material_text_has_no_codes(wall: dict[str, object]) -> None:
    """梯度報告「不貼（保留原牆：…）」那一句：三種原牆貼上新增層、以及原牆自己，都不外露英文代號（小寫英文字一個都沒有）。"""
    import re

    from aosr.gradient.report import material_text

    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    porous = {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 19100.0, "material_class": "fibrous"}
    mounted = adapter.validate_python(_mounted_document() | {"layers": [porous, {"kind": "air", "thickness_m": 0.05}], "wall": wall})
    for material in (mounted, adapter.validate_python(wall)):
        text = material_text(material)
        assert re.findall(r"[a-z_]{2,}", text) == [], text
    assert material_text(mounted).startswith("貼在原牆上的構造（房間側起 多孔層、空氣層；原牆 ")
