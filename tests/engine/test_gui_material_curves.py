"""材料編輯頁的吸音曲線介面（#1082 段 2-2b）：曲線照現成物理、標語照送來的內文、提醒分兩級、不碰使用者材料庫。

曲線走 JAX 要雙精度，pytest 行程刻意不開（引擎考卷 conftest 寫了為什麼）：全部案例交給冷子行程
``_material_curve_probe.py`` 一次跑完——它在正式開機設定下打曲線介面，同一行程另直接呼叫物理當對照；這裡只比。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.app import GuiSettings, create_app
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.materials.library import (
    PROVENANCE_LABELS,
    FixedEntry,
    OptimizableEntry,
    basis_label,
    builtin_entries,
    template_assembly,
)
from aosr.physics.wall_surfaces import extrapolated_frequency_text
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins

PROBE = Path(__file__).with_name("_material_curve_probe.py")
CURVE = "/api/material-curve"
LANE = list(GEOMETRIC_LANE_FREQUENCIES_HZ)
TABLE_AXIS = sorted({*GEOMETRIC_LANE_FREQUENCIES_HZ, 100.0, 400.0})
AIR = load_physics_constants(config_path("physics_constants.toml"))
RANGES = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
Reply = dict[str, object]


def _porous(flow: float, material_class: str = "fibrous", model: str = "komatsu") -> dict[str, object]:
    return {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": flow, "material_class": material_class, "model": model}


def _fixed(material: dict[str, object], material_id: str = "my_panel") -> dict[str, object]:
    return {"kind": "fixed", "id": material_id, "label": "我的板", "provenance": "model", "source": "自己估的", "material": material}


def _assembly(*layers: dict[str, object]) -> dict[str, object]:
    return {"kind": "assembly", "backing": "rigid", "layers": list(layers)}


def _measured(imaginary: bool) -> dict[str, object]:
    rows = [{"frequency_hz": 100.0, "real_pa_s_m": 950.0}, {"frequency_hz": 400.0, "real_pa_s_m": 700.0}]
    if imaginary:
        rows = [row | {"imag_pa_s_m": -250.0} for row in rows]
    return _fixed({"kind": "measured_impedance", "points": rows, "source_and_conditions": "阻抗管、垂直入射",
                   "base_assembly": _assembly(_porous(12000.0), {"kind": "air", "thickness_m": 0.1})})


def _family_with_cavity() -> dict[str, object]:
    entry = next(entry for entry in builtin_entries() if isinstance(entry, OptimizableEntry) and entry.cavity is not None)
    document = entry.model_dump(mode="json") | {"id": "my_wool", "label": "我的棉", "preset_members": [], "template_preset": None}
    cast(dict[str, object], document["cavity"])["initial_m"] = 0.08
    return document


def _post(document: object) -> dict[str, object]:
    return {"method": "POST", "path": CURVE, "content": json.dumps(document, ensure_ascii=False)}


def _case(*requests: dict[str, object], physics: tuple[object, ...] = (), axis: list[float] = LANE) -> dict[str, object]:
    return {"requests": list(requests), "physics": list(physics), "axis": axis}


def _cases() -> dict[str, dict[str, object]]:
    builtins = builtin_entries()
    family = OptimizableEntry.model_validate(_family_with_cavity())
    with_cavity = template_assembly(family)
    renamed = _fixed(_assembly(_porous(30000.0)), "curve")
    return {
        "builtins": _case(*(_post(entry.model_dump(mode="json")) for entry in builtins), physics=tuple(
            (template_assembly(entry) if isinstance(entry, OptimizableEntry) else entry.surface_material()).model_dump(mode="json")
            for entry in builtins)),
        "family": _case(_post(_family_with_cavity()), physics=(
            with_cavity.model_dump(mode="json"),
            with_cavity.model_copy(update={"layers": with_cavity.layers[:1]}).model_dump(mode="json"))),
        "below_samples": _case(_post(_fixed(_assembly(_porous(5000.0))))),
        "foam": _case(_post(_fixed(_assembly(_porous(20000.0, "foam"))))),
        "manual_low": _case(_post(_fixed(_assembly(_porous(3000.0))))),
        "manual_high": _case(_post(_fixed(_assembly(_porous(80000.0))))),
        **{f"measured_{imaginary}": _case(_post(_measured(imaginary)), axis=TABLE_AXIS, physics=(
            FixedEntry.model_validate(_measured(imaginary)).surface_material().model_dump(mode="json"),))
           for imaginary in (True, False)},
        "constant": _case(_post(_fixed({"kind": "constant_impedance", "impedance_pa_s_m": 800.0}))),
        "miki": _case(_post(_fixed(_assembly(_porous(15000.0, model="miki"))))),
        "incomplete": _case(_post(_fixed(_assembly(_porous(15000.0))) | {"provenance": None, "source": ""})),
        "not_json": _case({"method": "POST", "path": CURVE, "content": "{ broken"}),
        # 存檔那一案放最後：前面每一案每一步之後材料庫檔都還不在，才證明得了曲線介面沒寫檔。
        "library": _case(
            _post(_fixed(_assembly(_porous(15000.0)))),
            {"method": "PUT", "path": "/api/materials/curve", "headers": {"If-None-Match": "*"},
             "content": json.dumps(_fixed(_assembly(_porous(15000.0)), "curve"), ensure_ascii=False)},
            _post(renamed), _post("一段字串，不是材料"),
            physics=(FixedEntry.model_validate(renamed).surface_material().model_dump(mode="json"),)),
    }


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    folder = tmp_path_factory.mktemp("material-curves")
    data_dir = folder / "data"
    data_dir.mkdir()
    done = subprocess.run([sys.executable, "-B", str(PROBE), str(data_dir)], input=json.dumps(_cases(), ensure_ascii=False),
                          capture_output=True, text=True, encoding="utf-8", cwd=folder, timeout=900, check=False,
                          env=forward_cpu_pins(child_process_env(threads=1)))
    assert done.returncode == 0, done.stderr[-4000:]
    return cast(dict[str, dict[str, object]], json.loads(done.stdout))


def _steps(probe: dict[str, dict[str, object]], name: str) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], probe[name]["steps"])


def _reply(probe: dict[str, dict[str, object]], name: str, index: int = 0) -> Reply:
    step = _steps(probe, name)[index]
    assert step["status"] == 200, step
    return cast(Reply, step["body"])


def _expected(probe: dict[str, dict[str, object]], name: str, index: int = 0) -> list[float]:
    return cast(list[list[float]], probe[name]["expected"])[index]


def _codes(reply: Reply) -> list[tuple[str, str]]:
    return [(str(line["level"]), str(line["code"])) for line in cast(list[dict[str, object]], reply["warnings"])]


def _text(reply: Reply, code: str) -> str:
    return next(str(line["text"]) for line in cast(list[dict[str, object]], reply["warnings"]) if line["code"] == code)


def _runs(axis: list[float], flagged: set[float]) -> list[tuple[float, float]]:
    """考卷自己切段：被標的頻點在軸上序號相連才算同一段。"""
    runs: list[list[int]] = []
    for index in (index for index, frequency in enumerate(axis) if frequency in flagged):
        if runs and runs[-1][-1] == index - 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return [(axis[run[0]], axis[run[-1]]) for run in runs]


def test_every_builtin_is_drawn_on_the_room_axis_with_exactly_what_the_physics_returns(probe: dict[str, dict[str, object]]) -> None:
    for index, entry in enumerate(builtin_entries()):
        reply = _reply(probe, "builtins", index)
        assert reply["status"] == "drawn", entry.id
        assert reply["frequencies_hz"] == LANE
        assert reply["absorption"] == _expected(probe, "builtins", index), entry.id
        assert f"空氣用產品設定的標準值（聲速 {AIR.sound_speed_m_s:g} m/s、密度 {AIR.air_density_kg_m3:g} kg/m³）" in str(reply["axis_text"])
        assert (reply["basis"], reply["provenance"], reply["source"], reply["links"]) == (
            basis_label(entry), PROVENANCE_LABELS[entry.provenance], entry.source, list(entry.links))
        assert (reply["arrangement"] is None) == isinstance(entry, FixedEntry)


def test_an_optimizable_material_draws_its_initial_arrangement_and_says_it_is_one_of_a_family(
    probe: dict[str, dict[str, object]],
) -> None:
    reply = _reply(probe, "family")
    assert reply["absorption"] == _expected(probe, "family", 0)
    assert reply["absorption"] != _expected(probe, "family", 1)  # 漏了初值空腔、只畫多孔層會紅
    porous = cast(dict[str, object], _family_with_cavity()["porous"])
    assert f"多孔層 {porous['thickness_m']:g} m＋空腔 0.08 m" in str(reply["arrangement"])
    assert "一族" in str(reply["arrangement"]) and "起點" in str(reply["arrangement"])


def test_frequency_extrapolation_is_an_ordinary_note_with_its_bands_shaded(probe: dict[str, dict[str, object]]) -> None:
    index, entry = next((index, entry) for index, entry in enumerate(builtin_entries()) if entry.id == "glass_wool_48k")
    assert isinstance(entry, OptimizableEntry)
    lower, upper = RANGES.komatsu.f_over_sigma_bounds_m3_kg
    flagged = {f for f in LANE if not lower <= f / entry.porous.flow_resistivity_pa_s_m2 <= upper}
    reply = _reply(probe, "builtins", index)
    assert _codes(reply) == [("note", "komatsu_frequency_extrapolation")]
    assert reply["spans"] == [{"low_hz": low, "high_hz": high, "kind": "extrapolation"} for low, high in _runs(LANE, flagged)]
    text = _text(reply, "komatsu_frequency_extrapolation")
    assert f"第 1 層（小松纖維經驗式）在 {extrapolated_frequency_text(tuple(sorted(flagged)))} 是材料模型外推" in text
    assert "Komatsu" not in text


@pytest.mark.parametrize(("case", "strong"), [("below_samples", "komatsu_flow_below_samples"),
                                              ("foam", "komatsu_material_class_outside_fit")])
def test_parameters_outside_the_study_are_strong_warnings_listed_first(
    probe: dict[str, dict[str, object]], case: str, strong: str,
) -> None:
    reply = _reply(probe, case)
    assert reply["status"] == "drawn"
    assert _codes(reply)[0] == ("strong", strong)
    assert ("note", "komatsu_frequency_extrapolation") in _codes(reply)
    assert "Komatsu" not in _text(reply, strong)


@pytest.mark.parametrize("case", ["manual_low", "manual_high"])
def test_a_flow_outside_the_manual_range_cannot_be_drawn_and_says_why(probe: dict[str, dict[str, object]], case: str) -> None:
    reply = _reply(probe, case)
    assert reply["status"] == "uncomputable"
    assert "absorption" not in reply
    assert str(reply["message"]).startswith("這組參數算不出吸音曲線：第 1 層：小松纖維經驗式流阻")
    assert "超出手動試算範圍" in str(reply["message"])
    assert reply["basis"] == "模擬，之後要拿實測對照"


@pytest.mark.parametrize(("imaginary", "basis", "code"), [(True, "實測", "measured"), (False, "近似", "approximate")])
def test_a_measured_table_marks_what_is_measured_and_what_is_model_fill(
    probe: dict[str, dict[str, object]], imaginary: bool, basis: str, code: str,
) -> None:
    reply = _reply(probe, f"measured_{imaginary}")
    assert (reply["basis"], reply["basis_code"]) == (basis, code)
    assert ("虛部當 0" in str(reply["basis_detail"])) == (not imaginary)
    assert reply["frequencies_hz"] == TABLE_AXIS
    assert reply["absorption"] == _expected(probe, f"measured_{imaginary}")
    assert reply["measured_range_hz"] == [100.0, 400.0]
    fill = [span for span in cast(list[dict[str, object]], reply["spans"]) if span["kind"] == "model_fill"]
    assert [(span["low_hz"], span["high_hz"]) for span in fill] == _runs(TABLE_AXIS, {f for f in TABLE_AXIS if f < 100.0 or f > 400.0})
    assert len(fill) == 2
    assert _text(reply, "measured_table_model_fill").startswith("實測表只涵蓋 100–400 Hz；")
    assert "底構造第 1 層（小松纖維經驗式）" in _text(reply, "komatsu_frequency_extrapolation")


def test_a_constant_impedance_is_approximate_and_flat(probe: dict[str, dict[str, object]]) -> None:
    reply = _reply(probe, "constant")
    assert (reply["basis"], reply["spans"], reply["warnings"]) == ("近似", [], [])
    assert set(cast(list[float], reply["absorption"])) == {complex_random_incidence_absorption(800.0 / AIR.rho_c)}


def test_a_porous_model_without_registered_ranges_says_so_instead_of_staying_silent(probe: dict[str, dict[str, object]]) -> None:
    reply = _reply(probe, "miki")
    assert _codes(reply) == [("note", "model_range_not_registered")]
    assert _text(reply, "model_range_not_registered") == "第 1 層用三木纖維經驗式：這個模型沒有登記可信範圍，這頁沒有範圍可查。"
    assert "沒有提醒不代表在可信範圍內" in str(reply["range_scope"])


def test_an_unfinished_form_is_reported_in_the_form_words_not_drawn(probe: dict[str, dict[str, object]]) -> None:
    reply = _reply(probe, "incomplete")
    assert reply["status"] == "incomplete"
    assert reply["message"] == "表單還沒填齊，曲線畫不出來：數字從哪來：必填；出處與條件：不可空白"
    assert "basis" not in reply


def test_curves_never_write_the_user_library_and_curve_is_still_a_usable_material_id(probe: dict[str, dict[str, object]]) -> None:
    assert list(probe)[-1] == "library"
    assert {step["library"] for name in list(probe)[:-1] for step in _steps(probe, name)} == {None}
    first, saved, renamed, text = _steps(probe, "library")
    assert (first["status"], first["library"]) == (200, None)
    assert saved["status"] == 200 and saved["library"] is not None
    assert cast(Reply, renamed["body"])["absorption"] == _expected(probe, "library")
    assert cast(Reply, text["body"])["status"] == "incomplete"
    assert renamed["library"] == text["library"] == saved["library"]


def test_a_body_that_is_not_json_is_a_plain_400(probe: dict[str, dict[str, object]]) -> None:
    step = _steps(probe, "not_json")[0]
    assert (step["status"], step["body"]) == (400, {"error": "內文不是有效 JSON"})


def test_a_server_without_double_precision_blames_its_start_up_not_the_material(tmp_path: Path) -> None:
    """pytest 行程沒開雙精度（引擎 conftest 刻意的）：同程式起的網頁算不出曲線，要講是開機設定，不說成參數錯。"""
    entry = next(entry for entry in builtin_entries() if entry.id == "glass_wool_48k")
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost") as client:
        reply = client.post(CURVE, json=entry.model_dump(mode="json"))
    body = cast(Reply, reply.json())
    assert (reply.status_code, body["status"], body["basis"]) == (200, "no_precision", "模擬，之後要拿實測對照")
    assert body["message"] == "這個網頁行程沒有開雙精度，算不出吸音曲線；這是網頁開機設定的問題，不是材料參數的問題。"
