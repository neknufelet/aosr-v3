"""矩形分區方案：還沒接的計算路照舊明確拒收；網頁載入、表單、驗證、保存收下而且每一區都留著（#1083）。

只考 Python，不開瀏覽器；收下的考卷一律改最後一區（不是第一區），抓「只讀第一區」的路。
"""
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics import report_io, report_surfaces
from aosr.physics.report_measured import measured_seams
from aosr.physics.report_measured_text import construction_note_lines, measured_text
from aosr.reporting.modal_lookup import key_from_scheme, scheme_scope_reason
from aosr.reporting.scheme import Scheme
from aosr.reporting.stage_one_result import check_stage_one_prerequisites
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._surface_zone_cases import rectangle_document


@pytest.mark.parametrize("entry", ["gradient-endpoint", "seams", "text", "notes"])
def test_calculating_and_display_entries_refuse_rectangles(tmp_path: Path, entry: str) -> None:
    """抓任一路靜默取第一區；終點換材料也不能把分區覆蓋掉。"""
    scheme = Scheme.model_validate(rectangle_document())
    inputs = report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY)
    with pytest.raises(report_surfaces.RectangleZonesUnsupported, match=r"ceiling.*分區還不支援這一頁／這一路"):
        if entry == "gradient-endpoint":
            endpoint.scene_with_materials(scheme.scene, {wall: ConstantImpedance(impedance_pa_s_m=1000.0) for wall in Wall.all()})
        elif entry == "seams":
            measured_seams(inputs.surface_zones_by_wall, density_kg_m3=1.2, sound_speed_m_s=343.0)
        elif entry == "text":
            measured_text(inputs)
        else:
            construction_note_lines(inputs, {})


def test_existing_whole_construction_guards_also_refuse_rectangles() -> None:
    """模態與第一階段原本整份拒收構造，保持原界線。"""
    scheme = Scheme.model_validate(rectangle_document())
    assert scheme_scope_reason(scheme) == "construction_walls"
    assert key_from_scheme(scheme) is None
    scheme = scheme.model_copy(update={"source_model": "omnidirectional", "furniture": None})
    with pytest.raises(ValueError, match="第一階段不收牆面分區"):
        check_stage_one_prerequisites(scheme)


def test_web_python_load_reads_rectangle_file_with_every_zone(tmp_path: Path) -> None:
    """讀磁碟上的分區方案（#1083）：每一區原樣讀回，最後一區改過的材料也在，不是只留第一區。"""
    from aosr.gui.app import _read_scheme

    document = _last_zone_changed()
    path = tmp_path / "rectangles.json"
    path.write_text(Scheme.model_validate(document).model_dump_json())
    zones = _read_scheme(path).scene.surface_zones_by_wall or {}
    assert [zone.model_dump(mode="json") for zone in zones["ceiling"]] == _scene_zones(document)["ceiling"]
    assert _last_impedance(zones["ceiling"][-1].model_dump(mode="json")) == LAST_IMPEDANCE


def test_search_refuses_rectangles_before_creating_folder(tmp_path: Path) -> None:
    """搜尋建立時就拒收，不能先寫成等待計算的候選。"""
    from tests.engine.test_scheme_furniture import _store

    with pytest.raises(report_surfaces.RectangleZonesUnsupported, match="ceiling.*分區還不支援這一頁／這一路"):
        _store(tmp_path / "search", Scheme.model_validate(rectangle_document()))
    assert not (tmp_path / "search").exists()


LAST_IMPEDANCE = 1234.5


def _scene_zones(document: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    return cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])["surface_zones_by_wall"])


def _last_impedance(zone: dict[str, object]) -> object:
    return cast(dict[str, object], zone["material"])["impedance_pa_s_m"]


def _last_zone_changed() -> dict[str, object]:
    """天花最後一區（r1_c1，不是第一區）的阻抗換掉，其餘照分區樣本。"""
    document = rectangle_document()
    cast(dict[str, object], _scene_zones(document)["ceiling"][-1]["material"])["impedance_pa_s_m"] = LAST_IMPEDANCE
    return document


def _komatsu_out_of_range_last_zone() -> dict[str, object]:
    """天花最後一區換成流阻遠低於小松模型範圍的吸音層：範圍檢查要點名第 4 區、白話要寫分區代號。"""
    document = rectangle_document()
    _scene_zones(document)["ceiling"][-1]["material"] = {"kind": "assembly", "backing": "rigid", "layers": [
        {"kind": "porous", "thickness_m": .05, "model": "komatsu", "material_class": "fibrous",
         "flow_resistivity_pa_s_m2": 1.0}]}
    return document


@pytest.mark.parametrize("entry", ["web-load", "preview", "edit", "validate"])
def test_web_form_python_entry_accepts_rectangles_and_keeps_every_zone(entry: str) -> None:
    """表單載入／編輯／驗證收下分區方案（#1083）：編輯後每一區原樣、驗證不報問題。"""
    from aosr.gui.construction_input import web_scheme_problems
    from aosr.gui.input_setup import InputEdit, edit_input, input_preview
    from aosr.gui.problem_text import checked_scheme

    document = _last_zone_changed()
    if entry == "web-load":
        assert _scene_zones(checked_scheme(deepcopy(document)).model_dump(mode="json")) == _scene_zones(document)
    elif entry == "validate":
        assert web_scheme_problems(deepcopy(document), capabilities=CAPABILITIES, directivity=DIRECTIVITY) == ()
    elif entry == "preview":
        assert input_preview(deepcopy(document), CAPABILITIES, DIRECTIVITY)["field_values"]
    else:
        edited = edit_input(InputEdit(scheme=deepcopy(document), action="shortcut", value="listening"), CAPABILITIES, DIRECTIVITY)
        assert _scene_zones(edited) == _scene_zones(document)


def test_material_range_check_names_the_zone_it_found() -> None:
    """流阻範圍檢查掃每一區：最後一區超出範圍要點名第 4 區（路徑 .3.），白話寫出分區代號 r1_c1。"""
    from aosr.gui.construction_input import web_scheme_problems
    from aosr.gui.problem_text import plain_problems

    document = _komatsu_out_of_range_last_zone()
    problems = web_scheme_problems(document, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert [problem.path for problem in problems] == [
        "scene.surface_zones_by_wall.ceiling.3.material.assembly.layers.0.porous.flow_resistivity_pa_s_m2"]
    assert plain_problems(problems, document)[0]["fields"] == ["天花材料構造 › 分區 r1_c1 › 第 1 層 › 流阻，Pa·s/m²"]


def test_wall_preset_refuses_a_zoned_wall_and_leaves_it_alone() -> None:
    """構造預設只換整面一區的牆；分區的牆明確拒收，不會把第一區換掉。"""
    from aosr.gui.input_setup import InputEdit, InputShapeError, edit_input, input_defaults

    preset = cast(list[dict[str, object]], input_defaults()["construction_presets"])[0]["id"]
    document = _last_zone_changed()
    before = deepcopy(document)
    with pytest.raises(InputShapeError, match="這一面已經分區"):
        edit_input(InputEdit(scheme=document, action="wall_preset", value=str(preset), wall="ceiling"), CAPABILITIES, DIRECTIVITY)
    assert document == before


async def _web_requests(tmp_path: Path) -> None:
    """同一事件迴圈內使用正式 ASGI（網頁呼叫介面），不開 socket 或瀏覽器。"""
    import httpx
    from aosr.gui.app import GuiSettings, create_app

    # 開計算的那一支換成馬上結束的假程式：這裡只考網頁門口收不收，真的計算走 scheme_cli run（#1062 已算過 109 分區）。
    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path, runner=(sys.executable, "-c", "pass")))
    document = _last_zone_changed()
    (tmp_path / "schemes" / "rectangles.json").write_text(Scheme.model_validate(document).model_dump_json())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        loaded = await client.get("/api/schemes/rectangles")
        assert loaded.status_code == 200, loaded.text
        assert _scene_zones(loaded.json()["scheme"]) == _scene_zones(document)
        for response in [await client.post("/api/input-preview", json=document),
                         await client.post("/api/input-edit", json={"scheme": document, "action": "shortcut", "value": "listening"}),
                         await client.post("/api/validate", json=document)]:
            assert response.status_code == 200, response.text
        checked = (await client.post("/api/validate", json=document)).json()
        assert checked["problems"] == [] and set(checked["impedance_labels"]) == {"floor"}
        for planned in (await client.post("/api/plan", json=document), await client.get("/api/plan/rectangles")):
            assert planned.status_code == 200, planned.text
        started = await client.post("/api/runs", json={"scheme_id": "rectangles"})
        assert started.status_code == 200 and "run_id" in started.json(), started.text
        bad = await client.post("/api/validate", json=_komatsu_out_of_range_last_zone())
        assert "分區 r1_c1" in bad.text, bad.text
        saved = await client.put("/api/schemes/new", json=document | {"scheme_id": "new"})
        assert saved.status_code == 200, saved.text
    on_disk = Scheme.model_validate_json((tmp_path / "schemes" / "new.json").read_text()).model_dump(mode="json")
    assert _scene_zones(on_disk) == _scene_zones(document)


def test_web_requests_accept_rectangles_without_browser(tmp_path: Path) -> None:
    """正式載入、預覽、編輯、驗證及保存端點都收分區方案；存檔裡最後一區改過的值還在。"""
    import asyncio

    asyncio.run(_web_requests(tmp_path))


def test_result_page_notes_on_a_zoned_wall_say_zoned() -> None:
    """結果頁的材料外推提醒：分區的牆不拿第一區的層去對，照註記原文寫「（分區）」。"""
    from aosr.gui.result_view import _construction_notes
    from aosr.materials.construction_models import ConstructionNote
    from aosr.reporting.result import SchemeResult

    note = ConstructionNote(code="komatsu_frequency_extrapolation", layer_index=1, frequencies_hz=(20.0,),
                            material_class="fibrous", message="第 1 層：頻率外推照算")
    report = SimpleNamespace(construction_notes_by_wall={"ceiling": (note,)})
    result = SimpleNamespace(pairs=[SimpleNamespace(report=report)], scheme=Scheme.model_validate(rectangle_document()))
    lines = _construction_notes(cast(SchemeResult, result))
    assert len(lines) == 1 and lines[0].startswith("天花（分區）第 1 層：") and "外推照算" in lines[0]


def test_compare_lists_a_zone_edge_moved_by_a_micrometre_next_to_other_changes() -> None:
    """審查第 A 條：比較照完整座標比，不先縮成六位有效數字；縮完字一樣時印完整座標。"""
    from aosr.gui.compare_view import scheme_differences

    before = rectangle_document()
    after = deepcopy(before)
    for zone in _scene_zones(after)["x0"]:
        zone["u_m"] = [2.000001 if edge == 2.0 else edge for edge in cast(list[float], zone["u_m"])]
    after["purpose"] = "other_purpose"
    rows = {row.path: row for row in scheme_differences(Scheme.model_validate(before), Scheme.model_validate(after))}
    last = rows["scene.surface_zones_by_wall.x0.r1_c1"]
    assert "2.000001" in last.b_text and "2.0–" in last.a_text and "purpose" in rows
