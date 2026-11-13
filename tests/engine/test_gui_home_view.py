"""首頁第一屏（#1134）：最近算的那間房怎麼畫、三個入口連去哪、存著的方案每份一句白話。

俯視圖的方向照牆名那一條（wall_words）：前牆朝上、面向前牆時的左手邊在左。只考 109 分不出「推出來的」與
「寫死 x0 朝上」，所以同一間房轉 90、180、270 度再考一次。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, expect
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.home_view import (LAYOUT_NOTE, NO_WHOLE_LAYOUT_NOTE, ZONED_LAYOUT_NOTE, home_data, home_view,
                                latest_picks, name_words, room_overview, scheme_line)
from aosr.gui.room_words import treated, treated_words
from aosr.gui.result_list import ResultSummary
from aosr.gui.wall_words import wall_words
from aosr.materials.construction_models import Assembly
from aosr.materials.library import builtin_entries
from aosr.reporting.scheme import Scheme
from tests.engine._surface_zone_cases import ROOM_109, room109_quarters
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser
from tests.engine.test_gui_wall_words import Doc, _part2d_layout, _turned

MOUNTED: dict[str, object] = {"kind": "mounted_on_wall", "layers": [
    {"kind": "porous", "thickness_m": 0.1, "flow_resistivity_pa_s_m2": 19100.0, "flow_resistivity_source": "density_table",
     "model": "komatsu", "material_class": "fibrous"}, {"kind": "air", "thickness_m": 0.02}],
    "wall": {"kind": "constant_impedance", "impedance_pa_s_m": 100000.0}}


def _zones(document: Doc) -> dict[str, list[dict[str, object]]]:
    return cast(dict[str, list[dict[str, object]]], cast(Doc, document["scene"])["surface_zones_by_wall"])


def _at(item: dict[str, object], key: str) -> dict[str, float]:
    return cast(dict[str, float], item[key])


def _items(room: dict[str, object], key: str) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], room[key])


def _whole(document: Doc, wall: str) -> None:
    _zones(document)[wall] = [{"area": "whole_wall", "material": copy.deepcopy(MOUNTED)}]


def test_auto_saved_names_read_as_plain_words() -> None:
    assert name_words("room109demo-part2d") == ("「room109demo-part2d」", False)
    baseline, auto = name_words("room109demo-part2d-orig-3d07ea99")
    assert auto and baseline.startswith("「room109demo-part2d」拆掉貼上去的東西")
    adopted, auto = name_words("room109demo-search-c98c50b8")
    assert auto and adopted == "「room109demo」第三階段搜尋選出來的那一套（電腦自動存的）"
    refined, auto = name_words(f"{'a' * 32}-trial-000012-verification")
    assert auto and "第 12 號候選擺法" in refined
    assert name_words(f"{'a' * 32}-baseline-verification")[0].startswith("擺位搜尋：原本擺法")
    # 使用者自己取的名字剛好長得像：後面不是 8 碼十六進位就不算。
    assert name_words("room-orig-notahash") == ("「room-orig-notahash」", False)


def test_latest_skips_baselines_refines_and_unfinished_runs() -> None:
    rows = [("a-orig-3d07ea99", "r1", "/results/r1", "done"),
            (f"{'b' * 32}-baseline-verification", "r2", "/results/r2", "done"),
            ("mine", "r3", "/results/r3", "failed"),
            ("mine", "r3s", "/results/r3s", "stopped"),
            ("mine", "r3r", "/results/r3r", "running"),
            ("mine", "r4", "", "done"),
            ("mine-search-c98c50b8", "r5", "/results/r5", "none"),
            ("older", "r6", "/results/r6", "done")]
    assert list(latest_picks(rows)) == [("r5", "/results/r5"), ("r6", "/results/r6")]
    assert not list(latest_picks(rows[:6]))


def test_part2d_layout_draws_front_wall_on_top_and_left_wall_on_the_left() -> None:
    document = _part2d_layout()
    room109_quarters(document, "x0", (1e5, 1e5, 1e5, 1e5))
    _zones(document)["x0"][2]["material"] = copy.deepcopy(MOUNTED)  # u0v0, u0v1, u1v0…：第三塊是 u 後半、下半
    _whole(document, "y0")
    scheme = Scheme.model_validate(document)
    assert treated_words(scheme) == "貼了吸音材料：前牆 1 塊、左牆整面"
    room = room_overview(scheme, builtin_entries())
    assert (room["width_m"], room["depth_m"]) == (ROOM_109["Ly"], ROOM_109["Lx"])
    walls = {str(item["wall"]): item for item in _items(room, "walls")}
    assert _at(walls["x0"], "at")["y"] == 0 and walls["x0"]["name"] == "前牆"
    assert _at(walls["y0"], "at")["x"] == 0 and walls["y0"]["name"] == "左牆"
    segments = _items(room, "segments")
    front = next(item for item in segments if item["wall"] == "x0")
    # 前牆那一塊是 x0 上 u（沿 y）後半：畫面上前牆朝上，y0（左牆）在左，所以在右半邊。
    assert _at(front, "from")["y"] == _at(front, "to")["y"] == 0
    assert sorted((_at(front, "from")["x"], _at(front, "to")["x"])) == pytest.approx([ROOM_109["Ly"] / 2, ROOM_109["Ly"]])
    left = next(item for item in segments if item["wall"] == "y0")
    assert _at(left, "from")["x"] == _at(left, "to")["x"] == 0
    # 左聲道喇叭畫在左半邊、比主位靠前牆（畫面上方）。
    speakers = {str(item["name"]): _at(item, "at") for item in _items(room, "speakers")}
    primary = _at(room, "primary")
    assert speakers["左聲道喇叭"]["x"] < primary["x"] < speakers["右聲道喇叭"]["x"]
    assert speakers["左聲道喇叭"]["y"] < primary["y"]


MEASURED: dict[str, object] = {"kind": "measured_impedance", "points": [
    {"frequency_hz": 140.0, "real_pa_s_m": 1097.5, "imag_pa_s_m": -2404.1},
    {"frequency_hz": 144.0, "real_pa_s_m": 1086.6, "imag_pa_s_m": -2292.1}],
    "base_assembly": {"kind": "assembly", "layers": [{"kind": "porous", "thickness_m": 0.049, "flow_resistivity_pa_s_m2": 76100.0,
                                                      "model": "wilson_2015", "material_class": "fibrous", "porosity": 0.97,
                                                      "tortuosity": 1.5}], "backing": "rigid"},
    "source_and_conditions": "考卷用的兩列", "fill_rule": "log_linear_table_base_assembly_outside_v1"}


def test_measured_tables_draw_blue_and_plain_boards_count_as_the_original_wall() -> None:
    document = _part2d_layout()
    _zones(document)["yL"] = [{"area": "whole_wall", "material": copy.deepcopy(MEASURED)}]
    _zones(document)["ceiling"] = [{"area": "whole_wall", "material": {"kind": "assembly", "layers": [
        {"kind": "board", "surface_mass_kg_m2": 5.8, "resistance_pa_s_m": 40.0}, {"kind": "air", "thickness_m": 0.2}],
        "backing": "rigid"}}]
    scheme = Scheme.model_validate(document)
    assert treated_words(scheme) == "貼了吸音材料：右牆整面"
    segments = _items(room_overview(scheme, builtin_entries()), "segments")
    assert [(item["wall"], item["basis"]) for item in segments] == [("yL", "measured")]
    # 板＋空腔以外的層（這裡是吸音棉）直接取代整面牆也算有貼；只有板＋空腔（109 的天花）不算。
    porous = Assembly.model_validate(MEASURED["base_assembly"])
    board = Assembly.model_validate({"kind": "assembly", "backing": "rigid", "layers": [
        {"kind": "board", "surface_mass_kg_m2": 5.8, "resistance_pa_s_m": 40.0}, {"kind": "air", "thickness_m": 0.2}]})
    assert treated(porous) and not treated(board)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_turned_room_still_draws_front_on_top_and_left_on_the_left(turns: int) -> None:
    probe = _turned(_part2d_layout(), turns)
    names = {word: wall for wall, word in wall_words(probe).items()}
    document = _part2d_layout()
    _whole(document, names["前牆"])
    _whole(document, names["左牆"])
    scheme = _turned(document, turns)
    room = room_overview(scheme, builtin_entries())
    segments = {str(item["wall"]): item for item in _items(room, "segments")}
    front, left = segments[names["前牆"]], segments[names["左牆"]]
    assert _at(front, "from")["y"] == pytest.approx(0) and _at(front, "to")["y"] == pytest.approx(0)
    assert _at(left, "from")["x"] == pytest.approx(0) and _at(left, "to")["x"] == pytest.approx(0)
    speakers = {str(item["name"]): _at(item, "at") for item in _items(room, "speakers")}
    assert speakers["左聲道喇叭"]["x"] < speakers["右聲道喇叭"]["x"]


def test_links_follow_whether_the_latest_scheme_is_saved_and_zoned() -> None:
    document = _part2d_layout()
    room109_quarters(document, "x0", (1e5, 1e5, 1e5, 1e5))
    zoned = Scheme.model_validate(document | {"scheme_id": "cut"})
    whole = Scheme.model_validate(_part2d_layout() | {"scheme_id": "whole"})
    library = builtin_entries()
    view = home_view(("/results/r1", zoned), {"cut": zoned}, library)
    links = cast(dict[str, dict[str, str]], view["links"])
    assert links["result"]["url"] == "/results/r1"
    assert links["materials"]["url"] == "/gradient?scheme=cut"
    assert (links["layout"]["url"], links["layout"]["note"]) == ("/search-start", NO_WHOLE_LAYOUT_NOTE)
    links = cast(dict[str, dict[str, str]], home_view(("/results/r1", zoned), {"cut": zoned, "whole": whole},
                                                      library)["links"])
    assert links["layout"]["note"] == ZONED_LAYOUT_NOTE
    links = cast(dict[str, dict[str, str]], home_view(("/results/r2", whole), {"whole": whole}, library)["links"])
    assert (links["layout"]["url"], links["layout"]["note"]) == ("/search-start?scheme=whole", LAYOUT_NOTE)
    assert links["materials"]["url"] == "/gradient"
    # 結果的方案已經不在存檔裡（封存了）：入口不帶名字。
    links = cast(dict[str, dict[str, str]], home_view(("/results/r2", whole), {}, library)["links"])
    assert links["layout"]["url"] == "/search-start"
    empty = home_view(None, {}, library)
    assert empty["room"] is None and cast(dict[str, dict[str, str]], empty["links"])["result"]["url"] == ""


def test_scheme_line_says_cut_treated_and_searched() -> None:
    document = _part2d_layout()
    room109_quarters(document, "x0", (1e5, 1e5, 1e5, 1e5))
    scheme = Scheme.model_validate(document | {"scheme_id": "mine"})
    line = scheme_line(scheme, {"mine": scheme, "mine-search-c98c50b8": scheme})
    assert line["text"] == "牆面分成 4 區；牆面都是原樣，沒貼吸音材料；電腦挑過材料（第三階段搜尋）"
    assert line["search_start_refused"] and not line["auto"]


def test_home_api_without_results_lists_saved_schemes(tmp_path: Path) -> None:
    (tmp_path / "schemes").mkdir()
    document = _part2d_layout() | {"scheme_id": "mine"}
    _whole(document, "xL")
    (tmp_path / "schemes" / "mine.json").write_text(Scheme.model_validate(document).model_dump_json(), encoding="utf-8")
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost")
    reply = client.get("/api/home")
    assert reply.status_code == 200, reply.text
    data = reply.json()
    assert data["title"] == "還沒有算好的房間" and data["room"] is None
    assert [(item["name"], item["text"]) for item in data["schemes"]] == [("mine", "貼了吸音材料：後牆整面")]


def test_home_first_screen_opens_a_saved_scheme_into_the_form(tmp_path: Path, browser: Browser) -> None:
    (tmp_path / "schemes").mkdir()
    document = _part2d_layout() | {"scheme_id": "mine"}
    _whole(document, "xL")
    (tmp_path / "schemes" / "mine.json").write_text(Scheme.model_validate(document).model_dump_json(), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        expect(page.locator("#home-title")).to_have_text("還沒有算好的房間")
        expect(page.locator("#home-result")).to_be_hidden()
        expect(page.locator("#home-materials")).to_be_visible()
        row = page.locator('#home-schemes li[data-scheme="mine"]')
        expect(row).to_contain_text("貼了吸音材料：後牆整面")
        row.get_by_role("button", name="打開來改").click()
        expect(page.locator("#save-id")).to_have_value("mine")
        _assert_quiet(watched)


def test_home_data_reads_the_latest_result_snapshot_not_the_saved_scheme(tmp_path: Path) -> None:
    """最近算的房間畫的是結果檔裡那一份方案快照（存檔之後又改過也照算的那一份畫），基準那一份跳過。"""
    (tmp_path / "results").mkdir()
    computed = _part2d_layout() | {"scheme_id": "mine"}
    _whole(computed, "xL")
    (tmp_path / "results" / "r1.json").write_text(
        json.dumps({"scheme": Scheme.model_validate(computed).model_dump(mode="json")}), encoding="utf-8")
    saved = Scheme.model_validate(_part2d_layout() | {"scheme_id": "mine"})

    def row(scheme_id: str, run_id: str) -> ResultSummary:
        return ResultSummary(run_id=run_id, scheme_id=scheme_id, finished_text="", duration_text="", calculation_text="",
                             calculation_detail="", registry_text="", result_url=f"/results/{run_id}", run_status="done")

    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "mine.json").write_text(saved.model_dump_json(), encoding="utf-8")
    # 最新那一份讀不出（例如格式改版）：往下一份找，不說成「還沒有算好的」。
    (tmp_path / "results" / "r2.json").write_text("{}", encoding="utf-8")

    class Source:
        data_dir = tmp_path

        def _scheme_names(self) -> list[str]:
            return ["mine"]

        def _result_summaries(self, archived: bool = False) -> list[ResultSummary]:
            return [row("mine", "r2"), row("mine-orig-3d07ea99", "r0"), row("mine", "r1")]

    data = home_data(Source())
    assert [item["name"] for item in cast(list[dict[str, object]], data["schemes"])] == ["mine"]
    assert data["title"] == "最近算的房間：「mine」"
    assert str(data["facts"]).endswith("貼了吸音材料：後牆整面")
    assert cast(dict[str, dict[str, str]], data["links"])["result"]["url"] == "/results/r1"
