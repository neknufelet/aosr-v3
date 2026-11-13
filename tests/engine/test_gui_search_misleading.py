"""#1128（#1082 段 2-6c 示範找到的）會誤導人的三條：

1. 結果頁把「整個換成材料庫材料」的固定區當成沒動、改善算不進來，卻沒有在跑之前講：沒指原方案時，第三階段頁的預覽與
   這一跑的細節點名那幾區（``zone_result.baseline_warning``），結果頁也點名；存成方案照舊帶設計設定的原方案。
2. 交給搜尋的區寫「方案裡先放原牆」、其實放的是搜尋選的材料（畫面那一條在 ``test_gui_zone_modes_browser``）。
3. 換樣板整塊試過固定候選、沒選，網頁上看不到（``gradient_follow.template_tries``）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from aosr.gradient import folder_files, template_rounds
from aosr.gui import gradient_follow
from aosr.gui.gradient_follow import RIGID_BASELINE_WARNING, template_tries
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.wall_words import wall_words
from aosr.gui.zone_design import DesignSettings, ZoneDesign, read_design, write_design
from aosr.gui.zone_result import (
    baseline_warning, original_walls, replaced_unseen, replaced_unseen_note, replaced_unseen_text)
from aosr.gui.zone_result_routes import UNWRAP_NOTE, unwrap_notes
from aosr.materials.library import FixedEntry, builtin_entries
from aosr.reporting.scheme import Scheme
from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from tests.engine._directivity import DIRECTIVITY
from tests.engine._zone_result_cases import mounted, original_x0
from tests.engine._zoned_report_cases import split_x0, with_x0
from tests.engine.test_gui_gradient_follow import _done_job, _exported, _follow
from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _data, _jobs, _scheme

ENTRY = next(entry for entry in builtin_entries() if isinstance(entry, FixedEntry) and entry.id == "glass_wool_48k_50mm")


def _swapped(a_mode: str = "fixed") -> tuple[Scheme, DesignSettings]:
    """小房 x0 切兩區：a 整個換成材料庫的固定材料（不是貼在原牆上），b 在原牆上貼一層棉；其餘照舊。"""
    wall = original_x0()
    scheme = Scheme.model_validate(with_x0(split_x0(ENTRY.surface_material(), mounted(wall))).model_dump(mode="json")
                                   | {"scheme_id": "swapped"})
    design = DesignSettings(scheme_id="swapped", zones={"x0": {"a": ZoneDesign.model_validate({"mode": a_mode}),
                                                                "b": ZoneDesign(mode="fixed")}})
    return scheme, design


def test_only_fixed_zones_swapped_for_a_library_material_are_named() -> None:
    """抓：漏點名整個換掉的固定區、把貼在原牆上的區（拆得回原牆）或方案原本的牆也點進來、交給搜尋的區點進來、
    指了原方案還點名。"""
    scheme, design = _swapped()
    library = builtin_entries()
    expected = (f"{wall_words(scheme)['x0']} a「{ENTRY.label}」",)
    assert replaced_unseen(scheme, design, library) == expected
    assert replaced_unseen(scheme, None, library) == expected  # 沒有設計設定＝每一區都是固定
    assert replaced_unseen(*_swapped("search"), library) == ()
    assert replaced_unseen(scheme, design.model_copy(update={"original_scheme_id": "room"}), library) == ()
    assert replaced_unseen(scheme, design, ()) == ()  # 材料庫沒有這一筆：看不出是換上去的


def test_warning_says_both_rigid_and_swapped_zones_and_nothing_once_an_original_is_set() -> None:
    scheme, design = _swapped()
    library = builtin_entries()
    zones = replaced_unseen(scheme, design, library)
    assert baseline_warning(scheme, design, rigid=False, library=library) == replaced_unseen_text(zones)
    assert baseline_warning(scheme, design, rigid=True, library=library) == RIGID_BASELINE_WARNING + replaced_unseen_text(zones)
    pointed = design.model_copy(update={"original_scheme_id": "room"})
    assert baseline_warning(scheme, pointed, rigid=True, library=library) is None
    plain = with_x0(split_x0(mounted(original_x0()), original_x0()))
    assert baseline_warning(plain, None, rigid=False, library=library) is None


def test_result_page_names_the_swapped_zone_and_says_nothing_extra_with_an_original() -> None:
    scheme, design = _swapped()
    library = builtin_entries()
    zones = replaced_unseen(scheme, design, library)
    assert unwrap_notes(None, scheme, design, library) == [UNWRAP_NOTE, replaced_unseen_note(zones)]
    original = original_walls(with_x0(split_x0(original_x0(), original_x0())), scheme,
                              capabilities=load_capabilities(config_path("capabilities.toml")), directivity=DIRECTIVITY)
    assert unwrap_notes(original, scheme, design, library) == []


def _swap_design_zone(data_dir: Path) -> None:
    """考卷那份 zoned 方案的 x0 r0_c1（固定）整個換成材料庫的固定材料，設計設定原樣。"""
    path = data_dir / "schemes" / f"{NAME}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    zone = next(item for item in document["scene"]["surface_zones_by_wall"]["x0"] if item["id"] == "r0_c1")
    zone["material"] = ENTRY.surface_material().model_dump(mode="json")
    path.write_text(Scheme.model_validate(document).model_dump_json(), encoding="utf-8")


def test_preview_and_job_detail_warn_before_the_run_and_stop_once_the_adopted_design_points_an_original(
        tmp_path: Path) -> None:
    """抓：預覽（開跑前）不講、這一跑的細節不講，或存成方案後使用者已經替那一份指了原方案還一直講。"""
    data_dir = _data(tmp_path)
    _swap_design_zone(data_dir)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    preview = jobs.preview(NAME, GradientRequest())
    warning = cast(str, preview["baseline_warning"])
    assert warning.startswith("設計設定沒有指原方案") and f"「{ENTRY.label}」" in warning and "r0_c1" in warning
    job_id = _done_job(data_dir, launcher, _exported())
    follow = _follow(data_dir, launcher)
    assert follow.detail(job_id)["baseline_warning"] == warning
    adopted = cast(dict[str, object], follow.adopt(job_id)["adopted"])
    assert follow.detail(job_id)["baseline_warning"] == warning
    name = str(adopted["scheme_id"])
    (data_dir / "schemes" / "room-original.json").write_text(
        _scheme().model_copy(update={"scheme_id": "room-original"}).model_dump_json(), encoding="utf-8")
    design = read_design(data_dir, name)
    assert design is not None
    scheme = Scheme.model_validate_json((data_dir / "schemes" / f"{name}.json").read_text(encoding="utf-8"))
    write_design(data_dir, design.model_copy(update={"original_scheme_id": "room-original"}), scheme)
    assert follow.detail(job_id)["baseline_warning"] is None


def test_adopt_carries_the_original_scheme_the_design_points_at(tmp_path: Path) -> None:
    """抓存成方案時把設計設定指的原方案丟掉（結果頁就退回拆加貼層，固定換掉的區又算成沒動）。"""
    data_dir = _data(tmp_path)
    (data_dir / "schemes" / "room-original.json").write_text(
        _scheme().model_copy(update={"scheme_id": "room-original"}).model_dump_json(), encoding="utf-8")
    design = read_design(data_dir, NAME)
    assert design is not None
    write_design(data_dir, design.model_copy(update={"original_scheme_id": "room-original"}), _scheme())
    launcher = FakeLauncher()
    job_id = _done_job(data_dir, launcher, _exported())
    adopted = cast(dict[str, object], _follow(data_dir, launcher).adopt(job_id)["adopted"])
    saved = read_design(data_dir, str(adopted["scheme_id"]))
    assert saved is not None and saved.original_scheme_id == "room-original"


def _row(phase: str, round_index: int, **fields: object) -> str:
    return json.dumps({"phase": phase, "round": round_index, **fields})


def test_template_tries_lists_every_quick_score_with_the_round_baseline_and_what_was_taken(tmp_path: Path) -> None:
    """抓：漏列固定候選、現狀取錯數字、換上了的那一個沒標、還沒有決定列的半輪也列出來、讀不懂的列讓整段失敗。
    數字形狀照 2-6c 那一跑（b009b0ed）x 終點牆那四列。"""
    lines = [json.dumps({"ledger_version": "v3"}),
             _row("quick_score", 0, candidate="xL->fixed:demo_gw49", value=3.7413),
             _row("quick_score", 0, candidate="xL->fixed:glass_wool_48k_50mm_gap50", value=3.7415),
             _row("quick_score", 0, candidate="xL->none", value=4.1153),
             _row("quick_score", 0, candidate="x0@z1->glass_wool_48k", value=2.9),
             "not json",
             _row("template_decision", 0, value=2.9358, decision={"baseline_value": 2.9358, "accepted": None}),
             _row("quick_score", 1, candidate="xL->none", value=2.0),
             _row("template_decision", 1, value=2.0, decision={"baseline_value": 2.9358, "accepted": "xL->none"}),
             _row("quick_score", 2, candidate="xL->none", value=1.0)]
    (tmp_path / folder_files.TEMPLATES_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    labels = {"demo_gw49": "實測 49 mm", "glass_wool_48k_50mm_gap50": "48K 50＋50", "glass_wool_48k": "48K（可調厚度與空腔）"}
    first, second = template_tries(tmp_path, labels)
    assert first["round"] == 1 and first["baseline"] == 2.9358 and first["accepted"] is None
    assert [(item["surface"], item["label"], item["fixed"], item["value"], item["chosen"])
            for item in cast(list[dict[str, object]], first["tries"])] == [
        ("xL", "實測 49 mm（固定，整塊選或不選）", True, 3.7413, False),
        ("xL", "48K 50＋50（固定，整塊選或不選）", True, 3.7415, False),
        ("xL", "不貼（保留原牆）", False, 4.1153, False),
        ("x0@z1", "48K（可調厚度與空腔）", False, 2.9, False)]
    assert second["accepted"] == "xL->none"
    assert [item["chosen"] for item in cast(list[dict[str, object]], second["tries"])] == [True]
    assert template_tries(tmp_path / "missing", labels) == []


def test_candidate_separator_is_the_template_ledgers() -> None:
    """網頁那一份分隔字（網頁行程不載入 JAX，不能直接拿換樣板那一支的）要等於帳本寫的。"""
    assert gradient_follow.CANDIDATE_SEPARATOR == template_rounds.CANDIDATE_SEPARATOR
