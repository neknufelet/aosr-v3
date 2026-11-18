"""綁定表（#1206 第 1 支）修補第一輪的考卷：每一道守衛各一題直接打到它、新程式沒被釘住的四處、重播出結果時的檢查。

場景同 ``test_material_gradient_ties``（探針自編八片，y0@a 與 yL@a 成對；θ 六格，成對的格 (0, 4)、(1, 5)）。
代號（G1～G4、S4、S10、B7、B9、B10、G6）是兩位審查植錯清單裡的刀名：每一題的說明寫它守哪一刀。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from aosr.gradient import material_gradient as search
from aosr.gradient import material_gradient_loop as loop
from aosr.gradient import run
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.gradient.material_gradient_ledger import RoundDecision
from aosr.gradient.ties import TieBroken, check_ties_against_start, parse_ties, tied_cells
from tests.engine import _material_gradient_ties_probe as probe
from tests.engine.test_material_gradient_ties import (
    TIES, Record, _context, _one_ulp_off, _rows, _shared, _snapped, runs as runs,
)

MIDDLE = probe.MIDDLE_TIES
# 改綁 y0@a＝y0@b 那一場兩本帳的列數（庚二、換樣板）；本機量的，候選從錯的格算就會變。
MIDDLE_ROWS = (63, 57)


# ---- 一、守衛 ----

def test_pair_with_different_cell_counts_is_refused() -> None:
    """G1：成對兩片的 θ 格數不同（一片有空腔、一片沒有）時 ``tied_cells`` 要拋錯，不能配出不存在的格號。"""
    lean = rounds.designs_for_surfaces(tuple((name, "glass_wool_24k_50mm" if name == "yL@a" else preset)
                                             for name, preset in rounds.surfaces_of(probe.designs())), _context(TIES))
    with pytest.raises(TieBroken, match="成對的 y0@a 與 yL@a 的 θ 格數不同（2 對 1）"):
        tied_cells(lean, TIES)


def test_jump_refuses_a_candidate_of_the_following_surface() -> None:
    """G2：跳點收到「跟著走那一片」自己的候選要點名拒收（``ValueError``，不是等跳完才被不變量擋）。"""
    designs = probe.designs()
    follower = next(item for item in loop.candidates_from_gradient(_snapped(), np.zeros(6), designs, probe.settings())
                    if item.surface_id == "yL@a")
    with pytest.raises(ValueError, match="候選的面 yL@a 不在設計清單裡，或不是它那一對裡出候選的那一片"):
        loop.apply_candidate(_snapped(), follower, designs, ties=TIES)


def test_replay_refuses_a_candidate_named_after_the_following_surface() -> None:
    """G3：換樣板帳本不帶設定重播時，代號用跟著走那一片、只換它自己的那一列要拒收。"""
    start = rounds.surfaces_of(probe.designs())
    half = cast(templates.TemplateRow, SimpleNamespace(candidate="yL@a->glass_wool_96k", surfaces=tuple(
        (name, "glass_wool_96k_50mm" if name == "yL@a" else preset) for name, preset in start)))
    with pytest.raises(ValueError, match="候選 yL@a->glass_wool_96k 的配置不是起點配置只換那一個單位"):
        templates._swapped_surface(start, half, "第 0 輪", None, TIES)


def test_template_header_refuses_a_tie_onto_a_fixed_surface() -> None:
    """G4：換樣板帳本標頭的綁定片要在起點配置可換的面裡；綁到固定面 yL@b 的表拒收，原樣那一份照收。"""
    header = run.template_header_for(probe.header(TIES), probe.settings(), probe.kept_patches(),
                                     base_ledger_sha256="e" * 64)
    document = header.model_dump()
    assert templates.TemplateHeader.model_validate(document) == header
    with pytest.raises(ValueError, match="綁定表的片 .*yL@b.* 不在起點配置可換的面裡"):
        templates.TemplateHeader.model_validate({**document, "ties": (("y0@a", "yL@b"),)})


def test_resume_check_stops_a_decision_row_that_is_one_ulp_apart() -> None:
    """S4：接續時「該不該繼續」那一支自己就要擋成對不相等的起點（帳本最後一列決定），不等下一輪才擋。"""
    zeros = (0.0,) * len(search.COMPONENT_NAMES)
    start = loop._LoopStart(_one_ulp_off(_snapped()), 0.0, 0.0, zeros, 1, 0, 0)
    decision = cast(RoundDecision, SimpleNamespace(next_round_bound=1))
    with pytest.raises(TieBroken, match="接續起點（帳本最後一列決定）：成對的 y0@a.layer2.thickness_m 與 yL@a.layer2"):
        loop._check_should_continue(decision, start, probe.designs(), probe.settings(), TIES)


def test_pre_loop_never_writes_a_snap_row_with_a_split_pair(runs: tuple[Path, Record]) -> None:
    """S10：開跑前段取規格那一步就要查——取規格那一列是在重調之前寫進帳本的，等重調才擋的話壞的那一列已經落地。
    探針把下降交出來的 θ 換成成對空腔差一個最小浮點單位的（取規格不動空腔）：前段要在取規格拋錯，帳本裡只有下降的列。"""
    presnap = cast(Record, runs[1]["presnap"])
    assert presnap["error"] == "TieBroken" and str(presnap["message"]).startswith("取規格之後：成對的 y0@a.layer2")
    phases = cast(list[str], presnap["phases"])
    assert phases and set(phases) == {"descent"}


# ---- 二、新程式沒被釘住的四處 ----

def test_swap_under_a_depth_cap_projects_both_surfaces_of_the_pair() -> None:
    """B7：換樣板有個案上限時，成對兩片各投一次（只投面序在前那一片，另一片的初值超過上限、整個換法不可行）。
    兩片從沒有空腔的 24K 起跑、上限 0.08，一起換到有空腔的 48K：兩片的厚度與空腔逐位相等、真的被投過。"""
    settings, caps = probe.settings(), {"y0@a": 0.08, "yL@a": 0.08}
    presets = tuple("glass_wool_24k_50mm" if name in ("y0@a", "yL@a") else preset
                    for name, preset in zip(probe.NAMES, probe.PRESETS, strict=True))
    designs = run.designs_for(presets, settings, kept=probe.kept_patches(), fixed_surfaces=probe.FIXED, max_depths=caps,
                              table=run.presets_table())
    check_ties_against_start(TIES, designs, {})
    theta = np.asarray(search.snap_to_specs(search.baseline_theta(designs), designs, TIES).theta, dtype=np.float64)
    context = _context(TIES)._replace(max_depths=caps)
    swapped, moved = rounds.swap(designs, theta, "y0@a", "glass_wool_48k", context)
    names = search.design_names(swapped)
    left = [moved[names.index(f"y0@a.layer{layer}.thickness_m")] for layer in (1, 2)]
    right = [moved[names.index(f"yL@a.layer{layer}.thickness_m")] for layer in (1, 2)]
    assert [value.tobytes() for value in left] == [value.tobytes() for value in right]
    assert left[0] + left[1] <= 0.08 and left[1] < 0.05


@pytest.fixture(scope="module")
def wide(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[Path, Record]:
    """有綁、每輪候選上限 10 的那一場（含截帳本接續）。"""
    return _shared(tmp_path_factory, worker_id, "wide")


@pytest.fixture(scope="module")
def middle(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[Path, Record]:
    """改綁 y0@a 與 y0@b 的那一場：跟著走的 y0@b 後面還有 yL@a。"""
    return _shared(tmp_path_factory, worker_id, "middle")


def test_loop_budget_counts_units_when_the_cap_does_not_hide_it(wide: tuple[Path, Record]) -> None:
    """B9、B10：庚二一輪做完判停、接續時重算的下一輪最壞值都要照單位。每輪候選上限 3 時照單位與照片數截完一樣；
    這一場上限 10：記的下一輪最壞值是照單位的數（成對那一對還能往厚跳時比照片數少一段重調），截一輪接續要收、四個檔逐位相同。"""
    settings = probe.settings().model_copy(update={"candidates": probe.settings().candidates.model_copy(
        update={"max_per_round": 10})})
    designs, segment = probe.designs(), loop.retune_evaluation_bound(settings)
    rows = _rows(wide[0] / "wide" / run.LEDGER_FILE)
    decisions = [row for row in rows if row["phase"] == "round_decision"]
    assert len(decisions) >= 2
    seen = []
    for row in decisions:
        bound = cast(Record, row["decision"])["next_round_bound"]
        if bound is None:
            continue
        theta = np.asarray([float.fromhex(value) for value in cast(list[str], row["theta_hex"])])
        units = len(loop.candidates_from_gradient(theta, np.zeros(6), designs, settings, TIES))
        pieces = len(loop.candidates_from_gradient(theta, np.zeros(6), designs, settings))
        assert bound == 1 + (units + 1) * segment
        seen.append(pieces - units)
    assert seen and max(seen) > 0
    assert cast(Record, wide[1]["wide"])["resumed"] == {"rounds": True, "templates": True}


def test_surface_after_the_following_one_gets_candidates_from_its_own_cells(middle: tuple[Path, Record]) -> None:
    """G6：候選迴圈跳過跟著走那一片時 θ 格號要照樣推進。y0@a=y0@b、後面還有 yL@a：yL@a 往薄一格的位移寫在它自己的格、
    從它自己的厚度算；整條產品路徑兩本帳每一列成對相等、沒有 y0@b 自己的候選、yL@a 照常有候選，帳本列數固定。"""
    designs, settings = probe.designs(), probe.settings()
    theta = np.asarray([0.05, 0.05, 0.05, 0.05, 0.1, 0.05])
    found = loop.candidates_from_gradient(theta, np.asarray([-1.0, 0.0, -1.0, 0.0, 1.0, 0.0]), designs, settings, MIDDLE)
    thinner = next(item for item in found if item.candidate_id == "yL@a:thinner")
    assert thinner.delta[4] == 0.05 - 0.1 and thinner.delta[0] == thinner.delta[2] == 0.0
    assert [item.candidate_id for item in found if item.surface_id != "yL@a"] == ["y0@a:thicker"]
    rows = _rows(middle[0] / "middle" / run.LEDGER_FILE)
    assert all(cast(list[str], row["theta_hex"])[0:2] == cast(list[str], row["theta_hex"])[2:4] for row in rows)
    candidates = {str(row["candidate"]) for row in rows if row["candidate"] is not None}
    assert any(name.startswith("yL@a") for name in candidates) and not any(name.startswith("y0@b") for name in candidates)
    quick = {str(row["candidate"]) for row in _rows(middle[0] / "middle" / run.TEMPLATES_FILE) if row["phase"] == "quick_score"}
    assert any(name.startswith("yL@a") for name in quick) and not any(name.startswith("y0@b") for name in quick)
    assert (len(rows), len(_rows(middle[0] / "middle" / run.TEMPLATES_FILE))) == MIDDLE_ROWS
    assert cast(Record, middle[1]["middle"])["resumed"] == {"rounds": True, "templates": True}


# ---- 三、重播出結果時也檢查 ----

def test_result_rebuild_refuses_a_closed_ledger_whose_pair_differs(runs: tuple[Path, Record], tmp_path: Path) -> None:
    """已收尾、有綁的換樣板帳本，把最後交出去那一點裡 yL@a 的厚度改成另一個合法值（交出它的那幾列與決定列一起改、
    前後一致）：從帳本重播出結果（寫結果檔與終點核結果檔共用的 ``result_from_ledger``）要拋 ``TieBroken``，不寫結果檔。"""
    target = tmp_path / "tampered"
    shutil.copytree(runs[0] / "tied", target)
    path = target / run.TEMPLATES_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:]]
    last = rows[-1]
    cell = last["variable_names"].index("yL@a.layer1.thickness_m")
    handed = [index for index, row in enumerate(rows[:-1])
              if row["theta_hex"] == last["theta_hex"] and row["surfaces"] == last["surfaces"]]
    assert last["theta_hex"][cell] != (0.1).hex()
    for index in (*handed, len(rows) - 1):
        rows[index]["theta_hex"][cell] = (0.1).hex()
    path.write_text("\n".join([lines[0], *(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows)]) + "\n",
                    encoding="utf-8")
    (target / run.RESULT_FILE).unlink()
    with pytest.raises(TieBroken, match="最後交出去的那一點（重播出結果）：成對的 y0@a.layer1.thickness_m 與 yL@a.layer1"):
        run.write_result(target, probe.problem("bowl", TIES))
    assert not (target / run.RESULT_FILE).exists()


# ---- 四、拒收字句 ----

def test_unknown_surface_code_names_the_pair() -> None:
    """``--tie x0=zz``：代號不認識時要點名是哪一對（照打的順序），不只說哪一片不認識。"""
    with pytest.raises(ValueError, match="綁定 x0=zz：.*面片代號不認識"):
        parse_ties(["x0=zz"])
