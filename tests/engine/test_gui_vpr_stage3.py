"""第三階段頁與「逐區表與改善多少」的提醒（#1208 第 3 支）函式層考卷：方案裡有「鋼板直接貼在棉上」的構造時，開跑前的
預覽、每一跑的細節、改善多少各多一格；沒有時回覆跟原來一樣、不讀界線。

資料全是考卷自己的：第三階段用現有工作考卷那份 2×2 分區小房間（不開真搜尋，服務是替身）；改善多少用分區結果頁考卷那一份
真算的小房間（``_zone_result_cases.real_design``，同一次 pytest 只算一次），各題只把存的方案換成帶這類構造的版本再排字。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from aosr.config.quality_targets import load_quality_targets
from aosr.gui import reference_only
from aosr.gui.app import STATIC
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.zone_result import Improvement, improvement
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.scoring import gradient_objective_settings
from tests.engine._bonded_backing_cases import BONDED_VPR, MELAMINE_50
from tests.engine.test_gui_gradient_follow import _done_job, _exported, _follow
from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _data, _jobs, _scheme
from tests.engine.test_gui_vpr_results import (
    LOOSE, PLAIN_WALL, PURPOSE, TARGETS, WHY, _mounted, _registry, _saved, _zone, plain as plain)

Doc = dict[str, object]
STAGE_THREE = "有 VPR（鋼板共振吸音）這類「鋼板直接貼在棉上」的構造：第三階段只算 30～300 Hz，全在 300 Hz 以下，這一跑的數字都只能參考。"
RESULT = ("有 VPR（鋼板共振吸音）這類「鋼板直接貼在棉上」的構造：300 Hz 以下的數字只能參考；"
          "下表標了「含 300 Hz 以下」的幾列都受影響。")
# 改善表預期被標的五類（寫死成列上印的名字，不拿規則跟自己比）；低頻拖尾不計分、空間感尚未評估、反射與回聲下界剛好等於界線。
MARKED_ROWS = ["音色平衡", "聆聽區穩定性", "低頻頻響", "殘響", "聲道匹配"]
# 主線（82d76a9f）兩個回覆的每一個鍵。
PREVIEW_KEYS = {"plan", "arguments", "labels", "wall_names", "fixed_depths", "baseline_warning", "surrogate", "defaults", "busy"}
DETAIL_KEYS = {"job_id", "state", "max_steps", "descent_steps", "notes", "failures", "baseline_warning", "template_tries",
               "template_names_note", "adopted"}


def _block(walls: str) -> Doc:
    return {"sentence": walls + STAGE_THREE, "why_label": "為什麼", "why": WHY}


def _put(data_dir: Path, wall: str, zone_id: str, material: Doc, name: str = NAME) -> None:
    """存檔裡那份 2×2 分區方案的一區換材料（設計設定原樣：前牆 r0_c0、r1_c1 交給搜尋，天花 r0_c1 保留原牆，其餘固定）。"""
    path = data_dir / "schemes" / f"{name}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    zone = next(item for item in document["scene"]["surface_zones_by_wall"][wall] if item["id"] == zone_id)
    zone["material"] = material
    path.write_text(Scheme.model_validate(document).model_dump_json(), encoding="utf-8")


def _no_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("沒命中時不該讀品質登記簿")
    monkeypatch.setattr(reference_only, "load_quality_targets", refuse)
    monkeypatch.setattr(reference_only, "registry_band_hz", refuse)


# ---- 開跑前的預覽 ----

@pytest.mark.parametrize(("wall", "zone", "material", "mount", "walls"), [
    pytest.param("x0", "r0_c1", BONDED_VPR, "on_wall", "前牆", id="fixed_zone"),
    pytest.param("ceiling", "r0_c1", BONDED_VPR, "on_wall", "天花", id="kept_original_wall"),
    pytest.param("xL", "r1_c0", _mounted([MELAMINE_50], BONDED_VPR), "on_wall", "後牆", id="original_wall_under_plain_layers"),
    pytest.param("y0", "r0_c0", _mounted(cast("list[Doc]", BONDED_VPR["layers"]), PLAIN_WALL), "on_wall", "左牆", id="pasted_layers"),
    # 交給搜尋的區的原牆：貼在原牆上時搜尋貼的層背後接它；貼在硬底上時「不貼」那個候選保留的也是它——兩種裝法都算。
    pytest.param("x0", "r0_c0", BONDED_VPR, "on_wall", "前牆", id="searched_zone_on_wall"),
    pytest.param("x0", "r0_c0", BONDED_VPR, "rigid", "前牆", id="searched_zone_on_rigid"),
])
def test_the_preview_says_which_faces_keep_such_a_construction_before_the_run(
        tmp_path: Path, wall: str, zone: str, material: Doc, mount: str, walls: str) -> None:
    data_dir = _data(tmp_path)
    _put(data_dir, wall, zone, material)
    preview = _jobs(data_dir, FakeLauncher()).preview(NAME, GradientRequest.model_validate({"mount": mount}))
    assert preview["reference_only"] == _block(walls) and set(preview) == PREVIEW_KEYS | {"reference_only"}


def test_several_faces_are_named_once_each_in_face_order(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    for wall, zone in (("ceiling", "r0_c0"), ("yL", "r0_c0"), ("yL", "r1_c1"), ("x0", "r0_c1")):
        _put(data_dir, wall, zone, BONDED_VPR)
    preview = _jobs(data_dir, FakeLauncher()).preview(NAME, GradientRequest())
    assert preview["reference_only"] == _block("前牆、右牆、天花")


def test_without_such_a_construction_the_preview_is_the_ordinary_one_and_nothing_is_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """沒黏住的同一塊鋼板與棉不算：回覆沒有那一格（鍵跟主線一樣），而且不讀品質登記簿。"""
    data_dir = _data(tmp_path)
    _put(data_dir, "x0", "r0_c1", LOOSE)
    _no_registry(monkeypatch)
    assert set(_jobs(data_dir, FakeLauncher()).preview(NAME, GradientRequest())) == PREVIEW_KEYS


# ---- 每一跑的「注意」 ----

def test_a_run_gets_the_line_from_the_scheme_it_started_with_not_from_the_saved_one(tmp_path: Path) -> None:
    """開跑時方案有這類構造：細節多那一格、紀錄行那一串原樣。開跑後存檔那份改掉了，照開跑時的快照判。"""
    data_dir = _data(tmp_path)
    _put(data_dir, "x0", "r0_c1", BONDED_VPR)
    launcher = FakeLauncher()
    job_id = str(_jobs(data_dir, launcher).start(NAME, GradientRequest())["job_id"])
    detail = _follow(data_dir, launcher).detail(job_id)
    assert detail["reference_only"] == _block("前牆") and set(detail) == DETAIL_KEYS | {"reference_only"}
    assert detail["notes"] == []
    _put(data_dir, "x0", "r0_c1", PLAIN_WALL)
    assert _follow(data_dir, launcher).detail(job_id)["reference_only"] == _block("前牆")


def test_a_run_started_without_one_stays_silent_when_the_saved_scheme_gets_one_later(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    job_id = str(_jobs(data_dir, launcher).start(NAME, GradientRequest())["job_id"])
    _put(data_dir, "x0", "r0_c1", BONDED_VPR)
    _no_registry(monkeypatch)
    assert set(_follow(data_dir, launcher).detail(job_id)) == DETAIL_KEYS


@pytest.mark.parametrize("hit", ["started", "adopted"])
def test_after_adopting_the_line_follows_the_adopted_scheme(tmp_path: Path, hit: str) -> None:
    """存成方案之後照存成的那一份：開跑時有、存成的沒有就不講；開跑時沒有、存成的有就講。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    exported = _exported()
    if hit == "started":
        _put(data_dir, "x0", "r0_c1", BONDED_VPR)
    else:
        walls = cast("dict[str, list[Doc]]", cast(Doc, exported["scene"])["surface_zones_by_wall"])
        next(item for item in walls["yL"] if item["id"] == "r1_c1")["material"] = BONDED_VPR
    job_id = _done_job(data_dir, launcher, exported)
    follow = _follow(data_dir, launcher)
    assert follow.detail(job_id).get("reference_only") == (_block("前牆") if hit == "started" else None)
    follow.adopt(job_id)
    assert follow.detail(job_id).get("reference_only") == (None if hit == "started" else _block("右牆"))


def test_a_run_whose_scheme_snapshot_cannot_be_read_has_no_line(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    _put(data_dir, "x0", "r0_c1", BONDED_VPR)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    job_id = str(jobs.start(NAME, GradientRequest())["job_id"])
    (jobs.root / job_id / "scheme.json").write_text("{", encoding="utf-8")
    assert set(_follow(data_dir, launcher).detail(job_id)) == DETAIL_KEYS


# ---- 出句子的函式 ----

def test_the_sentence_changes_when_the_stage_three_range_reaches_above_the_band(tmp_path: Path) -> None:
    """現行 30～300、界線 300：上界剛好等於界線，整段都在界線以下。上界改到界線以上就只講界線以下那一部分。"""
    registry = load_quality_targets(TARGETS)
    assert reference_only.stage_three_block(["前牆", "天花"], registry, PURPOSE) == _block("前牆、天花")
    wider = _registry(tmp_path, reference_only.GRADIENT_RANGE_KEY, "[40.0, 300.5]", section="[[stage_one]]")
    assert reference_only.stage_three_block(["後牆"], wider, PURPOSE) == {
        "sentence": "後牆有 VPR（鋼板共振吸音）這類「鋼板直接貼在棉上」的構造：第三階段算 40～300.5 Hz，其中 300 Hz 以下的部分只能參考。",
        "why_label": "為什麼", "why": WHY}
    lower = _registry(tmp_path, reference_only.GRADIENT_RANGE_KEY, "[40.0, 250.0]", section="[[stage_one]]")
    block = reference_only.stage_three_block(["後牆"], lower, PURPOSE)
    assert block is not None and block["sentence"] == (
        "後牆有 VPR（鋼板共振吸音）這類「鋼板直接貼在棉上」的構造：第三階段只算 40～250 Hz，全在 300 Hz 以下，這一跑的數字都只能參考。")


def test_the_range_is_the_one_the_gradient_objective_reads_and_a_bad_one_is_an_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert reference_only.GRADIENT_RANGE_KEY == gradient_objective_settings.SCORED_RANGE_KEY
    registry = load_quality_targets(TARGETS)
    settings = gradient_objective_settings.gradient_objective_settings(registry, PURPOSE)
    assert settings.stage_one.scored_range_hz == (30.0, 300.0)
    assert reference_only.stage_three_block([], registry, PURPOSE) is None
    three = _registry(tmp_path, reference_only.GRADIENT_RANGE_KEY, "[30.0, 100.0, 300.0]", section="[[stage_one]]")
    with pytest.raises(TypeError, match="不是一段頻率範圍"):  # 三個數不是一段
        reference_only.stage_three_block(["前牆"], three, PURPOSE)
    # 同一段裡不是量法設定的（權重表）、值不是一組數的（溫度）：都報錯，不當成沒有。
    for key in ("stage_three_gradient.component_weights", "stage_three_gradient.big_peak_proxy_temperature_db"):
        monkeypatch.setattr(reference_only, "GRADIENT_RANGE_KEY", key)
        with pytest.raises(TypeError, match="不是一段頻率範圍"):
            reference_only.stage_three_block(["前牆"], registry, PURPOSE)


def test_walls_of_several_schemes_are_joined_in_face_order_and_named_once() -> None:
    plain_scheme = _scheme()

    def with_hit(*places: tuple[str, str]) -> Scheme:
        document = plain_scheme.model_dump(mode="json")
        for wall, zone_id in places:
            next(item for item in document["scene"]["surface_zones_by_wall"][wall] if item["id"] == zone_id)["material"] = BONDED_VPR
        return Scheme.model_validate(document)

    assert reference_only.scheme_walls(plain_scheme) == [] and reference_only.scheme_walls(plain_scheme, plain_scheme) == []
    assert reference_only.scheme_walls(with_hit(("ceiling", "r0_c0")), plain_scheme, with_hit(("x0", "r1_c1"), ("ceiling", "r1_c0"))) == [
        "前牆", "天花"]
    assert reference_only.scheme_walls(_scheme("whole")) == []


# ---- 改善多少 ----

def _renamed(result: SchemeResult, name: str) -> SchemeResult:
    """同一份結果換一個方案代號（改善要兩份不同代號的結果；數字原樣）。"""
    return SchemeResult.model_validate_json(result.model_dump_json().replace(f'"{result.scheme.scheme_id}"', f'"{name}"'))


@pytest.fixture(scope="module")
def baseline(plain: SchemeResult) -> SchemeResult:
    return _renamed(plain, "zone-base")


def _improvement(design: SchemeResult, baseline: SchemeResult) -> Improvement:
    return improvement(design, baseline, load_quality_targets(TARGETS))


def _reference(view: Improvement) -> Doc | None:
    return cast("Doc | None", view.model_dump(mode="json").get("reference_only"))


@pytest.mark.parametrize(("design_walls", "baseline_walls", "walls"), [
    pytest.param({"xL": _zone(BONDED_VPR)}, {}, "後牆", id="design"),
    pytest.param({}, {"ceiling": _zone(_mounted([MELAMINE_50], BONDED_VPR))}, "天花", id="baseline_only"),
    pytest.param({"floor": _zone(BONDED_VPR), "xL": _zone(BONDED_VPR)}, {"xL": _zone(BONDED_VPR), "y0": _zone(BONDED_VPR)},
                 "後牆、左牆、地板", id="both"),
])
def test_either_side_with_such_a_construction_marks_five_rows_and_the_total(
        plain: SchemeResult, baseline: SchemeResult, design_walls: dict[str, list[Doc]], baseline_walls: dict[str, list[Doc]],
        walls: str) -> None:
    """設計命中、只有原牆基準命中、兩邊都命中（牆名取聯集、一面一次）：句子、為什麼、標籤同結果頁那一塊；被標的列寫死。"""
    view = _improvement(_saved(plain, design_walls), _saved(baseline, baseline_walls))
    assert view.total is not None
    assert _reference(view) == {"sentence": walls + RESULT, "why_label": "為什麼", "why": WHY, "tag": "含 300 Hz 以下",
                                "rows": [*MARKED_ROWS, "總分"]}
    ordinary = _improvement(plain, baseline).model_dump(mode="json")
    assert {key: value for key, value in view.model_dump(mode="json").items() if key != "reference_only"} == ordinary


def test_without_one_on_either_side_the_improvement_is_the_ordinary_one_and_nothing_is_read(
        plain: SchemeResult, baseline: SchemeResult, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_registry(monkeypatch)
    view = _improvement(_saved(plain, {"xL": _zone(LOOSE)}), baseline)
    assert view.total is not None and view.reference_only is None
    assert set(view.model_dump(mode="json")) == {"rows", "total", "total_note", "note"}


def _without(result: SchemeResult, category: str) -> SchemeResult:
    kept = tuple(item for item in result.candidate.evaluations if item.category.value != category)
    return result.model_copy(update={"candidate": result.candidate.model_copy(update={"evaluations": kept})})


def test_a_row_with_a_number_on_one_side_is_marked_and_one_with_none_is_not(plain: SchemeResult, baseline: SchemeResult) -> None:
    """殘響只剩一邊有數字：那一列照標（那個數字就是受影響的數字）。兩邊都沒有：那一列不標，其餘照標。
    少一類就不可排名、沒有總分列，所以總分也不標。"""
    hit = _saved(plain, {"xL": _zone(BONDED_VPR)})
    one_side = _improvement(hit, _without(baseline, "reverberation"))
    assert next(row for row in one_side.rows if row.category == "殘響").baseline == "—" and one_side.total is None
    assert cast(Doc, _reference(one_side))["rows"] == MARKED_ROWS
    other_side = _improvement(_without(hit, "reverberation"), baseline)
    assert cast(Doc, _reference(other_side))["rows"] == MARKED_ROWS
    neither = _improvement(_without(hit, "timbre_balance"), _without(baseline, "timbre_balance"))
    assert neither.total is None and cast(Doc, _reference(neither))["rows"] == MARKED_ROWS[1:]


def _unavailable(result: SchemeResult, category: str) -> SchemeResult:
    """那一類的評估改成「不可估」（資料範圍不夠、沒有代價）：改善表照樣有那一列，兩格都是「—」。"""
    document = result.model_dump(mode="json")
    for item in cast("list[Doc]", cast(Doc, document["candidate"])["evaluations"]):
        if item["category"] == category:
            item.update(state="unavailable", payload=None, category_cost=None, raw_quantities=[], reason_codes=["insufficient_coverage"])
    return result.model_copy(update={"candidate": SchemeResult.model_validate(document).candidate})


def test_a_row_that_is_there_but_has_no_cost_on_either_side_is_not_marked(plain: SchemeResult, baseline: SchemeResult) -> None:
    """整張改善表：音色平衡兩邊都不可估——那一列在、兩格都是「—」、範圍也低於界線，但沒有算出代價，不標；其他四類照標。"""
    hit = _saved(_unavailable(plain, "timbre_balance"), {"xL": _zone(BONDED_VPR)})
    view = _improvement(hit, _unavailable(baseline, "timbre_balance"))
    row = next(row for row in view.rows if row.category == "音色平衡")
    assert (row.baseline, row.design) == ("—", "—")
    assert cast(Doc, _reference(view))["rows"] == MARKED_ROWS[1:]


def test_the_total_is_marked_only_when_a_marked_category_counts_into_it() -> None:
    registry = load_quality_targets(TARGETS)
    labels = {"timbre_balance": "音色平衡", "reflections_and_echo": "反射與回聲", "reverberation": "殘響"}
    costed = {"timbre_balance": True, "reflections_and_echo": True, "reverberation": False}

    def rows(in_total: dict[str, str]) -> object:
        block = reference_only.improvement_block(["前牆"], costed, in_total, registry, PURPOSE, labels)
        assert block is not None
        return block["rows"]

    assert rows({"timbre_balance": "總分", "reflections_and_echo": "總分"}) == ["音色平衡", "總分"]
    # 算進總分的只有沒被標的類別（反射下界等於界線、殘響沒算出代價）：總分不標。沒有總分列也不標。
    assert rows({"reflections_and_echo": "總分", "reverberation": "總分"}) == ["音色平衡"]
    assert rows({}) == ["音色平衡"]
    assert reference_only.improvement_block([], costed, {"timbre_balance": "總分"}, registry, PURPOSE, labels) is None


def test_the_pages_draw_from_the_reply_and_write_none_of_these_words_themselves() -> None:
    for name in ("gradient.js", "gradient.html", "zone_result.js"):
        text = (STATIC / name).read_text(encoding="utf-8")
        assert "鋼板直接貼在棉上" not in text.replace("方案裡有「鋼板直接貼在棉上」的構造（#1208 第 3 支）", "").replace(
            "設計或原牆基準有「鋼板直接貼在棉上」的構造（#1208 第 3 支）", ""), name
        for word in ("只能參考", "VPR", "Fraunhofer", "含 300", "為什麼"):
            assert word not in text, (name, word)
