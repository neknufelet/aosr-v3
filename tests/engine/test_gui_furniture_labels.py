"""第八支第一步：決策紙原文與存檔來源；不用現行材質登記簿猜答案。"""
from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.config.paths import config_path
from aosr.config.representative_speakers import load_representative_speakers
from aosr.gui import labels
from aosr.gui.result_view import build_result_view
from aosr.gui.search_best import _frequency
from aosr.gui.search_view import build_search_view
from aosr.reporting import display
from aosr.reporting.result import SchemeResult, save_result
from aosr.reporting.scheme import Scheme, SpeakerSetup
from aosr.scoring.contract import QualityCategory
from aosr.search.store import SearchStore
from aosr.search.labels import speaker_setup_text
from aosr.search.report import build_report, render_text
from aosr.search.run import SearchStatus
from tests.engine import _speaker_setup_cases as speakers
from tests.engine import _furniture_cases as furniture
from tests.engine._furniture_scheme_results import scheme_pair as scheme_pair
from tests.engine._search_blocked_cases import blocked_store
from tests.engine._search_run_cases import RUN_DATE, FakeCompute, make_store, run
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser
from tests.engine.test_gui_compare_view import _renamed, _view
from tests.engine.test_furniture_result_view import FINITE_LIMITATIONS
from tests.engine.test_scheme_pipeline import _run_control


@pytest.mark.parametrize("furnished", [False, True])
@pytest.mark.parametrize("kind,representative", [("bookshelf", True), ("floorstanding", True), ("floorstanding", False)])
def test_result_header_speaker_sentence_is_independent_of_furniture(
    scheme_pair: tuple[SchemeResult, ...], furnished: bool, kind: str, representative: bool,
) -> None:
    setup = SpeakerSetup.model_validate(speakers.setup(kind, "floor" if kind == "floorstanding" else "stand",
                                                     representative=representative))
    result = scheme_pair[1 if furnished else 0]
    result = result.model_copy(update={"scheme": result.scheme.model_copy(update={"speaker_setup": setup})})
    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    assert view.speaker_setup_line == speaker_setup_text(setup)
    assert ("代表模型，非實際型號" in view.speaker_setup_line) == representative
    assert ("資料只有 KEF 與 Arendal 兩家公開高音高度" in view.speaker_setup_line) == (
        representative and kind == "floorstanding")


def test_compare_lists_identical_materials_and_speaker_sentences(scheme_pair: tuple[SchemeResult, ...]) -> None:
    result = scheme_pair[1]
    setup = SpeakerSetup.model_validate(speakers.setup())
    result = result.model_copy(update={"scheme": result.scheme.model_copy(update={"speaker_setup": setup})})
    view = _view((result, _renamed(result, "same-furniture")))
    for side in ("A", "B"):
        assert f"{side}：沙發（seat）；布面；估計，非本件實測；未知（計算時用相鄰頻帶延伸代算）：63、8000 Hz；{speaker_setup_text(setup)}" in view.notes


def test_compare_speaker_only_still_lists_speaker_sentence(scheme_pair: tuple[SchemeResult, ...]) -> None:
    plain = scheme_pair[0]
    setup = SpeakerSetup.model_validate(speakers.setup())
    result = plain.model_copy(update={"scheme": plain.scheme.model_copy(update={"speaker_setup": setup})})
    view = _view((plain, _renamed(result, "speaker-only")))
    assert "A：無家具或喇叭設定" in view.notes
    assert f"B：{speaker_setup_text(setup)}" in view.notes


def test_compare_missing_material_table_does_not_claim_no_furniture(scheme_pair: tuple[SchemeResult, ...]) -> None:
    plain, furnished = scheme_pair
    pairs = tuple(pair.model_copy(update={"report": pair.report.model_copy(update={"path_table": None})})
                  for pair in furnished.pairs)
    changed = furnished.model_copy(update={"pairs": pairs})
    view = _view((plain, changed))
    assert "A：無家具或喇叭設定" in view.notes
    assert "B：無家具或喇叭設定" not in view.notes


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_compare_reverberation_names_side_and_preserves_verdict(
    scheme_pair: tuple[SchemeResult, ...], reverse: bool,
) -> None:
    view = _view((scheme_pair[1], scheme_pair[0]) if reverse else (scheme_pair[0], scheme_pair[1]))
    row = next(row for row in view.categories if row.category == "reverberation")
    assert row.better_text == "相同"
    side = "A" if reverse else "B"
    assert row.note_text == f"{side}：未包含家具吸音；不能用來判斷增加家具後的殘響改善量"


def _old_reverberation(result: SchemeResult) -> SchemeResult:
    evaluations = tuple(item.model_copy(update={"evaluator_version": item.evaluator_version + "-old"})
                        if item.category is QualityCategory.REVERBERATION else item
                        for item in result.candidate.evaluations)
    return result.model_copy(update={"candidate": result.candidate.model_copy(update={"evaluations": evaluations})})


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_compare_reverberation_keeps_conditions_difference(
    scheme_pair: tuple[SchemeResult, ...], reverse: bool,
) -> None:
    plain, furnished = scheme_pair
    pair = (plain, _old_reverberation(furnished))
    view = _view((pair[1], pair[0]) if reverse else pair)
    row = next(row for row in view.categories if row.category == "reverberation")
    side = "A" if reverse else "B"
    assert row.better_text == ""
    assert row.comparison_text == "評分條件不同，這一類代價不能直接比"
    assert row.note_text == (f"{side}：未包含家具吸音；不能用來判斷增加家具後的殘響改善量；"
                             "評分條件不同，這一類代價不能直接比")


def test_two_furnished_compare_reverberation_has_no_improvement_warning(scheme_pair: tuple[SchemeResult, ...]) -> None:
    furnished = scheme_pair[1]
    view = _view((furnished, _renamed(furnished, "both-furnished")))
    row = next(row for row in view.categories if row.category == "reverberation")
    assert row.better_text == "相同"
    assert row.note_text == "未包含家具吸音"


@pytest.mark.parametrize("furnished", [False, True])
def test_search_attachments_and_empty_refinement_marks(tmp_path: Path, furnished: bool,
                                                     scheme_pair: tuple[SchemeResult, ...]) -> None:
    store, registry = blocked_store(tmp_path, blocked=False, budget=3) if furnished else make_store(tmp_path, budget=3)
    run(store, registry, FakeCompute(store))
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
                             server_program=store.identity.program_fingerprint)
    blocks = {block.key: block for block in view.blocks}
    for key in ("crossover", "stability"):
        assert display.FURNITURE_MODEL_NOTE not in blocks[key].lines
    assert display.FURNITURE_MODEL_NOTE not in blocks["refine-best"].lines
    frequency = _frequency(scheme_pair[1 if furnished else 0].model_dump(mode="json"))
    assert frequency["furniture_note"] == (display.FURNITURE_MODEL_NOTE if furnished else "")


def test_no_ranked_candidate_has_no_approximate_rank_notice(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path, blocked=False, budget=3)
    status = run(store, registry, FakeCompute(store, missing=frozenset(range(store.settings.budget))))
    assert status.best_trial is None
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert not any("家具模型：近似" in line for line in report.ranks)
    assert "家具材質：第一名結果讀不回" not in report.furniture_notes
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
                             server_program=store.identity.program_fingerprint)
    assert display.FURNITURE_MODEL_NOTE not in next(block.lines for block in view.blocks if block.key == "search-best")


@pytest.mark.parametrize("refined", [False, True])
def test_report_materials_use_first_place_saved_path_table(tmp_path: Path, scheme_pair: tuple[SchemeResult, ...],
                                                        refined: bool) -> None:
    store, registry = blocked_store(tmp_path, blocked=False, budget=3)
    status = run(store, registry, FakeCompute(store))
    assert status.best_trial is not None
    result = scheme_pair[1]
    pairs = []
    for pair in result.pairs:
        table = pair.report.path_table
        assert table is not None and table.furniture_materials
        materials = tuple(row.model_copy(update={"material": "leather", "unknown_bands_hz": (125.0,)})
                          for row in table.furniture_materials)
        pairs.append(pair.model_copy(update={"report": pair.report.model_copy(update={
            "path_table": table.model_copy(update={"furniture_materials": materials})})}))
    changed = result.model_copy(update={"pairs": tuple(pairs)})
    if refined:
        from aosr.search.refine import RefineLedger, RefineRow
        from aosr.search.refine_run import header_for
        from aosr.search.store import refine_result_name

        save_result(result, store.candidate_path(status.best_trial))
        book = RefineLedger.create(store.refine_ledger_path, header_for(store))
        store.ensure_refine_dir()
        save_result(changed, store.refine_result_path(None))
        book.append(RefineRow(round=1, trial_number=None, result_file=refine_result_name(None),
                              outcome="scored", total_cost=0.0, seconds=0.0))
    else:
        save_result(changed, store.candidate_path(status.best_trial))
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    text = render_text(report)
    assert "家具模型說明\n" in text
    assert "沙發（seat）；皮面；估計，非本件實測；參考的是合成皮；未知（計算時用相鄰頻帶延伸代算）：125 Hz" in report.furniture_notes
    assert FINITE_LIMITATIONS in text
    assert "家具僅一次反射、混合反射未納入" in text
    assert "喇叭指向性往下的方向尚未獨立驗證，桌面反射強度靠這個假設" in text
    assert "整房殘響與交接頻率未包含家具吸音" in text
    assert "布面；估計" not in "\n".join(report.furniture_notes)


@pytest.mark.parametrize("refined", [False, True])
def test_report_unreadable_first_materials_are_explicit_without_search_fallback(
    tmp_path: Path, scheme_pair: tuple[SchemeResult, ...], refined: bool,
) -> None:
    from aosr.search.refine import RefineLedger, RefineRow
    from aosr.search.refine_run import header_for
    from aosr.search.store import refine_result_name

    store, registry = blocked_store(tmp_path, blocked=False, budget=3)
    status = run(store, registry, FakeCompute(store))
    assert status.best_trial is not None
    if refined:
        save_result(scheme_pair[1], store.candidate_path(status.best_trial))
        store.ensure_refine_dir()
        store.refine_result_path(None).write_text("{broken")
        book = RefineLedger.create(store.refine_ledger_path, header_for(store))
        book.append(RefineRow(round=1, trial_number=None, result_file=refine_result_name(None),
                              outcome="scored", total_cost=0.0, seconds=0.0))
    else:
        store.candidate_path(status.best_trial).write_text("{broken")
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert report.furniture_notes == (
        "家具模型說明", display.FURNITURE_REASON, "家具僅一次反射、混合反射未納入",
        "家具材質：第一名結果讀不回", FINITE_LIMITATIONS,
        "喇叭指向性往下的方向尚未獨立驗證，桌面反射強度靠這個假設",
        "透射未算", "整房殘響與交接頻率未包含家具吸音",
    )
    assert "家具材質：第一名結果讀不回" in render_text(report).splitlines()
    assert not any("估計，非本件實測" in line for line in report.furniture_notes)


def test_approximation_and_category_names_have_one_source() -> None:
    from aosr.search.report_calibration import CATEGORY_LABELS

    assert labels.LABELS["furniture_model_approximate"] == display.APPROXIMATE_TEXT == "近似"
    assert labels.LABELS["approximate"] == display.APPROXIMATE_TEXT
    assert all(value == labels.LABELS[key] for key, value in CATEGORY_LABELS.items())


def test_compare_browser_notes_and_reverberation_screenshot(
    browser: Browser, tmp_path: Path, scheme_pair: tuple[SchemeResult, ...],
) -> None:
    (tmp_path / "results").mkdir()
    for letter, result in zip(("a", "b"), scheme_pair, strict=True):
        save_result(result, tmp_path / "results" / f"{letter * 32}.json")
    with _serve(tmp_path) as base, _open(browser, f"{base}/compare/{'a' * 32}/{'b' * 32}") as watched:
        page = watched.page
        page.wait_for_selector("#content", state="visible")
        assert "B：沙發（seat）；布面；估計，非本件實測；未知（計算時用相鄰頻帶延伸代算）：63、8000 Hz" in page.locator("#notes").inner_text()
        row = page.locator("#categories tr").filter(has_text="殘響")
        assert "B：未包含家具吸音；不能用來判斷增加家具後的殘響改善量" in row.inner_text()
        page.locator("#notes").screenshot(path=str(tmp_path / "compare-notes.png"))
        row.screenshot(path=str(tmp_path / "compare-reverberation.png"))
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


@pytest.mark.parametrize("furnished", [False, True])
def test_result_browser_speaker_header_and_leather_material(
    browser: Browser, tmp_path: Path, furnished: bool,
) -> None:
    document = speakers.document("floor")
    representative = load_representative_speakers(config_path("representative_speakers.toml")).floorstanding
    document["speaker_setup"] = speakers.setup("floorstanding", "floor") | {"cabinet": representative.cabinet_fields_m()}
    if furnished:
        document["furniture"] = [furniture.relative_item(material="leather")]
    result = _run_control(Scheme.model_validate(document))
    assert result.scheme.speaker_setup is not None
    (tmp_path / "results").mkdir()
    save_result(result, tmp_path / "results" / f"{'c' * 32}.json")
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{'c' * 32}") as watched:
        page = watched.page
        page.wait_for_selector("#content", state="visible")
        header = page.locator("#speaker-setup")
        assert header.is_visible()
        assert header.inner_text() == speaker_setup_text(result.scheme.speaker_setup)
        assert "代表模型，非實際型號" in header.inner_text()
        assert "資料只有 KEF 與 Arendal 兩家公開高音高度" in header.inner_text()
        page.locator("header").screenshot(path=str(tmp_path / "result-header.png"))
        if furnished:
            assert "皮面；估計，非本件實測；參考的是合成皮" in page.locator("#furniture-list").inner_text()
            page.locator("#furniture").screenshot(path=str(tmp_path / "result-material.png"))
        else:
            assert not page.locator("#furniture").is_visible()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def _completed_attachments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, furnished: bool = True) -> SearchStore:
    from aosr.search.ledger import Ledger
    from aosr.search.refine import RefineLedger
    from tests.engine import _crossover_cases
    from tests.engine import _stability_attach_cases
    from tests.engine import _stability_report_cases as cases
    from tests.engine._crossover_cases import prepared

    original_make_store = make_store
    original_prepared = prepared

    def furnished_store(folder: Path) -> tuple[SearchStore, Path]:
        cloud = furniture.cloud_item(width_m=1.0, depth_m=1.0, height_m=0.1,
            placement={"bottom_center_m": [2.0, 2.0, 2.5], "yaw_deg": 0})
        return original_make_store(folder, furniture=[cloud] if furnished else None)

    monkeypatch.setattr(_crossover_cases, "make_store", furnished_store)

    def prepared_with_best(folder: Path) -> tuple[SearchStore, Path, SearchStatus]:
        store, registry, status = original_prepared(folder)
        search = min((row for row in Ledger.read(store.ledger_path)[1] if row.score is not None),
                     key=lambda row: row.score if row.score is not None else float("inf"))
        refined = min((row for row in RefineLedger.read(store.refine_ledger_path)[1] if row.total_cost is not None),
                      key=lambda row: row.total_cost if row.total_cost is not None else float("inf"))
        status = status.model_copy(update={"best_trial": search.trial_number, "best_score": search.score,
            "refine": status.refine.model_copy(update={"best": "baseline" if refined.trial_number is None else refined.trial_number,
                                                      "best_total_cost": refined.total_cost})})
        store.status_path.write_text(status.model_dump_json())
        return store, registry, status

    monkeypatch.setattr(_stability_attach_cases, "prepared", prepared_with_best)
    store, _, _, summary, _ = cases.report_case(tmp_path, monkeypatch)
    assert summary.completed
    return store


def test_search_browser_three_approximation_lines_with_completed_attachments(
    browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _completed_attachments(tmp_path, monkeypatch)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}") as watched:
        page = watched.page
        page.wait_for_function("() => document.querySelector('#best-frequency canvas') !== null")
        for key in ("best-frequency", "crossover", "stability"):
            block = page.locator(f"#{key}")
            assert display.FURNITURE_MODEL_NOTE in block.inner_text()
            assert "家具模型：近似" in block.inner_text()
            block.screenshot(path=str(tmp_path / f"search-{key}.png"))
        assert "已算" in page.locator("#stability").inner_text()
        assert "未開始" not in page.locator("#crossover").inner_text()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


@pytest.mark.parametrize("state", ["not_started", "running", "skipped", "unreadable"])
def test_search_attachments_without_results_have_no_approximation(
    tmp_path: Path, state: str,
) -> None:
    from aosr.search.crossover_record import CrossoverSummary, summary_path as crossover_path, write_summary as write_crossover
    from aosr.search.placement_stability_record import IdentityStamp, StabilitySummary
    from aosr.search.placement_stability_record import summary_path as stability_path, write_summary as write_stability

    store, _ = blocked_store(tmp_path, blocked=False, budget=3)
    store.status_path.write_text(SearchStatus().model_dump_json())
    if state in {"running", "skipped"}:
        write_crossover(store.path, CrossoverSummary.model_validate({"state": state, "completed": state == "skipped"}))
        write_stability(store.path, StabilitySummary.model_validate({"search_id": store.search_id, "state": state,
                        "completed": state == "skipped", "identity": IdentityStamp.of(store.identity)}))
    elif state == "unreadable":
        for target in (crossover_path(store.path), stability_path(store.path)):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{broken")
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
                             server_program=store.identity.program_fingerprint)
    for block in view.blocks:
        if block.key in {"crossover", "stability"}:
            assert display.FURNITURE_MODEL_NOTE not in block.lines
            assert block.warning == (state == "unreadable")


@pytest.mark.parametrize("furnished", [False, True])
@pytest.mark.parametrize("state", ["done", "running", "skipped"])
def test_search_attachments_only_mark_actual_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, furnished: bool, state: str,
) -> None:
    from aosr.search.crossover_record import read_summary as read_crossover, write_summary as write_crossover
    from aosr.search.placement_stability_record import read_summary as read_stability, write_summary as write_stability

    store = _completed_attachments(tmp_path, monkeypatch, furnished=furnished)
    crossover = read_crossover(store.path)
    stability = read_stability(store.path)
    assert crossover is not None and any(variant.ranking for variant in crossover.variants)
    assert stability is not None and any(point.total_cost is not None for point in stability.points)
    completed = state in {"done", "skipped"}
    write_crossover(store.path, crossover.model_copy(update={"state": state, "completed": completed}))
    write_stability(store.path, stability.model_copy(update={"state": state, "completed": completed,
        "arithmetic": stability.arithmetic if state == "done" else None,
        "boundaries": stability.boundaries if state == "done" else ()}))
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
                             server_program=store.identity.program_fingerprint)
    for block in view.blocks:
        if block.key in {"crossover", "stability"}:
            assert (display.FURNITURE_MODEL_NOTE in block.lines) == (furnished and state != "skipped")
