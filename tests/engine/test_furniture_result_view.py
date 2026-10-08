"""真家具結果頁；字句答案取自家具決策紙及第七步施工單。"""
from __future__ import annotations

from aosr.config.paths import config_path
from aosr.geometry.furniture import FaceDirection, FurnitureKind
from aosr.gui.furniture_view import furniture_surface
from aosr.gui.result_view import build_result_view
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import EvaluationState, ReasonCode
from aosr.scoring.reflections_contract import ReflectionSource, ReflectionsAndEchoPayload
from tests.engine._furniture_scheme_results import scheme_pair as scheme_pair


def test_furniture_coverage_validation_and_exact_surface_row(scheme_pair: tuple[SchemeResult, ...]) -> None:
    result = scheme_pair[1]
    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    payload = next(item.payload for item in result.candidate.evaluations
                   if item.category.value == "reflections_and_echo")
    assert isinstance(payload, ReflectionsAndEchoPayload)
    found = set()
    for channel in view.reflections:
        assert channel.coverage_text == "原本牆面覆蓋條件成立；家具僅一次反射、混合反射未納入"
        source = next(item for item in payload.channels
                      if (item.speaker_id, item.receiver_id) == (channel.speaker_id, channel.receiver_id))
        pair = next(item for item in result.pairs
                    if (item.speaker_id, item.receiver_id) == (channel.speaker_id, channel.receiver_id))
        assert pair.report.path_table is not None
        furniture_paths = [item for item in source.reflections
                           if item.source is ReflectionSource.PATH_TABLE and item.wall_sequence == ("furniture",)]
        expected = "牆面：已驗證"
        if furniture_paths:
            expected += "；家具反射只驗證公式實作一致，實際家具精度未驗證"
        assert channel.validation_text == expected
        ordered = sorted(source.reflections, key=lambda item: item.relative_direct_delay_s)
        for raw, shown in zip(ordered, channel.paths, strict=True):
            if raw in furniture_paths:
                table_row = pair.report.path_table.rows[raw.source_index]
                assert (table_row.furniture_id, table_row.furniture_face) == ("seat", FaceDirection.TOP)
                assert shown.surface_text == "沙發（seat）頂面"
                found.add((table_row.furniture_id, table_row.furniture_face))
            else:
                assert not shown.surface_text
    assert found == {("seat", FaceDirection.TOP)}
    assert {channel.validation_text for channel in view.reflections} == {
        "牆面：已驗證", "牆面：已驗證；家具反射只驗證公式實作一致，實際家具精度未驗證"}


def test_surface_names_the_piece_the_row_points_at_not_the_first_piece(scheme_pair: tuple[SchemeResult, ...]) -> None:
    # 只有一件家具時「查第一件」跟「查對的那件」分不出來；前面多塞一件代號排前面的書桌。
    result = scheme_pair[1]
    assert result.scheme.furniture is not None
    seat = result.scheme.furniture[0]
    desk = seat.model_copy(update={"furniture_id": "a-desk", "kind": FurnitureKind.DESK, "material": "wood"})
    scheme = result.scheme.model_copy(update={"furniture": (desk, seat)})
    payload = next(item.payload for item in result.candidate.evaluations
                   if item.category.value == "reflections_and_echo")
    assert isinstance(payload, ReflectionsAndEchoPayload)
    shown = []
    for channel in payload.channels:
        pair = next(item for item in result.pairs
                    if (item.speaker_id, item.receiver_id) == (channel.speaker_id, channel.receiver_id))
        shown += [furniture_surface(path, pair, scheme) for path in channel.reflections
                  if path.source is ReflectionSource.PATH_TABLE and path.wall_sequence == ("furniture",)]
    assert shown and set(shown) == {"沙發（seat）頂面"}


def test_furniture_section_ranking_reverb_flutter_and_response(scheme_pair: tuple[SchemeResult, ...]) -> None:
    view = build_result_view(scheme_pair[1], quality_targets_path=config_path("quality_targets.toml"))
    assert view.furniture == (("沙發（seat）", "布面；估計，非本件實測",
                              "未知（計算時用相鄰頻帶延伸代算）：63、8000 Hz"),)
    assert view.furniture_reason == ("已含家具一次反射、遮擋與有限尺寸鏡面修正；未含家具與牆之間的多次反射、"
                                     "完整繞射，以及家具吸音對整房殘響的影響")
    assert set(view.furniture_notes) == {"透射未算", "喇叭指向性往下的方向尚未獨立驗證，桌面反射強度靠這個假設",
                                         "遮擋邊界上的反射會突然出現或消失"}
    prefix, subject = view.ranking_approximation_text.split("；")
    assert prefix == "家具模型：近似"
    suffix = "以近似模型參與第二階段第一版的擺位排名"
    assert subject.endswith(suffix)
    assert set(subject.removesuffix(suffix).split("、")) == {"音色平衡", "聆聽區穩定性", "反射與回聲", "聲道匹配"}
    assert "家具模型" not in view.ranking_text
    assert "未包含家具吸音" in view.reverberation.caption_text
    assert next(row.note for row in view.categories if row.category == "reverberation") == "未包含家具吸音"
    assert view.furniture_flutter_text == "顫動警戒第一版只看三對牆，家具形成的平行面未評估"
    assert view.frequency_note == "家具模型：近似"


def test_plain_result_has_no_scene_furniture_notices(scheme_pair: tuple[SchemeResult, ...]) -> None:
    view = build_result_view(scheme_pair[0], quality_targets_path=config_path("quality_targets.toml"))
    assert not view.furniture
    assert not view.furniture_reason
    assert not view.furniture_notes
    assert not view.ranking_approximation_text
    assert not view.furniture_flutter_text
    assert not view.frequency_note
    assert "未包含家具吸音" not in view.reverberation.caption_text


def test_scene_notices_survive_missing_reflection_payload(scheme_pair: tuple[SchemeResult, ...]) -> None:
    result = scheme_pair[1]
    evaluations = tuple(item.model_copy(update={"payload": None, "flags": (),
                        "state": EvaluationState.UNAVAILABLE, "raw_quantities": (),
                        "category_cost": None, "reason_codes": (ReasonCode.SOLVER_UNAVAILABLE,)})
                        if item.category.value == "reflections_and_echo" else item
                        for item in result.candidate.evaluations)
    changed = result.model_copy(update={"candidate": result.candidate.model_copy(update={"evaluations": evaluations})})
    view = build_result_view(changed, quality_targets_path=config_path("quality_targets.toml"))
    assert view.furniture_flutter_text == "顫動警戒第一版只看三對牆，家具形成的平行面未評估"
    assert view.frequency_note == "家具模型：近似"
    assert all("家具反射只驗證" not in channel.validation_text for channel in view.reflections)
    assert "反射與回聲" not in view.ranking_approximation_text
    assert "音色平衡" in view.ranking_approximation_text


def test_furniture_materials_are_taken_from_saved_path_table(scheme_pair: tuple[SchemeResult, ...]) -> None:
    result = scheme_pair[1]
    pairs = []
    for pair in result.pairs:
        table = pair.report.path_table
        assert table is not None and table.furniture_materials
        materials = tuple(row.model_copy(update={"material": "leather", "unknown_bands_hz": (125.0,)})
                          for row in table.furniture_materials)
        table = table.model_copy(update={"furniture_materials": materials})
        pairs.append(pair.model_copy(update={"report": pair.report.model_copy(update={"path_table": table})}))
    view = build_result_view(result.model_copy(update={"pairs": tuple(pairs)}),
                             quality_targets_path=config_path("quality_targets.toml"))
    assert view.furniture == (("沙發（seat）", "皮面；估計，非本件實測",
                              "未知（計算時用相鄰頻帶延伸代算）：125 Hz"),)
