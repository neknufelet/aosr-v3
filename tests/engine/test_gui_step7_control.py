"""動刀前在乾淨 bbc7d17075010a846956003691efe9eac34bfa5c 錄無家具控制組。

2026-10-08 用 uv run --no-sync python，_control_result('wall-1'／'wall-2')；
只將求解計時四格固定為 0，再 build_result_view 與 test_gui_compare_view._view。
model_dump(mode='json') 用 ensure_ascii=False、sort_keys=True、separators=(',', ':')
序列化後 SHA-256。完整原始 JSON 留在本次 /tmp 暫存目錄。
新顯示欄位與共用標籤表新增的名稱從對照投影排除；第 5 步指定的三句能力說明依下列對照還原，
其餘原欄位逐位守住；動刀前的整份雜湊與既有考卷答案當時沒有改。
第五步改用原投影文字骨架與逐格③級（相對 5e-13／絕對 5e-12）答案，不宣稱跨處理器逐位。
錄製 HEAD 5339182ca3fd1831a684a19ab5c03462eb5b7e1f：工作區 785-tolerance-step5/evidence（repo 外）的錄製腳本；
整份舊雜湊同跑核過後拿掉。新答案住 _gui_step7_answers.py。投影與原本排序序列化方式不變。
機器相關：categories 的 cost 經評分 numpy.log2 等陣列操作；listening_area.pairs.value 經同族評分；
frequency_responses.points.level_db、frequency_plot_data 非軸列、overlay.series.levels_db
承接 Paris 節點影響的交接總能量，另經 display.level_db 的 math.log10；整題照紙「為什麼」③那一列歸評分層，③級。
無關：frequency_hz／center_frequency_hz／frequency_plot_data 軸列是輸入軸；timings 固定零；
listening_area.pairs.limit 是登記簿輸入，全部留骨架。所有顯示的四捨五入字串逐字。
剛好落在進位邊界時顯示字串可能翻位，這個風險仍在。rank_margin 同跑另守兩候選總代價差。
"""
from __future__ import annotations

import json
from functools import cache

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui.result_view import build_result_view
from aosr.reporting.result import SchemeResult, Timings
from aosr.reporting.calculation_fingerprint import short_fingerprint
from aosr.reporting.physics_identity import physics_identity_parts
from aosr.reporting.compare import compare_results
from tests.engine._cross_cpu_tolerance import SCORING_ABS, SCORING_REL
from tests.engine._gui_step7_answers import ANSWERS
from tests.engine._json_control import assert_json_control
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_gui_compare_view import _view
from tests.engine.test_scheme_pipeline import shared_control_result


NEW_FIELDS = {"furniture", "furniture_reason", "furniture_notes", "furniture_flutter_text",
              "frequency_note", "ranking_approximation_text", "surface_text", "coverage_text",
              "validation_text", "a_state_text", "b_state_text", "overlay_note", "no_changes_text",
              "speaker_setup_line", "ideal_source_note"}
NEW_LABELS = {"direction_zones", "ranking", "own_line_ripple_rms_db", "seat_robustness_rms_db"}

CAPABILITY_TEXT_CHANGES = {
    "牆面局部分區（這一版每面只收整面一區）": "隨頻率改變的牆面材料（這一版每面一個與頻率無關的阻抗）",
    '桌面、沙發、天雲都不進有限元素網格；家具吸音只以整件吸音量計入整房殘響與交接頻率、不分位置；家具反射相位假設為零': '家具、桌面、沙發等大型物件（房間是空的六面盒）',
    '家具只算一次反射；家具與牆的混合反射未納入': '家具與桌面的反射',
}


@cache
def _physics_text_changes() -> dict[str, str]:
    """第八步換物理身分；只還原整格短指紋，原本兩份雜湊照舊。

    題內才量：整輪的 JAX 設定夾具（conftest.py::calculation_environment）晚於收集，匯入時量會跟結果記的身分不同（#309）。
    """
    identity = physics_identity_parts(capabilities=CAPABILITIES, directivity=DIRECTIVITY).identity
    return {short_fingerprint(identity): "00eca7eaa663"}


def _original_fields(value: object, root: bool = True, labels: bool = False) -> object:
    if isinstance(value, dict):
        # #558 第二段 B 只投影掉低頻新說明；其餘七類每一格與原數值骨架照舊。
        # 新字句、原值與登記簿數字另由 test_low_frequency_response_display 獨立求答案。
        if value.get("category") == "low_frequency_response":
            value = {key: "" if key in {"note", "note_text"} else item for key, item in value.items()}
        return {key: _original_fields(item, False, key == "labels") for key, item in value.items()
                if (key not in NEW_FIELDS or key == "furniture" and not root)
                and (not labels or key not in NEW_LABELS)}
    if isinstance(value, list):
        return [_original_fields(item, False) for item in value]
    return {**CAPABILITY_TEXT_CHANGES, **_physics_text_changes()}.get(value, value) if isinstance(value, str) else value


def rank_margin(results: tuple[SchemeResult, SchemeResult]) -> tuple[float, float]:
    """各候選照正式排名算同跑總代價；分別算也讓同候選重複的突變在差值斷言紅。"""
    targets = load_quality_targets(config_path("quality_targets.toml"))
    totals = tuple(compare_results((result,), quality_targets=targets, run_date=result.run_date).rankable[0].total_cost
                   for result in results)
    return abs(totals[0] - totals[1]), max(SCORING_ABS, SCORING_REL * max(map(abs, totals)))


def test_no_furniture_original_view_json_hashes(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    def timed(name: str) -> SchemeResult:
        return shared_control_result(tmp_path_factory, worker_id, name).model_copy(
            update={"timings": Timings(solve_s=0, output_s=0, evaluate_s=0, total_s=0)})
    results = (timed("wall-1"), timed("wall-2"))
    gap, tolerance = rank_margin(results)
    assert gap > 1000 * tolerance, "兩候選總代價差沒有遠大於③級容差"
    views = {"result": build_result_view(results[0], quality_targets_path=config_path("quality_targets.toml")),
             "compare": _view(results)}
    for name, view in views.items():
        original = _original_fields(view.model_dump(mode="json"))
        text = json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert_json_control(text, "gui", ANSWERS[name])
