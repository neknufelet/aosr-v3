"""動刀前在乾淨 bbc7d17075010a846956003691efe9eac34bfa5c 錄無家具控制組。

2026-10-08 用 uv run --no-sync python，_control_result('wall-1'／'wall-2')；
只將求解計時四格固定為 0，再 build_result_view 與 test_gui_compare_view._view。
model_dump(mode='json') 用 ensure_ascii=False、sort_keys=True、separators=(',', ':')
序列化後 SHA-256。完整原始 JSON 留在本次 /tmp 暫存目錄；以下是動刀前整份雜湊。
新顯示欄位從對照投影排除；第 5 步指定的三句能力說明依下列對照還原，
其餘原欄位逐位守住；動刀前的整份雜湊與既有考卷答案沒有改。
"""
from __future__ import annotations

import hashlib
import json

import pytest

from aosr.config.paths import config_path
from aosr.gui.result_view import build_result_view
from aosr.reporting.result import SchemeResult, Timings
from tests.engine.test_gui_compare_view import _view
from tests.engine.test_scheme_pipeline import shared_control_result


HASHES = {"result": "2c05c99a2bb39cc6fe7ac574ac89322b49c408371a724335456c7dbd7daaf509",
          "compare": "bba2a9e285fa03d74adf02f6c59ea0aa8949328d037abb6e94cf76e0cf148ab5"}
NEW_FIELDS = {"furniture", "furniture_reason", "furniture_notes", "furniture_flutter_text",
              "frequency_note", "ranking_approximation_text", "surface_text", "coverage_text",
              "validation_text", "a_state_text", "b_state_text", "overlay_note", "no_changes_text"}

CAPABILITY_TEXT_CHANGES = {'搜尋比較時桌面、沙發、天雲都不進有限元素網格；家具只算一次反射；家具與牆的混合反射未納入；未包含家具吸音': '家具、桌面、沙發等大型物件（房間是空的六面盒）', '家具只算一次反射；家具與牆的混合反射未納入': '家具與桌面的反射', '桌面、控台、螢幕造成的早期反射要現場另外確認；喇叭指向性往下的方向尚未獨立驗證，桌面反射強度靠這個假設': '桌面、控台、螢幕造成的早期反射要現場另外確認'}

def _original_fields(value: object, root: bool = True) -> object:
    if isinstance(value, dict):
        return {key: _original_fields(item, False) for key, item in value.items()
                if key not in NEW_FIELDS or key == "furniture" and not root}
    if isinstance(value, list):
        return [_original_fields(item, False) for item in value]
    return CAPABILITY_TEXT_CHANGES.get(value, value) if isinstance(value, str) else value


def test_no_furniture_original_view_json_hashes(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    def timed(name: str) -> SchemeResult:
        return shared_control_result(tmp_path_factory, worker_id, name).model_copy(
            update={"timings": Timings(solve_s=0, output_s=0, evaluate_s=0, total_s=0)})
    results = (timed("wall-1"), timed("wall-2"))
    views = {"result": build_result_view(results[0], quality_targets_path=config_path("quality_targets.toml")),
             "compare": _view(results)}
    for name, view in views.items():
        original = _original_fields(view.model_dump(mode="json"))
        text = json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(text.encode()).hexdigest() == HASHES[name]
