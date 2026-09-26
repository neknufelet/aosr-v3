"""#505 第三刀控制組：全向的兩個候選，六類評估與排名跟改動前的主線相同。

評分層讀聲源模型之後，全向那條路的分數、旗標、狀態、名次與分表都不准動。答案是主對話在改動前的主線上實跑
``_scoring_source_model_control.py`` 錄下的（``_scoring_source_model_answers.py``）；這一題只拿現在的結果去比，
不拿被測函式現算的值當答案。按設計會變的格子（評估器版本、折了版本的兩個設定指紋、反射 payload 換型那一格、
新的聲源模型指紋）在管線那一支逐一列出並排除。整條管線放在同一題：平行跑法下模組層的準備會在每個工人重做。
"""
from __future__ import annotations

import pytest

from tests.engine import _scoring_source_model_answers as answers
from tests.engine import _scoring_source_model_control as control

# 評分層在陣列上用 numpy 的對數，雲端機器若走另一套指令集的實作，最後幾位可能不同（本機測不到）；
# 照 test_timbre_listening_area_support.py 音色控制組的比法用相對誤差，不逐位。
_REL = 1e-12


def _hex_leaves(node: object, path: str = "") -> list[tuple[str, str]]:
    if isinstance(node, dict):
        return [leaf for key in sorted(node) for leaf in _hex_leaves(node[key], f"{path}.{key}")]
    assert isinstance(node, str), path
    return [(path, node)]


def _assert_headline(found: dict[str, object]) -> None:
    got, expected = _hex_leaves(found), _hex_leaves(answers.HEADLINE)
    assert [path for path, _ in got] == [path for path, _ in expected]
    for (path, value), (_, answer) in zip(got, expected, strict=True):
        assert float.fromhex(value) == pytest.approx(float.fromhex(answer), rel=_REL), path


def _assert_section(name: str, found: dict[str, object], expected: dict[str, object]) -> None:
    assert found["exact"] == expected["exact"], f"{name}：非浮點那一半（結構、字串、旗標、狀態、名次）變了"
    groups, answer_groups = found["floats"], expected["floats"]
    assert isinstance(groups, dict) and isinstance(answer_groups, dict)
    assert groups.keys() == answer_groups.keys(), name
    for pattern, (count, weighted_abs, weighted_signed) in groups.items():
        answer_count, answer_abs, answer_signed = answer_groups[pattern]
        where = f"{name} {pattern}"
        assert count == answer_count, where
        scale = float.fromhex(answer_abs)
        assert float.fromhex(weighted_abs) == pytest.approx(scale, rel=_REL), where
        assert abs(float.fromhex(weighted_signed) - float.fromhex(answer_signed)) <= _REL * scale, where


def test_omnidirectional_scoring_and_ranking_match_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """全向兩候選：每類的狀態、旗標、原因碼，名次、總代價、類代價、分項、原始量，與每一段的其餘內容都跟主線相同。"""
    for module, name, stand_in in control.STAND_INS:
        monkeypatch.setattr(module, name, stand_in)
    candidates, ranking = control.run()

    assert control.summary(candidates, ranking) == answers.SUMMARY
    _assert_headline(control.headline(ranking))
    found = control.sections(candidates, ranking)
    assert found.keys() == answers.SECTIONS.keys()
    for name, section in found.items():
        expected = answers.SECTIONS[name]
        assert isinstance(expected, dict)
        _assert_section(name, section, expected)
