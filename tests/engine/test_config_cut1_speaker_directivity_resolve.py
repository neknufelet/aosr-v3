"""``aosr.config.speaker_directivity`` 解析函式的裁判。

v2 有 16 支考卷用到這一支，但**每一支**都整個帶不走（它們 import 了 physics／geometry／
materials／records／scoring 或週邊腳本那一包）。v2 考卷對這一支做的事也只有兩類：
把 ``_W = SPEAKER_PRESETS[...]`` 當輸入（``test_scene2_p1_oriented_schema``）、或只驗
開／關與 provenance（``test_scene17_default_directivity``）；``test_scene163_polygon_directivity``
甚至只斷言「不是 None」。沒有任何一支把五個欄位寫死。

所以裁判改由 donor 標準答案接手，並**補上 v2 考卷沒有的角度**：不支援的型別在 OFF 時
是「警告 ＋ 回 None」、在 ON 時是當場炸；覆寫值 0／負／非有限／布林要被擋下來。

判定與殘餘風險見 ``tests/engine/test_config_cut1_table``。
"""
from __future__ import annotations

from typing import TypedDict, cast

import pytest

import aosr.config.speaker_directivity as speaker_directivity

from tests.engine._config_answers import (
    is_approx,
    module_answers,
    probe_raised,
    probe_value,
    probe_warned,
)


class ResolveArgs(TypedDict, total=False):
    """``resolve_speaker_directivity`` 的參數（除了 ``enabled``，其餘都有預設）。"""

    enabled: bool
    speaker_type: str
    baffle_width_m: float | None
    piston_radius_m: float | None


def _table(node: object, where: str) -> dict[str, object]:
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是表：{node!r}")
    return {str(name): item for name, item in node.items()}


def _field(record: object, key: str) -> object:
    if not isinstance(record, dict) or key not in record:
        raise AssertionError(f"答案檔這一筆沒有 {key}：{record!r}")
    return record[key]


def _probes() -> list[object]:
    probes = module_answers("speaker_directivity")["probes"]
    if not isinstance(probes, list):
        raise AssertionError("答案檔的 speaker_directivity 探針不是一串東西")
    return probes


def _resolve_records() -> list[object]:
    picked: list[object] = []
    for record in _probes():
        args = _field(record, "args")
        if isinstance(args, dict) and "enabled" in args:
            picked.append(record)
    return picked


def _preset_record() -> dict[str, object] | None:
    for record in _probes():
        table = _table(record, "答案檔這一筆的探針")
        args = table.get("args")
        if isinstance(args, dict) and "presets" in args:
            return table
    return None


def test_resolve_matches_donor_on_every_probed_input() -> None:
    """每一組設定（開／關 × 型別 × 覆寫值）的解析結果都跟 donor 一樣。

    donor 說回一個物件就逐欄比；donor 說回 ``None`` 就 ``is None``；donor 說炸就炸，
    而且訊息逐字一樣。
    """
    records = _resolve_records()
    assert records, "答案檔裡沒有 resolve_speaker_directivity 的探針——這一支就沒有對象"
    for record in records:
        args = _table(_field(record, "args"), "答案檔這一筆的 args")
        expected = _table(_field(record, "expected"), "答案檔這一筆的 expected")
        raised = probe_raised(expected)
        if raised is not None:
            with pytest.raises(ValueError) as caught:
                speaker_directivity.resolve_speaker_directivity(**cast(ResolveArgs, args))
            assert str(caught.value) == raised["message"], f"訊息跟 donor 不一樣：{args}"
            continue
        built = speaker_directivity.resolve_speaker_directivity(**cast(ResolveArgs, args))
        want = probe_value(expected)
        if want is None:
            assert built is None, f"donor 回 None，新家卻回了東西：{args}"
        else:
            assert built is not None, f"donor 有回物件，新家回了 None：{args}"
            actual = {
                "speaker_type": built.speaker_type,
                "baffle_width_m": built.baffle_width_m,
                "piston_radius_m": built.piston_radius_m,
                "version": built.version,
            }
            assert is_approx(actual, want), f"解析出來的欄位跟 donor 不一樣：{args}"


def test_warning_path_matches_donor() -> None:
    """不支援的型別、指向性 OFF：donor 是「發一個警告再回 None」，這裡也要一樣。"""
    warned = [
        record
        for record in _resolve_records()
        if probe_warned(_table(_field(record, "expected"), "答案檔這一筆的 expected"))
    ]
    assert warned, "答案檔裡沒有警告那幾筆——這一條就沒有對象"
    for record in warned:
        args = _table(_field(record, "args"), "答案檔這一筆的 args")
        expected = _table(_field(record, "expected"), "答案檔這一筆的 expected")
        with pytest.warns(UserWarning) as caught:
            built = speaker_directivity.resolve_speaker_directivity(**cast(ResolveArgs, args))
        assert built is None
        assert str(caught[0].message) == probe_warned(expected)[0]["message"]


def test_speaker_presets_match_donor() -> None:
    """兩個喇叭預設（書架型／落地型）的五個欄位逐項比對。"""
    record = _preset_record()
    assert record is not None, "答案檔裡沒有 SPEAKER_PRESETS——這一格沒有裁判"
    expected = _table(_field(record, "expected"), "答案檔這一筆的 expected")
    fields = expected["preset_fields"]
    if not isinstance(fields, dict):
        raise AssertionError(f"答案檔的 preset_fields 不是表：{fields!r}")
    for name, want in fields.items():
        preset = speaker_directivity.SPEAKER_PRESETS[name]
        actual = {
            "width_m": preset.width_m,
            "height_m": preset.height_m,
            "depth_m": preset.depth_m,
            "acoustic_center_z_m": preset.acoustic_center_z_m,
            "piston_radius_m": preset.piston_radius_m,
        }
        assert is_approx(actual, want), f"預設 {name} 的欄位跟 donor 不一樣"


def test_enabled_default_is_the_keyword_default_of_the_resolver() -> None:
    """產品預設（指向性沒說就是開）真的走進解析函式的預設值（接線）。"""
    assert speaker_directivity.DIRECTIVITY_DEFAULT_ENABLED is True
    assert speaker_directivity.resolve_speaker_directivity(enabled=True) is not None
    assert speaker_directivity.resolve_speaker_directivity(enabled=False) is None
