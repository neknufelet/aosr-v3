"""``aosr.config.speaker_directivity`` 解析函式的裁判。

v2 有 16 支考卷用到這一支，但**每一支**都整個帶不走（它們 import 了 physics／geometry／
materials／records／scoring 或週邊腳本那一包）。v2 考卷對這一支做的事也只有兩類：
把 ``_W = SPEAKER_PRESETS[...]`` 當輸入（``test_scene2_p1_oriented_schema``）、或只驗
開／關與 provenance（``test_scene17_default_directivity``）；``test_scene163_polygon_directivity``
甚至只斷言「不是 None」。沒有任何一支把五個欄位寫死。

所以裁判改由 donor 標準答案接手，並**補上 v2 考卷沒有的角度**：不支援的型別在 OFF 時
是「警告 ＋ 回 None」、在 ON 時是當場炸；覆寫值 0／負／非有限／布林要被擋下來；
**該警告的那幾筆要警告、不該警告的一筆都不准多**（找碴 NOTE 2）。

判定與殘餘風險見 ``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。
"""
from __future__ import annotations

from typing import TypedDict, cast

import pytest

import aosr.config.speaker_directivity as speaker_directivity

from tests.engine._config_answers import (
    case_args,
    decode,
    expected_block,
    is_approx,
    observed_warnings,
    probe_by_id,
    probe_ids,
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


def _resolve_ids() -> list[str]:
    return sorted(
        case_id
        for case_id in probe_ids("speaker_directivity")
        if case_id.startswith("speaker_directivity.resolve.")
    )


def _call(case: dict[str, object]) -> object:
    """呼叫解析函式；warning 的捕捉由呼叫端自己決定（有的 case 要驗「不准有警告」）。"""
    return speaker_directivity.resolve_speaker_directivity(**cast(ResolveArgs, case_args(case)))


def test_resolve_matches_donor_on_every_declared_case() -> None:
    """每一組設定（開／關 × 型別 × 覆寫值）的解析結果、例外與警告都跟 donor 一樣。"""
    ids = _resolve_ids()
    assert ids, "答案檔裡沒有 resolve_speaker_directivity 的 case——這一支就沒有對象"
    for case_id in ids:
        case = probe_by_id(case_id)
        expected = expected_block(case)
        raised = probe_raised(expected)
        if raised is not None:
            with observed_warnings() as warnings_seen:
                try:
                    _call(case)
                except ValueError as exc:
                    assert str(exc) == raised["message"], f"訊息跟 donor 不一樣：{case_id}"
                else:
                    raise AssertionError(f"donor 說這一筆會炸，新家沒炸：{case_id}")
            assert not warnings_seen, f"炸掉的那一筆不該先發警告：{case_id}"
            continue
        want_warned = probe_warned(expected)
        with observed_warnings() as seen:
            built = _call(case)
        got = [str(item.message) for item in seen]
        assert got == [str(w["message"]) for w in want_warned], (
            f"警告跟 donor 不一樣（該警告的沒警告、或不該警告的多了）：{case_id}"
        )
        want = probe_value(expected)
        if want is None:
            assert built is None, f"donor 回 None，新家卻回了東西：{case_id}"
            continue
        resolved = _directivity(built, case_id)
        actual = {
            "speaker_type": resolved.speaker_type,
            "baffle_width_m": resolved.baffle_width_m,
            "piston_radius_m": resolved.piston_radius_m,
            "version": resolved.version,
        }
        assert is_approx(actual, want), f"解析出來的欄位跟 donor 不一樣：{case_id}"


def _directivity(built: object, case_id: str) -> speaker_directivity.SpeakerDirectivity:
    """把回傳值收窄成 ``SpeakerDirectivity``（donor 說這一筆有回物件）。"""
    if not isinstance(built, speaker_directivity.SpeakerDirectivity):
        raise AssertionError(f"donor 說這一筆回一個 SpeakerDirectivity，新家回了 {built!r}：{case_id}")
    return built


def test_unsupported_type_warns_through_pytest_and_returns_none() -> None:
    """不支援的型別、指向性 OFF：發一個 ``UserWarning`` 再回 ``None``（pytest 的寫法）。"""
    ids = [
        case_id
        for case_id in _resolve_ids()
        if probe_warned(expected_block(probe_by_id(case_id)))
    ]
    assert ids, "答案檔裡沒有警告那幾筆——這一條就沒有對象"
    for case_id in ids:
        case = probe_by_id(case_id)
        with pytest.warns(UserWarning):
            built = _call(case)
        assert built is None


def test_speaker_presets_match_donor() -> None:
    """兩個喇叭預設（書架型／落地型）的五個欄位逐項比對。

    這一格跟 ``constants.SPEAKER_PRESETS`` 是同一組數字：常數那一邊驗的是「值凍結了」，
    這裡驗的是「資料表拿得出來的那五個欄位跟 donor 一樣」（走 encode 的 hex 寫法，
    不是十進位）。
    """
    case = probe_by_id("speaker_directivity.SPEAKER_PRESETS.all")
    expected = expected_block(case)
    if "preset_fields" not in expected:
        raise AssertionError(f"答案檔那一筆沒有 preset_fields：{expected!r}")
    fields = decode(expected["preset_fields"])
    assert isinstance(fields, dict), f"答案檔的 preset_fields 不是表：{fields!r}"
    listed = case_args(case)
    assert listed["presets"] == sorted(speaker_directivity.SPEAKER_PRESETS), (
        "答案檔列的那幾個預設跟新家的鍵對不上"
    )
    for name in speaker_directivity.SPEAKER_PRESETS:
        assert name in fields, f"答案檔的 preset_fields 少了 {name}"
    for name, want in fields.items():
        preset = speaker_directivity.SPEAKER_PRESETS[str(name)]
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
