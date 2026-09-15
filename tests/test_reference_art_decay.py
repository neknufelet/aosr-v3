"""第九段晚期衰減答案的治理籃考卷；本檔不載入 ``aosr``。"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_CASES = ("flat", "varied", "lowabs")
_CARD = _ROOT / "governance" / "rules" / "answer-files-carry-provenance.toml"


def _path(case: str) -> Path:
    return _ROOT / "blueprint" / f"reference_art_decay_{case}.json"


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{where} 不是一層表")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError(f"{where} 不是一串值")
    return value


def _load(case: str) -> dict[str, object]:
    with _path(case).open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), case)


def _hex(value: object, where: str) -> str:
    cell = _mapping(value, where)
    hexadecimal = cell.get("hex")
    if not isinstance(hexadecimal, str):
        raise AssertionError(f"{where}.hex 不是字串")
    return hexadecimal


@pytest.mark.parametrize("case", _CASES)
def test_decay_answer_provenance_matches_registry(case: str) -> None:
    """抓 donor 的 tag、commit 或 clean 任一格漂離登記簿。"""
    answer = _load(case)
    card = tomllib.loads(_CARD.read_text(encoding="utf-8"))
    settings = _mapping(card.get("settings"), "settings")

    assert answer.get("donor") == {
        "tag": settings.get("donor_tag"),
        "commit": settings.get("donor_commit"),
        "clean": True,
    }
    assert isinstance(answer.get("env"), dict)


def test_decay_answers_cover_their_declared_bands_without_gaps() -> None:
    """抓任一材料少頻帶、多頻帶、重複頻帶或三份頻率軸不一致。"""
    observed_axes = []
    for case in _CASES:
        answer = _load(case)
        parameters = _mapping(answer.get("parameters"), "parameters")
        declared = tuple(
            _hex(cell, "frequencies_hz[]")
            for cell in _sequence(parameters.get("frequencies_hz"), "frequencies_hz")
        )
        bands = [
            _mapping(item, "bands[]")
            for item in _sequence(answer.get("bands"), "bands")
        ]
        observed = tuple(_hex(band.get("frequency_hz"), "frequency_hz") for band in bands)
        assert observed
        assert len(set(observed)) == len(observed)
        assert observed == declared
        observed_axes.append(observed)

    assert all(axis == observed_axes[0] for axis in observed_axes)


@pytest.mark.parametrize("case", _CASES)
def test_decay_answers_never_used_perron_fallback(case: str) -> None:
    """上一代三組參考題任一頻帶走 Perron 備援都必須判紅。"""
    bands = _sequence(_load(case).get("bands"), "bands")
    assert bands
    assert all(
        _mapping(band, "bands[]").get("fell_back_to_perron") is False
        for band in bands
    )
