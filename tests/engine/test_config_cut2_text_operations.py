"""整份原文操作要餵指定內容；矛盾操作要在跑載入器之前拒絕；餵上一代的物性表只拿掉新家多的那幾格。"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from aosr.config import stereo_layout
from aosr.config.physics_constants import PhysicsConstants
from blueprint import config_cut1_cases as cases
from blueprint import generate_config_cut1_answers as generator
from tests.engine import _config_answers as answers


@pytest.mark.parametrize("runner", ["generator", "exam"])
@pytest.mark.parametrize(
    ("text", "message"),
    [("", "must define [stereo]"), ("stereo = 1.0\n", "[stereo] must be a table")],
)
def test_config_cut2_text_feeds_the_whole_file(
    runner: str, text: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """忽略 text 而餵真的檔案會成功；這裡必須收到指定分支的報錯。"""
    op: dict[str, object] = {"fn": "load_stereo_layout", "text": text}
    actual: object
    if runner == "generator":
        actual = generator.loader_block(stereo_layout, "cut2_stereo_layout", op, answers.config_data_dir(), tmp_path)
    else:
        case_id = "cut2_stereo_layout.load.mut.stereo-section-missing"
        monkeypatch.setattr(cases, "op_for", lambda case: op)
        monkeypatch.setattr(answers, "_tmp_root", lambda: tmp_path)
        with answers.loader_case_run(case_id) as result:
            actual = result
    assert actual == {"raised": {"type": "ValueError", "message": f"<path>/stereo_layout.toml {message}"}}


@pytest.mark.parametrize("runner", ["generator", "exam"])
@pytest.mark.parametrize("conflict", [{"mutations": []}, {"path": "missing"}])
def test_config_cut2_text_rejects_conflicting_operations(
    runner: str, conflict: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """靜靜擇一會回報載入結果，矛盾操作必須直接拒絕。"""
    op = {"fn": "load_stereo_layout", "text": "", **conflict}
    if runner == "generator":
        with pytest.raises(SystemExit, match="text.*mutations.*missing"):
            generator.loader_block(stereo_layout, "cut2_stereo_layout", op, answers.config_data_dir(), tmp_path)
    else:
        monkeypatch.setattr(cases, "op_for", lambda case: op)
        monkeypatch.setattr(answers, "_tmp_root", lambda: tmp_path)
        with pytest.raises(AssertionError, match="text.*mutations.*missing"):
            with answers.loader_case_run("cut2_stereo_layout.load.mut.stereo-section-missing"):
                pass


def _physics_text() -> str:
    return (answers.config_data_dir() / "physics_constants.toml").read_text(encoding="utf-8")


def test_donor_physics_text_keeps_exactly_the_shared_fields() -> None:
    """上一代收到的表，鍵要剛好是新家序列化出來的那幾格（排除的不給），值逐格不變。"""
    full = tomllib.loads(_physics_text())
    donor = tomllib.loads(generator._donor_physics_text(_physics_text()))
    shared = {name for name, field in PhysicsConstants.model_fields.items() if not field.exclude}
    assert set(donor) == shared
    assert donor == {key: full[key] for key in shared}


@pytest.mark.parametrize("key", generator.TMM_ONLY_PHYSICS_KEYS)
@pytest.mark.parametrize("how", ["missing", "twice"])
def test_donor_physics_text_rejects_a_key_not_written_exactly_once(key: str, how: str) -> None:
    """要拿掉的那一格不在、或寫了兩行，都不准默默餵下去。"""
    text = _physics_text()
    line = next(row for row in text.splitlines(keepends=True) if row.startswith(f"{key} ="))
    broken = text.replace(line, "") if how == "missing" else text.replace(line, line + line)
    with pytest.raises(SystemExit, match=key):
        generator._donor_physics_text(broken)
