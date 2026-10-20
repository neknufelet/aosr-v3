"""第一階段命令列保留身分紀律，壞分片不退回求解。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.reporting import scheme_cli
from tests.engine import _stage_one_cases as cases
from tests.engine._fem_cli_cases import forbid_fem


def test_shuffled_parts_equal_regular_and_print_assumptions(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    item = cases.scheme()
    parts = cases.shards(monkeypatch, item)
    # 命令列的真實身分固定成分片的身分，其他指紋前後核對仍走原流程。
    monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", lambda **kwargs: parts[0].physics_identity)
    path = tmp_path / "scheme.json"
    path.write_text(item.model_dump_json())
    outputs = []
    for index, part in enumerate(parts):
        out = tmp_path / f"part-{index}.json"
        out.write_text(json.dumps({"shard": part.model_dump(mode="json")}))
        outputs.append(out)
    regular, injected = tmp_path / "regular.json", tmp_path / "injected.json"
    code = scheme_cli.main(cases.cli_args(path, regular))
    assert code == 0
    calls = forbid_fem(monkeypatch)
    code = scheme_cli.main(cases.cli_args(path, injected, list(reversed(outputs))))
    assert code == 0
    assert json.loads(regular.read_text()) == json.loads(injected.read_text())
    assert calls == []
    text = capsys.readouterr().out
    assert "未加入實際喇叭的低頻衰減，排名只適用於本次理想聲源基準；換播放系統後可能不同" in text
    assert "在這個牆面假設下：六面同一個頻率無關實數阻抗 4.0 ρc（ρc 是空氣特性阻抗）" in text
    for speaker in item.speakers:
        assert speaker in text
    for label in ("主項", "穩健性", "最差方向", "合分"):
        assert label in text


@pytest.mark.parametrize("damage", ["missing", "truncated", "identity", "axis", "coordinates", "overlap"])
def test_bad_parts_refuse_without_solving(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], damage: str) -> None:
    item = cases.scheme()
    parts = cases.shards(monkeypatch, item)
    monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", lambda **kwargs: "phys-v1:" + "0" * 64)
    path = tmp_path / "scheme.json"
    path.write_text(item.model_dump_json())
    if damage == "identity":
        parts[0] = parts[0].model_copy(update={"physics_identity": "different"})
    elif damage == "axis":
        item = item.model_copy(update={"scene": item.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.SEARCH_48})})
        path.write_text(item.model_dump_json())
    elif damage == "coordinates":
        point = next(iter(parts[0].candidates[item.scheme_id].speakers))
        parts[0].candidates[item.scheme_id].speakers[point] = (1.1.hex(), 1.2.hex(), 1.3.hex())
    outputs = []
    for index, part in enumerate(parts):
        out = tmp_path / f"part-{index}.json"
        out.write_text(json.dumps({"shard": part.model_dump(mode="json")}))
        outputs.append(out)
    if damage == "missing":
        outputs = outputs[1:]
    elif damage == "truncated":
        outputs[0].write_text('{"shard":')
    elif damage == "overlap":
        outputs.append(outputs[0])
    calls = forbid_fem(monkeypatch)
    out = tmp_path / "result.json"
    code = scheme_cli.main(cases.cli_args(path, out, outputs))
    assert code == (scheme_cli.PROGRAM_CHANGED_EXIT if damage == "identity" else 1)
    assert not out.exists()
    assert calls == []
    assert capsys.readouterr().err.strip()


@pytest.mark.parametrize("identity", ["program", "physics"])
def test_midrun_identity_change_leaves_no_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str) -> None:
    item = cases.scheme()
    cases.shards(monkeypatch, item)
    prefix = "calc-v1:" if identity == "program" else "phys-v1:"
    values = iter((prefix + "0" * 64, prefix + "1" * 64))
    target = "aosr.reporting.scheme_cli.calculation_fingerprint" if identity == "program" else "aosr.reporting.physics_identity.physics_identity"
    monkeypatch.setattr(target, lambda **kwargs: next(values))
    path, out = tmp_path / "scheme.json", tmp_path / "result.json"
    path.write_text(item.model_dump_json())
    previous = set(tmp_path.iterdir())
    assert scheme_cli.main(cases.cli_args(path, out)) == scheme_cli.PROGRAM_CHANGED_EXIT
    assert set(tmp_path.iterdir()) == previous


def test_atomic_replace_failure_preserves_previous_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = cases.scheme()
    cases.shards(monkeypatch, item)
    path, out = tmp_path / "scheme.json", tmp_path / "result.json"
    path.write_text(item.model_dump_json())
    out.write_text("previous")
    previous = set(tmp_path.iterdir())
    def fail_replace(source: str | Path, target: str | Path) -> None:
        assert Path(source).parent == out.parent and Path(target) == out
        assert json.loads(Path(source).read_text())["schema_version"] == "aosr.stage_one_result.v1"
        raise OSError("replacement interrupted")
    monkeypatch.setattr(os, "replace", fail_replace)
    code = scheme_cli.main(cases.cli_args(path, out))
    assert code == 1
    assert out.read_text() == "previous"
    assert set(tmp_path.iterdir()) == previous


def test_prerequisite_refusal_happens_before_solving(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    item = cases.scheme().model_copy(update={"source_model": "product_default"})
    path, out = tmp_path / "scheme.json", tmp_path / "result.json"
    path.write_text(item.model_dump_json())
    calls = forbid_fem(monkeypatch)
    assert scheme_cli.main(cases.cli_args(path, out)) == 1
    assert not out.exists()
    assert calls == []
    assert "全向" in capsys.readouterr().err
