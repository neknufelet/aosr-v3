"""開跑時另外指定往下走步數上限（#1073）：只換那一格、存進 project.json、接續與報告照記的套回去。

整條命令列（帶 ``--max-steps`` 開跑、截掉一段再接續、拿掉那一格要拒收）住 ``slow_material_gradient_cli``，合併前本機跑。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings, with_max_steps
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import cli, run
from tests.engine.test_material_gradient_mount import _project

REQUIRED = ("start", "--scheme", "s.json", "--surrogate", "m.npz", "--presets", ",".join(["none"] * len(Wall.all())),
            "--purpose", "p", "--root", "r")


def _settings() -> MaterialGradientSettings:
    return load_material_gradient_settings(config_path("material_gradient.toml"))


def _recorded(tmp_path: Path, max_steps: int | None) -> run.GradientProject:
    """設定檔抄一份到暫存目錄，project 指到它；指定步數時指紋記套上之後的（跟 ``cli._start`` 同一個寫法）。"""
    path = tmp_path / "settings.toml"
    path.write_bytes(config_path("material_gradient.toml").read_bytes())
    settings = _settings() if max_steps is None else with_max_steps(_settings(), max_steps)
    return _project().model_copy(update={"settings_path": str(path), "settings_fingerprint": settings.fingerprint,
                                         "max_steps": max_steps})


def test_with_max_steps_changes_only_the_step_cap() -> None:
    """抓：套步數時順手動到別格、指紋沒跟著變，或套回原值也換了指紋。"""
    base = _settings()
    wanted = base.descent.max_steps + 7
    changed = with_max_steps(base, wanted)
    assert changed.model_dump() == base.model_dump() | {"descent": base.descent.model_dump() | {"max_steps": wanted}}
    assert changed.fingerprint != base.fingerprint and changed.full_fingerprint != base.full_fingerprint
    assert with_max_steps(base, base.descent.max_steps).fingerprint == base.fingerprint


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_with_max_steps_keeps_the_descent_checks(value: object) -> None:
    """抓：另外指定的步數繞過設定檔那一格的檢查（整數、至少 1）。"""
    with pytest.raises(ValidationError):
        with_max_steps(_settings(), cast(int, value))


def test_start_flag_takes_only_positive_integers() -> None:
    """抓：旗標沒接上 ``start``、預設不是「照設定檔」、或收了零、負數、小數。"""
    parser = cli._parser()
    assert parser.parse_args(REQUIRED).max_steps is None
    assert parser.parse_args([*REQUIRED, "--max-steps", "5"]).max_steps == 5
    for text in ("0", "-3", "1.5", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._max_steps(text)


def test_project_records_the_cap_only_when_given(tmp_path: Path) -> None:
    """抓：沒指定也多寫一格（舊資料夾的 project.json 位元組會變）、指定了讀回來掉值，或收了零與字串。"""
    plain = _recorded(tmp_path, None)
    assert "max_steps" not in plain.model_dump_json()
    flagged = _recorded(tmp_path, 7)
    assert run.GradientProject.model_validate_json(flagged.model_dump_json()) == flagged
    for value in (0, "7"):
        with pytest.raises(ValidationError):
            run.GradientProject.model_validate({**flagged.model_dump(mode="json"), "max_steps": value})


def test_project_settings_reapplies_the_recorded_cap(tmp_path: Path) -> None:
    """抓：接續、終點、報告重讀設定檔時沒把開跑時指定的步數套回去（指紋對不上、拒收自己的資料夾）。"""
    plain, flagged = _recorded(tmp_path, None), _recorded(tmp_path, 7)
    assert run.project_settings(plain) == load_material_gradient_settings(plain.settings_path)
    applied = run.project_settings(flagged)
    assert applied.descent.max_steps == 7 and applied.fingerprint == flagged.settings_fingerprint
    assert applied == with_max_steps(load_material_gradient_settings(flagged.settings_path), 7)


def test_report_line_names_where_the_cap_came_from(tmp_path: Path) -> None:
    """抓：報告開頭那一行漏印、把開跑時指定的當成設定檔的，或設定讀不到、指紋不同時硬印一個數。"""
    flagged, plain = _recorded(tmp_path, 7), _recorded(tmp_path, None)
    assert cli._max_steps_line(flagged) == "往下走步數上限：7 步（開跑時指定）\n"
    assert cli._max_steps_line(plain) == f"往下走步數上限：{_settings().descent.max_steps} 步（設定檔）\n"
    other = plain.model_copy(update={"settings_fingerprint": "e" * 64})
    assert cli._max_steps_line(other) == "往下走步數上限：設定指紋不同，無法確認這一跑的上限\n"
    missing = plain.model_copy(update={"settings_path": str(tmp_path / "missing.toml")})
    assert cli._max_steps_line(missing) == "往下走步數上限：設定讀不到（FileNotFoundError）\n"
