"""實體範本 blueprint/scheme_stage_one_template.json：讀得進來、過前提、N 從檔裡的原數字現算。"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from aosr.reporting.scheme import load_scheme

BLUEPRINT = Path(__file__).resolve().parents[2] / "blueprint"
TEMPLATE = BLUEPRINT / "scheme_stage_one_template.json"


def test_template_loads_passes_prerequisites_and_computes_wall_multiple() -> None:
    module = importlib.import_module("aosr.reporting.stage_one_result")
    item = load_scheme(TEMPLATE)
    module.check_stage_one_prerequisites(item)
    assert item.scene.low_frequency_axis is None
    scene = json.loads(TEMPLATE.read_text(encoding="utf-8"))["scene"]
    (impedance,) = set(scene["impedance_pa_s_per_m_by_wall"].values())
    expected = impedance / (scene["density_kg_m3"] * scene["sound_speed_m_s"])
    assert f"{expected:.1f}" == "30.0"
    assert module.wall_assumption_text(item) == (
        f"在這個牆面假設下：六面同一個頻率無關實數阻抗 {expected:.1f} ρc（ρc 是空氣特性阻抗）")


def test_template_keeps_reference_room_seats_and_speakers() -> None:
    """座位與喇叭照參考方案原樣：方向標記沿用參考方案那一套，不另起一套。"""
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    reference = json.loads((BLUEPRINT / "scheme_reference_room.json").read_text(encoding="utf-8"))
    assert template["receiver_set"] == reference["receiver_set"]
    assert template["speakers"] == reference["speakers"]
    assert template["scene"]["room_m"] == reference["scene"]["room_m"]
