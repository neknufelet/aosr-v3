"""基礎物理量那張卡刻意不咬的幾種寫法：餵一棵綠樹下去必須回 0。

樣本樹只證明會紅；「設定層在函式裡讀不咬、函式裡 c = 343.0 不咬、rho_coefficient 這種鍵不咬、
只有幾何的答案不記 ρc 不咬」這四件事要另開一棵綠樹證明（第二輪找碴點的）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from governance.checks import physics_constants_single_source as subject

REPO = Path(__file__).resolve().parents[1]
MINI_CARD = REPO / "governance/fixtures/physics-constants-single-source/control/governance/rules/physics-constants-single-source.toml"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_deliberately_uncaught_shapes_are_green(tmp_path: Path) -> None:
    root = tmp_path / "green"
    shutil.copy(MINI_CARD, _mk(root / "governance/rules/physics-constants-single-source.toml"))
    _write(root / "src/aosr/config/data/physics_constants.toml", "air_density_kg_m3 = 1.2\nsound_speed_m_s = 343.0\nrho_coefficient = 0.5\n")
    _write(
        root / "src/aosr/config/physics_constants.py",
        "def load_physics_constants(path: str) -> dict[str, float]:\n    return {}\n\n\n"
        "def defaults() -> dict[str, float]:\n    return load_physics_constants('x')\n",
    )
    _write(
        root / "src" / "aosr" / "physics" / "thing.py",
        "def run(sound_speed_m_s: float) -> float:\n    c = 343.0\n    return sound_speed_m_s + c\n",
    )
    _write(
        root / "blueprint/reference_room_answers.json",
        json.dumps({"parameters": {"sound_speed_m_s": 343.0}}),
    )
    _write(
        root / "blueprint/reference_art_flat.json",
        json.dumps({"parameters": {"sound_speed_m_s": {"dec": "343.0", "hex": float(343.0).hex()}, "rho_c_pa_s_per_m": {"hex": float(411.6).hex()}}}),
    )
    assert subject.check(root, _tree(root)) == []


def _mk(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
