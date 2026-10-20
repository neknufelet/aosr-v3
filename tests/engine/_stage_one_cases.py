"""第一階段考卷的造數方案與有限元素替身；不讀工作區方案。"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import TypedDict

import pytest

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.reporting import fem_slices
from aosr.reporting.scheme import Scheme
from aosr.scoring.receiver_set import primary_cross
from tests.engine._directivity import DIRECTIVITY
from tests.engine._fem_cli_cases import fake_slices
from tests.engine.test_scheme_pipeline import _scheme


def scheme() -> Scheme:
    original = _scheme("wall-1")
    assert original.scene.impedance_pa_s_per_m_by_wall is not None
    impedance = 4.0 * original.scene.density_kg_m3 * original.scene.sound_speed_m_s
    scene = original.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": {
        wall: impedance for wall in original.scene.impedance_pa_s_per_m_by_wall}})
    return original.model_copy(update={"scene": scene, "receiver_set": primary_cross(
        primary_position_m=original.receiver_set.primary.position_m, offset_m=0.1)})


def shards(monkeypatch: pytest.MonkeyPatch, item: Scheme, slices: int = 2,
           solver: Callable[..., dict[str, dict[tuple[str, str], tuple[float, ...]]]] = fake_slices,
           ) -> list[fem_slices.FemShard]:
    monkeypatch.setattr(fem_slices, "solve_fem_energies_many", solver)
    return [fem_slices.solve_slice(
        [item], capabilities=load_capabilities(config_path("capabilities.toml")),
        directivity=DIRECTIVITY, slices=slices, slice_index=index,
        physics_identity="phys-v1:" + "0" * 64) for index in range(slices)]


class RunKwargs(TypedDict):
    """具名的第一階段執行條件。"""

    capabilities: CapabilityTable
    directivity: DirectivityDefaults
    registry: QualityTargets
    physics_identity: str
    program_fingerprint: str
    engine_commit: str
    run_date: date


def run_kwargs() -> RunKwargs:
    return {"capabilities": load_capabilities(config_path("capabilities.toml")),
            "directivity": DIRECTIVITY,
            "registry": load_quality_targets(config_path("quality_targets.toml")),
            "physics_identity": "phys-v1:" + "0" * 64,
            "program_fingerprint": "calc-v1:" + "0" * 64,
            "engine_commit": "control", "run_date": date(2026, 10, 20)}


def cli_args(path: Path, out: Path, parts: list[Path] | None = None) -> list[str]:
    return ["stage-one", str(path), "--out", str(out), "--capabilities",
            str(config_path("capabilities.toml")), "--engine-commit", "control",
            "--run-date", "2026-10-20",
            *([] if parts is None else ["--fem-parts", *map(str, parts)])]
