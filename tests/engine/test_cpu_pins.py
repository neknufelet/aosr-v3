"""共用小工具 forward_cpu_pins：父行程有設的三格原樣帶進子行程、沒設的不補、雲端每片都真的設了這三格。"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aosr.runtime import child_process_env
from tests.engine._cpu_pins import CPU_PIN_VARS, forward_cpu_pins

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github/workflows/verify.yml"
PINS = {"NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR", "OPENBLAS_CORETYPE": "Haswell",
        "XLA_FLAGS": "--xla_cpu_max_isa=AVX2"}


def test_pins_set_in_the_parent_reach_the_child_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PINS.items():
        monkeypatch.setenv(name, value)
    pinned = forward_cpu_pins(child_process_env(threads=1))
    assert {name: pinned[name] for name in PINS} == PINS
    assert pinned["OPENBLAS_NUM_THREADS"] == "1" and "PYTHONPATH" in pinned, "child_process_env 給的那幾格要原樣留著"


def test_pins_missing_in_the_parent_are_not_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PINS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENBLAS_CORETYPE", "Haswell")
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "")
    base = {"PATH": "/bin", "OPENBLAS_CORETYPE": "Prescott"}
    pinned = forward_cpu_pins(base)
    assert pinned == {"PATH": "/bin", "OPENBLAS_CORETYPE": "Haswell", "NPY_DISABLE_CPU_FEATURES": ""}, (
        "父行程有設的照抄（空字串也是設了、蓋掉傳進來的舊值），沒設的 XLA_FLAGS 不補")
    assert base == {"PATH": "/bin", "OPENBLAS_CORETYPE": "Prescott"}, "傳進來的那一份不准被改"


def test_every_cloud_shard_sets_the_pins_the_helper_forwards() -> None:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    shards = {name: job for name, job in jobs.items() if name.startswith("shard-")}
    assert shards, "verify.yml 裡找不到 shard- 開頭的 job"
    seen: set[tuple[str, ...]] = set()
    for name, job in shards.items():
        env = job.get("env") or {}
        assert set(CPU_PIN_VARS) <= env.keys(), f"{name} 的 job 層沒設齊三格：{sorted(set(CPU_PIN_VARS) - env.keys())}"
        seen.add(tuple(str(env[pin]) for pin in CPU_PIN_VARS))
    assert len(seen) == 1, f"各片的三格不一樣：{seen}"
