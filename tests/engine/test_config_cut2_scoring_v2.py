"""舊家的 scoring_v2 考卷原封不動帶過來，只改 import 與設定檔路徑。

這一支是上一代那一棵樹的 scoring_v2 考卷（13 題，不在這個 repo 裡、只 import 上一代的
``lib.config.scoring_v2``）的整支搬遷：題目、斷言、參數化一字不改，只換三件事——
import 走新家、設定檔路徑走新家的 ``src/aosr/config/data/``、以及檔名（``tests/engine/``
那一籃的命名）。

**為什麼可以直接帶。** 這支考卷從頭到尾只碰 ``scoring_v2`` 這一支——沒有拉 JAX、
沒有碰別的層、沒有 scripts。它驗的是「同一份設定檔載進來的值」與「突變幾個鍵之後
炸什麼」，正是這一刀的合約。它跟 donor 標準答案（``blueprint/config_cut1_answers.json``）
是**兩道不同的裁判**：答案檔管「我沒帶過來那些**別的** case 還對不對」，這一支管
「上一代真的寫死過的那些值還一樣」。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest

from aosr.config.paths import config_path
from aosr.config.scoring_v2 import (
    SCORING_V2_SCHEMA_VERSION,
    DecisionState,
    ProbeOnlyScoringV2ConfigError,
    ScoringV2ConfigError,
    load_scoring_v2,
)


CONFIG_PATH = config_path("scoring_v2.toml")


def test_alpha1_config_loads_with_traceable_locked_and_provisional_values() -> None:
    config = load_scoring_v2(CONFIG_PATH)

    # 這一格是 Literal 而 Literal 只吃字面值，所以同一份檔裡的常數 `SCORING_V2_SCHEMA_VERSION`
    # 與那一格的字面是兩份寫法；這一條把它們釘在一起（改一個沒改另一個就在這裡紅）。
    assert config.schema_version == SCORING_V2_SCHEMA_VERSION
    assert SCORING_V2_SCHEMA_VERSION == "aosr.scoring.v2.alpha1"
    assert config.status == "probe-only"
    assert (config.fem.low_hz, config.fem.high_hz) == (20.0, 200.0)
    assert (config.ism.low_hz, config.ism.high_hz) == (200.0, 1000.0)
    assert config.response.equivalence_db == 0.5
    assert config.response.equivalence_sweep_db == (0.25, 0.5, 1.0)
    assert config.listening_cross.order == (
        "center",
        "front",
        "back",
        "left",
        "right",
        "up",
        "down",
    )
    assert config.dip_eligibility.min_width_oct == pytest.approx(1.0 / 6.0)
    assert config.rfz_v2.band_status == "unresolved"
    assert config.rfz_v2.threshold_status == "unresolved"
    assert config.fingerprint().startswith("sha256:")
    assert "peak_deadband_db" not in config.model_dump()
    assert "dip_deadband_db" not in config.model_dump()


def test_alpha1_decision_states_preserve_all_four_semantics() -> None:
    config = load_scoring_v2(CONFIG_PATH)

    assert config.decision_state("fem.low_hz") is DecisionState.LOCKED
    assert config.decision_state("response.equivalence_db") is DecisionState.PROVISIONAL
    assert config.decision_state("rfz_v2.band") is DecisionState.UNRESOLVED
    assert config.decision_state("runtime.solver_path_coverage") is DecisionState.UNAVAILABLE


def test_production_mode_refuses_probe_only_config() -> None:
    with pytest.raises(ProbeOnlyScoringV2ConfigError, match="probe-only"):
        load_scoring_v2(CONFIG_PATH, mode="production")


@pytest.mark.parametrize("mode", ["prod", "Production", "SIDEcar", "sidecar "])
def test_loader_rejects_unknown_runtime_mode_spellings(mode: str) -> None:
    with pytest.raises(ScoringV2ConfigError, match="unsupported scoring-v2 load mode"):
        load_scoring_v2(CONFIG_PATH, mode=cast(Literal["sidecar", "production"], mode))


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        ("high_hz = 200.0", "high_hz = 201.0", "frequency ranges must join"),
        (
            'order = ["center", "front", "back", "left", "right", "up", "down"]',
            'order = ["center", "back", "front", "left", "right", "up", "down"]',
            "local_cross_7 order",
        ),
        ("radius_m = 0.10", "radius_m = 0.0", "greater than 0"),
        ("frame_tolerance = 1e-6", "frame_tolerance = 0.0", "greater than 0"),
        ("min_width_oct = 0.16666666666666666", "min_width_oct = 0.16", "1/6 octave"),
        ('band_status = "unresolved"', 'band_status = "ready"', "unresolved"),
    ],
)
def test_loader_rejects_contract_drift(
    tmp_path: Path,
    before: str,
    after: str,
    message: str,
) -> None:
    invalid_path = tmp_path / "scoring_v2.toml"
    invalid_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(before, after), encoding="utf-8"
    )

    with pytest.raises(ScoringV2ConfigError, match=message):
        load_scoring_v2(invalid_path)


def test_the_mutations_the_old_suite_used_are_declared_in_the_case_table() -> None:
    """這一支帶過來的 6 組突變，case 表裡也要有（考卷與 donor 兩邊看同一份清單）。

    這一條刻意比**具名的集合**而不是筆數（``assertions-not-pinned-to-counts`` 咬後者）。
    """
    from blueprint import config_cut1_cases as cases

    declared: set[tuple[str, str]] = set()
    for case in cases.cases_for("cut2_scoring_v2"):
        raw = cases.op_for(case).get("mutations")
        if isinstance(raw, list):
            for mutation in raw:
                declared.add((mutation["before"], mutation["after"]))
    carried = {
        ("high_hz = 200.0", "high_hz = 201.0"),
        (
            'order = ["center", "front", "back", "left", "right", "up", "down"]',
            'order = ["center", "back", "front", "left", "right", "up", "down"]',
        ),
        ("radius_m = 0.10", "radius_m = 0.0"),
        ("frame_tolerance = 1e-6", "frame_tolerance = 0.0"),
        ("min_width_oct = 0.16666666666666666", "min_width_oct = 0.16"),
        ('band_status = "unresolved"', 'band_status = "ready"'),
    }
    assert carried <= declared, f"case 表少了這幾筆：{sorted(carried - declared)}"
