"""`w_dip_from_bias` 的考卷（峰谷偏向那條旋鈕）——把上一代**不需要 JAX** 的那幾條帶過來。

**這一支在補什麼洞（找碴 F2）。** `scoring.py` 是這一刀唯一一支帶真數學的模組，而
`w_dip_from_bias` 在整份 repo **零考卷**（`grep -rn` 命中 0 次，控制組 `load_perceptual`
12 次）。上一代那一棵樹的 peak/dip 考卷（不在這個 repo 裡）有 5 條，其中 **3 條不需要 JAX**、
可以直接帶過來；這一支就是那 3 條。

**沒帶過來的 2 條**（它們要 `lib.scoring.core` 那兩個函式與 `jax`，住第 6 層，這一刀
沒有它們）：`test_dial_delivers_the_ratio_on_the_penalty_not_on_the_squared_term`
（用 `_asym_spread` 量回傳的罰則比值）與 `test_configured_bias_reaches_the_production_loss`
（用 `layer_a_loss` 驗配置值真的走到生產損失函式）、以及 `test_wiring_switched_the_target_term_from_std_to_rms`。
**誰接手**：scoring 那一塊（#135）——那三條驗的是「這條旋鈕在損失函式裡的角色」，
不是這個函式自己的算術。

第 1 條的參數表逐字照抄上一代：`[(3.0, 1/9), (1.5, 0.25), (0.0, 1.0), (-1.5, 4.0), (-3.0, 9.0)]`。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.scoring import load_scoring, w_dip_from_bias


@pytest.mark.parametrize(
    ("bias", "expected_w_dip"),
    [(3.0, 1.0 / 9.0), (1.5, 0.25), (0.0, 1.0), (-1.5, 4.0), (-3.0, 9.0)],
)
def test_dial_maps_to_the_documented_penalty_ratio(bias: float, expected_w_dip: float) -> None:
    """`bias` 對到 `w_dip` 的對應表（上一代那條一字不改）。"""
    assert w_dip_from_bias(bias) == pytest.approx(expected_w_dip)


def test_out_of_range_bias_is_rejected_at_both_ends() -> None:
    """超出刻度就丟 `ValueError`，兩端都擋（上一代那條一字不改）。"""
    for bad in (3.1, -3.1):
        with pytest.raises(ValueError, match=r"\[-3, 3\]"):
            w_dip_from_bias(bad)


def test_toml_rejects_a_bias_outside_the_dial(tmp_path: Path) -> None:
    """整份設定檔載入時也擋：`peak_dip_bias = 4.0` 要丟 `ValidationError`（同上一代）。"""
    broken = tmp_path / "scoring.toml"
    broken.write_text(
        config_path("scoring.toml").read_text().replace(
            "peak_dip_bias = 0.0", "peak_dip_bias = 4.0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_scoring(broken)


def test_the_configured_dial_value_is_the_one_the_loader_reads() -> None:
    """設定檔裡那一格真的被讀進來（不是第二個家）——接線到一半的那一條。

    上一代那條完整的接線考卷（`test_configured_bias_reaches_the_production_loss`）要
    `layer_a_loss`（#135）；這一條是它在這一刀的等價物：載入器讀到的那一格必須等於
    TOML 裡寫的那個值。
    """
    config = load_scoring(config_path("scoring.toml"))
    assert config.asymmetry.peak_dip_bias == 0.0
