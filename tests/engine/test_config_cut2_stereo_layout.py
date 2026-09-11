"""舊家的 stereo_layout 考卷帶過來——**切掉兩處**，其餘原封不動。

來源：上一代那一棵樹的 stereo layout 考卷（7 題，不在這個 repo 裡）。可以帶的理由：它只碰
``lib.config.fem_lane`` 與 ``lib.config.stereo_layout``。兩處切掉：

1. ``test_stereo_geometry_constants_match_toml_loader``——它比的是 ``lib.physics.stereo_geometry``
   底下那一排 ``STEREO_*`` 常數。那一支住第 5 層（``physics``），這一刀還沒長；它要驗的
   「同一組政策值只有一個家」在這一刀**沒有對象**。
2. ``test_mutated_toml_value_flows_through_loader_to_geometry_facade``——它 spawn 一個子程序
   去驗「突變過的 TOML 會流到 ``lib.physics.stereo_geometry``」。這是**行為**不是值，而它要驗
   的那條線同樣還沒長出來。

**這兩處的裁判改由誰接手**：19 個政策欄位的值由 donor 標準答案接手
（``blueprint/config_cut1_answers.json`` 的 ``cut2_stereo_layout.load.real`` 那一筆，逐欄比對）；
「那一排 ``STEREO_*`` 常數跟這份政策是不是同一組」這一條**沒有人接手**，那是 ``physics``
那一塊長出來時自己的事（見 PR 內文的「沒帶過來的 v2 考卷」那張表）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.fem_lane import POSITION_MIN_SOURCE_SOURCE_M_DEFAULT
from aosr.config.paths import config_path
from aosr.config.stereo_layout import (
    STEREO_LAYOUT,
    StereoLayoutConfig,
    load_stereo_layout,
)


_CONFIG_PATH = config_path("stereo_layout.toml")


def test_stereo_and_generic_source_spacing_keys_remain_distinct() -> None:
    assert STEREO_LAYOUT.min_source_source_m == 1.5
    assert POSITION_MIN_SOURCE_SOURCE_M_DEFAULT == 0.25
    assert "min_source_source_m = 1.5" in _CONFIG_PATH.read_text()
    fem_lane_path = _CONFIG_PATH.with_name("fem_lane.toml")
    assert "min_source_source_m = 0.25" in fem_lane_path.read_text()


def test_stereo4_policy_values_are_loaded_from_toml() -> None:
    assert STEREO_LAYOUT.horizontal_angle_max_deg == 100.0
    assert STEREO_LAYOUT.min_front_depth_m == 0.10
    assert STEREO_LAYOUT.search_horizontal_angle_max_deg == 70.0
    assert STEREO_LAYOUT.search_min_front_depth_m == 1.0
    assert STEREO_LAYOUT.speaker_front_wall_min_m == 0.15
    assert STEREO_LAYOUT.speaker_front_wall_max_m == 1.5
    assert STEREO_LAYOUT.speaker_lateral_wall_margin_m == 0.5
    assert STEREO_LAYOUT.seat_rear_wall_margin_frac == 0.15


def test_the_module_level_instance_is_the_loader_default() -> None:
    """模組層那個 ``STEREO_LAYOUT`` 就是「不傳路徑」載入出來的東西（載入期讀檔照抄不改）。

    上一代那一條子程序探針驗的是「突變會流到 geometry facade」，那一段沒帶；這一條驗的
    是同一件事**在這一塊裡面**的那一半：模組層常數與預設路徑指的是同一份檔案。
    """
    assert STEREO_LAYOUT == load_stereo_layout()
    assert load_stereo_layout(config_path("stereo_layout.toml")) == STEREO_LAYOUT


def test_every_policy_field_is_a_float_on_the_loaded_object() -> None:
    """19 個政策欄位**具名**列出來、一個都不少、而且都是浮點。

    欄位名寫死在這裡而不是從 ``__dataclass_fields__`` 撈，是為了讓「上一代有的那 19 個」
    在這一份考卷裡看得見——從類別自己撈的話，有人刪掉一個欄位這一條照樣綠。
    """
    expected_fields = (
        "source_wall_margin_m",
        "receiver_wall_margin_m",
        "min_source_source_m",
        "min_source_seat_m",
        "min_seat_pair_center_m",
        "min_front_depth_m",
        "search_min_front_depth_m",
        "speaker_front_wall_min_m",
        "speaker_front_wall_max_m",
        "speaker_lateral_wall_margin_m",
        "seat_rear_wall_margin_frac",
        "receiver_centerline_tol_m",
        "horizontal_angle_min_deg",
        "horizontal_angle_target_deg",
        "horizontal_angle_max_deg",
        "search_horizontal_angle_max_deg",
        "horizontal_angle_target_scale_deg",
        "pair_center_offset_max_m",
        "distance_imbalance_max",
    )
    assert set(expected_fields) == set(StereoLayoutConfig.__dataclass_fields__)
    loaded = load_stereo_layout()
    for name in expected_fields:
        assert isinstance(getattr(loaded, name), float), name


@pytest.mark.parametrize(
    "mutation",
    [
        "\n[unexpected]\nvalue = 1.0\n",
        "\nunknown_policy = 1.0\n",
    ],
)
def test_stereo_layout_loader_rejects_unknown_keys(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "stereo_layout.toml"
    text = _CONFIG_PATH.read_text()
    if mutation.startswith("\nunknown"):
        text = text.replace("[stereo]\n", f"[stereo]{mutation}")
    else:
        text += mutation
    path.write_text(text)

    with pytest.raises(ValueError, match="unknown"):
        load_stereo_layout(path)


def test_stereo_layout_loader_rejects_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "stereo_layout.toml"
    path.write_text(
        _CONFIG_PATH.read_text().replace(
            "distance_imbalance_max = 0.20",
            "distance_imbalance_max = nan",
        )
    )

    with pytest.raises(ValueError, match="finite"):
        load_stereo_layout(path)
