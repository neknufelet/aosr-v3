"""4 支**本來就有預設**的載入器：不帶參數那一條路要踩到（找碴 F3）。

**這一支在補什麼洞。** `calibration`／`mat_continuation`／`scoring_v2`／`stereo_layout`
這 4 支上一代就有 `path=None` 的預設，這一刀把那些預設從「自己往上數兩層」換成
`paths.config_path("<name>.toml")`——**那個換法本身沒有裁判**：找碴實測
`load_calibration()`／`load_mat_continuation()`／`load_scoring_v2()` 不帶參數的呼叫在
`tests/` 命中 0 次。`paths.py` 的檔頭自己就指名「深度少一層會很安靜地錯」，而剛好沒有
人踩那條路。

**這一支比的是「兩條路走到同一個地方」**：不帶參數載入出來的東西，必須等於明著餵
`config_path("<name>.toml")` 載入出來的東西。這樣一來，`config_path` 指錯地方、或某一支
的預設忘了跟著換，這裡當場紅——而且它不比對「路徑字串長什麼樣」，只比對「讀到的內容」，
所以換一棵 checkout 也成立。
"""
from __future__ import annotations

import pytest

from aosr.config.calibration import CalibrationConfig, load_calibration
from aosr.config.mat_continuation import MatContinuationConfig, load_mat_continuation
from aosr.config.paths import config_path
from aosr.config.scoring_v2 import ScoringV2Config, load_scoring_v2
from aosr.config.stereo_layout import STEREO_LAYOUT, StereoLayoutConfig, load_stereo_layout


def test_load_calibration_default_is_the_package_data_file() -> None:
    """`load_calibration()`（不帶參數）＝ `load_calibration(config_path("calibration.toml"))`。"""
    via_default = load_calibration()
    assert isinstance(via_default, CalibrationConfig)
    assert via_default == load_calibration(config_path("calibration.toml"))


def test_load_mat_continuation_default_is_the_package_data_file() -> None:
    """同上，`mat_continuation.toml`。"""
    via_default = load_mat_continuation()
    assert isinstance(via_default, MatContinuationConfig)
    assert via_default == load_mat_continuation(config_path("mat_continuation.toml"))


def test_load_scoring_v2_default_is_the_package_data_file() -> None:
    """同上，`scoring_v2.toml`（指紋那一格也一起比）。"""
    via_default = load_scoring_v2()
    assert isinstance(via_default, ScoringV2Config)
    explicit = load_scoring_v2(config_path("scoring_v2.toml"))
    assert via_default == explicit
    assert via_default.fingerprint() == explicit.fingerprint()


def test_load_stereo_layout_default_and_module_level_instance_agree() -> None:
    """`load_stereo_layout()`、明著餵路徑、以及模組層那個 `STEREO_LAYOUT` 三個同一份。

    `stereo_layout` 是**載入期讀檔**那一支（老闆拍板甲：模組層常數照抄不改），所以它的
    預設路徑有沒有指對，要看 `STEREO_LAYOUT` 是不是跟另外兩條路走到同一個地方。
    """
    via_default = load_stereo_layout()
    assert isinstance(via_default, StereoLayoutConfig)
    assert via_default == load_stereo_layout(config_path("stereo_layout.toml"))
    assert via_default == STEREO_LAYOUT


@pytest.mark.parametrize(
    "name",
    ["calibration.toml", "mat_continuation.toml", "scoring_v2.toml", "stereo_layout.toml"],
)
def test_the_four_defaulted_loaders_read_files_that_exist(name: str) -> None:
    """這 4 個檔真的在 `paths.config_path(...)` 指到的地方（不是安靜地指到別處）。"""
    assert config_path(name).is_file(), f"{config_path(name)} 不在——預設路徑指錯了"
