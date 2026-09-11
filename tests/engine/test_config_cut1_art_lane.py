"""``aosr.config.art_lane`` 的凍結常數：值要跟上一代（donor）一模一樣。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。這一支只做兩件事：
每一個公開常數各一條斷言（逐項具名比對，不是比數量），以及答案檔的檔頭還在不在。

值不是抄的：``blueprint`` 底下那一份是
``blueprint/generate_config_cut1_answers.py`` 在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.art_lane as art_lane

from tests.engine._config_answers import constant, is_approx, load_answers


def test_answer_file_carries_a_donor_mark() -> None:
    """答案檔的檔頭要有 donor 記號與完整 commit sha（沒有就不是證據）。"""
    donor = load_answers()["donor"]
    assert isinstance(donor, dict), "答案檔的 donor 檔頭不是表"
    assert donor["tag"]
    assert donor["commit"]


def test_art_interface_constants_are_frozen() -> None:
    """ART 介面那幾個常數（工作面網格、Neumann 尾段、冪迭代、形狀因子、互易）。"""
    assert is_approx(art_lane.ART_N_PER_WALL_DEFAULT, constant("art_lane", "ART_N_PER_WALL_DEFAULT"))
    assert is_approx(art_lane.ART_FACE_GRID_N, constant("art_lane", "ART_FACE_GRID_N"))
    assert is_approx(art_lane.ART_NEUMANN_EPS_TAIL, constant("art_lane", "ART_NEUMANN_EPS_TAIL"))
    assert is_approx(art_lane.ART_POWERITER_K, constant("art_lane", "ART_POWERITER_K"))
    assert is_approx(art_lane.ART_FORMFACTOR_EPS, constant("art_lane", "ART_FORMFACTOR_EPS"))
    assert is_approx(art_lane.ART_RECIPROCITY_TOL, constant("art_lane", "ART_RECIPROCITY_TOL"))


def test_art_bounds_are_frozen() -> None:
    """兩個上限（patch 數、Neumann 掃描長度）。"""
    assert is_approx(art_lane.ART_P_CAP, constant("art_lane", "ART_P_CAP"))
    assert is_approx(art_lane.ART_NEUMANN_K_MAX, constant("art_lane", "ART_NEUMANN_K_MAX"))


def test_art_wls_t20_window_is_frozen() -> None:
    """T20 擬合的視窗與軟化係數（VAL-F6）。"""
    assert is_approx(art_lane.ART_WLS_T20_HI_DB, constant("art_lane", "ART_WLS_T20_HI_DB"))
    assert is_approx(art_lane.ART_WLS_T20_LO_DB, constant("art_lane", "ART_WLS_T20_LO_DB"))
    assert is_approx(
        art_lane.ART_WLS_WINDOW_SOFTNESS_DB, constant("art_lane", "ART_WLS_WINDOW_SOFTNESS_DB")
    )
