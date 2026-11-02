"""#996 共用存檔的讀回檢查是逐位的：只差一個浮點符號位（+0.0 對 −0.0）的兩份要判不同。

三條讀回檢查——低頻控制組 ``SchemeResult``（``test_low_frequency_response_reload.saved``）、縮小的有限元素替身
``ReducedModel``（``_fem_reduced_cache.shared_model``）、留出驗證報告 ``ValidationReport``
（``_fem_reduced_cache.shared_holdout_report``）——都用 ``_scheme_cache.bitwise_equal``。``==`` 把 +0.0 與 −0.0
判成相等（審查實測），存讀若把符號位弄掉就會被當成一致；這裡每一種各做一對只差那一位的物件，必須判不同，
同一份做兩次的一對必須判相同（判準沒有寬到什麼都說不同）。
"""
from __future__ import annotations

import numpy as np
import pytest

from aosr.config.fem_reduced import load_reduced_model_settings
from aosr.config.paths import config_path
from aosr.physics.fem_reduced import BuildLedger, ReducedMatrices, ReducedModel
from aosr.physics.fem_reduced_check import Band, ValidationReport
from aosr.reporting.result import SchemeResult
from tests.engine._scheme_cache import bitwise_equal
from tests.engine.test_low_frequency_response_reload import saved as saved

ZERO, NEGATIVE_ZERO = 0.0, -0.0


def _model(total_s: float) -> ReducedModel:
    """一階的小替身；建模帳的 ``seconds["total"]`` 由參數給。"""
    return ReducedModel(
        matrices=ReducedMatrices(stiffness=np.ones((1, 1)), mass=np.ones((1, 1)), boundary=np.ones((6, 1, 1)),
                                 sources=np.ones((1, 1), dtype=np.complex128), receivers=np.ones((1, 1))),
        source_names=("left",), receiver_names=("front",), frequencies_hz=(100.0,), density_kg_m3=1.2,
        sound_speed_m_s=343.0, mesh_hash="0" * 64, physics_identity="phys-v1:" + "0" * 64, input_hash="0" * 64,
        ledger=BuildLedger(full_solves=1, snapshots=2, rank=1, rounds=(), stop_reason="考卷", seconds={"total": total_s},
                           settings=load_reduced_model_settings(config_path("fem_reduced.toml"))))


def _report(frequency_db: float) -> ValidationReport:
    """一組材料、一個頻點的小報告；座位 front 的逐頻 dB 差由參數給。"""
    pressure = np.ones((1, 1, 1, 1), dtype=np.complex128)
    return ValidationReport(
        frequency_db_by_seat={"front": frequency_db}, band_db_by_seat={"front": 0.5}, partial_band_db_by_seat={},
        bands=(Band(lower_hz=89.1, upper_hz=112.2, indices=(0,), partial=False),), full_solves=1,
        full_pressure=pressure, reduced_pressure=pressure.copy(), model_digest="0" * 64, computation_digest="1" * 64,
        source_points={"left": ("0x1.0p-1", "0x1.0p-1", "0x1.0p-1")},
        receiver_points={"front": ("0x1.0p+0", "0x1.0p-1", "0x1.0p-1")})


def _result(saved: SchemeResult, solve_s: float) -> SchemeResult:
    """真的控制組結果，只把牆鐘秒數的 ``solve_s`` 換成參數。"""
    return saved.model_copy(update={"timings": saved.timings.model_copy(update={"solve_s": solve_s})})


@pytest.mark.parametrize("kind", ("scheme_result", "reduced_model", "validation_report"))
def test_readback_check_tells_positive_from_negative_zero(saved: SchemeResult, kind: str) -> None:
    """抓讀回檢查只比值：+0.0 與 −0.0 值相等、位元不同，三條檢查都要判不同；同值同位的兩份判相同。"""
    build = {"scheme_result": lambda value: _result(saved, value), "reduced_model": _model,
             "validation_report": _report}[kind]
    assert not bitwise_equal(build(ZERO), build(NEGATIVE_ZERO))
    assert not bitwise_equal(build(NEGATIVE_ZERO), build(ZERO))
    assert bitwise_equal(build(NEGATIVE_ZERO), build(NEGATIVE_ZERO))
    assert bitwise_equal(build(ZERO), build(ZERO))
