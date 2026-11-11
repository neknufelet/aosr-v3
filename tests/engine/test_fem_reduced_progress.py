"""#1098 建模與留出驗證的進度回呼只讀：給不給回呼，替身與驗證報告逐位相同；回呼收到的每一輪跟建模帳一致。"""
from __future__ import annotations

import dataclasses

import numpy as np

from aosr.physics.fem_reduced_build import STOP_BUDGET, BuildProgress, build_reduced_model
from aosr.physics.fem_reduced_check import validate_reduced
from aosr.physics.fem_reduced_store import computation_digest, matrices_digest
from tests.engine._zone_rom_case import AXIS, SETTINGS, materials, operators


def test_build_progress_is_read_only_and_matches_the_ledger() -> None:
    """抓回呼改到選點或結果（替身摘要不同）、漏報某一輪、或報的次數與秩跟建模帳對不上。

    起點只解 1 點、每輪補 1 點、停止值小到不可能達標、預算 4 次：一定走過 4 輪補點、照預算停。
    回呼當下那一輪的點還沒解（``solved`` 是 0，下一輪開解才寫進帳），所以拿帳裡每一輪把 ``solved`` 換成 0 來比。
    """
    full = operators()
    training = materials(7, 8)
    settings = SETTINGS.model_copy(update={"initial_impedance_count": 1, "initial_frequency_count": 1, "batch": 1,
                                           "max_full_solves": 4, "stop_residual": 1e-30})
    plain = build_reduced_model(full, AXIS, training, settings=settings, dense_threads=1)
    reports: list[BuildProgress] = []
    watched = build_reduced_model(full, AXIS, training, settings=settings, dense_threads=1, progress=reports.append)
    assert matrices_digest(watched.matrices) == matrices_digest(plain.matrices)
    assert computation_digest(watched) == computation_digest(plain)
    assert watched.ledger.rounds == plain.ledger.rounds and watched.ledger.full_solves == plain.ledger.full_solves
    ledger = watched.ledger
    assert ledger.stop_reason == STOP_BUDGET and len(ledger.rounds) == 4
    assert [item.solved for item in ledger.rounds] == [1, 1, 1, 0]
    rounds = [report.latest for report in reports if report.latest is not None]
    assert len(reports) == len(rounds) == 4
    assert rounds == [dataclasses.replace(item, solved=0) for item in ledger.rounds]
    assert [report.full_solves for report in reports] == [1, 2, 3, 4]
    assert [report.rank for report in reports] == [item.rank for item in ledger.rounds]
    last = reports[-1]
    assert last.full_solves == ledger.full_solves and last.rank == ledger.rank
    assert last.max_full_solves == 4 and last.stop_residual == 1e-30
    assert all(report.elapsed_s >= 0.0 and "snapshot" in report.seconds for report in reports)


def test_validation_progress_counts_every_holdout_point_without_changing_the_report() -> None:
    """抓驗證回呼少報一組、總點數不是組數 × 頻點數、或給了回呼報告就變了。"""
    full = operators()
    model = build_reduced_model(full, AXIS, materials(7, 8), settings=SETTINGS, dense_threads=1)
    holdout = materials(11, 3)
    plain = validate_reduced(model, full, holdout, dense_threads=1)
    calls: list[tuple[int, int]] = []
    watched = validate_reduced(model, full, holdout, dense_threads=1, progress=lambda done, total: calls.append((done, total)))
    total = len(holdout) * len(AXIS)
    assert calls == [((index + 1) * len(AXIS), total) for index in range(len(holdout))]
    np.testing.assert_array_equal(watched.full_pressure, plain.full_pressure)
    np.testing.assert_array_equal(watched.reduced_pressure, plain.reduced_pressure)
    assert watched.frequency_db_by_seat == plain.frequency_db_by_seat and watched.band_db_by_seat == plain.band_db_by_seat
