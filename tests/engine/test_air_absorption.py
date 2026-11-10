"""空氣吸收的公開表值、能量換算與輸入邊界考卷。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value


class TableRow(TypedDict):
    """一列公開表值與其溫濕度。"""

    temperature_c: float
    relative_humidity_percent: float
    alpha_db_per_km: list[float]
    printed: list[str]


class PublicTable(TypedDict):
    """答案檔中本考卷使用的欄位。"""

    pressure_kpa: float
    nominal_frequencies_hz: list[float]
    printed_significant_figures: int
    rows: list[TableRow]


def _table() -> PublicTable:
    path = Path(__file__).resolve().parents[2] / "blueprint" / "air_absorption_iso9613_rows.json"
    return cast(PublicTable, json.loads(path.read_text(encoding="utf-8")))


def _exact_frequencies() -> NDArray[np.float64]:
    """ISO 266 的精確中心頻率，不能拿表頭名義頻率來算。"""
    return 1000.0 * 10.0 ** (np.arange(-13, 11, dtype=np.float64) / 10.0)


def _within_contract(actual: NDArray[np.float64], expected: NDArray[np.float64], ruler: float) -> bool:
    """跟變異題共用同一把逐格相對差裁判；非有限結果一定判紅。"""
    return bool(np.all(np.isfinite(actual)) and np.all(np.abs(actual - expected) <= ruler * np.abs(expected)))


def test_public_table_rows_within_ruler() -> None:
    """攝氏、百分比與公尺單位換錯，或改用名義頻率，都會讓公開表值超尺。"""
    from aosr.physics.air_absorption import air_attenuation_db_per_m

    table = _table()
    frequencies = _exact_frequencies()
    ruler = contract_value("air_absorption_vs_iso9613_table")
    assert len(table["rows"]) == 3
    assert len(table["nominal_frequencies_hz"]) == frequencies.size == 24
    assert table["printed_significant_figures"] == 3
    assert table["pressure_kpa"] * 1000.0 == 101325.0
    for row in table["rows"]:
        expected = np.asarray(row["alpha_db_per_km"], dtype=np.float64)
        assert expected.shape == (24,)
        assert len(row["printed"]) == 24
        assert np.array_equal(np.asarray(row["printed"], dtype=np.float64), expected)
        assert row["printed"] == [f"{value:.2E}" for value in expected]
        actual = air_attenuation_db_per_m(
            frequencies, temperature_c=row["temperature_c"],
            relative_humidity_percent=row["relative_humidity_percent"], pressure_pa=101325.0,
        )
        assert actual.shape == frequencies.shape
        assert actual.dtype == np.float64
        assert _within_contract(actual * 1000.0, expected, ruler), row


def test_mutant_beyond_tolerance_is_red() -> None:
    """比較器在尺兩側一綠一紅，並攔住攝氏、濕度、氣壓與名義頻率四種錯法。"""
    from aosr.physics.air_absorption import air_attenuation_db_per_m

    table = _table()
    ruler = contract_value("air_absorption_vs_iso9613_table")
    frequencies = _exact_frequencies()
    for row in table["rows"]:
        truth = np.asarray(row["alpha_db_per_km"], dtype=np.float64)
        inside = truth * (1.0 + (1.0 - MUTANT_MARGIN) * ruler)
        outside = truth * (1.0 + (1.0 + MUTANT_MARGIN) * ruler)
        assert _within_contract(inside, truth, ruler)
        assert not _within_contract(outside, truth, ruler)
        assert math.isclose(
            float(np.max(np.abs(outside - truth) / (ruler * truth))), 1.0 + MUTANT_MARGIN,
            rel_tol=MUTANT_MARGIN / 4.0,
        )
        temperature = row["temperature_c"]
        humidity = row["relative_humidity_percent"]
        for label, wrong_frequencies, wrong_temperature, wrong_humidity, wrong_pressure in (
            ("把 K 當攝氏", frequencies, temperature + 273.15, humidity, 101325.0),
            ("把濕度當小數", frequencies, temperature, humidity / 100.0, 101325.0),
            ("把 kPa 當 Pa", frequencies, temperature, humidity, 101.325),
            ("拿名義頻率來算", np.asarray(table["nominal_frequencies_hz"]), temperature, humidity, 101325.0),
        ):
            wrong = air_attenuation_db_per_m(
                wrong_frequencies, temperature_c=wrong_temperature,
                relative_humidity_percent=wrong_humidity, pressure_pa=wrong_pressure,
            )
            assert not _within_contract(wrong * 1000.0, truth, ruler), label
    assert not _within_contract(np.full(24, math.nan), np.ones(24), ruler)


@pytest.mark.parametrize("row", _table()["rows"])
@pytest.mark.parametrize("mean_free_path_m", (0.5, 2.9, 10.0))
def test_absorption_fraction_matches_decibel_form(row: TableRow, mean_free_path_m: float) -> None:
    """能量比例對獨立十次方寫法；若把能量換成振幅的除以二十，必須超尺。"""
    from aosr.physics.air_absorption import air_absorption_fraction, air_attenuation_db_per_m

    conditions = {
        "temperature_c": row["temperature_c"],
        "relative_humidity_percent": row["relative_humidity_percent"],
        "pressure_pa": 101325.0,
    }
    frequencies = _exact_frequencies()
    alpha = air_attenuation_db_per_m(frequencies, **conditions)
    expected = 1.0 - 10.0 ** (-alpha * mean_free_path_m / 10.0)
    actual = air_absorption_fraction(frequencies, mean_free_path_m, **conditions)
    ruler = contract_value("late_energy_physical_property")
    assert actual.shape == frequencies.shape
    assert actual.dtype == np.float64
    assert np.all((actual > 0.0) & (actual < 1.0))
    assert _within_contract(actual, expected, ruler)
    amplitude_mutant = -np.expm1(-alpha * np.log(10.0) / 20.0 * mean_free_path_m)
    assert not _within_contract(amplitude_mutant, expected, ruler)


@pytest.mark.parametrize(("frequencies", "overrides", "message"), (
    (1000.0, {}, "一維"),
    ([[1000.0]], {}, "一維"),
    ([0.0], {}, "頻率"),
    ([-1.0], {}, "頻率"),
    ([math.nan], {}, "頻率"),
    ([math.inf], {}, "頻率"),
    ([-math.inf], {}, "頻率"),
    ([1000.0], {"temperature_c": -273.15}, "溫度"),
    ([1000.0], {"temperature_c": -274.0}, "溫度"),
    ([1000.0], {"temperature_c": math.nan}, "溫度"),
    ([1000.0], {"temperature_c": math.inf}, "溫度"),
    ([1000.0], {"temperature_c": -math.inf}, "溫度"),
    ([1000.0], {"relative_humidity_percent": -1.0}, "濕度"),
    ([1000.0], {"relative_humidity_percent": 101.0}, "濕度"),
    ([1000.0], {"relative_humidity_percent": math.nan}, "濕度"),
    ([1000.0], {"relative_humidity_percent": math.inf}, "濕度"),
    ([1000.0], {"relative_humidity_percent": -math.inf}, "濕度"),
    ([1000.0], {"pressure_pa": 0.0}, "氣壓"),
    ([1000.0], {"pressure_pa": -1.0}, "氣壓"),
    ([1000.0], {"pressure_pa": math.nan}, "氣壓"),
    ([1000.0], {"pressure_pa": math.inf}, "氣壓"),
    ([1000.0], {"pressure_pa": -math.inf}, "氣壓"),
    ([1000.0], {"mean_free_path_m": 0.0}, "平均自由路徑"),
    ([1000.0], {"mean_free_path_m": -1.0}, "平均自由路徑"),
    ([1000.0], {"mean_free_path_m": math.nan}, "平均自由路徑"),
    ([1000.0], {"mean_free_path_m": math.inf}, "平均自由路徑"),
    ([1000.0], {"mean_free_path_m": -math.inf}, "平均自由路徑"),
))
def test_rejects_bad_inputs(frequencies: ArrayLike, overrides: dict[str, float], message: str) -> None:
    """兩個入口都拒絕壞頻率與大氣條件；比例入口另守平均自由路徑。"""
    from aosr.physics.air_absorption import air_absorption_fraction, air_attenuation_db_per_m

    conditions = {
        "temperature_c": overrides.get("temperature_c", 20.0),
        "relative_humidity_percent": overrides.get("relative_humidity_percent", 30.0),
        "pressure_pa": overrides.get("pressure_pa", 101325.0),
    }
    if "mean_free_path_m" not in overrides:
        with pytest.raises(ValueError, match=message):
            air_attenuation_db_per_m(frequencies, **conditions)
    with pytest.raises(ValueError, match=message):
        air_absorption_fraction(frequencies, overrides.get("mean_free_path_m", 2.9), **conditions)
