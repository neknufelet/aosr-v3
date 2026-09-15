"""票 #213：FEM 答案檔的出身、形狀與獨立解析物理健檢。"""
from __future__ import annotations

import copy
import json
import math
import tomllib
from pathlib import Path
from typing import Final

import pytest

from blueprint import reference_fem_check as check


_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_FEM_CASES: Final[tuple[str, ...]] = ("flat", "rigid")
_EXPECTED_SETS: Final[dict[str, set[str]]] = {
    "flat": {"A"},
    "rigid": {"A", "B"},
}
_PROVENANCE_CARD: Final[Path] = (
    _REPO_ROOT / "governance" / "rules" / "answer-files-carry-provenance.toml"
)


def _contract_value(name: str) -> float:
    path = _REPO_ROOT / "blueprint" / "precision_contracts.toml"
    with path.open("rb") as registry_file:
        loaded = tomllib.load(registry_file)
    contracts = loaded.get("contract")
    assert isinstance(contracts, list)
    match = next(item for item in contracts if isinstance(item, dict) and item.get("name") == name)
    value = match.get("value")
    assert isinstance(value, float)
    return value


_FEM_PHYSICS_TOLERANCE_REL = _contract_value("fem_rigid_modal_vs_analytic")


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{where} 不是表：{value!r}")
    return {str(key): item for key, item in value.items()}


def _list_of_mappings(value: object, where: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise AssertionError(f"{where} 不是清單：{value!r}")
    return [_mapping(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AssertionError(f"{where} 不是數字：{value!r}")
    return float(value)


def _xyz(value: object, where: str) -> tuple[float, float, float]:
    if not isinstance(value, list):
        raise AssertionError(f"{where} 不是清單：{value!r}")
    try:
        x_value, y_value, z_value = value
    except ValueError as exc:
        raise AssertionError(f"{where} 不是三軸座標：{value!r}") from exc
    return (
        _number(x_value, f"{where}.x"),
        _number(y_value, f"{where}.y"),
        _number(z_value, f"{where}.z"),
    )


def _load(case: str) -> dict[str, object]:
    path = _REPO_ROOT / "blueprint" / f"reference_fem_{case}.json"
    with path.open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), path.name)


def _points(case: str) -> list[dict[str, object]]:
    return _list_of_mappings(_load(case).get("points"), f"{case}.points")


def _complex_pressure(point: dict[str, object]) -> complex:
    pressure = _mapping(point.get("pressure"), "pressure")

    def component(name: str) -> float:
        cell = _mapping(pressure.get(name), f"pressure.{name}")
        raw = cell.get("hex")
        if not isinstance(raw, str):
            raise AssertionError(f"pressure.{name}.hex 不是字串：{raw!r}")
        return float.fromhex(raw)

    return complex(component("real"), component("imag"))


def _physics_inputs() -> check.ModalInputs:
    params = _mapping(_load("rigid").get("parameters"), "parameters")
    room = _mapping(params.get("room_m"), "room_m")
    return check.ModalInputs(
        room=(
            _number(room.get("Lx"), "room_m.Lx"),
            _number(room.get("Ly"), "room_m.Ly"),
            _number(room.get("Lz"), "room_m.Lz"),
        ),
        source=_xyz(params.get("source_xyz_m"), "source_xyz_m"),
        receiver=_xyz(params.get("receiver_xyz_m"), "receiver_xyz_m"),
        sound_speed=_number(params.get("c_m_s"), "c_m_s"),
    )


def _physics_points() -> list[dict[str, object]]:
    return [
        point
        for point in _points("rigid")
        if _number(point.get("frequency_hz"), "frequency_hz")
        <= check.FEM_PHYSICS_FMAX_HZ
    ]


def _physics_violations(
    points: list[dict[str, object]],
    *,
    contract_rel: float,
) -> tuple[list[str], float, str, list[check.ModalSolution]]:
    inputs = _physics_inputs()
    violations: list[str] = []
    usage = 0.0
    usage_at = "沒有考點"
    largest_relative_error = -1.0
    solutions: list[check.ModalSolution] = []
    for point in points:
        frequency = _number(point.get("frequency_hz"), "frequency_hz")
        solution = check.solve_modal_pressure(frequency, inputs)
        solutions.append(solution)
        judgment = check.judge_relative(
            solution.pressure,
            _complex_pressure(point),
            contract_rel,
        )
        if not judgment.within_contract:
            violations.append(
                f"{frequency!r} Hz 相對差 {judgment.relative_error!r} "
                f"> 界線 {contract_rel!r}"
            )
        if judgment.relative_error > largest_relative_error:
            largest_relative_error = judgment.relative_error
            usage_at = f"{point.get('set')} 集合 {frequency!r} Hz"
            if contract_rel > 0.0:
                usage = judgment.relative_error / contract_rel
    return violations, usage, usage_at, solutions


@pytest.mark.parametrize("case", _FEM_CASES)
def test_answer_provenance_matches_card_registry(case: str) -> None:
    """donor 的 tag／commit／clean 三格必須等於規矩卡登記值。"""
    with _PROVENANCE_CARD.open("rb") as handle:
        settings = _mapping(tomllib.load(handle).get("settings"), "settings")
    donor = _mapping(_load(case).get("donor"), f"{case}.donor")
    assert donor == {
        "tag": settings.get("donor_tag"),
        "commit": settings.get("donor_commit"),
        "clean": True,
    }


@pytest.mark.parametrize("case", _FEM_CASES)
def test_point_frequencies_are_unique_and_match_declared_sets(case: str) -> None:
    """points 不重複，set 只認 A／B，並逐點等於檔頭宣告的頻率集合。"""
    answer = _load(case)
    axis = _mapping(answer.get("frequency_axis"), "frequency_axis")
    declared = _mapping(axis.get("sets"), "frequency_axis.sets")
    points = _list_of_mappings(answer.get("points"), "points")
    actual = [
        (str(point.get("set")), _number(point.get("frequency_hz"), "frequency_hz"))
        for point in points
    ]
    expected = [
        (set_name, _number(frequency, f"sets.{set_name}"))
        for set_name, frequencies in declared.items()
        if isinstance(frequencies, list)
        for frequency in frequencies
    ]
    assert set(declared) == _EXPECTED_SETS[case]
    assert len(actual) == len(set(actual)), f"{case} 有重複的 (set, frequency)：{actual!r}"
    assert actual == expected


@pytest.mark.parametrize("case", _FEM_CASES)
def test_nearest_eigenfrequency_fields_use_relative_distance(case: str) -> None:
    """兩欄用 min(|f-f_n|/f_n) 獨立重算；不是先挑絕對 Hz 距離。"""
    inputs = _physics_inputs()
    modes = check.rigid_eigenfrequencies(inputs.room, inputs.sound_speed)
    diffs: list[str] = []
    for point in _points(case):
        frequency = _number(point.get("frequency_hz"), "frequency_hz")
        nearest, relative_distance = check.nearest_eigenfrequency_relative(
            frequency, modes
        )
        if point.get("nearest_eigenfrequency_hz") != nearest:
            diffs.append(f"{case} {frequency!r} Hz 的本徵頻不是 {nearest!r}")
        if point.get("nearest_eigenfrequency_relative_distance") != relative_distance:
            diffs.append(f"{case} {frequency!r} Hz 的相對距離不是 {relative_distance!r}")
    assert diffs == []


@pytest.mark.parametrize("case", _FEM_CASES)
def test_pressure_records_round_trip_and_magnitude_is_exact(case: str) -> None:
    """每格十進位可由 hex 重建，abs.hex 逐位等於 real／imag 的 hypot。"""
    for point in _points(case):
        pressure = _mapping(point.get("pressure"), "pressure")
        values: dict[str, float] = {}
        for name in ("real", "imag", "abs"):
            cell = _mapping(pressure.get(name), f"pressure.{name}")
            hexadecimal = cell.get("hex")
            assert isinstance(hexadecimal, str), f"pressure.{name}.hex 不是字串"
            value = float.fromhex(hexadecimal)
            values[name] = value
            assert cell.get("dec") == repr(value)
        magnitude = _mapping(pressure.get("abs"), "pressure.abs")
        assert magnitude.get("hex") == math.hypot(
            values["real"], values["imag"]
        ).hex()


def test_rigid_duplicate_frequencies_have_bitwise_identical_pressure() -> None:
    """從資料找 A／B 共有頻點；同頻的完整 pressure 記錄必須逐位一致。"""
    by_set = {
        set_name: {
            _number(point.get("frequency_hz"), "frequency_hz"): point.get("pressure")
            for point in _points("rigid")
            if point.get("set") == set_name
        }
        for set_name in ("A", "B")
    }
    shared_frequencies = by_set["A"].keys() & by_set["B"].keys()
    assert shared_frequencies, "rigid 的 A／B 集合沒有共同頻點"
    for frequency in shared_frequencies:
        assert by_set["A"][frequency] == by_set["B"][frequency], (
            f"rigid A／B 在 {frequency!r} Hz 的壓力不一致"
        )


def test_rigid_physics_points_match_modal_solution() -> None:
    """rigid 在物理上限內的全部點對解析模態解不超過契約。"""
    points = _physics_points()
    expected = [
        point
        for point in _points("rigid")
        if _number(point.get("frequency_hz"), "frequency_hz")
        <= check.FEM_PHYSICS_FMAX_HZ
    ]
    assert points == expected, "物理健檢漏掉頻率上限內的考點"
    assert points, "rigid 在物理契約頻率上限內沒有考點"
    violations, usage, usage_at, _solutions = _physics_violations(
        points,
        contract_rel=_FEM_PHYSICS_TOLERANCE_REL,
    )
    assert violations == [], (
        f"最大用到界線 {usage:.6%}，位置 {usage_at}；{violations!r}"
    )


def test_modal_series_converges_for_physics_points() -> None:
    """物理健檢實際用到的每個解析級數都通過原 oracle 的收斂判準。"""
    _violations, _usage, _usage_at, solutions = _physics_violations(
        _physics_points(),
        contract_rel=_FEM_PHYSICS_TOLERANCE_REL,
    )
    assert solutions
    assert all(solution.converged for solution in solutions)


def test_flat_pressures_are_finite_and_nonzero() -> None:
    """flat 每點複數壓力沒有 NaN／Infinity，且大小嚴格為正。"""
    for point in _points("flat"):
        pressure = _complex_pressure(point)
        assert math.isfinite(pressure.real) and math.isfinite(pressure.imag)
        assert abs(pressure) > 0.0


def test_control_scaled_pressure_exceeds_physics_contract() -> None:
    """控制組：任一物理健檢壓力乘 (1+2^-9)，同一裁判必須報超界。"""
    point = copy.deepcopy(_physics_points()[0])
    pressure = _complex_pressure(point) * (1.0 + 2.0 * _FEM_PHYSICS_TOLERANCE_REL)
    raw_pressure = point.get("pressure")
    assert isinstance(raw_pressure, dict), "pressure 不是表"
    for name, value in (("real", pressure.real), ("imag", pressure.imag)):
        component = raw_pressure[name]
        assert isinstance(component, dict), f"pressure.{name} 不是表"
        component["hex"] = value.hex()
    violations, _usage, _usage_at, _solutions = _physics_violations(
        [point],
        contract_rel=_FEM_PHYSICS_TOLERANCE_REL,
    )
    assert violations, "壓力乘 (1+2^-9) 後沒有被物理契約抓到"


def test_control_zero_contract_exposes_existing_numeric_difference() -> None:
    """控制組：把同一條界線換成 0，正常答案與解析解的數值差必須變紅。"""
    violations, _usage, _usage_at, _solutions = _physics_violations(
        _physics_points(),
        contract_rel=0.0,
    )
    assert violations, "界線換成 0 後沒有抓到 FEM 與解析解的數值差"


def test_control_omitting_zero_mode_changes_analytic_solution() -> None:
    """控制組：解析解漏掉全域零模態時，跟完整解析解的差超過物理契約。"""
    point = _physics_points()[0]
    frequency = _number(point.get("frequency_hz"), "frequency_hz")
    inputs = _physics_inputs()
    solution = check.solve_modal_pressure(frequency, inputs)
    without_zero_mode = solution.pressure - check.zero_mode_pressure(frequency, inputs)
    judgment = check.judge_relative(
        solution.pressure,
        without_zero_mode,
        _FEM_PHYSICS_TOLERANCE_REL,
    )
    assert not judgment.within_contract
