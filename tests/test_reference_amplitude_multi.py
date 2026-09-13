"""票 #219 第 2a 段考卷：多接收點與分格材料的凍結振幅答案。"""

from __future__ import annotations

import cmath
import copy
import json
import math
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from blueprint import reference_amplitude_check as check
from blueprint.reference_room_geometry import (
    NUM_AXES,
    Room,
    enumerate_identities,
    expand_bounces,
)

_ROOT = Path(__file__).resolve().parents[1]
_BLUEPRINT = _ROOT / "blueprint"
_RULE = _ROOT / "governance" / "rules" / "answer-files-carry-provenance.toml"
_MULTI_CASES = ("flat", "varied")
_ANSWER_NAMES = (
    "reference_amplitude_multi_flat.json",
    "reference_amplitude_multi_varied.json",
    "reference_amplitude_patch_varied.json",
)
_IDENTITY_LEN = 2 * NUM_AXES


@dataclass(frozen=True)
class _Receiver:
    """一個接收點與它的答案路徑。"""

    receiver_id: str
    inputs: check.Inputs
    paths: list[dict[str, object]]
    totals: dict[str, object]


def _mapping(node: object, where: str) -> dict[str, object]:
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是表：{node!r}")
    return {str(key): value for key, value in node.items()}


def _list(node: object, where: str) -> list[object]:
    if not isinstance(node, list):
        raise AssertionError(f"{where} 不是清單：{node!r}")
    return node


def _tables(node: object, where: str) -> list[dict[str, object]]:
    return [_mapping(item, where) for item in _list(node, where)]


def _number(node: object, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise AssertionError(f"{where} 不是數字：{node!r}")
    return float(node)


def _integer(node: object, where: str) -> int:
    if isinstance(node, bool) or not isinstance(node, int):
        raise AssertionError(f"{where} 不是整數：{node!r}")
    return node


def _numbers(node: object, where: str) -> tuple[float, ...]:
    return tuple(_number(item, where) for item in _list(node, where))


def _answer(name: str) -> dict[str, object]:
    with (_BLUEPRINT / name).open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), name)


def _multi(case: str) -> dict[str, object]:
    return _answer(f"reference_amplitude_multi_{case}.json")


def _identity(path: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    raw = _list(path.get("identity"), "identity")
    if len(raw) != _IDENTITY_LEN:
        raise AssertionError(f"identity 長度不是 {_IDENTITY_LEN}：{raw!r}")
    values = tuple(_integer(item, "identity item") for item in raw)
    return (values[0], values[1], values[2], values[3], values[4], values[5])


def _complex(cell: object) -> complex:
    table = _mapping(cell, "complex cell")
    real = _mapping(table.get("real"), "complex.real").get("dec")
    imag = _mapping(table.get("imag"), "complex.imag").get("dec")
    if not isinstance(real, str) or not isinstance(imag, str):
        raise AssertionError("複數格的 dec 不是字串")
    return complex(float(real), float(imag))


def _real(cell: object) -> float:
    dec = _mapping(cell, "real cell").get("dec")
    if not isinstance(dec, str):
        raise AssertionError("實數格的 dec 不是字串")
    return float(dec)


def _complexes(path: dict[str, object], key: str) -> tuple[complex, ...]:
    return tuple(_complex(cell) for cell in _list(path.get(key), key))


def _xyz(table: dict[str, object]) -> tuple[float, float, float]:
    return (
        _number(table.get("x"), "x"),
        _number(table.get("y"), "y"),
        _number(table.get("z"), "z"),
    )


def _inputs(params: dict[str, object], receiver: tuple[float, float, float]) -> check.Inputs:
    room = _mapping(params.get("room"), "room")
    return check.Inputs(
        lx=_number(room.get("Lx_m"), "Lx_m"),
        ly=_number(room.get("Ly_m"), "Ly_m"),
        lz=_number(room.get("Lz_m"), "Lz_m"),
        c=_number(params.get("sound_speed_m_s"), "sound_speed_m_s"),
        rho_c=_number(params.get("rho_c_pa_s_per_m"), "rho_c"),
        src=_xyz(_mapping(params.get("source_xyz_m"), "source_xyz_m")),
        recv=receiver,
        freqs_hz=_numbers(params.get("frequencies_hz"), "frequencies_hz"),
    )


def _receivers(case: str) -> list[_Receiver]:
    root = _multi(case)
    params = _mapping(root.get("parameters"), "parameters")
    coords = {
        str(item.get("id")): _xyz(item)
        for item in _tables(params.get("receivers_xyz_m"), "receivers_xyz_m")
    }
    records = _mapping(root.get("receivers"), "receivers")
    return [
        _Receiver(
            receiver_id=receiver_id,
            inputs=_inputs(params, coords[receiver_id]),
            paths=_tables(_mapping(record, receiver_id).get("paths"), "paths"),
            totals=_mapping(_mapping(record, receiver_id).get("totals"), "totals"),
        )
        for receiver_id, record in records.items()
    ]


def _impedance_values(table: dict[str, object]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        _numbers(table.get("per_frequency_real"), "per_frequency_real"),
        _numbers(table.get("per_frequency_imag"), "per_frequency_imag"),
    )


def _multi_impedance(case: str, params: dict[str, object]) -> check.WallImpedance:
    cases = _mapping(params.get("cases"), "cases")
    flat = _mapping(_mapping(cases.get("flat"), "flat").get("all_six_walls"), "flat Z")
    flat_real, flat_imag = _impedance_values(flat)
    if case == "flat":
        return lambda _wall, index: complex(flat_real[index], flat_imag[index])
    varied = _mapping(_mapping(cases.get("varied"), "varied").get("y0_wall"), "y0 Z")
    varied_real, varied_imag = _impedance_values(varied)

    def read(wall: str, index: int) -> complex:
        values = (varied_real, varied_imag) if wall == "y0" else (flat_real, flat_imag)
        return complex(values[0][index], values[1][index])

    return read


def _single_varied_impedance(params: dict[str, object]) -> check.WallImpedance:
    cases = _mapping(params.get("cases"), "cases")
    flat_material = _mapping(
        _mapping(cases.get("flat"), "flat").get("material"), "flat material"
    )
    flat = _mapping(flat_material.get("impedance"), "flat impedance")
    flat_value = complex(
        _number(flat.get("real"), "flat real"),
        _number(flat.get("imag"), "flat imag"),
    )
    varied_material = _mapping(
        _mapping(cases.get("varied"), "varied").get("material"), "varied material"
    )
    y0 = _mapping(varied_material.get("y0_wall"), "y0 wall")
    varied_real, varied_imag = _impedance_values(y0)

    def read(wall: str, band: int) -> complex:
        if wall == "y0":
            return complex(varied_real[band], varied_imag[band])
        return flat_value

    return read


def _recompute(receiver: _Receiver, impedance: check.WallImpedance) -> list[check.Recompute]:
    return [
        check.recompute_path(
            receiver.inputs,
            impedance,
            _integer(path.get("index"), "index"),
            _identity(path),
        )
        for path in receiver.paths
    ]


def _fraction(diff: float, bound: float) -> float:
    if bound == 0.0:
        return math.inf if diff != 0.0 else 0.0
    return diff / bound


def _path_violations(receiver: _Receiver, recomputed: list[check.Recompute]) -> tuple[list[str], float]:
    violations: list[str] = []
    maximum = 0.0
    for answer, result in zip(receiver.paths, recomputed, strict=True):
        tau = result.dist_m / receiver.inputs.c
        answer_refl = _complexes(answer, "reflection_product")
        answer_pressure = _complexes(answer, "path_pressure")
        for index, (ref, mine) in enumerate(zip(answer_refl, result.reflection_product, strict=True)):
            bound = check.reflection_tolerance(receiver.inputs.freqs_hz[index], tau, abs(ref))
            for slot, diff in (("real", abs(ref.real - mine.real)), ("imag", abs(ref.imag - mine.imag)), ("abs", abs(abs(ref) - abs(mine)))):
                maximum = max(maximum, _fraction(diff, bound))
                if diff > bound:
                    violations.append(f"{receiver.receiver_id} path {result.index} refl {index}.{slot}: {diff} > {bound}")
        for index, (ref, mine) in enumerate(zip(answer_pressure, result.path_pressure, strict=True)):
            relative = check.pressure_tolerance(receiver.inputs.freqs_hz[index], tau, abs(answer_refl[index]))
            bound = relative * abs(ref)
            diff = abs(ref - mine)
            maximum = max(maximum, _fraction(diff, bound))
            if diff > bound:
                violations.append(f"{receiver.receiver_id} path {result.index} pressure {index}: {diff} > {bound}")
    return violations, maximum


def _total_violations(receiver: _Receiver, recomputed: list[check.Recompute]) -> tuple[list[str], float]:
    violations: list[str] = []
    maximum = 0.0
    total_pressure = tuple(_complex(cell) for cell in _list(receiver.totals.get("pressure"), "pressure"))
    direct_energy = tuple(_real(cell) for cell in _list(receiver.totals.get("ism_direct_E"), "direct E"))
    reverb_energy = tuple(_real(cell) for cell in _list(receiver.totals.get("ism_rev_E"), "reverb E"))
    for band, frequency in enumerate(receiver.inputs.freqs_hz):
        pressures = [result.path_pressure[band] for result in recomputed]
        tolerances = [check.pressure_tolerance(frequency, result.dist_m / receiver.inputs.c, abs(result.reflection_product[band])) for result in recomputed]
        pressure_bound = math.sqrt(math.fsum((tol * abs(value)) ** 2 for tol, value in zip(tolerances, pressures, strict=True)))
        pressure_diff = abs(sum(pressures, 0j) - total_pressure[band])
        direct = [value for value, result in zip(pressures, recomputed, strict=True) if result.order == 0]
        reflected = [value for value, result in zip(pressures, recomputed, strict=True) if result.order > 0]
        direct_diff = abs(abs(sum(direct, 0j)) ** 2 - direct_energy[band])
        direct_bound = check.direct_energy_tolerance() * direct_energy[band]
        reflected_sum = sum(reflected, 0j)
        rss = math.sqrt(math.fsum((tol * abs(value)) ** 2 for tol, value, result in zip(tolerances, pressures, recomputed, strict=True) if result.order > 0))
        relative = 2.0 * rss / abs(reflected_sum) + check.REFLECTION_CONTRACT_ULP / 4.0
        reflected_diff = abs(abs(reflected_sum) ** 2 - reverb_energy[band])
        reflected_bound = relative * reverb_energy[band]
        for kind, diff, bound in (("pressure", pressure_diff, pressure_bound), ("direct E", direct_diff, direct_bound), ("reverb E", reflected_diff, reflected_bound)):
            maximum = max(maximum, _fraction(diff, bound))
            if diff > bound:
                violations.append(f"{receiver.receiver_id} {kind}[{band}]: {diff} > {bound}")
    return violations, maximum


@pytest.mark.parametrize("case", _MULTI_CASES)
def test_multi_r0_is_bit_exact_to_single_receiver(case: str) -> None:
    multi = _mapping(_multi(case).get("receivers"), "receivers")
    single = _answer(f"reference_amplitude_{case}.json")
    r0 = _mapping(multi.get("R0"), "R0")
    assert r0.get("paths") == single.get("paths")
    assert r0.get("totals") == single.get("totals")


@pytest.mark.parametrize("case", _MULTI_CASES)
def test_every_receiver_identity_set_is_independently_enumerated(case: str) -> None:
    params = _mapping(_multi(case).get("parameters"), "parameters")
    expected = enumerate_identities(_integer(params.get("max_order"), "max_order"))
    for receiver in _receivers(case):
        assert {_identity(path) for path in receiver.paths} == expected


@pytest.mark.parametrize("case", _MULTI_CASES)
def test_every_receiver_path_and_totals_are_within_contract(case: str) -> None:
    root = _multi(case)
    impedance = _multi_impedance(case, _mapping(root.get("parameters"), "parameters"))
    errors: list[str] = []
    fractions: list[str] = []
    for receiver in _receivers(case):
        recomputed = _recompute(receiver, impedance)
        path_errors, path_fraction = _path_violations(receiver, recomputed)
        total_errors, total_fraction = _total_violations(receiver, recomputed)
        errors.extend(path_errors + total_errors)
        fractions.append(
            f"{receiver.receiver_id} 路徑 {path_fraction:.2%}／總量 {total_fraction:.2%}"
        )
    assert errors == [], f"超界；{'；'.join(fractions)}：{errors!r}"


def _patch() -> tuple[dict[str, object], list[dict[str, object]]]:
    root = _answer("reference_amplitude_patch_varied.json")
    return root, _tables(root.get("paths"), "paths")


def _bounce_cells(path: dict[str, object]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(cell.get("wall")), _integer(cell.get("row"), "row"), _integer(cell.get("col"), "col"))
        for cell in _tables(path.get("bounce_cells"), "bounce_cells")
    )


def _patch_impedance(params: dict[str, object]) -> check.CellImpedance:
    cases = _mapping(params.get("cases"), "cases")
    patched = _mapping(cases.get("patched_y0_cell_0_0_10rhoc"), "patched case")
    walls = _mapping(patched.get("wall_cells"), "wall_cells")
    values: dict[tuple[str, int, int], tuple[tuple[float, ...], tuple[float, ...]]] = {}
    for wall, raw_cells in walls.items():
        for cell in _tables(raw_cells, wall):
            key = (wall, _integer(cell.get("row"), "row"), _integer(cell.get("col"), "col"))
            values[key] = _impedance_values(_mapping(cell.get("impedance"), "impedance"))

    def read(wall: str, row: int, col: int, band: int) -> complex:
        real, imag = values[(wall, row, col)]
        return complex(real[band], imag[band])

    return read


def _patch_inputs(root: dict[str, object]) -> check.Inputs:
    params = _mapping(root.get("parameters"), "parameters")
    receiver = _mapping(params.get("receiver_xyz_m"), "receiver_xyz_m")
    return _inputs(params, _xyz(receiver))


def _grid_shapes(params: dict[str, object]) -> dict[str, tuple[int, int]]:
    raw_shapes = _list(params.get("grid_shapes"), "grid_shapes")
    walls = check.canonical_walls()
    if len(raw_shapes) != len(walls):
        raise AssertionError("grid_shapes 與 canonical walls 長度不同")
    output: dict[str, tuple[int, int]] = {}
    for wall, raw_shape in zip(walls, raw_shapes, strict=True):
        shape = _list(raw_shape, f"{wall} grid shape")
        try:
            rows, cols = shape
        except ValueError:
            raise AssertionError(f"{wall} grid shape 不是二元組：{shape!r}")
        output[wall] = (
            _integer(rows, f"{wall} rows"),
            _integer(cols, f"{wall} cols"),
        )
    return output


def _geometric_cell(
    inputs: check.Inputs,
    shapes: dict[str, tuple[int, int]],
    wall: str,
    point: tuple[float, float, float],
) -> tuple[str, int, int]:
    axes = {
        "floor": (0, 1),
        "ceiling": (0, 1),
        "x0": (1, 2),
        "xL": (1, 2),
        "y0": (0, 2),
        "yL": (0, 2),
    }
    u_axis, v_axis = axes[wall]
    lengths = (inputs.lx, inputs.ly, inputs.lz)
    rows, cols = shapes[wall]
    col = min(max(int(point[u_axis] / lengths[u_axis] * cols), 0), cols - 1)
    row = min(max(int(point[v_axis] / lengths[v_axis] * rows), 0), rows - 1)
    return wall, row, col


def _patch_reflections(
    root: dict[str, object], paths: list[dict[str, object]], impedance: check.CellImpedance
) -> list[tuple[complex, ...]]:
    inputs = _patch_inputs(root)
    return [
        check.recompute_patch_reflection(inputs, impedance, _identity(path), _bounce_cells(path))
        for path in paths
    ]


def _patch_violations(
    root: dict[str, object], paths: list[dict[str, object]], impedance: check.CellImpedance
) -> tuple[list[str], float]:
    inputs = _patch_inputs(root)
    computed = _patch_reflections(root, paths, impedance)
    violations: list[str] = []
    maximum = 0.0
    for path, result in zip(paths, computed, strict=True):
        tau = _real(path.get("dist_m")) / inputs.c
        for band, (answer, mine) in enumerate(zip(_complexes(path, "reflection_product"), result, strict=True)):
            bound = check.reflection_tolerance(inputs.freqs_hz[band], tau, abs(answer))
            diff = abs(answer - mine)
            maximum = max(maximum, _fraction(diff, bound))
            if diff > bound:
                violations.append(f"patch path {_integer(path.get('index'), 'index')} band {band}: {diff} > {bound}")
    return violations, maximum


def test_patch_bounce_cells_match_order_and_identity_wall_multiset() -> None:
    _root, paths = _patch()
    for path in paths:
        cells = _bounce_cells(path)
        assert len(cells) == check.order_of(_identity(path)) == _integer(path.get("order"), "order")
        expected = Counter(check.wall_count_signature(_identity(path)))
        expected = Counter({wall: count for wall, count in expected.items() if count})
        assert Counter(wall for wall, _row, _col in cells) == expected


def test_patch_bounce_cells_are_independently_derived_from_geometry() -> None:
    root, paths = _patch()
    params = _mapping(root.get("parameters"), "parameters")
    inputs = _patch_inputs(root)
    room = Room(inputs.lx, inputs.ly, inputs.lz, inputs.c)
    shapes = _grid_shapes(params)
    for path in paths:
        bounces = expand_bounces(room, _identity(path), inputs.src, inputs.recv)
        # row-major 約定來自 donor flatten_cell_Z；這一題用幾何把每次落格獨立推一次。
        geometric = tuple(
            _geometric_cell(inputs, shapes, bounce.wall, bounce.point)
            for bounce in bounces
        )
        assert geometric == _bounce_cells(path)


def test_uniform_patch_and_whole_wall_reflection_formulas_agree() -> None:
    root, paths = _patch()
    varied = _answer("reference_amplitude_varied.json")
    params = _mapping(varied.get("parameters"), "parameters")
    wall_impedance = _single_varied_impedance(params)

    def cell_impedance(wall: str, _row: int, _col: int, band: int) -> complex:
        return wall_impedance(wall, band)

    inputs = _patch_inputs(root)
    for path in paths:
        patch_values = check.recompute_patch_reflection(
            inputs, cell_impedance, _identity(path), _bounce_cells(path)
        )
        whole_values = check.recompute_path(
            inputs,
            wall_impedance,
            _integer(path.get("index"), "index"),
            _identity(path),
        ).reflection_product
        for patch_value, whole_value in zip(patch_values, whole_values, strict=True):
            assert abs(patch_value - whole_value) <= check.REFLECTION_CONTRACT_ULP


def test_patch_changes_exactly_paths_hitting_replaced_cell() -> None:
    _root, paths = _patch()
    varied = _answer("reference_amplitude_varied.json")
    varied_by_identity = {_identity(path): path for path in _tables(varied.get("paths"), "paths")}
    hit = {_identity(path) for path in paths if ("y0", 0, 0) in _bounce_cells(path)}
    changed: set[tuple[int, int, int, int, int, int]] = set()
    for path in paths:
        identity = _identity(path)
        patch_values = path.get("reflection_product")
        varied_values = varied_by_identity[identity].get("reflection_product")
        if patch_values != varied_values:
            changed.add(identity)
        else:
            assert identity not in hit
    assert changed == hit


def test_patch_reflection_products_are_within_contract() -> None:
    root, paths = _patch()
    errors, fraction = _patch_violations(root, paths, _patch_impedance(_mapping(root.get("parameters"), "parameters")))
    assert errors == [], f"patch 超界；用到界線 {fraction:.2%}：{errors[:5]!r}"


def test_patch_control_swapped_y0_cells_goes_far_outside_contract() -> None:
    root, paths = _patch()
    original = _patch_impedance(_mapping(root.get("parameters"), "parameters"))

    def swapped(wall: str, row: int, col: int, band: int) -> complex:
        if (wall, row, col) == ("y0", 0, 0):
            return original("y0", 0, 1, band)
        if (wall, row, col) == ("y0", 0, 1):
            return original("y0", 0, 0, band)
        return original(wall, row, col, band)

    errors, fraction = _patch_violations(root, paths, swapped)
    assert errors != []
    assert fraction > 1_000.0


def test_patch_control_small_cell_perturbation_stays_within_contract() -> None:
    root, paths = _patch()
    original = _patch_impedance(_mapping(root.get("parameters"), "parameters"))

    def perturbed(wall: str, row: int, col: int, band: int) -> complex:
        value = original(wall, row, col, band)
        if (wall, row, col) == ("y0", 0, 0):
            return value * (1.0 + 1e-7)
        return value

    errors, fraction = _patch_violations(root, paths, perturbed)
    assert errors == []
    assert fraction <= 1.0


def test_patch_control_zero_boundary_goes_red(monkeypatch: pytest.MonkeyPatch) -> None:
    root, paths = _patch()
    monkeypatch.setattr(check, "reflection_tolerance", lambda _f, _tau, _refl: 0.0)
    errors, _fraction_used = _patch_violations(root, paths, _patch_impedance(_mapping(root.get("parameters"), "parameters")))
    assert errors != []


def test_new_answers_match_registered_donor() -> None:
    with _RULE.open("rb") as handle:
        settings = _mapping(tomllib.load(handle).get("settings"), "settings")
    expected = {"commit": settings.get("donor_commit"), "tag": settings.get("donor_tag")}
    for name in _ANSWER_NAMES:
        donor = _mapping(_answer(name).get("donor"), f"{name}.donor")
        assert {key: donor.get(key) for key in expected} == expected
        assert donor.get("clean") is True


def test_totals_by_receiver_repeat_each_receiver_totals() -> None:
    for case in _MULTI_CASES:
        root = _multi(case)
        receivers = _mapping(root.get("receivers"), "receivers")
        totals = _mapping(root.get("totals_by_receiver"), "totals_by_receiver")
        assert totals == {
            receiver_id: _mapping(record, receiver_id).get("totals")
            for receiver_id, record in receivers.items()
        }


def test_total_contract_control_zero_boundary_goes_red(monkeypatch: pytest.MonkeyPatch) -> None:
    receiver = _receivers("flat")[0]
    params = _mapping(_multi("flat").get("parameters"), "parameters")
    recomputed = _recompute(receiver, _multi_impedance("flat", params))
    monkeypatch.setattr(check, "pressure_tolerance", lambda _f, _tau, _refl: 0.0)
    errors, _fraction_used = _total_violations(receiver, recomputed)
    assert errors != []


def test_patch_cell_control_uses_the_selected_cell() -> None:
    root, paths = _patch()
    path = next(path for path in paths if _bounce_cells(path))
    impedance = _patch_impedance(_mapping(root.get("parameters"), "parameters"))
    calls: list[tuple[str, int, int]] = []

    def recording(wall: str, row: int, col: int, band: int) -> complex:
        calls.append((wall, row, col))
        return impedance(wall, row, col, band)

    check.recompute_patch_reflection(_patch_inputs(root), recording, _identity(path), _bounce_cells(path))
    expected = Counter(_bounce_cells(path))
    frequency_count = len(_patch_inputs(root).freqs_hz)
    assert Counter(calls) == Counter(
        {cell: count * frequency_count for cell, count in expected.items()}
    )


def test_all_varied_patch_control_matches_whole_wall_varied_answer() -> None:
    root, paths = _patch()
    varied = _answer("reference_amplitude_varied.json")
    params = _mapping(varied.get("parameters"), "parameters")
    wall_impedance = _single_varied_impedance(params)
    varied_by_identity = {
        _identity(path): path for path in _tables(varied.get("paths"), "paths")
    }

    def all_varied(wall: str, _row: int, _col: int, band: int) -> complex:
        return wall_impedance(wall, band)

    inputs = _patch_inputs(root)
    errors: list[str] = []
    for path in paths:
        identity = _identity(path)
        computed = check.recompute_patch_reflection(
            inputs, all_varied, identity, _bounce_cells(path)
        )
        answer = _complexes(varied_by_identity[identity], "reflection_product")
        for band, (mine, reference) in enumerate(zip(computed, answer, strict=True)):
            if abs(mine - reference) > check.REFLECTION_CONTRACT_ULP:
                errors.append(
                    f"path {_integer(path.get('index'), 'index')} band {band}: "
                    f"{abs(mine - reference)} > {check.REFLECTION_CONTRACT_ULP}"
                )
    assert errors == [], errors
