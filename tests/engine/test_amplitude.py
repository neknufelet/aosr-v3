"""票 #194 第二段的引擎考卷：v3 自己算反射乘積與路徑壓力，跟兩組振幅答案依契約比。

**這一支是引擎考卷**（住 ``tests/engine/``，規矩卡 ``green-must-be-real-green`` 的引擎籃），
它 import 新家的 ``aosr``（``aosr.physics.room_paths`` 與 ``aosr.physics.amplitude``）。
兩組振幅答案（flat／varied）是唯讀的 ``blueprint/reference_amplitude_{flat,varied}.json``；
考卷可以 import ``blueprint``（``src/`` 不行，規矩卡 layers-import-downward-only）。

**契約（決策紙 docs/decisions/precision-contract-amplitude-phase-scaled.md，選項 1）。**
反射乘積每分量絕對差 ≤ 2^-21；路徑壓力相對差 ≤ 2^-21·(ωτ+1) + 2^-21/|refl|（相對差以複數
差的模除以參考值的模計，``|Δp| ≤ tol·|p|``）；直達反射乘積恰等於 1。距離與到達
時間照 ``precision-contract-geometry-bit-exact`` 逐位元（hex 字串逐字相等）。界線常數與界線
函式**從一個地方來**：``aosr.physics.amplitude`` 是 src 版本的家（錨在決策紙），考卷另外
import ``blueprint.reference_amplitude_check`` 的版本斷言兩邊常數相等（防兩份漂掉）。

**材料換掉答案就變；沒有 materials 就照舊只算幾何。** 同一份輸入把阻抗改一格，反射乘積／
壓力要變；不給 ``materials`` 節，算出的路徑幾何格跟有材料時逐位相同（第三段的行為
不變，第三段會另驗，這裡只驗「有材料沒材料兩條路不互相污染」）。

**控制組（證明裁判真的咬得住）。** ①把 v3 算出的某格推到界線外，餵比對要報差；②把契約界線
函式換 0 要全體超界（紅）。壓力界線另有控制組：壓力界線獨自換 0 必全是 path_pressure 超界、
壓力乘 1+5e-4 必報那一格、壓力相位共軛必紅。載入器三種錯（缺牆、頻帶數對不上、阻抗實部
≤ 0）各要 ValueError，還有非有限、空頻帶、非正頻率、rho_c 訊息。命令列 ``--compare`` 兩組
exit 0、改壞的副本 exit 1、我方沒振幅 exit 1、答案檔沒振幅只比幾何並註明。

**不碰真環境。** 只寫 ``tmp_path``（規矩卡 tests-isolated-from-real-env）。
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from aosr.physics import amplitude as amp
from aosr.physics.amplitude import Materials
from aosr.physics.room_paths import (
    compare_paths,
    image_source_paths,
    load_room_input,
    main,
)
from aosr.physics.room_paths import RoomPath
from blueprint import reference_amplitude_check as check

# 兩組振幅答案檔位置（唯讀）。從這一支往上三層是 repo 根。
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_AMPLITUDE_CASES: tuple[str, ...] = ("flat", "varied")
_WALLS: tuple[str, ...] = ("floor", "ceiling", "x0", "xL", "y0", "yL")


def _amplitude_path(case: str) -> Path:
    return _REPO_ROOT / "blueprint" / f"reference_amplitude_{case}.json"


def _as_mapping(node: object, where: str) -> dict[str, object]:
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是一層表：{node!r}")
    return {str(k): v for k, v in node.items()}


def _as_number(node: object, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise AssertionError(f"{where} 不是數：{node!r}")
    return float(node)


def _answer_params(case: str) -> dict[str, object]:
    with _amplitude_path(case).open(encoding="utf-8") as handle:
        data = json.load(handle)
    root = _as_mapping(data, "answer")
    return _as_mapping(root.get("parameters"), "parameters")


def _answer_paths(case: str) -> list[dict[str, object]]:
    with _amplitude_path(case).open(encoding="utf-8") as handle:
        data = json.load(handle)
    root = _as_mapping(data, "answer")
    raw = root.get("paths")
    if not isinstance(raw, list):
        raise AssertionError("paths 不是一串東西")
    return [_as_mapping(e, "paths[i]") for e in raw]


def _input_case(case: str) -> dict[str, object]:
    """從振幅答案的 ``parameters`` 組出輸入檔形狀（含 ``materials``）。

    ``rho_c_pa_s_per_m`` → ``rho_c``；每面牆一列六個複數阻抗（``{real, imag}``）。flat 六面
    都是 4·ρc；varied 的 ``y0`` 讀 ``cases.varied.material.y0_wall`` 的六個頻帶複數、其餘五面
    維持 4·ρc——全由答案檔讀，不寫死數字。
    """
    params = _answer_params(case)
    rho_c = _as_number(params.get("rho_c_pa_s_per_m"), "rho_c_pa_s_per_m")
    freqs = params.get("frequencies_hz")
    if not isinstance(freqs, list):
        raise AssertionError("frequencies_hz 不是一串東西")
    n_freq = len(freqs)

    z_flat = complex(4.0 * rho_c, 0.0)
    y0_row: list[complex] | None = None
    if case == "varied":
        cases = _as_mapping(params.get("cases"), "cases")
        varied = _as_mapping(cases.get("varied"), "cases.varied")
        mat = _as_mapping(varied.get("material"), "cases.varied.material")
        y0 = _as_mapping(mat.get("y0_wall"), "y0_wall")
        reals = y0.get("per_frequency_real")
        imags = y0.get("per_frequency_imag")
        if not isinstance(reals, list) or not isinstance(imags, list):
            raise AssertionError("y0 的 per_frequency 不是一串東西")
        y0_row = [
            complex(_as_number(reals[k], "real"), _as_number(imags[k], "imag"))
            for k in range(n_freq)
        ]

    mats: dict[str, object] = {"rho_c": rho_c, "frequencies_hz": freqs}
    for wall in _WALLS:
        if wall == "y0" and y0_row is not None:
            mats[wall] = [{"real": z.real, "imag": z.imag} for z in y0_row]
        else:
            mats[wall] = [{"real": z_flat.real, "imag": z_flat.imag} for _ in range(n_freq)]

    return {
        "room": params["room"],
        "source_xyz_m": params["source_xyz_m"],
        "receiver_xyz_m": params["receiver_xyz_m"],
        "sound_speed_m_s": params["sound_speed_m_s"],
        "max_order": params["max_order"],
        "materials": mats,
    }


def _write_input(tmp_path: Path, case: str, **overrides: object) -> Path:
    params = _input_case(case)
    for key, value in overrides.items():
        params[key] = value
    path = tmp_path / f"room-{case}.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    return path


def _v3_paths(tmp_path: Path, case: str, **overrides: object) -> list[RoomPath]:
    inp = load_room_input(_write_input(tmp_path, case, **overrides))
    assert inp.materials is not None
    return image_source_paths(
        inp.room, inp.source, inp.receiver, inp.sound_speed, inp.max_order, inp.materials
    )


def _answer_complex(cell: object, where: str) -> complex:
    table = _as_mapping(cell, where)

    def _dec(slot: str) -> float:
        sub = _as_mapping(table.get(slot), f"{where}.{slot}")
        dec = sub.get("dec")
        if isinstance(dec, str):
            return float(dec)
        return _as_number(dec, f"{where}.{slot}.dec")

    return complex(_dec("real"), _dec("imag"))


def _answer_complex_list(node: object, where: str) -> list[complex]:
    if not isinstance(node, list):
        raise AssertionError(f"{where} 不是一串東西")
    return [_answer_complex(e, f"{where}[{idx}]") for idx, e in enumerate(node)]


def _answer_hex(cell: object, where: str) -> str:
    sub = _as_mapping(cell, where)
    hexed = sub.get("hex")
    if not isinstance(hexed, str):
        raise AssertionError(f"{where}.hex 不是字串")
    return hexed


# ── 兩組參考答案：每格在契約內（記最大用到幾成）；直達恰 1；距離／到達時間逐位 ─────────


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_amplitude_within_contract(tmp_path: Path, case: str) -> None:
    """v3 算出的反射乘積與壓力每格落在契約界線內；失敗訊息帶「用到界線幾成」最大值。"""
    paths = _v3_paths(tmp_path, case)
    answers = _answer_paths(case)
    assert len(paths) == len(answers)

    freq = tuple(_as_float_list(_answer_params(case)["frequencies_hz"], "freqs"))

    def _frac(diff: float, tol: float) -> float:
        if tol == 0.0:
            return float("inf") if diff != 0.0 else 0.0
        return diff / tol

    max_refl_frac = 0.0
    max_pp_frac = 0.0
    violations: list[str] = []
    for p, ans in zip(paths, answers):
        tau = p.delay_s
        their_refl = _answer_complex_list(ans.get("reflection_product"), "reflection_product")
        their_pp = _answer_complex_list(ans.get("path_pressure"), "path_pressure")
        for f_idx in range(len(freq)):
            refl_a = their_refl[f_idx]
            refl_m = p.reflection_product[f_idx]
            tol = amp.reflection_tolerance(freq[f_idx], tau, abs(refl_a))
            for slot, a, b in (
                ("real", refl_a.real, refl_m.real),
                ("imag", refl_a.imag, refl_m.imag),
                ("abs", abs(refl_a), abs(refl_m)),
            ):
                diff = abs(a - b)
                if diff > tol:
                    violations.append(
                        f"{case} {p.identity!r} reflection_product[{f_idx}].{slot} 差 {diff!r} > {tol!r}"
                    )
                max_refl_frac = max(max_refl_frac, _frac(diff, tol))

            pp_a = their_pp[f_idx]
            pp_m = p.path_pressure[f_idx]
            tol_rel = amp.pressure_tolerance(freq[f_idx], tau, abs(refl_a))
            abs_tol = tol_rel * abs(pp_a)
            diff = abs(pp_m - pp_a)
            if diff > abs_tol:
                violations.append(
                    f"{case} {p.identity!r} path_pressure[{f_idx}] 複數差模 {diff!r} > {abs_tol!r}"
                )
            max_pp_frac = max(max_pp_frac, _frac(diff, abs_tol))

    assert violations == [], (
        f"超出契約界線的格有 {len(violations)} 個；反射乘積用到界線 {max_refl_frac:.6f}、"
        f"壓力 {max_pp_frac:.6f}。前幾筆：{violations[:5]!r}"
    )


def _as_float_list(node: object, where: str) -> list[float]:
    if not isinstance(node, list):
        raise AssertionError(f"{where} 不是一串東西")
    return [_as_number(item, f"{where}[{idx}]") for idx, item in enumerate(node)]


# 直達路徑的 identity（全 0、全 +1），跟答案檔/獨立幾何一致。
_DIRECT_IDENTITY: tuple[int, int, int, int, int, int] = (0, 1, 0, 1, 0, 1)


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_direct_reflection_product_exactly_one(tmp_path: Path, case: str) -> None:
    """直達路徑（order 0）的反射乘積六個頻帶都恰等於 1+0j，不設容差。"""
    paths = _v3_paths(tmp_path, case)
    direct = [p for p in paths if p.identity == _DIRECT_IDENTITY]
    assert direct, "找不到直達路徑（identity 全 0 全 +1）"
    direct_path = direct[0]
    assert direct_path.order == 0
    for f_idx, z in enumerate(direct_path.reflection_product):
        assert z == complex(1.0, 0.0), f"{case} 直達 reflection_product[{f_idx}] 不是 1+0j"


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_dist_and_delay_hex_bit_exact(tmp_path: Path, case: str) -> None:
    """距離與到達時間的 hex 跟振幅答案檔逐字相等（幾何那份 precision-contract）。"""
    paths = _v3_paths(tmp_path, case)
    answers = _answer_paths(case)
    assert len(paths) == len(answers)
    for p, ans in zip(paths, answers):
        assert p.dist_m.hex() == _answer_hex(ans.get("dist_m"), "dist_m"), f"{p.identity!r} dist hex"
        assert p.delay_s.hex() == _answer_hex(ans.get("delay_s"), "delay_s"), f"{p.identity!r} delay hex"


# ── 材料換掉答案就變；沒有 materials 照舊只算幾何 ────────────────────────────────


def test_material_change_changes_amplitude(tmp_path: Path) -> None:
    """把一面牆的阻抗改一格，至少一條反射乘積／壓力的分量要變。"""
    baseline = _v3_paths(tmp_path, "flat")
    inp = load_room_input(_write_input(tmp_path, "flat"))
    assert inp.materials is not None
    new_walls = dict(inp.materials.walls)
    row = list(new_walls["floor"])
    row[0] = row[0] + complex(100.0, 0.0)
    new_walls["floor"] = tuple(row)
    changed_materials = Materials(
        rho_c=inp.materials.rho_c,
        frequencies_hz=inp.materials.frequencies_hz,
        walls=new_walls,
    )
    changed = image_source_paths(
        inp.room, inp.source, inp.receiver, inp.sound_speed, inp.max_order, changed_materials
    )
    differs = [
        (p.identity, f_idx)
        for p, q in zip(changed, baseline)
        for f_idx in range(len(p.reflection_product))
        if p.reflection_product[f_idx] != q.reflection_product[f_idx]
        or p.path_pressure[f_idx] != q.path_pressure[f_idx]
    ]
    assert differs, "把一面牆阻抗改一格之後振幅完全沒變"


def test_no_materials_keeps_geometry_output_identical(tmp_path: Path) -> None:
    """沒有 ``materials`` 節時，算出的路徑幾何格（距離、延遲、鏡像）跟有材料時逐位相同，只差
    振幅欄為空——幾何那條路沒被材料污染（``--json`` 的逐字不變由第三段另驗）。"""
    inp = load_room_input(_write_input(tmp_path, "flat"))
    assert inp.materials is not None
    with_materials = image_source_paths(
        inp.room, inp.source, inp.receiver, inp.sound_speed, inp.max_order, inp.materials
    )
    without = image_source_paths(
        inp.room, inp.source, inp.receiver, inp.sound_speed, inp.max_order
    )
    assert len(with_materials) == len(without)
    for p, q in zip(with_materials, without):
        assert p.dist_m.hex() == q.dist_m.hex()
        assert p.delay_s.hex() == q.delay_s.hex()
        assert p.image == q.image
        assert q.reflection_product == ()
        assert q.path_pressure == ()


# ── 契約控制組 ──────────────────────────────────────────────────────────────


def test_control_group_tampered_cell_reports_out_of_contract(tmp_path: Path) -> None:
    """把 v3 某格推到界線外，比對要報差（裁判咬得住）。"""
    paths = _v3_paths(tmp_path, "flat")
    answers = _answer_paths("flat")
    freq = tuple(_as_float_list(_answer_params("flat")["frequencies_hz"], "freqs"))

    victim = next(p for p in paths if p.order >= 1)
    tampered = replace(
        victim,
        reflection_product=(
            victim.reflection_product[0] + complex(2.0 * amp.REFLECTION_CONTRACT_ULP, 0.0),
        )
        + victim.reflection_product[1:],
    )
    tampered_list = [tampered if p is victim else p for p in paths]

    results = compare_paths(tampered_list, answers, freq)
    assert any(r.diffs for r in results), "掰到界線外之後比對沒報差"


def test_control_group_zero_bound_goes_red(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """把契約界線函式換 0，同一批振幅必全體超界（界線真的被吃進去）。"""
    paths = _v3_paths(tmp_path, "flat")
    answers = _answer_paths("flat")
    freq = tuple(_as_float_list(_answer_params("flat")["frequencies_hz"], "freqs"))

    import aosr.physics.compare as cmp

    monkeypatch.setattr(cmp, "reflection_tolerance", lambda f, tau, ar: 0.0)
    monkeypatch.setattr(cmp, "pressure_tolerance", lambda f, tau, ar: 0.0)
    results = compare_paths(paths, answers, freq)
    assert any(r.diffs for r in results), "界線換 0 之後居然沒有超界（界線沒被吃進去）"


def test_control_group_pressure_bound_zero_goes_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壓力界線換 0（反射不動）→ 違規非空且全是 path_pressure（壓力界線真的被吃）。"""
    import aosr.physics.compare as cmp

    paths = _v3_paths(tmp_path, "flat")
    answers = _answer_paths("flat")
    freq = tuple(_as_float_list(_answer_params("flat")["frequencies_hz"], "freqs"))
    monkeypatch.setattr(cmp, "pressure_tolerance", lambda f, tau, ar: 0.0)
    results = compare_paths(paths, answers, freq)
    diffs = [d for r in results for d in r.diffs]
    assert diffs, "壓力界線換 0 之後居然沒有超界"
    assert all("path_pressure" in d for d in diffs), (
        f"壓力界線換 0 之後違規不全是 path_pressure：{diffs[:5]!r}"
    )
    assert not any("reflection_product" in d for d in diffs), (
        f"壓力界線換 0 不該動到 reflection_product：{diffs[:5]!r}"
    )


def _set_complex_cell(cell: object, z: complex) -> None:
    """把答案檔一個複數格的實／虛 dec 重寫成 ``z``（abs 那格不碰）。就地改、不回新表。"""
    table = _as_mapping(cell, "cell")
    for slot, value in (("real", z.real), ("imag", z.imag)):
        sub = table[slot]
        assert isinstance(sub, dict)
        sub["dec"] = repr(value)


def _tampered_answers(mutate_pressure: Callable[[complex], complex]) -> list[dict[str, object]]:
    """把答案檔每條路徑每個頻帶的壓力都改成 ``mutate_pressure(z)``，回掰過的路徑清單。"""
    answers = copy.deepcopy(_answer_paths("flat"))
    for ans in answers:
        pp = ans.get("path_pressure")
        if not isinstance(pp, list):
            continue
        for cell in pp:
            _set_complex_cell(cell, mutate_pressure(_answer_complex(cell, "cell")))
    return answers


def test_control_group_pressure_small_error_reported(tmp_path: Path) -> None:
    """把 125 Hz 那格壓力乘 (1+5e-4)，判契約函式必報那一格。"""
    freq = tuple(_as_float_list(_answer_params("flat")["frequencies_hz"], "freqs"))
    f_125 = min((i for i, f in enumerate(freq) if f == 125.0), default=None)
    assert f_125 is not None, "答案檔沒有 125 Hz 的頻帶"

    answers = copy.deepcopy(_answer_paths("flat"))
    pp = answers[0]["path_pressure"]
    if not isinstance(pp, list):
        raise AssertionError("path_pressure 不是一串東西")
    cell = pp[f_125]
    z = _answer_complex(cell, "cell")
    _set_complex_cell(cell, z * (1.0 + 5e-4))

    paths = _v3_paths(tmp_path, "flat")
    results = compare_paths(paths, answers, freq)
    diffs = [d for r in results for d in r.diffs]
    assert any(f"path_pressure[{f_125}]" in d for d in diffs), (
        f"壓力乘 1+5e-4 之後 125 Hz 那格沒被報出：{diffs[:5]!r}"
    )


def test_control_group_pressure_conjugate_goes_red(tmp_path: Path) -> None:
    """正常界線下把全部壓力相位共軛（只動壓力、反射不動）→ 必紅（壓力界線有意義）。"""
    freq = tuple(_as_float_list(_answer_params("flat")["frequencies_hz"], "freqs"))
    answers = _tampered_answers(lambda z: complex(z.real, -z.imag))
    paths = _v3_paths(tmp_path, "flat")
    results = compare_paths(paths, answers, freq)
    diffs = [d for r in results for d in r.diffs]
    assert diffs, "壓力相位共軛之後居然沒有超界"
    assert all("path_pressure" in d for d in diffs), (
        f"共軛只動壓力，違規卻不全是 path_pressure：{diffs[:5]!r}"
    )


# ── 載入器三種錯 ──────────────────────────────────────────────────────────────


def test_loader_rejects_missing_wall(tmp_path: Path) -> None:
    """缺一面牆要 ValueError，訊息指名哪一面。"""
    params = _input_case("flat")
    mats = _as_mapping(params["materials"], "materials")
    mats.pop("xL")
    params["materials"] = mats
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="xL"):
        load_room_input(path)


def test_loader_rejects_wrong_band_count(tmp_path: Path) -> None:
    """一面牆頻帶數對不上要 ValueError，訊息指名哪一面。"""
    params = _input_case("flat")
    mats = _as_mapping(params["materials"], "materials")
    row = mats["yL"]
    if isinstance(row, list):
        mats["yL"] = row[:-1]
    params["materials"] = mats
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="yL"):
        load_room_input(path)


@pytest.mark.parametrize("bad_real", [0.0, -5.0])
def test_loader_rejects_nonpositive_real_impedance(tmp_path: Path, bad_real: float) -> None:
    """阻抗實部 ≤ 0 要 ValueError（上一代的公式在這裡沒定義）。"""
    params = _input_case("flat")
    mats = _as_mapping(params["materials"], "materials")
    row = mats["ceiling"]
    if isinstance(row, list):
        first = _as_mapping(row[0], "row[0]")
        first["real"] = bad_real
        row[0] = first
        mats["ceiling"] = row
    params["materials"] = mats
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="ceiling"):
        load_room_input(path)


# ── 命令列 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_cli_compare_exit_zero(tmp_path: Path, case: str) -> None:
    """--compare 對兩組振幅答案都 exit 0。"""
    input_path = _write_input(tmp_path, case)
    exit_code = main([str(input_path), "--compare", str(_amplitude_path(case))])
    assert exit_code == 0


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_cli_compare_tampered_exit_one(tmp_path: Path, case: str) -> None:
    """--compare 對一份改壞振幅格的答案副本 exit 1。"""
    root = None
    with _amplitude_path(case).open(encoding="utf-8") as handle:
        root = json.load(handle)
    paths = root["paths"]
    if isinstance(paths, list) and paths:
        first = paths[0]
        if isinstance(first, dict):
            refl = first["reflection_product"]
            if isinstance(refl, list) and refl:
                real_dec = float(refl[0]["real"]["dec"])
                refl[0]["real"]["dec"] = repr(real_dec + 2.0 * amp.REFLECTION_CONTRACT_ULP)
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(root, sort_keys=True), encoding="utf-8")
    input_path = _write_input(tmp_path, case)
    exit_code = main([str(input_path), "--compare", str(tampered_path)])
    assert exit_code == 1


# ── 契約常數兩邊相等（防 src 跟獨立檢查漂掉）──────────────────────────────────


def test_contract_constant_matches_reference_check() -> None:
    """src 的契約常數等於 blueprint/reference_amplitude_check 的同名常數。"""
    from blueprint import reference_amplitude_check as check

    assert amp.REFLECTION_CONTRACT_ULP == check.REFLECTION_CONTRACT_ULP


def test_tolerance_functions_match_reference_check() -> None:
    """src 跟獨立檢查的界線函式對同一組 (f, τ, |refl|) 回同一個數。"""
    from blueprint import reference_amplitude_check as check

    for f, tau in ((125.0, 0.01), (4000.0, 0.05), (1000.0, 0.02)):
        for abs_refl in (1.0, 0.5, 0.2):
            assert amp.reflection_tolerance(f, tau, abs_refl) == check.reflection_tolerance(f, tau, abs_refl)
            assert amp.pressure_tolerance(f, tau, abs_refl) == check.pressure_tolerance(f, tau, abs_refl)


# ── --compare 的「我方／答案檔有沒有振幅」兩條路 ─────────────────────────────────


def _write_input_without_materials(tmp_path: Path) -> Path:
    """一份沒有 ``materials`` 節的輸入（照舊只算幾何）。"""
    params = _input_case("flat")
    params.pop("materials", None)
    path = tmp_path / "room-nomat.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    return path


def test_cli_compare_ours_no_amplitude_reports(tmp_path: Path) -> None:
    """答案檔有振幅欄、我方沒有（沒 materials）→ 每條報「我方沒有振幅可比」、exit 1。"""
    input_path = _write_input_without_materials(tmp_path)
    exit_code = main([str(input_path), "--compare", str(_amplitude_path("flat"))])
    assert exit_code == 1


def test_cli_compare_answer_no_amplitude_only_geometry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """答案檔沒振幅欄（第三段幾何檔）、我方有 → 只比幾何、判決行註明「答案檔無振幅，未比」、exit 0。"""
    geom = _REPO_ROOT / "blueprint" / "reference_room_answers_order3.json"
    input_path = _write_input(tmp_path, "flat")
    exit_code = main([str(input_path), "--compare", str(geom)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "答案檔無振幅，未比" in out


# ── 三份 wall_count_signature 與 check↔src 數字逐位 ────────────────────────────


def test_three_wall_count_signatures_match_all_identities() -> None:
    """src、reference_room_geometry、reference_amplitude_check 三份 wall_count_signature 對
    63 個 identity（order 3）逐個相等。"""
    from aosr.geometry.shoebox import wall_count_signature as src_sig
    from aosr.geometry.shoebox import enumerate_identities as src_enum
    from blueprint import reference_amplitude_check as check
    from blueprint.reference_room_geometry import wall_count_signature as geom_sig
    from blueprint.reference_room_geometry import enumerate_identities as _blueprint_enumerate_identities

    identities = src_enum(3)
    assert set(identities) == set(
        _blueprint_enumerate_identities(3)
    ), "src 的 identity 枚舉跟獨立幾何的集合不同"
    for ident in identities:
        src = src_sig(ident)
        geom = geom_sig(ident)
        chk = check.wall_count_signature(ident)
        assert src == geom, f"identity {ident!r}: src 跟獨立幾何的 wall_count_signature 不同"
        assert src == chk, f"identity {ident!r}: src 跟獨立檢查的 wall_count_signature 不同"


def _check_inputs_and_compute(
    case: str,
) -> tuple[check.Inputs, check.WallImpedance, Materials]:
    """把振幅答案的 parameters 搭成獨立檢查的 :class:`check.Inputs` 與 :class:`check.WallImpedance`
    的饋料、以及 v3 的 :class:`Materials`，回三者（供逐位題共用）。"""
    params = _answer_params(case)
    room = _as_mapping(params.get("room"), "room")
    freqs = _as_float_list(params["frequencies_hz"], "frequencies_hz")
    src = _as_mapping(params["source_xyz_m"], "source_xyz_m")
    recv = _as_mapping(params["receiver_xyz_m"], "receiver_xyz_m")

    inputs = check.Inputs(
        lx=_as_number(room.get("Lx_m"), "Lx_m"),
        ly=_as_number(room.get("Ly_m"), "Ly_m"),
        lz=_as_number(room.get("Lz_m"), "Lz_m"),
        c=_as_number(params.get("sound_speed_m_s"), "sound_speed_m_s"),
        rho_c=_as_number(params.get("rho_c_pa_s_per_m"), "rho_c_pa_s_per_m"),
        src=(_as_number(src["x"], "x"), _as_number(src["y"], "y"), _as_number(src["z"], "z")),
        recv=(_as_number(recv["x"], "x"), _as_number(recv["y"], "y"), _as_number(recv["z"], "z")),
        freqs_hz=tuple(freqs),
    )

    rho_c = inputs.rho_c
    z_const = complex(4.0 * rho_c, 0.0)
    y0_row: list[complex] | None = None
    if case == "varied":
        cases = _as_mapping(params.get("cases"), "cases")
        varied = _as_mapping(cases.get("varied"), "cases.varied")
        mat = _as_mapping(varied.get("material"), "cases.varied.material")
        y0 = _as_mapping(mat.get("y0_wall"), "y0_wall")
        reals = y0.get("per_frequency_real")
        imags = y0.get("per_frequency_imag")
        if not isinstance(reals, list) or not isinstance(imags, list):
            raise AssertionError("y0 的 per_frequency 不是一串東西")
        y0_row = [
            complex(_as_number(reals[k], "real"), _as_number(imags[k], "imag"))
            for k in range(len(freqs))
        ]

    def impedance(wall: str, f_idx: int) -> complex:
        if wall == "y0" and y0_row is not None:
            return y0_row[f_idx]
        return z_const

    walls: dict[str, tuple[complex, ...]] = {}
    for wall in ("floor", "ceiling", "x0", "xL", "y0", "yL"):
        if wall == "y0" and y0_row is not None:
            walls[wall] = tuple(y0_row)
        else:
            walls[wall] = tuple(z_const for _ in freqs)
    materials = Materials(rho_c=rho_c, frequencies_hz=tuple(freqs), walls=walls)

    return inputs, impedance, materials


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_check_and_src_amplitude_bit_exact(case: str) -> None:
    """獨立檢查的 reflection_coefficient／incidence_cos／recompute_path 跟 src 算的數字逐位
    相同（雙精度對雙精度）。"""
    from blueprint import reference_amplitude_check as check

    inputs, impedance, materials = _check_inputs_and_compute(case)

    # 反射係數、入射 cos：同一批參數逐位相同。
    for Z, cos in ((complex(1646.4, 0.0), 0.5), (complex(823.2, 123.48), 0.3), (complex(1.0, 0.0), 1.0)):
        assert check.reflection_coefficient(Z, cos, inputs.rho_c) == amp.reflection_coefficient(
            Z, cos, inputs.rho_c
        )
    for axis, dist_m, recv, image in (
        (0, 2.5, inputs.recv, inputs.src),
        (1, 3.7, inputs.recv, inputs.src),
        (2, 1.9, inputs.recv, inputs.src),
    ):
        assert check.incidence_cos(axis, dist_m, recv, image) == amp.incidence_cos(
            axis, dist_m, recv, image
        )

    # 每條路徑：check.recompute_path 的反射乘積／壓力 跟 src.path_amplitude 逐位相同。
    from aosr.geometry.shoebox import enumerate_identities, image_from_identity, distance
    from aosr.geometry.shoebox import Point, Room

    room = Room(Lx=inputs.lx, Ly=inputs.ly, Lz=inputs.lz)
    src_p = Point(*inputs.src)
    recv_p = Point(*inputs.recv)
    for index, ident in enumerate(enumerate_identities(3)):
        recomputed = check.recompute_path(inputs, impedance, index, ident)
        image_pt = image_from_identity(room, ident, src_p)
        dist = distance(image_pt, recv_p)
        refl, pp = amp.path_amplitude(
            materials, ident, dist, inputs.c, recv_p.as_tuple(), image_pt.as_tuple()
        )
        diffs = check.compare_cells(recomputed.reflection_product, refl, f"id {ident!r}", "reflection_product")
        diffs += check.compare_cells(recomputed.path_pressure, pp, f"id {ident!r}", "path_pressure")
        assert diffs == [], f"{case} identity {ident!r} 逐位不同：{diffs[:3]!r}"


# ── 載入器：非有限、空頻帶、rho_c 訊息 ─────────────────────────────────────────


@pytest.mark.parametrize("bad", ["inf", "nan"], ids=["inf-imag", "nan-imag"])
def test_loader_rejects_nonfinite_impedance(tmp_path: Path, bad: str) -> None:
    """阻抗實／虛部非有限（inf／nan）→ ValueError。"""
    params = _input_case("flat")
    mats = _as_mapping(params["materials"], "materials")
    row = mats["floor"]
    if isinstance(row, list):
        first = _as_mapping(row[0], "row[0]")
        first["imag"] = float(bad)
        row[0] = first
        mats["floor"] = row
    params["materials"] = mats
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="floor"):
        load_room_input(path)


def test_loader_rejects_empty_frequencies(tmp_path: Path) -> None:
    """frequencies_hz 空 → ValueError。"""
    params = _input_case("flat")
    mats = _as_mapping(params["materials"], "materials")
    mats["frequencies_hz"] = []
    for wall in _WALLS:
        mats[wall] = []
    params["materials"] = mats
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="空的"):
        load_room_input(path)


def test_loader_rejects_nonpositive_frequency(tmp_path: Path) -> None:
    """frequencies_hz 裡有 ≤ 0 → ValueError。"""
    params = _input_case("flat")
    mats = _as_mapping(params["materials"], "materials")
    freqs_raw = mats["frequencies_hz"]
    if isinstance(freqs_raw, list):
        freqs = list(freqs_raw)
        freqs[0] = 0.0
        mats["frequencies_hz"] = freqs
    params["materials"] = mats
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="frequencies"):
        load_room_input(path)


def test_loader_rho_c_error_message_is_about_rho_c(tmp_path: Path) -> None:
    """rho_c ≤ 0 的錯誤訊息要講 rho_c，不是借房間三邊的尾巴。"""
    params = _input_case("flat")
    mats = _as_mapping(params["materials"], "materials")
    mats["rho_c"] = 0.0
    params["materials"] = mats
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="rho_c"):
        load_room_input(path)


def test_path_pressure_rejects_zero_distance() -> None:
    """path_pressure 距離為 0 → ValueError 說接收點落在鏡像上（不准 ZeroDivisionError）。"""
    materials = Materials(
        rho_c=411.6,
        frequencies_hz=(125.0,),
        walls={w: (complex(1646.4, 0.0),) for w in _WALLS},
    )
    with pytest.raises(ValueError, match="鏡像"):
        amp.path_pressure(materials, 0.0, 343.0, (complex(1.0, 0.0),))
