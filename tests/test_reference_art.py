"""票 #215 第 2 段考卷：獨立核對三份上一代 ART 晚期混響能量答案。

這支考卷住治理籃且不 import ``aosr``。答案的吸收率是 donor 以 float32 算出的；考卷用
``blueprint.reference_art_check`` 的標準庫實作重算 ``alpha = 1 - |R|**2``，允許相鄰一個
float32 格。Sabine 比對描述上一代均勻材料的數值性質，不是 v3 精確解的契約。
"""
from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import pytest

from blueprint import reference_art_check as check

_ROOT = Path(__file__).resolve().parents[1]
_CASES = ("flat", "varied", "lowabs")
_UNIFORM_CASES = ("flat", "lowabs")
_PROVENANCE_CARD = _ROOT / "governance" / "rules" / "answer-files-carry-provenance.toml"


def _answer_path(case: str) -> Path:
    return _ROOT / "blueprint" / f"reference_art_{case}.json"


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{where} 不是一層表：{value!r}")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError(f"{where} 不是一串值：{value!r}")
    return value


def _real(value: object, where: str) -> float:
    cell = _mapping(value, where)
    hexadecimal = cell.get("hex")
    if not isinstance(hexadecimal, str):
        raise AssertionError(f"{where}.hex 不是字串：{hexadecimal!r}")
    return float.fromhex(hexadecimal)


def _integer(value: object, where: str) -> int:
    cell = _mapping(value, where)
    hexadecimal = cell.get("hex")
    if not isinstance(hexadecimal, str):
        raise AssertionError(f"{where}.hex 不是字串：{hexadecimal!r}")
    return int(hexadecimal, 16)


def _load(case: str) -> dict[str, object]:
    with _answer_path(case).open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), case)


def _bands(answer: dict[str, object]) -> list[dict[str, object]]:
    return [_mapping(item, "band") for item in _sequence(answer.get("bands"), "bands")]


def _impedance(parameters: dict[str, object], wall: str, band_index: int) -> complex:
    material = _mapping(parameters.get("material"), "parameters.material")
    by_wall = _mapping(material.get("impedance_by_wall"), "material.impedance_by_wall")
    values = _sequence(by_wall.get(wall), f"impedance_by_wall.{wall}")
    cell = _mapping(values[band_index], f"impedance_by_wall.{wall}[{band_index}]")
    return complex(_real(cell.get("real"), "Z.real"), _real(cell.get("imag"), "Z.imag"))


def _room(parameters: dict[str, object]) -> check.Room:
    room = _mapping(parameters.get("room_m"), "parameters.room_m")
    return check.Room(
        lx=_real(room.get("Lx"), "room.Lx"),
        ly=_real(room.get("Ly"), "room.Ly"),
        lz=_real(room.get("Lz"), "room.Lz"),
    )


@pytest.mark.parametrize("case", _CASES)
def test_provenance_matches_registered_card(case: str) -> None:
    """donor 的 tag、commit、clean 三格等於規矩卡登記的出身。"""
    answer = _load(case)
    card = tomllib.loads(_PROVENANCE_CARD.read_text(encoding="utf-8"))
    settings = _mapping(card.get("settings"), "card.settings")
    assert answer.get("donor") == {
        "tag": settings.get("donor_tag"),
        "commit": settings.get("donor_commit"),
        "clean": True,
    }


@pytest.mark.parametrize("case", _CASES)
def test_alpha_by_wall_matches_independent_normal_incidence_formula(case: str) -> None:
    """逐面逐頻帶以獨立 float32 算式核對 α；答案至多差一個 float32 格。"""
    answer = _load(case)
    parameters = _mapping(answer.get("parameters"), "parameters")
    rho_c = _real(parameters.get("rho_c_pa_s_per_m"), "rho_c")
    for band_index, band in enumerate(_bands(answer)):
        alpha_by_wall = _mapping(band.get("alpha_by_wall"), "band.alpha_by_wall")
        for wall, cell in alpha_by_wall.items():
            observed = _real(cell, f"alpha_by_wall.{wall}")
            expected = check.normal_incidence_absorption_float32(
                _impedance(parameters, wall, band_index), rho_c
            )
            assert check.float32_cells_apart(observed, expected) <= 1


@pytest.mark.parametrize("case", _CASES)
def test_alpha_bar_matches_independent_area_weighting(case: str) -> None:
    """每頻帶 alpha_bar 等於六面牆按實際房間面積加權的獨立結果。"""
    answer = _load(case)
    parameters = _mapping(answer.get("parameters"), "parameters")
    room = _room(parameters)
    for band in _bands(answer):
        alpha_cells = _mapping(band.get("alpha_by_wall"), "band.alpha_by_wall")
        alpha = {wall: _real(cell, f"alpha_by_wall.{wall}") for wall, cell in alpha_cells.items()}
        assert _real(band.get("alpha_bar"), "alpha_bar") == check.area_weighted_alpha(room, alpha)


@pytest.mark.parametrize("case", _CASES)
def test_domain_flag_is_derived_from_answer_limit(case: str) -> None:
    """in_domain 恰等於 alpha_bar 不超過答案檔所載的 donor 適用域上限。"""
    answer = _load(case)
    parameters = _mapping(answer.get("parameters"), "parameters")
    limit = _mapping(parameters.get("ART_RADIOSITY_ALPHA_BAR_MAX"), "domain limit")
    maximum = _real(limit, "domain limit")
    for band in _bands(answer):
        assert band.get("in_domain") is (_real(band.get("alpha_bar"), "alpha_bar") <= maximum)


def test_reference_cases_cover_both_sides_of_domain_from_data() -> None:
    """資料自己判出 flat／varied 域外、lowabs 域內，不以 case 名直接寫死旗標。"""
    flags = {case: {band.get("in_domain") for band in _bands(_load(case))} for case in _CASES}
    outside = {case for case, values in flags.items() if values == {False}}
    inside = {case for case, values in flags.items() if values == {True}}
    assert outside == {"flat", "varied"}
    assert inside == {"lowabs"}


@pytest.mark.parametrize("case", _UNIFORM_CASES)
def test_uniform_raw_art_has_sabine_diffuse_field_scale(case: str) -> None:
    """上一代均勻材料 raw/Sabine 在 1e-3 內；這是性質描述，不是 v3 契約。"""
    answer = _load(case)
    parameters = _mapping(answer.get("parameters"), "parameters")
    room = _room(parameters)
    for band in _bands(answer):
        alpha_cells = _mapping(band.get("alpha_by_wall"), "band.alpha_by_wall")
        alpha = {wall: _real(cell, f"alpha_by_wall.{wall}") for wall, cell in alpha_cells.items()}
        estimate = check.sabine_diffuse_energy(room, alpha)
        ratio = _real(band.get("raw_art_rev_E"), "raw_art_rev_E") / estimate
        assert abs(ratio - 1.0) <= check.SABINE_CHARACTERIZATION_REL


@pytest.mark.parametrize("case", _CASES)
def test_late_energy_is_raw_times_eyring_ratio_bit_exact(case: str) -> None:
    """每頻帶 late = raw * eyring_ratio，只守浮點可逆。

    eyring_ratio 就是從 late/raw 算回來的；Eyring 修正本身沒有獨立核對，決策紙也明說
    修正後比值不算證據。
    """
    for band in _bands(_load(case)):
        raw = _real(band.get("raw_art_rev_E"), "raw_art_rev_E")
        ratio = _real(band.get("eyring_ratio"), "eyring_ratio")
        late = _mapping(band.get("late_rev_E"), "late_rev_E")
        assert (raw * ratio).hex() == late.get("hex")


def test_control_using_reflection_magnitude_instead_of_power_differs() -> None:
    """控制組：把 α 的 |R|² 寫錯成 |R|，每格都必須與答案相隔逾一個 float32 格。"""
    answer = _load("varied")
    parameters = _mapping(answer.get("parameters"), "parameters")
    rho_c = _real(parameters.get("rho_c_pa_s_per_m"), "rho_c")
    distances = []
    for band_index, band in enumerate(_bands(answer)):
        observed = _mapping(band.get("alpha_by_wall"), "alpha_by_wall")
        for wall, cell in observed.items():
            z = _impedance(parameters, wall, band_index)
            observed_cell = _real(cell, f"alpha.{wall}")
            wrong = check.normal_incidence_absorption_without_square(z, rho_c)
            distances.append(check.float32_cells_apart(observed_cell, wrong))
    assert all(distance > 1 for distance in distances)


def test_control_zero_contract_rejects_nonidentical_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    """控制組：契約常數換成 0，原本容許的非零相對差會被判決函式判紅。"""
    reference = _real(_bands(_load("flat"))[0].get("late_rev_E"), "late_rev_E")
    candidate = math.nextafter(reference, math.inf)
    assert check.art_contract_accepts(candidate, reference)
    monkeypatch.setattr(check, "ART_CONTRACT_REL", 0.0)
    assert not check.art_contract_accepts(candidate, reference)


def test_contract_accepts_below_boundary_and_rejects_above_boundary() -> None:
    """控制組：相對差 0.9 倍契約門檻判綠，1.1 倍判紅。"""
    reference = 1.0
    below = reference * (1.0 + 0.9 * check.ART_CONTRACT_REL)
    above = reference * (1.0 + 1.1 * check.ART_CONTRACT_REL)
    assert check.art_contract_accepts(below, reference)
    assert not check.art_contract_accepts(above, reference)
