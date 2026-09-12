"""票 #194 的考卷：讀振幅答案檔，用獨立雙精度重算逐格比對，斷言落在精度契約界線內。

**這支考卷不 import ``aosr``。** 它住的籃子是 ``tests/``（治理層的考卷，見規矩卡
``green-must-be-real-green``）：它驗的是「產生器跑出來的凍結振幅答案」跟「只 import 標準庫的
獨立雙精度重算」兩邊是否在契約界線內一致，跟新家引擎無關。

**跟產生器走同一套形狀、但算式完全獨立。** 產生器
（``blueprint/generate_amplitude_answers.py``）在唯讀的 donor 工作樹上跑上一代的
``solve_ism_shoebox_patched``（單精度 complex64）寫出答案檔；這一支用
``blueprint/reference_amplitude_check.py``（只 import ``math``/``cmath``/``dataclasses``）把每條
路徑的反射乘積與路徑壓力用雙精度重新算一遍，斷言跟答案的**差落在契約界線內**（不是逐位相同，
見決策紙 ``docs/decisions/precision-contract-amplitude-phase-scaled.md``——上一代是單精度算的，
逐位元相同做不到）。

**契約界線由獨立檢查的函式算，不寫死在考卷。** :func:`check.reflection_tolerance` 與
:func:`check.pressure_tolerance` 吃 f、τ、|refl| 回容差；考卷 import 它們，不重抄 ``2^-21``
（thresholds-live-only-in-registry 那張卡管的是**門檻數字**，這一格的契約常數住在獨立檢查
的模組常數並錨在決策紙，考卷從那裡 import）。斷言差 ≤ 界線，並把「用到界線幾成」的最大值
印在失敗訊息裡——證明不是拿一個鬆到沒用的界線放水。壓力界線是相對容差，判法用複數模
``|Δp| ≤ tol_rel·|p|``（``Δp`` 是複數差），不是逐分量各比。

**兩個特殊題。** ① 距離與到達時間照 ``precision-contract-geometry-bit-exact`` 逐位元：答案檔
的 ``dist_m.hex``／``delay_s.hex`` 逐字等於第三段 order3 答案檔對應路徑的同名格（兩邊 identity
對得上才比）。② 直達路徑（order 0）的反射乘積**恰等於 1+0j**，不設容差。

**控制組（證明裁判真的咬得住）。** ① 把答案檔某條路徑某一頻帶的反射乘積掰到界線外，餵同一支
判契約的函式，必須報出超界（紅）；② 把契約界線函式（``check.reflection_tolerance`` 與
``check.pressure_tolerance``）暫時換成 0，同一批對得上的資料必須全體超界（紅）——後者證明
那個界線真的被吃進去、不是裝飾。另外三個壓力界線控制組：壓力界線獨自換 0（反射不動）必全
是 ``path_pressure`` 超界；把 125 Hz 那格壓力乘 (1+5e-4) 必報那一格；正常界線下把壓力相位
共軛（只動壓力、反射不動）必紅——皆證明「壓力那條界線有意義」、不是跟著反射一起才被抓到。

**varied 那組的形狀題。** 打到 y0 牆的路徑，六個頻帶的反射乘積**互異**（real+imag 的 hex 對
各不相同）且虛部非零；沒打到 y0 的路徑，反射乘積與壓力跟 flat 那組**逐位相同**（hex 全等）——
證明 varied 的差只長在 y0 那一面、別的面沒被波及。
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from blueprint import reference_amplitude_check as check
from blueprint.reference_room_geometry import NUM_AXES, enumerate_identities

# 答案檔的位置：從這一支往上兩層是 repo 根，答案檔住在 blueprint 底下。
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
# 兩組振幅答案（flat／varied），參數化來源（不是把數量寫死——規矩卡
# assertions-not-pinned-to-counts）。
_AMPLITUDE_CASES: tuple[str, ...] = ("flat", "varied")

# 第三段的 order3 答案檔（距離與到達時間的逐位元契約錨點）。
_ORDER3_PATH: Path = _REPO_ROOT / "blueprint" / "reference_room_answers_order3.json"

# identity 六元組的長度：每軸一組（次數、正負號），共 2 * 軸數——軸數由獨立幾何供應
# （``blueprint.reference_room_geometry.NUM_AXES``），不寫死 6。
_IDENTITY_LEN: int = 2 * NUM_AXES

# 直達路徑的 identity（全 0、全 +1）。跟答案檔／獨立幾何一致。
_DIRECT_IDENTITY: tuple[int, int, int, int, int, int] = (0, 1, 0, 1, 0, 1)


def _amplitude_path(case: str) -> Path:
    return _REPO_ROOT / "blueprint" / f"reference_amplitude_{case}.json"


@dataclass(frozen=True)
class _Loaded:
    """一份振幅答案檔的收窄形狀：case 名、參數、獨立重算的輸入。"""

    case: str
    params: dict[str, object]
    inputs: check.Inputs


def _as_mapping(node: object, where: str) -> dict[str, object]:
    """把一個節點收窄成一層表。不是表就當場炸——沒看懂就不出結論。"""
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是一層表：{node!r}")
    return {str(key): value for key, value in node.items()}


def _as_number(node: object, where: str) -> float:
    """把 json 讀出來的一個節點收窄成 float。"""
    if not isinstance(node, (int, float)) or isinstance(node, bool):
        raise AssertionError(f"{where} 不是數：{node!r}")
    return float(node)


def _as_int(node: object, where: str) -> int:
    """把 json 讀出來的一個節點收窄成 int。"""
    if isinstance(node, bool) or not isinstance(node, int):
        raise AssertionError(f"{where} 不是整數：{node!r}")
    return node


def _xyz(params: dict[str, object], key: str) -> tuple[float, float, float]:
    table = _as_mapping(params.get(key), key)
    return (
        _as_number(table.get("x"), f"{key}.x"),
        _as_number(table.get("y"), f"{key}.y"),
        _as_number(table.get("z"), f"{key}.z"),
    )


def _split_list(node: object, where: str) -> list[dict[str, object]]:
    """把 ``paths`` 這類的 list 收窄成一串表。"""
    if not isinstance(node, list):
        raise AssertionError(f"{where} 不是一串東西")
    return [_as_mapping(item, where) for item in node]


def _as_number_list(node: object, where: str) -> list[float]:
    """把 json 的一串數字收窄成 ``list[float]``。"""
    if not isinstance(node, list):
        raise AssertionError(f"{where} 不是一串東西")
    return [_as_number(item, f"{where}[{idx}]") for idx, item in enumerate(node)]


def _identity_of(path: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    raw = path.get("identity")
    if not isinstance(raw, list) or len(raw) != _IDENTITY_LEN:
        raise AssertionError(f"identity 不是 {_IDENTITY_LEN} 元組：{raw!r}")
    return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]), int(raw[4]), int(raw[5]))


def _read_dec(cell: dict[str, object], slot: str) -> float:
    sub = _as_mapping(cell.get(slot), slot)
    dec = sub.get("dec")
    if dec is None:
        raise AssertionError(f"{slot}.dec 是 null")
    # ``dec`` 由產生器用 ``repr(float)`` 存成 JSON 字串；數值也可能是 int。
    if isinstance(dec, str):
        return float(dec)
    return _as_number(dec, f"{slot}.dec")


def _hex_of(cell: dict[str, object], slot: str) -> str:
    """一個分量的 hex 字串（比對逐位元用）。"""
    sub = _as_mapping(cell.get(slot), slot)
    raw = sub.get("hex")
    if not isinstance(raw, str):
        raise AssertionError(f"{slot}.hex 不是字串")
    return raw


def _complex_from_cell(cell: dict[str, object]) -> complex:
    """把答案檔一個複數格子（real/imag 的 dec）還原成複數。"""
    return complex(_read_dec(cell, "real"), _read_dec(cell, "imag"))


def _load(case: str) -> _Loaded:
    """讀一份振幅答案檔，抽參數、建 :class:`check.Inputs`。"""
    path = _amplitude_path(case)
    with path.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _as_mapping(data, path.name)
    params = _as_mapping(root.get("parameters"), "parameters")
    room = _as_mapping(params.get("room"), "room")
    freqs = _as_number_list(params.get("frequencies_hz"), "frequencies_hz")
    inputs = check.Inputs(
        lx=_as_number(room.get("Lx_m"), "Lx_m"),
        ly=_as_number(room.get("Ly_m"), "Ly_m"),
        lz=_as_number(room.get("Lz_m"), "Lz_m"),
        c=_as_number(params.get("sound_speed_m_s"), "sound_speed_m_s"),
        rho_c=_as_number(params.get("rho_c_pa_s_per_m"), "rho_c_pa_s_per_m"),
        src=_xyz(params, "source_xyz_m"),
        recv=_xyz(params, "receiver_xyz_m"),
        freqs_hz=tuple(freqs),
    )
    return _Loaded(case=case, params=params, inputs=inputs)


def _impedance_reader(loaded: _Loaded) -> check.WallImpedance:
    """從答案檔 ``parameters.cases`` 讀出「牆名＋頻帶 → Z」的饋料（全由答案檔讀，不寫死）。"""
    cases = _as_mapping(loaded.params.get("cases"), "cases")
    flat_case = _as_mapping(cases.get("flat"), "cases.flat")
    flat_mat = _as_mapping(flat_case.get("material"), "cases.flat.material")
    imp = _as_mapping(flat_mat.get("impedance"), "impedance")
    z_const = complex(
        _as_number(imp.get("real"), "impedance.real"),
        _as_number(imp.get("imag"), "impedance.imag"),
    )
    if loaded.case == "flat":
        return lambda _wall, _f: z_const

    varied_case = _as_mapping(cases.get("varied"), "cases.varied")
    mat = _as_mapping(varied_case.get("material"), "cases.varied.material")
    y0 = _as_mapping(mat.get("y0_wall"), "y0_wall")
    reals = _as_number_list(y0.get("per_frequency_real"), "y0.per_frequency_real")
    imags = _as_number_list(y0.get("per_frequency_imag"), "y0.per_frequency_imag")

    def reader(wall: str, f_idx: int) -> complex:
        if wall == "y0":
            return complex(reals[f_idx], imags[f_idx])
        return z_const

    return reader


def _answer_paths(loaded: _Loaded) -> list[dict[str, object]]:
    """讀答案檔的 ``paths``（讀檔層跟比對層分開，比對不重開檔）。"""
    with _amplitude_path(loaded.case).open(encoding="utf-8") as handle:
        root = _as_mapping(json.load(handle), "answer")
    return _split_list(root.get("paths"), "paths")


def _recompute_all(loaded: _Loaded) -> list[check.Recompute]:
    """對答案檔每一條路徑做獨立重算（反射乘積與路徑壓力）。"""
    impedance = _impedance_reader(loaded)
    return [
        check.recompute_path(loaded.inputs, impedance, _as_int(p.get("index"), "index"), _identity_of(p))
        for p in _answer_paths(loaded)
    ]


def _cell_list(path: dict[str, object], key: str) -> list[dict[str, object]]:
    """一條路徑的 ``reflection_product``／``path_pressure``（六頻帶的複數格清單）。"""
    return _split_list(path.get(key), f"path.{key}")


def _contract_violations(
    loaded: _Loaded,
    answer_paths: list[dict[str, object]] | None = None,
    recomputes: list[check.Recompute] | None = None,
) -> tuple[list[str], float, float]:
    """用獨立重算對答案檔逐格判契約，回（超界差異清單、反射乘積用到界線幾成、壓力幾成）。

    界線一律走 :func:`check.reflection_tolerance` 與 :func:`check.pressure_tolerance`（模組
    參考，控制組用 monkeypatch 換成 0 或 1e9 就真的吃到）。壓力界線是**相對**容差，判法
    用複數模：``|Δp| ≤ tol_rel·|p|``（``Δp`` 是複數差，不是逐分量各比）。``answer_paths``／
    ``recomputes`` 可覆寫（控制組餵掰過的答案／重算用），預設從答案檔重算。
    """
    if answer_paths is None:
        answer_paths = _answer_paths(loaded)
    if recomputes is None:
        recomputes = _recompute_all(loaded)
    diffs: list[str] = []
    max_refl_frac = 0.0
    max_pp_frac = 0.0

    def _frac(diff: float, tol: float) -> float:
        if tol == 0.0:
            return float("inf") if diff != 0.0 else 0.0
        return diff / tol

    for rp, p in zip(recomputes, answer_paths, strict=True):
        ident = rp.identity
        tau = rp.dist_m / loaded.inputs.c
        answer_refl = _cell_list(p, "reflection_product")
        answer_pp = _cell_list(p, "path_pressure")

        for f_idx in range(len(answer_refl)):
            ref = _complex_from_cell(answer_refl[f_idx])
            mine = rp.reflection_product[f_idx]
            tol = check.reflection_tolerance(loaded.inputs.freqs_hz[f_idx], tau, abs(ref))
            for slot, a, b in (
                ("real", ref.real, mine.real),
                ("imag", ref.imag, mine.imag),
                ("abs", abs(ref), abs(mine)),
            ):
                diff = abs(a - b)
                if diff > tol:
                    diffs.append(
                        f"identity {ident!r} reflection_product[{f_idx}].{slot} "
                        f"差 {diff!r} > 界線 {tol!r}"
                    )
                max_refl_frac = max(max_refl_frac, _frac(diff, tol))

        for f_idx in range(len(answer_pp)):
            ref = _complex_from_cell(answer_pp[f_idx])
            mine = rp.path_pressure[f_idx]
            abs_refl = abs(_complex_from_cell(answer_refl[f_idx]))
            tol_rel = check.pressure_tolerance(loaded.inputs.freqs_hz[f_idx], tau, abs_refl)
            diff = abs(mine - ref)
            abs_tol = tol_rel * abs(ref)
            if diff > abs_tol:
                diffs.append(
                    f"identity {ident!r} path_pressure[{f_idx}] "
                    f"複數差模 {diff!r} > 界線 {abs_tol!r}"
                )
            max_pp_frac = max(max_pp_frac, _frac(diff, abs_tol))

    return diffs, max_refl_frac, max_pp_frac


# ── 考題 ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_dist_and_delay_hex_match_order3(case: str) -> None:
    """dist_m.hex／delay_s.hex 逐字等於第三段 order3 答案檔對應路徑的同名格。"""
    loaded = _load(case)
    paths = _answer_paths(loaded)
    with _ORDER3_PATH.open(encoding="utf-8") as handle:
        order3 = _as_mapping(json.load(handle), "order3")
    by_identity = {
        _identity_of(entry): entry for entry in _split_list(order3.get("paths"), "order3 paths")
    }

    diffs: list[str] = []
    for p in paths:
        ident = _identity_of(p)
        ref = by_identity.get(ident)
        if ref is None:
            diffs.append(f"identity {ident!r} 不在 order3 答案檔")
            continue
        ref_dist = _as_mapping(ref.get("dist_m"), "dist_m")
        ref_delay = _as_mapping(ref.get("delay_s"), "delay_s")
        mine_dist = _as_mapping(p.get("dist_m"), "dist_m")
        mine_delay = _as_mapping(p.get("delay_s"), "delay_s")
        if mine_dist.get("hex") != ref_dist.get("hex"):
            diffs.append(f"identity {ident!r} dist_m.hex 不同")
        if mine_delay.get("hex") != ref_delay.get("hex"):
            diffs.append(f"identity {ident!r} delay_s.hex 不同")
    assert diffs == []


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_direct_reflection_product_is_exactly_one(case: str) -> None:
    """直達路徑（order 0）的反射乘積六個頻帶都恰等於 1+0j，不設容差。"""
    loaded = _load(case)
    direct = [p for p in _answer_paths(loaded) if _as_int(p.get("order"), "order") == 0]
    # order 0 的那幾條，identity 集合恰等於 {直達 identity}（不寫死「恰好一條」這種數字）。
    if {_identity_of(p) for p in direct} != {_DIRECT_IDENTITY}:
        raise AssertionError(f"order 0 路徑的 identity 不是恰等於直達：{[_identity_of(p) for p in direct]!r}")
    # 先用「直達 identity」對到那一條再比，不靠 direct 列表的長度。
    refl = _cell_list(direct[0], "reflection_product")
    for f_idx, cell in enumerate(refl):
        assert _complex_from_cell(cell) == complex(1.0, 0.0), (
            f"直達路徑反射乘積[{f_idx}] 不是 1+0j"
        )


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_reflection_and_pressure_within_contract(case: str) -> None:
    """每格反射乘積與壓力落在契約界線內；失敗訊息帶「用到界線幾成」的最大值。"""
    loaded = _load(case)
    diffs, max_refl_frac, max_pp_frac = _contract_violations(loaded)
    assert diffs == [], (
        f"超出契約界線的格有 {len(diffs)} 個；反射乘積用到界線 {
            max_refl_frac:.6f}、壓力 {max_pp_frac:.6f}。前幾筆：{diffs[:5]!r}"
    )


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_identity_set_matches_independent_enumeration(case: str) -> None:
    """答案檔的 identity 集合等於獨立幾何照 donor 去重規則枚舉出的集合。"""
    loaded = _load(case)
    answer_set = {_identity_of(p) for p in _answer_paths(loaded)}
    max_order = int(_as_number(loaded.params.get("max_order"), "max_order"))
    enumerated = enumerate_identities(max_order)
    assert answer_set == enumerated


def _hits_y0(identity: tuple[int, int, int, int, int, int]) -> bool:
    return check.wall_count_signature(identity)["y0"] > 0


def test_varied_y0_paths_six_distinct_bands_and_nonzero_imag() -> None:
    """varied：打到 y0 六頻帶反射乘積互異且虛部非零；沒打到的跟 flat 逐位相同。"""
    flat = _load("flat")
    varied = _load("varied")
    flat_paths = _answer_paths(flat)
    varied_paths = _answer_paths(varied)
    if len(flat_paths) != len(varied_paths):
        raise AssertionError(f"flat {len(flat_paths)} 條 ≠ varied {len(varied_paths)} 條")

    hit_y0_count = 0
    for vp, fp in zip(varied_paths, flat_paths, strict=True):
        ident = _identity_of(vp)
        var_refl = _cell_list(vp, "reflection_product")
        flat_refl = _cell_list(fp, "reflection_product")
        var_pp = _cell_list(vp, "path_pressure")
        flat_pp = _cell_list(fp, "path_pressure")
        if _hits_y0(ident):
            hit_y0_count += 1
            refl_hexes = [
                (_hex_of(cell, "real"), _hex_of(cell, "imag")) for cell in var_refl
            ]
            assert len(set(refl_hexes)) == len(refl_hexes), (
                f"identity {ident!r} 打到 y0 但六頻帶反射乘積不是互異"
            )
            assert any(
                _complex_from_cell(cell).imag != 0.0 for cell in var_refl
            ), f"identity {ident!r} 打到 y0 但全部虛部為 0"
        else:
            for f_idx in range(len(var_refl)):
                assert var_refl[f_idx] == flat_refl[f_idx], (
                    f"identity {ident!r} 沒打到 y0 但 reflection_product[{f_idx}] 跟 flat 不同"
                )
                assert var_pp[f_idx] == flat_pp[f_idx], (
                    f"identity {ident!r} 沒打到 y0 但 path_pressure[{f_idx}] 跟 flat 不同"
                )
    assert hit_y0_count > 0, "varied 沒有打到 y0 的路徑"


# ── 控制組 ────────────────────────────────────────────────────────────────────


def _set_complex_cell(cell: dict[str, object], z: complex) -> None:
    """把答案檔一個複數格的實／虛 dec 重寫成 ``z``（abs 那格不碰）。就地改、不回新表。"""
    real_cell = cell["real"]
    imag_cell = cell["imag"]
    assert isinstance(real_cell, dict) and isinstance(imag_cell, dict)
    real_cell["dec"] = repr(z.real)
    imag_cell["dec"] = repr(z.imag)


def _tamper_pressure_cell(
    loaded: _Loaded,
    cell_selector: Callable[[tuple[int, int, int, int, int, int], int], bool],
    mutate: Callable[[complex], complex],
) -> list[dict[str, object]]:
    """把答案檔某一格 ``path_pressure`` 的複數改成 ``mutate(z)``，回一份掰過的 paths 清單。

    ``cell_selector`` 是 ``(identity, f_idx) -> bool``，選中就走 ``mutate`` 改那格（實、虛各
    重存 dec，abs 那格留著是答案檔自己的格式，契約比對不看它）。
    """
    paths = copy.deepcopy(_answer_paths(loaded))
    changed = 0
    for p in paths:
        ident = _identity_of(p)
        pp = _cell_list(p, "path_pressure")
        for f_idx, cell in enumerate(pp):
            if not cell_selector(ident, f_idx):
                continue
            z = _complex_from_cell(cell)
            _set_complex_cell(cell, mutate(z))
            changed += 1
    assert changed > 0, "控制組：沒有選到任何 path_pressure 格"
    return paths


def test_control_group_tampered_cell_goes_red() -> None:
    """控制組 ①：把答案某一格反射乘積掰到界線外，餵判契約的函式必報超界。"""
    loaded = _load("flat")
    paths = copy.deepcopy(_answer_paths(loaded))
    changed = 0
    for p in paths:
        if _as_int(p.get("order"), "order") == 0:
            continue
        refl = _cell_list(p, "reflection_product")
        _set_complex_cell(
            refl[0],
            complex(
                _complex_from_cell(refl[0]).real + 2.0 * check.REFLECTION_CONTRACT_ULP,
                _complex_from_cell(refl[0]).imag,
            ),
        )
        changed += 1
        # 每條都掰就算了數量，這裡只掰「恰好」選中的第一條就夠驗——但為讓違規明確定為這條，
        # 只掰一條 order 1 就走（order 0 跳過、其餘不動）。
        break
    assert changed == 1, "控制組 ①：沒掰到反射乘積格"
    diffs, _rf, _pf = _contract_violations(loaded, answer_paths=paths)
    assert any("reflection_product" in d for d in diffs), (
        f"控制組 ①：掰到界線外的反射乘積沒被報出（diffs={diffs!r}）"
    )


def test_control_group_zero_bound_goes_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """控制組 ②：把契約界線函式換成 0，同一批資料必全體超界（界線真的在吃）。"""
    loaded = _load("flat")
    monkeypatch.setattr(check, "reflection_tolerance", lambda f, tau, ar: 0.0)
    monkeypatch.setattr(check, "pressure_tolerance", lambda f, tau, ar: 0.0)
    diffs, _rf, _pf = _contract_violations(loaded)
    assert diffs != [], "控制組 ②：界線換 0 之後居然沒有超界（界線沒被吃進去）"


def test_control_group_pressure_bound_zero_goes_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """壓力界線換 0（反射不動）→ 違規非空且全是 path_pressure（壓力界線真的被吃）。"""
    loaded = _load("flat")
    monkeypatch.setattr(check, "pressure_tolerance", lambda f, tau, ar: 0.0)
    diffs, _rf, _pf = _contract_violations(loaded)
    assert diffs != [], "壓力界線換 0 之後居然沒有超界"
    assert all("path_pressure" in d for d in diffs), (
        f"壓力界線換 0 之後違規不全是 path_pressure：{diffs[:5]!r}"
    )
    assert not any("reflection_product" in d for d in diffs), (
        f"壓力界線換 0 不該動到 reflection_product：{diffs[:5]!r}"
    )


def test_control_group_pressure_small_error_reported() -> None:
    """把 125 Hz 那格壓力乘 (1+5e-4)，判契約函式必報那一格。"""
    loaded = _load("flat")
    freqs = loaded.inputs.freqs_hz
    f_125 = min((i for i, f in enumerate(freqs) if f == 125.0), default=None)
    assert f_125 is not None, "答案檔沒有 125 Hz 的頻帶"

    def selector(ident: tuple[int, int, int, int, int, int], f_idx: int) -> bool:
        del ident
        return f_idx == f_125

    paths = _tamper_pressure_cell(loaded, selector, lambda z: z * (1.0 + 5e-4))
    diffs, _rf, _pf = _contract_violations(loaded, answer_paths=paths)
    assert any(f"path_pressure[{f_125}]" in d for d in diffs), (
        f"壓力乘 1+5e-4 之後 125 Hz 那格沒被報出：{diffs[:5]!r}"
    )


def test_control_group_pressure_conjugate_goes_red() -> None:
    """正常界線下把全部壓力相位共軛（只動壓力、反射不動）→ 必紅（壓力界線有意義）。"""
    loaded = _load("flat")
    paths = _tamper_pressure_cell(loaded, lambda ident, f_idx: True, lambda z: complex(z.real, -z.imag))
    diffs, _rf, _pf = _contract_violations(loaded, answer_paths=paths)
    assert diffs != [], "壓力相位共軛之後居然沒有超界"
    assert all("path_pressure" in d for d in diffs), (
        f"共軛只動壓力，違規卻不全是 path_pressure：{diffs[:5]!r}"
    )
