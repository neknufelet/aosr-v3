"""答案檔的讀取與逐條比對，以及共用序列化輔助（dec/hex、複數格）。

**這一支是純函式模組，不 print。** 它從 :mod:`aosr.physics.room_paths` 拆出來（票 #206 第 2a 段）：
把「答案檔的讀取與逐條比對」整組搬到這裡，``room_paths.py`` 只留載入輸入、算路徑、印表、``main``。
命令列輸出一字不變（``--json``、表格、``--compare`` 印的字），比對的判決行照舊從這裡的判決
結構印，不靠字串 startswith 去撈。

**比對照契約（決策紙 ``docs/decisions/precision-contract-amplitude-phase-scaled.md``）。**
每條路徑比 index、order、identity、牆名序列、鏡像三軸 hex、``dist_m`` hex、``delay_s`` hex、
反彈數與反彈點；答案檔有振幅欄時依契約逐格比反射乘積與路徑壓力。界線常數與界線函式從
:mod:`aosr.physics.amplitude` 這一個地方來（``reflection_tolerance``、``pressure_tolerance``），
本模組用 ``from aosr.physics.amplitude import pressure_tolerance`` 匯入後裸名呼叫，所以
``monkeypatch`` 要打 ``compare`` 模組的名字；打 ``amplitude`` 會靜默無效。

**序列化輔助住這裡。** ``_dec_hex``／``_complex_hex_dec`` 從 ``room_paths.py`` 搬過來，供
``room_paths.path_to_dict``（答案檔 ``paths`` 同形）與 ``totals.totals_to_payload``（答案檔
``totals`` 同形）兩邊共用，不抄兩份。

**不 print。** ``style-guard`` 的輸出層白名單只列 ``room_paths.py``，這一支回值、不解讀、
不對人說話。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from aosr.physics.amplitude import (
    pressure_tolerance,
    reflection_tolerance,
)

if TYPE_CHECKING:
    from aosr.physics.room_paths import RoomPath

# 直達路徑的 identity：全 0、全 +1。
_DIRECT_IDENTITY: Final[tuple[int, int, int, int, int, int]] = (0, 1, 0, 1, 0, 1)


@dataclass(frozen=True)
class PathComparison:
    """``--compare`` 對一條路徑的判決：index、牆名序列、差異清單（空＝完全相同）。

    ``max_refl_frac`` 是反射乘積「用到界線幾成」的最大值、``max_pp_frac`` 是壓力「用到界線
    幾成」的最大值（都是差／界線的比值：≤1 在契約內，>1 就超界；沒比振幅時兩者都是 0）。
    ``worst`` 記「超界最多」的那一格（``(量名, 頻帶, 差, 界線, 倍數)``），只在有超界時非空——
    判決行有超界時要印出它，而不是只印總數。判決行從這個結構印，不用字串 startswith 去撈。
    """

    index: int
    wall_seq: str
    diffs: list[str]
    max_refl_frac: float = 0.0
    max_pp_frac: float = 0.0
    worst: tuple[str, int, float, float, float] | None = None


@dataclass(frozen=True)
class AnswerFile:
    """命令列答案檔：逐條 ``paths``，以及可選但要能區分 null 的頂層 ``totals``。"""

    paths: list[dict[str, object]]
    has_totals: bool
    totals: object


def _group_total_diffs(
    diffs: list[str], n_freq: int
) -> tuple[tuple[str, ...], ...]:
    """把總量裁判的 ``量名[頻帶] 超界：細節`` 依頻帶分組，並拿掉重複判詞。"""
    grouped: list[list[str]] = [[] for _ in range(n_freq)]
    for diff in diffs:
        cell, separator, detail = diff.partition(" 超界：")
        kind, bracket, index_text = cell.rpartition("[")
        if not separator or not bracket or not index_text.endswith("]"):
            raise ValueError(f"總量差異格式不對：{diff!r}")
        try:
            index = int(index_text[:-1])
        except ValueError as exc:
            raise ValueError(f"總量差異頻帶不是整數：{diff!r}") from exc
        if not 0 <= index < n_freq:
            raise ValueError(f"總量差異頻帶超出範圍：{diff!r}")
        grouped[index].append(f"{kind}[{index}] {detail}")
    return tuple(tuple(items) for items in grouped)


def _mapping(node: object, where: str) -> dict[str, object]:
    """把一個節點收窄成一層表。不是表就丟 ValueError（指名在哪一格）。"""
    if not isinstance(node, dict):
        raise ValueError(f"{where} 不是一層表：{node!r}")
    return {str(key): value for key, value in node.items()}


def wall_name_seq_from_identity(identity: tuple[int, int, int, int, int, int]) -> str:
    """identity 六元組 → 牆名序列（canonical 順序展開，從 identity 純推、不靠幾何）。

    六元組 ``(nx, sx, ny, sy, nz, sz)`` 每軸一組（反射次數、正負號）。每軸兩面牆的反射
    次數拆成「zero 面幾次、L 面幾次」（``sign=+1`` 兩面各 ``|n|``、``sign=-1`` 依 ``n``
    的正負把奇數次分給兩面）；牆名照 canonical 順序 floor/ceiling/x0/xL/y0/yL 展開、用
    ``→`` 接起來。直達（全 0 全 +1）回 ``"direct"``。這是**牆次的簽名**（哪面牆幾次），
    不是逐次反彈的時間順序——時間順序是幾何量，答案檔沒有存，比對時不看。
    """
    if identity == _DIRECT_IDENTITY:
        return "direct"
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])

    def _zero_l(nx: int, sign: int) -> tuple[int, int]:
        """一軸的 (zero 面次數, L 面次數)。"""
        if sign == 1:
            return abs(nx), abs(nx)
        m = 2 * nx - 1
        order = abs(m)
        if m > 0:
            return order // 2, (order + 1) // 2
        return (order + 1) // 2, order // 2

    z_lo, z_hi = _zero_l(n[2], s[2])
    x_lo, x_hi = _zero_l(n[0], s[0])
    y_lo, y_hi = _zero_l(n[1], s[1])
    parts: list[str] = []
    parts.extend(["floor"] * z_lo)
    parts.extend(["ceiling"] * z_hi)
    parts.extend(["x0"] * x_lo)
    parts.extend(["xL"] * x_hi)
    parts.extend(["y0"] * y_lo)
    parts.extend(["yL"] * y_hi)
    return "→".join(parts)


def _dec_hex(value: float) -> dict[str, str]:
    """一個浮點同時存十進位與十六進位兩格（跟答案檔的 ``dec``／``hex`` 同形）。"""
    return {"dec": repr(value), "hex": value.hex()}


def _complex_hex_dec(value: complex) -> dict[str, object]:
    """一個複數存實／虛／abs 三格，各含 ``dec`` 與 ``hex``（跟答案檔同形）。"""
    return {
        "real": _dec_hex(value.real),
        "imag": _dec_hex(value.imag),
        "abs": _dec_hex(abs(value)),
    }


def _hex_cell(node: object, where: str) -> str:
    """答案檔裡 ``dist_m``／``delay_s``／``image_xyz.<axis>`` 的 hex 字串。"""
    cell = _mapping(node, where)
    hexed = cell.get("hex")
    if not isinstance(hexed, str):
        raise ValueError(f"{where} 沒有 hex 那一格")
    return hexed


def _answer_hexes(
    path: dict[str, object],
) -> tuple[tuple[str, str, str] | None, str, str]:
    """一條答案檔 path → ((img_x_hex, img_y_hex, img_z_hex)|None, dist_hex, delay_hex)。

    ``image_xyz`` 只在幾何答案檔（``reference_room_answers*.json``）有；振幅答案檔
    （``reference_amplitude_*.json``）沒有那一格，回 ``None``（比對時跳過鏡像）。
    """
    image = path.get("image_xyz")
    if image is None:
        image_hexes: tuple[str, str, str] | None = None
    else:
        table = _mapping(image, "paths.image_xyz")
        image_hexes = (
            _hex_cell(table.get("x"), "image_xyz.x"),
            _hex_cell(table.get("y"), "image_xyz.y"),
            _hex_cell(table.get("z"), "image_xyz.z"),
        )
    return (
        image_hexes,
        _hex_cell(path.get("dist_m"), "dist_m"),
        _hex_cell(path.get("delay_s"), "delay_s"),
    )


def _dec_cell(node: object, where: str) -> float:
    """答案檔一個 ``{dec, hex}`` 格的 ``dec``，還原成 float（字串或數都收）。"""
    cell = _mapping(node, where)
    dec = cell.get("dec")
    if dec is None:
        raise ValueError(f"{where}.dec 是 null")
    if isinstance(dec, str):
        return float(dec)
    if isinstance(dec, (int, float)) and not isinstance(dec, bool):
        return float(dec)
    raise ValueError(f"{where}.dec 不是數：{dec!r}")


def _answer_complex_cell(node: object, where: str) -> complex:
    """答案檔一個複數格（``{real:{dec,…}, imag:{dec,…}, …}``），還原成複數。"""
    cell = _mapping(node, where)
    return complex(_dec_cell(cell.get("real"), f"{where}.real"), _dec_cell(cell.get("imag"), f"{where}.imag"))


def _answer_complex_list(node: object, where: str) -> list[complex]:
    """答案檔一條路徑的 ``reflection_product``／``path_pressure``（六頻帶複數清單）。"""
    if not isinstance(node, list):
        raise ValueError(f"{where} 不是一串東西：{node!r}")
    return [_answer_complex_cell(entry, f"{where}[{idx}]") for idx, entry in enumerate(node)]


def _compare_amplitude(
    ours: RoomPath,
    their_refl: list[complex],
    their_pp: list[complex],
    frequencies: tuple[float, ...],
) -> tuple[list[str], float, float, tuple[str, int, float, float, float] | None]:
    """比一條路徑的反射乘積與路徑壓力依契約逐格，回（差異清單、反射用到幾成、壓力用到
    幾成、超界最多的那一格）。

    反射乘積每分量（實、虛、abs）絕對差 ≤ :func:`aosr.physics.amplitude.reflection_tolerance`；
    路徑壓力相對差用**複數差的模**：``|Δp| ≤ pressure_tolerance·|p|``（``Δp`` 是複數差，
    不是逐分量各比）。``max_refl_frac``／``max_pp_frac`` 是「差／界線」的最大值（≤1 在契約
    內，>1 就超界）；``worst`` 記超界最多那格 ``(量名, 頻帶, 差, 界線, 倍數)``。
    """
    tau = ours.delay_s
    diffs: list[str] = []
    max_refl_frac = 0.0
    max_pp_frac = 0.0
    worst: tuple[str, int, float, float, float] | None = None

    def _frac(diff: float, tol: float) -> float:
        if tol == 0.0:
            return float("inf") if diff != 0.0 else 0.0
        return diff / tol

    def _note_worst(kind: str, f_idx: int, diff: float, tol: float) -> None:
        nonlocal worst
        frac = _frac(diff, tol)
        if worst is None or frac > worst[4]:
            worst = (kind, f_idx, diff, tol, frac)

    # 反射乘積：每分量絕對差 ≤ 2^-21（實、虛、abs 各比一次）。
    for f_idx, freq in enumerate(frequencies):
        if f_idx >= len(ours.reflection_product) or f_idx >= len(their_refl):
            break
        refl_answer = their_refl[f_idx]
        refl_mine = ours.reflection_product[f_idx]
        for comp_name, a_val, b_val in (
            ("real", refl_answer.real, refl_mine.real),
            ("imag", refl_answer.imag, refl_mine.imag),
            ("abs", abs(refl_answer), abs(refl_mine)),
        ):
            diff = abs(a_val - b_val)
            tol = reflection_tolerance(freq, tau, abs(refl_answer))
            if diff > tol:
                diffs.append(
                    f"reflection_product[{f_idx}].{comp_name} 超界：差 {diff!r} > 界線 {tol!r}"
                )
                _note_worst("reflection_product", f_idx, diff, tol)
            max_refl_frac = max(max_refl_frac, _frac(diff, tol))

    # 路徑壓力：複數差的模 ≤ tol·|p|（不是逐分量各比）。
    for f_idx, freq in enumerate(frequencies):
        if f_idx >= len(ours.path_pressure) or f_idx >= len(their_pp):
            break
        pp_answer = their_pp[f_idx]
        pp_mine = ours.path_pressure[f_idx]
        abs_refl = abs(their_refl[f_idx])
        tol_rel = pressure_tolerance(freq, tau, abs_refl)
        abs_tol = tol_rel * abs(pp_answer)
        diff = abs(pp_mine - pp_answer)
        if diff > abs_tol:
            diffs.append(
                f"path_pressure[{f_idx}] 超界：複數差模 {diff!r} > 界線 {abs_tol!r}"
            )
            _note_worst("path_pressure", f_idx, diff, abs_tol)
        max_pp_frac = max(max_pp_frac, _frac(diff, abs_tol))

    return diffs, max_refl_frac, max_pp_frac, worst


def _answer_identity(entry: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    """答案檔一條路徑的 identity 六元組。"""
    raw = entry.get("identity")
    if not isinstance(raw, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in raw):
        raise ValueError(f"答案檔 identity 不是整數串：{raw!r}")
    if len(raw) != 6:
        raise ValueError(f"答案檔 identity 不是六元組：{raw!r}")
    return (raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])


def _answer_index(entry: dict[str, object]) -> int:
    """答案檔一條路徑的 index。缺了或不是整數就丟 ValueError。"""
    raw = entry.get("index")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"答案檔 index 不是整數：{raw!r}")
    return raw


def _answer_order(entry: dict[str, object]) -> int:
    """答案檔一條路徑的 order。缺了或不是整數就丟 ValueError。"""
    raw = entry.get("order")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"答案檔 order 不是整數：{raw!r}")
    return raw


def _amplitude_comparison(
    ours: RoomPath,
    theirs: dict[str, object],
    frequencies: tuple[float, ...] | None,
    one: list[str],
) -> tuple[float, float, tuple[str, int, float, float, float] | None]:
    """比一條路徑的振幅（反射乘積與壓力）依契約，把差異寫進 ``one``、回（反射幾成、壓力
    幾成、超界最多那格）。

    條件反過來看「答案檔那一側有沒有振幅欄」：答案檔有而我方沒有（沒 materials／頻帶空）→
    報「我方沒有振幅可比」；答案檔沒振幅欄 → 不進這支（幾何那側已比完）；兩邊都有才逐格比。
    """
    answer_has_amp = "reflection_product" in theirs or "path_pressure" in theirs
    if not answer_has_amp:
        return 0.0, 0.0, None
    if not ours.reflection_product:
        one.append("我方沒有振幅可比")
        return 0.0, 0.0, None
    if "reflection_product" not in theirs or "path_pressure" not in theirs:
        one.append("答案檔沒有 reflection_product／path_pressure 振幅欄")
        return 0.0, 0.0, None
    if frequencies is None:
        one.append("我方沒有振幅可比")
        return 0.0, 0.0, None
    try:
        their_refl = _answer_complex_list(theirs["reflection_product"], "reflection_product")
        their_pp = _answer_complex_list(theirs["path_pressure"], "path_pressure")
    except ValueError as exc:
        one.append(f"答案檔振幅欄讀不進：{exc}")
        return 0.0, 0.0, None
    if len(their_refl) != len(frequencies) or len(their_pp) != len(frequencies):
        one.append(
            f"答案檔振幅頻帶數 {len(their_refl)}/{len(their_pp)} "
            f"不是 {len(frequencies)}"
        )
        return 0.0, 0.0, None
    amp_diffs, max_refl_frac, max_pp_frac, worst = _compare_amplitude(
        ours, their_refl, their_pp, frequencies
    )
    one.extend(amp_diffs)
    return max_refl_frac, max_pp_frac, worst


def _compare_one_path(
    ours: RoomPath,
    theirs: dict[str, object],
    position: int,
    frequencies: tuple[float, ...] | None,
) -> PathComparison:
    """比 v3 一條路徑跟答案檔一條：index、order、identity、牆名序列、hex、反彈、振幅。"""
    theirs_identity = _answer_identity(theirs)
    our_wall = wall_name_seq_from_identity(ours.identity)
    their_wall = wall_name_seq_from_identity(theirs_identity)

    one: list[str] = []
    if ours.index != position:
        one.append(f"index 是 {ours.index}，不等於位置 {position}")
    if ours.index != _answer_index(theirs):
        one.append(f"index 不同：v3={ours.index}，答案={_answer_index(theirs)}")
    if ours.order != _answer_order(theirs):
        one.append(f"order 不同：v3={ours.order}，答案={_answer_order(theirs)}")
    if ours.identity != theirs_identity:
        one.append(f"identity 不同：v3={list(ours.identity)!r}，答案={list(theirs_identity)!r}")
    if our_wall != their_wall:
        one.append(f"牆名序列不同：v3={our_wall!r}，答案={their_wall!r}")
    if len(ours.bounces) != ours.order:
        one.append(f"反彈數 {len(ours.bounces)} 不等於 order {ours.order}")

    our_image = tuple(v.hex() for v in ours.image)
    their_image, their_dist, their_delay = _answer_hexes(theirs)
    if their_image is not None:
        for axis_index, axis_label in enumerate(("x", "y", "z")):
            if our_image[axis_index] != their_image[axis_index]:
                one.append(
                    f"鏡像 {axis_label} hex 不同：v3={our_image[axis_index]!r}，"
                    f"答案={their_image[axis_index]!r}"
                )

    if ours.dist_m.hex() != their_dist:
        one.append(f"dist_m.hex 不同：v3={ours.dist_m.hex()!r}，答案={their_dist!r}")
    if ours.delay_s.hex() != their_delay:
        one.append(f"delay_s.hex 不同：v3={ours.delay_s.hex()!r}，答案={their_delay!r}")
    for bounce in ours.bounces:
        if bounce.in_wall is False:
            one.append(f"反射點（{bounce.wall}）不在牆上")

    max_refl_frac, max_pp_frac, worst = _amplitude_comparison(
        ours, theirs, frequencies, one
    )
    return PathComparison(
        index=ours.index,
        wall_seq=our_wall,
        diffs=one,
        max_refl_frac=max_refl_frac,
        max_pp_frac=max_pp_frac,
        worst=worst,
    )


def compare_paths(
    paths: list[RoomPath],
    answer_paths: list[dict[str, object]],
    frequencies: tuple[float, ...] | None = None,
) -> list[PathComparison]:
    """逐條比 v3 算出來的路徑跟答案檔，回結構化結果（每一條一筆：index、牆名序列、差異清單）。

    每一條比：index 等於位置、order、identity、牆名序列（答案檔沒有牆名欄，兩邊 identity 各
    推一次 :func:`wall_name_seq_from_identity` 再比）、``image_xyz`` 三個 hex、``dist_m``
    hex、``delay_s`` hex，以及反彈數等於 order、每個反彈點的 ``in_wall`` 為 True；答案檔有
    振幅欄時依契約比反射乘積與壓力（我方沒振幅就報「我方沒有振幅可比」）。兩邊條數不同要
    逐條報「少了哪一條（牆名序列）」，不靠 zip 截斷。
    """
    diffs: list[PathComparison] = []
    wall_of = wall_name_seq_from_identity
    common = min(len(paths), len(answer_paths))

    for position in range(common):
        theirs = _mapping(answer_paths[position], "answer_paths 的一筆")
        diffs.append(_compare_one_path(paths[position], theirs, position, frequencies))

    # 條數不同：逐條報少了哪一條（牆名序列），不用 zip 截斷。
    if len(paths) > common:
        for ours in paths[common:]:
            diffs.append(
                PathComparison(
                    index=ours.index,
                    wall_seq=wall_of(ours.identity),
                    diffs=[f"答案檔少了這一條（{wall_of(ours.identity)}）"],
                )
            )
    elif len(answer_paths) > common:
        for position in range(common, len(answer_paths)):
            theirs = _mapping(answer_paths[position], "answer_paths 的一筆")
            name = wall_of(_answer_identity(theirs))
            diffs.append(
                PathComparison(
                    index=position,
                    wall_seq=name,
                    diffs=[f"v3 少了這一條（{name}）"],
                )
            )

    return diffs


def _load_answer_file(path: Path) -> AnswerFile:
    """讀一份答案檔，保留 ``paths`` 與頂層 ``totals`` 是否存在。"""
    with path.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _mapping(data, "答案檔")
    raw_paths = root.get("paths")
    if not isinstance(raw_paths, list):
        raise ValueError("答案檔沒有 paths 那一欄，或它不是一串東西")
    return AnswerFile(
        paths=[_mapping(entry, "答案檔 paths 的一筆") for entry in raw_paths],
        has_totals="totals" in root,
        totals=root.get("totals"),
    )


def _load_answer(path: Path) -> list[dict[str, object]]:
    """相容舊呼叫端：讀一份答案檔，只回它的 ``paths``。"""
    return _load_answer_file(path).paths
