"""票 #181 第二段的考卷：v3 自己算參考房，跟凍結答案逐位比。

**這一支是引擎考卷**（住 ``tests/engine/``，規矩卡 ``green-must-be-real-green`` 的引擎籃），
它 import 新家的 ``aosr``（``aosr.geometry.shoebox`` 與 ``aosr.physics.room_paths``）。
比對的標準答案是唯讀的 ``blueprint/reference_room_answers.json``——考卷可以 import
``blueprint``，``src/`` 不行（規矩卡 ``layers-import-downward-only`` 只准 ``src/aosr``
往下引，藍圖在它外面）。

**逐位相等不是 isclose。** 七條路徑的 ``image/dist/delay`` 的 ``hex`` 字串逐字相等
（``float.hex()`` 的十六進位是精確表示，跨平台不漂）。數量不鎖死成一個整數：牆名用
集合相等，條數用兩邊現算的 ``len`` 互比（規矩卡 ``assertions-not-pinned-to-counts``）。

**不碰真環境。** 只寫 ``tmp_path``（規矩卡 ``tests-isolated-from-real-env``）：改壞的
答案檔、範例輸入檔全部在 ``tmp_path`` 生成，不寫進 repo。
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Point, Wall
from aosr.physics.room_paths import (
    RoomInput,
    RoomPath,
    compare_paths,
    image_source_paths,
    load_room_input,
    main,
)

# 答案檔位置（唯讀）：repo 根往上兩層是 tests/engine，答案是 blueprint 底下。
ANSWER_PATH: Path = (
    Path(__file__).resolve().parents[2] / "blueprint" / "reference_room_answers.json"
)

# 期望的牆名集合：六面牆 + 直達。牆名由新家的 Wall 枚舉現算（不鎖死數量）。
_EXPECTED_WALLS: frozenset[str] = frozenset(Wall.wall_names()) | {"direct"}


def _frozen_parameters() -> dict[str, object]:
    """答案檔的 parameters（凍結、唯讀）。"""
    with ANSWER_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    params = data["parameters"]
    if not isinstance(params, dict):
        raise AssertionError("parameters 不是一層表")
    return params


def _frozen_answer_paths() -> list[dict[str, object]]:
    """答案檔的 paths（唯讀）。"""
    with ANSWER_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    raw = data["paths"]
    if not isinstance(raw, list):
        raise AssertionError("paths 不是一串東西")
    return [entry for entry in raw if isinstance(entry, dict)]


# 輸入檔只吃這些欄位（答案檔 parameters 還帶 units/convention/grid_shapes 這種 donor 自己的
# 中介資料，v3 的 CLI 讀的是幾何/聲學欄位加 max_order）。
_INPUT_KEYS: tuple[str, ...] = (
    "room",
    "source_xyz_m",
    "receiver_xyz_m",
    "sound_speed_m_s",
    "max_order",
)


def _input_params() -> dict[str, object]:
    """從凍結 parameters 摘出輸入檔欄位（唯讀，回一份新的 dict，不污染凍結那份）。"""
    params = _frozen_parameters()
    return {key: params[key] for key in _INPUT_KEYS}


def _write_input(tmp_path: Path, params: dict[str, object]) -> Path:
    """把一份輸入欄位寫成範例輸入檔到 tmp_path，回路徑。"""
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    return path


def _room_input(tmp_path: Path, **overrides: object) -> RoomInput:
    """用答案檔輸入欄位寫輸入檔再餵 loader（可逐格覆寫頂層欄位）。"""
    params = _input_params()
    for key, value in overrides.items():
        params[key] = value
    return load_room_input(_write_input(tmp_path, params))


def _v3_paths(inp: RoomInput) -> list[RoomPath]:
    """用一份輸入跑 v3，回路徑（直達＋六面牆）。"""
    return image_source_paths(inp.room, inp.source, inp.receiver, inp.sound_speed, inp.max_order)


def _answer_identity(entry: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    """答案檔一條路徑的 identity 六元組。"""
    raw = entry["identity"]
    assert isinstance(raw, list)
    return (raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])


def _answer_hex_str(entry: dict[str, object], key: str) -> str:
    """答案檔 ``dist_m``／``delay_s`` 那一格的 hex 字串。"""
    cell = entry[key]
    assert isinstance(cell, dict)
    hexed = cell["hex"]
    assert isinstance(hexed, str)
    return hexed


def _answer_image_hex(entry: dict[str, object]) -> tuple[str, str, str]:
    """答案檔 ``image_xyz`` 三軸的 hex 字串。"""
    cell = entry["image_xyz"]
    assert isinstance(cell, dict)
    out: list[str] = []
    for axis in ("x", "y", "z"):
        one = cell[axis]
        assert isinstance(one, dict)
        hexed = one["hex"]
        assert isinstance(hexed, str)
        out.append(hexed)
    return (out[0], out[1], out[2])


# ── 逐位比對 ────────────────────────────────────────────────────────────────


def test_paths_match_frozen_hex_exactly(tmp_path: Path) -> None:
    """用答案檔 parameters 跑 v3，每一條的 image/dist/delay hex 逐字相等。"""
    v3 = _v3_paths(_room_input(tmp_path))
    answers = _frozen_answer_paths()
    assert len(v3) == len(answers), "v3 條數跟答案檔不同"

    for ours, theirs in zip(v3, answers):
        assert ours.dist_m.hex() == _answer_hex_str(theirs, "dist_m"), "dist_m.hex 不同"
        assert ours.delay_s.hex() == _answer_hex_str(theirs, "delay_s"), "delay_s.hex 不同"
        assert tuple(v.hex() for v in ours.image) == _answer_image_hex(theirs), "鏡像 hex 不同"
        assert ours.identity == _answer_identity(theirs), "identity 不同"


def test_wall_names_are_six_plus_direct(tmp_path: Path) -> None:
    """路徑的牆名集合等於 {direct, x0, xL, y0, yL, floor, ceiling} 且不重複。"""
    v3 = _v3_paths(_room_input(tmp_path))
    names = [p.wall for p in v3]
    assert set(names) == _EXPECTED_WALLS, f"牆名集合 {set(names)!r} ≠ {_EXPECTED_WALLS!r}"
    assert len(names) == len(_EXPECTED_WALLS), "牆名有重複或缺"


def test_moving_receiver_changes_at_least_one_dist(tmp_path: Path) -> None:
    """接收點移一格，至少一條距離 hex 變。"""
    inp = _room_input(tmp_path)
    baseline = [p.dist_m.hex() for p in _v3_paths(inp)]

    moved = RoomInput(
        room=inp.room,
        source=inp.source,
        receiver=Point(inp.receiver.x + 1.0, inp.receiver.y, inp.receiver.z),
        sound_speed=inp.sound_speed,
        max_order=inp.max_order,
    )
    changed = [p.dist_m.hex() for p in _v3_paths(moved)]

    differs = [b for b, c in zip(baseline, changed) if b != c]
    assert differs, "接收點移一格後沒有一條 dist hex 變"


# ── 反射點（refl_pt／t／in_wall） ────────────────────────────────────────────


def test_all_seven_paths_in_wall_true_and_t_in_open_interval(tmp_path: Path) -> None:
    """七條路徑的 ``in_wall`` 全為 True、``t`` 在 (0,1)；直達路徑兩者皆 None。"""
    v3 = _v3_paths(_room_input(tmp_path))
    for p in v3:
        if p.wall == "direct":
            assert p.refl_pt is None and p.t is None and p.in_wall is None
        else:
            assert p.in_wall is True, f"{p.wall} 的反射點不在牆上"
            assert p.refl_pt is not None, f"{p.wall} 缺反射點"
            assert p.t is not None and 0.0 < p.t < 1.0, f"{p.wall} 的 t={p.t!r} 不在 (0,1)"


def test_reflection_points_match_independent_geometry(tmp_path: Path) -> None:
    """用 blueprint.reference_room_geometry 重算反射點，逐位比 t 與反射點座標 hex。"""
    from blueprint import reference_room_geometry as geom

    v3 = _v3_paths(_room_input(tmp_path))
    room = geom.Room(lx=6.0, ly=4.0, lz=3.0, c=343.0)
    src = (1.5, 1.0, 1.2)
    recv = (4.0, 3.0, 1.5)

    for p in v3:
        if p.wall == "direct":
            continue
        actual = geom.wall_reflection(room, p.wall, src, recv)
        assert actual.refl_pt is not None, f"{p.wall} 獨立幾何沒算出反射點"
        assert p.refl_pt is not None, f"{p.wall} v3 沒算出反射點"
        assert actual.t is not None and p.t is not None
        assert p.t.hex() == actual.t.hex(), f"{p.wall} 的 t hex 不同"
        for axis_index, axis_label in enumerate(("x", "y", "z")):
            assert (
                p.refl_pt[axis_index].hex() == actual.refl_pt[axis_index].hex()
            ), f"{p.wall} 反射點 {axis_label} hex 不同"


# ── 比對要咬得住（逐筆刪、identity 改一位、缺一條報牆名）────────────────────────


def test_compare_all_same_for_frozen(tmp_path: Path) -> None:
    """凍結參數整份比對，回空差異清單。"""
    v3 = _v3_paths(_room_input(tmp_path))
    answers = _frozen_answer_paths()
    results = compare_paths(v3, answers)
    assert all(not r.diffs for r in results), f"有差異：{[r for r in results if r.diffs]}"


def test_compare_reports_a_missing_path(tmp_path: Path) -> None:
    """把 v3 的七條少一條餵給比對，要報條數不同，並指名少了哪一條（牆名）。"""
    v3 = _v3_paths(_room_input(tmp_path))
    answers = _frozen_answer_paths()
    results = compare_paths(v3, answers)
    assert all(not r.diffs for r in results)

    # 少最後一條（xL）。
    truncated = compare_paths(v3[:-1], answers)
    missing = [r for r in truncated if r.diffs]
    assert missing, "少一條沒被抓到"
    assert any("少了這一條" in d for r in missing for d in r.diffs), "沒指名少了哪一條"


def test_compare_reports_identity_flipped(tmp_path: Path) -> None:
    """identity 改一位，比對要報 identity 不同。"""
    v3 = _v3_paths(_room_input(tmp_path))
    answers = _frozen_answer_paths()

    victim = v3[0]
    flipped = list(victim.identity)
    flipped[1] = -flipped[1]
    flipped_identity = (flipped[0], flipped[1], flipped[2], flipped[3], flipped[4], flipped[5])
    tampered = replace(victim, identity=flipped_identity)
    tampered_list = [tampered, *v3[1:]]

    results = compare_paths(tampered_list, answers)
    assert any(r.diffs for r in results), "identity 改一位沒被抓到"


def test_compare_reports_a_wall_name_mismatch(tmp_path: Path) -> None:
    """牆名由 identity 推出來：改 identity 使得牆名變成另一面牆，比對要報牆名不同。"""
    v3 = _v3_paths(_room_input(tmp_path))
    answers = _frozen_answer_paths()

    # x0 的 identity 是 (0,-1,0,1,0,1)；把它改成 y0 的 (0,1,0,-1,0,1)，牆名就不同。
    # 但直達改會連帶 order 一起亂，這裡拿 index 1（x0）改。
    victim = v3[1]
    tampered = replace(victim, identity=(0, 1, 0, -1, 0, 1))
    tampered_list = [v3[0], tampered, *v3[2:]]

    results = compare_paths(tampered_list, answers)
    assert any("牆名不同" in d for r in results for d in r.diffs), "牆名改掉沒被抓到"


# ── 載入器 ──────────────────────────────────────────────────────────────────


def test_loader_reads_max_order_and_passes_down(tmp_path: Path) -> None:
    """max_order 從輸入檔讀進來，餵給 image_source_paths。"""
    inp = _room_input(tmp_path)
    assert inp.max_order == 1
    # 直接呼叫 image_source_paths 用讀到的 max_order，等於七條。
    assert len(_v3_paths(inp)) == len(_frozen_answer_paths())


@pytest.mark.parametrize("bad_order", [0, 2, -1])
def test_image_source_paths_rejects_unsupported_max_order(tmp_path: Path, bad_order: int) -> None:
    """max_order 不是 1（0、2、-1）就 NotImplementedError。"""
    inp = _room_input(tmp_path)
    with pytest.raises(NotImplementedError):
        image_source_paths(inp.room, inp.source, inp.receiver, inp.sound_speed, bad_order)


def test_loader_rejects_missing_field(tmp_path: Path) -> None:
    """缺欄位（sound_speed_m_s）要丟 ValueError，訊息講「缺欄位」不是「不是一個數」。"""
    params = _input_params()
    params.pop("sound_speed_m_s")
    with pytest.raises(ValueError, match="缺欄位"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_missing_xyz_axis(tmp_path: Path) -> None:
    """缺 source_xyz_m.z 要講「缺欄位」，不是「不是一個數」。"""
    params = _input_params()
    source = params["source_xyz_m"]
    if isinstance(source, dict):
        source.pop("z")
    with pytest.raises(ValueError, match="缺欄位"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_negative(tmp_path: Path) -> None:
    """負數（room.Lz_m）要丟 ValueError 指名哪一格。"""
    params = _input_params()
    room = params["room"]
    if isinstance(room, dict):
        room["Lz_m"] = -3.0
    with pytest.raises(ValueError, match="Lz_m"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_zero_sound_speed(tmp_path: Path) -> None:
    """聲速 0 要丟 ValueError 指名哪一格。"""
    params = _input_params()
    params["sound_speed_m_s"] = 0.0
    with pytest.raises(ValueError, match="sound_speed_m_s"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_zero_room_side(tmp_path: Path) -> None:
    """房間三邊其中一邊 0 要丟 ValueError 指名哪一格。"""
    params = _input_params()
    room = params["room"]
    if isinstance(room, dict):
        room["Lx_m"] = 0.0
    with pytest.raises(ValueError, match="Lx_m"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_extra_field(tmp_path: Path) -> None:
    """多餘欄位（寫錯字）要丟 ValueError，並列出多的欄位名。"""
    params = _input_params()
    params["sound_speed_ms"] = 343.0  # 寫錯字：多打一個 s 掉（正確是 sound_speed_m_s）
    with pytest.raises(ValueError, match="sound_speed_ms"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_missing_path(tmp_path: Path) -> None:
    """路徑不存在要炸（OSError 的那一類）。"""
    with pytest.raises((ValueError, OSError, FileNotFoundError)):
        load_room_input(tmp_path / "does_not_exist.json")


# ── 命令列 ──────────────────────────────────────────────────────────────────


def _write_answer(tmp_path: Path, mutate: bool, name: str) -> Path:
    """把答案檔整份抄到 tmp_path（選擇性弄壞一格 hex）。"""
    with ANSWER_PATH.open(encoding="utf-8") as handle:
        root = json.load(handle)
    if mutate:
        paths = root["paths"]
        if isinstance(paths, list) and paths:
            first = paths[0]
            if isinstance(first, dict):
                raw = first["dist_m"]["hex"]
                bumped = math.nextafter(float.fromhex(raw), math.inf)
                first["dist_m"]["hex"] = bumped.hex()
    path = tmp_path / name
    path.write_text(json.dumps(root, sort_keys=True), encoding="utf-8")
    return path


def test_cli_compare_frozen_exit_zero(tmp_path: Path) -> None:
    """--compare 對凍結參數離開碼 0。"""
    input_path = _write_input(tmp_path, _input_params())
    exit_code = main([str(input_path), "--compare", str(ANSWER_PATH)])
    assert exit_code == 0


def test_cli_compare_tampered_exit_one(tmp_path: Path) -> None:
    """--compare 對一份改過 hex 的答案檔副本離開碼 1。"""
    input_path = _write_input(tmp_path, _input_params())
    tampered = _write_answer(tmp_path, mutate=True, name="tampered.json")
    exit_code = main([str(input_path), "--compare", str(tampered)])
    assert exit_code == 1


def test_cli_missing_input_exit_two(tmp_path: Path) -> None:
    """讀不到輸入檔離開碼 2。"""
    exit_code = main([str(tmp_path / "nope.json")])
    assert exit_code == 2


def test_cli_zero_sound_speed_exit_two(tmp_path: Path) -> None:
    """聲速 0 的輸入檔 → 離開碼 2（ValueError → main 裡回 2）。"""
    params = _input_params()
    params["sound_speed_m_s"] = 0.0
    input_path = _write_input(tmp_path, params)
    exit_code = main([str(input_path)])
    assert exit_code == 2
