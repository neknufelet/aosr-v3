"""票 #175 的考卷：讀答案檔，用獨立幾何逐格比對七條 shoebox 反射路徑。

**這支考卷不 import ``aosr``。** 它住的籃子是 ``tests/``（治理層的考卷，見規矩卡
``green-must-be-real-green``）：它驗的是「產生器跑出來的凍結答案」跟「只 import 標準庫的
獨立幾何重算」兩邊是否逐格一致，跟新家引擎無關。

**跟產生器走同一套形狀、但算式完全獨立。** 產生器（``blueprint/generate_reference_room_answers.py``）
在唯讀的 donor 工作樹上跑上一代的 ``precompute_shoebox_patch_paths`` 寫出答案檔；
這一支用 ``blueprint/reference_room_geometry.py``（只 import ``math``／``dataclasses``）
把每一條路徑重新算一遍，斷言兩邊的 ``hex`` **字串相等**（不是 isclose）——hex 是
``float.hex()`` 的精確表示，跨平台不漂，正合「數值要跟上一代一致」這張票的合約。鏡像座標
也一樣比 ``hex`` 字串，不是比 float() 的浮點。

**逐格比對的回傳不是布林。** :func:`_compare_one` 回傳差異清單（空清單＝完全相同），
布林在一題裡說不出是哪一格壞了；控制組直接證明這個裁判真的咬得住——把某一條的
``dist_m.hex``（或 ``delay_s``、鏡像座標、反射點的 ``t``）弄壞再餵同一支比對函式，
必須報出那格不同。另有整檔控制組：把一整份刻意弄壞的答案檔餵進整條讀取＋比對，
斷言每種壞法都被抓到。
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from blueprint import reference_room_geometry as geom
from blueprint.generate_reference_room_answers import (
    ANSWER_SCHEMA,
    CANONICAL_WALL_ORDER,
    FROZEN_PARAMETERS,
)

# 答案檔的位置：從這一支往上兩層是 repo 根，答案檔住在 blueprint 底下。
ANSWER_PATH: Path = Path(__file__).resolve().parents[1] / "blueprint" / "reference_room_answers.json"

# donor commit 的形狀：40 位小寫十六進位（git 的 sha-1）。用正則逐字比對是「形狀比對」，
# 不是把數量鎖死成一個整數（規矩卡 assertions-not-pinned-to-counts）。
_COMMIT_PATTERN: re.Pattern[str] = re.compile(r"[0-9a-f]{40}\Z")

# 六面牆 + 直達，共七條路徑的名字。這份名單也是獨立幾何聲明的規範順序——
# 逐項具名比對（集合相等），不把數量鎖死成一個數字（規矩卡 assertions-not-pinned-to-counts）。
_EXPECTED_NAMES: frozenset[str] = frozenset(geom.canonical_walls()) | {"direct"}

# identity 六元組的長度：每軸一組（次數、正負號），共 2 * 軸數——軸數由獨立幾何供應，
# 不寫死 6（規矩卡 assertions-not-pinned-to-counts）。
_IDENTITY_LEN: int = 2 * geom.NUM_AXES

# 檔頭必填欄位（缺一格這份答案檔就不算證據，照 tests/engine/_materials_answers.py 的同一套）。
_REQUIRED_DONOR_KEYS: tuple[str, ...] = ("tag", "commit", "clean")
_REQUIRED_ENV_KEYS: tuple[str, ...] = ("versions", "backend", "x64", "python", "platform")
_REQUIRED_VERSION_KEYS: tuple[str, ...] = ("jax", "jaxlib", "numpy", "flax")
_REQUIRED_COMMAND_KEYS: tuple[str, ...] = ("interpreter", "argv", "env")
_REQUIRED_COMMAND_ENV_KEYS: tuple[str, ...] = (
    "PYTHONPATH",
    "JAX_PLATFORMS",
    "PYTHONDONTWRITEBYTECODE",
)


@dataclass(frozen=True)
class _Path:
    """答案檔裡一條路徑的收窄形狀（浮點從 hex 逐位元還原；hex 字串留著比對用）。"""

    index: int
    order: int
    identity: tuple[int, int, int, int, int, int]
    image_hex: tuple[str, str, str]
    image: tuple[float, float, float]
    dist_hex: str
    delay_hex: str
    dist_m: float
    delay_s: float


@dataclass(frozen=True)
class _Inputs:
    """餵給獨立幾何的輸入：房間（尺寸＋聲速）與聲源、接收點。"""

    room: geom.Room
    src: tuple[float, float, float]
    recv: tuple[float, float, float]


def _as_mapping(node: object, where: str) -> dict[str, object]:
    """把一個節點收窄成一層表。不是表就當場炸——沒看懂就不出結論。"""
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是一層表：{node!r}")
    return {str(key): value for key, value in node.items()}


def _as_xyz(node: object, where: str) -> tuple[tuple[str, str, str], tuple[float, float, float]]:
    """答案檔 ``image_xyz`` 那一格：三個軸的 hex 字串與還原的浮點三元組。"""
    table = _as_mapping(node, where)
    hexes: list[str] = []
    values: list[float] = []
    for axis in ("x", "y", "z"):
        cell = _as_mapping(table.get(axis), f"{where}.{axis}")
        raw = cell.get("hex")
        if not isinstance(raw, str):
            raise AssertionError(f"{where}.{axis} 沒有 hex 那一格")
        hexes.append(raw)
        values.append(float.fromhex(raw))
    return (hexes[0], hexes[1], hexes[2]), (values[0], values[1], values[2])


def _as_float_hex(node: object, where: str) -> tuple[str, float]:
    """答案檔 ``dist_m``／``delay_s`` 那一格：hex 字串與還原的浮點。"""
    cell = _as_mapping(node, where)
    raw = cell.get("hex")
    if not isinstance(raw, str):
        raise AssertionError(f"{where} 沒有 hex 那一格")
    return raw, float.fromhex(raw)


def _as_int(node: object, where: str) -> int:
    """把 json 讀出來的一個節點收窄成 int（不是整數就當場炸）。"""
    if isinstance(node, bool) or not isinstance(node, int):
        raise AssertionError(f"{where} 不是整數：{node!r}")
    return node


def _as_number(node: object, where: str) -> float:
    """把 json 讀出來的一個節點收窄成 float（不是數就當場炸）。"""
    if not isinstance(node, (int, float)) or isinstance(node, bool):
        raise AssertionError(f"{where} 不是數：{node!r}")
    return float(node)


def _as_text(node: object, where: str) -> str:
    """把 json 讀出來的一個節點收窄成 str。"""
    if not isinstance(node, str):
        raise AssertionError(f"{where} 不是字串：{node!r}")
    return node


def _as_xyz_value(params: dict[str, object], key: str) -> tuple[float, float, float]:
    """``source_xyz_m``／``receiver_xyz_m`` 那一格：三個軸的浮點三元組。"""
    table = _as_mapping(params.get(key), key)
    return (
        _as_number(table.get("x"), f"{key}.x"),
        _as_number(table.get("y"), f"{key}.y"),
        _as_number(table.get("z"), f"{key}.z"),
    )


def _inputs_from_params(params: dict[str, object]) -> _Inputs:
    """從答案檔的 ``parameters`` 讀出餵給獨立幾何的輸入（房間、聲源、接收點）。"""
    room = _as_mapping(params.get("room"), "room")
    return _Inputs(
        room=geom.Room(
            lx=_as_number(room.get("Lx_m"), "room.Lx_m"),
            ly=_as_number(room.get("Ly_m"), "room.Ly_m"),
            lz=_as_number(room.get("Lz_m"), "room.Lz_m"),
            c=_as_number(params.get("sound_speed_m_s"), "sound_speed_m_s"),
        ),
        src=_as_xyz_value(params, "source_xyz_m"),
        recv=_as_xyz_value(params, "receiver_xyz_m"),
    )


def _validate_header(root: dict[str, object], source_name: str) -> None:
    """檔頭必填欄位：schema、donor（tag/commit/clean）、env（versions/backend/x64/python/platform）。"""
    if root.get("schema") != ANSWER_SCHEMA:
        raise AssertionError(
            f"{source_name} 的 schema 是 {root.get('schema')!r}，不是 {ANSWER_SCHEMA}"
        )

    donor = _as_mapping(root.get("donor"), f"{source_name} 的 donor 檔頭")
    for key in _REQUIRED_DONOR_KEYS:
        if not donor.get(key):
            raise AssertionError(f"{source_name} 的 donor 檔頭少了 {key}")
    if donor.get("clean") is not True:
        raise AssertionError(
            f"{source_name} 的 donor 檔頭沒有 clean=true"
            "——那代表它是在一棵被改過的樹上跑出來的，不是上一代的值"
        )
    if donor.get("tag") != "v3-donor":
        raise AssertionError(f"{source_name} 的 donor 檔頭 tag 不是 'v3-donor'")

    env = _as_mapping(root.get("env"), f"{source_name} 的 env 檔頭")
    for key in _REQUIRED_ENV_KEYS:
        if key not in env:
            raise AssertionError(
                f"{source_name} 的 env 檔頭少了 {key}"
                "——dtype 跟著 x64 開關與函式庫版本走，沒記就比不出兩邊是不是同一組設定"
            )
    versions = _as_mapping(env.get("versions"), f"{source_name} 的 env.versions")
    for key in _REQUIRED_VERSION_KEYS:
        if not versions.get(key):
            raise AssertionError(f"{source_name} 的 env.versions 少了 {key}")


def _load_path(path: Path | None = None) -> tuple[dict[str, object], list[_Path], _Inputs]:
    """讀一份答案檔，驗證檔頭，把 ``paths`` 收窄成 :class:`_Path` 清單，並抽出幾何輸入。

    ``path`` 是整檔控制組的入口（把一份刻意的壞答案檔餵進來，證明裁判真的咬得住）；
    正規答案檔走 ``None`` 就讀 ``ANSWER_PATH``。
    """
    source = ANSWER_PATH if path is None else path
    with source.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _as_mapping(data, source.name)

    _validate_header(root, source.name)

    params = _as_mapping(root.get("parameters"), f"{source.name} 的 parameters")
    inputs = _inputs_from_params(params)

    raw_paths = root.get("paths")
    if not isinstance(raw_paths, list):
        raise AssertionError(f"{source.name} 的 paths 不是一串東西")
    paths: list[_Path] = []
    for position, raw in enumerate(raw_paths):
        entry = _as_mapping(raw, "paths 的一筆")
        raw_identity = entry.get("identity")
        if not isinstance(raw_identity, list) or len(raw_identity) != _IDENTITY_LEN:
            raise AssertionError(
                f"identity 不是 {_IDENTITY_LEN} 元組（2×{geom.NUM_AXES} 軸）：{raw_identity!r}"
            )
        identity = (
            _as_int(raw_identity[0], "identity[0]"),
            _as_int(raw_identity[1], "identity[1]"),
            _as_int(raw_identity[2], "identity[2]"),
            _as_int(raw_identity[3], "identity[3]"),
            _as_int(raw_identity[4], "identity[4]"),
            _as_int(raw_identity[5], "identity[5]"),
        )
        dist_hex, dist_m = _as_float_hex(entry.get("dist_m"), "dist_m")
        delay_hex, delay_s = _as_float_hex(entry.get("delay_s"), "delay_s")
        image_hex, image = _as_xyz(entry.get("image_xyz"), "image_xyz")

        index = _as_int(entry.get("index"), "index")
        # 檔案的 index 必須等於它在 paths 清單裡的位置（0, 1, 2, …），不是拿寫死數字比。
        if index != position:
            raise AssertionError(f"paths[{position}] 的 index 是 {index}，不是位置 {position}")

        order = _as_int(entry.get("order"), "order")
        # order 必須等於獨立幾何從 identity 算出來的階數（直達 0、一次反射 1）。
        expected_order = geom.order_of(identity)
        if order != expected_order:
            raise AssertionError(
                f"paths[{position}] 的 order 是 {order}，獨立幾何算出來是 {expected_order}"
            )

        paths.append(
            _Path(
                index=index,
                order=order,
                identity=identity,
                image_hex=image_hex,
                image=image,
                dist_hex=dist_hex,
                delay_hex=delay_hex,
                dist_m=dist_m,
                delay_s=delay_s,
            )
        )
    return root, paths, inputs


def _compare_one(path: _Path, inputs: _Inputs) -> list[str]:
    """用獨立幾何重建一條路徑，逐格對答案檔，回傳差異清單（空＝完全相同）。

    ``inputs``（房間、聲源、接收點）由呼叫端讀一次傳來，這裡**不重讀檔**——
    每條路徑重開一次答案檔既是浪費，也會讓「比對的那份」跟「載入的那份」可以是兩份不同的檔。
    """
    name = geom.decode_identity(path.identity)
    if name is None:
        return [f"identity {path.identity!r} 不是 order 0/1，看不懂牆名"]

    actual = (
        geom.direct_reflection(inputs.room, inputs.src, inputs.recv)
        if name == "direct"
        else geom.wall_reflection(inputs.room, name, inputs.src, inputs.recv)
    )

    diffs: list[str] = []
    actual_image_hex = (actual.image[0].hex(), actual.image[1].hex(), actual.image[2].hex())
    if actual_image_hex != path.image_hex:
        diffs.append(
            f"identity {path.identity!r}（{name}）鏡像坐標 hex：幾何算 {actual_image_hex!r}，"
            f"答案檔是 {path.image_hex!r}"
        )
    if actual.dist_m.hex() != path.dist_hex:
        diffs.append(
            f"identity {path.identity!r}（{name}）dist_m.hex：幾何算 {actual.dist_m.hex()!r}，"
            f"答案檔是 {path.dist_hex!r}"
        )
    if actual.delay_s.hex() != path.delay_hex:
        diffs.append(
            f"identity {path.identity!r}（{name}）delay_s.hex：幾何算 {actual.delay_s.hex()!r}，"
            f"答案檔是 {path.delay_hex!r}"
        )
    if name != "direct":
        if actual.t is None or not (0.0 < actual.t < 1.0):
            diffs.append(f"{name} 的反射點 t={actual.t!r} 不在 (0,1)")
        if actual.in_wall is not True:
            diffs.append(f"{name} 的反射點 ({actual.refl_pt!r}) 不在牆面內")
    return diffs


def _parameter_diffs(root: dict[str, object]) -> list[str]:
    """答案檔的 ``parameters`` 跟產生器的 ``FROZEN_PARAMETERS`` 逐格比，回差異清單。"""
    params = _as_mapping(root.get("parameters"), "parameters")
    diffs: list[str] = []
    keys = sorted(set(FROZEN_PARAMETERS) | set(params))
    for key in keys:
        expected = FROZEN_PARAMETERS.get(key)
        actual = params.get(key)
        if actual != expected:
            diffs.append(f"parameters.{key}：答案檔是 {actual!r}，凍結常數是 {expected!r}")
    return diffs


def _wall_name_diffs(paths: list[_Path]) -> list[str]:
    """每條 identity 推出的牆名集合要等於 {direct, x0, xL, y0, yL, floor, ceiling} 且不重複。"""
    names = [geom.decode_identity(path.identity) for path in paths]
    diffs: list[str] = []
    if set(names) != _EXPECTED_NAMES:
        diffs.append(f"牆名集合 {set(names)!r} ≠ {_EXPECTED_NAMES!r}（可能少一條或多條）")
    # 集合相等看不出「同一個名字兩筆」；再比一次「名字個數 = 期望名字個數」。
    if len(names) != len(_EXPECTED_NAMES):
        diffs.append(f"牆名有 {len(names)} 個，期望 {len(_EXPECTED_NAMES)} 個（有重複或缺）")
    return diffs


def _validate_and_compare(path: Path) -> list[str]:
    """整條讀取＋比對：檔頭欄位、index/order、parameters、牆名、每一條路徑。回全部差異。

    檔頭缺格、index/order 錯、identity 長度錯這些「結構壞」是當場 raise；數值不一致
    進差異清單。整檔控制組把一份刻意弄壞的檔餵進來，這一支要嘛 raise、要嘛回非空清單。
    """
    root, paths, inputs = _load_path(path)
    diffs: list[str] = list(_parameter_diffs(root))
    diffs.extend(_wall_name_diffs(paths))
    for entry in paths:
        diffs.extend(_compare_one(entry, inputs))
    return diffs


def _bump_ulp(value: float) -> tuple[str, float]:
    """一個浮點 +1 ULP（尾數最後一位），回傳新的 hex 字串與浮點。"""
    bumped = math.nextafter(value, math.inf)
    return bumped.hex(), bumped


def _flip_dist(path: _Path) -> _Path:
    """把 dist_m 翻 1 ULP。"""
    hexed, bumped = _bump_ulp(path.dist_m)
    return replace(path, dist_m=bumped, dist_hex=hexed)


def _flip_delay(path: _Path) -> _Path:
    """把 delay_s 翻 1 ULP。"""
    hexed, bumped = _bump_ulp(path.delay_s)
    return replace(path, delay_s=bumped, delay_hex=hexed)


def _flip_image(path: _Path) -> _Path:
    """把鏡像坐標的 x 分量翻 1 ULP。"""
    hexed, bumped = _bump_ulp(path.image[0])
    image = (bumped, path.image[1], path.image[2])
    image_hex = (hexed, path.image_hex[1], path.image_hex[2])
    return replace(path, image=image, image_hex=image_hex)


def _receiver_off_wall(inputs: _Inputs) -> _Inputs:
    """把接收點換到鏡像那一側（房間外），讓反射點的 t 掉出 (0,1)，弄壞反射點。"""
    recv = (-2.0, 3.0, 1.5)
    return replace(inputs, recv=recv)


# ── 整檔控制組：把一份刻意弄壞的答案檔寫進 tmp_path ──────────────────────────────


def _write_tampered(
    tmp_path: Path, mutate: Callable[[dict[str, object]], None], name: str
) -> Path:
    """讀正本答案檔、套一份動手腳、寫進 tmp_path，回傳那份壞檔的路徑。"""
    with ANSWER_PATH.open(encoding="utf-8") as handle:
        root = json.load(handle)
    mutate(root)
    out = tmp_path / name
    out.write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _tamper_drop_one(root: dict[str, object]) -> None:
    """少一條路徑（把最後一條砍掉）。"""
    paths = root["paths"]
    if isinstance(paths, list):
        paths.pop()


def _tamper_shuffle_index(root: dict[str, object]) -> None:
    """把每一條的 index 都換錯（改到不匹配它自己的位置）。"""
    paths = root["paths"]
    if isinstance(paths, list):
        for position, raw in enumerate(paths):
            raw["index"] = (position + 1) % len(paths)


def _tamper_flip_hex(root: dict[str, object]) -> None:
    """把第一條 dist_m.hex 翻 1 ULP。"""
    paths = root["paths"]
    if isinstance(paths, list):
        first = paths[0]
        raw = first["dist_m"]["hex"]
        bumped = math.nextafter(float.fromhex(raw), math.inf)
        first["dist_m"]["hex"] = bumped.hex()


def _tamper_parameters(root: dict[str, object]) -> None:
    """把 parameters 的聲速改掉一格。"""
    params = root.get("parameters")
    if isinstance(params, dict):
        params["sound_speed_m_s"] = 340.0


# ── 考題 ──────────────────────────────────────────────────────────────────────


def test_schema_and_donor_commit() -> None:
    """答案檔 schema 等於產生器登記的那一版，且 donor.commit 是 40 位十六進位字元。"""
    root, _paths, _inputs = _load_path()
    assert root.get("schema") == ANSWER_SCHEMA
    donor = _as_mapping(root.get("donor"), "donor")
    commit = donor.get("commit")
    assert isinstance(commit, str)
    assert _COMMIT_PATTERN.fullmatch(commit) is not None


def test_header_requires_donor_and_env_keys() -> None:
    """檔頭必填欄位齊全：donor 有 tag/commit/clean 且 clean=True、tag=v3-donor；env 全五格與四個版本。"""
    root, _paths, _inputs = _load_path()
    donor = _as_mapping(root.get("donor"), "donor")
    for key in _REQUIRED_DONOR_KEYS:
        assert donor.get(key)
    assert donor.get("clean") is True
    assert donor.get("tag") == "v3-donor"
    env = _as_mapping(root.get("env"), "env")
    for key in _REQUIRED_ENV_KEYS:
        assert key in env
    versions = _as_mapping(env.get("versions"), "env.versions")
    for key in _REQUIRED_VERSION_KEYS:
        assert versions.get(key)


def test_parameters_match_frozen() -> None:
    """答案檔的 parameters 逐格等於產生器檔頭的 FROZEN_PARAMETERS（單一來源）。"""
    root, _paths, _inputs = _load_path()
    assert _parameter_diffs(root) == []


def test_every_path_matches_independent_geometry() -> None:
    """每一條路徑用獨立幾何重算，dist/delay/image 的 hex 逐格字串相等、反射點在牆內。"""
    _root, paths, inputs = _load_path()
    all_diffs: list[str] = []
    for path in paths:
        all_diffs.extend(_compare_one(path, inputs))
    assert all_diffs == []


def test_wall_names_are_the_six_plus_direct_without_duplicates() -> None:
    """每條 identity 推出的牆名集合等於 {direct, x0, xL, y0, yL, floor, ceiling} 且不重複。"""
    _root, paths, _inputs = _load_path()
    assert _wall_name_diffs(paths) == []


def test_canonical_wall_order_matches_convention() -> None:
    """獨立幾何的規範牆順序等於產生器的 CANONICAL_WALL_ORDER，且 convention 文字跟它對得上。"""
    assert geom.canonical_walls() == CANONICAL_WALL_ORDER
    root, _paths, _inputs = _load_path()
    params = _as_mapping(root.get("parameters"), "parameters")
    convention = _as_text(params.get("convention"), "convention")
    assert ", ".join(CANONICAL_WALL_ORDER) in convention


def test_command_records_rerun_info() -> None:
    """command 欄記的是可重跑資訊：直譯器、argv、三個環境變數（沒設就是 null）。"""
    root, _paths, _inputs = _load_path()
    command = _as_mapping(root.get("command"), "command")
    for key in _REQUIRED_COMMAND_KEYS:
        if key not in command:
            raise AssertionError(f"command 檔頭少了 {key}")
    interpreter = command.get("interpreter")
    assert isinstance(interpreter, str) and interpreter
    argv = command.get("argv")
    assert isinstance(argv, list) and argv
    env = _as_mapping(command.get("env"), "command.env")
    for key in _REQUIRED_COMMAND_ENV_KEYS:
        if key not in env:
            raise AssertionError(f"command.env 少了 {key}")


def test_control_group_flags_each_cell() -> None:
    """控制組：四個比對格各弄壞一次，同一支 _compare_one 都要報出那格（差異非空且指名）。"""
    _root, paths, inputs = _load_path()

    victim = next(p for p in paths if geom.decode_identity(p.identity) != "direct")

    dist_diffs = _compare_one(_flip_dist(victim), inputs)
    assert any("dist_m.hex" in diff for diff in dist_diffs)

    delay_diffs = _compare_one(_flip_delay(victim), inputs)
    assert any("delay_s.hex" in diff for diff in delay_diffs)

    image_diffs = _compare_one(_flip_image(victim), inputs)
    assert any("鏡像坐標" in diff for diff in image_diffs)

    # 把接收點換到鏡像那一側（房間外），讓反射點的 t 掉出 (0,1)。
    t_diffs = _compare_one(victim, _receiver_off_wall(inputs))
    assert any("反射點" in diff for diff in t_diffs)


def test_whole_file_control_group(tmp_path: Path) -> None:
    """整檔控制組：四種壞法各寫一份壞檔，餵整條讀取＋比對，每一種都要被抓到。"""
    cases = [
        _write_tampered(tmp_path, _tamper_drop_one, "drop_one.json"),
        _write_tampered(tmp_path, _tamper_shuffle_index, "shuffle_index.json"),
        _write_tampered(tmp_path, _tamper_flip_hex, "flip_hex.json"),
        _write_tampered(tmp_path, _tamper_parameters, "change_params.json"),
    ]
    for path in cases:
        caught: bool
        try:
            diffs = _validate_and_compare(path)
        except AssertionError:
            caught = True
        else:
            caught = bool(diffs)
        assert caught, f"整檔控制組：這份被動過手腳的檔沒有被抓到：{path.name}"
