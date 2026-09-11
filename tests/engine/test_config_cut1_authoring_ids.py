"""``aosr.config.authoring_defaults`` 三支 id 助手的裁判。

v2 有 17 支考卷用到這三支函式，但它們全部只當**輸入**（拿回來的 id 字串再塞進別的層），
沒有任何一支把回傳值寫死。所以裁判改由 donor 標準答案接手：正常輸入、空字串、
帶負數的補丁座標各餵一組，比對回傳字串。

判定與殘餘風險見 ``tests/engine/test_config_cut1_table``。這一支把答案檔那幾筆輸入真的
餵進去，不是另外抄一份期望值。
"""
from __future__ import annotations

import aosr.config.authoring_defaults as authoring

from tests.engine._config_answers import module_answers, probe_args, probe_value


def _field(record: dict[str, object], key: str) -> dict[str, object]:
    """答案檔裡一筆探針的某一格（收窄成表；形狀不對就當場炸）。"""
    value = record[key]
    if not isinstance(value, dict):
        raise AssertionError(f"答案檔這一筆的 {key} 不是表：{value!r}")
    return {str(name): item for name, item in value.items()}


def _records(*keys: str) -> list[dict[str, object]]:
    """答案檔裡參數名完全等於 ``keys`` 的那幾筆探針。"""
    wanted = set(keys)
    probes = module_answers("authoring_defaults")["probes"]
    if not isinstance(probes, list):
        raise AssertionError("答案檔的 authoring_defaults 探針不是一串東西")
    records: list[dict[str, object]] = []
    for item in probes:
        if not isinstance(item, dict):
            raise AssertionError(f"答案檔有一筆探針不是表：{item!r}")
        args = item.get("args")
        if isinstance(args, dict) and set(args) == wanted:
            records.append({str(key): value for key, value in item.items()})
    return records


def _table(record: object, where: str) -> dict[str, object]:
    if not isinstance(record, dict):
        raise AssertionError(f"{where} 不是表：{record!r}")
    return {str(name): item for name, item in record.items()}


def _check(records: list[dict[str, object]], key: str) -> None:
    """每一筆探針都真的餵進去，比對 donor 的回傳字串。

    ``key`` 是答案檔那個模組底下的哪一支（這一支考卷的三支助手簽章各不相同，
    所以在這裡分派，不用一個泛型回呼把它們抹平）。
    """
    assert records, "答案檔裡沒有這一支的探針——這一支就沒有對象"
    for record in records:
        expected = probe_value(_field(record, "expected"))
        actual = _call(key, probe_args(_table(record, "答案檔這一筆的探針")))
        assert actual == expected, f"這一組輸入的結果跟 donor 不一樣：{record!r}"


def _call(key: str, args: dict[str, object]) -> object:
    """把一筆探針的參數餵給那一支助手（參數名與簽章都是固定的，這裡逐支寫出來）。"""
    if key == "rhino_boundary_id":
        return authoring.rhino_boundary_id(str(args["guid"]))
    wall = str(args["wall_id"])
    if key == "app_wall_id":
        return authoring.app_wall_id(wall)
    if key == "app_patch_id":
        row = args["row"]
        col = args["col"]
        assert isinstance(row, int) and isinstance(col, int)
        return authoring.app_patch_id(wall, row, col)
    raise AssertionError(f"不認識的助手：{key}")


def test_wall_id_helper_matches_donor() -> None:
    """``app_wall_id``：正常牆名、另一個牆名、空字串。"""
    _check(_records("wall_id"), "app_wall_id")


def test_patch_id_helper_matches_donor() -> None:
    """``app_patch_id``：正常座標、原點、帶負數的列。"""
    _check(_records("wall_id", "row", "col"), "app_patch_id")


def test_rhino_boundary_id_helper_matches_donor() -> None:
    """``rhino_boundary_id``：一般 guid、空字串、沒有破折號的字串。"""
    _check(_records("guid"), "rhino_boundary_id")


def test_id_prefixes_really_are_what_the_helpers_use() -> None:
    """三個前綴常數真的是那三支助手做進字串裡的東西（接線，不只是各自凍結）。

    這一條不抄前綴的字面值：那三個字面值已經在凍結常數那一支比過了，這裡比的是
    「助手有沒有用那個常數」——只比常數各自沒變，蓋不到「助手改用自己的字面值」。
    """
    assert authoring.app_wall_id("y0") == f"{authoring.APP_WALL_PREFIX}y0"
    assert authoring.app_patch_id("y0", 1, 2) == f"{authoring.APP_PATCH_PREFIX}y0:r1c2"
    assert authoring.rhino_boundary_id("g") == f"{authoring.RHINO_PREFIX}g"
