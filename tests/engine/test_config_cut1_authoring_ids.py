"""``aosr.config.authoring_defaults`` 三支 id 助手的裁判。

v2 有 17 支考卷用到這三支函式，但它們全部只當**輸入**（拿回來的 id 字串再塞進別的層），
沒有任何一支把回傳值寫死。所以裁判改由 donor 標準答案接手：正常輸入、空字串、
帶負數的補丁座標各餵一組，比對回傳字串。

哪一筆屬於哪一支助手由 case 表（``blueprint/config_cut1_cases.py``）的 id 決定
（``authoring_defaults.app_wall_id.…``／``….app_patch_id.…``／``….rhino_boundary_id.…``）。
判定與殘餘風險見 ``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。
"""
from __future__ import annotations

import aosr.config.authoring_defaults as authoring

from tests.engine._config_answers import (
    case_args,
    expected_block,
    probe_by_id,
    probe_ids,
    probe_value,
)


def _ids(helper: str) -> list[str]:
    return sorted(case_id for case_id in probe_ids("authoring_defaults") if f".{helper}." in case_id)


def _check(helper: str, key: str) -> None:
    """每一筆 case 都真的餵進去，比對 donor 的回傳字串。

    ``key`` 是三支助手之一（它們的簽章各不相同，所以在這裡分派，不用一個泛型回呼
    把它們抹平）。
    """
    ids = _ids(helper)
    assert ids, f"答案檔裡沒有 {helper} 的 case——這一支就沒有對象"
    for case_id in ids:
        case = probe_by_id(case_id)
        args = case_args(case)
        expected = probe_value(expected_block(case))
        actual = _call(key, args)
        assert actual == expected, f"這一組輸入的結果跟 donor 不一樣：{case_id}"


def _call(key: str, args: dict[str, object]) -> object:
    """把一筆 case 的參數餵給那一支助手（參數名與簽章都是固定的，這裡逐支寫出來）。"""
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
    _check("app_wall_id", "app_wall_id")


def test_patch_id_helper_matches_donor() -> None:
    """``app_patch_id``：正常座標、原點、帶負數的列。"""
    _check("app_patch_id", "app_patch_id")


def test_rhino_boundary_id_helper_matches_donor() -> None:
    """``rhino_boundary_id``：一般 guid、空字串、沒有破折號的字串。"""
    _check("rhino_boundary_id", "rhino_boundary_id")


def test_id_prefixes_really_are_what_the_helpers_use() -> None:
    """三個前綴常數真的是那三支助手做進字串裡的東西（接線，不只是各自凍結）。

    這一條不抄前綴的字面值：那三個字面值已經在凍結常數那一支比過了，這裡比的是
    「助手有沒有用那個常數」——只比常數各自沒變，蓋不到「助手改用自己的字面值」。
    """
    assert authoring.app_wall_id("y0") == f"{authoring.APP_WALL_PREFIX}y0"
    assert authoring.app_patch_id("y0", 1, 2) == f"{authoring.APP_PATCH_PREFIX}y0:r1c2"
    assert authoring.rhino_boundary_id("g") == f"{authoring.RHINO_PREFIX}g"
