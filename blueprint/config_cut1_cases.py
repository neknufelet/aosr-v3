"""票 #127 那 20 支 config 模組的 case 表：**「有哪些 case」的唯一住處**。

**為什麼要有這一支（PR #150 的找碴 F1）。** 原本的形狀是「產生器自己去上一代跑一輪、
考卷逐筆跟答案檔比」——那份答案檔住 ``blueprint/``，而 ``identity-strings-generated`` 與
``refs-and-links-resolve`` 都扣掉那一層，於是**沒有任何機制保證它還有幾筆**：把答案檔的
探針逐筆刪掉，只有極少數幾筆刪了會讓考卷變紅，其餘刪掉整套照樣綠。這一支把「有哪些 case」
搬進版控、當成產生器與考卷**共用的同一份清單**：

* 產生器（``blueprint/generate_config_cut1_answers.py``）只跑**這裡宣告的**參數；
* 考卷（``tests/engine/``）斷言答案檔裡的 case id 集合**等於**這裡宣告的集合
  （少一筆紅、多一筆也紅——比的是具名的集合，不是筆數）。

**兩個前後半。** 第 1 刀（票 #127 前半）那 11 支不讀檔：模組名就是它自己的名字，case
是「拿一組參數呼叫一支純函式」。第 2 刀（後半，讀檔那 9 支）名字帶 ``cut2_`` 前綴，case
多了一格 ``op``：餵哪一個設定檔、要不要先突變它、要不要傳路徑。``engine_module_name()``
把表上的名字換成新家的模組名，產生器與考卷都走它。

**參數怎麼寫。** 值是純量或這幾個小幫手（都是普通 dict，沒有任何 v2 依賴，兩邊都 import
得到）：``flt``（浮點）、``nan``／``plus_inf``／``minus_inf``、``true``／``false``
（布林——有些參數要驗「餵布林會被擋」，所以布林不能跟整數混在一起）、``strs``（字串）、
``items``／``floats``（一串值）。
參數自己的**種類**（哪個參數是 int、哪個是 float）由各支考卷的 ``TypedDict`` 講清楚，
這一張表只管「有哪些 case」。

**這一支不准 import 上一代的任何東西。** 它會被 ``style-guard``／``type-guard``／
``uv-single-entrypoint`` 掃（``blueprint/`` 只被 ``refs-and-links-resolve`` 扣掉）。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final, TypedDict

# 一個參數值就是一個普通 dict（不是自訂類別：答案檔用 json 存，寫成普通 dict 才不必
# 為序列化再補一層 representer）。
Tagged = dict[str, object]


class CaseEntry(TypedDict, total=False):
    """一筆 case：穩定的 id ＋ 那一筆的參數（參數值是 :data:`Tagged` 或純量）。

    ``op`` 只有這一刀（讀檔那 9 支）用得到：第 1 刀那些 case 是「拿一組參數去呼叫一支
    函式」，這一刀多了「餵哪一個檔案、要不要先突變它」。沒寫 ``op`` 就是第 1 刀那種形狀。
    """

    id: str
    args: dict[str, Tagged | int | str]
    op: CaseOp


class ModuleSpec(TypedDict):
    """一支模組在這一張表裡的三件事：對應的檔名、要凍結的常數、要跑的 case。"""

    file: str
    constants: list[str]
    cases: list[CaseEntry]


def flt(value: float) -> Tagged:
    """一個浮點參數。"""
    return {"k": "float", "v": value}


def nan() -> Tagged:
    """``float("nan")``（非有限值的守門要驗它）。"""
    return {"k": "nan"}


def plus_inf() -> Tagged:
    """``float("inf")``。"""
    return {"k": "plus_inf"}


def minus_inf() -> Tagged:
    """``float("-inf")``。"""
    return {"k": "minus_inf"}


def true() -> Tagged:
    """布林 ``True``（跟整數 1 是兩件事：有的參數會特別擋布林）。"""
    return {"k": "bool", "v": True}


def false() -> Tagged:
    """布林 ``False``。"""
    return {"k": "bool", "v": False}


def items(*values: Tagged | int | str) -> Tagged:
    """一串值，還原成 ``list``（目前只有「要列出哪幾個預設」那一筆在用）。

    上游有兩處收一串數字，但**容器不一樣**（`SplOutputConfig` 的欄位宣告是
    ``tuple[float, ...]``），而 pydantic 的訊息會把 ``input_value=[nan]`` 與
    ``input_value=(nan,)`` 講成兩句不同的話——所以「是哪一種容器」在這一張表裡是
    看得見的：數字那一串用 :func:`floats`（tuple），其餘用這一支（list）。
    """
    return {"k": "items", "v": list(values)}


def floats(*values: Tagged | float | int) -> Tagged:
    """一串數字，還原成 ``tuple``（`SplOutputConfig.sensitivity_db` 那一格）。"""
    return {"k": "float_tuple", "v": list(values)}


def strs(value: str) -> Tagged:
    """一個字串參數，**明寫成記號**。

    第 1 刀只有純量與幾個小幫手；這一刀有幾筆（`profile="critical"`、
    `mode="production"`）要說清楚「這是一個字串參數」，所以多這一支。
    """
    return {"k": "str", "v": value}


# ── 這一刀（票 #127 後半，讀檔那 9 支）加的小幫手 ─────────────────────────────


class Mutation(TypedDict):
    """一筆 TOML 突變：把真的設定檔讀進來、把其中一段換掉，餵給載入器。

    兩段都是**原文**，不是行號也不是正則：「原文在不在」本身就是一條下限——原文被改掉
    或搬家，這一筆就產生不出來（不是靜靜地跳過）。
    """

    before: str
    after: str


class CaseOp(TypedDict, total=False):
    """這一刀每一筆 case 的執行方式（第 1 刀那些 case 沒有這一格）。

    * ``fn``：要呼叫哪一支載入器（不給就用 case id 的第二段）。
    * ``module``：要載入哪一支新家模組（不給就用 case id 第一段的 ``cut2_`` 後面那個名字）。
    * ``mutations``：依序套用的突變（見 :class:`Mutation`）；真的設定檔讀進來、改過再餵。
    * ``path``：餵給載入器的路徑（``"default"`` ＝ 不傳路徑、走載入器自己的預設；
      ``"missing"`` ＝ 一個不存在的路徑，驗「檔案不存在」那條分支）。
    * ``kwargs``：其他關鍵字參數（例如 ``profile=``、``mode=``）。
    """

    fn: str
    module: str
    mutations: list[Mutation]
    path: str
    kwargs: dict[str, Tagged | int | str]


# 這一刀新增的 9 支模組（產生器靠這一份名單決定要不要載入新家那一支跑載入器；
# 第 1 刀那 11 支只凍結常數與純呼叫，沒有這一格）。
CUT2_MODULES: Final[tuple[str, ...]] = (
    "calibration",
    "fem_lane",
    "mat_continuation",
    "membrane_opt",
    "perceptual",
    "physics_constants",
    "scoring",
    "scoring_v2",
    "stereo_layout",
)


def engine_module_name(table_name: str) -> str:
    """case 表裡的名字對應到新家的模組名（去掉這一刀加的 ``cut2_`` 前綴）。

    「有哪幾支模組」只寫在這一張表裡，產生器與考卷都走這一支，不各留一份名單。
    """
    return table_name.removeprefix("cut2_")


def missing_path() -> str:
    """「檔案不存在」那幾筆餵進去的路徑。

    刻意是一個**看起來像路徑、但一定不存在**的固定字串（不開暫存檔、不碰真目錄）：
    載入器要炸在開檔那一刻，這條路徑只要不存在就驗得到。產生器與考卷共用這一支，
    所以兩邊餵的是同一個字串；比對訊息時只要把這一段正規化掉。
    """
    return "/tmp/aosr-cut2-missing/nope.toml"


def tmp_root() -> str:
    """這一刀的探針檔寫在哪。

    產生器與考卷都把突變過的 TOML 寫在這裡、都比到同一個固定位置；錯誤訊息裡那一段
    路徑不逐位元比（暫存目錄每一跑都不一樣），兩邊都把它換成 ``<tmp>`` 再比。
    """
    return "/tmp/aosr-cut2-probe"


# 錯誤訊息裡的**絕對路徑**一律換成這個記號再比。理由：暫存目錄每一跑都不一樣，
# 而真的設定檔那一份的絕對路徑**在每一棵 checkout 上也不一樣**（CI 的 runner 跟本機
# 不同）。把「repo 在哪、暫存在哪」抽掉之後，留在訊息裡的就是契約那一段：哪一支載入器、
# 哪一個欄位、哪一種錯。產生器與考卷兩邊共用這一支，不各寫一份。
#
# 這一支換的是**已知的兩個根**（repo 根與暫存根），不是用正則去猜「哪裡像路徑」：
# 猜的那一種會把訊息裡本來就有的相對路徑（`src/aosr/config/data/x.toml`）留下來，
# 於是答案檔在另一棵 checkout 上比不過——那是這一條第一版真的踩到的坑。
NORMALISED_PATH: Final[str] = "<path>"

# 一個**絕對路徑**：從 `/` 開始（前面不可以是斜線或英數字——`file:///x` 那種網址
# 例外，見檔尾），一路吃到最後一個斜線（含）。故意貪心——多收一段只是把「在哪一棵樹、
# 哪一個暫存目錄」收掉，留下來的尾段（`scoring_v2.toml`）才是契約。
_ABSOLUTE_PREFIX = re.compile(r"(?<![\w/])/(?:[^\s'\":,()\[\]]*/)+")


def normalise_paths(message: str) -> str:
    """把訊息裡的**絕對路徑前綴**換成 ``<path>/``。

    這是訊息本身的純函式，不吃「這一跑在哪一棵樹跑」這種外部狀態——產生器與考卷在
    不同的 checkout、不同的暫存目錄上跑，套完這一支之後留下的字串逐位元相同。
    相對路徑（`src/aosr/config/data/x.toml`）刻意不動：那一種不是「在哪一棵樹」，
    它是載入器訊息本身的一部分。

    **它連 pydantic 訊息裡的網址前綴一起收掉**（`https://errors.pydantic.dev/2.13/v/finite_number`
    變成 `https:<path>/finite_number`）。那是刻意的：那個前綴帶著庫的版本號，換一個
    pydantic 版本就漂一次，而「哪一種錯」已經由 `[type=…]` 那一格比到了（見
    `_config_answers.validate_message`）。**不要把這一段當成漏抓去修 regex**——修了要
    重生答案檔，而現在的行為才是對的。
    """
    return _ABSOLUTE_PREFIX.sub(NORMALISED_PATH + "/", message)


# 這一刀的載入器有兩種共同的入口形狀，收成兩支小工廠（產生器與考卷都照同一組旗標跑）。
def perimeter_default(
    fn: str,
    *,
    mutations: list[Mutation] | None = None,
    kwargs: dict[str, Tagged | int | str] | None = None,
) -> CaseOp:
    """一筆「餵真的設定檔、不傳路徑（走載入器自己的預設）」的 case。"""
    op: CaseOp = {"fn": fn, "path": "default"}
    if mutations is not None:
        op["mutations"] = mutations
    if kwargs is not None:
        op["kwargs"] = kwargs
    return op


def perimeter_missing(fn: str) -> CaseOp:
    """一筆「餵一個不存在的路徑」的 case（驗各支載入器的開檔失敗分支）。"""
    return {"fn": fn, "path": "missing"}


def perimeter_mutation(fn: str, before: str, after: str) -> CaseOp:
    """一筆「把真的設定檔改一格再餵」的 case。"""
    return {"fn": fn, "path": "default", "mutations": [Mutation(before=before, after=after)]}


def op_for(case: CaseEntry) -> dict[str, object]:
    """一筆 case 的執行方式（沒寫 ``op`` 就是第 1 刀那種「呼叫帶參數的函式」）。"""
    op = case.get("op")
    if op is None:
        return {}
    return {str(key): value for key, value in op.items()}


def cut2_case_ids() -> set[str]:
    """這一刀那 9 支模組宣告的每一個 case id（考卷用它逐支挑出自己的 case）。"""
    ids: set[str] = set()
    for name in CUT2_MODULES:
        for case in cases_for(f"cut2_{name}"):
            ids.add(case["id"])
    return ids


MODULES: Final[dict[str, ModuleSpec]] = {
    "art_lane": {
        "file": "art_lane",
        "constants": [
            "ART_FACE_GRID_N",
            "ART_FORMFACTOR_EPS",
            "ART_NEUMANN_EPS_TAIL",
            "ART_NEUMANN_K_MAX",
            "ART_N_PER_WALL_DEFAULT",
            "ART_POWERITER_K",
            "ART_P_CAP",
            "ART_RECIPROCITY_TOL",
            "ART_WLS_T20_HI_DB",
            "ART_WLS_T20_LO_DB",
            "ART_WLS_WINDOW_SOFTNESS_DB",
        ],
        "cases": [
            {"id": "art_lane.guard_art_patch_count.n=0", "args": {"n_per_wall": 0}},
            {"id": "art_lane.guard_art_patch_count.n=-1", "args": {"n_per_wall": -1}},
            {"id": "art_lane.guard_art_patch_count.n=1", "args": {"n_per_wall": 1}},
            {"id": "art_lane.guard_art_patch_count.n=6", "args": {"n_per_wall": 6}},
            {"id": "art_lane.guard_art_patch_count.n=36", "args": {"n_per_wall": 36}},
            {"id": "art_lane.guard_art_patch_count.n=37", "args": {"n_per_wall": 37}},
            {"id": "art_lane.guard_art_patch_count.n=116", "args": {"n_per_wall": 116}},
            {"id": "art_lane.guard_art_patch_count.custom-context.n=37", "args": {"context": 'ART-CUSTOM', "n_per_wall": 37}},
            {"id": "art_lane.guard_polygon_art_patch_count.n=0", "args": {"n_tris": 0}},
            {"id": "art_lane.guard_polygon_art_patch_count.n=-5", "args": {"n_tris": -5}},
            {"id": "art_lane.guard_polygon_art_patch_count.n=1", "args": {"n_tris": 1}},
            {"id": "art_lane.guard_polygon_art_patch_count.n=8192", "args": {"n_tris": 8192}},
            {"id": "art_lane.guard_polygon_art_patch_count.n=8193", "args": {"n_tris": 8193}},
            {"id": "art_lane.guard_polygon_art_patch_count.custom-context.n=5000", "args": {"context": 'polygon ART-CUSTOM', "n_tris": 5000}},
        ],
    },
    "art_rt_guard": {
        "file": "art_rt_guard",
        "constants": [
            "ART_RT_GUARD_SMOOTHNESS_S",
            "ART_RT_T_CAP_S",
            "RENDER_DEFAULT_SAMPLE_RATE_HZ",
        ],
        "cases": [
        ],
    },
    "authoring_defaults": {
        "file": "authoring_defaults",
        "constants": [
            "APP_PATCH_PREFIX",
            "APP_WALL_PREFIX",
            "BARE_LAYER_STACK_SPEC",
            "BARE_MATERIAL_ID",
            "BARE_TEMPLATE_ID",
            "BROADBAND_POROUS_LAYER_STACK_SPEC",
            "BROADBAND_POROUS_TEMPLATE_ID",
            "CUSTOM_STACK_TEMPLATE_ID",
            "FABRIC_MS",
            "FABRIC_RS",
            "FIXED_ALPHA_TEMPLATE_ID",
            "FLOW_RESISTIVITY_MAX",
            "FLOW_RESISTIVITY_MIN",
            "GRID_PRESETS",
            "HOLE_RATIO_MAX",
            "HOLE_RATIO_MIN",
            "PERFORATED_FIXED_THICKNESS_M",
            "RESONANT_PANEL_LAYER_STACK_SPEC",
            "RESONANT_PANEL_TEMPLATE_ID",
            "RHINO_PREFIX",
            "SPACING_MAX_M",
            "SPACING_MIN_M",
            "TOTAL_THICKNESS_MAX_M",
        ],
        "cases": [
            {"id": "authoring_defaults.app_wall_id.'y0'", "args": {"wall_id": 'y0'}},
            {"id": "authoring_defaults.app_wall_id.'north'", "args": {"wall_id": 'north'}},
            {"id": "authoring_defaults.app_wall_id.''", "args": {"wall_id": ''}},
            {"id": "authoring_defaults.app_patch_id.'y0'.r1c2", "args": {"col": 2, "row": 1, "wall_id": 'y0'}},
            {"id": "authoring_defaults.app_patch_id.'x1'.r0c0", "args": {"col": 0, "row": 0, "wall_id": 'x1'}},
            {"id": "authoring_defaults.app_patch_id.''.r-3c7", "args": {"col": 7, "row": -3, "wall_id": ''}},
            {"id": "authoring_defaults.rhino_boundary_id.'7d8ecb8f-0000-4000-8000-000000000000'", "args": {"guid": '7d8ecb8f-0000-4000-8000-000000000000'}},
            {"id": "authoring_defaults.rhino_boundary_id.''", "args": {"guid": ''}},
            {"id": "authoring_defaults.rhino_boundary_id.'no-dashes'", "args": {"guid": 'no-dashes'}},
        ],
    },
    "crossover_axis": {
        "file": "crossover_axis",
        "constants": [
            "DEFAULT_T_C_CENTER_S",
            "DEFAULT_T_C_FADE_S",
        ],
        "cases": [
            {"id": "crossover_axis.build.seam=321.0", "args": {"seam_f_s": flt(321.0)}},
            {"id": "crossover_axis.build.seam=321.0.center=0.08.fade=0.02", "args": {"seam_f_s": flt(321.0), "t_c_center": flt(0.08), "t_c_fade": flt(0.02)}},
            {"id": "crossover_axis.build.seam=250.0.center=0.08.fade=0.013", "args": {"seam_f_s": flt(250.0), "t_c_center": flt(0.08), "t_c_fade": flt(0.013)}},
            {"id": "crossover_axis.build.seam=0.0", "args": {"seam_f_s": flt(0.0)}},
        ],
    },
    "default_geometry": {
        "file": "default_geometry",
        "constants": [
            "DEFAULT_DIMS_M",
            "DEFAULT_EAR_HEIGHT_M",
            "DEFAULT_RECEIVER_XYZ",
            "DEFAULT_SOURCE_XYZ",
        ],
        "cases": [
        ],
    },
    "ism_lane": {
        "file": "ism_lane",
        "constants": [
            "ISM_FACE_GRID_N",
            "ISM_FACE_GRID_SAMPLES",
        ],
        "cases": [
        ],
    },
    "phase2_report_bands": {
        "file": "phase2_report_bands",
        "constants": [
            "PHASE2_LATE_RT_BAND_HZ",
            "PHASE2_RFZ_BAND_HZ",
            "PHASE2_RFZ_THRESHOLD_DB",
            "PHASE2_RFZ_WINDOW_S",
            "PHASE2_TARGET_SLOPE_DB_OCT",
        ],
        "cases": [
        ],
    },
    "receiver_grid": {
        "file": "receiver_grid",
        "constants": [
            "EAR_HEIGHT_M",
            "GRID_MARGIN_M",
            "GRID_N",
        ],
        "cases": [
        ],
    },
    "source_reference": {
        "file": "source_reference",
        "constants": [
            "CANONICAL_MONOPOLE_STRENGTH",
            "DIFFUSE_MONOPOLE_4PI",
            "HIGH_SPL_REFERENCE_DB_SPL_1M_1W",
            "HIGH_SPL_REFERENCE_PRESSURE_PA",
            "REFERENCE_ANCHOR_KIND",
            "REFERENCE_DRIVE_POWER_W",
            "REFERENCE_SENSITIVITY_DB_SPL_1M_1W",
            "REFERENCE_SENSITIVITY_PRESSURE_PA",
        ],
        "cases": [
        ],
    },
    "speaker_directivity": {
        "file": "speaker_directivity",
        "constants": [
            "DEFAULT_SPEAKER_TYPE",
            "DIRECTIVITY_DEFAULT_ENABLED",
            "DIRECTIVITY_ENABLED_KEY",
            "DIRECTIVITY_GOLDEN_VERSION",
            "DIRECTIVITY_MODEL_VERSION",
            "DIRECTIVITY_NORMALIZATION_GL_ORDER",
            "DIRECTIVITY_REAR_GAIN_MIN",
            "DIRECTIVITY_TOE_IN_ANCHOR",
            "PISTON_RADIUS_OVERRIDE_KEY",
            "SPEAKER_PRESETS",
            "SPEAKER_TYPES",
            "SPEAKER_TYPE_KEY",
            "SPEAKER_WIDTH_OVERRIDE_KEY",
        ],
        "cases": [
            {"id": "speaker_directivity.resolve.enabled=True", "args": {"enabled": true()}},
            {"id": "speaker_directivity.resolve.enabled=True.type='bookshelf'", "args": {"enabled": true(), "speaker_type": 'bookshelf'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='floorstanding'", "args": {"enabled": true(), "speaker_type": 'floorstanding'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='inwall'", "args": {"enabled": true(), "speaker_type": 'inwall'}},
            {"id": "speaker_directivity.resolve.enabled=False", "args": {"enabled": false()}},
            {"id": "speaker_directivity.resolve.enabled=False.type='inwall'", "args": {"enabled": false(), "speaker_type": 'inwall'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='  Bookshelf  '", "args": {"enabled": true(), "speaker_type": '  Bookshelf  '}},
            {"id": "speaker_directivity.resolve.enabled=True.type='bookshelf'.baffle=0.21", "args": {"baffle_width_m": flt(0.21), "enabled": true(), "speaker_type": 'bookshelf'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='bookshelf'.piston=0.05", "args": {"enabled": true(), "piston_radius_m": flt(0.05), "speaker_type": 'bookshelf'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='floorstanding'.baffle=0.3.piston=0.1", "args": {"baffle_width_m": flt(0.3), "enabled": true(), "piston_radius_m": flt(0.1), "speaker_type": 'floorstanding'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='unknown-box'", "args": {"enabled": true(), "speaker_type": 'unknown-box'}},
            {"id": "speaker_directivity.resolve.enabled=False.type='soundbar'", "args": {"enabled": false(), "speaker_type": 'soundbar'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='bookshelf'.baffle=0.0", "args": {"baffle_width_m": flt(0.0), "enabled": true(), "speaker_type": 'bookshelf'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='bookshelf'.piston=-0.1", "args": {"enabled": true(), "piston_radius_m": flt(-0.1), "speaker_type": 'bookshelf'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='bookshelf'.baffle=nan", "args": {"baffle_width_m": nan(), "enabled": true(), "speaker_type": 'bookshelf'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='bookshelf'.baffle=inf", "args": {"baffle_width_m": plus_inf(), "enabled": true(), "speaker_type": 'bookshelf'}},
            {"id": "speaker_directivity.resolve.enabled=True.type='floorstanding'.piston=True", "args": {"enabled": true(), "piston_radius_m": true(), "speaker_type": 'floorstanding'}},
            {"id": "speaker_directivity.SPEAKER_PRESETS.all", "args": {"presets": items('bookshelf', 'floorstanding')}},
        ],
    },
    "spl_output": {
        "file": "spl_output",
        "constants": [
            "DEFAULT_PLAYBACK_LEVEL_DB",
            "DEFAULT_SENSITIVITY_DB",
            "FLAT_EQUAL_TOL_DB",
        ],
        "cases": [
            {"id": "spl_output.SplOutputConfig.defaults", "args": {}},
            {"id": "spl_output.SplOutputConfig.playback_level_db=0.0.sensitivity_db=[0.0]", "args": {"playback_level_db": flt(0.0), "sensitivity_db": floats(flt(0.0))}},
            {"id": "spl_output.SplOutputConfig.playback_level_db=-6.0.sensitivity_db=[88.0]", "args": {"playback_level_db": flt(-6.0), "sensitivity_db": floats(flt(88.0))}},
            {"id": "spl_output.SplOutputConfig.playback_level_db=0.0.sensitivity_db=[85.0, 85.0]", "args": {"playback_level_db": flt(0.0), "sensitivity_db": floats(flt(85.0), flt(85.0))}},
            {"id": "spl_output.SplOutputConfig.sensitivity_db=[85.0, 85.0000005]", "args": {"sensitivity_db": floats(flt(85.0), flt(85.0000005))}},
            {"id": "spl_output.SplOutputConfig.sensitivity_db=[85.0, 88.0]", "args": {"sensitivity_db": floats(flt(85.0), flt(88.0))}},
            {"id": "spl_output.SplOutputConfig.sensitivity_db=[nan]", "args": {"sensitivity_db": floats(nan())}},
            {"id": "spl_output.SplOutputConfig.sensitivity_db=[inf]", "args": {"sensitivity_db": floats(plus_inf())}},
            {"id": "spl_output.SplOutputConfig.sensitivity_db=[-inf]", "args": {"sensitivity_db": floats(minus_inf())}},
            {"id": "spl_output.SplOutputConfig.playback_level_db=nan", "args": {"playback_level_db": nan()}},
            {"id": "spl_output.SplOutputConfig.playback_level_db=inf", "args": {"playback_level_db": plus_inf()}},
            {"id": "spl_output.SplOutputConfig.sensitivity_db=[]", "args": {"sensitivity_db": floats()}},
        ],
    },
    # ── 這一刀（票 #127 後半）：讀檔那 9 支 ────────────────────────────────────
    # 每一支都有一筆「真的設定檔」的 case（`op` 說「餵真的那個檔、不傳路徑就走預設」），
    # 其餘是突變（把真的設定檔改一格再餵）或「檔案不存在」那條分支。`constants` 是模組層
    # 常數的凍結值；載入器回傳的物件（含巢狀子模型）由那一筆 case 的值一起凍結。
    "cut2_calibration": {
        "file": "calibration",
        "constants": ["CalibrationBandConfig", "CalibrationConfig", "CalibrationRtConfig", "CalibrationSmoothingConfig"],
        "cases": [
            {
                "id": "cut2_calibration.load.real",
                "args": {},
                "op": perimeter_default("load_calibration"),
            },
            {
                "id": "cut2_calibration.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_calibration"),
            },
            {
                "id": "cut2_calibration.load.mut.band-f_hi-not-above-f_lo",
                "args": {},
                "op": perimeter_mutation("load_calibration", "f_hi_hz = 135.0", "f_hi_hz = 30.0"),
            },
        ],
    },
    "cut2_fem_lane": {
        "file": "fem_lane",
        "constants": [
            "CEIL_PLANARITY_TOL",
            "FACE_PLANARITY_TOL",
            "FEM3D_ENERGY_FLOOR_E",
            "FEM_FMAX_CAP_HZ",
            "FEM_N_DEFAULT",
            "FULLBAND_FLATNESS_FMAX_HZ",
            "MAX_CEILING_SLOPE",
            "MIN_OCTAGON_EDGE_M",
            "OBJECTIVE_FLATNESS_EPS_DEFAULT",
            "OBJECTIVE_RULE_DEFAULT",
            "POSITION_CENTERLINE_TOL_M_DEFAULT",
            "POSITION_DEFAULT_HEIGHT_M",
            "POSITION_MIN_SOURCE_SOURCE_M_DEFAULT",
            "POSITION_N_TRIALS_DEFAULT",
            "SHAPE6_UNASSIGNED_ALPHA",
            "SHAPE_EVAL_REFERENCE_ALPHA",
            "SPLAY_CAP",
            "SV_BARRIER_W",
            "SV_FLOOR_EPS",
            "SV_FLOOR_FRAC",
            "VOL_CONSERVE_W",
        ],
        "cases": [],
    },
    "cut2_mat_continuation": {
        "file": "mat_continuation",
        "constants": ["MatContinuationConfig"],
        "cases": [
            {
                "id": "cut2_mat_continuation.load.real",
                "args": {},
                "op": perimeter_default("load_mat_continuation"),
            },
            {
                "id": "cut2_mat_continuation.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_mat_continuation"),
            },
            {
                "id": "cut2_mat_continuation.load.mut.beta-cap-below-beta0",
                "args": {},
                "op": perimeter_mutation("load_mat_continuation", "beta_cap = 500.0", "beta_cap = 10.0"),
            },
        ],
    },
    "cut2_membrane_opt": {
        "file": "membrane_opt",
        "constants": ["MembraneOptConfig"],
        "cases": [
            {
                "id": "cut2_membrane_opt.load.real",
                "args": {},
                "op": perimeter_default("load_membrane_opt"),
            },
            {
                "id": "cut2_membrane_opt.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_membrane_opt"),
            },
            # `_validate_ranges` 的 6 條分支：4 組 lower<upper ＋ w_decay/w_rt60 必為 0。
            {
                "id": "cut2_membrane_opt.load.mut.air_gap-lower-equals-upper",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "air_gap_min_m = 0.005", "air_gap_min_m = 0.40"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.mass-lower-above-upper",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "mass_min_kg_m2 = 0.10", "mass_min_kg_m2 = 30.0"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.porous-lower-above-upper",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "porous_min_m = 0.005", "porous_min_m = 0.30"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.flow_resistivity-lower-above-upper",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "flow_resistivity_min = 1000.0", "flow_resistivity_min = 200000.0"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.w_decay-nonzero",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "w_decay = 0.0", "w_decay = 0.5"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.w_rt60-nonzero",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "w_rt60 = 0.0", "w_rt60 = 1.0"),
            },
        ],
    },
    "cut2_perceptual": {
        "file": "perceptual",
        "constants": ["MATERIAL_RFZ_PROFILE"],
        "cases": [
            {
                "id": "cut2_perceptual.load.real.default",
                "args": {},
                "op": perimeter_default("load_perceptual"),
            },
            {
                "id": "cut2_perceptual.load.real.profile-monitoring",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("monitoring")}),
            },
            {
                "id": "cut2_perceptual.load.real.profile-critical",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("critical")}),
            },
            {
                "id": "cut2_perceptual.load.real.profile-home_theater",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("home_theater")}),
            },
            {
                "id": "cut2_perceptual.load.real.profile-material_rfz",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("material_rfz")}),
            },
            {
                "id": "cut2_perceptual.load.real.unknown-profile",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("nope")}),
            },
            {
                "id": "cut2_perceptual.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_perceptual"),
            },
            {
                "id": "cut2_perceptual.load.material_rfz-profile.real",
                "args": {},
                "op": perimeter_default("load_material_rfz_profile"),
            },
            {
                "id": "cut2_perceptual.load.material_rfz-profile.missing-file",
                "args": {},
                "op": perimeter_missing("load_material_rfz_profile"),
            },
            {
                "id": "cut2_perceptual.load.mut.extra-top-level-key",
                "args": {},
                "op": perimeter_mutation("load_perceptual", 'default_profile = "monitoring"', 'default_profile = "monitoring"\nbogus = true'),
            },
            {
                "id": "cut2_perceptual.load.mut.rt60-target-ragged-lengths",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": "tol_lo_s = [0.05, 0.05, 0.05, 0.05, 0.10]",
                            "after": "tol_lo_s = [0.05, 0.05, 0.05, 0.05]",
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.modal-curve-ragged-lengths",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": "t60_s = [0.90, 0.50, 0.35, 0.25, 0.20]",
                            "after": "t60_s = [0.90, 0.50, 0.35, 0.25]",
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.default-profile-not-in-profiles",
                "args": {},
                "op": perimeter_mutation("load_perceptual", 'default_profile = "monitoring"', 'default_profile = "nope"'),
            },
            {
                "id": "cut2_perceptual.load.mut.profile-level-not-in-curves",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": '[profiles.monitoring]\nrfz = { threshold_db = -10.0, window_ms = 15.0, role = "hard_gate" }\nmodal_decay = { level = "music" }',
                            "after": '[profiles.monitoring]\nrfz = { threshold_db = -10.0, window_ms = 15.0, role = "hard_gate" }\nmodal_decay = { level = "nope" }',
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.rfz-both-windows",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": '[profiles.critical]\nrfz = { threshold_db = -25.0, window_mode = "dynamic", role = "soft_target" }',
                            "after": '[profiles.critical]\nrfz = { threshold_db = -25.0, window_mode = "dynamic", window_ms = 15.0, role = "soft_target" }',
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.rfz-no-window",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": '[profiles.monitoring]\nrfz = { threshold_db = -10.0, window_ms = 15.0, role = "hard_gate" }',
                            "after": '[profiles.monitoring]\nrfz = { threshold_db = -10.0, role = "hard_gate" }',
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.early-reflection-extra-key",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": "required_attenuation_db = 10.0\n# 10dB 檢查的聚合帶",
                            "after": "required_attenuation_db = 10.0\nbogus_key = 1.0\n# 10dB 檢查的聚合帶",
                        }
                    ],
                },
            },
        ],
    },
    "cut2_physics_constants": {
        "file": "physics_constants",
        "constants": ["PhysicsConstants"],
        "cases": [
            {
                "id": "cut2_physics_constants.load.real",
                "args": {},
                "op": perimeter_default("load_physics_constants"),
            },
            {
                "id": "cut2_physics_constants.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_physics_constants"),
            },
            {
                "id": "cut2_physics_constants.load.mut.sound-speed-zero",
                "args": {},
                "op": perimeter_mutation("load_physics_constants", "sound_speed_m_s = 343.0", "sound_speed_m_s = 0.0"),
            },
        ],
    },
    "cut2_scoring": {
        "file": "scoring",
        "constants": ["AsymmetryScoringConfig", "DecayScoringConfig", "ScoringConfig", "SmoothingScoringConfig", "SpatialScoringConfig"],
        "cases": [
            {
                "id": "cut2_scoring.load.real",
                "args": {},
                "op": perimeter_default("load_scoring"),
            },
            {
                "id": "cut2_scoring.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_scoring"),
            },
            {
                "id": "cut2_scoring.load.mut.hf-rolloff-fc-zero",
                "args": {},
                "op": perimeter_mutation("load_scoring", "hf_rolloff_fc_hz = 2000.0", "hf_rolloff_fc_hz = 0.0"),
            },
            {
                "id": "cut2_scoring.load.mut.peak-dip-bias-above-range",
                "args": {},
                "op": perimeter_mutation("load_scoring", "peak_dip_bias = 0.0", "peak_dip_bias = 9.0"),
            },
            {
                "id": "cut2_scoring.load.mut.frac-oct-zero",
                "args": {},
                "op": perimeter_mutation("load_scoring", "frac_oct = 3.0", "frac_oct = 0.0"),
            },
        ],
    },
    "cut2_scoring_v2": {
        "file": "scoring_v2",
        "constants": ["LOCAL_CROSS_7_ORDER", "SCORING_V2_SCHEMA_VERSION"],
        "cases": [
            {
                "id": "cut2_scoring_v2.load.real",
                "args": {},
                "op": perimeter_default("load_scoring_v2"),
            },
            {
                "id": "cut2_scoring_v2.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_scoring_v2"),
            },
            {
                "id": "cut2_scoring_v2.load.production-mode-refused",
                "args": {},
                "op": perimeter_default("load_scoring_v2", kwargs={"mode": strs("production")}),
            },
            {
                "id": "cut2_scoring_v2.load.mode-spelling-rejected",
                "args": {},
                "op": perimeter_default("load_scoring_v2", kwargs={"mode": strs("prod")}),
            },
            {
                "id": "cut2_scoring_v2.load.mut.frequency-ranges-do-not-join",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", "high_hz = 200.0", "high_hz = 201.0"),
            },
            {
                "id": "cut2_scoring_v2.load.mut.local-cross-order-swapped",
                "args": {},
                "op": {
                    "fn": "load_scoring_v2",
                    "mutations": [
                        {
                            "before": 'order = ["center", "front", "back", "left", "right", "up", "down"]',
                            "after": 'order = ["center", "back", "front", "left", "right", "up", "down"]',
                        }
                    ],
                },
            },
            {
                "id": "cut2_scoring_v2.load.mut.radius-zero",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", "radius_m = 0.10", "radius_m = 0.0"),
            },
            {
                "id": "cut2_scoring_v2.load.mut.frame-tolerance-zero",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", "frame_tolerance = 1e-6", "frame_tolerance = 0.0"),
            },
            {
                "id": "cut2_scoring_v2.load.mut.dip-min-width-not-one-sixth",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", "min_width_oct = 0.16666666666666666", "min_width_oct = 0.16"),
            },
            {
                "id": "cut2_scoring_v2.load.mut.rfz-band-status-ready",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", 'band_status = "unresolved"', 'band_status = "ready"'),
            },
        ],
    },
    "cut2_stereo_layout": {
        "file": "stereo_layout",
        # 這一支只有一個常數物件（19 個政策欄位是它的欄位，不是模組層常數）。那一筆
        # `load.real` 的值就是整個物件，逐欄比；19 個欄位名另外由
        # `tests/engine/test_config_cut2_stereo_layout.py` 具名列出並逐欄核對
        # （上一代有一條考卷逐欄比的是 `lib.physics.stereo_geometry` 那一排 `STEREO_*`，
        # 那一支不在這一刀裡）。
        "constants": ["StereoLayoutConfig"],
        "cases": [
            {
                "id": "cut2_stereo_layout.load.real",
                "args": {},
                "op": perimeter_default("load_stereo_layout"),
            },
            {
                "id": "cut2_stereo_layout.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_stereo_layout"),
            },
            {
                "id": "cut2_stereo_layout.load.mut.unknown-top-level-table",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "distance_imbalance_max = 0.20", "distance_imbalance_max = 0.20\n\n[unexpected]\nvalue = 1.0"),
            },
            {
                "id": "cut2_stereo_layout.load.mut.unknown-policy-key",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "distance_imbalance_max = 0.20", "distance_imbalance_max = 0.20\nunknown_policy = 1.0"),
            },
            {
                "id": "cut2_stereo_layout.load.mut.non-numeric-value",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "distance_imbalance_max = 0.20", 'distance_imbalance_max = "0.20"'),
            },
            {
                "id": "cut2_stereo_layout.load.mut.non-finite-value",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "distance_imbalance_max = 0.20", "distance_imbalance_max = nan"),
            },
            {
                "id": "cut2_stereo_layout.load.mut.missing-policy-key",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "speaker_lateral_wall_margin_m = 0.5\n", ""),
            },
            {
                "id": "cut2_stereo_layout.load.mut.stereo-section-missing",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "[stereo]", "[not_stereo]"),
            },
        ],
    },
}


def resolve(value: object) -> object:
    """把一個參數值還原成 Python 值（產生器餵給上一代、考卷餵給新家用同一支）。

    不是記號（例如 ``0``、``"y0"``）就原樣回傳。記號的種類看不懂就當場炸——
    看不懂的參數靜靜地餵下去，比炸掉糟。
    """
    if not isinstance(value, dict) or "k" not in value:
        return value
    kind = value["k"]
    if kind == "float":
        return _as_number(value.get("v"), "float")
    if kind == "bool":
        return bool(value.get("v"))
    if kind == "str":
        raw_text = value.get("v")
        if not isinstance(raw_text, str):
            raise AssertionError(f"str 記號裡不是字串：{raw_text!r}")
        return raw_text
    if kind == "nan":
        return float("nan")
    if kind == "plus_inf":
        return float("inf")
    if kind == "minus_inf":
        return float("-inf")
    if kind in ("items", "float_tuple"):
        raw = value["v"]
        if not isinstance(raw, list):
            raise AssertionError(f"{kind} 記號的內容不是一串東西：{raw!r}")
        resolved = [resolve(item) for item in raw]
        return tuple(resolved) if kind == "float_tuple" else resolved
    raise AssertionError(f"不認識的參數記號：{kind!r}")


def _as_number(raw: object, where: str) -> float:
    """把一個記號裡的數字收窄成 ``float``（不是數字就當場炸）。"""
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise AssertionError(f"{where} 記號裡不是數字：{raw!r}")
    return float(raw)


def resolve_args(case: CaseEntry) -> dict[str, object]:
    """一筆 case 的參數，整格還原成 Python 值。"""
    return {str(name): resolve(item) for name, item in case["args"].items()}


def all_case_ids() -> set[str]:
    """這一張表宣告的每一個 case id（考卷用它跟答案檔比對集合）。"""
    ids: set[str] = set()
    for module in MODULES.values():
        for case in module["cases"]:
            ids.add(case["id"])
    return ids


def constant_ids() -> set[str]:
    """每一個常數的凍結值那一筆的 id（``<module>.const.<NAME>``）。"""
    ids: set[str] = set()
    for name, module in MODULES.items():
        for constant in module["constants"]:
            ids.add(f"{name}.const.{constant}")
    return ids


def cases_for(module: str) -> list[CaseEntry]:
    """某一支模組宣告的探針 case（產生器照這一串去跑）。"""
    return list(MODULES[module]["cases"])


def constants_for(module: str) -> list[str]:
    """某一支模組宣告的公開常數名（產生器照這一串去取凍結值）。"""
    return list(MODULES[module]["constants"])
