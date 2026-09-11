"""票 #127 前半那 11 支 config 模組的 case 表：**「有哪些 case」的唯一住處**。

**為什麼要有這一支（PR #150 的找碴 F1）。** 原本的形狀是「產生器自己去上一代跑一輪、
考卷逐筆跟答案檔比」——那份答案檔住 ``blueprint/``，而 ``identity-strings-generated`` 與
``refs-and-links-resolve`` 都扣掉那一層，於是**沒有任何機制保證它還有幾筆**：把答案檔的
探針逐筆刪掉，只有極少數幾筆刪了會讓考卷變紅，其餘刪掉整套照樣綠。這一支把「有哪些 case」
搬進版控、當成產生器與考卷**共用的同一份清單**：

* 產生器（``blueprint/generate_config_cut1_answers.py``）只跑**這裡宣告的**參數；
* 考卷（``tests/engine/``）斷言答案檔裡的 case id 集合**等於**這裡宣告的集合
  （少一筆紅、多一筆也紅——比的是具名的集合，不是筆數）。

**參數怎麼寫。** 值是純量或這幾個小幫手（都是普通 dict，沒有任何 v2 依賴，兩邊都 import
得到）：``flt``（浮點）、``nan``／``plus_inf``／``minus_inf``、``true``／``false``
（布林——有些參數要驗「餵布林會被擋」，所以布林不能跟整數混在一起）、``str_list``。
參數自己的**種類**（哪個參數是 int、哪個是 float）由各支考卷的 ``TypedDict`` 講清楚，
這一張表只管「有哪些 case」。

**這一支不准 import 上一代的任何東西。** 它會被 ``style-guard``／``type-guard``／
``uv-single-entrypoint`` 掃（``blueprint/`` 只被 ``refs-and-links-resolve`` 扣掉）。
"""
from __future__ import annotations

from typing import Final, TypedDict

# 一個參數值就是一個普通 dict（不是自訂類別：答案檔用 json 存，寫成普通 dict 才不必
# 為序列化再補一層 representer）。
Tagged = dict[str, object]


class CaseEntry(TypedDict):
    """一筆 case：穩定的 id ＋ 那一筆的參數（參數值是 :data:`Tagged` 或純量）。"""

    id: str
    args: dict[str, Tagged | int | str]


class ModuleSpec(TypedDict):
    """一支模組在這一張表裡的兩件事：對應的檔名、要凍結的常數、要跑的 case。"""

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
