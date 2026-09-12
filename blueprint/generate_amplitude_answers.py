"""把凍結的 v2 donor 的振幅（反射乘積與每條路徑的壓力）跑出來，寫成新家考卷要用的標準答案檔。

**為什麼要有這一支。** 票 #194 把第 4 段第一步（票 #188）凍結的振幅參考值入庫到 v3：上一代
在唯讀的 donor 工作樹上跑 ``precompute_shoebox_patch_paths``（鏡像聲源法的反射路徑）與
``solve_ism_shoebox_patched``（解出反射乘積、總壓力、能量），把每條路徑每個頻率的反射乘積
（``reflection_product``）與路徑壓力（``path_pressure``）存成檔案。這一支就是那個產生器：它照
上一代 ``run/amplitude_reference.py`` 的**同一條算式、同一個 dtype（complex64／float32）**重跑
一遍，寫成 ``blueprint/reference_amplitude_{flat,varied}.json``，``paths`` 與 ``totals`` 兩塊
跟凍結原件（``/home/florian/aosr-v3-work/188/evidence/frozen-seg3/``）逐位元相同。

**兩組材料。** ``--case flat``：六面牆同一種純實數、頻率無關的表面阻抗 ``4·ρc``（材料
``ref-absorber-4rhoc``，用 ``constant_impedance`` 建）。``--case varied``：五面牆維持 ``4·ρc``，
``y0`` 那一面換成六個頻帶各不同、帶虛部的阻抗 ``Z_k = (2 + 0.5k)·ρc + i·(0.3·ρc)·(−1)^k``
（``k=0..5``，材料 ``ref-absorber-y0-varied``，用 ``MaterialResponse.from_impedance`` 建）。
兩組的 ``paths`` 結構相同（都是 63 條、六個頻帶），只有數值不同。

**path_pressure 是照 donor 的式子重現，不是函式直接回傳。** ``solve_ism_shoebox_patched``
內部算 ``path_pressure = (1/dist)·refl·exp(−iωτ)`` 但不回傳它（回傳的 ``ShoeboxIsmResult`` 只帶
``pressure, delay, reflection_product, distance, ism_direct_E, ism_rev_E, order``）。所以這一支在
v2 環境裡用 donor 自己的公式（``lib/physics/ism_patch.py``，``dist_safe = jnp.maximum(dist,
1e-12)``、``omega = 2π·freqs``、``phase = exp(−i·omega·delay)``、``refl = ∏ R``、
``path_pressure = (1/dist_safe)·refl·phase``）在同一個 dtype 下重算，並把加總跟
``result.pressure`` 的最大差存進檔頭——那是「重現對不對」的證據，不是要修的目標。

**反射乘積的定義。** ``refl = ∏ R``，每一跳的 ``R = z_to_r_from_cos(Z, cosθ, ρc)``
（``(Z·cosθ − ρc)/(Z·cosθ + ρc)``）；directivity 關著所以沒有別的因子。直達路徑（order 0）沒有
反射，乘積恰等於 ``1+0j``。

**donor 的出身用量的，不是宣稱。** 解析 ``v3-donor`` 記號、確認那棵樹乾淨、確認 HEAD 就在
記號上，三格一起寫進檔頭（照 ``blueprint/generate_reference_room_answers.py`` 的同一套）。
donor 不在 CI 上，所以那個 sha 沒有機器在守——它是「產生時量的 ＋ 人工可重跑」。

**答案檔頂層**：``schema``（1）、``note``（由…產生、不准手改）、``donor``、``env``（python 與
jax/jaxlib/numpy/flax 版本、JAX 後端與 x64 開關、dtype 六格、platform）、``command``、
``parameters``（含 ``cases`` 兩組材料的完整描述）、``paths``（每條 ``index, order, identity,
dist_m, delay_s, reflection_product[6], path_pressure[6]``，複數的實／虛／abs 各存 ``dec`` 與
``hex``）、``totals``（``pressure, ism_direct_E, ism_rev_E``，每頻率一格）。**不存 platform 以外
會隨機器變的東西**進 ``paths``：``paths`` 只放純計算結果的浮點值與 identity。

**路徑數值同時存 ``dec`` 與 ``hex``**：``hex()`` 是精確的、可 ``float.fromhex()`` 還原，
跨平台不漂；``dec`` 是給人看的 ``repr``。``totals`` 只有 ``pressure, ism_direct_E, ism_rev_E``
三格（照凍結原件——``reproduced_pressure_sum`` 那一格是重現與總壓力的差，屬證據不屬答案，
考卷不拿它當契約）。考卷用 hex 逐字字串相等比對，不是 isclose。

**這一支刻意不印任何東西。** ``style-guard`` 只准具名的輸出層 ``print``，而這一支還沒有
登記（白名單上現在三支產生器、一個引擎工具與輸出層本身，這一支不在）；要看它跑出什麼就讀
產物本身，跑壞了走離開碼。

**怎麼跑**（donor 自己一套 venv，在 v3 外面；``cd`` 在 v3 repo 根）::

    PYTHONPATH=/home/florian/aosr-v3-work/175/donor JAX_PLATFORMS=cpu \\
        PYTHONDONTWRITEBYTECODE=1 <donor venv>/bin/python \\
        -m blueprint.generate_amplitude_answers --case flat --out blueprint/reference_amplitude_flat.json

``--case`` 是 ``flat`` 或 ``varied``（必填）；``--out`` 必填。``PYTHONPATH`` 是**在呼叫時**帶進去
的環境變數，這支程式碼裡一個字都不碰 ``sys.path``（規矩卡 ``uv-single-entrypoint`` 第二條把
那一族判紅）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform as _platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

DONOR_TAG: Final[str] = "v3-donor"

ANSWER_SCHEMA: Final[int] = 1

# ── 規範牆順序（跟 v2 的 CANONICAL_WALL_ORDER 一樣）────────────────────────────
CANONICAL_WALL_ORDER: Final[tuple[str, ...]] = (
    "floor",
    "ceiling",
    "x0",
    "xL",
    "y0",
    "yL",
)

# ── 凍結參數（票 #188 的合約，不准改）─────────────────────────────────────────
#   房間 (6,4,3) m、聲源 (1.5,1.0,1.2)、接收點 (4.0,3.0,1.5)、聲速 343.0 m/s、
#   rho_c 411.6、max_order 3（63 條，跟第三段對齊）、六個頻帶、grid 六面各 (1,1)。
ROOM_LX: Final[float] = 6.0
ROOM_LY: Final[float] = 4.0
ROOM_LZ: Final[float] = 3.0
SOURCE_XYZ: Final[tuple[float, float, float]] = (1.5, 1.0, 1.2)
RECEIVER_XYZ: Final[tuple[float, float, float]] = (4.0, 3.0, 1.5)
SOUND_SPEED: Final[float] = 343.0
RHO_C: Final[float] = 411.6
MAX_ORDER: Final[int] = 3
GRID_SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
)
FREQS_HZ: Final[list[float]] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]

# 材料 id 與阻抗（兩組材料，照凍結原件）。
MATERIAL_ID_FLAT: Final[str] = "ref-absorber-4rhoc"
MATERIAL_ID_VARIED: Final[str] = "ref-absorber-y0-varied"
MATERIAL_Z_FLAT_REAL: Final[float] = 4.0 * RHO_C  # 1646.4，純實數
# varied 的 y0 牆：Z_k = (2 + 0.5k)·ρc + i·(0.3·ρc)·(−1)^k，k=0..5（k == 頻帶 index）。
VARIED_Y0_REAL_KK: Final[list[float]] = [
    (2.0 + 0.5 * k) * RHO_C for k in range(len(FREQS_HZ))
]
VARIED_Y0_IMAG_KK: Final[list[float]] = [
    (0.3 * RHO_C) * ((-1) ** k) for k in range(len(FREQS_HZ))
]

UNITS: Final[str] = (
    "metres (m), seconds (s), metres/second (m/s), hertz (Hz), "
    "Pa·s/m (impedance, rho_c)"
)
CONVENTION: Final[str] = (
    "origin at room corner; x in [0,Lx], y in [0,Ly], z in [0,Lz]; "
    "z=0 is floor, z=Lz ceiling. Wall ids in canonical order: "
    + ", ".join(CANONICAL_WALL_ORDER)
    + "."
)

# 檔頭要記的依賴版本（dtype 與數值跟著它們走，所以要記下來比對）。
RECORDED_PACKAGES: Final[tuple[str, ...]] = ("jax", "jaxlib", "numpy", "flax")

# ``command`` 檔頭要記的三個環境變數（這一跑能不能重現的鑰匙）。沒設就記 null。
RECORDED_ENV_VARS: Final[tuple[str, ...]] = (
    "PYTHONPATH",
    "JAX_PLATFORMS",
    "PYTHONDONTWRITEBYTECODE",
)

# dtype 六格的名字（答案是它們的量測值，不是手打的字串——見 ``_build_case``）。
DTYPE_SLOTS: Final[tuple[str, ...]] = (
    "Z_cells",
    "reflection_product",
    "pressure",
    "path_pressure_reproduced",
    "ism_direct_E",
    "ism_rev_E",
)

# 兩個 material case 的合法名（``--case`` 只認這兩個）。
CASES: Final[tuple[str, str]] = ("flat", "varied")


def frozen_parameters() -> dict[str, object]:
    """凍結參數那一塊（單一來源）：房間、聲源接收點、介質、材料、頻帶、慣例。

    這一份必須逐格等於凍結原件 ``parameters`` 的相應欄位（去掉 donor 自己加的
    ``cases`` 之外的部分僅描述、不改數值）。考卷 import 這一支拿同一組材料描述來
    比對，不另寫一份會漂的副本。
    """
    return {
        "room": {"Lx_m": ROOM_LX, "Ly_m": ROOM_LY, "Lz_m": ROOM_LZ},
        "source_xyz_m": {
            "x": SOURCE_XYZ[0],
            "y": SOURCE_XYZ[1],
            "z": SOURCE_XYZ[2],
        },
        "receiver_xyz_m": {
            "x": RECEIVER_XYZ[0],
            "y": RECEIVER_XYZ[1],
            "z": RECEIVER_XYZ[2],
        },
        "sound_speed_m_s": SOUND_SPEED,
        "rho_c_pa_s_per_m": RHO_C,
        "max_order": MAX_ORDER,
        "grid_shapes": [list(shape) for shape in GRID_SHAPES],
        "wall_edges": None,
        "directivity": None,
        "frequencies_hz": list(FREQS_HZ),
        "frequency_resolution": "custom",
        "cases": _material_cases(),
        "convention": CONVENTION,
        "units": UNITS,
    }


def _material_cases() -> dict[str, object]:
    """兩組材料的完整描述（照凍結原件 ``parameters.cases``，一字不差）。"""
    return {
        "flat": {
            "material": {
                "source": "fallback (no named absorbing preset in lib/materials/registry.py)",
                "constructor": "constant_impedance",
                "material_id": MATERIAL_ID_FLAT,
                "impedance": {
                    "z_value": "4.0 * rho_c = 1646.4 + 0.0j",
                    "real": MATERIAL_Z_FLAT_REAL,
                    "imag": 0.0,
                    "unit": "Pa·s/m",
                },
                "boundary_model": "local_impedance",
                "is_locally_reacting": True,
                "applied_to": "all six walls (floor, ceiling, x0, xL, y0, yL)",
            },
        },
        "varied": {
            "material": {
                "source": "MaterialResponse.from_impedance with a per-frequency complex array",
                "constructor": "from_impedance",
                "material_id": MATERIAL_ID_VARIED,
                "five_walls": {
                    "constructor": "constant_impedance",
                    "z_value": "4.0 * rho_c = 1646.4 + 0.0j",
                    "applied_to": "floor, ceiling, x0, xL, yL",
                },
                "y0_wall": {
                    "formula": "Z_k = (2 + 0.5*k)*rho_c + 1j*(0.3*rho_c)*((-1)**k), k=0..5",
                    "per_frequency_real": list(VARIED_Y0_REAL_KK),
                    "per_frequency_imag": list(VARIED_Y0_IMAG_KK),
                    "note": (
                        "six frequency bands each with a DIFFERENT impedance and a "
                        "non-zero imaginary part; k == frequency index k=0..5."
                    ),
                },
                "boundary_model": "local_impedance",
                "is_locally_reacting": True,
            },
        },
    }


def donor_root_from_env() -> Path:
    """從呼叫端帶進來的 ``PYTHONPATH`` 找出 donor 工作樹的根。

    刻意不碰 ``sys.path``。找不到就當場炸——找錯一棵樹比找不到更糟。
    """
    for item in sys.path:
        if not item:
            continue
        candidate = Path(item)
        if (candidate / "lib" / "physics" / "ism_patch.py").is_file():
            return candidate.resolve()
    raise SystemExit("找不到 donor 工作樹：請用 PYTHONPATH=<donor 工作樹> 跑這一支")


def _git(root: Path, *args: str) -> str:
    """在 donor 工作樹上跑一個唯讀的 git 指令。"""
    done = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def donor_provenance(root: Path) -> dict[str, object]:
    """量 donor 的出身：記號解析出來的 commit、樹乾不乾淨、HEAD 在記號上。"""
    commit = _git(root, "rev-parse", f"{DONOR_TAG}^{{commit}}")
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise SystemExit(
            f"{root} 不乾淨（status --porcelain 有 {len(dirty.splitlines())} 行）"
            f"——在一棵被改過的樹上跑出來的值不是 {DONOR_TAG} 的值，拒絕產出"
        )
    if _git(root, "rev-parse", "HEAD") != commit:
        raise SystemExit(
            f"{root} 的 HEAD 不是 {DONOR_TAG}（{DONOR_TAG}^{{commit}}={commit}）"
            "——這一支只准在記號那一筆上跑"
        )
    return {"tag": DONOR_TAG, "commit": commit, "clean": True}


def _module_version(module: object) -> str:
    """一個模組的 ``__version__``，沒有就回 ``"unknown"``。"""
    return str(getattr(module, "__version__", "unknown"))


def environment() -> dict[str, object]:
    """這一跑的環境：python 與依賴版本、JAX 的後端與 x64 開關、platform。

    dtype 與數值跟著 jax/jaxlib/numpy 版本與 x64 開關走，必須記下來；platform 是唯一
    允許隨機器變、又不該進 ``paths`` 的東西（答案檔的檔頭，不是路徑值）。dtype 六格在
    ``_build_case`` 現場量（``str(arr.dtype)``），這裡只放版本與開關。
    """
    jax = importlib.import_module("jax")
    versions: dict[str, str] = {}
    for name in RECORDED_PACKAGES:
        module = importlib.import_module(name)
        versions[name] = _module_version(module)
    return {
        "python": sys.version,
        "versions": versions,
        "backend": str(jax.default_backend()),
        "x64": bool(jax.config.jax_enable_x64),
        "platform": {
            "system": _platform.system(),
            "release": _platform.release(),
            "machine": _platform.machine(),
            "python_implementation": _platform.python_implementation(),
        },
    }


def command_record() -> dict[str, object]:
    """記這一跑怎麼重現：直譯器、argv、以及三個環境變數的實際值（沒設記 null）。"""
    return {
        "interpreter": sys.executable,
        "argv": sys.argv,
        "env": {name: os.environ.get(name) for name in RECORDED_ENV_VARS},
    }


def _guard_nonfinite(value: float) -> dict[str, object] | None:
    """JSON 裡不准出現 Infinity/NaN——記 null 並附原因。

    回傳 ``None`` 表示值有限（呼叫端照常存 dec/hex），否則回一個把該格標成 null 並寫
    明原因的 dict。
    """
    if math.isinf(float(value)):
        return {"dec": None, "hex": None, "null_reason": "inf"}
    if math.isnan(float(value)):
        return {"dec": None, "hex": None, "null_reason": "nan"}
    return None


def _real_hex_and_dec(value: float) -> dict[str, object]:
    """一個實數同時存十進位與十六進位兩格（null-guarded）。"""
    guard = _guard_nonfinite(value)
    if guard is not None:
        return guard
    return {"dec": repr(float(value)), "hex": float(value).hex()}


def _complex_hex_and_dec(value: complex) -> dict[str, object]:
    """一個複數存實／虛／abs 三格，各含 dec 與 hex（null-guarded 防 Infinity/NaN）。"""
    out: dict[str, object] = {}
    for key, scalar in (("real", float(value.real)), ("imag", float(value.imag))):
        guard = _guard_nonfinite(scalar)
        out[key] = (
            guard
            if guard is not None
            else {"dec": repr(scalar), "hex": float(scalar).hex()}
        )
    absv = float(abs(value))
    guard = _guard_nonfinite(absv)
    out["abs"] = (
        guard
        if guard is not None
        else {"dec": repr(absv), "hex": float(absv).hex()}
    )
    return out


def _donor_modules() -> dict[str, ModuleType]:
    """載入 donor 的六支模組（字串名字，不碰 ``sys.path``），回名字對模組的表。"""
    return {
        name: importlib.import_module(f"lib.{path}")
        for name, path in (
            ("ism", "physics.ism_patch"),
            ("proxy", "physics.proxy_solver"),
            ("adapter", "materials.adapter"),
            ("registry", "materials.registry"),
            ("response", "materials.response"),
            ("patch", "geometry.patch_materials"),
        )
    }


def _material_grids(mods: dict[str, ModuleType], freq_axis: object, case: str) -> object:
    """建一組材料的六面牆網格（flat 全同／varied 的 y0 隨頻帶）。"""
    flat_mat = mods["registry"].constant_impedance(
        freq_axis, complex(MATERIAL_Z_FLAT_REAL, 0.0), material_id=MATERIAL_ID_FLAT
    )
    if case == "varied":
        z_arr = _np_module().asarray(
            [complex(VARIED_Y0_REAL_KK[k], VARIED_Y0_IMAG_KK[k]) for k in range(len(FREQS_HZ))],
            dtype=complex,
        )
        y0_mat = mods["response"].MaterialResponse.from_impedance(
            z_arr, freq_axis, material_id=MATERIAL_ID_VARIED
        )
        cells = {"floor": flat_mat, "ceiling": flat_mat, "x0": flat_mat,
                 "xL": flat_mat, "y0": y0_mat, "yL": flat_mat}
        return tuple(mods["patch"].WallPatchGrid(w, 1, 1, (cells[w],)) for w in CANONICAL_WALL_ORDER)
    return tuple(
        mods["patch"].WallPatchGrid(w, 1, 1, (flat_mat,)) for w in CANONICAL_WALL_ORDER
    )


def _np_module() -> ModuleType:
    """載入 numpy（donor 的數值型別），只在此處用，新家不搬進來。"""
    return importlib.import_module("numpy")


def _jnp_module() -> ModuleType:
    """載入 jax.numpy（上一代的數值後端），只在此處用。"""
    return importlib.import_module("jax.numpy")


def _build_case(case: str) -> tuple[dict[str, object], dict[str, str]]:
    """跑 donor 對一組材料算反射乘積與路徑壓力，組出 ``paths`` 與 ``totals``。

    回傳 ``(payload, dtypes)``：``paths`` 與 ``totals`` 必須跟凍結原件逐位元相同；
    ``dtypes`` 是 dtype 六格現場量（``str(arr.dtype)``）。這一段照凍結原著
    ``run/amplitude_reference.py`` 的 ``_build_case``：同一組 import、同一條公式、同一個
    dtype（complex64），所以數值逐位元相同。
    """
    np = _np_module()
    jnp = _jnp_module()
    mods = _donor_modules()

    freq_axis = mods["response"].FrequencyAxis.from_hz(
        np.asarray(FREQS_HZ, dtype=float), resolution="custom"
    )
    wpm = mods["patch"].WallPatchMaterials(_material_grids(mods, freq_axis, case), freq_axis)
    mods["ism"].validate_patch_materials(wpm, RHO_C)
    Z_cells, _offsets = mods["ism"].flatten_cell_Z(wpm)

    geom = mods["proxy"].ShoeboxGeometry.of(ROOM_LX, ROOM_LY, ROOM_LZ)
    src = np.array(SOURCE_XYZ, dtype=float)
    rec = np.array([RECEIVER_XYZ], dtype=float)
    dims = np.array([ROOM_LX, ROOM_LY, ROOM_LZ], dtype=float)
    paths = mods["ism"].precompute_shoebox_patch_paths(
        geom, src, rec, tuple(tuple(s) for s in GRID_SHAPES),
        max_order=MAX_ORDER, sound_speed=SOUND_SPEED, wall_edges=None,
    )
    result = mods["ism"].solve_ism_shoebox_patched(
        paths, Z_cells, freq_axis, rho_c=RHO_C, directivity=None
    )
    image_paths = mods["ism"]._enumerate_image_paths(src, dims, MAX_ORDER)
    if len(image_paths) != int(paths.dist.shape[1]):
        raise SystemExit(
            f"_enumerate_image_paths 回 {len(image_paths)} 條，"
            f"precompute 回 {paths.dist.shape[1]} 條——兩邊不一致，拒絕產出"
        )

    # 照 donor 的式子重現 path_pressure（同 dtype）；refl = ∏R、phase=exp(−iωτ)。
    gid = jnp.asarray(paths.cell_gid)
    cos = jnp.asarray(paths.cos)[..., None]
    vmask = jnp.asarray(paths.valid_mask)[None, :, :, None]
    rb = mods["adapter"].z_to_r_from_cos(Z_cells[gid], cos, RHO_C)
    rb = jnp.where(vmask, rb, 1.0 + 0.0j)
    refl = jnp.prod(rb, axis=2)
    dist_safe = jnp.maximum(jnp.asarray(paths.dist), 1e-12)
    omega = 2.0 * jnp.pi * jnp.asarray(freq_axis.freqs_hz)
    phase = jnp.exp(-1j * omega[None, None, :] * jnp.asarray(paths.delay)[:, :, None])
    path_pressure = (1.0 / dist_safe)[:, :, None] * refl * phase
    reproduced_pressure = jnp.sum(path_pressure, axis=1)
    n_freq = int(freq_axis.n_freq)

    # 抽出純 Python 值（index 都在這裡，``paths``／``result`` 是 donor 傳回的動態值）。
    n_paths = len(image_paths)
    dists = [float(paths.dist[0, i]) for i in range(n_paths)]
    delays = [float(paths.delay[0, i]) for i in range(n_paths)]
    orders = [int(ip.order) for ip in image_paths]
    identities = [[int(v) for v in ip.identity] for ip in image_paths]
    refl_rows = [[complex(result.reflection_product[0, i, f]) for f in range(n_freq)] for i in range(n_paths)]
    pp_rows = [[complex(path_pressure[0, i, f]) for f in range(n_freq)] for i in range(n_paths)]
    total_pressure = [complex(result.pressure[0, f]) for f in range(n_freq)]
    total_repro = [complex(reproduced_pressure[0, f]) for f in range(n_freq)]
    total_direct = [float(result.ism_direct_E[0, f]) for f in range(n_freq)]
    total_rev = [float(result.ism_rev_E[0, f]) for f in range(n_freq)]

    payload: dict[str, object] = {
        "paths": _format_paths(dists, delays, orders, identities, refl_rows, pp_rows),
        "totals": _format_totals(total_pressure, total_repro, total_direct, total_rev),
        "parameters": frozen_parameters(),
    }
    return payload, _live_dtypes(Z_cells, result, path_pressure)


def _format_paths(
    dists: list[float],
    delays: list[float],
    orders: list[int],
    identities: list[list[int]],
    refl_rows: list[list[complex]],
    pp_rows: list[list[complex]],
) -> list[dict[str, object]]:
    """純 Python 值 → 每條路徑的 JSON 形狀（identity 是清單，照凍結原件）。"""
    records: list[dict[str, object]] = []
    for index in range(len(orders)):
        records.append(
            {
                "index": index,
                "order": orders[index],
                "identity": identities[index],
                "dist_m": _real_hex_and_dec(dists[index]),
                "delay_s": _real_hex_and_dec(delays[index]),
                "reflection_product": [_complex_hex_and_dec(c) for c in refl_rows[index]],
                "path_pressure": [_complex_hex_and_dec(c) for c in pp_rows[index]],
            }
        )
    return records


def _format_totals(
    pressure: list[complex],
    repro: list[complex],
    direct_e: list[float],
    rev_e: list[float],
) -> dict[str, object]:
    """接收點 0 的 totals（pressure、重現加總、能量，照凍結原著四格）。"""
    return {
        "pressure": [_complex_hex_and_dec(c) for c in pressure],
        "reproduced_pressure_sum": [_complex_hex_and_dec(c) for c in repro],
        "ism_direct_E": [_real_hex_and_dec(v) for v in direct_e],
        "ism_rev_E": [_real_hex_and_dec(v) for v in rev_e],
    }


def _live_dtypes(Z_cells: object, result: object, path_pressure: object) -> dict[str, str]:
    """dtype 六格現場量（``str(arr.dtype)``），不手打字串。"""
    return {
        "Z_cells": str(getattr(Z_cells, "dtype", None)),
        "reflection_product": str(getattr(getattr(result, "reflection_product"), "dtype", None)),
        "pressure": str(getattr(getattr(result, "pressure"), "dtype", None)),
        "path_pressure_reproduced": str(getattr(path_pressure, "dtype", None)),
        "ism_direct_E": str(getattr(getattr(result, "ism_direct_E"), "dtype", None)),
        "ism_rev_E": str(getattr(getattr(result, "ism_rev_E"), "dtype", None)),
    }


def build_payload(root: Path, case: str) -> dict[str, object]:
    """組成一組材料答案檔的完整內容（檔頭 ＋ paths ＋ totals）。"""
    payload, dtypes = _build_case(case)
    env = environment()
    env["dtypes"] = dtypes
    return {
        "schema": ANSWER_SCHEMA,
        "note": (
            "由 blueprint/generate_amplitude_answers.py 在唯讀的 donor 工作樹上跑出來，"
            "不准手改；要改就重跑產生器。paths 與 totals 跟第 4 段凍結原件逐位元相同。"
        ),
        "donor": donor_provenance(root),
        "env": env,
        "command": command_record(),
        "parameters": payload["parameters"],
        "paths": payload["paths"],
        "totals": payload["totals"],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    """``--case``（flat|varied）與 ``--out``（答案檔要寫到哪裡）。"""
    parser = argparse.ArgumentParser(
        description="把 donor 的振幅（反射乘積與路徑壓力）跑成標準答案檔"
    )
    parser.add_argument("--case", required=True, choices=list(CASES))
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """跑一次：量 donor 出身、跑一組材料、寫檔。"""
    args = parse_args(argv)
    root = donor_root_from_env()
    payload = build_payload(root, args.case)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
