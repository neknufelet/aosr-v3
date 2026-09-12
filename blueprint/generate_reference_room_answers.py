"""把凍結的 v2 donor 的 shoebox 反射路徑跑出來，寫成新家考卷要用的標準答案檔。

**為什麼要有這一支。** 票 #175 在 v3 工作樹裡長出一支「獨立幾何」模組
（``blueprint/reference_room_geometry.py``，只 import 標準庫），它把直達 ＋ 六面牆各一次
反射的七條路徑重新算一遍；要驗它算得對，得先有一份「上一代真的算出來的」凍結值當標準
答案。這一支就是產生那份答案的產生器：它在唯讀的 v2 donor 工作樹上跑
``precompute_shoebox_patch_paths``（直達＋六面牆一次反射）與 ``_enumerate_image_paths``
（每條路徑的 identity），把純計算結果寫成 ``blueprint/reference_room_answers.json``。

**參數寫死在檔頭一份 ``FROZEN_PARAMETERS``。** 房間 (6,4,3) m、聲源 (1.5,1.0,1.2)、
接收點 (4.0,3.0,1.5)、聲速 343.0 m/s、max_order 1、六面牆各 (1,1)。這些是票 #175 的
合約，這一支不改它們。這一個 dict 是**單一來源**：答案檔的 ``parameters`` 直接由它寫出，
考卷也 import 它逐格比對——不會在產生器、獨立幾何、答案檔各長一份各自漂的副本
（獨立幾何一支尺寸都不寫死，全部靠呼叫端餵）。

**donor 的出身用量的，不是宣稱**：解析 ``v3-donor`` 這個記號、確認那棵樹乾淨、確認 HEAD
就在記號上，三格一起寫進檔頭（照 ``blueprint/generate_materials_cut1_answers.py`` 的同一套）。
donor 不在 CI 上，所以那個 sha 沒有機器在守——它是「產生時量的 ＋ 人工可重跑」。

**答案檔頂層**：``schema``（1）、``note``（由…產生、不准手改）、``donor``、``env``
（python 與 jax/jaxlib/numpy/flax 版本、JAX 後端與 x64 開關、platform）、``command``
（直譯器 ``interpreter``、``argv``、三個環境變數的實際值——沒設就記 null）、``parameters``
（含 ``units`` 與 ``convention`` 文字）、``paths``（每條 ``index, order, identity,
image_xyz, dist_m, delay_s``，浮點同時存 ``dec`` 與 ``hex``）。**不存 platform 以外會隨
機器變的東西**進 ``paths``：paths 只放純計算結果的浮點值與 identity。

**路徑數值同時存 ``dec`` 與 ``hex``**：``hex()`` 是精確的、可 ``float.fromhex()`` 還原，
跨平台不漂；``dec`` 是給人看的 ``repr``。考卷用 hex 逐字串相等比對，不是 isclose。

**這一支刻意不印任何東西。** ``style-guard`` 只准具名的輸出層 ``print``，而這一支還沒有
登記；要看它跑出什麼就讀產物本身。跑壞了走離開碼。

**怎麼跑**（donor 自己一套 venv，在 v3 外面；``cd`` 在 v3 repo 根）::

    PYTHONPATH=/home/florian/aosr-v3-work/175/donor JAX_PLATFORMS=cpu \\
        PYTHONDONTWRITEBYTECODE=1 <donor venv>/bin/python \\
        -m blueprint.generate_reference_room_answers --out blueprint/reference_room_answers.json

``PYTHONPATH`` 是**在呼叫時**帶進去的環境變數，這支程式碼裡一個字都不碰 ``sys.path``
（規矩卡 ``uv-single-entrypoint`` 第二條把那一族判紅）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform as _platform
import subprocess
import sys
from pathlib import Path
from typing import Final

DONOR_TAG: Final[str] = "v3-donor"

ANSWER_SCHEMA: Final[int] = 1

# ── 規範牆順序（跟 v2 的 CANONICAL_WALL_ORDER 一致，也是 convention 文字的那一段）────
CANONICAL_WALL_ORDER: Final[tuple[str, ...]] = (
    "floor",
    "ceiling",
    "x0",
    "xL",
    "y0",
    "yL",
)

# ── 凍結參數（票 #175 的合約，不准改；單一來源，見模組說明的第二段）────────────────
ROOM_LX: Final[float] = 6.0
ROOM_LY: Final[float] = 4.0
ROOM_LZ: Final[float] = 3.0
SOURCE_XYZ: Final[tuple[float, float, float]] = (1.5, 1.0, 1.2)
RECEIVER_XYZ: Final[tuple[float, float, float]] = (4.0, 3.0, 1.5)
SOUND_SPEED: Final[float] = 343.0
MAX_ORDER: Final[int] = 1
GRID_SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
)
UNITS: Final[str] = "metres (m), seconds (s), metres/second (m/s)"
CONVENTION: Final[str] = (
    "origin at room corner; x in [0,Lx], y in [0,Ly], z in [0,Lz]; "
    "z=0 is floor, z=Lz ceiling. Wall ids in canonical order: "
    + ", ".join(CANONICAL_WALL_ORDER)
    + "."
)

# 上面那些型別化名字拼成答案檔 ``parameters`` 那一塊的凍結內容。考卷 import 這一格
# （`from blueprint.generate_reference_room_answers import FROZEN_PARAMETERS`）逐格比對，
# 所以「答案檔的 parameters」跟「產生器的凍結常數」是同一份，不是兩份各自會漂。
FROZEN_PARAMETERS: Final[dict[str, object]] = {
    "room": {"Lx_m": ROOM_LX, "Ly_m": ROOM_LY, "Lz_m": ROOM_LZ},
    "source_xyz_m": {"x": SOURCE_XYZ[0], "y": SOURCE_XYZ[1], "z": SOURCE_XYZ[2]},
    "receiver_xyz_m": {
        "x": RECEIVER_XYZ[0],
        "y": RECEIVER_XYZ[1],
        "z": RECEIVER_XYZ[2],
    },
    "sound_speed_m_s": SOUND_SPEED,
    "max_order": MAX_ORDER,
    "grid_shapes": [list(shape) for shape in GRID_SHAPES],
    "units": UNITS,
    "convention": CONVENTION,
}

# 檔頭要記的依賴版本（dtype 與數值跟著它們走，所以要記下來比對）。
RECORDED_PACKAGES: Final[tuple[str, ...]] = ("jax", "jaxlib", "numpy", "flax")

# ``command`` 檔頭要記的三個環境變數（這一跑能不能重現的鑰匙）。沒設就記 null。
RECORDED_ENV_VARS: Final[tuple[str, ...]] = (
    "PYTHONPATH",
    "JAX_PLATFORMS",
    "PYTHONDONTWRITEBYTECODE",
)


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


def environment() -> dict[str, object]:
    """這一跑的環境：python 與依賴版本、JAX 的後端與 x64 開關、platform。

    這些是**答案的一部分**：dtype 與數值跟著 jax/jaxlib/numpy 版本與 x64 開關走，
    必須記下來；platform 是唯一允許隨機器變、又不該進 ``paths`` 的東西（答案檔的
    檔頭，不是路徑值）。
    """
    jax = importlib.import_module("jax")
    versions: dict[str, str] = {}
    for name in RECORDED_PACKAGES:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
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


def parameters() -> dict[str, object]:
    """凍結參數那一塊：回 ``FROZEN_PARAMETERS``（單一來源，不另寫一份副本）。"""
    return dict(FROZEN_PARAMETERS)


def command_record() -> dict[str, object]:
    """記這一跑怎麼重現：直譯器、argv、以及三個環境變數的實際值（沒設記 null）。"""
    return {
        "interpreter": sys.executable,
        "argv": sys.argv,
        "env": {name: os.environ.get(name) for name in RECORDED_ENV_VARS},
    }


def _floats_hex_and_dec(value: float) -> dict[str, str]:
    """一個浮點同時存十進位與十六進位兩格。"""
    return {"dec": repr(float(value)), "hex": float(value).hex()}


def _paths() -> list[dict[str, object]]:
    """跑 donor 的 ``_enumerate_image_paths`` 與 ``precompute_shoebox_patch_paths``。

    兩支都是 donor 的程式，用 ``importlib.import_module`` 的**字串**名字載入（照
    ``generate_config_cut1_answers.py`` 的同一招）：路徑由呼叫端的 ``PYTHONPATH`` 決定，
    這一支不直接 ``from lib… import …``（不碰 ``sys.path``，也讓 mypy 不用去找 donor 那棵樹）。
    路徑順序由 ``_enumerate_image_paths`` 的排序（order、identity）決定，``precompute``
    用同一份順序，所以 index 一一對得上。
    """
    import numpy as np  # 只在此處 import：donor 的數值型別要用它，新家不搬進來

    ism_patch = importlib.import_module("lib.physics.ism_patch")
    proxy_solver = importlib.import_module("lib.physics.proxy_solver")
    enumerate_image_paths = ism_patch._enumerate_image_paths
    precompute_paths = ism_patch.precompute_shoebox_patch_paths
    shoebox_geometry = proxy_solver.ShoeboxGeometry

    geom = shoebox_geometry.of(ROOM_LX, ROOM_LY, ROOM_LZ)
    src = np.array(SOURCE_XYZ, dtype=float)
    rec = np.array([RECEIVER_XYZ], dtype=float)
    dims = np.array([ROOM_LX, ROOM_LY, ROOM_LZ], dtype=float)

    image_paths = enumerate_image_paths(src, dims, MAX_ORDER)
    result = precompute_paths(
        geom,
        src,
        rec,
        tuple(tuple(shape) for shape in GRID_SHAPES),
        max_order=MAX_ORDER,
        sound_speed=SOUND_SPEED,
        wall_edges=None,
    )

    if result.dist.shape[1] != len(image_paths):
        raise SystemExit(
            f"_enumerate_image_paths 回 {len(image_paths)} 條，"
            f"precompute 回 {result.dist.shape[1]} 條——兩邊不一致，拒絕產出"
        )

    paths: list[dict[str, object]] = []
    for index, ip in enumerate(image_paths):
        image_xyz = np.asarray(ip.image_xyz, dtype=float)
        dist_m = float(result.dist[0, index])
        delay_s = float(result.delay[0, index])
        paths.append(
            {
                "index": index,
                "order": int(ip.order),
                "identity": [int(value) for value in ip.identity],
                "image_xyz": {
                    "x": _floats_hex_and_dec(image_xyz[0]),
                    "y": _floats_hex_and_dec(image_xyz[1]),
                    "z": _floats_hex_and_dec(image_xyz[2]),
                },
                "dist_m": _floats_hex_and_dec(dist_m),
                "delay_s": _floats_hex_and_dec(delay_s),
            }
        )
    return paths


def build_payload(root: Path) -> dict[str, object]:
    """組成答案檔的內容。"""
    return {
        "schema": ANSWER_SCHEMA,
        "note": (
            "由 blueprint/generate_reference_room_answers.py 在唯讀的 donor 工作樹上跑出來，"
            "不准手改；要改就重跑產生器。"
        ),
        "donor": donor_provenance(root),
        "env": environment(),
        "command": command_record(),
        "parameters": parameters(),
        "paths": _paths(),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    """``--out``：答案檔要寫到哪裡。"""
    parser = argparse.ArgumentParser(description="把 donor 的 shoebox 反射路徑跑成標準答案檔")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """跑一次：量 donor 出身、蒐集路徑、寫檔。"""
    args = parse_args(argv)
    root = donor_root_from_env()
    payload = build_payload(root)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
