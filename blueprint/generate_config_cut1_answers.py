#!/usr/bin/env python3
"""把上一代（v2）那 11 支 config 模組的公開值跑出來，寫成新家考卷要用的標準答案檔。

**為什麼要有這一支。** 票 #127 第 1 刀照抄了 v2 的 11 支模組（``art_lane``／``art_rt_guard``／
``authoring_defaults``／``crossover_axis``／``default_geometry``／``ism_lane``／
``phase2_report_bands``／``receiver_grid``／``source_reference``／``speaker_directivity``／
``spl_output``）。那 11 支在 v2 的考卷**一支都帶不走**（每一支考卷都還 import 了別的層或搬走
的模組），所以它們的裁判改由這一份標準答案接手：常數比凍結值、函式比「同一組輸入下的輸出」。

**數字以後用跑的，不要用抄的**（票 #138 的原則）。這支程式在唯讀的 v2 工作樹上把值真的
跑出來，寫成 ``blueprint`` 底下那一份標準答案（它的產物跟它住同一層）；考卷讀那個檔比對，
人不碰數字。

**怎麼跑**（v2 的 venv 自己一套依賴，不要拿 v3 的環境）：

    PYTHONPATH=<v2 工作樹> <v2 工作樹>/.venv/bin/python \
        blueprint/generate_config_cut1_answers.py --out blueprint/config_cut1_answers.json

``PYTHONPATH`` 是**在呼叫時**帶進去的環境變數，這支程式碼裡一個字都不碰 ``sys.path``
（規矩卡 ``uv-single-entrypoint`` 第二條把 ``sys.path`` 的 append／insert／extend 與指派
一律判紅，而 ``blueprint`` 底下每一支 .py 也在它的掃描面裡）。

**為什麼要在 ``sys.modules`` 放 stub。** 載入 ``lib.config`` 底下任何一支都會先執行那個
套件的門面檔，而那 5 行 re-export 會把 JAX 拉進來——那不是新家的形狀，也會讓
「這一支模組自己有沒有碰 JAX」這個問題問不出答案。先在 ``sys.modules`` 放 ``lib`` 與
``lib.config`` 兩個空殼（只設 ``__path__``），再逐一載入，就等於跳過門面。那個空殼的目錄
由呼叫端的 ``PYTHONPATH`` 決定（``lib`` 套件登記在哪，就從哪裡拿）。

**答案檔裡不准有絕對路徑或 ``~/``。** 它住在 ``blueprint`` 底下——那一層是
``refs-and-links-resolve`` 與 `style-guard` 都扣掉的前綴（那些檔是資料、不是這棵樹的引用），
所以這一條規矩在這裡靠人守：產生器只寫 donor 記號、完整 commit sha 與值本身。
所以輸出只有：donor 記號、完整 commit sha（十六進位，解析得到）、以及值本身。
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import json
import subprocess
import sys
import types
import warnings
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final

DONOR_TAG: Final[str] = "v3-donor"

# 這 11 支（第 2 刀那 9 支讀檔模組不在這裡）。
MODULES: Final[tuple[str, ...]] = (
    "art_lane",
    "art_rt_guard",
    "authoring_defaults",
    "crossover_axis",
    "default_geometry",
    "ism_lane",
    "phase2_report_bands",
    "receiver_grid",
    "source_reference",
    "speaker_directivity",
    "spl_output",
)

# 這一刀只碰得到一個「算出來的」值：`4*math.pi`。以運算式記號存，考卷自己算出來比，
# 不要在這裡寫成小數（寫成小數就等於把一個算出來的東西抄成一個常數）。
PI_EXPR: Final[str] = "4*math.pi"

def _donor_commit(v2_root: Path) -> str:
    """v2 工作樹的完整 commit sha（答案檔的檔頭，驗得到的那一個）。"""
    done = subprocess.run(
        ["git", "-C", str(v2_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def stub_v2_packages(v2_root: Path) -> None:
    """在 ``sys.modules`` 放 ``lib`` 與 ``lib.config`` 的空殼，跳過 v2 的門面檔。

    只設 ``__path__``（告訴 import 系統「這是一個套件、它的檔在這兩個目錄底下」），
    不執行那個套件的門面檔——那 5 行 re-export 會把 JAX 拉進來。
    """
    lib_pkg = types.ModuleType("lib")
    lib_pkg.__path__ = [str(v2_root / "lib")]
    config_pkg = types.ModuleType("lib.config")
    config_pkg.__path__ = [str(v2_root / "lib" / "config")]
    sys.modules["lib"] = lib_pkg
    sys.modules["lib.config"] = config_pkg


def donor_root_from_env() -> Path:
    """從呼叫端帶進來的 ``PYTHONPATH`` 找出 v2 工作樹的根。

    刻意不碰 ``sys.path``：規矩卡把那個字串整族判紅。找不到就當場炸——找錯一棵樹
    比找不到更糟（會產出一份看起來很合理的答案檔）。
    """
    entries = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    for entry in entries:
        candidate = Path(entry)
        if (candidate / "lib" / "config" / "art_lane.py").is_file():
            return candidate.resolve()
    raise SystemExit("找不到 v2 工作樹：請用 PYTHONPATH=<v2 工作樹> 跑這一支")


def load_module(name: str) -> types.ModuleType:
    """載入 v2 的 ``lib.config.<name>``（門面已被跳過）。"""
    return importlib.import_module(f"lib.config.{name}")


def encode(value: object) -> dict[str, object]:
    """把一個值編成可以 JSON 化、又能逐位元還原的記號。

    * ``float`` -> ``{"kind": "float", "hex": float.hex()}``：``hex()`` 是精確的、
      可以 ``float.fromhex()`` 還原，跨平台不會漂（十進位字串會）。
    * ``tuple``／``list`` -> 同一種 ``list`` 記號：上游那幾支的 tuple 與 list 在
      ``==`` 底下本來就互通，而「哪一種」是結構不是值（這一刀的合約是數值一致、
      結構隨便改）。
    * dataclass 與 pydantic 模型 -> 欄位表（不比記憶體位址）。
    * 其餘（``int``／``str``／``bool``／``None``）原樣存。
    """
    if isinstance(value, bool):
        return {"kind": "scalar", "value": value}
    if isinstance(value, float):
        out: dict[str, object] = {"kind": "float", "hex": value.hex()}
        if value == 4.0 * math.pi:
            out["expr"] = PI_EXPR
        return out
    if isinstance(value, (int, str)) or value is None:
        return {"kind": "scalar", "value": value}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "object",
            "fields": {field.name: encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, dict):
        return {"kind": "dict", "items": {str(key): encode(item) for key, item in value.items()}}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        # pydantic 的模型（這一刀只有 `SplOutputConfig`）：比它自己的欄位表。
        return {"kind": "model", "fields": encode(dump())}
    if isinstance(value, (tuple, list)):
        return {"kind": "list", "items": [encode(item) for item in value]}
    raise TypeError(f"這個值編不出來：{type(value).__name__}")


def public_constant_names(module: types.ModuleType) -> list[str]:
    """模組裡的全大寫公開常數（名字不以 ``_`` 開頭、也不是匯入進來的模組）。"""
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and name.isupper() and not isinstance(value, types.ModuleType)
    )


def constants_block(module: types.ModuleType) -> dict[str, object]:
    """這一支模組的每一個公開常數各一筆凍結值。"""
    return {name: encode(getattr(module, name)) for name in public_constant_names(module)}


def call_block(fn: Callable[..., object], args: dict[str, object]) -> dict[str, object]:
    """用 ``args`` 呼叫 ``fn``：回值編成記號，炸掉記型別與訊息，警告也記下來。"""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = fn(**args)
    except Exception as exc:  # noqa: BLE001  # expires=2026-12-08 reason=這一支要記錄「炸什麼」而不是讓它往外炸，例外的種類不影響判準
        return {"raised": {"type": type(exc).__name__, "message": str(exc)}}
    block: dict[str, object] = {"value": encode(value)}
    if caught:
        block["warned"] = [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ]
    return block


def probe(args: dict[str, object], expected: dict[str, object]) -> dict[str, object]:
    """一筆探針：輸入參數，以及那組輸入底下的結果。"""
    return {"args": args, "expected": expected}


def art_lane_probes(module: types.ModuleType) -> list[dict[str, object]]:
    """``art_lane`` 兩支護欄函式：負值／正常／剛好在上限／跨過上限。

    ``ART_P_CAP = 8192``、patch 數是 ``6*n*n``，所以 ``n=36`` 剛好塞得下（7776）、
    ``n=37`` 就跨過去（8214）——邊界用跑的找，不用心算。
    """
    records = [
        probe({"n_per_wall": n}, call_block(module.guard_art_patch_count, {"n_per_wall": n}))
        for n in (0, -1, 1, 6, 36, 37, 116)
    ]
    records.append(
        probe(
            {"n_per_wall": 37, "context": "ART-CUSTOM"},
            call_block(module.guard_art_patch_count, {"n_per_wall": 37, "context": "ART-CUSTOM"}),
        )
    )
    records.extend(
        probe({"n_tris": n}, call_block(module.guard_polygon_art_patch_count, {"n_tris": n}))
        for n in (0, -5, 1, 8192, 8193)
    )
    records.append(
        probe(
            {"n_tris": 5000, "context": "polygon ART-CUSTOM"},
            call_block(
                module.guard_polygon_art_patch_count,
                {"n_tris": 5000, "context": "polygon ART-CUSTOM"},
            ),
        )
    )
    return records


def authoring_defaults_probes(module: types.ModuleType) -> list[dict[str, object]]:
    """``authoring_defaults`` 三支 id 助手：正常輸入與空字串。"""
    records = [
        probe({"wall_id": wall}, call_block(module.app_wall_id, {"wall_id": wall}))
        for wall in ("y0", "north", "")
    ]
    records.extend(
        probe(
            {"wall_id": wall, "row": row, "col": col},
            call_block(module.app_patch_id, {"wall_id": wall, "row": row, "col": col}),
        )
        for wall, row, col in (("y0", 1, 2), ("x1", 0, 0), ("", -3, 7))
    )
    records.extend(
        probe({"guid": guid}, call_block(module.rhino_boundary_id, {"guid": guid}))
        for guid in ("7d8ecb8f-0000-4000-8000-000000000000", "", "no-dashes")
    )
    return records


def crossover_axis_probes(module: types.ModuleType) -> list[dict[str, object]]:
    """``build_canonical_crossover``：預設值與兩種覆寫。"""
    cases: list[dict[str, object]] = [
        {"seam_f_s": 321.0},
        {"seam_f_s": 321.0, "t_c_center": 0.08, "t_c_fade": 0.02},
        {"seam_f_s": 250.0, "t_c_center": 0.08, "t_c_fade": 0.013},
        {"seam_f_s": 0.0},
    ]
    return [probe(args, call_block(module.build_canonical_crossover, args)) for args in cases]


def speaker_directivity_probes(module: types.ModuleType) -> list[dict[str, object]]:
    """``resolve_speaker_directivity``：開／關、三種型別、覆寫值、不支援的字串、壞的覆寫值。

    一筆一筆把參數寫出來而不是用推導：這幾組是「哪幾種輸入值得探」的決定，寫成一張
    看得見的表比藏在推導裡好。
    """
    resolve = module.resolve_speaker_directivity
    cases: list[dict[str, object]] = [
        {"enabled": True},
        {"enabled": True, "speaker_type": "bookshelf"},
        {"enabled": True, "speaker_type": "floorstanding"},
        {"enabled": True, "speaker_type": "inwall"},
        {"enabled": False},
        {"enabled": False, "speaker_type": "inwall"},
        {"enabled": True, "speaker_type": "  Bookshelf  "},
        {"enabled": True, "speaker_type": "bookshelf", "baffle_width_m": 0.21},
        {"enabled": True, "speaker_type": "bookshelf", "piston_radius_m": 0.05},
        {"enabled": True, "speaker_type": "floorstanding", "baffle_width_m": 0.30, "piston_radius_m": 0.10},
        {"enabled": True, "speaker_type": "unknown-box"},
        {"enabled": False, "speaker_type": "soundbar"},
        {"enabled": True, "speaker_type": "bookshelf", "baffle_width_m": 0.0},
        {"enabled": True, "speaker_type": "bookshelf", "piston_radius_m": -0.1},
        {"enabled": True, "speaker_type": "bookshelf", "baffle_width_m": float("nan")},
        {"enabled": True, "speaker_type": "bookshelf", "baffle_width_m": float("inf")},
        {"enabled": True, "speaker_type": "floorstanding", "piston_radius_m": True},
    ]
    records = [probe(args, call_block(resolve, args)) for args in cases]
    records.append(
        probe(
            {"presets": sorted(module.SPEAKER_PRESETS)},
            {"preset_fields": {name: vars(preset) for name, preset in module.SPEAKER_PRESETS.items()}},
        )
    )
    return records


def _config_block(cls: object, args: dict[str, object]) -> dict[str, object]:
    """建一個 ``SplOutputConfig``，連它的兩個屬性一起收。

    屬性要在**還沒編碼之前**、從真的物件上讀（編碼之後它只是一個 JSON 記號，沒有屬性）。
    """
    block = call_block(cls, args)  # type: ignore[arg-type]  # expires=2026-12-08 reason=cls 是從模組上拿下來的類別物件，型別樁只知道它是 object；這裡要的就是「執行期那一格」
    if "value" not in block:
        return block
    try:
        built = cls(**args)  # type: ignore[operator]  # expires=2026-12-08 reason=同上一行，執行期才知道的類別物件，型別樁表達不了
    except Exception:  # noqa: BLE001  # expires=2026-12-08 reason=建不起來的那幾組已經由 call_block 記下例外了，這裡只是不再多收屬性
        return block
    for name in ("l_ref_db", "is_relative"):
        block[name] = _read_attr(built, name)
    return block


def _read_attr(built: object, name: str) -> dict[str, object]:
    """讀一個屬性；讀的當下炸掉（例如守門）就記成例外。"""
    try:
        value = getattr(built, name)
    except Exception as exc:  # noqa: BLE001  # expires=2026-12-08 reason=屬性本身就是守門（值不合法時丟例外），這裡要記錄它而不是讓它往外炸
        return {"raised": {"type": type(exc).__name__, "message": str(exc)}}
    return {"value": encode(value)}


def spl_output_probes(module: types.ModuleType) -> list[dict[str, object]]:
    """``SplOutputConfig`` 的建構與兩個屬性：預設、相加、廣播、非有限、不相等守門。"""
    cls = module.SplOutputConfig
    cases: list[dict[str, object]] = [
        {},
        {"sensitivity_db": (0.0,), "playback_level_db": 0.0},
        {"sensitivity_db": (88.0,), "playback_level_db": -6.0},
        {"sensitivity_db": (85.0, 85.0), "playback_level_db": 0.0},
        {"sensitivity_db": (85.0, 85.0 + module.FLAT_EQUAL_TOL_DB / 2.0)},
        {"sensitivity_db": (85.0, 88.0)},
        {"sensitivity_db": (float("nan"),)},
        {"sensitivity_db": (float("inf"),)},
        {"sensitivity_db": (float("-inf"),)},
        {"playback_level_db": float("nan")},
        {"playback_level_db": float("inf")},
        {"sensitivity_db": ()},
    ]
    return [probe(args, _config_block(cls, args)) for args in cases]


PROBE_BUILDERS: Final[dict[str, Callable[[types.ModuleType], list[dict[str, object]]]]] = {
    "art_lane": art_lane_probes,
    "authoring_defaults": authoring_defaults_probes,
    "crossover_axis": crossover_axis_probes,
    "speaker_directivity": speaker_directivity_probes,
    "spl_output": spl_output_probes,
}


def public_callable_names(module: types.ModuleType) -> list[str]:
    """模組裡自己定義的公開函式／類別（不含匯入進來的名字）。"""
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (isinstance(value, type) or callable(value))
        and getattr(value, "__module__", None) == module.__name__
    )


def coverage_problems() -> list[str]:
    """每一支有公開函式／類別的模組都要有探針——沒有探針的公開函式不准靜靜溜過去。

    這一條是給未來的人看的：第 2 刀或多加一支模組進來的時候，忘了寫探針會在這裡回紅，
    而不是產出一份「看起來很完整」的答案檔。
    """
    problems: list[str] = []
    for name in MODULES:
        module = load_module(name)
        wanted = public_callable_names(module)
        if wanted and name not in PROBE_BUILDERS:
            problems.append(f"{name} 有公開函式／類別 {wanted}，但沒有探針產生器")
    return problems


def build_payload(v2_root: Path) -> dict[str, object]:
    """把 11 支的常數與探針全部跑出來，組成答案檔的內容。"""
    problems = coverage_problems()
    if problems:
        raise SystemExit("探針沒蓋完：" + "；".join(problems))
    per_module: dict[str, object] = {}
    for name in MODULES:
        module = load_module(name)
        entry: dict[str, object] = {"constants": constants_block(module)}
        builder = PROBE_BUILDERS.get(name)
        if builder is not None:
            entry["probes"] = builder(module)
        per_module[name] = entry
    return {
        "donor": {"tag": DONOR_TAG, "commit": _donor_commit(v2_root), "modules": list(MODULES)},
        "note": "由 blueprint/generate_config_cut1_answers.py 在唯讀的 v2 工作樹上跑出來，不准手改。",
        "modules": per_module,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    """``--out``：答案檔要寫到哪裡。"""
    parser = argparse.ArgumentParser(description="把 v2 config 的公開值跑成標準答案檔")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """跑一次：載入 v2、蒐集值、寫檔、印出筆數。"""
    args = parse_args(argv)
    v2_root = donor_root_from_env()
    stub_v2_packages(v2_root)
    payload = build_payload(v2_root)
    modules = payload["modules"]
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"寫出 {args.out}（{len(text)} bytes）")
    for name in MODULES:
        entry = modules[name]  # type: ignore[index]  # expires=2026-12-08 reason=modules 是這支剛剛組出來的 dict，形狀由 build_payload 保證
        probes = entry.get("probes", [])
        print(f"  {name}: 常數 {len(entry['constants'])} 筆，探針 {len(probes)} 筆")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
