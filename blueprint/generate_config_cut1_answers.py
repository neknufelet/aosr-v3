#!/usr/bin/env python3
"""把上一代（v2）那 11 支 config 模組的公開值跑出來，寫成新家考卷要用的標準答案檔。

**為什麼要有這一支。** 票 #127 第 1 刀照抄了 v2 的 11 支模組（``art_lane``／``art_rt_guard``／
``authoring_defaults``／``crossover_axis``／``default_geometry``／``ism_lane``／
``phase2_report_bands``／``receiver_grid``／``source_reference``／``speaker_directivity``／
``spl_output``）。那 11 支在 v2 的考卷**一支都帶不走**（每一支考卷都還 import 了別的層或搬走
的模組），所以它們的裁判改由這一份標準答案接手：常數比凍結值、函式比「同一組輸入下的輸出」。

**要跑哪些 case 不是這一支決定的。** 「有哪些 case」住在 ``blueprint/config_cut1_cases.py``
（版控裡的單一來源），這一支照它跑、照它的 id 寫進答案檔。答案檔本身住 ``blueprint/``，
是 ``identity-strings-generated`` 與 ``refs-and-links-resolve`` 都扣掉的地方——所以**不能**
靠那兩支守「答案檔還有幾筆」；守它的是考卷那一邊的集合比對（見
``tests/engine/_config_answers.py``）與 case 表。

**數字以後用跑的，不要用抄的**（票 #138 的原則）。這支程式在唯讀的 v2 工作樹上把值真的
跑出來；考卷讀那個檔比對，人不碰數字。

**怎麼跑**（v2 的 venv 自己一套依賴，不要拿 v3 的環境；``cd`` 在 repo 根，``-m`` 讓
``blueprint`` 這個套件 import 得到）：

    PYTHONPATH=<v2 工作樹> <v2 工作樹>/.venv/bin/python -m blueprint.generate_config_cut1_answers \
        --out blueprint/config_cut1_answers.json

``PYTHONPATH`` 是**在呼叫時**帶進去的環境變數，這支程式碼裡一個字都不碰 ``sys.path``
（規矩卡 ``uv-single-entrypoint`` 第二條把 ``sys.path`` 的 append／insert／extend 與指派
一律判紅，而 ``blueprint`` 底下每一支 .py 也在它的掃描面裡）。

**為什麼要在 ``sys.modules`` 放 stub。** 載入 ``lib.config`` 底下任何一支都會先執行那個
套件的門面檔，而那 5 行 re-export 會把 JAX 拉進來——那不是新家的形狀，也會讓
「這一支模組自己有沒有碰 JAX」這個問題問不出答案。先在 ``sys.modules`` 放 ``lib`` 與
``lib.config`` 兩個空殼（只設 ``__path__``），再逐一載入，就等於跳過門面。那個空殼的目錄
由呼叫端的 ``PYTHONPATH`` 決定（``lib`` 套件登記在哪，就從哪裡拿）。

**donor 的出身用量的，不是宣稱**（PR #150 的找碴 F3）：這一支會
①解析 ``v3-donor`` 這個記號（不是讀一個寫死的字串、也不是拿 ``HEAD`` 冒充），
②確認那棵樹乾淨（``status --porcelain`` 空），③把 tag、解析出來的完整 commit sha、
以及 ``clean: true`` 一起寫進檔頭。**v2 不在 CI**，所以那個 sha 沒有機器在守——
它是「產生時量的 ＋ 人工可重跑」，不是 CI 驗的。

**答案檔裡不准有絕對路徑或 ``~/``。** 它住在 ``blueprint`` 底下——那一層是
``refs-and-links-resolve`` 扣掉的前綴，所以這一條規矩在這裡靠人守：產生器只寫 donor 的出身、
每個 case 的 id 與值本身。
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import types
import warnings
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final, cast

from blueprint import config_cut1_cases as cases

DONOR_TAG: Final[str] = "v3-donor"
# 答案檔的形狀版本。加了 case 表（id）與 donor 的 clean 那一格之後是第 2 版。
ANSWER_SCHEMA: Final[int] = 2


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


def _git(v2_root: Path, *args: str) -> str:
    """在 v2 工作樹上跑一個唯讀的 git 指令，回傳去掉尾端空白的輸出。"""
    done = subprocess.run(
        ["git", "-C", str(v2_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def donor_provenance(v2_root: Path) -> dict[str, object]:
    """量 donor 的出身：記號解析出來的 commit，以及那棵樹乾不乾淨。

    三個都不是宣稱：``rev-parse <tag>^{commit}`` 是記號真的指到哪一筆提交；
    ``status --porcelain`` 空才准產出（一棵被改過的樹跑出來的值不是上一代的值）。
    """
    commit = _git(v2_root, "rev-parse", f"{DONOR_TAG}^{{commit}}")
    dirty = _git(v2_root, "status", "--porcelain")
    if dirty:
        raise SystemExit(
            f"{v2_root} 不乾淨（status --porcelain 有 {len(dirty.splitlines())} 行）"
            f"——在一棵被改過的樹上跑出來的值不是 {DONOR_TAG} 的值，拒絕產出"
        )
    if _git(v2_root, "rev-parse", "HEAD") != commit:
        raise SystemExit(
            f"{v2_root} 的 HEAD 不是 {DONOR_TAG}（{DONOR_TAG}^{{commit}}={commit}）"
            "——這一支只准在記號那一筆上跑"
        )
    return {"tag": DONOR_TAG, "commit": commit, "clean": True}


def load_module(name: str) -> types.ModuleType:
    """載入 v2 的 ``lib.config.<name>``（門面已被跳過）。"""
    return importlib.import_module(f"lib.config.{name}")


def resolve(value: object) -> object:
    """把 case 表裡的一個參數值還原成 Python 值（跟考卷用同一支）。"""
    return cases.resolve(value)


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
            out["expr"] = "4*math.pi"
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


def constants_block(module: types.ModuleType, names: list[str]) -> dict[str, object]:
    """這一支模組宣告的每一個公開常數各一筆凍結值（清單來自 case 表）。"""
    return {name: encode(getattr(module, name)) for name in names}


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


def _read_attr(built: object, name: str) -> dict[str, object]:
    """讀一個屬性；讀的當下炸掉（例如守門）就記成例外。"""
    try:
        value = getattr(built, name)
    except Exception as exc:  # noqa: BLE001  # expires=2026-12-08 reason=屬性本身就是守門（值不合法時丟例外），這裡要記錄它而不是讓它往外炸
        return {"raised": {"type": type(exc).__name__, "message": str(exc)}}
    return {"value": encode(value)}


def _spl_output_block(cls: object, args: dict[str, object]) -> dict[str, object]:
    """建一個 ``SplOutputConfig``，連它的兩個屬性一起收（屬性要從真的物件上讀）。

    屬性要在**還沒編碼之前**、從真的物件上讀（編碼之後它只是一個 JSON 記號，沒有屬性）；
    建不起來的那幾筆已經由 ``call_block`` 記下例外了，這裡不再多收屬性。
    """
    block = call_block(cls, args)  # type: ignore[arg-type]  # expires=2026-12-08 reason=cls 是從模組上拿下來的類別物件，型別樁只知道它是 object；這裡要的就是「執行期那一格」
    if "value" not in block:
        return block
    try:
        built = cls(**args)  # type: ignore[operator]  # expires=2026-12-08 reason=執行期才知道的類別物件，型別樁表達不了
    except Exception:  # noqa: BLE001  # expires=2026-12-08 reason=同上一行：建不起來的那些 case 不是這一格要報的事
        return block
    for name in ("l_ref_db", "is_relative"):
        block[name] = _read_attr(built, name)
    return block


# 一個 case 要在上一代哪一支函式上跑。名字就是這一刀那 11 支裡的公開函式；
# `SplOutputConfig` 那幾筆走 `_spl_output_block`（它要順手讀兩個屬性）。
CALL_NAMES: Final[dict[str, str]] = {
    "art_lane": "guard_art_patch_count",
    "authoring_defaults": "app_wall_id",
    "crossover_axis": "build_canonical_crossover",
    "speaker_directivity": "resolve_speaker_directivity",
    "spl_output": "SplOutputConfig",
}


def _pick_function(module: types.ModuleType, case_id: str) -> Callable[..., object]:
    """這一筆 case 要用哪一支函式跑（case id 的第三段就是那一支的名字）。"""
    if ".guard_polygon_art_patch_count." in case_id:
        return cast(Callable[..., object], module.guard_polygon_art_patch_count)
    if ".app_patch_id." in case_id:
        return cast(Callable[..., object], module.app_patch_id)
    if ".rhino_boundary_id." in case_id:
        return cast(Callable[..., object], module.rhino_boundary_id)
    if ".app_wall_id." in case_id:
        return cast(Callable[..., object], module.app_wall_id)
    name = CALL_NAMES[module.__name__.rsplit(".", 1)[-1]]
    return cast(Callable[..., object], getattr(module, name))


def presets_block(module: types.ModuleType) -> dict[str, object]:
    """``SPEAKER_PRESETS`` 那一筆：兩個預設的欄位表，**走 ``encode()``**（浮點存 hex）。

    這一格跟 ``constants.SPEAKER_PRESETS`` 是同一組數字——刻意留著是因為考卷那一邊要
    拿它去對「解析函式回傳的物件」與「資料表本身」一不一致（值一樣、容器不同）。
    """
    return {
        "preset_fields": encode(
            {name: preset for name, preset in module.SPEAKER_PRESETS.items()}
        )
    }


def probe_record(module_name: str, case: cases.CaseEntry) -> dict[str, object]:
    """一筆探針：case 的 id、那一筆的參數、以及那組參數底下的結果。"""
    module = load_module(module_name)
    case_id = case["id"]
    args = cases.resolve_args(case)
    if ".SPEAKER_PRESETS." in case_id:
        expected = presets_block(module)
    elif module_name == "spl_output":
        expected = _spl_output_block(module.SplOutputConfig, args)
    else:
        expected = call_block(_pick_function(module, case_id), args)
    return {"id": case_id, "args": args, "expected": expected}


def _case_id(record: dict[str, object]) -> str:
    """答案檔那一筆的 case id（排序用；這也讓輸出逐位元可重跑）。"""
    case_id = record.get("id")
    if not isinstance(case_id, str):
        raise SystemExit(f"這一筆探針沒有 id：{record!r}")
    return case_id


def build_payload(v2_root: Path) -> dict[str, object]:
    """把 case 表宣告的每一筆跑出來，組成答案檔的內容。"""
    per_module: dict[str, object] = {}
    for name in sorted(cases.MODULES):
        module = load_module(name)
        records: list[dict[str, object]] = [probe_record(name, case) for case in cases.cases_for(name)]
        records.sort(key=_case_id)
        entry: dict[str, object] = {
            "constants": constants_block(module, cases.constants_for(name)),
            "probes": records,
        }
        per_module[name] = entry
    return {
        "schema": ANSWER_SCHEMA,
        "donor": donor_provenance(v2_root),
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
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(text)
    modules = payload["modules"]
    print(f"寫出 {args.out}（{len(text)} bytes）")
    for name in sorted(cases.MODULES):
        entry = modules[name]  # type: ignore[index]  # expires=2026-12-08 reason=modules 是這支剛剛組出來的 dict，形狀由 build_payload 保證
        print(f"  {name}: 常數 {len(entry['constants'])} 筆，探針 {len(entry['probes'])} 筆")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
