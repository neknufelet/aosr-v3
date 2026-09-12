"""把上一代（donor）材料那三支的公開行為跑出來，寫成新家考卷要用的標準答案檔。

**為什麼要有這一支。** 票 #134 第一候選的六支裡，``response``／``source``／``registry``
在上一代**沒有一支帶得走的考卷**——唯一直接驗它們的 ``tests/test_response.py`` 只有 4 題，
其餘 160 多支測試檔雖然 import 得到它們，但每一支都綁著 physics／scoring／geometry 或
週邊腳本那一包（見 ``scope.md``）。所以它們的裁判改由這一份標準答案接手：常數與型別別名
比凍結值，公開函式與方法比「同一組輸入下的輸出（或它炸出來的例外）」。

**要跑哪些 case 不是這一支決定的。** 「有哪些 case」住在 ``blueprint/materials_cut1_cases.py``
（版控裡的單一來源），**怎麼跑**住在 ``blueprint/materials_cut1_probe.py``（產生器與考卷
共用同一支）。這一支只做三件事：確認 donor 的出身、把模組交給探針、把結果寫成檔。

**怎麼跑**（donor 自己一套依賴，不要拿 v3 的環境；``cd`` 在 v3 repo 根，``-m`` 讓
``blueprint`` 這個套件 import 得到）::

    PYTHONPATH=<donor 工作樹> JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \\
        <外部 donor venv>/bin/python -m blueprint.generate_materials_cut1_answers \\
        --out blueprint/materials_cut1_answers.json

``PYTHONPATH`` 是**在呼叫時**帶進去的環境變數，這支程式碼裡一個字都不碰 ``sys.path``
（規矩卡 ``uv-single-entrypoint`` 第二條把那一族判紅）。

**為什麼要在 ``sys.modules`` 放 stub。** 載入 ``lib.materials`` 底下任何一支都會先執行那個
套件的門面檔（一段 re-export）。新家不打算有那個門面，跳過它才問得出「這一支模組自己需要
什麼」。放兩個空殼（只設 ``__path__``）就等於跳過門面。

**donor 的出身用量的，不是宣稱**：解析 ``v3-donor`` 這個記號、確認那棵樹乾淨、確認 HEAD
就在記號上，三格一起寫進檔頭。donor 不在 CI 上，所以那個 sha 沒有機器在守——它是
「產生時量的 ＋ 人工可重跑」。

**這一支刻意不印任何東西。** ``style-guard`` 只准具名的輸出層 ``print``，而這一支還沒有
登記；要看它跑出什麼就讀產物本身（答案檔的檔頭記了環境與 donor 出身）。跑壞了走離開碼。
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Final

from blueprint import materials_cut1_cases as cases
from blueprint import materials_cut1_probe as probe

DONOR_TAG: Final[str] = "v3-donor"

# 檔頭要記的依賴版本（兩邊的 dtype 與數值都跟著它們走，所以要記下來比對）。
RECORDED_PACKAGES: Final[tuple[str, ...]] = ("jax", "jaxlib", "flax", "numpy")


def stub_donor_packages(donor_root: Path) -> None:
    """在 ``sys.modules`` 放 ``lib``／``lib.materials``／``lib.config`` 的空殼。

    只設 ``__path__``（告訴 import 系統「這是一個套件、它的檔在這個目錄底下」），
    不執行那幾個套件的門面檔。
    """
    for name, parts in (
        ("lib", ("lib",)),
        ("lib.materials", ("lib", "materials")),
        ("lib.config", ("lib", "config")),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(donor_root.joinpath(*parts))]
        sys.modules[name] = package


def donor_root_from_env() -> Path:
    """從呼叫端帶進來的 ``PYTHONPATH`` 找出 donor 工作樹的根。

    刻意不碰 ``sys.path``。找不到就當場炸——找錯一棵樹比找不到更糟（會產出一份看起來
    很合理的答案檔）。
    """
    entries = [item for item in sys.path if item]
    for entry in entries:
        candidate = Path(entry)
        if (candidate / "lib" / "materials" / "response.py").is_file():
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
    """量 donor 的出身：記號解析出來的 commit，以及那棵樹乾不乾淨。"""
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
    """這一跑的環境：依賴版本、JAX 的後端與 x64 開關。

    這三格是**答案的一部分**：``jax_enable_x64`` 一開，同一段程式跑出來的 dtype 從
    ``float32`` 變 ``float64``，值也跟著變。考卷那一邊會比同樣三格。
    """
    metadata = importlib.import_module("importlib.metadata")
    jax = importlib.import_module("jax")
    versions = {name: metadata.version(name) for name in RECORDED_PACKAGES}
    return {
        "versions": versions,
        "backend": str(jax.default_backend()),
        "x64": bool(jax.config.jax_enable_x64),
    }


def donor_lookup(name: str) -> ModuleType:
    """表上的模組名換成 donor 那一支真的模組。"""
    return importlib.import_module(cases.MODULES[name]["donor"])


def module_block(name: str) -> dict[str, object]:
    """一支模組那一塊：常數、型別別名、每一筆 case 的結果。"""
    module = donor_lookup(name)
    constants = {
        item: probe.encode(getattr(module, item)) for item in cases.constants_for(name)
    }
    aliases = {
        item: probe.encode(list(probe.alias_values(module, item)))
        for item in cases.aliases_for(name)
    }
    records = [
        {"id": case["id"], "result": probe.run_case(donor_lookup, case)}
        for case in cases.cases_for(name)
    ]
    records.sort(key=lambda record: str(record["id"]))
    return {"constants": constants, "aliases": aliases, "cases": records}


def build_payload(donor_root: Path) -> dict[str, object]:
    """把 case 表宣告的每一筆跑出來，組成答案檔的內容。"""
    return {
        "schema": probe.ANSWER_SCHEMA,
        "donor": donor_provenance(donor_root),
        "env": environment(),
        "note": (
            "由 blueprint/generate_materials_cut1_answers.py 在唯讀的 donor 工作樹上跑出來，"
            "不准手改；要改就重跑產生器。"
        ),
        "modules": {name: module_block(name) for name in sorted(cases.MODULES)},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    """``--out``：答案檔要寫到哪裡。"""
    parser = argparse.ArgumentParser(description="把 donor 材料那三支的公開行為跑成標準答案檔")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """跑一次：確認 donor 出身、蒐集結果、寫檔。"""
    args = parse_args(argv)
    donor_root = donor_root_from_env()
    stub_donor_packages(donor_root)
    payload = build_payload(donor_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    out: Path = args.out
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
