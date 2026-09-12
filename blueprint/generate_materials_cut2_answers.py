"""把上一代（donor）繼承那三支的公開行為跑出來，寫成新家考卷要用的標準答案檔。

**這是票 #134 的第二刀**：``freq_axis``／``experiment_schema``／``material_loader``。第一刀
那三支（``response``／``source``／``registry``）的答案在 ``materials_cut1_answers.json``，
**這一支不碰它**。

「有哪些 case」住 :mod:`blueprint.materials_cut2_cases`（版控裡的單一來源），「怎麼跑」住
:mod:`blueprint.materials_cut2_probe`（產生器與考卷共用同一支）。這一支只做四件事：確認
donor 的出身、把模組交給探針、另起一個開了 x64 的行程問那一格、把結果寫成檔。

**怎麼跑**（donor 自己一套依賴，不要拿 v3 的環境；``cd`` 在 v3 repo 根，``-m`` 讓
``blueprint`` 這個套件 import 得到）::

    PYTHONPATH=<donor 工作樹>:. JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \\
        PYTHONDONTWRITEBYTECODE=1 <外部 donor venv>/bin/python \\
        -m blueprint.generate_materials_cut2_answers \\
        --out blueprint/materials_cut2_answers.json

``PYTHONPATH`` 是**在呼叫時**帶進去的環境變數，這支程式碼裡一個字都不碰 ``sys.path``
（規矩卡 ``uv-single-entrypoint`` 第二條把那一族判紅）。

**為什麼要另起一個行程。** ``FREQS_HZ_IDENTITY`` 被釘成 float32，所以不管載入的時候
``jax_enable_x64`` 開還是關都該是同一組值——而常數在 import 的時候就算完了，同一個行程裡
問不出這件事。這一支把開關打開、另起一個行程再載入一次，把兩邊的值都寫進答案檔；考卷那邊
在新家做同一件事。

**這一支刻意不印任何東西**（``style-guard`` 只准具名的輸出層 ``print``）：要看它跑出什麼就
讀產物本身。跑壞了走離開碼。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import ModuleType
from typing import Final

from blueprint import materials_cut2_cases as cases
from blueprint import materials_cut2_probe as probe

DONOR_TAG: Final[str] = "v3-donor"

# 檔頭要記的依賴版本（兩邊的 dtype、數值與錯誤訊息都跟著它們走，所以要記下來比對；
# pydantic 與 pyyaml 是這一刀才進來的——那兩支的錯誤訊息逐字進答案檔）。
RECORDED_PACKAGES: Final[tuple[str, ...]] = ("jax", "jaxlib", "flax", "numpy", "pydantic", "PyYAML")

# 另起那個行程要跑的東西：把 lib 那三個套件放空殼、載入 donor 的 freq_axis、
# 用**同一支 codec** 把那條 tuple 編成記號印出來。
X64_SNIPPET: Final[str] = """
import importlib, json, sys, types
from pathlib import Path

donor_root = Path(sys.argv[1])
module_name = sys.argv[2]
for name, parts in (
    ("lib", ("lib",)),
    ("lib.materials", ("lib", "materials")),
    ("lib.config", ("lib", "config")),
):
    package = types.ModuleType(name)
    package.__path__ = [str(donor_root.joinpath(*parts))]
    sys.modules[name] = package

from blueprint import materials_cut1_probe as codec

import jax

module = importlib.import_module(module_name)
sys.stdout.write(json.dumps({
    "x64": bool(jax.config.jax_enable_x64),
    "identity": codec.encode(module.FREQS_HZ_IDENTITY),
}))
"""


def stub_donor_packages(donor_root: Path) -> None:
    """在 ``sys.modules`` 放 ``lib``／``lib.materials``／``lib.config`` 的空殼。

    只設 ``__path__``，不執行那幾個套件的門面檔（新家不打算有那個門面，跳過它才問得出
    「這一支模組自己需要什麼」）。
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
    """從呼叫端帶進來的 ``PYTHONPATH`` 找出 donor 工作樹的根（刻意不碰 ``sys.path``）。"""
    for entry in [item for item in sys.path if item]:
        candidate = Path(entry)
        if (candidate / "lib" / "config" / "material_loader.py").is_file():
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
    """這一跑的環境：依賴版本、JAX 的後端與 x64 開關（三格都是答案的一部分）。"""
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
    return importlib.import_module(cases.lookup_name(name, "donor"))


def donor_module_paths() -> dict[str, str]:
    """表上每一個名字在 donor 這一邊真的 import 路徑（訊息裡那一段要換成替身）。"""
    names = [*cases.MODULES, *cases.INGREDIENTS]
    return {name: cases.lookup_name(name, "donor") for name in names}


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
    paths = donor_module_paths()
    records = [
        {"id": case["id"], "result": probe.run_case(donor_lookup, case, paths)}
        for case in cases.cases_for(name)
    ]
    records.sort(key=lambda record: str(record["id"]))
    return {"constants": constants, "aliases": aliases, "cases": records}


def x64_identity_probe(donor_root: Path) -> dict[str, object]:
    """另起一個**開了 x64** 的行程，問同一條 tuple 的值。

    回兩格：``on`` 是那個行程量到的、``off`` 是這一跑（x64 關著）量到的。兩格相等就是
    「那條 tuple 跟 x64 開關無關」的證據；考卷在新家做同一件事，再跟這裡逐格比。
    """
    module = donor_lookup("freq_axis")
    with tempfile.TemporaryDirectory(prefix="aosr-cut2-x64-") as workdir:
        script = Path(workdir) / "probe_x64_identity.py"
        script.write_text(X64_SNIPPET, encoding="utf-8")
        env = dict(os.environ)
        env["JAX_ENABLE_X64"] = "true"
        env["JAX_PLATFORMS"] = "cpu"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        done = subprocess.run(
            [sys.executable, str(script), str(donor_root), cases.lookup_name("freq_axis", "donor")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
    on: object = json.loads(done.stdout)
    return {"on": on, "off": {"x64": False, "identity": probe.encode(module.FREQS_HZ_IDENTITY)}}


def build_payload(donor_root: Path) -> dict[str, object]:
    """把 case 表宣告的每一筆跑出來，組成答案檔的內容。"""
    return {
        "schema": probe.ANSWER_SCHEMA,
        "donor": donor_provenance(donor_root),
        "env": environment(),
        "note": (
            "由 blueprint/generate_materials_cut2_answers.py 在唯讀的 donor 工作樹上跑出來，"
            "不准手改；要改就重跑產生器。"
        ),
        "modules": {name: module_block(name) for name in sorted(cases.MODULES)},
        "probes": {"x64_identity": x64_identity_probe(donor_root)},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    """``--out``：答案檔要寫到哪裡。"""
    parser = argparse.ArgumentParser(description="把 donor 繼承那三支的公開行為跑成標準答案檔")
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
