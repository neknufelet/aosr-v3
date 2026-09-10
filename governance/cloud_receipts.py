"""三張收據卡共用的零件：讀鏡像、讀卡上的登記簿、一份收據的形狀收窄、按 schema 版本挑規則。

鏡像（`governance/status/mirror_receipts.py` 抄出來的那一層）長這樣：

    <mirror_dir>/receipts/<run id>-<attempt>.json   一跑一份
    <mirror_dir>/provenance.json                     每一份是哪筆提交寫的、宣稱的身分

**鏡像不在列舉集合裡。** 那個目錄在 .gitignore 裡（真樹上），卡的 scope 也不宣告它——跟 green-must-be-real-green
讀 junit 收據一樣是照實記：檢查會讀它，但它不是版控裡的檔。樣本樹裡則是真的檔（.gitignore 那一行只蓋根層）。

**按版本判。** 每一份收據自帶 `schema`；卡上的登記簿按版本列欄位。收據的版本卡不認識（比卡新）就回 2
——不是紅：新收據我沒有尺，「沒問題」這句話不算數；比卡舊的按它自己那版的欄位判——規矩卡
receipt-authority-is-the-cloud-run 第④條：新規矩不准回頭把舊收據判成無效。

**放行。** 卡上 `[[settings.allow]]` 可以具名放過某一份收據（`path = "<run id>-<attempt>.json"`，帶 reason 與
expires，由 exemptions-need-expiry 守）。收據落在 status 分支就是永久的，一份出過事的收據會讓主線永遠紅，
放行是唯一的出口，而且要寫得出為什麼、什麼時候要回頭看。這裡沒有任何數字：門檻與名單全住在卡上。
"""
from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TypeGuard

from governance.exit_codes import ToolBroken, note
from governance.loader import RULES_DIR, exemption_field_problems, setting_int, setting_strings, setting_tables, setting_text

ALLOW_KEY = "allow"
ALLOW_PATH_KEY = "path"


@dataclass(frozen=True)
class Receipt:
    """鏡像裡的一份收據：檔名、解開的 JSON、版本。"""

    name: str
    body: dict[str, object]
    schema: int


@dataclass(frozen=True)
class Mirror:
    """整份鏡像：收據們，加上來源。"""

    receipts: tuple[Receipt, ...]
    provenance: dict[str, object]
    folder: Path


# ── 讀卡 ──────────────────────────────────────────────────────────────────────


def card_settings(scan_root: Path, files: list[Path], check_rel: str) -> tuple[Path, dict[str, object]]:
    """掃描根自己的 governance/rules/ 底下宣告 check 是那一支的卡。要剛好一張、要有 [settings]。"""
    rules_dir = scan_root / RULES_DIR
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == check_rel:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={check_rel} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——登記簿只寫在卡上，讀不到卡這一跑就不算數"
        )
    path, data = mine[0]
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{path.relative_to(scan_root)} 沒有 [settings] 表（登記簿全寫在那裡）")
    return path, {str(k): v for k, v in settings.items()}


def by_schema(settings: Mapping[str, object], key: str, schema: int, where: str) -> list[str]:
    """登記簿裡按版本列的一組欄位：``[settings.<key>]`` 底下 ``"1" = [...]``、``"2" = [...]``。

    往下找最近的一版：舊收據按它自己那版判，登記簿只在欄位有變的那一版才寫一次。
    「收據比卡新」那一關在 :func:`read_mirror` 用卡上的 ``newest_schema`` 一次擋掉，這裡不重判。
    """
    table = settings.get(key)
    if not isinstance(table, dict) or not table:
        raise ToolBroken(f"{where} 的 [settings.{key}] 缺席或是空的——登記簿沒有這一組欄位就不出結論")
    versions: dict[int, list[str]] = {}
    for raw, value in table.items():
        try:
            version = int(str(raw))
        except ValueError as exc:
            raise ToolBroken(f"{where} 的 [settings.{key}] 有一個不是版本號的鍵 {raw!r}") from exc
        versions[version] = setting_strings({"x": value}, "x")
    older = [v for v in versions if v <= schema]
    if not older:
        raise ToolBroken(f"{where} 的 [settings.{key}] 沒有任何一版蓋得到 schema {schema}（登記的是 {sorted(versions)}）")
    return versions[max(older)]


def allowed_names(settings: Mapping[str, object], today: date) -> set[str]:
    """卡上具名放過的收據檔名——只算還沒到期的那幾筆。

    放行有牙：過期的那一筆當作沒放行，那份收據照樣紅（跟 merge-gate-read-back 的准上網放行同一個做法）。
    兩格（reason、expires）的形狀不對就 ToolBroken——寫壞的放行不是放行。理由與到期日另外由
    exemptions-need-expiry 守著整張卡；這裡只讀名字與日期。
    """
    picked: set[str] = set()
    for entry in setting_tables(settings, ALLOW_KEY):
        if ALLOW_PATH_KEY not in entry:
            continue
        name = setting_text(entry, ALLOW_PATH_KEY)
        problems = exemption_field_problems(entry, f"放過 {name} 的那一筆")
        if problems:
            raise ToolBroken("；".join(problems))
        expires = date.fromisoformat(str(entry["expires"]).strip())
        if expires < today:
            note(f"放行 {name} 已於 {expires} 到期，當作沒放行——要續就重新看它還需不需要，改卡走 PR")
            continue
        picked.add(name)
    return picked


# ── 讀鏡像 ────────────────────────────────────────────────────────────────────


def _load_json(path: Path, what: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不開{what} {path.name}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ToolBroken(f"{what} {path.name} 不是 JSON：{exc}——解不開的收據沒得判，這一跑不算數") from exc


def read_mirror(scan_root: Path, settings: Mapping[str, object], where: str) -> Mirror:
    """讀整份鏡像。目錄不在、一份都沒有、來源檔不在，一律回 2：沒有收據就沒有東西可判。"""
    folder = scan_root / setting_text(settings, "mirror_dir")
    receipts_dir = folder / setting_text(settings, "receipts_subdir")
    provenance_path = folder / setting_text(settings, "provenance_file")
    newest = setting_int(settings, "newest_schema")
    if not receipts_dir.is_dir():
        raise ToolBroken(
            f"鏡像目錄 {receipts_dir.relative_to(scan_root)} 不在——先跑 governance.status.mirror_receipts 把 status 分支上的收據鏡過來"
        )
    paths = sorted(receipts_dir.glob("*.json"))
    if not paths:
        raise ToolBroken(f"鏡像目錄 {receipts_dir.relative_to(scan_root)} 底下一份收據都沒有——沒有收據就沒有東西可判")
    if not provenance_path.is_file():
        raise ToolBroken(f"鏡像缺來源檔 {provenance_path.relative_to(scan_root)}——不知道收據是誰寫的，這一跑不算數")
    provenance = _load_json(provenance_path, "來源檔")
    if not isinstance(provenance, dict):
        raise ToolBroken(f"來源檔 {provenance_path.name} 不是一張表")
    receipts: list[Receipt] = []
    for path in paths:
        body = _load_json(path, "收據")
        if not isinstance(body, dict):
            raise ToolBroken(f"收據 {path.name} 不是一張表：{str(body)[:80]!r}")
        schema = body.get("schema")
        if isinstance(schema, bool) or not isinstance(schema, int):
            raise ToolBroken(f"收據 {path.name} 沒有整數的 schema 欄（實際是 {schema!r}）——不知道版本就沒有尺")
        if schema > newest:
            raise ToolBroken(
                f"收據 {path.name} 的 schema {schema} 比 {where} 登記的最新版 {newest} 還新——這張卡沒有這一版的尺，"
                "「沒問題」這句話不算數；先改卡登記那一版"
            )
        receipts.append(Receipt(name=path.name, body={str(k): v for k, v in body.items()}, schema=schema))
    note(f"鏡像 {folder.relative_to(scan_root)}：{len(receipts)} 份收據，來源 {str(provenance.get('ref', '?'))}@{str(provenance.get('commit', '?'))[:12]}（{where}）")
    return Mirror(receipts=tuple(receipts), provenance={str(k): v for k, v in provenance.items()}, folder=folder)


def table(value: object, what: str) -> dict[str, object]:
    """收窄成一張表；不是就 ToolBroken（形狀看不懂就不出結論）。"""
    if not isinstance(value, dict):
        raise ToolBroken(f"{what} 不是一張表：{str(value)[:80]!r}")
    return {str(k): v for k, v in value.items()}


def rows(value: object, what: str) -> list[object]:
    """收窄成清單；不是就 ToolBroken。"""
    if not isinstance(value, list):
        raise ToolBroken(f"{what} 不是清單：{str(value)[:80]!r}")
    return value


def is_int(value: object) -> TypeGuard[int]:
    """整數（``true``／``false`` 不算）。宣告成 TypeGuard，呼叫端之後就當它是 int。"""
    return isinstance(value, int) and not isinstance(value, bool)
