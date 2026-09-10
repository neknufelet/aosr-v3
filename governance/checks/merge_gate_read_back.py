"""規矩卡 merge-gate-read-back：合併門口的設定要從伺服器回讀比對。

卡上 ``[settings]`` 登記的是**期望**（哪個 repo、哪個 ruleset、必須有哪幾條規則、必要檢查叫什麼、
bypass 名單必須是空的）；這支檢查用 ``gh``（GitHub 的命令列工具）把主線的 ruleset **從伺服器讀回來**，
逐格比對，任何一格對不上就回 1；登記的 ruleset 在伺服器上根本不存在也回 1（主線沒鎖）。

**這是唯一准上網的檢查。** 決策紙 ``docs/decisions/merge-gate-check-may-read-github.md``：只准這一支、
只准讀、放行登記在卡的 ``[[settings.allow]]`` 並帶到期日。放行有牙——這裡先讀那一筆放行，缺席或
過期就回 2、不上網。到期那天不只 exemptions-need-expiry 紅，這支也拒絕上網。

**回 2 的路（這一跑不算數，不准當綠）：** ``gh`` 不在 PATH 上、沒 token、API 回不了話或逾時、回來的
不是 JSON、形狀跟 GitHub 文件對不上、卡讀不到或放行缺席／過期。

**刻意不比的：** ``current_user_can_bypass``（那一格看的是問的人是誰，不是設定本身）；reflog 與
非快進推送的痕跡（雲端 checkout 是冷 clone，看不到）。門檻與期望只住在卡上，這裡沒有任何數字。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from governance import gh_replay
from governance.exit_codes import ToolBroken, note, run
from governance.loader import (
    EXEMPTION_KEYS,
    RULES_DIR,
    exemption_field_problems,
    setting_int,
    setting_strings,
    setting_tables,
    setting_text,
)

CHECK_REL = "governance/checks/merge_gate_read_back.py"
GH = "gh"
ALLOW_KEY = "allow"
# 放行條目裡「准上網」那一格的鍵。找的是帶這一格的那一筆，不是任何一筆 allow。
NETWORK_KEY = "network"



@dataclass(frozen=True)
class Expected:
    """卡上登記的期望。"""

    repo: str
    ruleset_id: int
    enforcement: str
    ref_include: tuple[str, ...]
    rule_types: tuple[str, ...]
    required_checks: tuple[str, ...]
    strict: bool
    dismiss_stale: bool
    bypass_empty: bool
    timeout: int


# ── 讀卡 ──────────────────────────────────────────────────────────────────────


def _card_path(scan_root: Path, files: list[Path]) -> Path:
    """掃描根自己的 governance/rules/ 底下宣告 check 是這支的那張卡。要剛好一張。"""
    rules_dir = scan_root / RULES_DIR
    mine: list[Path] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append(path)
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——期望只寫在卡上，讀不到卡這一跑就不算數"
        )
    return mine[0]


def _settings(card: Path, scan_root: Path) -> dict[str, object]:
    data = tomllib.loads(card.read_bytes().decode("utf-8"))
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{card.relative_to(scan_root)} 沒有 [settings] 表（期望全寫在那裡）")
    return settings


def _bool(settings: Mapping[str, object], key: str, where: str) -> bool:
    value = settings.get(key)
    if not isinstance(value, bool):
        raise ToolBroken(f"{where} 的 [settings] {key} 必須是 true／false，實際是 {value!r}")
    return value


def read_expected(settings: Mapping[str, object], where: str) -> Expected:
    """把卡上的期望讀成一個具體型別。少一格、形狀不對，loader 的讀值函式自己就回 2。"""
    return Expected(
        repo=setting_text(settings, "repo"),
        ruleset_id=setting_int(settings, "ruleset_id"),
        enforcement=setting_text(settings, "enforcement"),
        ref_include=tuple(setting_strings(settings, "ref_include")),
        rule_types=tuple(setting_strings(settings, "rule_types")),
        required_checks=tuple(setting_strings(settings, "required_checks")),
        strict=_bool(settings, "strict_required_status_checks", where),
        dismiss_stale=_bool(settings, "dismiss_stale_reviews_on_push", where),
        bypass_empty=_bool(settings, "bypass_actors_empty", where),
        timeout=setting_int(settings, "api_timeout_seconds"),
    )


def network_allowed(settings: Mapping[str, object], where: str, today: date) -> str:
    """找那一筆「准上網」的放行。缺席、欄位不齊、過期，一律回 2——不上網。回那一筆的 reason。"""
    entries = [e for e in setting_tables(settings, ALLOW_KEY) if NETWORK_KEY in e]
    if len(entries) != 1:
        raise ToolBroken(
            f"{where} 的 [[settings.allow]] 裡帶 {NETWORK_KEY} 那一格的放行有 {len(entries)} 筆，要剛好 1 筆"
            "——沒有登記的放行就不准上網（決策紙 merge-gate-check-may-read-github）"
        )
    entry = entries[0]
    problems = exemption_field_problems(entry, f"{where} 的准上網放行")
    if problems:
        raise ToolBroken("；".join(problems))
    expires = date.fromisoformat(str(entry["expires"]).strip())
    if expires < today:
        raise ToolBroken(
            f"{where} 的准上網放行 {expires} 已到期（今天 {today}）——到期就不准上網，"
            "要續就重新看它還需不需要、範圍有沒有變寬，改卡走 PR"
        )
    return str(entry[EXEMPTION_KEYS[0]])


# ── 問伺服器 ───────────────────────────────────────────────────────────────────


def gh_json(path: str, timeout: int) -> object:
    """打一支只讀的 GitHub API。叫不動、逾時、非零、不是 JSON，一律回 2。

    後設測試會把同一句問題問很多次（六回合裡有三回合走到這裡，再乘上「產收據那一跑」），
    所以先問一次 :mod:`governance.gh_replay`——那一層只在後設測試設了錄音目錄的時候有東西，
    而且重播之前一樣要求 ``gh`` 真的在 PATH 上，抽掉工具那一回合照樣回 2。
    雲端那一跑身上沒有那個環境變數，每一句都真的去問伺服器。
    """
    argv = [GH, "api", path]
    what = f"{GH} api {path}"
    recorded = gh_replay.replay(argv, what)
    if recorded is not None:
        return _parse(recorded, path)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ToolBroken(f"{GH} 不在 PATH 上（{exc}）——讀不到伺服器就不出結論") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolBroken(f"{GH} api {path} 超過 {timeout} 秒沒回話") from exc
    if proc.returncode != 0:
        raise ToolBroken(f"{GH} api {path} 回 {proc.returncode}：{proc.stderr.strip()[:300]}")
    gh_replay.record(argv, proc.stdout)
    return _parse(proc.stdout, path)


def _parse(text: str, path: str) -> object:
    """把伺服器（或這一跑的錄音）吐出來的字串解成 JSON。解不開就回 2。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolBroken(f"{GH} api {path} 回來的不是 JSON：{exc}") from exc


def _table(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ToolBroken(f"伺服器回來的 {what} 不是一張表：{value!r}"[:300])
    return {str(k): v for k, v in value.items()}


def _rows(value: object, what: str) -> list[object]:
    if not isinstance(value, list):
        raise ToolBroken(f"伺服器回來的 {what} 不是清單：{value!r}"[:300])
    return value


def _text(row: Mapping[str, object], key: str, what: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ToolBroken(f"伺服器回來的 {what} 沒有字串欄 {key}：{value!r}"[:300])
    return value


def _ids(rows: Sequence[object]) -> set[int]:
    out: set[int] = set()
    for item in rows:
        value = _table(item, "ruleset 清單的一筆").get("id")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolBroken(f"ruleset 清單的一筆沒有整數 id：{value!r}")
        out.add(value)
    return out


# ── 比對 ──────────────────────────────────────────────────────────────────────


def _rules_by_type(detail: Mapping[str, object]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for item in _rows(detail.get("rules"), "ruleset 的 rules"):
        rule = _table(item, "一條 rule")
        out[_text(rule, "type", "一條 rule")] = _table(rule.get("parameters", {}), "rule 的 parameters")
    return out


def _compare_shape(detail: Mapping[str, object], want: Expected) -> list[str]:
    """生效狀態、掛哪些分支、bypass 名單。"""
    hits: list[str] = []
    got = _text(detail, "enforcement", "ruleset")
    if got != want.enforcement:
        hits.append(f"enforcement 期望 {want.enforcement!r}，伺服器上是 {got!r}")
    conditions = _table(detail.get("conditions"), "conditions")
    include = _rows(_table(conditions.get("ref_name"), "ref_name").get("include"), "ref_name.include")
    for ref in want.ref_include:
        if ref not in include:
            hits.append(f"ruleset 沒掛在 {ref!r} 上（include={include!r}）")
    if want.bypass_empty:
        actors = detail.get("bypass_actors")
        if actors is None:
            # GitHub 只把 bypass 名單交給 admin；雲端 CI 的 token 拿到的 JSON 沒這一格
            # （2026-09-10 實測：本機用老闆的帳號看得到、雲端看不到）。看不到就明說，不當成空的。
            note("伺服器沒交出 bypass_actors（只給 admin 看）——「沒有人能繞過」這一格這一跑沒守，只有 admin 的 token 跑得到")
        elif _rows(actors, "bypass_actors"):
            hits.append(f"bypass 名單不是空的：{actors!r}"[:300])
    return hits


def _compare_rules(detail: Mapping[str, object], want: Expected) -> list[str]:
    """四條規則在不在、關鍵參數對不對。"""
    hits: list[str] = []
    rules = _rules_by_type(detail)
    for kind in want.rule_types:
        if kind not in rules:
            hits.append(f"少了規則 {kind!r}（伺服器上有：{sorted(rules)}）")
    pr = rules.get("pull_request")
    if pr is not None and want.dismiss_stale and pr.get("dismiss_stale_reviews_on_push") is not True:
        hits.append("pull_request 規則沒開 dismiss_stale_reviews_on_push——候選一改舊核准要作廢")
    rsc = rules.get("required_status_checks")
    if rsc is not None:
        if want.strict and rsc.get("strict_required_status_checks_policy") is not True:
            hits.append("required_status_checks 沒開 strict——分支要跟上主線才准合")
        contexts = {
            _text(_table(c, "一個必要檢查"), "context", "一個必要檢查")
            for c in _rows(rsc.get("required_status_checks"), "required_status_checks 清單")
        }
        for name in want.required_checks:
            if name not in contexts:
                hits.append(f"必要檢查少了 {name!r}（伺服器上有：{sorted(contexts)}）")
    return hits


def compare(detail: Mapping[str, object], want: Expected) -> list[str]:
    """伺服器讀回來的 ruleset 跟卡上期望逐格比。回違規清單，空就是一致。"""
    return _compare_shape(detail, want) + _compare_rules(detail, want)


# ── 外殼接線 ──────────────────────────────────────────────────────────────────


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：只有自己那張卡。"""
    return [_card_path(scan_root, files)]


def check(scan_root: Path, files: list[Path]) -> list[str]:
    card = _card_path(scan_root, files)
    where = str(card.relative_to(scan_root))
    settings = _settings(card, scan_root)
    want = read_expected(settings, where)
    reason = network_allowed(settings, where, date.today())
    note(f"准上網的放行在（{reason[:60]}…）；只讀 {want.repo} 的 ruleset {want.ruleset_id}")
    listed = _ids(_rows(gh_json(f"repos/{want.repo}/rulesets", want.timeout), "ruleset 清單"))
    if want.ruleset_id not in listed:
        return [f"登記的 ruleset {want.ruleset_id} 在 {want.repo} 上不存在（伺服器上有：{sorted(listed)}）——主線沒鎖"]
    detail = _table(gh_json(f"repos/{want.repo}/rulesets/{want.ruleset_id}", want.timeout), "ruleset")
    hits = compare(detail, want)
    note(f"ruleset {want.ruleset_id}（{_text(detail, 'name', 'ruleset')}）比了 {len(want.rule_types)} 條規則，違規 {len(hits)}")
    return hits


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="合併門口的設定從伺服器回讀比對：跟卡上期望不一致就紅，讀不到回 2",
            targets=targets,
        )
    )
