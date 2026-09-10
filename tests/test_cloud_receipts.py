"""三張收據卡共用的放行那條路：具名放過一份壞收據的時候，三支檢查都要跳過它；過期或寫壞就不跳。

不 spawn 任何東西：直接餵 judge() 一個手造的 Mirror 與 settings。掃描根必須在 repo 裡（外殼會擋），
所以這裡不走整支檢查，只戳判決那一層——放行的名單與日期是共用零件 allowed_names 算的，三支都走它。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from governance import cloud_receipts
from governance.checks import four_roles_different_actors, receipt_authority_is_the_cloud_run, receipt_schema_complete
from governance.exit_codes import ToolBroken

TODAY = date(2026, 9, 11)
NAME = "34439894217-1.json"
BOT = "41898282+github-actions[bot]@users.noreply.github.com"
FIXTURE = Path(__file__).resolve().parents[1] / "governance" / "fixtures" / "receipt-authority-is-the-cloud-run" / "control"


def bad_receipt() -> dict[str, object]:
    """三張卡都會咬的一份收據：authority 是 local、checks 是空的、寫收據的就是被查的那一跑。"""
    body = json.loads((FIXTURE / "governance" / "receipts" / "cloud" / "receipts" / NAME).read_text(encoding="utf-8"))
    assert isinstance(body, dict)
    body["checks"] = []
    body["written_by"] = {"run_id": body["run"]["run_id"], "attempt": 1}
    return body


def mirror_of(body: dict[str, object]) -> cloud_receipts.Mirror:
    receipt = cloud_receipts.Receipt(name=NAME, body=body, schema=int(str(body["schema"])))
    provenance: dict[str, object] = {"ref": "x", "commit": "y", "files": {NAME: {"commit": "z", "author_email": BOT, "committer_email": BOT}}}
    return cloud_receipts.Mirror(receipts=(receipt,), provenance=provenance, folder=Path())


def settings_with(allow: list[dict[str, str]] | None) -> dict[str, object]:
    base: dict[str, object] = {
        "newest_schema": 2,
        "authority_value": "cloud-run",
        "success_conclusion": "success",
        "signal_exit_base": 128,
        "machine_emails": [BOT],
        "sha_hex_len": 40,
        "sha256_hex_len": 64,
        "report_optional_names": ["pytest"],
        "required_top": {"1": ["schema", "run", "checks", "written_by"]},
        "required_run": {"1": ["run_id", "attempt", "head_sha"]},
        "required_written_by": {"1": ["run_id"]},
        "required_check": {"1": ["name", "exit_code"]},
        "required_pytest": {"1": ["tests"]},
    }
    if allow is not None:
        base["allow"] = allow
    return base


Judge = Callable[[cloud_receipts.Mirror, dict[str, object]], list[str]]


def judges() -> list[tuple[str, Judge]]:
    return [
        ("schema", lambda m, s: receipt_schema_complete.judge(m, s, "卡")),
        ("authority", receipt_authority_is_the_cloud_run.judge),
        ("four-roles", four_roles_different_actors.judge),
    ]


def test_without_allow_every_card_bites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_receipts, "date", _FixedDate)
    for name, judge in judges():
        assert judge(mirror_of(bad_receipt()), settings_with(None)), f"{name} 沒咬那份壞收據"


def test_named_unexpired_allow_skips_the_receipt_on_every_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_receipts, "date", _FixedDate)
    allow = [{"path": NAME, "reason": "測試：假裝這一份出過事、拍板放過。", "expires": "2026-12-08"}]
    for name, judge in judges():
        assert judge(mirror_of(bad_receipt()), settings_with(allow)) == [], f"{name} 沒跳過被放行的那一份"


def test_expired_allow_does_not_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_receipts, "date", _FixedDate)
    allow = [{"path": NAME, "reason": "測試：放過但到期了。", "expires": "2026-01-01"}]
    for name, judge in judges():
        assert judge(mirror_of(bad_receipt()), settings_with(allow)), f"{name} 把過期的放行當成有效"


def test_allow_naming_another_receipt_does_not_skip() -> None:
    allow = [{"path": "999-1.json", "reason": "測試：放的是別份。", "expires": "2026-12-08"}]
    assert cloud_receipts.allowed_names(settings_with(allow), TODAY) == {"999-1.json"}
    assert receipt_authority_is_the_cloud_run.judge(mirror_of(bad_receipt()), settings_with(allow))


def test_malformed_allow_is_tool_broken() -> None:
    for broken in ({"path": NAME, "reason": "沒有到期日"}, {"path": NAME, "expires": "2026-12-08"}, {"path": NAME, "reason": "日期壞掉", "expires": "someday"}):
        with pytest.raises(ToolBroken):
            cloud_receipts.allowed_names(settings_with([broken]), TODAY)


class _FixedDate(date):
    """把 date.today() 釘在測試的今天，放行的到期判斷才不會跟著日曆漂。"""

    @classmethod
    def today(cls) -> "_FixedDate":
        return cls(TODAY.year, TODAY.month, TODAY.day)
