"""identity-strings-generated 的正向對照：對得到的 sha 與 run id 不咬、網址與 UUID 裡的十六進位段塗掉。

不 spawn 整支檢查（外殼要求掃描根在 repo 裡）；sha 那一半用 git_sandbox 造一筆真提交來對，
run id 那一半餵一個手造的 Mirror。判準（長度、位數、塗法）從主卡讀，不在這裡重寫一份。
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from governance import cloud_receipts
from governance.checks import identity_strings_generated as isg
from tests.conftest import REPO, GitSandbox

CARD = REPO / "governance" / "rules" / "identity-strings-generated.toml"


def rules() -> isg.Rules:
    data = tomllib.loads(CARD.read_text(encoding="utf-8"))
    settings = data["settings"]
    assert isinstance(settings, dict)
    return isg.read_rules(settings)


def test_urls_and_uuids_are_masked_before_looking_for_shas() -> None:
    text = "示範頁 https://example.invalid/artifact/07a6f39a-21a6-4993-8844-f1a95b78750d 與 UUID 0123abcd-0000-4000-8000-0123456789ab。"
    assert isg.sha_candidates(isg.masked(text, rules()), rules()) == []


def test_short_hex_without_letters_or_too_short_is_not_a_sha() -> None:
    text = "ruleset 22615925、行號 1234567、短碼 abc123。"
    assert isg.sha_candidates(text, rules()) == []


def test_a_real_commit_resolves_and_a_fake_one_does_not(git_sandbox: GitSandbox) -> None:
    root = git_sandbox.root
    (root / "f").write_text("x\n", encoding="utf-8")
    git_sandbox.git("add", "f")
    git_sandbox.git("commit", "-q", "-m", "one")
    sha = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    assert isg.sha_resolves(root, sha)
    assert isg.sha_resolves(root, sha[:10])
    assert not isg.sha_resolves(root, "f" * len(sha))


def test_run_ids_in_the_mirror_are_known_and_others_are_not(tmp_path: Path) -> None:
    body: dict[str, object] = {"schema": 1, "run": {"run_id": 34439894217}, "written_by": {"run_id": "34440152750"}}
    receipt = cloud_receipts.Receipt(name="34439894217-1.json", body=body, schema=1)
    mirror = cloud_receipts.Mirror(receipts=(receipt,), provenance={}, folder=tmp_path)
    known = isg.receipt_run_ids(mirror)
    assert {"34439894217", "34440152750"} <= known
    assert isg.run_id_candidates("run 34439894217 綠了、run 99999999999 沒有", rules()) == ["34439894217", "99999999999"]
    assert "99999999999" not in known
