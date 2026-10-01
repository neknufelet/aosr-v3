"""CI 那兩步改用「深度上限」與「跳過三針」的設定之後，那個設定本身也要被樣本餵過。

規矩卡必填欄位與離開碼誠實在真 repo 頂層不再重做後設測試第 2～5 回已做的事（#586）：CI 那一步
以步驟層 ``env:`` 帶設定。後設測試第 2 回照舊在深度 0、不開開關餵樣本，那個設定下會不會紅
沒有人證明——有人把「只驗欄位」那條路或開關的範圍改寬，CI 那一步照綠，第 2 回也照綠。

這一題讀 verify.yml 那一步**實際**帶的 ``env:``，用同一組設定逐份餵兩張卡的必紅樣本與控制樣本：
只有「交給後設測試接手」的那幾份准不紅，其餘每一份都必須回 1。准不紅的名單寫死在這裡：它就是
開關的範圍，名單外的樣本變成不紅（開關變寬）或名單內的樣本變紅（名單過期）都要有人看見。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from governance.exit_codes import CLEAN, VIOLATION
from governance.loader import Card
from tests.test_fixture_runner import CARDS, REPO, _run

WORKFLOW = REPO / ".github" / "workflows" / "verify.yml"

# 卡名 → 那個設定下准回 0 的樣本（由後設測試哪一回接手）。
META_COVERED = {
    # 第二關（會咬）專屬的樣本：深度到上限就不遞迴，由第 2 回在深度 0 餵。
    "rule-card-required-fields": {"case-canary-check-never-bites"},
    # 跳過的三針各自的樣本：由第 3、4、5 回接手。
    "check-exit-code-honest": {
        "case-control-sample-not-red",
        "case-missing-scan-root-returns-zero",
        "case-missing-tool-counted-as-clean",
    },
}


def _step_env(card: Card) -> dict[str, str]:
    """verify.yml 裡跑這張卡那一步的步驟層 env；找不到那一步或沒有 env 就紅（設定不在就沒有東西可驗）。"""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if f"-m {card.check_module} " in str(step.get("run", ""))
    ]
    assert steps, f"verify.yml 找不到跑 {card.check_module} 的那一步"
    envs = [step.get("env") for step in steps]
    assert all(isinstance(env, dict) and env for env in envs), f"{card.id} 那一步沒有步驟層 env：{envs}"
    first = {str(key): str(value) for key, value in envs[0].items()}
    assert all({str(k): str(v) for k, v in env.items()} == first for env in envs), f"{card.id} 各步 env 不一致"
    return first


@pytest.mark.parametrize("card_id", sorted(META_COVERED))
def test_ci_step_settings_still_bite_every_sample_not_handed_to_meta_tests(card_id: str) -> None:
    card = next(card for card in CARDS if card.id == card_id)
    env = _step_env(card)
    cases = card.negative_cases(REPO)
    control = card.control_path(REPO)
    if control.is_dir() and control not in cases:
        cases.append(control)
    assert cases, f"{card_id} 沒有樣本可餵"

    quiet: set[str] = set()
    for case in cases:
        proc = _run(card, case, extra_env=env)
        assert proc.returncode in (CLEAN, VIOLATION), (
            f"{card_id} 用 CI 設定 {env} 餵 {case.name} 回 {proc.returncode}"
        )
        if proc.returncode == CLEAN:
            quiet.add(Path(case).name)
    assert quiet == META_COVERED[card_id], (
        f"{card_id} 用 CI 那一步的設定 {env}，不紅的樣本是 {sorted(quiet)}，"
        f"應該剛好是交給後設測試接手的 {sorted(META_COVERED[card_id])}"
    )
