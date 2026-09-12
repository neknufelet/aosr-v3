"""票 #134 第二刀：``FREQS_HZ_IDENTITY`` 跟 x64 開關無關（另起一個行程才問得出來）。

上一代那一支的檔頭寫著：這條 tuple 被釘成 float32，好讓 ``config_hash`` 與存下去的酬載
**不管載入的時候 ``jax_enable_x64`` 是開還是關**都一樣。同一個行程裡問不出這件事——常數在
import 的時候就算完了，開關後來再開也不會回頭重算。所以這一支另起一個**開了 x64** 的行程，
在新家載入一次，跟答案檔裡 donor 那一跑的兩格逐格比。

子行程的東西寫在 pytest 的 ``tmp_path`` 底下（不是真的 repo），``PYTHONPATH`` 由這個模組
實際從哪裡被載入推出來，不是寫死 repo 路徑。
"""
from __future__ import annotations

from pathlib import Path

from tests.engine import _materials_cut2_answers as answers


def test_the_identity_tuple_is_the_same_with_x64_on(tmp_path: Path) -> None:
    """新家在 x64 開著的行程裡量到的那條 tuple，要等於 donor 在同樣設定下量到的。"""
    here = answers.x64_identity_here(tmp_path)
    there = answers.as_mapping(answers.probe_block("x64_identity").get("on"), "答案檔的 x64 on")
    assert here.get("x64") is True, "子行程沒有真的把 x64 打開——那這一題什麼都沒問到"
    assert there.get("x64") is True, "答案檔那一格不是在 x64 開著的時候量的"
    gaps = answers.differences(here.get("identity"), there.get("identity"), "x64_identity.on")
    assert gaps == [], "x64 開著的時候新家跟上一代對不上：\n" + "\n".join(gaps)


def test_turning_x64_on_does_not_change_the_identity_tuple(tmp_path: Path) -> None:
    """開與關量到的是同一組值——這就是「釘成 float32」那句話的意思。

    兩邊都比：donor 自己的 on／off 要相等（答案檔裡那兩格），新家的 on 要等於新家的 off
    （後者就是凍結常數 ``freq_axis.const.FREQS_HZ_IDENTITY``，由產品考卷在 x64 關著的
    這一跑讀出來）。
    """
    block = answers.probe_block("x64_identity")
    donor_on = answers.as_mapping(block.get("on"), "答案檔的 x64 on")
    donor_off = answers.as_mapping(block.get("off"), "答案檔的 x64 off")
    assert answers.differences(donor_on.get("identity"), donor_off.get("identity"), "donor") == []
    frozen = answers.frozen_block("freq_axis", "constants", "FREQS_HZ_IDENTITY")
    assert answers.differences(donor_off.get("identity"), frozen, "donor.off_vs_frozen") == []
    here = answers.x64_identity_here(tmp_path)
    gaps = answers.differences(here.get("identity"), frozen, "engine.on_vs_frozen")
    assert gaps == [], "新家在 x64 開著的時候那條 tuple 變了：\n" + "\n".join(gaps)
