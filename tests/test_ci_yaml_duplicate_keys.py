"""雲端那張卡第⑧條（同一層不准重複鍵）的節點走訪：別名指回自己不准卡死、同一塊被別名引用幾次都只報一次（#326 複查）。

卡死不是 0／1／2 任何一個離開碼，到了雲端就是無聲掛到 timeout 被砍——正好是這張卡要抓的事。
"""
import pytest

from governance.checks import ci_jobs_cannot_die_quietly as ci


@pytest.mark.parametrize("text", ["x-loop: &r [1, *r]\n", "x-loop: &r {k: *r}\n"], ids=["sequence", "mapping"])
def test_self_referencing_aliases_finish_clean(text: str) -> None:
    assert ci._duplicate_key_problems("w.yml", text) == []


def test_a_duplicate_inside_an_anchor_is_reported_once() -> None:
    text = "base: &b\n  run: a\n  run: b\nx: *b\ny: *b\nz: *b\n"
    assert ci._duplicate_key_problems("w.yml", text) == [
        "w.yml 第 3 行：同一層重複的鍵 'run'（第 2 行已經有了）——剖析只留最後一個，前一個靜靜不見"]
