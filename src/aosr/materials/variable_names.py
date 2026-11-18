"""構造樣板變數的名字怎麼寫（只有這一份）。

單獨一檔、什麼都不載入：``construction_diff``（算樣板，載入 JAX）與 ``gradient.surface_design``（起跑配置的設計變數名，
網頁行程也用、不准載入 JAX，#1206 第 2 支）都從這裡拿。不放進 ``construction_models``：那一檔在物理身分的程式閉包裡，
動它等於宣告物理變了（``test_physics_identity_pinned``）。
"""
from __future__ import annotations


def variable_name(layer_index: int, field: str) -> str:
    """參數字典的鍵：layer<層號>.<欄位名>，層號從房間側的一起算，跟註記的層編號同一套。"""
    return f"layer{layer_index}.{field}"
