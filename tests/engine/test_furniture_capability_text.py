"""能力表字句取自家具決策紙第 27 條所指範圍；模態身分不變。"""
from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.gui.capability_view import capability_lists
from aosr.reporting.modal_diagnosis import modal_identity


def test_capabilities_describe_furniture_scope_and_preserve_identities() -> None:
    table = load_capabilities(config_path("capabilities.toml"))
    texts = tuple(text for entry in table.entry for text in (*entry.not_modeled, *entry.manual_checks))
    assert "家具與桌面的反射" not in texts
    assert all("家具不在模型" not in text and "房間是空的六面盒" not in text for text in texts)
    assert "家具只算一次反射；家具與牆的混合反射未納入" in texts
    assert any("控台、螢幕" in text for text in texts)
    assert modal_identity() == "modal-v1:5dcb7e682c04552785515d72567623f0bd8c42ae82a742ac4b7f5be015955e9e"


def test_display_lists_never_repeat_a_clause_across_items() -> None:
    # 清單只合併整句相同的項目；兩節各寫一半相同的句子，畫面上就會重複一句（第七步截圖看到過）。
    lists = capability_lists(load_capabilities(config_path("capabilities.toml")))
    for field, items in lists.items():
        clauses = [clause for item in items for clause in item.split("；")]
        assert [clause for clause in dict.fromkeys(clauses) if clauses.count(clause) > 1] == [], field
