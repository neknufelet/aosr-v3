"""能力表只改說明，物理與模態身分不變；字句取自家具決策紙第 27 條所指範圍。"""
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.reporting.physics_identity import physics_identity_parts
from aosr.reporting.modal_diagnosis import modal_identity


def test_capabilities_describe_furniture_scope_and_preserve_identities() -> None:
    table = load_capabilities(config_path("capabilities.toml"))
    texts = tuple(text for entry in table.entry for text in (*entry.not_modeled, *entry.manual_checks))
    assert "家具與桌面的反射" not in texts
    assert all("家具不在模型" not in text and "房間是空的六面盒" not in text for text in texts)
    assert "家具只算一次反射；家具與牆的混合反射未納入" in texts
    assert any("控台、螢幕" in text for text in texts)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    parts = physics_identity_parts(capabilities=table, directivity=directivity)
    assert parts.identity == "phys-v1:00eca7eaa663212a188c14de76e1a9a69b29f98ce2a6fb3247fe08e4faf75cf1"
    assert modal_identity() == "modal-v1:5dcb7e682c04552785515d72567623f0bd8c42ae82a742ac4b7f5be015955e9e"
