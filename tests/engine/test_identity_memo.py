"""#586：引擎考卷的身分記憶掛在每個消費者自己的名字上，交出的值等於真算。"""
from __future__ import annotations

import importlib
from pathlib import Path

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.reporting import modal_diagnosis, physics_identity
from tests.engine import _identity_memo as memo


def _resolve(name: str) -> object:
    module, attribute = name.rsplit(".", 1)
    return getattr(importlib.import_module(module), attribute)


def test_every_consumer_name_carries_the_memo() -> None:
    assert {id(_resolve(name)) for name in memo.PHYSICS_IDENTITY_NAMES} == {id(memo.remembered_physics_identity)}
    assert {id(_resolve(name)) for name in memo.MODAL_IDENTITY_NAMES} == {id(memo.remembered_modal_identity)}


def test_memo_returns_the_real_identity() -> None:
    capabilities = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    assert Path(memo.real_physics_identity.__code__.co_filename).name == "physics_identity.py"
    assert Path(memo.real_modal_identity.__code__.co_filename).name == "modal_diagnosis.py"
    real = physics_identity.physics_identity_parts(capabilities=capabilities, directivity=directivity).identity
    assert physics_identity.physics_identity(capabilities=capabilities, directivity=directivity) == real
    assert modal_diagnosis.modal_identity() == memo.real_modal_identity()
