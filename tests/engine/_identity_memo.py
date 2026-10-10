"""引擎考卷共用：物理身分與模態身分按完整呼叫記憶（#586）；每個工人第一次照舊真算。"""
from __future__ import annotations

import pytest

from aosr.reporting import modal_diagnosis, physics_identity
from tests.engine._gui_cache import _memoize

# 考身分本身的三支照真算：它們會改身分裡面的零件、改暫存程式樹再重算，記憶會讓它們看不到改變。
IDENTITY_SELF_TESTS = frozenset({"test_physics_identity.py", "test_modal_diagnosis.py", "test_modal_lookup.py"})
# 每個消費者自己綁的名字都要換；在函式裡才匯入的（scheme_cli、gui.__main__）走來源模組那一格。
PHYSICS_IDENTITY_NAMES = ("aosr.reporting.physics_identity.physics_identity", "aosr.search.cli.physics_identity",
                          "aosr.gui.app.physics_identity")
MODAL_IDENTITY_NAMES = ("aosr.reporting.modal_diagnosis.modal_identity", "aosr.gui.modal_jobs.modal_identity",
                        "aosr.search.modal_attach.modal_identity", "aosr.search.report_modal.modal_identity")
remembered_physics_identity = _memoize(physics_identity.physics_identity)
remembered_modal_identity = _memoize(modal_diagnosis.modal_identity)


def apply_identity_memo(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.path.name in IDENTITY_SELF_TESTS:
        return
    for name in PHYSICS_IDENTITY_NAMES:
        monkeypatch.setattr(name, remembered_physics_identity)
    for name in MODAL_IDENTITY_NAMES:
        monkeypatch.setattr(name, remembered_modal_identity)
