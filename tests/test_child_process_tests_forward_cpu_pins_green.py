"""三格釘選那張卡該放過的寫法：餵一棵綠樹下去必須回 0。

樣本樹只證明會紅；「小工具取別名或用模組.函式叫也算包好、凍結處數剛好相等不咬、tests/ 外面不管、
把函式當值傳出去與 getattr 取（卡上寫明看不到的那兩種）不咬」要另開一棵綠樹證明。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from governance.checks import child_process_tests_forward_cpu_pins as subject

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "governance/fixtures/child-process-tests-forward-cpu-pins/control"
CARD_REL = "governance/rules/child-process-tests-forward-cpu-pins.toml"
ENGINE = ("tests", "engine")
FROZEN_REL = "/".join((*ENGINE, "test_frozen.py"))
FROZEN = (
    f'\n[[settings.allow]]\npath = "{FROZEN_REL}"\ncalls = 2\n'
    'reason = "綠樹：凍結當時兩處，現在也兩處。"\nexpires = "2099-12-31"\n'
)
HELPER_REL = "/".join((*ENGINE, "_cpu_pins.py"))
WRAPPED_REL = "/".join((*ENGINE, "test_wrapped.py"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _green_root(tmp_path: Path) -> Path:
    root = tmp_path / "green"
    for rel in (CARD_REL, HELPER_REL, WRAPPED_REL):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES / rel, root / rel)
    return root


def test_shapes_the_card_lets_through_are_green(tmp_path: Path) -> None:
    root = _green_root(tmp_path)
    card = root / CARD_REL
    card.write_text(card.read_text(encoding="utf-8") + FROZEN, encoding="utf-8")
    _write(root.joinpath(*ENGINE, "test_helper_alias.py"),
           "from aosr.runtime import child_process_env\nfrom tests.engine._cpu_pins import forward_cpu_pins as pin\n\n\n"
           "def test_child() -> None:\n    assert pin(child_process_env(threads=1))\n")
    _write(root.joinpath(*ENGINE, "test_helper_attr.py"),
           "from aosr import runtime\nfrom tests.engine import _cpu_pins\n\n\n"
           "def test_child() -> None:\n    assert _cpu_pins.forward_cpu_pins(runtime.child_process_env())\n")
    _write(root.joinpath(*ENGINE, "test_frozen.py"),
           "from aosr.runtime import child_process_env\n\n\n"
           "def test_child() -> None:\n    assert child_process_env() and child_process_env(threads=1)\n")
    _write(root.joinpath(*ENGINE, "test_known_gaps.py"),
           "from aosr import runtime\nfrom aosr.runtime import child_process_env\n\n\n"
           "def test_child() -> None:\n    factory = child_process_env\n    assert factory and getattr(runtime, 'child_process_env')()\n")
    _write(root.joinpath("src", "aosr", "search", "worker.py"),
           "from aosr.runtime import child_process_env\n\n\ndef spawn() -> dict[str, str]:\n    return child_process_env(threads=1)\n")
    assert subject.check(root, _tree(root)) == []


def test_targets_are_the_tests_python_and_the_rule_cards(tmp_path: Path) -> None:
    root = _green_root(tmp_path)
    _write(root.joinpath("src", "aosr", "runtime.py"), "def child_process_env() -> dict[str, str]:\n    return {}\n")
    _write(root.joinpath(*ENGINE, "data.json"), "{}\n")
    picked = {p.relative_to(root).as_posix() for p in subject.targets(root, _tree(root))}
    assert picked == {CARD_REL, HELPER_REL, WRAPPED_REL}
