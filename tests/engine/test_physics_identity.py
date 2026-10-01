"""物理身分的載入邊界、複本變異、環境分離與命令列契約。"""
from __future__ import annotations

import ast
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

from aosr.config.capabilities import Capability, CapabilityEntry, CapabilityTable, load_capabilities
from aosr.config.directivity_defaults import DirectivityDefaults, load_directivity_defaults
from aosr.config.paths import config_path
from aosr.reporting import physics_identity as identity_module
from aosr.reporting import scheme_cli
from aosr.reporting.calculation_fingerprint import calculation_fingerprint


def _root() -> Path:
    return config_path("capabilities.toml").parents[2]


def _module_path(root: Path, module: str) -> Path:
    return root.joinpath(*module.split(".")[1:]).with_suffix(".py")


def _data_path(root: Path, name: str) -> Path:
    return root / "config" / "data" / name


def _table(root: Path) -> CapabilityTable:
    return load_capabilities(_data_path(root, config_path("capabilities.toml").name))


def _directivity(root: Path) -> DirectivityDefaults:
    return load_directivity_defaults(
        _data_path(root, config_path("directivity_defaults.toml").name))


def _parts(root: Path, *, table: CapabilityTable | None = None,
           directivity: DirectivityDefaults | None = None) -> identity_module.PhysicsIdentityParts:
    return identity_module.physics_identity_parts(
        capabilities=table if table is not None else _table(root),
        directivity=directivity if directivity is not None else _directivity(root),
        package_root=root,
    )


def _copy(tmp_path: Path) -> Path:
    return Path(shutil.copytree(_root(), tmp_path / "aosr",
                               ignore=shutil.ignore_patterns("__pycache__")))


def _replace(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    assert old in source
    updated = source.replace(old, new)
    assert updated != source
    path.write_text(updated, encoding="utf-8")


def test_closure_matches_modules_loaded_in_subprocess() -> None:
    probe = ("import sys; import aosr.reporting.physics_stage; "
             "print('\\n'.join(sorted(name for name in sys.modules "
             "if name == 'aosr' or name.startswith('aosr.'))))")
    completed = subprocess.run([sys.executable, "-B", "-c", probe], check=True,
                               capture_output=True, text=True)
    loaded = set(completed.stdout.splitlines())
    parts = _parts(_root())
    closure = set(parts.closure)
    assert "aosr.reporting.physics_stage" in loaded
    assert loaded <= closure
    for module in closure - loaded:
        imports = [item for item in parts.imports if item.module == module]
        assert imports, module
        assert all(item.in_function for item in imports), module


def test_closure_excludes_scoring_registry_evaluation_search_and_gui() -> None:
    parts = _parts(_root())
    assert {name for name in parts.closure if name.startswith("aosr.scoring")} <= {
        "aosr.scoring", "aosr.scoring.channel_group", "aosr.scoring.receiver_set"}
    forbidden = {"aosr.config.quality_targets", "aosr.reporting.result",
                 "aosr.reporting.evaluation", "aosr.reporting.pipeline",
                 "aosr.physics.reflection_window"}
    assert set(parts.closure).isdisjoint(forbidden)
    assert not any(name.startswith(("aosr.search", "aosr.gui")) for name in parts.closure)
    assert set(parts.third_party).isdisjoint({"jax", "flax"})


def test_config_path_literals_are_declared_data_files() -> None:
    parts = _parts(_root())
    literals: set[str] = set()
    for module in parts.closure:
        path = _module_path(_root(), module)
        if not path.is_file():
            path = _root().joinpath(*module.split(".")[1:], "__init__").with_suffix(".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "config_path":
                    argument = node.args[0]
                    assert isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                    literals.add(argument.value)
    assert literals == set(parts.data_files)
    assert all(_data_path(_root(), name).is_file() for name in literals)
    assert literals.isdisjoint({config_path("quality_targets.toml").name,
                                config_path("physics_constants.toml").name})


def test_third_party_imports_are_whitelisted() -> None:
    parts = _parts(_root())
    allowed = dict(identity_module.physics_dependency_versions())
    providers = metadata.packages_distributions()
    # 錨點：走訪器真的記到第三方 import（不然下面的迴圈在空集合上永遠綠）。
    assert {"numpy", "scipy", "gmsh", "pydiso", "skfem", "pydantic"} <= set(parts.third_party)
    for name in parts.third_party:
        assert name in providers
        distributions = {identity_module.normalized_distribution_name(item)
                         for item in providers[name]}
        assert distributions <= allowed.keys()
    assert allowed.keys().isdisjoint({"optuna", "sqlalchemy", "alembic", "starlette",
                                      "uvicorn", "jax", "flax"})


def _display_table(table: CapabilityTable, change: str) -> CapabilityTable:
    entry = table.for_entry("three_lane_report")
    updates: dict[str, object] = {change: "換一段說明" if change == "note" else ("換一項說明",)}
    if change == "row_note":
        row = entry.capability[0].model_copy(update={"note": "換一段條件說明"})
        updates = {"capability": (row, *entry.capability[1:])}
    changed = entry.model_copy(update=updates)
    return table.model_copy(update={"entry": tuple(
        changed if item.name == entry.name else item for item in table.entry)})


def _unrelated_change(root: Path, change: str) -> CapabilityTable:
    table = _table(root)
    if change in {"note", "row_note", "not_modeled", "manual_checks"}:
        return _display_table(table, change)
    if change == "quality":
        path = _data_path(root, config_path("quality_targets.toml").name)
        source = path.read_text(encoding="utf-8")
        tree = tomllib.loads(source)
        assert tree
        # 改登記簿真數值；無須載入品質模型，該資料不在物理閉包。
        updated, changed = re.subn(r"(=\s*)(-?\d+\.\d+)", r"\g<1>0.12345", source, count=1)
        assert changed and updated != source
        assert tomllib.loads(updated) != tree
        path.write_text(updated, encoding="utf-8")
    elif change == "search":
        for module in ("aosr.search.__init__", "aosr.search.x"):
            path = _module_path(root, module)
            path.parent.mkdir(exist_ok=True)
            path.write_text("SEARCH_ONLY = 42\n", encoding="utf-8")
    else:
        module = {"scoring": "aosr.scoring.timbre", "gui": "aosr.gui.app",
                  "comment": "aosr.physics.totals", "docstring": "aosr.physics.totals"}[change]
        path = _module_path(root, module)
        if change == "docstring":
            _replace(path, "鏡像法的總壓力", "鏡像法算出的總壓力")
        else:
            source = path.read_text(encoding="utf-8")
            addition = "\n# 新註解\n" if change == "comment" else "\ndef unrelated() -> int:\n    return 42\n"
            path.write_text(source + addition, encoding="utf-8")
    return table


@pytest.mark.parametrize("change", ["scoring", "quality", "search", "gui", "comment",
                                    "docstring", "note", "row_note", "not_modeled", "manual_checks"])
def test_changes_that_must_not_flip_identity(tmp_path: Path, change: str) -> None:
    root = _copy(tmp_path)
    before = _parts(root)
    table = _unrelated_change(root, change)
    after = _parts(root, table=table)
    assert after == before


def _physical_table(table: CapabilityTable, change: str) -> CapabilityTable:
    name = "source_directivity" if change == "source_status" else "three_lane_report"
    entry = table.for_entry(name)
    row = entry.capability[0]
    # 每一刀只改一格：狀態那兩刀只改 status（experimental 換 unsupported，evidence 照舊是空的）。
    choices: dict[str, dict[str, object]] = {
        "evidence": {"evidence": ("new-evidence",)},
        "status": {"status": "unsupported"},
        "source_status": {"status": "unsupported"},
        "frequency": {"frequency_hz": (row.frequency_hz[0], row.frequency_hz[1] * 0.5)},
        "outputs": {"outputs": (*row.outputs, "extra_output")},
    }
    updates = choices[change]
    assert row.status == "experimental" and not row.evidence
    row = row.model_copy(update=updates)
    changed = entry.model_copy(update={"capability": (row, *entry.capability[1:])})
    return table.model_copy(update={"entry": tuple(
        changed if item.name == name else item for item in table.entry)})


def _physical_change(root: Path, change: str) -> tuple[CapabilityTable, DirectivityDefaults]:
    table, directivity = _table(root), _directivity(root)
    replacements = {
        "operation": ("aosr.physics.totals", "highest + 1", "highest - 1"),
        "validation": ("aosr.reporting.validation", '"必填"', '"一定要填"'),
        "runtime": ("aosr.runtime", "DEFAULT_PARDISO_THREADS = 1", "DEFAULT_PARDISO_THREADS = 2"),
    }
    if change in replacements:
        module, old, new = replacements[change]
        _replace(_module_path(root, module), old, new)
    elif change == "data":
        _replace(_data_path(root, config_path("fem_lane.toml").name), "splay_cap = 0.5", "splay_cap = 0.4")
    elif change == "directivity":
        curve = directivity.two_parameter.model_copy(update={
            "beta_corner_hz": directivity.two_parameter.beta_corner_hz + 1.0})
        directivity = directivity.model_copy(update={"two_parameter": curve})
    elif change not in {"numpy", "machine"}:
        table = _physical_table(table, change)
    return table, directivity


@pytest.mark.parametrize("change", ["operation", "validation", "runtime", "data", "status",
                                    "evidence", "source_status", "frequency", "outputs",
                                    "directivity", "numpy", "machine"])
def test_changes_that_must_flip_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    root = _copy(tmp_path)
    before = _parts(root)
    table, directivity = _physical_change(root, change)
    if change == "numpy":
        original = metadata.version
        monkeypatch.setattr(metadata, "version", lambda name:
                            original(name) + "+changed" if name == "numpy" else original(name))
    if change == "machine":
        monkeypatch.setattr(platform, "machine", lambda: "another-cpu")
    after = _parts(root, table=table, directivity=directivity)
    assert after.identity != before.identity
    if change in {"numpy", "machine"}:
        assert after.code_digest == before.code_digest
        assert after.environment_digest != before.environment_digest
    else:
        assert after.code_digest != before.code_digest
        assert after.environment_digest == before.environment_digest


def test_physics_reads_only_declared_capability_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.physics import three_lane_report
    from aosr.reporting.physics_stage import solve_scheme_physics
    from tests.engine import _scoring_source_model_control as control
    from tests.engine.test_scheme_pipeline import _many_fem, _scheme

    recorded: set[str] = set()
    original = CapabilityTable.for_entry

    def record(table: CapabilityTable, name: str) -> CapabilityEntry:
        recorded.add(name)
        return original(table, name)

    monkeypatch.setattr(CapabilityTable, "for_entry", record)
    for module, name, fake in control.STAND_INS:
        monkeypatch.setattr(module, name, fake)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
    table, directivity = _table(_root()), _directivity(_root())
    for model in ("omnidirectional", "product_default"):
        recorded.clear()
        scheme = _scheme("wall-1").model_copy(update={"source_model": model})
        checked, physics = solve_scheme_physics(scheme, capabilities=table, directivity=directivity)
        assert physics.pairs and checked.source_model == model
        assert "three_lane_report" in recorded
        assert recorded <= set(identity_module.PHYSICS_CAPABILITY_ENTRIES)
        if model == "product_default":
            assert "source_directivity" in recorded


def test_scheme_cli_identity_prints_both(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / config_path("capabilities.toml").name
    path.write_bytes(config_path("capabilities.toml").read_bytes())
    expected = identity_module.physics_identity(capabilities=load_capabilities(path),
                                                directivity=_directivity(_root()))
    whole = calculation_fingerprint(capabilities_path=path)
    exit_code = scheme_cli.main(["identity", "--capabilities", str(path)])
    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        f"物理身分 {expected}", f"整支程式指紋 {whole}"]


def _write_module(root: Path, name: str, source: str) -> None:
    path = _module_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "aosr"
    for name in ("aosr.__init__", "aosr.reporting.__init__"):
        _write_module(root, name, "")
    return root


def test_import_forms_and_parent_initializers(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    sources = {
        "aosr.reporting.physics_stage": (
            "import aosr.leaf\nfrom . import sibling\nfrom .nested import value\n"
            "from ..box import child\nfrom typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    import aosr.typed\n"
            "async def later():\n    from . import lazy\n"),
        "aosr.box.__init__": "import aosr.init_dependency\n",
        "aosr.reporting.nested": "value = 1\n",
        "aosr.box.child": "",
        "aosr.leaf": "",
        "aosr.reporting.sibling": "",
        "aosr.reporting.lazy": "",
        "aosr.typed": "",
        "aosr.init_dependency": "",
    }
    for name, source in sources.items():
        _write_module(root, name, source)
    closure = identity_module.physics_import_closure(package_root=root)
    expected = {name.removesuffix(".__init__") for name in sources} | {"aosr", "aosr.reporting"}
    assert set(closure.modules) == expected
    assert closure.modules == tuple(sorted(expected))
    assert any(item.module == "aosr.typed" and not item.in_function for item in closure.imports)
    assert any(item.module == "aosr.reporting.lazy" and item.in_function for item in closure.imports)


@pytest.mark.parametrize("failure", ["missing_entry", "missing_source", "syntax", "dynamic_data", "missing_data"])
def test_invalid_source_or_data_raises(tmp_path: Path, failure: str) -> None:
    root = _synthetic_root(tmp_path)
    source = {"missing_source": "import aosr.absent\n", "syntax": "def broken(\n",
              "dynamic_data": "config_path(variable)\n",
              "missing_data": f"config_path({config_path('fem_lane.toml').name!r})\n"}.get(failure, "")
    if failure != "missing_entry":
        _write_module(root, "aosr.reporting.physics_stage", source)
    with pytest.raises((FileNotFoundError, SyntaxError, ValueError)):
        _parts(root, table=_table(_root()), directivity=_directivity(_root()))


def test_docstrings_are_removed_at_every_scope(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    module = "aosr.reporting.physics_stage"
    source = ('"""old docs"""\nclass C:\n    """old docs"""\n'
              '    def f(self):\n        """old docs"""\n        return "value"\n'
              'async def g():\n    """old docs"""\n    return 3\n')
    _write_module(root, module, source)
    table, directivity = _table(_root()), _directivity(_root())
    before = _parts(root, table=table, directivity=directivity)
    _replace(_module_path(root, module), "old docs", "new docs")
    assert _parts(root, table=table, directivity=directivity) == before
    _replace(_module_path(root, module), 'return "value"', 'return "new value"')
    assert _parts(root, table=table, directivity=directivity).identity != before.identity


class _Distribution:
    def __init__(self, requirements: tuple[str, ...]) -> None:
        self.requires = requirements


def test_dependency_markers_normalization_and_missing_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    def distribution(name: str) -> _Distribution:
        requirements = {"numpy": ("Some_Name>=1", "extra-only; extra == 'test'",
                                  "wrong-python; python_version < '0'"),
                        "some-name": ("leaf.name; python_version >= '3'",),
                        "leaf-name": ("numpy",)}
        return _Distribution(requirements.get(name, ()))

    monkeypatch.setattr(metadata, "distribution", distribution)
    monkeypatch.setattr(metadata, "version", lambda name: "1.0")
    versions = dict(identity_module.physics_dependency_versions())
    assert {"some-name", "leaf-name", "numpy"} <= versions.keys()
    assert versions.keys().isdisjoint({"extra-only", "wrong-python"})

    def missing(name: str) -> _Distribution:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    with pytest.raises(metadata.PackageNotFoundError):
        identity_module.physics_dependency_versions()


def test_capability_fields_are_each_decided() -> None:
    """能力表每一格都要明確決定收不收：新增一格不准被安靜略過。"""
    assert set(Capability.model_fields) - {"note"} == set(identity_module._CAPABILITY_FIELDS)
    assert set(CapabilityEntry.model_fields) - {"note", "not_modeled", "manual_checks"} == {
        "name", "module", "capability"}


@pytest.mark.parametrize("source", [
    "from ..config.paths import config_path as _cp\n_cp(variable)\n",
    "from aosr.config.paths import CONFIG_DIR\nTABLE = CONFIG_DIR / 'x'\n",
    "import aosr.config.paths\nTABLE = aosr.config.paths.CONFIG_DIR / 'x'\n",
])
def test_unmeasurable_data_paths_are_refused(tmp_path: Path, source: str) -> None:
    """相對 import 加別名的 config_path 也要認得；自己拿 CONFIG_DIR 組路徑一律拒收。"""
    root = _synthetic_root(tmp_path)
    _write_module(root, "aosr.reporting.physics_stage", source)
    with pytest.raises(ValueError):
        identity_module.physics_import_closure(package_root=root)
