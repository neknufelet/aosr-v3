"""#1142 子行程（雙精度）：同一組起跑樣板與准用清單，產品設定檔與「總深度上限收窄」的那一份各算一次固定材料候選的
真核對配置（:func:`aosr.cli.start_designs`→:func:`~aosr.gradient.fixed_candidates.candidate_configurations`，不是樁），
寫兩份阻抗摘要與展開的准用清單到 ``probe.json``。``argv``：工作資料夾。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aosr import runtime


def main() -> None:
    runtime.configure_calculation_jax()

    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.material_gradient import MOUNT_RIGID, load_material_gradient_settings
    from aosr.config.paths import config_path
    from aosr.gradient import cli, fixed_candidates
    from aosr.gradient.surrogate_fixed_checks import impedances_sha256
    from tests.engine.test_fixed_checks_reuse import AXIS, CARPET_TARGET, NARROWED_DEPTH
    from tests.engine.test_material_gradient_allowed import GAP50, _table
    from tests.engine.test_material_gradient_cli import _schemes

    directory = Path(sys.argv[1])
    text = config_path("material_gradient.toml").read_text(encoding="utf-8")
    narrowed = text
    for old, new in NARROWED_DEPTH:
        narrowed = narrowed.replace(old, new, 1)
    path = directory / "settings.toml"
    path.write_text(narrowed, encoding="utf-8")
    scheme = _schemes()["scheme.json"]
    capabilities = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    allowed: dict[str, tuple[str, ...]] = {"x0": ("glass_wool_48k", CARPET_TARGET)}
    found: dict[str, object] = {"changed": narrowed != text}
    for name, settings in (("built", load_material_gradient_settings(config_path("material_gradient.toml"))),
                           ("now", load_material_gradient_settings(path))):
        designs, expanded = cli.start_designs(scheme, settings, _table(), (None, GAP50, GAP50, GAP50, None, None),
                                              mount=MOUNT_RIGID, fixed={}, caps={}, allowed=allowed,
                                              capabilities=capabilities, directivity=directivity)
        rows = fixed_candidates.candidate_configurations(designs, expanded, _table(), AXIS,
                                                         density_kg_m3=scheme.scene.density_kg_m3,
                                                         sound_speed_m_s=scheme.scene.sound_speed_m_s)
        found[name] = impedances_sha256(rows[CARPET_TARGET])
        found["expanded"] = {surface: list(targets) for surface, targets in expanded.items()}
    (directory / "probe.json").write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
