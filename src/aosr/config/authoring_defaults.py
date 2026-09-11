"""M13 authoring schema defaults and id namespace helpers."""

from __future__ import annotations

from collections.abc import Mapping

APP_WALL_PREFIX = "app:wall:"
APP_PATCH_PREFIX = "app:patch:"
RHINO_PREFIX = "rhino:"

GRID_PRESETS: Mapping[str, tuple[int, int]] = {
    "coarse": (2, 3),
    "medium": (3, 4),
    "fine": (4, 6),
}

TOTAL_THICKNESS_MAX_M = 0.20
BARE_MATERIAL_ID = "masonry_bare"

BARE_TEMPLATE_ID = "bare"
BROADBAND_POROUS_TEMPLATE_ID = "broadband_porous"
RESONANT_PANEL_TEMPLATE_ID = "resonant_panel"
CUSTOM_STACK_TEMPLATE_ID = "custom_stack"
# Fixed-alpha materials: authored as a direct 8-band absorption table (α[K])
# instead of a TMM layer stack. They never enter the optimiser (design §3).
FIXED_ALPHA_TEMPLATE_ID = "fixed_alpha"

FLOW_RESISTIVITY_MIN = 1.0e3
FLOW_RESISTIVITY_MAX = 1.0e5
SPACING_MIN_M = 0.010
SPACING_MAX_M = 0.050
HOLE_RATIO_MIN = 0.05
HOLE_RATIO_MAX = 0.60
PERFORATED_FIXED_THICKNESS_M = 0.009
FABRIC_RS = 250.0
FABRIC_MS = 0.05

LayerSpecDict = dict[str, str | float]

BARE_LAYER_STACK_SPEC: tuple[LayerSpecDict, ...] = ()
BROADBAND_POROUS_LAYER_STACK_SPEC: tuple[LayerSpecDict, ...] = (
    {"type": "air", "thickness_mm": 50.0},
    {"type": "porous", "thickness_mm": 50.0, "flow_resistivity": 10000.0},
    {"type": "fabric", "Rs": 250.0, "ms": 0.05},
)
RESONANT_PANEL_LAYER_STACK_SPEC: tuple[LayerSpecDict, ...] = (
    {"type": "air", "thickness_mm": 80.0},
    {"type": "porous", "thickness_mm": 40.0, "flow_resistivity": 10000.0},
    {
        "type": "perforated",
        "thickness_mm": 9.0,
        "hole_dia_mm": 6.0,
        "spacing_mm": 28.0,
    },
)


def app_wall_id(wall_id: str) -> str:
    """Return the app namespace id for a parent shoebox wall."""
    return f"{APP_WALL_PREFIX}{wall_id}"


def app_patch_id(wall_id: str, row: int, col: int) -> str:
    """Return the deterministic app namespace id for a uniform wall patch."""
    return f"{APP_PATCH_PREFIX}{wall_id}:r{row}c{col}"


def rhino_boundary_id(guid: str) -> str:
    """Return the reserved Rhino namespace id for an imported boundary."""
    return f"{RHINO_PREFIX}{guid}"
