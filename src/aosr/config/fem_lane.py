"""FEM-lane structural constants (M15-P3.3 SSOT).

Frozen, optimiser-untouched STRUCTURAL constants for the differentiable 3D FEM
shape-optimisation lane (design ``design_20260615-m15-p3.3-production-fem-solver.md``
§4.4, §6). Solver / anti-cheat *structure* lives here as Python constants.
The shape-search range is the deliberate exception: ``SPLAY_CAP`` is loaded
from the ``fem_lane.toml`` in this package's ``data`` directory so geometry construction and later optimizer
decode share the requested runtime-config SSOT.

They were provisional locals (``_T0_SV_*``) in the previous generation's physics layer (``aosr-v2``'s `optimize_material` module; that tree is not in this repo)
during the T0 plumbing stage; T4 promotes them here as the single source of
truth. No other module may redefine them — ``import`` from here.
"""

from __future__ import annotations

import logging
import math
import tomllib
from pathlib import Path
from .paths import config_path

logger = logging.getLogger(__name__)


def _load_splay_cap(path: Path) -> float:
    """Load the symmetric shape-splay bound from the FEM-lane TOML SSOT."""
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    try:
        value = float(data["shape"]["splay_cap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} must define a numeric [shape].splay_cap"
        ) from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            f"{path} [shape].splay_cap must be finite and in (0, 1), got {value}"
        )
    logger.debug("loaded FEM-lane symmetric splay cap %.6g from %s", value, path)
    return value


def _load_min_octagon_edge_m(path: Path) -> float:
    """Load the octagon shape-search minimum edge from the TOML SSOT."""
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    try:
        value = float(data["shape"]["min_octagon_edge_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} must define a numeric [shape].min_octagon_edge_m"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{path} [shape].min_octagon_edge_m must be finite and > 0, got {value}"
        )
    logger.debug("loaded octagon minimum edge %.6g m from %s", value, path)
    return value


def _load_position_defaults(path: Path) -> tuple[int, float, float, float, str, float]:
    """Load PMO6 runtime defaults from the FEM-lane TOML SSOT."""
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    try:
        section = data["position"]
        n_trials = int(section["n_trials_default"])
        default_height = float(section["default_height_m"])
        centerline_tol = float(section["centerline_tol_m"])
        min_source_source = float(section["min_source_source_m"])
        objective_rule = str(section["objective_rule_default"])
        objective_eps = float(section["objective_flatness_eps_default"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} must define [position] n_trials_default, default_height_m, "
            "centerline_tol_m, min_source_source_m, objective_rule_default, "
            "objective_flatness_eps_default"
        ) from exc
    if n_trials < 1:
        raise ValueError(
            f"{path} [position].n_trials_default must be >= 1, got {n_trials}"
        )
    if not math.isfinite(default_height) or default_height <= 0.0:
        raise ValueError(
            f"{path} [position].default_height_m must be finite and > 0, "
            f"got {default_height}"
        )
    if not math.isfinite(centerline_tol) or centerline_tol <= 0.0:
        raise ValueError(
            f"{path} [position].centerline_tol_m must be finite and > 0, "
            f"got {centerline_tol}"
        )
    if not math.isfinite(min_source_source) or min_source_source <= 0.0:
        raise ValueError(
            f"{path} [position].min_source_source_m must be finite and > 0, "
            f"got {min_source_source}"
        )
    if objective_rule != "lexicographic":
        raise ValueError(
            f"{path} [position].objective_rule_default must be 'lexicographic', "
            f"got {objective_rule!r}"
        )
    if not math.isfinite(objective_eps) or objective_eps < 0.0:
        raise ValueError(
            f"{path} [position].objective_flatness_eps_default must be finite "
            f"and >= 0, got {objective_eps}"
        )
    return (
        n_trials,
        default_height,
        centerline_tol,
        min_source_source,
        objective_rule,
        objective_eps,
    )


_FEM_LANE_CONFIG_PATH = config_path("fem_lane.toml")

# --- frequency cap (structural solver ceiling; DECOUPLED from F_SCHROEDER) -----
# Upper FEM mesh frequency (Hz). The differentiable dense-3D lane only resolves
# the low-frequency modal region up to this cap; bins above it are hard-zeroed
# before the crossover stitch (design §4.8, wired at the dispatch seam in T6).
# PROVISIONAL 250 Hz — the GO DOF/OOM envelope pins the real value on the
# extruded-quad system (design §1/§9).
#
# NOTE: the crossover-coverage guard ``FEM_FMAX_CAP_HZ >= f_schroeder*2**half_width``
# (design §4.8) is evaluated against the RUNTIME ``cfg.f_schroeder`` at the
# dispatch seam (T6), NOT as a static module-level relation here — keeping it
# static would let a caller leave ``f_schroeder`` at its default and silently
# reopen the coverage hole.
FEM_FMAX_CAP_HZ: float = 250.0

# M15-P5-3a linear-ceiling search bound. The cap is set by frozen-point
# interiority at the lowered edge, not by tetrahedron inversion.
MAX_CEILING_SLOPE: float = 0.5

# Maximum best-fit-plane residual (metres) accepted for the deformed ceiling
# node group. The fp32 production mesh measures about 1e-5 m for the exact
# linear map, leaving one decimal decade for platform-level SVD variation.
CEIL_PLANARITY_TOL: float = 1.0e-4

# General polygon-face coplanarity tolerance. It intentionally matches the 3a
# ceiling gate: both gmsh polygon walls and the linear ceiling are exact planes
# in geometry, while fp32/SVD evaluation needs a small platform allowance.
FACE_PLANARITY_TOL: float = CEIL_PLANARITY_TOL

# Verified-feasible interior wall-splay bound from T7-A. This runtime range is
# config-backed because the shape template and future optimizer decode must
# share one value. Do not redefine it in geometry/optimizer modules.
SPLAY_CAP: float = _load_splay_cap(_FEM_LANE_CONFIG_PATH)

# Minimum usable edge length in the chamfered-octagon shape search. The default
# 0.5 m keeps normal construction-detail chamfers reachable in a 5.4 x 4.2 m
# room while excluding near-diamond residual wall segments below usable width.
MIN_OCTAGON_EDGE_M: float = _load_min_octagon_edge_m(_FEM_LANE_CONFIG_PATH)

(
    POSITION_N_TRIALS_DEFAULT,
    POSITION_DEFAULT_HEIGHT_M,
    POSITION_CENTERLINE_TOL_M_DEFAULT,
    POSITION_MIN_SOURCE_SOURCE_M_DEFAULT,
    OBJECTIVE_RULE_DEFAULT,
    OBJECTIVE_FLATNESS_EPS_DEFAULT,
) = _load_position_defaults(_FEM_LANE_CONFIG_PATH)

# --- FEM-lane energy floor (design §6 #6) --------------------------------------
# Named, fp32-representable energy floor for the dense-3D FEM lane's pressure→energy
# conversion. Used with a DOUBLE-WHERE guard (``jnp.where(nonzero, |p|², 0)``), NOT
# an additive ``+1e-12`` (which sits ~28 orders above a quiet-region energy and kills
# the reverse-mode gradient there, design §6 #6). ≥1e-30 keeps it a true hardware
# guard above the fp32 subnormal floor (~1.18e-38), aligned with the stitch
# ``tiny=1e-30``. This is the magnitude added ONLY inside a ``log10`` for the dB
# observable; the energy itself uses exact-zero double-where (matching the proxy lane
# ``m4_pipeline.py`` ism_/fem_ double-where).
FEM3D_ENERGY_FLOOR_E: float = 1e-30

# --- positive-volume (non-inversion) barrier (design §4.4) ---------------------
# The barrier engages once the minimum signed tet volume drops below this
# fraction of the nominal (theta=0) minimum.
SV_FLOOR_FRAC: float = 0.5
# Barrier weight on the FLOOR-NORMALISED ``relu²`` shortfall ``(short/floor)²``.
# Sized so a mesh-inverting over-splay incurs a cost that dominates plausible
# objective gains — T0's provisional ``1.0`` (on an un-normalised volume² term,
# ~m⁶, far too weak) was only a leaf-liveness probe; T4 steepens. Seeded from the
# calibrated 2D-probe flip weight (``shape_opt_2d_probe.py`` ``w_flip``). Never
# ``clip|det|`` — that makes a dead-gradient region (design §6).
SV_BARRIER_W: float = 200.0

# --- volume-conservation anti-cheat (design §4.4) ------------------------------
# Penalises wall-splay (theta) that inflates/shrinks the room volume away from
# the nominal (theta=0) volume at FROZEN dims: ``(V(theta)/V0 - 1)²``. Keeps the
# shape search volume-preserving so the optimiser cannot trivially game modal
# density by changing volume via splay (the dims axis is orthogonal and handled
# separately). Seeded from the 2D-probe area-conservation weight (``w_area``).
VOL_CONSERVE_W: float = 50.0

# --- division guard for the floor/volume normalisation -------------------------
# A tiny GEOMETRIC (m³) epsilon guarding the barrier's ratio denominators against
# a degenerate nominal mesh. NOT an energy floor (those are ≥1e-30, design §6);
# this guards a normalisation against div-by-zero only, on quantities that are
# ``stop_gradient``-frozen constants (so a plain ``maximum`` guard is correct — no
# gradient flows through the denominator). fp32-representable.
SV_FLOOR_EPS: float = 1e-12

# --- shape-ranking reference absorption (M15-P5-2 shape-eval yardstick) ---------
# The uniform per-face absorption the shape-ranking lanes BOTH rank candidate
# shapes against — one yardstick: the FEM membrane-flatness scan (inc-2) builds a
# 6-wall ValidatedMaterials at this α, and the RFZ early-reflection metric (inc-3)
# builds a uniform N-face FacePatchMaterials at this α. It is NOT a shipped
# material — it is the PO-decision-A "processed-room low-frequency stand-in" so the
# seed Schroeder f_s stays under the FEM cap (probe: 5.4×4.2×2.8 → f_s≈135 ≤ 176.78;
# α=0.20 overshoots). ORDINAL: absolute RFZ gap magnitude scales with this α, so any
# threshold/weight tuned on gap size MUST declare it (calibration deferred to S4).
SHAPE_EVAL_REFERENCE_ALPHA: float = 0.30

# --- SHAPE-6 applied-room unassigned face material ----------------------------
# Default absorption for polygon applied-room faces with no explicit per-face
# assignment. This is separate from SHAPE_EVAL_REFERENCE_ALPHA, which remains the
# shape-ranking yardstick for ISM-specular geometric energy and RFZ.
SHAPE6_UNASSIGNED_ALPHA: float = 0.05

# --- M15-P5 F1: full-band (FEM⊕ISM) flatness window upper cutoff ----------------
# Upper Hz bound of the across-bin flatness window for the octagon full-band score
# (``score_octagon_shape(compute_fullband=True)``). ``inf`` = full production axis —
# the F1 window derisk probe (上一代那一棵樹的 F1 window derisk 探針紀錄，不在這個 repo 裡) showed
# 0 floored (-300 dB) bins at full-axis (the coherent geometric energy never nulls to
# the 1e-30 floor), so no cutoff is needed. S4-PROVISIONAL like
# SHAPE_EVAL_REFERENCE_ALPHA: this is a rank-affecting frequency-aggregation constant;
# any final value is a listening-calibration decision owned by S4, NOT to be tuned by
# which ranking it produces. (NOTE: the full-band rank is empirically == FEM-only —
# the modal band dominates the across-bin variance ~4×; the full-band score is the
# honest audible-band display, validated to not mis-rank.)
FULLBAND_FLATNESS_FMAX_HZ: float = float("inf")

# --- provisional mesh resolution (geometry-opt leaf-liveness fallback) ---------
# Used ONLY as a leaf-liveness probe fallback (a 2x2x2 mesh cannot represent a
# half-wavelength mode near the 250 Hz cap — it is NOT a physics mesh). The real
# production real3d resolution (6 P1-elements/wavelength up to FEM_FMAX_CAP_HZ) is
# computed by ``fem_kernel_3d.fem3d_mesh_resolution`` and wired as the real3d
# default in T6 (the forward-scoring lane in ``m4_pipeline.run_three_band_multisource``).
FEM_N_DEFAULT: int = 2
