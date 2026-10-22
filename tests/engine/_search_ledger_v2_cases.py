"""第二版帳本的合成輸入；方案、設定與登記簿都在暫存樹建立。"""
from __future__ import annotations

from pathlib import Path

from aosr.reporting.evaluation import purpose_settings
from aosr.reporting.scheme import Scheme
from aosr.search.sampler import Proposal
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._search_store_cases import settings_document


SPACE = {"front_distance": (0.0, 1.0), "spacing": (0.0, 1.0),
         "listening_distance": (0.0, 1.0), "room_width": (-2.0, 3.0)}
METERS = ("front_distance", "spacing", "listening_distance", "room_width_m", "derived_depth")


def store_in(tmp_path: Path, *, locked: bool = False) -> tuple[SearchStore, Path]:
    project = Scheme.model_validate({
        "schema_version": "aosr.scheme.v1", "scheme_id": "synthetic", "purpose": "listening",
        "source_model": "product_default",
        "scene": {"room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0}, "sound_speed_m_s": 343.0,
                  "density_kg_m3": 1.2, "impedance_pa_s_per_m_by_wall": {
                      wall: 1646.4 for wall in ("floor", "ceiling", "x0", "xL", "y0", "yL")}},
        "speakers": {"left": {"x": 1.0, "y": 1.3, "z": 1.2}, "right": {"x": 1.0, "y": 2.5, "z": 1.2}},
        "channel_group": {"channels": [{"role": "left", "speaker_id": "left"},
                                       {"role": "right", "speaker_id": "right"}],
                          "comparisons": [{"left_role": "left", "right_role": "right"}],
                          "feature_match_tolerance_hz": 10.0},
        "receiver_set": {"points": [{"receiver_id": "main", "position_m": [3.2, 1.9, 1.2],
                                     "role": "primary", "importance": 1.0},
                                    {"receiver_id": "near", "position_m": [3.3, 1.9, 1.2],
                                     "role": "surrounding", "importance": 1.0,
                                     "direction_relative_to_primary": "back"}]},
    })
    document = settings_document() | {"budget": 5, "convergence_run": 100}
    settings = SearchSettings.model_validate(document)
    if locked:
        settings = SearchSettings.model_validate(document | {"layout": settings.layout.model_dump() | {
            "seat_locked": True, "axis_offset_m": -0.1, "ear_height_m": 1.2, "listening_distance_m": None}})
    registry = tmp_path / "registry"
    registry.write_text('schema_version = 1\n[[purpose]]\nname = "listening"\n'
                        'setting = []\ntarget = []\nweight = []\nqualification = []\n')
    identity = SearchIdentity("phys-v1:" + "b" * 64, "calc-v1:" + "a" * 64,
                              purpose_settings(registry, "listening"))
    return SearchStore.create(tmp_path / "searches", project=project, settings=settings,
        identity=identity, versions={"python": "test", "optuna": "test", "numpy": "test"}), registry


def header_document() -> dict[str, object]:
    return {"ledger_version": "aosr.search_ledger.v2", "search_id": "synthetic",
            "settings_fingerprint": "settings", "project_fingerprint": "project",
            "sampler": {"seed": 19, "n_startup_trials": 2, "constant_liar": True},
            "batch_size": 3, "search_space": dict(SPACE), "meter_quantities": METERS,
            "physics_identity": "physical", "program_fingerprint": "program", "purpose_fingerprint": "purpose"}


def row_document(proposal: Proposal | None = None, *, batch: int = 0) -> dict[str, object]:
    proposal = proposal or Proposal(0, {"front_distance": 0.25, "spacing": 0.5,
                                        "listening_distance": 0.75, "room_width": -1.0})
    return {"batch_index": batch, "trial_number": proposal.trial_number,
            "unit_params_hex": {key: value.hex() for key, value in proposal.params.items()},
            "params_m": dict(zip(METERS, (0.5, 1.0, 2.0, 4.0, 6.0), strict=True)),
            "outcome": "illegal", "score": None, "reason": "wall_gap", "violation_m": 0.125,
            "seconds": 0.25, "result_file": None}
