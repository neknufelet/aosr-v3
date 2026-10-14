"""設定第 2 刀讀檔那 9 支模組的 case 清單與模組名單。

``blueprint/config_cut1_parts.py`` 放型別、值與操作的小工廠、參數還原及路徑正規化；
``blueprint/config_cut1_cases.py`` 留第 1 刀清單、合併的 MODULES 與存取函式，並再匯出零件與本表。
拆成三支是為了避開 style-guard（寫法警衛）第⑤條的單檔 1000 行上限，case 內容與順序不變。
本表也由上一代的 Python 載入，只能依賴標準庫與 blueprint，不載入新家引擎。
"""
from __future__ import annotations

from typing import Final

from blueprint.config_cut1_parts import (
    ModuleSpec,
    perimeter_default,
    perimeter_missing,
    perimeter_mutation,
    strs,
)

# 這一刀新增的 9 支模組（產生器靠這一份名單決定要不要載入新家那一支跑載入器；
# 第 1 刀那 11 支只凍結常數與純呼叫，沒有這一格）。
CUT2_MODULES: Final[tuple[str, ...]] = (
    "calibration",
    "fem_lane",
    "mat_continuation",
    "membrane_opt",
    "perceptual",
    "physics_constants",
    "scoring",
    "scoring_v2",
    "stereo_layout",
)


CUT2_CASES: Final[dict[str, ModuleSpec]] = {
    # ── 這一刀（票 #127 後半）：讀檔那 9 支 ────────────────────────────────────
    # 每一支都有一筆「真的設定檔」的 case（`op` 說「餵真的那個檔、不傳路徑就走預設」），
    # 其餘是突變（把真的設定檔改一格再餵）或「檔案不存在」那條分支。`constants` 是模組層
    # 常數的凍結值；載入器回傳的物件（含巢狀子模型）由那一筆 case 的值一起凍結。
    "cut2_calibration": {
        "file": "calibration",
        "constants": ["CalibrationBandConfig", "CalibrationConfig", "CalibrationRtConfig", "CalibrationSmoothingConfig"],
        "cases": [
            {
                "id": "cut2_calibration.load.real",
                "args": {},
                "op": perimeter_default("load_calibration"),
            },
            {
                "id": "cut2_calibration.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_calibration"),
            },
            {
                "id": "cut2_calibration.load.mut.band-f_hi-not-above-f_lo",
                "args": {},
                "op": perimeter_mutation("load_calibration", "f_hi_hz = 135.0", "f_hi_hz = 30.0"),
            },
        ],
    },
    "cut2_fem_lane": {
        "file": "fem_lane",
        "constants": [
            "CEIL_PLANARITY_TOL",
            "FACE_PLANARITY_TOL",
            "FEM3D_ENERGY_FLOOR_E",
            "FEM_FMAX_CAP_HZ",
            "FEM_N_DEFAULT",
            "FULLBAND_FLATNESS_FMAX_HZ",
            "MAX_CEILING_SLOPE",
            "MIN_OCTAGON_EDGE_M",
            "OBJECTIVE_FLATNESS_EPS_DEFAULT",
            "OBJECTIVE_RULE_DEFAULT",
            "POSITION_CENTERLINE_TOL_M_DEFAULT",
            "POSITION_DEFAULT_HEIGHT_M",
            "POSITION_MIN_SOURCE_SOURCE_M_DEFAULT",
            "POSITION_N_TRIALS_DEFAULT",
            "SHAPE6_UNASSIGNED_ALPHA",
            "SHAPE_EVAL_REFERENCE_ALPHA",
            "SPLAY_CAP",
            "SV_BARRIER_W",
            "SV_FLOOR_EPS",
            "SV_FLOOR_FRAC",
            "VOL_CONSERVE_W",
        ],
        # 這一格**刻意維持空的**（票 #162 的「或明說為什麼不補」；完整理由寫在
        # `tests/engine/test_config_cut2_fem_lane_wiring.py` 的檔頭）：`fem_lane` 沒有公開
        # 載入器可以餵一筆「突變過的 TOML」，而這一張表的一筆 case 就是「餵一個公開載入器」。
        # 「值是從 `fem_lane.toml` 算出來的」那條接線由那一支考卷直接餵突變檔 ＋ reload 守。
        "cases": [],
    },
    "cut2_mat_continuation": {
        "file": "mat_continuation",
        "constants": ["MatContinuationConfig"],
        "cases": [
            {
                "id": "cut2_mat_continuation.load.real",
                "args": {},
                "op": perimeter_default("load_mat_continuation"),
            },
            {
                "id": "cut2_mat_continuation.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_mat_continuation"),
            },
            {
                "id": "cut2_mat_continuation.load.mut.beta-cap-below-beta0",
                "args": {},
                "op": perimeter_mutation("load_mat_continuation", "beta_cap = 500.0", "beta_cap = 10.0"),
            },
        ],
    },
    "cut2_membrane_opt": {
        "file": "membrane_opt",
        "constants": ["MembraneOptConfig"],
        "cases": [
            {
                "id": "cut2_membrane_opt.load.real",
                "args": {},
                "op": perimeter_default("load_membrane_opt"),
            },
            {
                "id": "cut2_membrane_opt.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_membrane_opt"),
            },
            # `_validate_ranges` 的 6 條分支：4 組 lower<upper ＋ w_decay/w_rt60 必為 0。
            {
                "id": "cut2_membrane_opt.load.mut.air_gap-lower-equals-upper",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "air_gap_min_m = 0.005", "air_gap_min_m = 0.40"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.mass-lower-above-upper",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "mass_min_kg_m2 = 0.10", "mass_min_kg_m2 = 30.0"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.porous-lower-above-upper",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "porous_min_m = 0.005", "porous_min_m = 0.30"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.flow_resistivity-lower-above-upper",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "flow_resistivity_min = 1000.0", "flow_resistivity_min = 200000.0"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.w_decay-nonzero",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "w_decay = 0.0", "w_decay = 0.5"),
            },
            {
                "id": "cut2_membrane_opt.load.mut.w_rt60-nonzero",
                "args": {},
                "op": perimeter_mutation("load_membrane_opt", "w_rt60 = 0.0", "w_rt60 = 1.0"),
            },
        ],
    },
    "cut2_perceptual": {
        "file": "perceptual",
        "constants": ["MATERIAL_RFZ_PROFILE"],
        "cases": [
            {
                "id": "cut2_perceptual.load.real.default",
                "args": {},
                "op": perimeter_default("load_perceptual"),
            },
            {
                "id": "cut2_perceptual.load.real.profile-monitoring",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("monitoring")}),
            },
            {
                "id": "cut2_perceptual.load.real.profile-critical",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("critical")}),
            },
            {
                "id": "cut2_perceptual.load.real.profile-home_theater",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("home_theater")}),
            },
            {
                "id": "cut2_perceptual.load.real.profile-material_rfz",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("material_rfz")}),
            },
            {
                "id": "cut2_perceptual.load.real.unknown-profile",
                "args": {},
                "op": perimeter_default("load_perceptual", kwargs={"profile": strs("nope")}),
            },
            {
                "id": "cut2_perceptual.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_perceptual"),
            },
            {
                "id": "cut2_perceptual.load.material_rfz-profile.real",
                "args": {},
                "op": perimeter_default("load_material_rfz_profile"),
            },
            {
                "id": "cut2_perceptual.load.material_rfz-profile.missing-file",
                "args": {},
                "op": perimeter_missing("load_material_rfz_profile"),
            },
            {
                "id": "cut2_perceptual.load.mut.extra-top-level-key",
                "args": {},
                "op": perimeter_mutation("load_perceptual", 'default_profile = "monitoring"', 'default_profile = "monitoring"\nbogus = true'),
            },
            {
                "id": "cut2_perceptual.load.mut.rt60-target-ragged-lengths",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": "tol_lo_s = [0.05, 0.05, 0.05, 0.05, 0.10]",
                            "after": "tol_lo_s = [0.05, 0.05, 0.05, 0.05]",
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.modal-curve-ragged-lengths",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": "t60_s = [0.90, 0.50, 0.35, 0.25, 0.20]",
                            "after": "t60_s = [0.90, 0.50, 0.35, 0.25]",
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.default-profile-not-in-profiles",
                "args": {},
                "op": perimeter_mutation("load_perceptual", 'default_profile = "monitoring"', 'default_profile = "nope"'),
            },
            {
                "id": "cut2_perceptual.load.mut.profile-level-not-in-curves",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": '[profiles.monitoring]\nrfz = { threshold_db = -10.0, window_ms = 15.0, role = "hard_gate" }\nmodal_decay = { level = "music" }',
                            "after": '[profiles.monitoring]\nrfz = { threshold_db = -10.0, window_ms = 15.0, role = "hard_gate" }\nmodal_decay = { level = "nope" }',
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.rfz-both-windows",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": '[profiles.critical]\nrfz = { threshold_db = -25.0, window_mode = "dynamic", role = "soft_target" }',
                            "after": '[profiles.critical]\nrfz = { threshold_db = -25.0, window_mode = "dynamic", window_ms = 15.0, role = "soft_target" }',
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.rfz-no-window",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": '[profiles.monitoring]\nrfz = { threshold_db = -10.0, window_ms = 15.0, role = "hard_gate" }',
                            "after": '[profiles.monitoring]\nrfz = { threshold_db = -10.0, role = "hard_gate" }',
                        }
                    ],
                },
            },
            {
                "id": "cut2_perceptual.load.mut.early-reflection-extra-key",
                "args": {},
                "op": {
                    "fn": "load_perceptual",
                    "mutations": [
                        {
                            "before": "required_attenuation_db = 10.0\n# 10dB 檢查的聚合帶",
                            "after": "required_attenuation_db = 10.0\nbogus_key = 1.0\n# 10dB 檢查的聚合帶",
                        }
                    ],
                },
            },
        ],
    },
    "cut2_physics_constants": {
        "file": "physics_constants",
        "constants": ["PhysicsConstants"],
        "cases": [
            {
                "id": "cut2_physics_constants.load.real",
                "args": {},
                "op": perimeter_default("load_physics_constants"),
            },
            {
                "id": "cut2_physics_constants.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_physics_constants"),
            },
            {
                "id": "cut2_physics_constants.load.mut.sound-speed-zero",
                "args": {},
                "op": perimeter_mutation("load_physics_constants", "sound_speed_m_s = 343.0", "sound_speed_m_s = 0.0"),
            },
        ],
    },
    "cut2_scoring": {
        "file": "scoring",
        "constants": ["AsymmetryScoringConfig", "DecayScoringConfig", "ScoringConfig", "SmoothingScoringConfig", "SpatialScoringConfig"],
        "cases": [
            {
                "id": "cut2_scoring.load.real",
                "args": {},
                "op": perimeter_default("load_scoring"),
            },
            {
                "id": "cut2_scoring.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_scoring"),
            },
            {
                "id": "cut2_scoring.load.mut.hf-rolloff-fc-zero",
                "args": {},
                "op": perimeter_mutation("load_scoring", "hf_rolloff_fc_hz = 2000.0", "hf_rolloff_fc_hz = 0.0"),
            },
            {
                "id": "cut2_scoring.load.mut.peak-dip-bias-above-range",
                "args": {},
                "op": perimeter_mutation("load_scoring", "peak_dip_bias = 0.0", "peak_dip_bias = 9.0"),
            },
            {
                "id": "cut2_scoring.load.mut.frac-oct-zero",
                "args": {},
                "op": perimeter_mutation("load_scoring", "frac_oct = 3.0", "frac_oct = 0.0"),
            },
        ],
    },
    "cut2_scoring_v2": {
        "file": "scoring_v2",
        "constants": ["LOCAL_CROSS_7_ORDER", "SCORING_V2_SCHEMA_VERSION"],
        "cases": [
            {
                "id": "cut2_scoring_v2.load.real",
                "args": {},
                "op": perimeter_default("load_scoring_v2"),
            },
            {
                "id": "cut2_scoring_v2.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_scoring_v2"),
            },
            {
                "id": "cut2_scoring_v2.load.production-mode-refused",
                "args": {},
                "op": perimeter_default("load_scoring_v2", kwargs={"mode": strs("production")}),
            },
            {
                "id": "cut2_scoring_v2.load.mode-spelling-rejected",
                "args": {},
                "op": perimeter_default("load_scoring_v2", kwargs={"mode": strs("prod")}),
            },
            {
                "id": "cut2_scoring_v2.load.mut.frequency-ranges-do-not-join",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", "high_hz = 200.0", "high_hz = 201.0"),
            },
            {
                "id": "cut2_scoring_v2.load.mut.local-cross-order-swapped",
                "args": {},
                "op": {
                    "fn": "load_scoring_v2",
                    "mutations": [
                        {
                            "before": 'order = ["center", "front", "back", "left", "right", "up", "down"]',
                            "after": 'order = ["center", "back", "front", "left", "right", "up", "down"]',
                        }
                    ],
                },
            },
            {
                "id": "cut2_scoring_v2.load.mut.radius-zero",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", "radius_m = 0.10", "radius_m = 0.0"),
            },
            {
                "id": "cut2_scoring_v2.load.mut.frame-tolerance-zero",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", "frame_tolerance = 1e-6", "frame_tolerance = 0.0"),
            },
            {
                "id": "cut2_scoring_v2.load.mut.dip-min-width-not-one-sixth",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", "min_width_oct = 0.16666666666666666", "min_width_oct = 0.16"),
            },
            {
                "id": "cut2_scoring_v2.load.mut.rfz-band-status-ready",
                "args": {},
                "op": perimeter_mutation("load_scoring_v2", 'band_status = "unresolved"', 'band_status = "ready"'),
            },
        ],
    },
    "cut2_stereo_layout": {
        "file": "stereo_layout",
        # 這一支只有一個常數物件（19 個政策欄位是它的欄位，不是模組層常數）。那一筆
        # `load.real` 的值就是整個物件，逐欄比；19 個欄位名另外由
        # `tests/engine/test_config_cut2_stereo_layout.py` 具名列出並逐欄核對
        # （上一代有一條考卷逐欄比的是 `lib.physics.stereo_geometry` 那一排 `STEREO_*`，
        # 那一支不在這一刀裡）。
        "constants": ["StereoLayoutConfig"],
        "cases": [
            {
                "id": "cut2_stereo_layout.load.real",
                "args": {},
                "op": perimeter_default("load_stereo_layout"),
            },
            {
                "id": "cut2_stereo_layout.load.missing-file",
                "args": {},
                "op": perimeter_missing("load_stereo_layout"),
            },
            {
                "id": "cut2_stereo_layout.load.mut.unknown-top-level-table",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "distance_imbalance_max = 0.20", "distance_imbalance_max = 0.20\n\n[unexpected]\nvalue = 1.0"),
            },
            {
                "id": "cut2_stereo_layout.load.mut.unknown-policy-key",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "distance_imbalance_max = 0.20", "distance_imbalance_max = 0.20\nunknown_policy = 1.0"),
            },
            {
                "id": "cut2_stereo_layout.load.mut.non-numeric-value",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "distance_imbalance_max = 0.20", 'distance_imbalance_max = "0.20"'),
            },
            {
                "id": "cut2_stereo_layout.load.mut.non-finite-value",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "distance_imbalance_max = 0.20", "distance_imbalance_max = nan"),
            },
            {
                "id": "cut2_stereo_layout.load.mut.missing-policy-key",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "speaker_lateral_wall_margin_m = 0.5\n", ""),
            },
            {
                "id": "cut2_stereo_layout.load.mut.stereo-section-missing",
                "args": {},
                "op": perimeter_mutation("load_stereo_layout", "[stereo]", "[not_stereo]"),
            },
        ],
    },
}
