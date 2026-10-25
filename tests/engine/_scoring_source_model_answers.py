"""#505 第三刀控制組的答案：主對話在改動前的主線（31573f47f94b505bf0b475e9521902cae392d64b，src 沒有未提交改動）上跑
``_scoring_source_model_control.py`` 錄下的，不是被測函式現算的；施工的人不准改這一支。

產生腳本是工作區 505-directivity-cut3-scoring 底下 scripts 目錄的 record_answers.py（repo 外）。
浮點記成 float.hex；SECTIONS 每段一個非浮點雜湊，加上每種浮點路徑的（個數, 加權絕對值和, 加權帶號和）。

#518（聆聽區左右各算）重錄了照設計會變的那幾格，同樣在改動前的主線（bb20de068d379258db084f6d4c53a632342b1e6a）上跑、不是被測函式現算的：
排名表表頭（拿掉聆聽區那一格）與每一列（拿掉 total_cost）重錄；兩個候選與排名表的聆聽區段整段不比、從這裡拿掉，
改成聆聽區左右兩支各自一段（payload、出身、旗標，由改動前的單支評估錄）；
HEADLINE 的聆聽區那一格與 total_cost 換成由改動前左右兩支單錄、手算平均的新值（分項與原始量加角色前綴）。
產生腳本是工作區 518-listening-channels 底下 evidence 目錄的 record_control_answers_518.py（repo 外），跑兩次逐位相同。

#577（刪掉沒有評估器讀的驗證名單）讓品質登記簿整份的指紋改變；控制組改把那個指紋換成佔位字
（``_scoring_source_model_control.py::version_folding``），在改動前的主線（6e7c0136651368f55937e7374ca93ff1225b932e）
上重錄受影響那 17 段的非浮點雜湊；浮點那一半、SUMMARY、HEADLINE 核對過跟原答案相同、沒有重錄。
產生腳本是工作區 577-drop-verification-list 底下 evidence 目錄的 record_control_answers_577.py（repo 外），跑兩次逐位相同。

#785 第七步：在 HEAD（e7ba8d1eff313673bce7fd39979f951d6241482d）與 verify.yml 三格設定，
先掛 STAND_INS；舊答案同跑逐位核過（24 段 exact 與分組個數／兩個加權和、HEADLINE、SUMMARY），
再從同一行程同一份結果分出骨架與逐格 hex；不再存分組和。HEADLINE、SUMMARY 沒重錄。
錄答案腳本在暫存根（tmp）下 785-step7-cellproof 目錄的 record.py（repo 外），逐格資料拆到 _scoring_source_model_cells_answers.py。
此題守骨架逐字與機器相關浮點③級，不證明跨處理器逐位；錄前工作樹只有考卷裁判變更、src 未改。
接手在延遲例外縮為五條完整路徑後，用暫存根（tmp）下 785-step7r-recheck 目錄的 record.py 重核：
同一行程先過舊 SECTIONS、HEADLINE、SUMMARY 逐位門檻，再由同份結果重建並核對現有骨架雜湊與逐格 hex 全部相同；答案值未改。

#668（音色起伏計分範圍下限 40→20 Hz，老闆 2026-10-20「音色改 傾斜不改」）：在主線 f2dfffd7 加本票前兩顆提交（登記簿改值、網頁標示）上，
先只把品質登記簿換回改動前主線那份同跑，本題與另 7 題舊答案全過（改動的其餘部分不動任何一格）；
再用新登記簿、掛 STAND_INS，同一行程取出會變的格子重錄：SECTIONS 骨架雜湊 8 段、逐格 hex 12 段、HEADLINE 34 格；SUMMARY 沒變、沒重錄。
這次 HEADLINE 取的是產品在新登記簿下算出的值，不是手算；依據是上面那一道同跑核對。
錄答案腳本在工作區 668-ripple-from-20 底下 record 目錄的 record_source_model.py（repo 外）。
"""

from tests.engine._scoring_source_model_cells_answers import FLOATS

SUMMARY: dict[str, object] = {
    "evaluations": {
        "wall-1/timbre_balance": ["measured", ["unvalidated", "feature_boundary_incomplete", "feature_narrower_than_axis", "baseline_settings"], []],
        "wall-1/listening_area_stability": ["measured", ["unvalidated", "feature_boundary_incomplete", "feature_narrower_than_axis", "baseline_settings"], []],
        "wall-1/reflections_and_echo": ["measured", ["unvalidated", "no_directivity", "baseline_settings", "window_only_delay_screen", "geometry_material_conservative_screen"], []],
        "wall-1/channel_matching": ["measured", ["unvalidated", "feature_boundary_incomplete", "feature_narrower_than_axis", "baseline_settings"], []],
        "wall-1/reverberation": ["measured", ["unvalidated"], []],
        "wall-2/timbre_balance": ["measured", ["unvalidated", "feature_boundary_incomplete", "feature_narrower_than_axis", "baseline_settings"], []],
        "wall-2/listening_area_stability": ["measured", ["unvalidated", "feature_boundary_incomplete", "feature_narrower_than_axis", "baseline_settings"], []],
        "wall-2/reflections_and_echo": ["measured", ["unvalidated", "no_directivity", "baseline_settings", "window_only_delay_screen", "geometry_material_conservative_screen"], []],
        "wall-2/channel_matching": ["measured", ["unvalidated", "feature_boundary_incomplete", "feature_narrower_than_axis", "baseline_settings"], []],
        "wall-2/reverberation": ["measured", ["unvalidated"], []],
    },
    "rows": {
        "wall-1": [1, ["unvalidated", "no_directivity", "feature_boundary_incomplete", "feature_narrower_than_axis", "baseline_settings", "listening_area_peer_group_missing", "window_only_delay_screen", "geometry_material_conservative_screen", "reflection_lateral_above_threshold", "reflection_vertical_above_threshold"]],
        "wall-2": [2, ["unvalidated", "no_directivity", "feature_boundary_incomplete", "feature_narrower_than_axis", "baseline_settings", "listening_area_peer_group_missing", "window_only_delay_screen", "geometry_material_conservative_screen", "reflection_front_above_threshold", "reflection_lateral_above_threshold", "reflection_rear_above_threshold", "reflection_vertical_above_threshold"]],
    },
    "other_blocks": [[], [], []],
}

HEADLINE: dict[str, object] = {
  "wall-1": {
    "total_cost": "0x1.0ff7619588f5ep+2",
    "categories": {
      "channel_matching": {
        "category_cost": "0x1.37fddee455ba5p-4",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.37fddee455ba5p-4",
        "components": {
          "broadband_level_difference.weighted_mean": "0x1.d3fcce5680978p-3",
          "broadband_level_difference.weighted_mean.left-right": "0x1.d3fcce5680978p-3",
          "broadband_level_difference.worst": "0x0.0p+0",
          "broadband_level_difference.worst.left-right.front": "0x0.0p+0",
          "ripple_rms_difference.weighted_mean": "0x0.0p+0",
          "ripple_rms_difference.weighted_mean.left-right": "0x0.0p+0",
          "ripple_rms_difference.worst": "0x0.0p+0",
          "ripple_rms_difference.worst.left-right.front": "0x0.0p+0",
          "tilt_difference.weighted_mean": "0x0.0p+0",
          "tilt_difference.weighted_mean.left-right": "0x0.0p+0",
          "tilt_difference.worst": "0x0.0p+0",
          "tilt_difference.worst.left-right.main": "0x0.0p+0"
        },
        "raw_quantities": {
          "assessed_point_count": "0x1.0000000000000p+1",
          "comparison.0.tilt_difference.weighted_mean_absolute_difference": "0x1.c7cfe1510c280p-3",
          "comparison.0.tilt_difference.worst_absolute_difference": "0x1.12524a0b68200p-2",
          "comparison.0.ripple_rms_difference.weighted_mean_absolute_difference": "0x1.1a0f9dd259b80p-1",
          "comparison.0.ripple_rms_difference.worst_absolute_difference": "0x1.fbe02e6554450p-1",
          "comparison.0.broadband_level_difference.weighted_mean_absolute_difference": "0x1.e9fe672b404bcp+0",
          "comparison.0.broadband_level_difference.worst_absolute_difference": "0x1.3dd366bce433cp+1",
          "comparison.0.direct_time_difference.weighted_mean_absolute_difference": "0x1.4e12af1cbc9b0p+1",
          "comparison.0.direct_time_difference.worst_absolute_difference": "0x1.a00deb4cfe7d0p+1"
        }
      },
      "listening_area_stability": {
        "category_cost": "0x1.2fc79c9ca48f8p-3",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.2fc79c9ca48f8p-3",
        "components": {
          "left.tilt_weighted_mean_deviation": "0x0.0p+0",
          "left.tilt_weighted_mean_deviation.primary_to_surrounding": "0x0.0p+0",
          "left.tilt_worst_deviation": "0x0.0p+0",
          "left.tilt_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "left.ripple_rms_weighted_mean_deviation": "0x1.ae85aa93eaaabp-11",
          "left.ripple_rms_weighted_mean_deviation.primary_to_surrounding": "0x1.ae85aa93eaaabp-11",
          "left.ripple_rms_worst_deviation": "0x0.0p+0",
          "left.ripple_rms_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "left.overall_level_weighted_mean_deviation": "0x1.f6d3aa9ac4464p-3",
          "left.overall_level_weighted_mean_deviation.primary_to_surrounding": "0x1.f6d3aa9ac4464p-3",
          "left.overall_level_worst_deviation": "0x0.0p+0",
          "left.overall_level_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.tilt_weighted_mean_deviation": "0x1.9c3423cfc3b80p-5",
          "right.tilt_weighted_mean_deviation.primary_to_surrounding": "0x1.9c3423cfc3b80p-5",
          "right.tilt_worst_deviation": "0x0.0p+0",
          "right.tilt_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.ripple_rms_weighted_mean_deviation": "0x0.0p+0",
          "right.ripple_rms_weighted_mean_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.ripple_rms_worst_deviation": "0x0.0p+0",
          "right.ripple_rms_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.overall_level_weighted_mean_deviation": "0x0.0p+0",
          "right.overall_level_weighted_mean_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.overall_level_worst_deviation": "0x0.0p+0",
          "right.overall_level_worst_deviation.primary_to_surrounding": "0x0.0p+0"
        },
        "raw_quantities": {
          "left.tilt.primary_to_surrounding.weighted_mean_deviation": "0x1.3e7a4f45c98c0p-3",
          "left.tilt.primary_to_surrounding.worst_deviation": "0x1.3e7a4f45c98c0p-3",
          "left.ripple_rms.primary_to_surrounding.weighted_mean_deviation": "0x1.00a1721ff7780p+0",
          "left.ripple_rms.primary_to_surrounding.worst_deviation": "0x1.00a1721ff7780p+0",
          "left.overall_level.primary_to_surrounding.weighted_mean_deviation": "0x1.4d6b87fb8733cp+1",
          "left.overall_level.primary_to_surrounding.worst_deviation": "0x1.4d6b87fb8733cp+1",
          "left.peak_dip_consistency.primary_to_surrounding.weighted_mean_deviation": "0x1.b800000000000p+5",
          "left.peak_dip_consistency.primary_to_surrounding.worst_deviation": "0x1.b800000000000p+5",
          "right.tilt.primary_to_surrounding.weighted_mean_deviation": "0x1.33868479f8770p-1",
          "right.tilt.primary_to_surrounding.worst_deviation": "0x1.33868479f8770p-1",
          "right.ripple_rms.primary_to_surrounding.weighted_mean_deviation": "0x1.6647f11ed2370p-1",
          "right.ripple_rms.primary_to_surrounding.worst_deviation": "0x1.6647f11ed2370p-1",
          "right.overall_level.primary_to_surrounding.weighted_mean_deviation": "0x1.77864359fe300p+0",
          "right.overall_level.primary_to_surrounding.worst_deviation": "0x1.77864359fe300p+0",
          "right.peak_dip_consistency.primary_to_surrounding.weighted_mean_deviation": "0x1.c800000000000p+5",
          "right.peak_dip_consistency.primary_to_surrounding.worst_deviation": "0x1.c800000000000p+5"
        }
      },
      "reflections_and_echo": {
        "category_cost": "0x1.368508b6f8924p-3",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.368508b6f8924p-3",
        "components": {
          "left.front": "0x0.0p+0",
          "left.lateral": "0x1.8d4fd110a3773p-3",
          "left.rear": "0x0.0p+0",
          "left.vertical": "0x1.cb45d49968fdap-3",
          "right.front": "0x0.0p+0",
          "right.lateral": "0x1.0cf1c5a50614ep-1",
          "right.rear": "0x0.0p+0",
          "right.vertical": "0x1.13e5c4bccfe4dp-2"
        },
        "raw_quantities": {
          "round_trip_delay_x0_xL": "0x1.2cee3c9aa5180p-5",
          "round_trip_delay_y0_yL": "0x1.87affd9c95de4p-6",
          "round_trip_delay_floor_ceiling": "0x1.150c17494ac8fp-6"
        }
      },
      "reverberation": {
        "category_cost": "0x1.fc60ad5e826c0p-7",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.fc60ad5e826c0p-7",
        "components": {
          "adjacent_t20_jump": "0x0.0p+0",
          "adjacent_t20_jump.1000-2000Hz": "0x0.0p+0",
          "adjacent_t20_jump.125-250Hz": "0x0.0p+0",
          "adjacent_t20_jump.2000-4000Hz": "0x0.0p+0",
          "adjacent_t20_jump.250-500Hz": "0x0.0p+0",
          "adjacent_t20_jump.4000-8000Hz": "0x0.0p+0",
          "adjacent_t20_jump.500-1000Hz": "0x0.0p+0",
          "t20_target_interval": "0x1.fc60ad5e826c0p-6",
          "t20_target_interval.1000Hz": "0x0.0p+0",
          "t20_target_interval.125Hz": "0x0.0p+0",
          "t20_target_interval.2000Hz": "0x0.0p+0",
          "t20_target_interval.250Hz": "0x0.0p+0",
          "t20_target_interval.4000Hz": "0x0.0p+0",
          "t20_target_interval.500Hz": "0x0.0p+0",
          "t20_target_interval.8000Hz": "0x1.bcd497b2b21e8p-3"
        },
        "raw_quantities": {
          "band_count": "0x1.c000000000000p+2",
          "125Hz.t20_s": "0x1.39bcd497b2b21p-2",
          "125Hz.t30_s": "0x1.6da9fd47d79e1p-2",
          "125Hz.fitting_difference": "0x1.2a5ecc60df50dp+0",
          "250Hz.t20_s": "0x1.404675fc32311p-2",
          "250Hz.t30_s": "0x1.74ed942948d5cp-2",
          "250Hz.fitting_difference": "0x1.2a16070150eafp+0",
          "500Hz.t20_s": "0x1.4d59b8c5312eep-2",
          "500Hz.t30_s": "0x1.8374c1ec2b453p-2",
          "500Hz.fitting_difference": "0x1.298d0c65b1d09p+0",
          "1000Hz.t20_s": "0x1.67803e572f2a8p-2",
          "1000Hz.t30_s": "0x1.a0831d71f023fp-2",
          "1000Hz.fitting_difference": "0x1.2898fb86db088p+0",
          "2000Hz.t20_s": "0x1.9bcd497b2b21fp-2",
          "2000Hz.t30_s": "0x1.da9fd47d79e17p-2",
          "2000Hz.fitting_difference": "0x1.270dd7f52f163p+0",
          "4000Hz.t20_s": "0x1.0233afe191885p-1",
          "4000Hz.t30_s": "0x1.276ca14a46ae3p-1",
          "4000Hz.fitting_difference": "0x1.24e7ae88f8ff6p+0",
          "8000Hz.t20_s": "0x1.6acdc62989770p-1",
          "8000Hz.t30_s": "0x1.9ba60f615a294p-1",
          "8000Hz.fitting_difference": "0x1.227737eaf3c44p+0",
          "125-250Hz.signed_log_ratio": "0x1.e77c33b6c74c4p-6",
          "250-500Hz.signed_log_ratio": "0x1.d8e7654d708e0p-5",
          "500-1000Hz.signed_log_ratio": "0x1.be487087e8acep-4",
          "1000-2000Hz.signed_log_ratio": "0x1.9150ce69e023ep-3",
          "2000-4000Hz.signed_log_ratio": "0x1.4e62e588c19d6p-2",
          "4000-8000Hz.signed_log_ratio": "0x1.f6776d80e84a9p-2"
        }
      },
      "timbre_balance": {
        "category_cost": "0x1.edcda93156e96p+1",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.edcda93156e96p+1",
        "components": {
          "left.tilt": "0x1.e588e2a90c222p+0",
          "left.residual_rms": "0x1.f95629fd260f6p+0",
          "left.target_deviation": "0x1.149e4b1fbe0f7p+1",
          "left.peaks_dips": "0x1.f3278e4c5b986p+1",
          "right.tilt": "0x1.c33e99679f1e2p+0",
          "right.residual_rms": "0x1.0a8c7f5bc52aep+1",
          "right.target_deviation": "0x1.0778af618315ep+1",
          "right.peaks_dips": "0x1.7f5a5f30bfefbp+2"
        },
        "raw_quantities": {
          "left.tilt_db_per_octave": "0x1.32c4715486111p+2",
          "left.target_deviation_rms_db": "0x1.9eed70af9d173p+3",
          "left.residual_rms_db": "0x1.f95629fd260f6p+2",
          "left.feature_count": "0x1.5800000000000p+5",
          "right.tilt_db_per_octave": "0x1.219f4cb3cf8f1p+2",
          "right.target_deviation_rms_db": "0x1.8b35071244a0dp+3",
          "right.residual_rms_db": "0x1.0a8c7f5bc52aep+3",
          "right.feature_count": "0x1.6800000000000p+5"
        }
      }
    }
  },
  "wall-2": {
    "total_cost": "0x1.3cf1acb358955p+2",
    "categories": {
      "channel_matching": {
        "category_cost": "0x1.a7fdb54285c00p-5",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.a7fdb54285c00p-5",
        "components": {
          "broadband_level_difference.weighted_mean": "0x1.3dfe47f1e4500p-3",
          "broadband_level_difference.weighted_mean.left-right": "0x1.3dfe47f1e4500p-3",
          "broadband_level_difference.worst": "0x0.0p+0",
          "broadband_level_difference.worst.left-right.front": "0x0.0p+0",
          "ripple_rms_difference.weighted_mean": "0x0.0p+0",
          "ripple_rms_difference.weighted_mean.left-right": "0x0.0p+0",
          "ripple_rms_difference.worst": "0x0.0p+0",
          "ripple_rms_difference.worst.left-right.front": "0x0.0p+0",
          "tilt_difference.weighted_mean": "0x0.0p+0",
          "tilt_difference.weighted_mean.left-right": "0x0.0p+0",
          "tilt_difference.worst": "0x0.0p+0",
          "tilt_difference.worst.left-right.main": "0x0.0p+0"
        },
        "raw_quantities": {
          "assessed_point_count": "0x1.0000000000000p+1",
          "comparison.0.tilt_difference.weighted_mean_absolute_difference": "0x1.11023e5376ca0p-2",
          "comparison.0.tilt_difference.worst_absolute_difference": "0x1.65d3dfad5d5f0p-2",
          "comparison.0.ripple_rms_difference.weighted_mean_absolute_difference": "0x1.a1ab5b1c79b90p-2",
          "comparison.0.ripple_rms_difference.worst_absolute_difference": "0x1.89d1735a94980p-1",
          "comparison.0.broadband_level_difference.weighted_mean_absolute_difference": "0x1.9eff23f8f2280p+0",
          "comparison.0.broadband_level_difference.worst_absolute_difference": "0x1.0808b2ab9fc1cp+1",
          "comparison.0.direct_time_difference.weighted_mean_absolute_difference": "0x1.4e12af1cbc9b0p+1",
          "comparison.0.direct_time_difference.worst_absolute_difference": "0x1.a00deb4cfe7d0p+1"
        }
      },
      "listening_area_stability": {
        "category_cost": "0x1.26831d8ba9244p-3",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.26831d8ba9244p-3",
        "components": {
          "left.tilt_weighted_mean_deviation": "0x0.0p+0",
          "left.tilt_weighted_mean_deviation.primary_to_surrounding": "0x0.0p+0",
          "left.tilt_worst_deviation": "0x0.0p+0",
          "left.tilt_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "left.ripple_rms_weighted_mean_deviation": "0x1.1864871a29eabp-7",
          "left.ripple_rms_weighted_mean_deviation.primary_to_surrounding": "0x1.1864871a29eabp-7",
          "left.ripple_rms_worst_deviation": "0x0.0p+0",
          "left.ripple_rms_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "left.overall_level_weighted_mean_deviation": "0x1.6ea6e9e44db72p-3",
          "left.overall_level_weighted_mean_deviation.primary_to_surrounding": "0x1.6ea6e9e44db72p-3",
          "left.overall_level_worst_deviation": "0x0.0p+0",
          "left.overall_level_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.tilt_weighted_mean_deviation": "0x0.0p+0",
          "right.tilt_weighted_mean_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.tilt_worst_deviation": "0x0.0p+0",
          "right.tilt_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.ripple_rms_weighted_mean_deviation": "0x1.99b21182c3e55p-4",
          "right.ripple_rms_weighted_mean_deviation.primary_to_surrounding": "0x1.99b21182c3e55p-4",
          "right.ripple_rms_worst_deviation": "0x0.0p+0",
          "right.ripple_rms_worst_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.overall_level_weighted_mean_deviation": "0x0.0p+0",
          "right.overall_level_weighted_mean_deviation.primary_to_surrounding": "0x0.0p+0",
          "right.overall_level_worst_deviation": "0x0.0p+0",
          "right.overall_level_worst_deviation.primary_to_surrounding": "0x0.0p+0"
        },
        "raw_quantities": {
          "left.tilt.primary_to_surrounding.weighted_mean_deviation": "0x1.559acf591e9a0p-3",
          "left.tilt.primary_to_surrounding.worst_deviation": "0x1.559acf591e9a0p-3",
          "left.ripple_rms.primary_to_surrounding.weighted_mean_deviation": "0x1.06925b2a9cfb8p+0",
          "left.ripple_rms.primary_to_surrounding.worst_deviation": "0x1.06925b2a9cfb8p+0",
          "left.overall_level.primary_to_surrounding.weighted_mean_deviation": "0x1.271ef1c835db8p+1",
          "left.overall_level.primary_to_surrounding.worst_deviation": "0x1.271ef1c835db8p+1",
          "left.peak_dip_consistency.primary_to_surrounding.weighted_mean_deviation": "0x1.e800000000000p+5",
          "left.peak_dip_consistency.primary_to_surrounding.worst_deviation": "0x1.e800000000000p+5",
          "right.tilt.primary_to_surrounding.weighted_mean_deviation": "0x1.773714fa5e470p-2",
          "right.tilt.primary_to_surrounding.worst_deviation": "0x1.773714fa5e470p-2",
          "right.ripple_rms.primary_to_surrounding.weighted_mean_deviation": "0x1.4cd1634884bb0p+0",
          "right.ripple_rms.primary_to_surrounding.worst_deviation": "0x1.4cd1634884bb0p+0",
          "right.overall_level.primary_to_surrounding.weighted_mean_deviation": "0x1.6c1960d3d1000p+0",
          "right.overall_level.primary_to_surrounding.worst_deviation": "0x1.6c1960d3d1000p+0",
          "right.peak_dip_consistency.primary_to_surrounding.weighted_mean_deviation": "0x1.0400000000000p+6",
          "right.peak_dip_consistency.primary_to_surrounding.worst_deviation": "0x1.0400000000000p+6"
        }
      },
      "reflections_and_echo": {
        "category_cost": "0x1.3c0a9bdffdd4bp-2",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.3c0a9bdffdd4bp-2",
        "components": {
          "left.front": "0x1.e5f256961c620p-3",
          "left.lateral": "0x1.1de853644dacdp-2",
          "left.rear": "0x1.14dd539af5f0ep-4",
          "left.vertical": "0x1.903398ca5e28fp-2",
          "right.front": "0x1.e00c84e2ac2c6p-3",
          "right.lateral": "0x1.4e69aee101428p-1",
          "right.rear": "0x1.37b576584aeecp-3",
          "right.vertical": "0x1.d154173ff90fep-2"
        },
        "raw_quantities": {
          "round_trip_delay_x0_xL": "0x1.2cee3c9aa5180p-5",
          "round_trip_delay_y0_yL": "0x1.87affd9c95de4p-6",
          "round_trip_delay_floor_ceiling": "0x1.150c17494ac8fp-6"
        }
      },
      "reverberation": {
        "category_cost": "0x1.fc60ad5e826c0p-7",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.fc60ad5e826c0p-7",
        "components": {
          "adjacent_t20_jump": "0x0.0p+0",
          "adjacent_t20_jump.1000-2000Hz": "0x0.0p+0",
          "adjacent_t20_jump.125-250Hz": "0x0.0p+0",
          "adjacent_t20_jump.2000-4000Hz": "0x0.0p+0",
          "adjacent_t20_jump.250-500Hz": "0x0.0p+0",
          "adjacent_t20_jump.4000-8000Hz": "0x0.0p+0",
          "adjacent_t20_jump.500-1000Hz": "0x0.0p+0",
          "t20_target_interval": "0x1.fc60ad5e826c0p-6",
          "t20_target_interval.1000Hz": "0x0.0p+0",
          "t20_target_interval.125Hz": "0x0.0p+0",
          "t20_target_interval.2000Hz": "0x0.0p+0",
          "t20_target_interval.250Hz": "0x0.0p+0",
          "t20_target_interval.4000Hz": "0x0.0p+0",
          "t20_target_interval.500Hz": "0x0.0p+0",
          "t20_target_interval.8000Hz": "0x1.bcd497b2b21e8p-3"
        },
        "raw_quantities": {
          "band_count": "0x1.c000000000000p+2",
          "125Hz.t20_s": "0x1.39bcd497b2b21p-2",
          "125Hz.t30_s": "0x1.6da9fd47d79e1p-2",
          "125Hz.fitting_difference": "0x1.2a5ecc60df50dp+0",
          "250Hz.t20_s": "0x1.404675fc32311p-2",
          "250Hz.t30_s": "0x1.74ed942948d5cp-2",
          "250Hz.fitting_difference": "0x1.2a16070150eafp+0",
          "500Hz.t20_s": "0x1.4d59b8c5312eep-2",
          "500Hz.t30_s": "0x1.8374c1ec2b453p-2",
          "500Hz.fitting_difference": "0x1.298d0c65b1d09p+0",
          "1000Hz.t20_s": "0x1.67803e572f2a8p-2",
          "1000Hz.t30_s": "0x1.a0831d71f023fp-2",
          "1000Hz.fitting_difference": "0x1.2898fb86db088p+0",
          "2000Hz.t20_s": "0x1.9bcd497b2b21fp-2",
          "2000Hz.t30_s": "0x1.da9fd47d79e17p-2",
          "2000Hz.fitting_difference": "0x1.270dd7f52f163p+0",
          "4000Hz.t20_s": "0x1.0233afe191885p-1",
          "4000Hz.t30_s": "0x1.276ca14a46ae3p-1",
          "4000Hz.fitting_difference": "0x1.24e7ae88f8ff6p+0",
          "8000Hz.t20_s": "0x1.6acdc62989770p-1",
          "8000Hz.t30_s": "0x1.9ba60f615a294p-1",
          "8000Hz.fitting_difference": "0x1.227737eaf3c44p+0",
          "125-250Hz.signed_log_ratio": "0x1.e77c33b6c74c4p-6",
          "250-500Hz.signed_log_ratio": "0x1.d8e7654d708e0p-5",
          "500-1000Hz.signed_log_ratio": "0x1.be487087e8acep-4",
          "1000-2000Hz.signed_log_ratio": "0x1.9150ce69e023ep-3",
          "2000-4000Hz.signed_log_ratio": "0x1.4e62e588c19d6p-2",
          "4000-8000Hz.signed_log_ratio": "0x1.f6776d80e84a9p-2"
        }
      },
      "timbre_balance": {
        "category_cost": "0x1.1baebe47c7223p+2",
        "category_weight": "0x1.0000000000000p+0",
        "weighted_cost": "0x1.1baebe47c7223p+2",
        "components": {
          "left.tilt": "0x1.1bb1b4e680b53p+1",
          "left.residual_rms": "0x1.299d58b180209p+1",
          "left.target_deviation": "0x1.207de6a2a26efp+1",
          "left.peaks_dips": "0x1.832fa24cd97cap+2",
          "right.tilt": "0x1.055476ebaadf4p+1",
          "right.residual_rms": "0x1.2417749b70d3bp+1",
          "right.target_deviation": "0x1.163bee67a9127p+1",
          "right.peaks_dips": "0x1.12921a4f651f4p+3"
        },
        "raw_quantities": {
          "left.tilt_db_per_octave": "0x1.5bb1b4e680b53p+2",
          "left.target_deviation_rms_db": "0x1.b0bcd9f3f3a67p+3",
          "left.residual_rms_db": "0x1.299d58b180209p+3",
          "left.feature_count": "0x1.9000000000000p+5",
          "right.tilt_db_per_octave": "0x1.455476ebaadf4p+2",
          "right.target_deviation_rms_db": "0x1.a159e59b7d9bbp+3",
          "right.residual_rms_db": "0x1.2417749b70d3bp+3",
          "right.feature_count": "0x1.9800000000000p+5"
        }
      }
    }
  }
}

SECTIONS: dict[str, dict[str, object]] = {
    'wall-1/timbre_balance': {"exact": '3bf420ed77118be27a759b460f70997c3070280b345f5891177938d15ccbf7f2', "floats": FLOATS['wall-1/timbre_balance']},
    'wall-1/reflections_and_echo': {"exact": 'd574d38ea401a4e486c422ba5884deb45de418ec0d48b4bb2b606f64aa38db01', "floats": FLOATS['wall-1/reflections_and_echo']},
    'wall-1/channel_matching': {"exact": '67762d11196964d7595702aa21a4f4caee1907ff3879015e39d0c0e8697defd9', "floats": FLOATS['wall-1/channel_matching']},
    'wall-1/reverberation': {"exact": '46472247aa7493bc50bed7a4005b7c8d5ab537b1f8813c62604ba2f936b9b4e7', "floats": FLOATS['wall-1/reverberation']},
    'wall-2/timbre_balance': {"exact": 'b7390e53f25c0b94affb559cc2e59994736ca7b36a22c0df38971a7c2f666bad', "floats": FLOATS['wall-2/timbre_balance']},
    'wall-2/reflections_and_echo': {"exact": 'a002b8886f9d120de63c25dc85ffb7522df5aa47682a1d4358ee7804a3a3f61f', "floats": FLOATS['wall-2/reflections_and_echo']},
    'wall-2/channel_matching': {"exact": 'b9703038ba4b889c37ed5ec6f94b34aef3e771fb232421d3da456ba865d18f16', "floats": FLOATS['wall-2/channel_matching']},
    'wall-2/reverberation': {"exact": '492b306d0235802e722d9baddf91f79dfbf2ca1d83dd21d1dc59982ac2a7448f', "floats": FLOATS['wall-2/reverberation']},
    'wall-1/listening_area_stability/left': {"exact": 'faf3ebc0fc1a056efb93a0979a6135ca5babd9f249f996229bed5fbde0a7eb17', "floats": FLOATS['wall-1/listening_area_stability/left']},
    'wall-1/listening_area_stability/right': {"exact": '57f2f6c4b85e85d15b48eea0e8fbd76354ae6ebd4896f916c56691ddae0dcaea', "floats": FLOATS['wall-1/listening_area_stability/right']},
    'wall-2/listening_area_stability/left': {"exact": 'f43647d0cd3109b3c439d062dcbfe9f833fb477c83d2c8d6cc81379c4bc17541', "floats": FLOATS['wall-2/listening_area_stability/left']},
    'wall-2/listening_area_stability/right': {"exact": 'e99be712bd60642e180855a2bf36aa27edb65b25deb485a9188cdd6a5b217716', "floats": FLOATS['wall-2/listening_area_stability/right']},
    'ranking/header': {"exact": 'a8f729217dd9b385cbb4a9c5dd8b9596e02a5c708c4a60baa4048e1082ee7a17', "floats": FLOATS['ranking/header']},
    'ranking/wall-1/channel_matching': {"exact": '1eade5f8275e43b72eccd4e9c13bf6944093a77ee9663362730bb3f323cc341b', "floats": FLOATS['ranking/wall-1/channel_matching']},
    'ranking/wall-1/reflections_and_echo': {"exact": '61c2689d544294287b83d065aafcf8a96fa65b9ffca032822916a08042d21cf8', "floats": FLOATS['ranking/wall-1/reflections_and_echo']},
    'ranking/wall-1/reverberation': {"exact": 'ad62c728ab25de860bafae7b32faac95585bc7fb5962d80b888b2980494be0c4', "floats": FLOATS['ranking/wall-1/reverberation']},
    'ranking/wall-1/timbre_balance': {"exact": '732ac14a68e991b52d94f28f2f01024312be80ee54923f9975b52a155bd7927b', "floats": FLOATS['ranking/wall-1/timbre_balance']},
    'ranking/wall-1/row': {"exact": '16b2e8ae8b9961b4e5d83db06fb419aaa012777d61518e04c5f30a0ad8f52e33', "floats": FLOATS['ranking/wall-1/row']},
    'ranking/wall-2/channel_matching': {"exact": '54ec671bf7c7dc805e04b442a6be199c48632fd49f53c2f43c7f16d046b33d62', "floats": FLOATS['ranking/wall-2/channel_matching']},
    'ranking/wall-2/reflections_and_echo': {"exact": 'feae1ec728ec26cdc25edc3cd37458ef51bfa64633d7595c94fec128f1fe4348', "floats": FLOATS['ranking/wall-2/reflections_and_echo']},
    'ranking/wall-2/reverberation': {"exact": '6a199186c7c8591126acbde5519b97301993e72eae5a8a93b30fe5b17517a137', "floats": FLOATS['ranking/wall-2/reverberation']},
    'ranking/wall-2/timbre_balance': {"exact": '36c45f927630d80fbde7150a5dc8d9f7ad57234a5e7d68e8aae59df4b9e2e5c4', "floats": FLOATS['ranking/wall-2/timbre_balance']},
    'ranking/wall-2/row': {"exact": '2b3a9ad2c06774cc0b08c2ff45c726e81eb3c415b4c3556752487080363d02c0', "floats": FLOATS['ranking/wall-2/row']},
    'ranking/rest': {"exact": 'a97235ebbda44bcb72e0851421b5046647ed41805b11ace954732c055fdff211', "floats": FLOATS['ranking/rest']},
}

# #558（2026-10-25）：先實跑確認舊答案紅，再按頻段分工更新起伏、聽區與衍生代價；
# 原曲線與物理原料不改；新增峰谷範圍、契約 v4 的骨架也重錄。逐格差異見 REPORT-1.md。
