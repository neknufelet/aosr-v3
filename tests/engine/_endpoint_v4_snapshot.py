"""4b8223f7 改前的 v4 終點段落快照；只守讀取與逐字輸出。"""

SNAPSHOT_JSON = r'''{
  "source_commit": "4b8223f788b4f9579c0c6345be127a9bacadff40",
  "record": {
    "format_version": "aosr.material_gradient_endpoint.v4",
    "gradient_id": "4ecb5b625a13f0e221e8ab15f828c435",
    "result_ledger_sha256": "b42c60b0c30ca1c74e90fbc6931ea698a8a8ef98a08ef6b8e3a47298b6efdf48",
    "result_template_ledger_sha256": "092bd5cb8bcc75abf814e74452daf918cd5394fe079025ef3f21d8ecc1352ab4",
    "result_sha256": "a5223091bbae4133bc428143e1d7e06853098bedc04b37ed8dfdef6e64af4fde",
    "report": {
      "low_frequency_axis": "search_octave_24",
      "speaker_ids": [
        "left",
        "right"
      ],
      "receiver_ids": [
        "main",
        "front",
        "back",
        "left",
        "right",
        "up",
        "down"
      ],
      "search_theta": [
        0.2,
        0.07929806256953623,
        0.05,
        0.25,
        0.2,
        0.05
      ],
      "exported_theta": [
        0.2,
        0.07929806256953623,
        0.05,
        0.25,
        0.2,
        0.05
      ],
      "surfaces": [
        [
          "floor",
          "glass_wool_48k_50mm_gap50",
          0.2,
          0.07929806256953623,
          "64067cbc21c4ed16397af5af4a3aec65599c74d3f28ecbab0f98d4d11da0266d"
        ],
        [
          "ceiling",
          null,
          null,
          null,
          "7589327f8f541a444bcd45f3328dd361c5f8c98076fa1c4085066b12a6c95d78"
        ],
        [
          "x0",
          "glass_wool_48k_50mm_gap50",
          0.05,
          0.25,
          "64067cbc21c4ed16397af5af4a3aec65599c74d3f28ecbab0f98d4d11da0266d"
        ],
        [
          "xL",
          "glass_wool_48k_50mm",
          0.2,
          null,
          "ed4a66abb87eb2d3d438861eae63460efbee4654b40d3baeeb6af2d46f5f1692"
        ],
        [
          "y0",
          null,
          null,
          null,
          "7589327f8f541a444bcd45f3328dd361c5f8c98076fa1c4085066b12a6c95d78"
        ],
        [
          "yL",
          "glass_wool_24k_50mm",
          0.05,
          null,
          "9aa0be661d4d1836d45bda2695460415acb4847912bb17f78b140d673c4e4fec"
        ]
      ],
      "dropped_cavities": [],
      "reduced_total": 5.866104821124377,
      "reduced_components": [
        2.2019653413419262,
        3.6641394797824507
      ],
      "full_total": 3.879824545544901,
      "full_components": [
        2.203220623192582,
        1.6766039223523193
      ],
      "total_difference": -1.9862802755794764,
      "total_difference_bound": 61.48673829358548,
      "max_frequency_db": 20.495579431194777,
      "rms_floor_db": 5.764786594407953e-13,
      "frequency_db_by_seat": {
        "main": 0.025651898097436725,
        "front": 18.297593974522865,
        "back": 15.195424498943508,
        "left": 13.75261834128866,
        "right": 17.861028564875838,
        "up": 20.495579431194777,
        "down": 19.43666034339053
      },
      "band_db_by_seat": {
        "main": 0.0018139244029140107,
        "front": 16.584135441834164,
        "back": 8.356992045051419,
        "left": 12.678137796916689,
        "right": 12.706876829165385,
        "up": 11.000304674437746,
        "down": 7.909556541374526
      },
      "partial_band_db_by_seat": {
        "main": 0.00763078804586579,
        "front": 11.276571723754445,
        "back": 14.999187770759654,
        "left": 9.455373439862568,
        "right": 12.696119258280552,
        "up": 7.68527654535024,
        "down": 4.222757565207381
      },
      "big_peak_db_by_speaker": {
        "left": 2.4960283652648094,
        "right": 1.757840507747222
      },
      "big_peak_db": 2.4960283652648094,
      "full_solves": 94,
      "model_digest": "b1c80c85a4905f3dd1af9a8a72725de2c937b3ba951c86f932e9f331442836c6",
      "reduced_objective_total": 7.252863881365175,
      "reduced_proxy_db": 5.54703624096319,
      "full_objective_total": 5.266755747314222,
      "full_proxy_db": 5.547724807077284,
      "proxy_gap_bound_db": 1.2592381506034074,
      "reduced_interior_max_db": 5.543809379611888,
      "full_interior_max_db": 5.544503544356956,
      "below_scoring": {
        "range_hz": [
          20.0,
          30.0
        ],
        "by_speaker": {
          "left": {
            "strongest_peak": null,
            "seat_mean_max": {
              "frequency_hz": 20.0,
              "db": 27.026067511016585
            }
          },
          "right": {
            "strongest_peak": null,
            "seat_mean_max": {
              "frequency_hz": 20.0,
              "db": 26.857128611338425
            }
          }
        }
      }
    },
    "baseline": {
      "full_total": 4.042060376897451,
      "full_components": [
        2.6841964921637538,
        1.3578638847336972
      ],
      "big_peak_db_by_speaker": {
        "left": 3.9740472571057524,
        "right": 2.8090966166362428
      },
      "big_peak_db": 3.9740472571057524,
      "full_solves": 94,
      "full_objective_total": 5.209546303611589,
      "full_proxy_db": 4.669943706856551,
      "full_interior_max_db": 4.480971067728485,
      "below_scoring": {
        "range_hz": [
          20.0,
          30.0
        ],
        "by_speaker": {
          "left": {
            "strongest_peak": null,
            "seat_mean_max": {
              "frequency_hz": 20.0,
              "db": 23.192509355625187
            }
          },
          "right": {
            "strongest_peak": null,
            "seat_mean_max": {
              "frequency_hz": 20.0,
              "db": 23.125565549338454
            }
          }
        }
      }
    },
    "improvement": {
      "total": 0.16223583135255026,
      "components": [
        0.48097586897117184,
        -0.318740037618622
      ],
      "objective_total": -0.057209443702633145,
      "proxy_db": -0.8777811002207327,
      "below_scoring": {
        "left": {
          "strongest_peak_db": null,
          "seat_mean_max_db": -3.8335581553913975
        },
        "right": {
          "strongest_peak_db": null,
          "seat_mean_max_db": -3.731563061999971
        }
      }
    },
    "exported_scheme": "scheme-endpoint.json",
    "exported_scheme_sha256": "d52f9d53e1015b3800547440cbe6d4c549c11901a794dc4a5ee9943189306a70",
    "seconds": {
      "export_operators": 0.9150578528642654,
      "endpoint": 9.49331720173359,
      "baseline": 8.006993132643402
    },
    "finished_at": "2026-11-07T21:32:28+00:00",
    "engine_commit": "test",
    "device": "gpu",
    "max_refinement_steps": 2
  },
  "lines": [
    "終點完整模型重算（2026-11-07T21:32:28+00:00，引擎 test）：",
    "  替身：舊合分 5.866105／新合分 7.252864／代理 5.547 dB",
    "  完整模型：舊合分 3.879825／新合分 5.266756／代理 5.548 dB／真大峰 2.496 dB／上界 1.259 dB（主位起伏 2.203221 dB、座位穩健性 1.676604 dB）",
    "  終點代理 − 內部最大殘差 ＝ +0.003221 dB（應在 0～上界 1.259238 dB 之間）",
    "  終點真大峰 2.496 dB（偵測器；低於內部最大殘差 5.545 dB 表示那個最大值不是嚴格局部極大）",
    "  舊合分的差（完整減替身）-1.986280；上界 61.486738",
    "  每座位最大逐頻差（替身對完整模型）：main 0.026 dB、front 18.298 dB、back 15.195 dB、left 13.753 dB、right 17.861 dB、up 20.496 dB、down 19.437 dB；全部最大 20.495579 dB",
    "  真大峰（完整模型）：left 2.496 dB、right 1.758 dB；較差 2.496 dB",
    "  基準（方案原本六面，完整模型）：舊合分 4.042060／新合分 5.209546／代理 4.670 dB／真大峰 3.974 dB／上界 1.259 dB（主位起伏 2.684196 dB、座位穩健性 1.357864 dB）；真大峰 left 3.974 dB、right 2.809 dB",
    "  基準代理 − 內部最大殘差 ＝ +0.188973 dB（應在 0～上界 1.259238 dB 之間）",
    "  基準真大峰 3.974 dB（偵測器；低於內部最大殘差 4.481 dB 表示那個最大值不是嚴格局部極大）",
    "  改善（基準減終點，完整模型）：舊合分 +0.162236（主位起伏 +0.480976 dB、座位穩健性 -0.318740 dB）；新合分 -0.057209、代理 -0.878 dB",
    "  left：30 Hz 以下（不計分），自 20 Hz 起：沒有可辨識峰；原牆 沒有可辨識峰；座位平均能量最高 20.0 Hz、27.026 dB，原牆 20.0 Hz、23.193 dB。",
    "  right：30 Hz 以下（不計分），自 20 Hz 起：沒有可辨識峰；原牆 沒有可辨識峰；座位平均能量最高 20.0 Hz、26.857 dB，原牆 20.0 Hz、23.126 dB。",
    "  拿掉的空腔（小於最小空腔不放空氣層）：沒有",
    "  匯出方案：scheme-endpoint.json（SHA-256 d52f9d53e1015b3800547440cbe6d4c549c11901a794dc4a5ee9943189306a70）",
    "  秒數：匯出算子 0.9、終點重算 9.5、基準 8.0"
  ]
}
'''
