"""主線 bbc7d17075010a846956003691efe9eac34bfa5c 的命令列實跑答案。

2026-10-08，乾淨主 checkout，由指揮者錄製；原文取 control 目錄兩份 .txt。
shared_control_result(..., "wall-1") 與 test_gui_compare_view.moved_primary_result，
save_result 成 wall-1.json、wall-2.json；scheme_cli.main(compare, 兩份路徑,
--capabilities config_path("capabilities.toml"), --run-date 2026-09-27)，capsys 取 stdout。
下列常數保留原文每一行與末尾換行，不以雜湊取代字句斷言。

#668（音色起伏計分範圍下限 40→20 Hz，老闆 2026-10-20）：在主線 f2dfffd7 加本票前兩顆提交（登記簿改值、網頁標示）上，同一棵樹只把品質登記簿換回
改動前主線那份同跑，照上面的錄法重現舊原文（代價逐位相同）；換新登記簿同法重錄兩份原文，只有代價數字變了
（工作區 668-ripple-from-20 底下 record 目錄的 record_furniture_cli.py）。

#558（音色起伏計分範圍改 300～8000 Hz，峰谷與左右匹配起伏留 20 Hz，老闆 2026-10-25）：先實跑確認只有 wall-2 的音色與聆聽區兩格代價變，
其餘每一行逐字不變；兩格換成同一棵樹新登記簿實跑的值（原文在外部工作區 558-stage2-lf-cost 底下 checks 目錄的 compare-r300.txt）。
"""

PLAIN_THEN_MOVED = """wall-1.json | 讀回等級 current
wall-2.json | 讀回等級 current
wall-1.json | 座位組指紋 73cf50d5ea73b32e97fc427faf368c49e38fdc1f9a6caa690cac398a606f95c4 | 與第一份相同 | rankable | 原因 無
wall-2.json | 座位組指紋 ddfff11df207287eba3de4f6649bdcb1e113b32167485caf9ebbc1da40f29a60 | 與第一份不同 | not_comparable | 原因 與主表的比較身分不同：同一類但身分不同 channel_matching,listening_area_stability
方案 wall-1：rankable
  timbre_balance | measured | 代價 3.8578387728610535 | 原因 無
  listening_area_stability | measured | 代價 0.14832994797740873 | 原因 無
  reflections_and_echo | measured | 代價 0.15162092979796526 | 原因 無
  low_frequency_response | unavailable | 代價 未計 | 原因 low_frequency_seat_layout_invalid
  channel_matching | measured | 代價 0.07616984431420677 | 原因 無
  reverberation | measured | 代價 0.015514454510611286 | 原因 無
低頻拖尾不計分，另有模態診斷報告
低頻模態診斷（不計分）：modal <方案檔> --out <診斷檔> --cache-dir <共用快取資料夾>
方案 wall-2：not_comparable
  timbre_balance | measured | 代價 未計 | 原因 無
  listening_area_stability | measured | 代價 未計 | 原因 無
  reflections_and_echo | measured | 代價 未計 | 原因 無
  low_frequency_response | unavailable | 代價 未計 | 原因 low_frequency_seat_layout_invalid
  channel_matching | measured | 代價 未計 | 原因 無
  reverberation | measured | 代價 未計 | 原因 無
低頻拖尾不計分，另有模態診斷報告
低頻模態診斷（不計分）：modal <方案檔> --out <診斷檔> --cache-dir <共用快取資料夾>
類別 | wall-1 | wall-2
timbre_balance | 3.8578387728610535 | not_comparable
listening_area_stability | 0.14832994797740873 | not_comparable
low_frequency_response | unavailable：low_frequency_seat_layout_invalid | not_comparable
reflections_and_echo | 0.15162092979796526 | not_comparable
reverberation | 0.015514454510611286 | not_comparable
channel_matching | 0.07616984431420677 | not_comparable
spatial_impression | 未評估 | not_comparable
"""

MOVED_THEN_PLAIN = """wall-2.json | 讀回等級 current
wall-1.json | 讀回等級 current
wall-2.json | 座位組指紋 ddfff11df207287eba3de4f6649bdcb1e113b32167485caf9ebbc1da40f29a60 | 與第一份相同 | not_comparable | 原因 與主表的比較身分不同：同一類但身分不同 channel_matching,listening_area_stability
wall-1.json | 座位組指紋 73cf50d5ea73b32e97fc427faf368c49e38fdc1f9a6caa690cac398a606f95c4 | 與第一份不同 | rankable | 原因 無
方案 wall-2：not_comparable
  timbre_balance | measured | 代價 未計 | 原因 無
  listening_area_stability | measured | 代價 未計 | 原因 無
  reflections_and_echo | measured | 代價 未計 | 原因 無
  low_frequency_response | unavailable | 代價 未計 | 原因 low_frequency_seat_layout_invalid
  channel_matching | measured | 代價 未計 | 原因 無
  reverberation | measured | 代價 未計 | 原因 無
低頻拖尾不計分，另有模態診斷報告
低頻模態診斷（不計分）：modal <方案檔> --out <診斷檔> --cache-dir <共用快取資料夾>
方案 wall-1：rankable
  timbre_balance | measured | 代價 3.8578387728610535 | 原因 無
  listening_area_stability | measured | 代價 0.14832994797740873 | 原因 無
  reflections_and_echo | measured | 代價 0.15162092979796526 | 原因 無
  low_frequency_response | unavailable | 代價 未計 | 原因 low_frequency_seat_layout_invalid
  channel_matching | measured | 代價 0.07616984431420677 | 原因 無
  reverberation | measured | 代價 0.015514454510611286 | 原因 無
低頻拖尾不計分，另有模態診斷報告
低頻模態診斷（不計分）：modal <方案檔> --out <診斷檔> --cache-dir <共用快取資料夾>
類別 | wall-2 | wall-1
timbre_balance | not_comparable | 3.8578387728610535
listening_area_stability | not_comparable | 0.14832994797740873
low_frequency_response | not_comparable | unavailable：low_frequency_seat_layout_invalid
reflections_and_echo | not_comparable | 0.15162092979796526
reverberation | not_comparable | 0.015514454510611286
channel_matching | not_comparable | 0.07616984431420677
spatial_impression | not_comparable | 未評估
"""
