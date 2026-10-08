"""主線 bbc7d17075010a846956003691efe9eac34bfa5c 的命令列實跑答案。

2026-10-08，乾淨主 checkout，由指揮者錄製；原文取 control 目錄兩份 .txt。
shared_control_result(..., "wall-1") 與 test_gui_compare_view.moved_primary_result，
save_result 成 wall-1.json、wall-2.json；scheme_cli.main(compare, 兩份路徑,
--capabilities config_path("capabilities.toml"), --run-date 2026-09-27)，capsys 取 stdout。
下列常數保留原文每一行與末尾換行，不以雜湊取代字句斷言。
"""

PLAIN_THEN_MOVED = """wall-1.json | 讀回等級 current
wall-2.json | 讀回等級 current
wall-1.json | 座位組指紋 73cf50d5ea73b32e97fc427faf368c49e38fdc1f9a6caa690cac398a606f95c4 | 與第一份相同 | not_comparable | 原因 與主表的比較身分不同：同一類但身分不同 channel_matching,listening_area_stability
wall-2.json | 座位組指紋 ddfff11df207287eba3de4f6649bdcb1e113b32167485caf9ebbc1da40f29a60 | 與第一份不同 | rankable | 原因 無
方案 wall-1：not_comparable
  timbre_balance | measured | 代價 未計 | 原因 無
  listening_area_stability | measured | 代價 未計 | 原因 無
  reflections_and_echo | measured | 代價 未計 | 原因 無
  channel_matching | measured | 代價 未計 | 原因 無
  reverberation | measured | 代價 未計 | 原因 無
低頻拖尾不計分，另有模態診斷報告
低頻模態診斷（不計分）：modal <方案檔> --out <診斷檔> --cache-dir <共用快取資料夾>
方案 wall-2：rankable
  timbre_balance | measured | 代價 4.441570107090119 | 原因 無
  listening_area_stability | measured | 代價 0.10258367136835034 | 原因 無
  reflections_and_echo | measured | 代價 0.31047979363252864 | 原因 無
  channel_matching | measured | 代價 0.040995172929975565 | 原因 無
  reverberation | measured | 代價 0.015514454510611286 | 原因 無
低頻拖尾不計分，另有模態診斷報告
低頻模態診斷（不計分）：modal <方案檔> --out <診斷檔> --cache-dir <共用快取資料夾>
類別 | wall-1 | wall-2
timbre_balance | not_comparable | 4.441570107090119
listening_area_stability | not_comparable | 0.10258367136835034
reflections_and_echo | not_comparable | 0.31047979363252864
reverberation | not_comparable | 0.015514454510611286
channel_matching | not_comparable | 0.040995172929975565
spatial_impression | not_comparable | 未評估
"""

MOVED_THEN_PLAIN = """wall-2.json | 讀回等級 current
wall-1.json | 讀回等級 current
wall-2.json | 座位組指紋 ddfff11df207287eba3de4f6649bdcb1e113b32167485caf9ebbc1da40f29a60 | 與第一份相同 | rankable | 原因 無
wall-1.json | 座位組指紋 73cf50d5ea73b32e97fc427faf368c49e38fdc1f9a6caa690cac398a606f95c4 | 與第一份不同 | not_comparable | 原因 與主表的比較身分不同：同一類但身分不同 channel_matching,listening_area_stability
方案 wall-2：rankable
  timbre_balance | measured | 代價 4.441570107090119 | 原因 無
  listening_area_stability | measured | 代價 0.10258367136835034 | 原因 無
  reflections_and_echo | measured | 代價 0.31047979363252864 | 原因 無
  channel_matching | measured | 代價 0.040995172929975565 | 原因 無
  reverberation | measured | 代價 0.015514454510611286 | 原因 無
低頻拖尾不計分，另有模態診斷報告
低頻模態診斷（不計分）：modal <方案檔> --out <診斷檔> --cache-dir <共用快取資料夾>
方案 wall-1：not_comparable
  timbre_balance | measured | 代價 未計 | 原因 無
  listening_area_stability | measured | 代價 未計 | 原因 無
  reflections_and_echo | measured | 代價 未計 | 原因 無
  channel_matching | measured | 代價 未計 | 原因 無
  reverberation | measured | 代價 未計 | 原因 無
低頻拖尾不計分，另有模態診斷報告
低頻模態診斷（不計分）：modal <方案檔> --out <診斷檔> --cache-dir <共用快取資料夾>
類別 | wall-2 | wall-1
timbre_balance | 4.441570107090119 | not_comparable
listening_area_stability | 0.10258367136835034 | not_comparable
reflections_and_echo | 0.31047979363252864 | not_comparable
reverberation | 0.015514454510611286 | not_comparable
channel_matching | 0.040995172929975565 | not_comparable
spatial_impression | 未評估 | not_comparable
"""

