"""目前模態入口靜態閉包與程式摘要的快取變更偵測答案。

改 CLOSURE 或 CODE_DIGEST 的合併請求，內文要寫明「模態快取作廢」與原因
（每間房重解約 505～564 秒）。這是說明約定，機器不檢查合併請求內文。
只釘程式摘要；執行環境由完整模態身分另行反映，不釘在這裡。
程式摘要使用 ast.dump（語法樹文字），格式跟著 Python 小版本走；換小版本要重錄，
那不算模態變更。
"""

CLOSURE = (
    "aosr", "aosr.config", "aosr.config.fem_lane", "aosr.config.frequency_axis", "aosr.config.paths",
    "aosr.geometry", "aosr.geometry.shoebox", "aosr.geometry.shoebox_mesh", "aosr.physics",
    "aosr.physics.fem_helmholtz", "aosr.physics.fem_modal", "aosr.physics.fem_modal_check",
    "aosr.physics.fem_modal_participation", "aosr.physics.modal_convention", "aosr.reporting",
    "aosr.reporting.import_closure", "aosr.reporting.modal_diagnosis", "aosr.reporting.modal_diagnosis_cache",
    "aosr.reporting.modal_diagnosis_model", "aosr.runtime",
)

PYTHON_MINOR = (3, 12)
# #309 之五甲第二段審查修補：runtime 晚設定核生效值與冷布林讀法改變，實跑重釘；
# #693 已讓不同身分並存。
# #435 模態快取作廢：frequency_axis 加入 48 點搜尋軸，靜態閉包納入整個模組；閉包不變，#693 仍讓新舊快取並存。
CODE_DIGEST = "ae6b705e6234e9e21a7aa9e60429cc5264f126f38e5d1c5d4e0ced27f6b09699"

# #693 老闆 2026-10-10 拍：改模態程式之前必須先修 #693（快取鑰匙或路徑納入模態身分，不同版本並存、不互相覆寫）。
# 這是修 #693 之前的模態程式摘要。快取還不能讓兩個身分並存時，模態程式只准是這一份：
# 只重錄上面的 CODE_DIGEST 過不了 test_modal_lookup.py::test_modal_code_change_waits_for_693，要先修 #693。
# 換 Python 小版本時兩個一起重錄（那不算模態變更）。
# #693 已修：兩個身分讀得回來時那道閘門直接放行，這一份只留作歷史，不再比。
PRE_693_CODE_DIGEST = "b6bc3c1019aa393d573ad6cd0487f305189d472ba5c06df726826d740400c6b5"
