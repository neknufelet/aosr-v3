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
CODE_DIGEST = "b6bc3c1019aa393d573ad6cd0487f305189d472ba5c06df726826d740400c6b5"
