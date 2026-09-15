"""樣本產品程式：把契約門檻藏在資料類別的欄位預設值裡。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Judge:
    tolerance_rel: float = 2.0 ** -20
