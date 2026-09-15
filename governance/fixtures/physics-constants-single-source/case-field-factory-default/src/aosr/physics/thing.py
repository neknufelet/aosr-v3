"""樣本產品程式：資料模型的欄位用工廠函式包著數字預設值。"""

from pydantic import BaseModel, Field


class Air(BaseModel):
    sound_speed_m_s: float = Field(default=343.0)
