"""樣本道具：型別存根檔用帶錯誤碼的抑制註解壓掉真的 import 錯誤。"""
import package_that_does_not_exist  # type: ignore[import-not-found]

def parse(text: str) -> int: ...
