"""細算尚未寫入時，把保留的使用者停止交回第一階段外圈。"""


class StageOneRefineStopped(Exception):
    """不建立細算狀態；由外圈重讀記號並保存使用者停止結論。"""
