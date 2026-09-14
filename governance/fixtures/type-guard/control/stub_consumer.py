"""控制樣本道具：真的匯入同一棵樹裡的乾淨型別存根檔。"""
from stub_api import render


def rendered(value: int) -> str:
    """經過存根宣告的具體型別呼叫。"""
    return render(value)
