"""控制樣本用的假命令列工具：print 照卡頂層 [[by_design]] 登記的函式與處數，一處不多一處不少。

只給 style-guard 的樣本當道具用，不是真的程式。
"""


class Report:
    def show(self, rows):
        print(len(rows))


def main(rows):
    print("共", len(rows), "列")
    print("完成")
    return 0


print("sample_cli 載入")
