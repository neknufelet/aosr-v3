"""控制樣本用的道具：抑制標記照卡頂層 [[by_design]] 登記的函式與處數，一處不多一處不少。

只給 exemptions-need-expiry 的樣本當道具用，不是真的考卷。
"""
LONG = 1  # noqa: E501


class Probe:
    def run(self, build):
        return build(frequency=1)  # type: ignore[arg-type]


def guarded(build, flag):
    if flag:
        def first():
            return build(frequency=2)
        return build(frequency=first)  # type: ignore[arg-type]
    try:
        def second():
            return build(frequency=3)
        return build(frequency=second)  # type: ignore[arg-type]
    except TypeError:
        return None
