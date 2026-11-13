"""樣本道具：一支考卷，刻意餵錯型別看它炸。

只給 exemptions-need-expiry 的樣本當道具用，不是真的考卷。
"""


def probe(build):
    return build(frequency=1)  # type: ignore[arg-type]
