"""樣本道具：一支考卷，刻意餵錯型別看它炸。

只給 exemptions-need-expiry 的樣本當道具用，不是真的考卷。
"""


def probe(build):
    build(frequency=1)  # type: ignore[arg-type]
    return build(frequency=2)  # type: ignore[arg-type]
