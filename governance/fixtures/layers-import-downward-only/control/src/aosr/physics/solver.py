"""合規：延遲 import 也是往下引。"""


def solve() -> str:
    from aosr import config

    return str(config)
