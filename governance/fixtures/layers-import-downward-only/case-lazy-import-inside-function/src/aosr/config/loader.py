"""違規：第二層引第三層，那一行寫在函式裡。"""


def load() -> str:
    from aosr.physics import solver

    return str(solver)
