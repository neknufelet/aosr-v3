from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class option:
    @staticmethod
    def setNumber(name: str, value: float) -> None: ...

class model:
    @staticmethod
    def add(name: str) -> None: ...

    class occ:
        @staticmethod
        def addBox(
            x: float,
            y: float,
            z: float,
            dx: float,
            dy: float,
            dz: float,
            tag: int = ...,
        ) -> int: ...
        @staticmethod
        def synchronize() -> None: ...

    class mesh:
        @staticmethod
        def generate(dim: int = ...) -> None: ...
        @staticmethod
        def getNodes(
            dim: int = ...,
            tag: int = ...,
            includeBoundary: bool = ...,
            returnParametricCoord: bool = ...,
        ) -> tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.float64]]: ...
        @staticmethod
        def getElements(
            dim: int = ...,
            tag: int = ...,
        ) -> tuple[
            NDArray[np.int32],
            list[NDArray[np.uint64]],
            list[NDArray[np.uint64]],
        ]: ...

def initialize(
    argv: list[str] = ...,
    readConfigFiles: bool = ...,
    run: bool = ...,
    interruptible: bool = ...,
) -> None: ...
def finalize() -> None: ...
