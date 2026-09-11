"""5 支**必填** ``path`` 的載入器：餵 ``None`` 要丟跟上一代**逐字相同**的 ``TypeError``。

**這一支在守什麼（找碴第二輪）。** 上一代那 5 支的開檔寫法是 ``with open(path, "rb")``——
`path` 是 `None` 的時候，`open()` 自己丟的那句話是
``expected str, bytes or os.PathLike object, not NoneType``。這一刀一度把開檔改成
``resolved = Path(path)`` ＋ ``resolved.open("rb")``（為了讓路徑來源統一），型別一樣是
`TypeError`，**訊息不一樣**（變成 ``argument should be a str or an os.PathLike object where
__fspath__ returns a str, not 'NoneType'``）。

這個 repo 一向把驗證訊息當行為比（答案檔的逐筆比對就是），所以「訊息不同」就是行為不同。
修法是把那 3 支的開檔寫法改回跟上一代**逐字相同**（`physics_constants`／`scoring`／
`perceptual`；`membrane_opt` 上一代本來就是 `Path(path)` ＋ `.open(...)`，所以它的訊息
本來就是 `__fspath__` 那一句——這一條把它一起釘住，免得有人「順手統一」）。

**為什麼要有一條考卷而不是只靠交叉對帳**：交叉對帳那支是一次性工具、不進版控；這一條
進版控之後，任何人把那三支的開檔寫法改漂亮，這裡當場紅。
"""
from __future__ import annotations

from collections.abc import Callable

import pytest

from aosr.config.membrane_opt import load_membrane_opt
from aosr.config.perceptual import load_material_rfz_profile, load_perceptual
from aosr.config.physics_constants import load_physics_constants
from aosr.config.scoring import load_scoring

# 上一代 `open(path, "rb")` 收到 `None` 時那句話（Python 直譯器自己講的）。
OPEN_MESSAGE = "expected str, bytes or os.PathLike object, not NoneType"
# 上一代 `Path(path)` ＋ `.open("rb")` 收到 `None` 時那句話（`pathlib` 自己講的）。
PATHLIB_MESSAGE = (
    "argument should be a str or an os.PathLike object where __fspath__ returns a str, "
    "not 'NoneType'"
)


@pytest.mark.parametrize(
    ("loader", "message"),
    [
        (load_physics_constants, OPEN_MESSAGE),
        (load_scoring, OPEN_MESSAGE),
        (load_perceptual, OPEN_MESSAGE),
        (load_material_rfz_profile, OPEN_MESSAGE),
        (load_membrane_opt, PATHLIB_MESSAGE),
    ],
)
def test_required_path_loaders_refuse_none_with_the_previous_generation_message(
    loader: Callable[..., object], message: str
) -> None:
    """餵 ``None``：`TypeError`，而且訊息跟上一代逐字相同（開檔寫法沒被「統一」掉）。"""
    with pytest.raises(TypeError) as caught:
        loader(None)  # 這一條要驗的就是「餵 None 會炸、炸什麼訊息」，`None` 是刻意的
    assert str(caught.value) == message, (
        f"{loader.__module__}.{loader.__name__} 餵 None 的訊息跟上一代不一樣\n"
        f"  上一代: {message!r}\n"
        f"  新家  : {str(caught.value)!r}"
    )
