"""樣本道具：一支設定讀取器，開檔寫法跟上一代逐字相同。

只給 exemptions-need-expiry 的樣本當道具用，不是真的考卷。
"""

import os


def probe(path):
    with open(os.path.join(path, "a.toml"), "rb") as file:  # noqa: PTH123, PTH118
        return file.read()
