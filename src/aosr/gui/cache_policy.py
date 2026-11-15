"""網頁回應的快取規則（#1181）。

網頁與靜態檔原本只帶 ``etag``／``last-modified``，瀏覽器照自己猜的期限沿用舊檔：合併、重開網頁之後，
老闆打開看到的還是舊的 js（11-15「我看還是舊的葉面阿」）。GET 回應沒自己設 ``Cache-Control`` 的一律
``no-cache``（每次先回伺服器問；這一版 Starlette 的檔案回應不回 304，等於每次重抓，檔都小）；
已經設成 ``no-store`` 的 API 照舊。
"""
from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response


def ask_every_time(request: Request, response: Response) -> Response:
    """GET 回應沒設 ``Cache-Control`` 的補 ``no-cache``，其餘原樣回。"""
    if request.method == "GET" and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response
