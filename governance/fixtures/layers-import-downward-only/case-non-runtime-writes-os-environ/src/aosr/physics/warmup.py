"""違規：第三層寫環境變數。"""
from os import environ

environ["AOSR_THREADS"] = "1"
