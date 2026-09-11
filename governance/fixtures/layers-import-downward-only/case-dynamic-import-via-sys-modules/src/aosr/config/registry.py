"""違規：第二層從已載入模組表裡撈第三層。"""
import sys

VALUE = sys.modules["aosr.physics"]
