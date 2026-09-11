"""違規：第二層用動態取模組把第三層載進來。"""
import importlib as il

VALUE = il.import_module("aosr.physics.solver")
