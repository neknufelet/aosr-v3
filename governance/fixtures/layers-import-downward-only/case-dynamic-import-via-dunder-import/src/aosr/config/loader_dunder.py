"""違規：第二層用內建的動態取模組入口把第三層載進來。"""

VALUE = __import__("aosr.physics.solver")
